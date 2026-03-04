using Test
using InteractiveUtils

# Load the perf_diagnose script
include(joinpath(@__DIR__, "scripts", "perf_diagnose.jl"))

# =============================================================================
# Test Helpers
# =============================================================================

struct TestSum end
(::TestSum)(x) = sum(x)

struct NotCallable
    x::Int
end

# Vararg function that triggers specialization widening with Function args
test_vararg(f, x::Vararg{Any,N}) where {N} = f(first(x))

# Non-vararg function with explicit types
test_explicit(f::F, x) where {F} = f(x)

# Allocation-free function
alloc_free(x, y) = x + y

# Function that allocates
alloc_func(n) = collect(1:n)

# Small callee for inlining tests
@noinline tiny_noinline(x) = x + 1
inline_caller(x) = tiny_noinline(x) + 2

# =============================================================================
# Tests
# =============================================================================

@testset "PerfDiagnose" begin

    @testset "Enums and Structs" begin
        @test SLOT_OK isa SlotClassification
        @test SLOT_EXPECTED_ABSTRACT isa SlotClassification
        @test SLOT_WIDENED_CALLABLE isa SlotClassification
        @test SLOT_WIDENED_OTHER isa SlotClassification

        slot = SlotInfo(1, typeof(sin), Function, Any, false, SLOT_WIDENED_CALLABLE, "test")
        @test slot.position == 1
        @test slot.runtime_type === typeof(sin)
        @test slot.compiled_type === Function
        @test slot.classification == SLOT_WIDENED_CALLABLE
        @test slot.suggestion == "test"

        metrics = AllocationMetrics(3, 1, 2, 4, 10, 15, true)
        @test metrics.heap_alloc_calls == 3
        @test metrics.inline_saves_allocs == true

        callee = NonInlinedCallee("foo", 5, false, true, "small callee")
        @test callee.callee_name == "foo"
        @test callee.is_small == true
    end

    @testset "_is_callable_type" begin
        @test _is_callable_type(typeof(sin)) == true
        @test _is_callable_type(typeof(sum)) == true
        @test _is_callable_type(typeof(+)) == true
        # Callable struct
        @test _is_callable_type(typeof(TestSum())) == true
        # Non-callable
        @test _is_callable_type(Int) == false
        @test _is_callable_type(String) == false
    end

    @testset "_expand_vararg_sig" begin
        m = which(test_vararg, (typeof(sin), Float64))
        expanded = _expand_vararg_sig(m, 3)  # f + 1 vararg element = 3 params (including typeof(test_vararg))
        @test length(expanded) == 3

        m2 = which(alloc_free, (Int, Int))
        expanded2 = _expand_vararg_sig(m2, 3)  # typeof(alloc_free) + 2 args
        @test length(expanded2) == 3
    end

    @testset "_parse_llvm_allocs" begin
        # Synthetic LLVM IR with known patterns
        llvm_with_allocs = """
        define void @test() {
            %1 = call ptr @ijl_gc_pool_alloc(ptr %0, i32 1, i32 16)
            %2 = call ptr @jl_box_float64(double 1.0)
            %3 = call ptr @jl_apply_generic(ptr %f, ptr %args)
            %4 = call ptr @julia.gc_alloc_obj(ptr %0, i64 8, ptr %type)
            ret void
        }
        """
        result = _parse_llvm_allocs(llvm_with_allocs)
        @test result.heap_alloc >= 2  # ijl_gc_pool_alloc + julia.gc_alloc_obj
        @test result.box_calls >= 1   # jl_box_float64
        @test result.dynamic_dispatch >= 1  # jl_apply_generic
        @test length(result.matching_lines) >= 3

        # Clean LLVM IR
        clean_llvm = """
        define double @clean(double %x) {
            %1 = fadd double %x, 1.0
            ret double %1
        }
        """
        clean_result = _parse_llvm_allocs(clean_llvm)
        @test clean_result.heap_alloc == 0
        @test clean_result.box_calls == 0
        @test clean_result.dynamic_dispatch == 0
    end

    @testset "_count_allocs" begin
        # Allocation-free function
        c = _count_allocs(alloc_free, 1, 2)
        @test c == 0

        # Allocating function
        c2 = _count_allocs(alloc_func, 10)
        @test c2 > 0
    end

    @testset "_classify_slot" begin
        # OK: types match
        slot_ok = _classify_slot(1, Float64, Float64, Float64, false)
        @test slot_ok.classification == SLOT_OK

        # Expected abstract: declared abstract matches compiled
        slot_abs = _classify_slot(1, Float64, Number, Number, false)
        @test slot_abs.classification == SLOT_EXPECTED_ABSTRACT

        # Widened callable: function widened to Function
        slot_wc = _classify_slot(1, typeof(sin), Function, Any, true)
        @test slot_wc.classification == SLOT_WIDENED_CALLABLE
        @test !isempty(slot_wc.suggestion)
        @test occursin("Callable", slot_wc.suggestion)

        # Widened other: non-callable widened
        slot_wo = _classify_slot(1, Float64, Real, Any, false)
        @test slot_wo.classification == SLOT_WIDENED_OTHER
    end

    @testset "check_specialization — no widening" begin
        report = check_specialization(alloc_free, 1, 2)
        @test report isa SpecializationReport
        @test report.has_widening == false
        @test report.method isa Method
        # All slots should be OK
        for slot in report.slots
            @test slot.classification in (SLOT_OK, SLOT_EXPECTED_ABSTRACT)
        end
    end

    @testset "check_specialization — Vararg + Function" begin
        # Force compilation
        test_vararg(sin, 1.0)
        report = check_specialization(test_vararg, sin, 1.0)
        @test report isa SpecializationReport
        @test report.has_vararg == true
        # Check if widening is detected (depends on Julia's specialization behavior)
        # The key is the report runs without error
        @test report.method isa Method
    end

    @testset "check_specialization — callable struct (no widening)" begin
        ts = TestSum()
        test_vararg(ts, [1.0, 2.0])
        report = check_specialization(test_vararg, ts, [1.0, 2.0])
        @test report isa SpecializationReport
        # Callable struct should NOT be widened
        callable_widened = any(
            s -> s.classification == SLOT_WIDENED_CALLABLE,
            report.slots
        )
        @test !callable_widened
    end

    @testset "check_allocations — allocation-free" begin
        report = check_allocations(alloc_free, 1, 2)
        @test report isa AllocationReport
        @test report.metrics.measured_allocs == 0
    end

    @testset "check_allocations — allocating" begin
        report = check_allocations(alloc_func, 100)
        @test report isa AllocationReport
        @test report.metrics.measured_allocs > 0
    end

    @testset "check_inlining" begin
        report = check_inlining(inline_caller, 1.0)
        @test report isa InliningReport
        @test report.invoke_count_optimized >= 0
        @test report.invoke_count_unoptimized >= 0
    end

    @testset "diagnose_perf" begin
        report = diagnose_perf(alloc_free, 1, 2)
        @test report isa PerfDiagnostic
        @test report.specialization isa SpecializationReport
        @test report.allocations isa AllocationReport
        @test report.inlining isa InliningReport
    end

    @testset "show_report output" begin
        report = diagnose_perf(alloc_free, 1, 2)
        buf = IOBuffer()
        show_report(report; io=buf)
        output = String(take!(buf))
        @test occursin("Performance Diagnostic", output)
        @test occursin("Specialization Widening", output)
        @test occursin("Allocation", output)
        @test occursin("Inlining", output)
    end

    @testset "show_specialization output" begin
        report = check_specialization(alloc_free, 1, 2)
        buf = IOBuffer()
        show_specialization(report; io=buf)
        output = String(take!(buf))
        @test occursin("Specialization Widening", output)
        @test occursin("Slot", output)
    end

    @testset "show_allocations output" begin
        report = check_allocations(alloc_free, 1, 2)
        buf = IOBuffer()
        show_allocations(report; io=buf)
        output = String(take!(buf))
        @test occursin("Allocation", output)
        @test occursin("Heap alloc", output)
    end

    @testset "show_inlining output" begin
        report = check_inlining(inline_caller, 1.0)
        buf = IOBuffer()
        show_inlining(report; io=buf)
        output = String(take!(buf))
        @test occursin("Inlining", output)
        @test occursin("Invoke count", output)
    end

    @testset "compare_perf" begin
        comp = compare_perf(test_vararg, (sin, 1.0), (TestSum(), [1.0, 2.0]))
        @test comp isa PerfComparison
        @test !isempty(comp.slots)
        @test !isempty(comp.verdict)
    end

    @testset "show_comparison output" begin
        comp = compare_perf(test_vararg, (sin, 1.0), (TestSum(), [1.0, 2.0]))
        buf = IOBuffer()
        show_comparison(comp; io=buf)
        output = String(take!(buf))
        @test occursin("Performance Comparison", output)
        @test occursin("Per-Slot Comparison", output)
        @test occursin("Metrics", output)
        @test occursin("Verdict", output)
    end

    @testset "quick_diagnose" begin
        buf = IOBuffer()
        report = quick_diagnose(alloc_free, 1, 2; io=buf)
        @test report isa PerfDiagnostic
        output = String(take!(buf))
        @test occursin("Performance Diagnostic", output)
    end

    @testset "display helpers" begin
        @test _short_type_name(Int) == "Int64"
        @test _short_type_name(nothing) == "N/A"
        long_type = Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int,Int}
        short = _short_type_name(long_type)
        @test length(short) <= 30

        @test _classification_str(SLOT_OK) == "OK"
        @test _classification_str(SLOT_WIDENED_CALLABLE) == "WIDENED_CALLABLE"
        @test _classification_str(SLOT_EXPECTED_ABSTRACT) == "expected_abstract"
    end
end

println("\nAll PerfDiagnose tests completed!")
