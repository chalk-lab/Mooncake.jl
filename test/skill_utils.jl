using Mooncake:
    # IR Inspection
    inspect_ir,
    inspect_fwd,
    inspect_rvs,
    quick_inspect,
    show_ir,
    show_stage,
    show_diff,
    show_all_diffs,
    diff_ir,
    cfg_dot,
    write_ir,
    world_age_info,
    show_world_info,
    forward_stage_graph,
    reverse_stage_graph,
    forward_stage_order,
    reverse_stage_order,
    render_ir,
    extract_meta,
    simple_diff,
    StageMeta,
    IRStage,
    IRInspection,
    WorldAgeReport,
    # Performance Diagnostics
    SlotClassification,
    SLOT_OK,
    SLOT_EXPECTED_ABSTRACT,
    SLOT_WIDENED_CALLABLE,
    SLOT_WIDENED_OTHER,
    SlotInfo,
    AllocationMetrics,
    NonInlinedCallee,
    SpecializationReport,
    AllocationReport,
    InliningReport,
    PerfDiagnostic,
    PerfComparison,
    _is_callable_type,
    _expand_vararg_sig,
    _parse_llvm_allocs,
    _count_allocs_min,
    _classify_slot,
    check_specialization,
    check_allocations,
    check_inlining,
    diagnose_perf,
    compare_perf,
    quick_diagnose,
    show_report,
    show_specialization,
    show_allocations,
    show_inlining,
    show_comparison,
    _short_type_name,
    _classification_str

# =============================================================================
# Test helpers
# =============================================================================

test_fn(x) = sin(x) * cos(x)
multi_arg_fn(x, y) = x * y + sin(x)

struct TestSum end
(::TestSum)(x) = sum(x)

struct NotCallable
    x::Int
end

test_vararg(f, x::Vararg{Any,N}) where {N} = f(first(x))
test_explicit(f::F, x) where {F} = f(x)
alloc_free(x, y) = x + y
alloc_func(n) = collect(1:n)
@noinline tiny_noinline(x) = x + 1
inline_caller(x) = tiny_noinline(x) + 2

# =============================================================================
# IR Inspection Tests
# =============================================================================

@testset "ir_inspect" begin
    @testset "inspect_ir reverse mode" begin
        ins = inspect_ir(sin, 1.0)
        @test ins.mode == :reverse
        @test ins.sig == Tuple{typeof(sin),Float64}
        @test ins.world isa UInt
        @test isempty(ins.notes)

        expected_stages =
            [:raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs]
        @test ins.stage_order == expected_stages
        for s in expected_stages
            @test haskey(ins.stages, s)
        end

        for s in expected_stages
            stage = ins.stages[s]
            @test stage.name == s
            @test stage.meta.block_count > 0
            @test stage.meta.inst_count > 0
            @test !isempty(stage.text)
        end

        @test length(ins.diffs) == length(ins.stage_graph)
        for (from, to) in ins.stage_graph
            @test haskey(ins.diffs, from => to)
        end

        @test length(ins.cfgs) == length(ins.stages)
        for (name, dot) in ins.cfgs
            @test occursin("digraph", dot)
        end
    end

    @testset "inspect_ir forward mode" begin
        ins = inspect_fwd(sin, 1.0)
        @test ins.mode == :forward

        expected_stages = [:raw, :normalized, :bbcode, :dual_ir, :optimized]
        @test ins.stage_order == expected_stages
        for s in expected_stages
            @test haskey(ins.stages, s)
            @test ins.stages[s].meta.block_count > 0
        end
    end

    @testset "inspect_ir optimize=false" begin
        ins = inspect_ir(sin, 1.0; optimize=false)
        @test ins.mode == :reverse
        @test :raw in ins.stage_order
        @test :fwd_ir in ins.stage_order
        @test :rvs_ir in ins.stage_order
        @test !haskey(ins.stages, :optimized_fwd)
        @test !haskey(ins.stages, :optimized_rvs)

        ins_fwd = inspect_fwd(sin, 1.0; optimize=false)
        @test :dual_ir in ins_fwd.stage_order
        @test !haskey(ins_fwd.stages, :optimized)
    end

    @testset "inspect_ir multi-arg function" begin
        ins = inspect_ir(multi_arg_fn, 1.0, 2.0)
        @test ins.sig == Tuple{typeof(multi_arg_fn),Float64,Float64}
        @test length(ins.stages) > 0
    end

    @testset "show_ir" begin
        ins = inspect_fwd(sin, 1.0)
        io = IOBuffer()
        show_ir(ins; io)
        output = String(take!(io))
        @test occursin("IR Inspection", output)
        @test occursin("Mode: forward", output)
        for s in ins.stage_order
            @test occursin("Stage: $s", output)
        end
    end

    @testset "show_stage" begin
        ins = inspect_ir(sin, 1.0)
        io = IOBuffer()
        show_stage(ins, :raw; io)
        output = String(take!(io))
        @test occursin("Stage: raw", output)
        @test !occursin("Stage: normalized", output)
    end

    @testset "diff_ir and show_diff" begin
        ins = inspect_fwd(sin, 1.0)

        d = diff_ir(ins; from=:raw, to=:normalized)
        @test d isa String
        @test occursin("---", d)
        @test occursin("+++", d)

        d2 = diff_ir(ins; from=:raw, to=:dual_ir)
        @test d2 isa String

        d3 = diff_ir(ins; from=:nonexistent, to=:raw)
        @test occursin("not found", d3)

        io = IOBuffer()
        show_diff(ins; from=:raw, to=:normalized, io)
        output = String(take!(io))
        @test occursin("Diff:", output)
    end

    @testset "show_all_diffs" begin
        ins = inspect_fwd(sin, 1.0)
        io = IOBuffer()
        show_all_diffs(ins; io)
        output = String(take!(io))
        @test !isempty(output)
        for (from, to) in ins.stage_graph
            @test occursin("Diff: $from", output)
        end
    end

    @testset "cfg_dot" begin
        ins = inspect_ir(sin, 1.0)

        dot = cfg_dot(ins, :raw)
        @test occursin("digraph", dot)
        @test occursin("->", dot)
        @test occursin("Block", dot)

        dot_bb = cfg_dot(ins, :bbcode)
        @test occursin("digraph", dot_bb)
        @test occursin("ID=", dot_bb)

        @test cfg_dot(ins, :nonexistent) == ""
    end

    @testset "world_age_info" begin
        ins = inspect_ir(sin, 1.0)
        report = world_age_info(ins)
        @test report isa WorldAgeReport
        @test report.inspection_world isa UInt
        @test report.inspection_world > 0
        @test length(report.stage_worlds) == length(ins.stages)
    end

    @testset "show_world_info" begin
        ins = inspect_ir(sin, 1.0)
        io = IOBuffer()
        show_world_info(ins; io)
        output = String(take!(io))
        @test occursin("World Age Report", output)
        @test occursin("Inspection world", output)
    end

    @testset "write_ir" begin
        ins = inspect_fwd(sin, 1.0)
        tmpdir = mktempdir()
        write_ir(ins, tmpdir)
        files = readdir(tmpdir)

        @test length(files) ==
            length(ins.stages) + length(ins.diffs) + length(ins.cfgs)

        for s in keys(ins.stages)
            @test "$(s).txt" in files
        end
        for s in keys(ins.cfgs)
            @test "cfg_$(s).dot" in files
        end
        for (from, to) in keys(ins.diffs)
            @test "diff_$(from)_$(to).txt" in files
        end
    end

    @testset "simple_diff" begin
        d = simple_diff("line1\nline2\nline3", "line1\nchanged\nline3")
        @test occursin("-line2", d)
        @test occursin("+changed", d)
        @test !occursin("-line1", d)
        @test !occursin("-line3", d)

        d_longer = simple_diff("a", "a\nb")
        @test occursin("+b", d_longer)
    end

    @testset "render_ir" begin
        ins = inspect_ir(sin, 1.0)
        @test !isempty(render_ir(ins.stages[:raw].ir))
        @test !isempty(render_ir(ins.stages[:bbcode].ir))
        @test occursin("Block", render_ir(ins.stages[:bbcode].ir))
    end

    @testset "convenience functions" begin
        ins_rvs = inspect_rvs(sin, 1.0)
        @test ins_rvs.mode == :reverse

        ins_fwd = inspect_fwd(sin, 1.0)
        @test ins_fwd.mode == :forward

        ins = quick_inspect(sin, 1.0; mode=:forward, stages=:raw)
        @test ins isa IRInspection
        @test ins.mode == :forward
    end

    @testset "stage graph structure" begin
        fg = forward_stage_graph()
        @test fg == [
            :raw => :normalized,
            :normalized => :bbcode,
            :bbcode => :dual_ir,
            :dual_ir => :optimized,
        ]

        rg = reverse_stage_graph()
        @test (:raw => :normalized) in rg
        @test (:bbcode => :fwd_ir) in rg
        @test (:bbcode => :rvs_ir) in rg
        @test (:fwd_ir => :optimized_fwd) in rg
        @test (:rvs_ir => :optimized_rvs) in rg

        @test forward_stage_order() ==
            [:raw, :normalized, :bbcode, :dual_ir, :optimized]
        @test reverse_stage_order() == [
            :raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd,
            :optimized_rvs,
        ]
    end

    @testset "StageMeta" begin
        meta = StageMeta()
        @test meta.block_count == 0
        @test meta.inst_count == 0
        @test meta.edge_count == 0
        @test meta.has_ssa == true
        @test meta.uses_bbcode_ids == false
        @test meta.valid_worlds === nothing
        @test meta.misty_world === nothing

        meta2 = StageMeta(; block_count=5, inst_count=10, uses_bbcode_ids=true)
        @test meta2.block_count == 5
        @test meta2.inst_count == 10
        @test meta2.uses_bbcode_ids == true
    end

    @testset "extract_meta" begin
        ins = inspect_ir(sin, 1.0)

        raw_meta = extract_meta(ins.stages[:raw].ir)
        @test raw_meta.block_count > 0
        @test raw_meta.inst_count > 0
        @test raw_meta.edge_count > 0
        @test raw_meta.has_ssa == true
        @test raw_meta.uses_bbcode_ids == false

        bb_meta = extract_meta(ins.stages[:bbcode].ir)
        @test bb_meta.block_count > 0
        @test bb_meta.inst_count > 0
        @test bb_meta.has_ssa == false
        @test bb_meta.uses_bbcode_ids == true

        fallback = extract_meta("not an IR")
        @test fallback.block_count == 0
    end

    @testset "custom function reverse mode" begin
        ins = inspect_ir(test_fn, 1.0)
        @test ins.mode == :reverse
        @test length(ins.stages) == 7
        @test all(s -> ins.stages[s].meta.inst_count > 0, ins.stage_order)
    end

    @testset "custom function forward mode" begin
        ins = inspect_fwd(test_fn, 1.0)
        @test ins.mode == :forward
        @test length(ins.stages) == 5
        @test all(s -> ins.stages[s].meta.inst_count > 0, ins.stage_order)
    end
end

# =============================================================================
# Performance Diagnostics Tests
# =============================================================================

@testset "perf_diagnose" begin
    @testset "Enums and Structs" begin
        @test SLOT_OK isa SlotClassification
        @test SLOT_EXPECTED_ABSTRACT isa SlotClassification
        @test SLOT_WIDENED_CALLABLE isa SlotClassification
        @test SLOT_WIDENED_OTHER isa SlotClassification

        slot =
            SlotInfo(1, typeof(sin), Function, Any, false, SLOT_WIDENED_CALLABLE, "test")
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
        @test _is_callable_type(typeof(TestSum())) == true
        @test _is_callable_type(Int) == false
        @test _is_callable_type(String) == false
    end

    @testset "_expand_vararg_sig" begin
        m = which(test_vararg, (typeof(sin), Float64))
        expanded = _expand_vararg_sig(m, 3)
        @test length(expanded) == 3

        m2 = which(alloc_free, (Int, Int))
        expanded2 = _expand_vararg_sig(m2, 3)
        @test length(expanded2) == 3
    end

    @testset "_parse_llvm_allocs" begin
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
        @test result.heap_alloc >= 2
        @test result.box_calls >= 1
        @test result.dynamic_dispatch >= 1
        @test length(result.matching_lines) >= 3

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

    @testset "_count_allocs_min" begin
        c = _count_allocs_min(alloc_free, 1, 2)
        @test c == 0

        c2 = _count_allocs_min(alloc_func, 10)
        @test c2 > 0
    end

    @testset "_classify_slot" begin
        slot_ok = _classify_slot(1, Float64, Float64, Float64, false)
        @test slot_ok.classification == SLOT_OK

        slot_abs = _classify_slot(1, Float64, Number, Number, false)
        @test slot_abs.classification == SLOT_EXPECTED_ABSTRACT

        slot_wc = _classify_slot(1, typeof(sin), Function, Any, true)
        @test slot_wc.classification == SLOT_WIDENED_CALLABLE
        @test !isempty(slot_wc.suggestion)
        @test occursin("Callable", slot_wc.suggestion)

        slot_wo = _classify_slot(1, Float64, Real, Any, false)
        @test slot_wo.classification == SLOT_WIDENED_OTHER
    end

    @testset "check_specialization — no widening" begin
        report = check_specialization(alloc_free, 1, 2)
        @test report isa SpecializationReport
        @test report.has_widening == false
        @test report.method isa Method
        for slot in report.slots
            @test slot.classification in (SLOT_OK, SLOT_EXPECTED_ABSTRACT)
        end
    end

    @testset "check_specialization — Vararg + Function" begin
        test_vararg(sin, 1.0)
        report = check_specialization(test_vararg, sin, 1.0)
        @test report isa SpecializationReport
        @test report.has_vararg == true
        @test report.method isa Method
    end

    @testset "check_specialization — callable struct (no widening)" begin
        ts = TestSum()
        test_vararg(ts, [1.0, 2.0])
        report = check_specialization(test_vararg, ts, [1.0, 2.0])
        @test report isa SpecializationReport
        callable_widened =
            any(s -> s.classification == SLOT_WIDENED_CALLABLE, report.slots)
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
