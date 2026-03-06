#=
Performance Diagnostics for Mooncake.jl
=======================================
Detects Julia compiler boundary issues: specialization widening, SROA/allocation
failures, and inlining failures that are invisible to Mooncake's internal IR.

## Quick Start
    using Revise
    includet(".claude/skills/perf-diagnose/scripts/perf_diagnose.jl")

    # Full diagnostic
    report = diagnose_perf(f, args...)
    show_report(report)

    # Individual detectors
    check_specialization(f, args...)
    check_allocations(f, args...)
    check_inlining(f, args...)

    # Compare two calls
    compare_perf(f, (slow_args...,), (fast_args...,))
=#

using InteractiveUtils
using Printf

# =============================================================================
# Enums and Data Structures
# =============================================================================

@enum SlotClassification begin
    SLOT_OK
    SLOT_EXPECTED_ABSTRACT
    SLOT_WIDENED_CALLABLE
    SLOT_WIDENED_OTHER
end

struct SlotInfo
    position::Int
    runtime_type::Type
    compiled_type::Type
    declared_type::Type
    is_vararg_slot::Bool
    classification::SlotClassification
    suggestion::String
end

struct SpecializationReport
    method::Method
    method_instance::Union{Core.MethodInstance,Nothing}
    runtime_sig::Type
    compiled_sig::Union{Type,Nothing}
    slots::Vector{SlotInfo}
    has_widening::Bool
    has_vararg::Bool
    notes::Vector{String}
end

struct AllocationMetrics
    heap_alloc_calls::Int
    box_calls::Int
    dynamic_dispatch_calls::Int
    gc_frame_slots::Int
    measured_allocs::Int
    measured_allocs_noinline::Int
    inline_saves_allocs::Bool
end

struct AllocationReport
    sig::Type
    metrics::AllocationMetrics
    matching_lines::Vector{String}
    suggestions::Vector{String}
    notes::Vector{String}
end

struct NonInlinedCallee
    callee_name::String
    statement_count::Union{Int,Nothing}
    is_noinline::Bool
    is_small::Bool
    suggestion::String
end

struct InliningReport
    sig::Type
    invoke_count_optimized::Int
    invoke_count_unoptimized::Int
    non_inlined_callees::Vector{NonInlinedCallee}
    suggestions::Vector{String}
    notes::Vector{String}
end

struct PerfDiagnostic
    sig::Type
    specialization::SpecializationReport
    allocations::AllocationReport
    inlining::InliningReport
    overall_suggestions::Vector{String}
    notes::Vector{String}
end

struct SlotComparison
    position::Int
    runtime_type_a::Type
    runtime_type_b::Type
    compiled_type_a::Union{Type,Nothing}
    compiled_type_b::Union{Type,Nothing}
    classification_a::SlotClassification
    classification_b::SlotClassification
end

struct PerfComparison
    sig_a::Type
    sig_b::Type
    slots::Vector{SlotComparison}
    invoke_count_a::Int
    invoke_count_b::Int
    heap_alloc_calls_a::Int
    heap_alloc_calls_b::Int
    measured_allocs_a::Int
    measured_allocs_b::Int
    verdict::String
    suggestions::Vector{String}
end

# =============================================================================
# Internal Helpers
# =============================================================================

"""Check if a type is callable (Function subtype or has call methods)."""
function _is_callable_type(@nospecialize(T::Type))::Bool
    T <: Function && return true
    # Check for callable structs (types with () methods)
    try
        return !isempty(methods(T.instance)) || hasmethod(T, Tuple{Vararg{Any}})
    catch
        return false
    end
end

"""Get specializations portably across Julia versions."""
function _get_specializations(method::Method)
    mis = Core.MethodInstance[]
    for mi in Base.specializations(method)
        mi === nothing && continue
        push!(mis, mi)
    end
    return mis
end

"""Find the MethodInstance that Julia compiled for a given call."""
function _find_method_instance(f, args...)
    tt = Tuple{typeof(f),map(typeof, args)...}
    # Force compilation
    try
        f(args...)
    catch
    end
    meth = which(f, typeof.(args))
    # Exact match first
    for mi in _get_specializations(meth)
        if mi.specTypes === tt
            return mi
        end
    end
    # Closest match: tt <: mi.specTypes, pick the tightest
    best = nothing
    for mi in _get_specializations(meth)
        if tt <: mi.specTypes
            if best === nothing || mi.specTypes <: best.specTypes
                best = mi
            end
        end
    end
    return best
end

"""Expand a method signature's Vararg to the actual arity, returning a Vector of Any (may contain TypeVar)."""
function _expand_vararg_sig(method::Method, nargs::Int)
    sig = Base.unwrap_unionall(method.sig)
    params = sig.parameters
    declared = Any[]
    for (i, p) in enumerate(params)
        if Base.isvarargtype(p)
            elem_type = Base.unwrapva(p)
            remaining = nargs - length(declared)
            append!(declared, fill(elem_type, remaining))
            break
        else
            push!(declared, p)
        end
    end
    # Pad if needed (shouldn't happen normally)
    while length(declared) < nargs
        push!(declared, Any)
    end
    return declared
end

"""Classify a single argument slot."""
function _classify_slot(
    position::Int,
    @nospecialize(runtime_type),
    @nospecialize(compiled_type),
    @nospecialize(declared_type),
    is_vararg::Bool,
)::SlotInfo
    # Coerce TypeVars to their upper bounds for comparison
    rt = runtime_type isa TypeVar ? runtime_type.ub : runtime_type
    ct = compiled_type isa TypeVar ? compiled_type.ub : compiled_type
    dt = declared_type isa TypeVar ? declared_type.ub : declared_type

    # Ensure we have Type values for the SlotInfo struct
    rt isa Type || (rt = Any)
    ct isa Type || (ct = Any)
    dt isa Type || (dt = Any)

    if rt === ct
        return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_OK, "")
    end

    # Is the declared type intentionally abstract (non-Any)?
    expected_abstract = (
        dt isa DataType &&
        isabstracttype(dt) &&
        dt !== Any &&
        ct === dt
    )
    if expected_abstract
        return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_EXPECTED_ABSTRACT, "")
    end

    # Is this a callable type that got widened?
    is_callable = try rt <: Function catch; false end
    if !is_callable
        # Also check for callable structs (singleton types with call methods)
        try
            is_callable = isconcretetype(rt) && hasmethod(rt.instance, Tuple{Vararg{Any}})
        catch
        end
    end

    if is_callable && (ct === Function || ct === Any)
        suggestion = if is_vararg
            "Callable in Vararg position widened to $(ct). Fix: wrap in a callable struct, or add @inline to the caller"
        else
            "Callable argument widened to $(ct). Fix: add explicit type parameter (::F where {F}) or add @inline"
        end
        return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_WIDENED_CALLABLE, suggestion)
    end

    widened = try rt !== ct && rt <: ct catch; false end
    if widened
        return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_WIDENED_OTHER,
            "Runtime type widened: $(rt) -> $(ct)")
    end

    return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_OK, "")
end

# --- Allocation counting (replicating Mooncake's pattern from test_utils.jl) ---
# Use @eval-generated per-arity methods to avoid Vararg specialization issues
const __MAX_DIAG_ARGS = 10

for nargs in 0:__MAX_DIAG_ARGS
    args_syms = [Symbol("x", i) for i in 1:nargs]
    types_syms = [Symbol("X", i) for i in 1:nargs]
    sigs = [:($(args_syms[i])::$(types_syms[i])) for i in 1:nargs]
    @eval function _count_allocs(f::F, $(sigs...)) where {F,$(types_syms...)}
        # Warmup
        f($(args_syms...))
        # Measure
        best = typemax(Int)
        for _ in 1:5
            stats = Base.gc_num()
            f($(args_syms...))
            diff = Base.GC_Diff(Base.gc_num(), stats)
            c = Base.gc_alloc_count(diff)
            c < best && (best = c)
        end
        return best
    end
end

# Vararg fallback for > __MAX_DIAG_ARGS arguments
function _count_allocs(f::F, x::Vararg{Any,N}) where {F,N}
    f(x...)
    best = typemax(Int)
    for _ in 1:5
        stats = Base.gc_num()
        f(x...)
        diff = Base.GC_Diff(Base.gc_num(), stats)
        c = Base.gc_alloc_count(diff)
        c < best && (best = c)
    end
    return best
end

"""Parse LLVM IR text for allocation patterns."""
function _parse_llvm_allocs(llvm_text::String)
    lines = split(llvm_text, '\n')
    heap_re = r"@ij?l_gc_pool_alloc\b|@julia\.gc_alloc_obj\b|@julia\.gc_alloc_bytes\b|@jl_alloc_array"
    box_re = r"@jl_box_[A-Za-z0-9_]+\b"
    dispatch_re = r"@jl_apply_generic\b|@jl_invoke\b"
    gcframe_re = r"gcframe|julia\.new_gc_frame"

    heap = count(l -> occursin(heap_re, l), lines)
    box = count(l -> occursin(box_re, l), lines)
    dispatch = count(l -> occursin(dispatch_re, l), lines)
    gc_slots = count(l -> occursin(gcframe_re, l), lines)

    matching = filter(l -> any(re -> occursin(re, l), [heap_re, box_re, dispatch_re]), lines)

    return (heap_alloc=heap, box_calls=box, dynamic_dispatch=dispatch,
        gc_slots=gc_slots, matching_lines=String.(matching))
end

"""Count :invoke expressions in CodeInfo."""
function _count_invokes(ci::Core.CodeInfo)::Int
    return count(st -> st isa Expr && st.head === :invoke, ci.code)
end

"""Extract non-inlined callees from optimized CodeInfo."""
function _extract_non_inlined(ci_opt::Core.CodeInfo; max_stmts::Int=20)
    callees = NonInlinedCallee[]
    for st in ci_opt.code
        st isa Expr && st.head === :invoke || continue
        mi = st.args[1]
        mi isa Core.MethodInstance || continue
        m = mi.def
        m isa Method || continue

        name = string(m.name)
        # Try to count lowered statements
        stmt_count = nothing
        try
            src = Base.uncompressed_ast(m)
            stmt_count = count(s -> !(s isa Core.LineNumberNode), src.code)
        catch
        end

        is_small = stmt_count !== nothing && stmt_count <= max_stmts
        # We can't easily check @noinline portably; assume not noinline
        is_noinline = false

        suggestion = if is_small
            "Small callee '$name' ($stmt_count stmts) not inlined. Consider adding @inline or checking specialization"
        else
            ""
        end

        push!(callees, NonInlinedCallee(name, stmt_count, is_noinline, is_small, suggestion))
    end
    return callees
end

# =============================================================================
# Public API — Individual Detectors
# =============================================================================

"""
    check_specialization(f, args...) -> SpecializationReport

Detect specialization widening at the Julia compiler boundary. Compares runtime
argument types against the compiled MethodInstance's specTypes to find cases where
concrete callable types (like `typeof(sum)`) were widened to `Function` or `Any`.
"""
function check_specialization(f, args...)
    tt = Tuple{typeof(f),map(typeof, args)...}
    meth = which(f, typeof.(args))
    mi = _find_method_instance(f, args...)

    nparams = length(tt.parameters)
    declared = _expand_vararg_sig(meth, nparams)
    compiled = mi !== nothing ? collect(Base.unwrap_unionall(mi.specTypes).parameters) : fill(Any, nparams)
    runtime = collect(tt.parameters)

    # Determine vararg start position
    sig_params = Base.unwrap_unionall(meth.sig).parameters
    vararg_start = typemax(Int)
    for (i, p) in enumerate(sig_params)
        if Base.isvarargtype(p)
            vararg_start = i
            break
        end
    end

    slots = SlotInfo[]
    has_widening = false
    for i in 1:nparams
        is_va = i >= vararg_start
        # Pad if compiled has fewer params (shouldn't happen)
        ct = i <= length(compiled) ? compiled[i] : Any
        dt = i <= length(declared) ? declared[i] : Any
        slot = _classify_slot(i, runtime[i], ct, dt, is_va)
        push!(slots, slot)
        if slot.classification == SLOT_WIDENED_CALLABLE
            has_widening = true
        end
    end

    notes = String[]
    if has_widening
        push!(notes, "Specialization widening detected. Julia compiled a less-specific method instance than the runtime types warrant.")
    end

    return SpecializationReport(
        meth, mi, tt, mi !== nothing ? mi.specTypes : nothing,
        slots, has_widening, meth.isva, notes,
    )
end

"""
    check_allocations(f, args...) -> AllocationReport

Analyze allocation behavior at the Julia compiler boundary. Parses LLVM IR for
heap allocation patterns and measures runtime allocations with inline-sensitivity
probing to detect SROA failures.
"""
function check_allocations(f, args...)
    argtt = Tuple{map(typeof, args)...}
    sig = Tuple{typeof(f),argtt.parameters...}

    # LLVM IR analysis
    llvm_text = try
        io = IOBuffer()
        code_llvm(io, f, argtt; raw=true, debuginfo=:none)
        String(take!(io))
    catch e
        ""
    end

    parsed = _parse_llvm_allocs(llvm_text)

    # Runtime allocation measurement
    measured = try
        _count_allocs(f, args...)
    catch
        -1
    end

    # Inline-sensitivity probe: wrap in noinline barrier
    noinline_measured = try
        wrapper = @noinline (f, args...) -> f(args...)
        wrapper(f, args...) # warmup
        _count_allocs(wrapper, f, args...)
    catch
        -1
    end

    inline_saves = measured >= 0 && noinline_measured >= 0 && noinline_measured > measured

    metrics = AllocationMetrics(
        parsed.heap_alloc, parsed.box_calls, parsed.dynamic_dispatch,
        parsed.gc_slots, measured, noinline_measured, inline_saves,
    )

    suggestions = String[]
    if metrics.dynamic_dispatch_calls > 0
        push!(suggestions, "Dynamic dispatch detected in LLVM IR ($(metrics.dynamic_dispatch_calls) calls). Ensure all call targets have concrete types.")
    end
    if metrics.box_calls > 0
        push!(suggestions, "Boxing detected ($(metrics.box_calls) calls). This suggests type instability at the compiler boundary.")
    end
    if inline_saves
        push!(suggestions, "Allocations depend on inlining ($(measured) direct vs $(noinline_measured) with @noinline barrier). Add @inline to enable SROA.")
    end
    if metrics.heap_alloc_calls > 0 && measured == 0
        push!(suggestions, "LLVM IR shows allocation calls but runtime reports zero. Likely optimized away.")
    end

    notes = String[]
    if isempty(llvm_text)
        push!(notes, "Could not generate LLVM IR for this function.")
    end

    return AllocationReport(sig, metrics, parsed.matching_lines, suggestions, notes)
end

"""
    check_inlining(f, args...) -> InliningReport

Detect inlining failures by comparing optimized vs unoptimized code_typed output.
Flags small callees that survive optimization without being inlined.
"""
function check_inlining(f, args...)
    argtt = Tuple{map(typeof, args)...}
    sig = Tuple{typeof(f),argtt.parameters...}

    ci_opt = nothing
    ci_unopt = nothing
    notes = String[]

    try
        result = code_typed(f, argtt; optimize=true)
        ci_opt = first(result)[1]
    catch e
        push!(notes, "code_typed(optimize=true) failed: $e")
    end

    try
        result = code_typed(f, argtt; optimize=false)
        ci_unopt = first(result)[1]
    catch e
        push!(notes, "code_typed(optimize=false) failed: $e")
    end

    invoke_opt = ci_opt !== nothing ? _count_invokes(ci_opt) : 0
    invoke_unopt = ci_unopt !== nothing ? _count_invokes(ci_unopt) : 0

    callees = ci_opt !== nothing ? _extract_non_inlined(ci_opt) : NonInlinedCallee[]

    suggestions = String[]
    small_count = count(c -> c.is_small, callees)
    if small_count > 0
        push!(suggestions, "$small_count small callee(s) not inlined. This may indicate specialization widening prevented inlining.")
    end
    if invoke_opt > invoke_unopt
        push!(notes, "More invokes after optimization ($invoke_opt) than before ($invoke_unopt) — unusual, may indicate issues.")
    end

    return InliningReport(sig, invoke_opt, invoke_unopt, callees, suggestions, notes)
end

# =============================================================================
# Public API — Full Diagnostic and Compare
# =============================================================================

"""
    diagnose_perf(f, args...) -> PerfDiagnostic

Run all three detectors and produce a prioritized diagnostic report.
"""
function diagnose_perf(f, args...)
    sig = Tuple{typeof(f),map(typeof, args)...}
    spec = check_specialization(f, args...)
    allocs = check_allocations(f, args...)
    inl = check_inlining(f, args...)

    overall = String[]
    # Priority 1: specialization widening
    for slot in spec.slots
        if slot.classification == SLOT_WIDENED_CALLABLE
            push!(overall, "[HIGH] Slot $(slot.position): $(slot.suggestion)")
        end
    end
    # Priority 2: inline-dependent allocations
    if allocs.metrics.inline_saves_allocs
        push!(overall, "[HIGH] Allocations disappear when inlined — add @inline to this function")
    end
    # Priority 3: small non-inlined callees
    for c in inl.non_inlined_callees
        if c.is_small && !isempty(c.suggestion)
            push!(overall, "[MED] $(c.suggestion)")
        end
    end
    # Priority 4: dynamic dispatch
    if allocs.metrics.dynamic_dispatch_calls > 0
        push!(overall, "[MED] Dynamic dispatch in LLVM IR — check type stability")
    end
    # Priority 5: remaining allocations
    if allocs.metrics.measured_allocs > 0 && !allocs.metrics.inline_saves_allocs
        push!(overall, "[LOW] $(allocs.metrics.measured_allocs) allocation(s) measured at runtime")
    end

    notes = String[]
    return PerfDiagnostic(sig, spec, allocs, inl, overall, notes)
end

"""
    compare_perf(f, args_a::Tuple, args_b::Tuple) -> PerfComparison

Compare two calls to the same function with different argument types.
"""
function compare_perf(f, args_a::Tuple, args_b::Tuple)
    spec_a = check_specialization(f, args_a...)
    spec_b = check_specialization(f, args_b...)
    allocs_a = check_allocations(f, args_a...)
    allocs_b = check_allocations(f, args_b...)
    inl_a = check_inlining(f, args_a...)
    inl_b = check_inlining(f, args_b...)

    # Align slots
    n = max(length(spec_a.slots), length(spec_b.slots))
    slot_comparisons = SlotComparison[]
    for i in 1:n
        sa = i <= length(spec_a.slots) ? spec_a.slots[i] : nothing
        sb = i <= length(spec_b.slots) ? spec_b.slots[i] : nothing
        push!(slot_comparisons, SlotComparison(
            i,
            sa !== nothing ? sa.runtime_type : Any,
            sb !== nothing ? sb.runtime_type : Any,
            sa !== nothing ? sa.compiled_type : nothing,
            sb !== nothing ? sb.compiled_type : nothing,
            sa !== nothing ? sa.classification : SLOT_OK,
            sb !== nothing ? sb.classification : SLOT_OK,
        ))
    end

    # Build verdict
    a_widened = count(s -> s.classification_a == SLOT_WIDENED_CALLABLE, slot_comparisons)
    b_widened = count(s -> s.classification_b == SLOT_WIDENED_CALLABLE, slot_comparisons)
    verdict = if a_widened > 0 && b_widened == 0
        "Call A has $(a_widened) widened callable slot(s); Call B has none. This likely explains the performance difference."
    elseif b_widened > 0 && a_widened == 0
        "Call B has $(b_widened) widened callable slot(s); Call A has none."
    elseif a_widened > 0 && b_widened > 0
        "Both calls have widened callable slots."
    else
        alloc_diff = allocs_a.metrics.measured_allocs - allocs_b.metrics.measured_allocs
        if alloc_diff > 0
            "Call A allocates more ($(allocs_a.metrics.measured_allocs) vs $(allocs_b.metrics.measured_allocs))."
        elseif alloc_diff < 0
            "Call B allocates more ($(allocs_b.metrics.measured_allocs) vs $(allocs_a.metrics.measured_allocs))."
        else
            "No significant difference detected at the compiler boundary."
        end
    end

    suggestions = String[]
    for sc in slot_comparisons
        if sc.classification_a == SLOT_WIDENED_CALLABLE && sc.classification_b == SLOT_OK
            push!(suggestions, "Slot $(sc.position): A uses $(sc.runtime_type_a) (widened), B uses $(sc.runtime_type_b) (concrete). Use B's pattern.")
        end
    end

    return PerfComparison(
        spec_a.runtime_sig, spec_b.runtime_sig,
        slot_comparisons,
        inl_a.invoke_count_optimized, inl_b.invoke_count_optimized,
        allocs_a.metrics.heap_alloc_calls, allocs_b.metrics.heap_alloc_calls,
        allocs_a.metrics.measured_allocs, allocs_b.metrics.measured_allocs,
        verdict, suggestions,
    )
end

"""Quick diagnostic with immediate display."""
function quick_diagnose(f, args...; io=stdout)
    report = diagnose_perf(f, args...)
    show_report(report; io)
    return report
end

# =============================================================================
# Display Functions
# =============================================================================

function show_specialization(report::SpecializationReport; io=stdout)
    println(io, "--- Specialization Widening ---")
    println(io, "Method: $(report.method)")
    println(io, "Vararg: $(report.has_vararg)")
    if report.method_instance !== nothing
        println(io, "Compiled as: $(report.compiled_sig)")
    else
        println(io, "Compiled as: (no MethodInstance found)")
    end
    println(io)
    # Header
    @printf(io, "  %-5s  %-30s  %-30s  %-20s  %s\n",
        "Slot", "Runtime Type", "Compiled Type", "Classification", "Note")
    println(io, "  ", "-"^100)
    for slot in report.slots
        rt = _short_type_name(slot.runtime_type)
        ct = _short_type_name(slot.compiled_type)
        cls = _classification_str(slot.classification)
        flag = slot.classification == SLOT_WIDENED_CALLABLE ? " [!]" : ""
        @printf(io, "  %-5d  %-30s  %-30s  %-20s  %s\n",
            slot.position, rt, ct, cls * flag, slot.suggestion)
    end
    for note in report.notes
        println(io, "\n  ", note)
    end
end

function show_allocations(report::AllocationReport; io=stdout)
    println(io, "--- Allocation / SROA Analysis ---")
    m = report.metrics
    println(io, "  Heap alloc calls (LLVM):  $(m.heap_alloc_calls)")
    println(io, "  Box calls (LLVM):         $(m.box_calls)")
    println(io, "  Dynamic dispatch (LLVM):  $(m.dynamic_dispatch_calls)")
    println(io, "  GC frame refs (LLVM):     $(m.gc_frame_slots)")
    println(io, "  Measured allocs:          $(m.measured_allocs)")
    println(io, "  Measured allocs (noinl):  $(m.measured_allocs_noinline)")
    if m.inline_saves_allocs
        println(io, "  Inline-sensitive:         YES — allocations disappear when inlined [!]")
    else
        println(io, "  Inline-sensitive:         no")
    end
    if !isempty(report.matching_lines)
        println(io, "\n  Matching LLVM lines:")
        for (i, line) in enumerate(report.matching_lines)
            i > 5 && (println(io, "  ... ($(length(report.matching_lines) - 5) more)"); break)
            println(io, "    ", strip(line))
        end
    end
    for s in report.suggestions
        println(io, "\n  Suggestion: ", s)
    end
    for note in report.notes
        println(io, "\n  Note: ", note)
    end
end

function show_inlining(report::InliningReport; io=stdout)
    println(io, "--- Inlining Analysis ---")
    println(io, "  Invoke count (optimized):   $(report.invoke_count_optimized)")
    println(io, "  Invoke count (unoptimized): $(report.invoke_count_unoptimized)")
    if !isempty(report.non_inlined_callees)
        small = filter(c -> c.is_small, report.non_inlined_callees)
        if !isempty(small)
            println(io, "\n  Suspicious non-inlined callees:")
            for c in small
                stmts = c.statement_count !== nothing ? "$(c.statement_count) stmts" : "? stmts"
                println(io, "    - $(c.callee_name) ($stmts)")
                !isempty(c.suggestion) && println(io, "      $(c.suggestion)")
            end
        end
    end
    for s in report.suggestions
        println(io, "\n  Suggestion: ", s)
    end
    for note in report.notes
        println(io, "\n  Note: ", note)
    end
end

function show_report(report::PerfDiagnostic; io=stdout)
    println(io, "=" ^ 70)
    println(io, "Performance Diagnostic: $(report.sig)")
    println(io, "=" ^ 70)
    println(io)
    show_specialization(report.specialization; io)
    println(io)
    show_allocations(report.allocations; io)
    println(io)
    show_inlining(report.inlining; io)
    if !isempty(report.overall_suggestions)
        println(io, "\n", "=" ^ 70)
        println(io, "Overall Suggestions (prioritized)")
        println(io, "=" ^ 70)
        for (i, s) in enumerate(report.overall_suggestions)
            println(io, "  $(i). $(s)")
        end
    end
    println(io)
end

function show_comparison(comp::PerfComparison; io=stdout)
    println(io, "=" ^ 70)
    println(io, "Performance Comparison")
    println(io, "  A: $(comp.sig_a)")
    println(io, "  B: $(comp.sig_b)")
    println(io, "=" ^ 70)

    println(io, "\n--- Per-Slot Comparison ---")
    @printf(io, "  %-5s  %-25s  %-25s  %-15s  %-15s  %s\n",
        "Slot", "A Runtime", "B Runtime", "A Class", "B Class", "Delta")
    println(io, "  ", "-"^95)
    for sc in comp.slots
        rta = _short_type_name(sc.runtime_type_a)
        rtb = _short_type_name(sc.runtime_type_b)
        ca = _classification_str(sc.classification_a)
        cb = _classification_str(sc.classification_b)
        delta = if sc.classification_a != sc.classification_b
            sc.classification_a == SLOT_WIDENED_CALLABLE ? "A WIDENED" :
            sc.classification_b == SLOT_WIDENED_CALLABLE ? "B WIDENED" : "differs"
        else
            "-"
        end
        @printf(io, "  %-5d  %-25s  %-25s  %-15s  %-15s  %s\n",
            sc.position, rta, rtb, ca, cb, delta)
    end

    println(io, "\n--- Metrics ---")
    @printf(io, "  %-25s  %8s  %8s  %8s\n", "Metric", "A", "B", "Delta")
    println(io, "  ", "-"^55)
    _metric_row(io, "Invoke count", comp.invoke_count_a, comp.invoke_count_b)
    _metric_row(io, "Heap allocs (LLVM)", comp.heap_alloc_calls_a, comp.heap_alloc_calls_b)
    _metric_row(io, "Measured allocs", comp.measured_allocs_a, comp.measured_allocs_b)

    println(io, "\n  Verdict: $(comp.verdict)")
    for s in comp.suggestions
        println(io, "  Suggestion: ", s)
    end
    println(io)
end

# --- Display helpers ---

function _short_type_name(@nospecialize(T::Type))
    s = string(T)
    length(s) <= 30 && return s
    # Truncate long type names
    return s[1:27] * "..."
end
_short_type_name(::Nothing) = "N/A"

function _classification_str(c::SlotClassification)
    c == SLOT_OK && return "OK"
    c == SLOT_EXPECTED_ABSTRACT && return "expected_abstract"
    c == SLOT_WIDENED_CALLABLE && return "WIDENED_CALLABLE"
    c == SLOT_WIDENED_OTHER && return "widened_other"
    return string(c)
end

function _metric_row(io, name, a::Int, b::Int)
    delta = a - b
    ds = delta == 0 ? "-" : (delta > 0 ? "+$delta" : "$delta")
    @printf(io, "  %-25s  %8d  %8d  %8s\n", name, a, b, ds)
end

# =============================================================================
# Load confirmation
# =============================================================================

println("Perf Diagnose module loaded. Try:")
println("  report = diagnose_perf(f, args...)")
println("  show_report(report)")
println("  comp = compare_perf(f, (slow_args...,), (fast_args...,))")
println("  show_comparison(comp)")
