#=
Skill Utilities for Mooncake.jl
================================
Developer tools for IR inspection and performance diagnostics, used by Claude Code skills.

This file provides two main tool sets:
1. IR Inspection — view and diff IR at each stage of the AD pipeline
2. Performance Diagnostics — detect compiler-boundary issues (specialization widening,
   allocation/SROA failures, inlining failures)

These are NOT part of the public API and may change in non-breaking releases.
=#

using InteractiveUtils: code_llvm, code_typed
using Printf: @printf

# =============================================================================
# IR Inspection
# =============================================================================

struct StageMeta
    block_count::Int
    inst_count::Int
    edge_count::Int
    has_ssa::Bool
    uses_bbcode_ids::Bool
    valid_worlds::Union{UnitRange{UInt},Nothing}
    misty_world::Union{UInt,Nothing}
end

function StageMeta(;
    block_count=0,
    inst_count=0,
    edge_count=0,
    has_ssa=true,
    uses_bbcode_ids=false,
    valid_worlds=nothing,
    misty_world=nothing,
)
    return StageMeta(
        block_count,
        inst_count,
        edge_count,
        has_ssa,
        uses_bbcode_ids,
        valid_worlds,
        misty_world,
    )
end

struct IRStage
    name::Symbol
    ir::Any
    text::String
    meta::StageMeta
end

struct RuleInspection
    rule::Any
    misty_stages::Dict{Symbol,IRStage}
end

struct IRInspection
    mode::Symbol
    sig::Type
    world::UInt
    stages::Dict{Symbol,IRStage}
    stage_order::Vector{Symbol}
    stage_graph::Vector{Pair{Symbol,Symbol}}
    diffs::Dict{Pair{Symbol,Symbol},String}
    cfgs::Dict{Symbol,String}
    rules::Vector{RuleInspection}
    notes::Vector{String}
end

struct WorldAgeReport
    inspection_world::UInt
    method_worlds::Union{UnitRange{UInt},Nothing}
    stage_worlds::Dict{Symbol,Union{UInt,Nothing}}
    rule_worlds::Vector{Union{UInt,Nothing}}
    mismatches::Vector{String}
end

# --- Stage Graphs ---

forward_stage_order() = [:raw, :normalized, :bbcode, :dual_ir, :optimized]

function forward_stage_graph()
    return [
        :raw => :normalized,
        :normalized => :bbcode,
        :bbcode => :dual_ir,
        :dual_ir => :optimized,
    ]
end

function reverse_stage_order()
    return [:raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs]
end

function reverse_stage_graph()
    return [
        :raw => :normalized,
        :normalized => :bbcode,
        :bbcode => :fwd_ir,
        :bbcode => :rvs_ir,
        :fwd_ir => :optimized_fwd,
        :rvs_ir => :optimized_rvs,
    ]
end

# --- IR Rendering ---

function render_ir(ir::IRCode)::String
    io = IOBuffer()
    show(io, ir)
    return String(take!(io))
end

function render_ir(bb::BBCode)::String
    io = IOBuffer()
    for (i, block) in enumerate(bb.blocks)
        println(io, "Block $(i) (id=$(block.id)):")
        for (id, inst) in zip(block.inst_ids, block.insts)
            println(io, "  $id: $(inst.stmt) :: $(inst.type)")
        end
    end
    return String(take!(io))
end

function render_ir(x)::String
    io = IOBuffer()
    show(io, MIME"text/plain"(), x)
    return String(take!(io))
end

# --- Metadata Extraction ---

function extract_meta(ir::IRCode)::StageMeta
    cfg = ir.cfg
    block_count = length(cfg.blocks)
    inst_count = length(ir.stmts)
    edge_count = sum(length(b.succs) for b in cfg.blocks)
    valid_worlds = nothing
    if hasproperty(ir, :valid_worlds)
        vw = ir.valid_worlds
        valid_worlds = UInt(CC.min_world(vw)):UInt(CC.max_world(vw))
    end
    return StageMeta(;
        block_count=block_count,
        inst_count=inst_count,
        edge_count=edge_count,
        has_ssa=true,
        uses_bbcode_ids=false,
        valid_worlds=valid_worlds,
    )
end

function extract_meta(bb::BBCode)::StageMeta
    block_count = length(bb.blocks)
    inst_count = sum(length(b.inst_ids) for b in bb.blocks)
    succs = BasicBlockCode.compute_all_successors(bb)
    edge_count = sum(length(v) for v in values(succs))
    return StageMeta(;
        block_count=block_count,
        inst_count=inst_count,
        edge_count=edge_count,
        has_ssa=false,
        uses_bbcode_ids=true,
    )
end

extract_meta(x) = StageMeta()

# --- CFG to DOT ---

function cfg_to_dot(ir::IRCode; name="cfg")::String
    cfg = ir.cfg
    io = IOBuffer()
    println(io, "digraph $name {")
    println(io, "  rankdir=TB;")
    println(io, "  node [shape=box];")
    for (i, block) in enumerate(cfg.blocks)
        label = "Block $i\\n[$(block.stmts.start):$(block.stmts.stop)]"
        println(io, "  b$i [label=\"$label\"];")
    end
    for (i, block) in enumerate(cfg.blocks)
        for succ in block.succs
            println(io, "  b$i -> b$succ;")
        end
    end
    println(io, "}")
    return String(take!(io))
end

function cfg_to_dot(bb::BBCode; name="cfg")::String
    io = IOBuffer()
    println(io, "digraph $name {")
    println(io, "  rankdir=TB;")
    println(io, "  node [shape=box];")
    id_to_idx = Dict(b.id => i for (i, b) in enumerate(bb.blocks))
    succs = BasicBlockCode.compute_all_successors(bb)
    for (i, block) in enumerate(bb.blocks)
        label = "Block $i\\nID=$(block.id.id)\\n$(length(block.inst_ids)) insts"
        println(io, "  b$i [label=\"$label\"];")
    end
    for block in bb.blocks
        from_idx = id_to_idx[block.id]
        for succ_id in succs[block.id]
            to_idx = id_to_idx[succ_id]
            println(io, "  b$from_idx -> b$to_idx;")
        end
    end
    println(io, "}")
    return String(take!(io))
end

# --- Text Diff ---

function simple_diff(text1::String, text2::String; context=3)::String
    lines1 = split(text1, '\n')
    lines2 = split(text2, '\n')
    io = IOBuffer()
    println(io, "--- stage1")
    println(io, "+++ stage2")
    max_lines = max(length(lines1), length(lines2))
    for i in 1:max_lines
        l1 = i <= length(lines1) ? lines1[i] : ""
        l2 = i <= length(lines2) ? lines2[i] : ""
        if l1 != l2
            if !isempty(l1)
                println(io, "-$l1")
            end
            if !isempty(l2)
                println(io, "+$l2")
            end
        end
    end
    return String(take!(io))
end

# --- Main Inspection ---

"""
    inspect_ir(f, args...; kwargs...) -> IRInspection

!!! warning
    This is not part of the public interface of Mooncake.

Inspect IR transformations for a function call. Returns an `IRInspection` struct
containing all stages, diffs, and CFG data.

# Keyword Arguments
- `mode::Symbol = :reverse`: `:forward` or `:reverse` mode
- `world::UInt = Base.get_world_counter()`: World age for compilation
- `optimize::Bool = true`: Whether to run optimization passes
- `do_inline::Bool = true`: Whether to inline during optimization
- `compute_diffs::Bool = true`: Whether to compute diffs between stages
- `compute_cfgs::Bool = true`: Whether to compute CFG DOT strings
- `debug_mode::Bool = false`: Enable Mooncake debug mode
"""
function inspect_ir(
    f,
    args...;
    mode::Symbol=:reverse,
    world::UInt=Base.get_world_counter(),
    optimize::Bool=true,
    do_inline::Bool=true,
    compute_diffs::Bool=true,
    compute_cfgs::Bool=true,
    debug_mode::Bool=false,
)
    sig = Tuple{typeof(f),map(typeof, args)...}
    interp_mode = mode == :forward ? ForwardMode : ReverseMode
    interp = get_interpreter(interp_mode)

    stages = Dict{Symbol,IRStage}()
    notes = String[]

    seed_id!()

    try
        # Stage 1: Raw IR
        raw_ir, _ = lookup_ir(interp, sig)
        stages[:raw] = IRStage(:raw, raw_ir, render_ir(raw_ir), extract_meta(raw_ir))

        # Stage 2: Normalized IR
        isva, spnames = is_vararg_and_sparam_names(sig)
        normalized_ir = CC.copy(raw_ir)
        normalise!(normalized_ir, spnames)
        stages[:normalized] = IRStage(
            :normalized,
            normalized_ir,
            render_ir(normalized_ir),
            extract_meta(normalized_ir),
        )

        # Stage 3: BBCode
        bbcode = BBCode(normalized_ir)
        bbcode = remove_unreachable_blocks!(bbcode)
        stages[:bbcode] =
            IRStage(:bbcode, bbcode, render_ir(bbcode), extract_meta(bbcode))

        # Mode-specific stages
        if mode == :forward
            dual_ir, captures, info =
                generate_dual_ir(interp, sig; debug_mode, do_inline=false)
            stages[:dual_ir] = IRStage(
                :dual_ir, dual_ir, render_ir(dual_ir), extract_meta(dual_ir)
            )
            if optimize
                opt_ir = optimise_ir!(CC.copy(dual_ir); do_inline)
                stages[:optimized] = IRStage(
                    :optimized, opt_ir, render_ir(opt_ir), extract_meta(opt_ir)
                )
            end
        else
            dri = generate_ir(interp, sig; debug_mode, do_inline=false)
            stages[:fwd_ir] = IRStage(
                :fwd_ir, dri.fwd_ir, render_ir(dri.fwd_ir), extract_meta(dri.fwd_ir)
            )
            stages[:rvs_ir] = IRStage(
                :rvs_ir, dri.rvs_ir, render_ir(dri.rvs_ir), extract_meta(dri.rvs_ir)
            )
            if optimize
                opt_fwd = optimise_ir!(CC.copy(dri.fwd_ir); do_inline)
                opt_rvs = optimise_ir!(CC.copy(dri.rvs_ir); do_inline)
                stages[:optimized_fwd] = IRStage(
                    :optimized_fwd, opt_fwd, render_ir(opt_fwd), extract_meta(opt_fwd)
                )
                stages[:optimized_rvs] = IRStage(
                    :optimized_rvs, opt_rvs, render_ir(opt_rvs), extract_meta(opt_rvs)
                )
            end
        end
    catch e
        push!(notes, "Error during IR generation: $e")
        @error "IR inspection failed" exception = (e, catch_backtrace())
    end

    stage_order = mode == :forward ? forward_stage_order() : reverse_stage_order()
    stage_graph = mode == :forward ? forward_stage_graph() : reverse_stage_graph()
    stage_order = filter(s -> haskey(stages, s), stage_order)
    stage_graph =
        filter(p -> haskey(stages, p.first) && haskey(stages, p.second), stage_graph)

    diffs = Dict{Pair{Symbol,Symbol},String}()
    if compute_diffs
        for (from, to) in stage_graph
            if haskey(stages, from) && haskey(stages, to)
                diffs[from => to] = simple_diff(stages[from].text, stages[to].text)
            end
        end
    end

    cfgs = Dict{Symbol,String}()
    if compute_cfgs
        for (name, stage) in stages
            try
                cfgs[name] = cfg_to_dot(stage.ir; name=string(name))
            catch
            end
        end
    end

    return IRInspection(
        mode, sig, world, stages, stage_order, stage_graph, diffs, cfgs,
        RuleInspection[], notes,
    )
end

# --- Display Functions ---

"""
    show_ir(ins::IRInspection; stages=:all, io=stdout)

Display IR stages from an inspection result.
"""
function show_ir(ins::IRInspection; stages=:all, io=stdout)
    stage_list = if stages == :all
        ins.stage_order
    elseif stages isa Symbol
        [stages]
    else
        collect(stages)
    end

    println(io, "=" ^ 60)
    println(io, "IR Inspection: $(ins.sig)")
    println(io, "Mode: $(ins.mode), World: $(ins.world)")
    println(io, "=" ^ 60)

    for name in stage_list
        if haskey(ins.stages, name)
            stage = ins.stages[name]
            println(io, "\n", "-" ^ 40)
            println(io, "Stage: $name")
            println(
                io,
                "Blocks: $(stage.meta.block_count), Insts: $(stage.meta.inst_count), Edges: $(stage.meta.edge_count)",
            )
            if stage.meta.valid_worlds !== nothing
                println(io, "Valid worlds: $(stage.meta.valid_worlds)")
            end
            println(io, "-" ^ 40)
            println(io, stage.text)
        end
    end
end

"""
    show_stage(ins::IRInspection, stage::Symbol; io=stdout)

Display a single stage.
"""
function show_stage(ins::IRInspection, stage::Symbol; io=stdout)
    return show_ir(ins; stages=[stage], io)
end

"""
    diff_ir(ins::IRInspection; from::Symbol, to::Symbol)

Get the diff between two stages.
"""
function diff_ir(ins::IRInspection; from::Symbol, to::Symbol)
    key = from => to
    if haskey(ins.diffs, key)
        return ins.diffs[key]
    else
        if haskey(ins.stages, from) && haskey(ins.stages, to)
            return simple_diff(ins.stages[from].text, ins.stages[to].text)
        else
            return "Stages not found: $from, $to"
        end
    end
end

"""
    show_diff(ins::IRInspection; from::Symbol, to::Symbol, io=stdout)

Display diff between two stages.
"""
function show_diff(ins::IRInspection; from::Symbol, to::Symbol, io=stdout)
    println(io, "=" ^ 60)
    println(io, "Diff: $from → $to")
    println(io, "=" ^ 60)
    return println(io, diff_ir(ins; from, to))
end

"""
    show_all_diffs(ins::IRInspection; io=stdout)

Display all consecutive diffs.
"""
function show_all_diffs(ins::IRInspection; io=stdout)
    for (from, to) in ins.stage_graph
        show_diff(ins; from, to, io)
        println(io)
    end
end

"""
    cfg_dot(ins::IRInspection, stage::Symbol)

Get DOT string for a stage's CFG.
"""
cfg_dot(ins::IRInspection, stage::Symbol) = get(ins.cfgs, stage, "")

"""
    world_age_info(ins::IRInspection) -> WorldAgeReport

Extract world age information from inspection.
"""
function world_age_info(ins::IRInspection)
    stage_worlds = Dict{Symbol,Union{UInt,Nothing}}()
    for (name, stage) in ins.stages
        stage_worlds[name] =
            stage.meta.valid_worlds !== nothing ? first(stage.meta.valid_worlds) : nothing
    end

    mismatches = String[]
    for (name, w) in stage_worlds
        if w !== nothing && w > ins.world
            push!(mismatches, "Stage $name has world $w > inspection world $(ins.world)")
        end
    end

    return WorldAgeReport(ins.world, nothing, stage_worlds, Union{UInt,Nothing}[], mismatches)
end

"""
    show_world_info(ins::IRInspection; io=stdout)

Display world age information.
"""
function show_world_info(ins::IRInspection; io=stdout)
    report = world_age_info(ins)
    println(io, "=" ^ 60)
    println(io, "World Age Report")
    println(io, "=" ^ 60)
    println(io, "Inspection world: $(report.inspection_world)")
    println(io, "\nStage worlds:")
    for (name, w) in report.stage_worlds
        println(io, "  $name: $(w === nothing ? "N/A" : w)")
    end
    if !isempty(report.mismatches)
        println(io, "\nMismatches:")
        for m in report.mismatches
            println(io, "  ⚠ $m")
        end
    end
end

# --- Convenience Functions ---

"""
    inspect_fwd(f, args...; kwargs...)

Shorthand for `inspect_ir` with forward mode.
"""
inspect_fwd(f, args...; kwargs...) = inspect_ir(f, args...; mode=:forward, kwargs...)

"""
    inspect_rvs(f, args...; kwargs...)

Shorthand for `inspect_ir` with reverse mode.
"""
inspect_rvs(f, args...; kwargs...) = inspect_ir(f, args...; mode=:reverse, kwargs...)

"""
    quick_inspect(f, args...; mode=:reverse)

Quick inspection with immediate display.
"""
function quick_inspect(f, args...; mode=:reverse, stages=:all)
    ins = inspect_ir(f, args...; mode)
    show_ir(ins; stages)
    return ins
end

"""
    write_ir(ins::IRInspection, outdir::String)

Write all stages, diffs, and CFGs to files.
"""
function write_ir(ins::IRInspection, outdir::String)
    mkpath(outdir)
    for (name, stage) in ins.stages
        open(joinpath(outdir, "$(name).txt"), "w") do f
            println(f, stage.text)
        end
    end
    for ((from, to), diff) in ins.diffs
        open(joinpath(outdir, "diff_$(from)_$(to).txt"), "w") do f
            println(f, diff)
        end
    end
    for (name, dot) in ins.cfgs
        open(joinpath(outdir, "cfg_$(name).dot"), "w") do f
            println(f, dot)
        end
    end
    return println(
        "Wrote $(length(ins.stages)) stages, $(length(ins.diffs)) diffs, $(length(ins.cfgs)) CFGs to $outdir",
    )
end

# =============================================================================
# Performance Diagnostics
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

# --- Internal Helpers ---

function _is_callable_type(@nospecialize(T::Type))::Bool
    T <: Function && return true
    try
        return !isempty(methods(T.instance)) || hasmethod(T, Tuple{Vararg{Any}})
    catch
        return false
    end
end

function _get_specializations(method::Method)
    mis = Core.MethodInstance[]
    for mi in Base.specializations(method)
        mi === nothing && continue
        push!(mis, mi)
    end
    return mis
end

function _find_method_instance(f, args...)
    tt = Tuple{typeof(f),map(typeof, args)...}
    try
        f(args...)
    catch
    end
    meth = which(f, typeof.(args))
    for mi in _get_specializations(meth)
        if mi.specTypes === tt
            return mi
        end
    end
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
    while length(declared) < nargs
        push!(declared, Any)
    end
    return declared
end

function _classify_slot(
    position::Int,
    @nospecialize(runtime_type),
    @nospecialize(compiled_type),
    @nospecialize(declared_type),
    is_vararg::Bool,
)::SlotInfo
    rt = runtime_type isa TypeVar ? runtime_type.ub : runtime_type
    ct = compiled_type isa TypeVar ? compiled_type.ub : compiled_type
    dt = declared_type isa TypeVar ? declared_type.ub : declared_type

    rt isa Type || (rt = Any)
    ct isa Type || (ct = Any)
    dt isa Type || (dt = Any)

    if rt === ct
        return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_OK, "")
    end

    expected_abstract = (
        dt isa DataType &&
        isabstracttype(dt) &&
        dt !== Any &&
        ct === dt
    )
    if expected_abstract
        return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_EXPECTED_ABSTRACT, "")
    end

    is_callable = try
        rt <: Function
    catch
        false
    end
    if !is_callable
        try
            is_callable =
                isconcretetype(rt) && hasmethod(rt.instance, Tuple{Vararg{Any}})
        catch
        end
    end

    if is_callable && (ct === Function || ct === Any)
        suggestion = if is_vararg
            "Callable in Vararg position widened to $(ct). Fix: wrap in a callable struct, or add @inline to the caller"
        else
            "Callable argument widened to $(ct). Fix: add explicit type parameter (::F where {F}) or add @inline"
        end
        return SlotInfo(
            position, rt, ct, dt, is_vararg, SLOT_WIDENED_CALLABLE, suggestion
        )
    end

    widened = try
        rt !== ct && rt <: ct
    catch
        false
    end
    if widened
        return SlotInfo(
            position, rt, ct, dt, is_vararg, SLOT_WIDENED_OTHER,
            "Runtime type widened: $(rt) -> $(ct)",
        )
    end

    return SlotInfo(position, rt, ct, dt, is_vararg, SLOT_OK, "")
end

# Allocation counting: reuse TestUtils.__count_allocs from test_utils.jl (a @generated function
# that avoids Vararg specialization issues) and wrap with warmup + min-of-5.
function _count_allocs_min(f, args...)
    try
        f(args...)
    catch
    end
    best = typemax(Int)
    for _ in 1:5
        c = TestUtils.__count_allocs(f, args...)
        c < best && (best = c)
    end
    return best
end

function _parse_llvm_allocs(llvm_text::String)
    lines = split(llvm_text, '\n')
    heap_re =
        r"@ij?l_gc_pool_alloc\b|@julia\.gc_alloc_obj\b|@julia\.gc_alloc_bytes\b|@jl_alloc_array"
    box_re = r"@jl_box_[A-Za-z0-9_]+\b"
    dispatch_re = r"@jl_apply_generic\b|@jl_invoke\b"
    gcframe_re = r"gcframe|julia\.new_gc_frame"

    heap = count(l -> occursin(heap_re, l), lines)
    box = count(l -> occursin(box_re, l), lines)
    dispatch = count(l -> occursin(dispatch_re, l), lines)
    gc_slots = count(l -> occursin(gcframe_re, l), lines)

    matching =
        filter(l -> any(re -> occursin(re, l), [heap_re, box_re, dispatch_re]), lines)

    return (
        heap_alloc=heap,
        box_calls=box,
        dynamic_dispatch=dispatch,
        gc_slots=gc_slots,
        matching_lines=String.(matching),
    )
end

function _count_invokes(ci::Core.CodeInfo)::Int
    return count(st -> st isa Expr && st.head === :invoke, ci.code)
end

function _extract_non_inlined(ci_opt::Core.CodeInfo; max_stmts::Int=20)
    callees = NonInlinedCallee[]
    for st in ci_opt.code
        st isa Expr && st.head === :invoke || continue
        mi = st.args[1]
        mi isa Core.MethodInstance || continue
        m = mi.def
        m isa Method || continue

        name = string(m.name)
        stmt_count = nothing
        try
            src = Base.uncompressed_ast(m)
            stmt_count = count(s -> !(s isa Core.LineNumberNode), src.code)
        catch
        end

        is_small = stmt_count !== nothing && stmt_count <= max_stmts
        is_noinline = false

        suggestion = if is_small
            "Small callee '$name' ($stmt_count stmts) not inlined. Consider adding @inline or checking specialization"
        else
            ""
        end

        push!(
            callees, NonInlinedCallee(name, stmt_count, is_noinline, is_small, suggestion)
        )
    end
    return callees
end

# --- Public API: Individual Detectors ---

"""
    check_specialization(f, args...) -> SpecializationReport

!!! warning
    This is not part of the public interface of Mooncake.

Detect specialization widening at the Julia compiler boundary.
"""
function check_specialization(f, args...)
    tt = Tuple{typeof(f),map(typeof, args)...}
    meth = which(f, typeof.(args))
    mi = _find_method_instance(f, args...)

    nparams = length(tt.parameters)
    declared = _expand_vararg_sig(meth, nparams)
    compiled = if mi !== nothing
        collect(Base.unwrap_unionall(mi.specTypes).parameters)
    else
        fill(Any, nparams)
    end
    runtime = collect(tt.parameters)

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
        push!(
            notes,
            "Specialization widening detected. Julia compiled a less-specific method instance than the runtime types warrant.",
        )
    end

    return SpecializationReport(
        meth,
        mi,
        tt,
        mi !== nothing ? mi.specTypes : nothing,
        slots,
        has_widening,
        meth.isva,
        notes,
    )
end

"""
    check_allocations(f, args...) -> AllocationReport

!!! warning
    This is not part of the public interface of Mooncake.

Analyze allocation behavior at the Julia compiler boundary.
"""
function check_allocations(f, args...)
    argtt = Tuple{map(typeof, args)...}
    sig = Tuple{typeof(f),argtt.parameters...}

    llvm_text = try
        io = IOBuffer()
        code_llvm(io, f, argtt; raw=true, debuginfo=:none)
        String(take!(io))
    catch e
        ""
    end

    parsed = _parse_llvm_allocs(llvm_text)

    measured = try
        _count_allocs_min(f, args...)
    catch
        -1
    end

    noinline_measured = try
        wrapper = @noinline (f, args...) -> f(args...)
        wrapper(f, args...)
        _count_allocs_min(wrapper, f, args...)
    catch
        -1
    end

    inline_saves = measured >= 0 && noinline_measured >= 0 && noinline_measured > measured

    metrics = AllocationMetrics(
        parsed.heap_alloc,
        parsed.box_calls,
        parsed.dynamic_dispatch,
        parsed.gc_slots,
        measured,
        noinline_measured,
        inline_saves,
    )

    suggestions = String[]
    if metrics.dynamic_dispatch_calls > 0
        push!(
            suggestions,
            "Dynamic dispatch detected in LLVM IR ($(metrics.dynamic_dispatch_calls) calls). Ensure all call targets have concrete types.",
        )
    end
    if metrics.box_calls > 0
        push!(
            suggestions,
            "Boxing detected ($(metrics.box_calls) calls). This suggests type instability at the compiler boundary.",
        )
    end
    if inline_saves
        push!(
            suggestions,
            "Allocations depend on inlining ($(measured) direct vs $(noinline_measured) with @noinline barrier). Add @inline to enable SROA.",
        )
    end
    if metrics.heap_alloc_calls > 0 && measured == 0
        push!(
            suggestions,
            "LLVM IR shows allocation calls but runtime reports zero. Likely optimized away.",
        )
    end

    notes = String[]
    if isempty(llvm_text)
        push!(notes, "Could not generate LLVM IR for this function.")
    end

    return AllocationReport(sig, metrics, parsed.matching_lines, suggestions, notes)
end

"""
    check_inlining(f, args...) -> InliningReport

!!! warning
    This is not part of the public interface of Mooncake.

Detect inlining failures by comparing optimized vs unoptimized code_typed output.
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
        push!(
            suggestions,
            "$small_count small callee(s) not inlined. This may indicate specialization widening prevented inlining.",
        )
    end
    if invoke_opt > invoke_unopt
        push!(
            notes,
            "More invokes after optimization ($invoke_opt) than before ($invoke_unopt) — unusual, may indicate issues.",
        )
    end

    return InliningReport(sig, invoke_opt, invoke_unopt, callees, suggestions, notes)
end

# --- Full Diagnostic and Compare ---

"""
    diagnose_perf(f, args...) -> PerfDiagnostic

!!! warning
    This is not part of the public interface of Mooncake.

Run all three detectors and produce a prioritized diagnostic report.
"""
function diagnose_perf(f, args...)
    sig = Tuple{typeof(f),map(typeof, args)...}
    spec = check_specialization(f, args...)
    allocs = check_allocations(f, args...)
    inl = check_inlining(f, args...)

    overall = String[]
    for slot in spec.slots
        if slot.classification == SLOT_WIDENED_CALLABLE
            push!(overall, "[HIGH] Slot $(slot.position): $(slot.suggestion)")
        end
    end
    if allocs.metrics.inline_saves_allocs
        push!(
            overall,
            "[HIGH] Allocations disappear when inlined — add @inline to this function",
        )
    end
    for c in inl.non_inlined_callees
        if c.is_small && !isempty(c.suggestion)
            push!(overall, "[MED] $(c.suggestion)")
        end
    end
    if allocs.metrics.dynamic_dispatch_calls > 0
        push!(overall, "[MED] Dynamic dispatch in LLVM IR — check type stability")
    end
    if allocs.metrics.measured_allocs > 0 && !allocs.metrics.inline_saves_allocs
        push!(
            overall,
            "[LOW] $(allocs.metrics.measured_allocs) allocation(s) measured at runtime",
        )
    end

    return PerfDiagnostic(sig, spec, allocs, inl, overall, String[])
end

"""
    compare_perf(f, args_a::Tuple, args_b::Tuple) -> PerfComparison

!!! warning
    This is not part of the public interface of Mooncake.

Compare two calls to the same function with different argument types.
"""
function compare_perf(f, args_a::Tuple, args_b::Tuple)
    spec_a = check_specialization(f, args_a...)
    spec_b = check_specialization(f, args_b...)
    allocs_a = check_allocations(f, args_a...)
    allocs_b = check_allocations(f, args_b...)
    inl_a = check_inlining(f, args_a...)
    inl_b = check_inlining(f, args_b...)

    n = max(length(spec_a.slots), length(spec_b.slots))
    slot_comparisons = SlotComparison[]
    for i in 1:n
        sa = i <= length(spec_a.slots) ? spec_a.slots[i] : nothing
        sb = i <= length(spec_b.slots) ? spec_b.slots[i] : nothing
        push!(
            slot_comparisons,
            SlotComparison(
                i,
                sa !== nothing ? sa.runtime_type : Any,
                sb !== nothing ? sb.runtime_type : Any,
                sa !== nothing ? sa.compiled_type : nothing,
                sb !== nothing ? sb.compiled_type : nothing,
                sa !== nothing ? sa.classification : SLOT_OK,
                sb !== nothing ? sb.classification : SLOT_OK,
            ),
        )
    end

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
        if sc.classification_a == SLOT_WIDENED_CALLABLE &&
            sc.classification_b == SLOT_OK
            push!(
                suggestions,
                "Slot $(sc.position): A uses $(sc.runtime_type_a) (widened), B uses $(sc.runtime_type_b) (concrete). Use B's pattern.",
            )
        end
    end

    return PerfComparison(
        spec_a.runtime_sig,
        spec_b.runtime_sig,
        slot_comparisons,
        inl_a.invoke_count_optimized,
        inl_b.invoke_count_optimized,
        allocs_a.metrics.heap_alloc_calls,
        allocs_b.metrics.heap_alloc_calls,
        allocs_a.metrics.measured_allocs,
        allocs_b.metrics.measured_allocs,
        verdict,
        suggestions,
    )
end

"""
    quick_diagnose(f, args...; io=stdout)

Quick diagnostic with immediate display.
"""
function quick_diagnose(f, args...; io=stdout)
    report = diagnose_perf(f, args...)
    show_report(report; io)
    return report
end

# --- Display Functions ---

function _short_type_name(@nospecialize(T::Type))
    s = string(T)
    length(s) <= 30 && return s
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
    return @printf(io, "  %-25s  %8d  %8d  %8s\n", name, a, b, ds)
end

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
    @printf(
        io,
        "  %-5s  %-30s  %-30s  %-20s  %s\n",
        "Slot",
        "Runtime Type",
        "Compiled Type",
        "Classification",
        "Note"
    )
    println(io, "  ", "-" ^ 100)
    for slot in report.slots
        rt = _short_type_name(slot.runtime_type)
        ct = _short_type_name(slot.compiled_type)
        cls = _classification_str(slot.classification)
        flag = slot.classification == SLOT_WIDENED_CALLABLE ? " [!]" : ""
        @printf(
            io, "  %-5d  %-30s  %-30s  %-20s  %s\n",
            slot.position, rt, ct, cls * flag, slot.suggestion
        )
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
        println(
            io,
            "  Inline-sensitive:         YES — allocations disappear when inlined [!]",
        )
    else
        println(io, "  Inline-sensitive:         no")
    end
    if !isempty(report.matching_lines)
        println(io, "\n  Matching LLVM lines:")
        for (i, line) in enumerate(report.matching_lines)
            i > 5 &&
                (println(io, "  ... ($(length(report.matching_lines) - 5) more)"); break)
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
                stmts = c.statement_count !== nothing ? "$(c.statement_count) stmts" :
                        "? stmts"
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
    return println(io)
end

function show_comparison(comp::PerfComparison; io=stdout)
    println(io, "=" ^ 70)
    println(io, "Performance Comparison")
    println(io, "  A: $(comp.sig_a)")
    println(io, "  B: $(comp.sig_b)")
    println(io, "=" ^ 70)

    println(io, "\n--- Per-Slot Comparison ---")
    @printf(
        io,
        "  %-5s  %-25s  %-25s  %-15s  %-15s  %s\n",
        "Slot",
        "A Runtime",
        "B Runtime",
        "A Class",
        "B Class",
        "Delta"
    )
    println(io, "  ", "-" ^ 95)
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
        @printf(
            io, "  %-5d  %-25s  %-25s  %-15s  %-15s  %s\n",
            sc.position, rta, rtb, ca, cb, delta
        )
    end

    println(io, "\n--- Metrics ---")
    @printf(io, "  %-25s  %8s  %8s  %8s\n", "Metric", "A", "B", "Delta")
    println(io, "  ", "-" ^ 55)
    _metric_row(io, "Invoke count", comp.invoke_count_a, comp.invoke_count_b)
    _metric_row(io, "Heap allocs (LLVM)", comp.heap_alloc_calls_a, comp.heap_alloc_calls_b)
    _metric_row(io, "Measured allocs", comp.measured_allocs_a, comp.measured_allocs_b)

    println(io, "\n  Verdict: $(comp.verdict)")
    for s in comp.suggestions
        println(io, "  Suggestion: ", s)
    end
    return println(io)
end
