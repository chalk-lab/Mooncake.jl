"""
    module SkillUtils

Developer tools for viewing and diffing IR at each stage of the AD pipeline.
These are NOT part of the public API and may change in non-breaking releases.
"""
module SkillUtils

using ..Mooncake:
    CC,
    IRCode,
    BBCode,
    BasicBlockCode,
    ForwardMode,
    ReverseMode,
    get_interpreter,
    lookup_ir,
    is_vararg_and_sparam_names,
    normalise!,
    remove_unreachable_blocks!,
    generate_dual_ir,
    generate_ir,
    optimise_ir!,
    seed_id!

struct StageMeta
    block_count::Int
    inst_count::Int
    edge_count::Int
    valid_worlds::Union{UnitRange{UInt},Nothing}
end

function StageMeta(; block_count=0, inst_count=0, edge_count=0, valid_worlds=nothing)
    return StageMeta(block_count, inst_count, edge_count, valid_worlds)
end

struct IRStage
    name::Symbol
    ir::Any
    text::String
    meta::StageMeta
end

struct IRInspection
    mode::Symbol
    sig::Type
    world::UInt
    stages::Dict{Symbol,IRStage}
    stage_order::Vector{Symbol}
    stage_graph::Vector{Pair{Symbol,Symbol}}
    diffs::Dict{Pair{Symbol,Symbol},String}
    notes::Vector{String}
end

struct WorldAgeReport
    inspection_world::UInt
    stage_worlds::Dict{Symbol,Union{UInt,Nothing}}
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
    valid_worlds = nothing
    if hasproperty(ir, :valid_worlds)
        vw = ir.valid_worlds
        valid_worlds = UInt(CC.min_world(vw)):UInt(CC.max_world(vw))
    end
    return StageMeta(;
        block_count=length(cfg.blocks),
        inst_count=length(ir.stmts),
        edge_count=sum(length(b.succs) for b in cfg.blocks),
        valid_worlds=valid_worlds,
    )
end

function extract_meta(bb::BBCode)::StageMeta
    succs = BasicBlockCode.compute_all_successors(bb)
    return StageMeta(;
        block_count=length(bb.blocks),
        inst_count=sum(length(b.inst_ids) for b in bb.blocks),
        edge_count=sum(length(v) for v in values(succs)),
    )
end

extract_meta(x) = StageMeta()

# --- Text Diff ---

function simple_diff(text1::String, text2::String)::String
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
        _, spnames = is_vararg_and_sparam_names(sig)
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
        stages[:bbcode] = IRStage(:bbcode, bbcode, render_ir(bbcode), extract_meta(bbcode))

        # Mode-specific stages
        if mode == :forward
            dual_ir, _, _ = generate_dual_ir(interp, sig; debug_mode, do_inline=false)
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
    stage_graph = filter(
        p -> haskey(stages, p.first) && haskey(stages, p.second), stage_graph
    )

    diffs = Dict{Pair{Symbol,Symbol},String}()
    if compute_diffs
        for (from, to) in stage_graph
            diffs[from => to] = simple_diff(stages[from].text, stages[to].text)
        end
    end

    return IRInspection(mode, sig, world, stages, stage_order, stage_graph, diffs, notes)
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
    for note in ins.notes
        println(io, "\n⚠ ", note)
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
    world_age_info(ins::IRInspection) -> WorldAgeReport

Extract world age information from inspection.
"""
function world_age_info(ins::IRInspection)
    stage_worlds = Dict{Symbol,Union{UInt,Nothing}}()
    for (name, stage) in ins.stages
        vw = stage.meta.valid_worlds
        stage_worlds[name] = vw !== nothing ? first(vw) : nothing
    end

    mismatches = String[]
    for (name, stage) in ins.stages
        vw = stage.meta.valid_worlds
        vw === nothing && continue
        if ins.world < first(vw)
            push!(
                mismatches,
                "Stage $name: min_world $(first(vw)) > inspection world $(ins.world)",
            )
        elseif ins.world > last(vw)
            push!(
                mismatches,
                "Stage $name: max_world $(last(vw)) < inspection world $(ins.world) (stale)",
            )
        end
    end

    return WorldAgeReport(ins.world, stage_worlds, mismatches)
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
    return println(
        "Wrote $(length(ins.stages)) stages, $(length(ins.diffs)) diffs to $outdir"
    )
end

end # module SkillUtils
