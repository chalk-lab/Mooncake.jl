#=
IR Inspection Tools for Mooncake.jl
====================================
Tools for viewing and debugging IR transformations at each stage of Mooncake's AD pipeline.
## Quick Start
    using Revise
    includet(".claude/skills/ir-inspect/scripts/ir_inspect.jl")
    # Inspect reverse-mode AD for a function
    ins = inspect_ir(sin, 1.0)
    # View all stages
    show_ir(ins)
    # View a specific stage
    show_stage(ins, :raw)        # Original IR from Julia
    show_stage(ins, :normalized) # After Mooncake normalization
    show_stage(ins, :bbcode)     # BBCode representation (stable IDs)
    show_stage(ins, :fwd_ir)     # Forward pass IR (reverse-mode)
    show_stage(ins, :rvs_ir)     # Pullback IR (reverse-mode)
    show_stage(ins, :optimized_fwd)  # Optimized forward pass
    show_stage(ins, :optimized_rvs)  # Optimized pullback
    # View diffs between stages
    show_diff(ins; from=:raw, to=:normalized)
    show_all_diffs(ins)
    # World age debugging
    show_world_info(ins)
    # Get CFG as DOT/Graphviz string
    println(cfg_dot(ins, :bbcode))
    # Write all outputs to files
    write_ir(ins, "/tmp/ir_output")
## Forward Mode
    ins = inspect_ir(sin, 1.0; mode=:forward)
    # Stages: :raw -> :normalized -> :bbcode -> :dual_ir -> :optimized
## Options
    inspect_ir(f, args...;
        mode = :reverse,       # :forward or :reverse
        optimize = true,       # run optimization passes
        do_inline = true,      # inline during optimization
        debug_mode = false,    # enable Mooncake debug mode
    )
=#
using Mooncake
using Mooncake:
# Interpreter internals
    MooncakeInterpreter,
    get_interpreter,
    ForwardMode,
    ReverseMode,
    lookup_ir,
    normalise!,
    is_vararg_and_sparam_names,
# BBCode
    BBCode,
    BBlock,
    ID,
    collect_stmts,
    remove_unreachable_blocks!,
    seed_id!,
# Forward mode
    generate_dual_ir,
# Reverse mode
    generate_ir,
# Optimization
    optimise_ir!
# BBCode utilities (accessed via full path since not exported)
using Core.Compiler: IRCode, CFG
const CC = Core.Compiler
# =============================================================================
# Data Structures
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
# =============================================================================
# Stage Graphs
# =============================================================================
forward_stage_order() = [:raw, :normalized, :bbcode, :dual_ir, :optimized]
function forward_stage_graph()
    [
        :raw => :normalized,
        :normalized => :bbcode,
        :bbcode => :dual_ir,
        :dual_ir => :optimized,
    ]
end
function reverse_stage_order()
    [:raw, :normalized, :bbcode, :fwd_ir, :rvs_ir, :optimized_fwd, :optimized_rvs]
end
function reverse_stage_graph()
    [
        :raw => :normalized,
        :normalized => :bbcode,
        :bbcode => :fwd_ir,
        :bbcode => :rvs_ir,
        :fwd_ir => :optimized_fwd,
        :rvs_ir => :optimized_rvs,
    ]
end
# =============================================================================
# IR Rendering (stable text for diffs)
# =============================================================================
function render_ir(ir::IRCode)::String
    io = IOBuffer()
show(io, ir)
return String(take!(io))
end
function render_ir(bb::BBCode)::String
    io = IOBuffer()
# Render BBCode blocks
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
# =============================================================================
# Metadata Extraction
# =============================================================================
function extract_meta(ir::IRCode)::StageMeta
    cfg = ir.cfg
    block_count = length(cfg.blocks)
    inst_count = length(ir.stmts)
    edge_count = sum(length(b.succs) for b in cfg.blocks)
# Try to get valid_worlds (Julia 1.12+)
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
# Count edges from successors
    succs = Mooncake.BasicBlockCode.compute_all_successors(bb)
    edge_count = sum(length(v) for v in values(succs))
return StageMeta(;
        block_count=block_count,
        inst_count=inst_count,
        edge_count=edge_count,
        has_ssa=false,
        uses_bbcode_ids=true,
    )
end
function extract_meta(x)::StageMeta
return StageMeta()
end
# =============================================================================
# CFG to DOT
# =============================================================================
function cfg_to_dot(ir::IRCode; name="cfg")::String
    cfg = ir.cfg
    io = IOBuffer()
println(io, "digraph $name {")
println(io, "  rankdir=TB;")
println(io, "  node [shape=box];")
for (i, block) in enumerate(cfg.blocks)
# Node with line range
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
# Map block IDs to indices
    id_to_idx = Dict(b.id => i for (i, b) in enumerate(bb.blocks))
    succs = Mooncake.BasicBlockCode.compute_all_successors(bb)
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
# =============================================================================
# Text Diff (simple line-based)
# =============================================================================
function simple_diff(text1::String, text2::String; context=3)::String
    lines1 = split(text1, '\n')
    lines2 = split(text2, '\n')
    io = IOBuffer()
println(io, "--- stage1")
println(io, "+++ stage2")
# Simple diff: show changed lines
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
# =============================================================================
# Main Inspection Functions
# =============================================================================
"""
    inspect_ir(f, args...; kwargs...) -> IRInspection
Inspect IR transformations for a function call. Returns an IRInspection struct
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
# Build signature
    sig = Tuple{typeof(f),map(typeof, args)...}
# Get interpreter
    interp_mode = mode == :forward ? ForwardMode : ReverseMode
    interp = get_interpreter(interp_mode)
# Initialize containers
    stages = Dict{Symbol,IRStage}()
    notes = String[]
# Reset IDs for reproducibility
seed_id!()
try
# ===================
# Stage 1: Raw IR
# ===================
        raw_ir, _ = lookup_ir(interp, sig)
        stages[:raw] = IRStage(:raw, raw_ir, render_ir(raw_ir), extract_meta(raw_ir))
# ===================
# Stage 2: Normalized IR
# ===================
        isva, spnames = is_vararg_and_sparam_names(sig)
        normalized_ir = CC.copy(raw_ir)
normalise!(normalized_ir, spnames)
        stages[:normalized] = IRStage(
            :normalized,
            normalized_ir,
render_ir(normalized_ir),
extract_meta(normalized_ir),
        )
# ===================
# Stage 3: BBCode
# ===================
        bbcode = BBCode(normalized_ir)
        bbcode = remove_unreachable_blocks!(bbcode)
        stages[:bbcode] = IRStage(:bbcode, bbcode, render_ir(bbcode), extract_meta(bbcode))
# ===================
# Mode-specific stages
# ===================
if mode == :forward
# Forward mode: dual_ir
            dual_ir, captures, info = generate_dual_ir(
                interp, sig; debug_mode, do_inline=false
            )
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
# Reverse mode: fwd_ir and rvs_ir
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
@error "IR inspection failed" exception=(e, catch_backtrace())
end
# Choose stage order and graph based on mode
    stage_order = mode == :forward ? forward_stage_order() : reverse_stage_order()
    stage_graph = mode == :forward ? forward_stage_graph() : reverse_stage_graph()
# Filter to only include stages we actually have
    stage_order = filter(s -> haskey(stages, s), stage_order)
    stage_graph = filter(
        p -> haskey(stages, p.first) && haskey(stages, p.second), stage_graph
    )
# Compute diffs
    diffs = Dict{Pair{Symbol,Symbol},String}()
if compute_diffs
for (from, to) in stage_graph
if haskey(stages, from) && haskey(stages, to)
                diffs[from => to] = simple_diff(stages[from].text, stages[to].text)
end
end
end
# Compute CFGs
    cfgs = Dict{Symbol,String}()
if compute_cfgs
for (name, stage) in stages
try
                cfgs[name] = cfg_to_dot(stage.ir; name=string(name))
catch
# Some stages may not support CFG
end
end
end
return IRInspection(
        mode,
        sig,
        world,
        stages,
        stage_order,
        stage_graph,
        diffs,
        cfgs,
        RuleInspection[],
        notes,
    )
end
# =============================================================================
# Display Functions
# =============================================================================
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
println(io, "="^60)
println(io, "IR Inspection: $(ins.sig)")
println(io, "Mode: $(ins.mode), World: $(ins.world)")
println(io, "="^60)
for name in stage_list
if haskey(ins.stages, name)
            stage = ins.stages[name]
println(io, "\n", "-"^40)
println(io, "Stage: $name")
println(
                io,
"Blocks: $(stage.meta.block_count), Insts: $(stage.meta.inst_count), Edges: $(stage.meta.edge_count)",
            )
if stage.meta.valid_worlds !== nothing
println(io, "Valid worlds: $(stage.meta.valid_worlds)")
end
println(io, "-"^40)
println(io, stage.text)
end
end
end
"""
    show_stage(ins::IRInspection, stage::Symbol; io=stdout)
Display a single stage.
"""
function show_stage(ins::IRInspection, stage::Symbol; io=stdout)
show_ir(ins; stages=[stage], io)
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
# Compute on demand
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
println(io, "="^60)
println(io, "Diff: $from → $to")
println(io, "="^60)
println(io, diff_ir(ins; from, to))
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
function cfg_dot(ins::IRInspection, stage::Symbol)
return get(ins.cfgs, stage, "")
end
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
# Check for mismatches
    mismatches = String[]
for (name, w) in stage_worlds
if w !== nothing && w > ins.world
push!(mismatches, "Stage $name has world $w > inspection world $(ins.world)")
end
end
return WorldAgeReport(
        ins.world,
nothing,  # TODO: extract from method
        stage_worlds,
        Union{UInt,Nothing}[],
        mismatches,
    )
end
"""
    show_world_info(ins::IRInspection; io=stdout)
Display world age information.
"""
function show_world_info(ins::IRInspection; io=stdout)
    report = world_age_info(ins)
println(io, "="^60)
println(io, "World Age Report")
println(io, "="^60)
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
# =============================================================================
# Convenience Functions
# =============================================================================
"""
    inspect_fwd(f, args...; kwargs...)
Shorthand for inspect_ir with forward mode.
"""
inspect_fwd(f, args...; kwargs...) = inspect_ir(f, args...; mode=:forward, kwargs...)
"""
    inspect_rvs(f, args...; kwargs...)
Shorthand for inspect_ir with reverse mode.
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
# =============================================================================
# File Output
# =============================================================================
"""
    write_ir(ins::IRInspection, outdir::String)
Write all stages, diffs, and CFGs to files.
"""
function write_ir(ins::IRInspection, outdir::String)
mkpath(outdir)
# Write stages
for (name, stage) in ins.stages
open(joinpath(outdir, "$(name).txt"), "w") do f
println(f, stage.text)
end
end
# Write diffs
for ((from, to), diff) in ins.diffs
open(joinpath(outdir, "diff_$(from)_$(to).txt"), "w") do f
println(f, diff)
end
end
# Write CFGs
for (name, dot) in ins.cfgs
open(joinpath(outdir, "cfg_$(name).dot"), "w") do f
println(f, dot)
end
end
println(
"Wrote $(length(ins.stages)) stages, $(length(ins.diffs)) diffs, $(length(ins.cfgs)) CFGs to $outdir",
    )
end
# =============================================================================
# Example usage (uncomment to test)
# =============================================================================
# Simple test function
test_fn(x) = sin(x) * cos(x)
println("IR Inspect module loaded. Try:")
println("  ins = inspect_ir(sin, 1.0)")
println("  show_ir(ins)")
println("  show_diff(ins; from=:raw, to=:normalized)")
println("  show_world_info(ins)")
