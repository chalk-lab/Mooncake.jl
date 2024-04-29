# See the docstring for `BBCode` for some context on this file.

_id_count::Int32 = 0

"""
    ID()

An `ID` (read: unique name) is just a wrapper around an `Int32`. Uniqueness is ensured via a
global counter, which is incremented each time that an `ID` is created.

This counter can be reset using `seed_id!` if you need to ensure deterministic `ID`s are
produced, in the same way that seed for random number generators can be set.
"""
struct ID
    id::Int32
    function ID()
        global _id_count += 1 
        return new(_id_count)
    end
end

Base.copy(id::ID) = id

"""
    seed_id!()

Set the global counter used to ensure ID uniqueness to 0. This is useful when you want to
ensure determinism between two runs of the same function which makes use of `ID`s.

This is akin to setting the random seed associated to a random number generator globally.
"""
function seed_id!()
    global _id_count = 0
end

"""
    IDPhiNode(edges::Vector{ID}, values::Vector{Any})

Like a `PhiNode`, but `edges` are `ID`s rather than `Int32`s.
"""
struct IDPhiNode
    edges::Vector{ID}
    values::Vector{Any}
end

Base.copy(node::IDPhiNode) = IDPhiNode(copy(node.edges), copy(node.values))

"""
    IDGotoNode(label::ID)

Like a `GotoNode`, but `label` is an `ID` rather than an `Int64`.
"""
struct IDGotoNode
    label::ID
end

Base.copy(node::IDGotoNode) = IDGotoNode(copy(node.label))

"""
    IDGotoIfNot(cond::Any, dest::ID)

Like a `GotoIfNot`, but `dest` is an `ID` rather than an `Int64`.
"""
struct IDGotoIfNot
    cond::Any
    dest::ID
end

Base.copy(node::IDGotoIfNot) = IDGotoIfNot(copy(node.cond), copy(node.dest))

"""
    Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)

A switch-statement node. These can be inserted in the `BBCode` representation of Julia IR.
`Switch` has the following semantics:
```julia
goto dests[1] if not conds[1]
goto dests[2] if not conds[2]
...
goto dests[N] if not conds[N]
goto fallthrough_dest
```
where the value associated to each element of `conds` is a `Bool`, and `dests` indicate
which block to jump to. If none of the conditions are met, then we go to whichever block is
specified by `fallthrough_dest`.

`Switch` statements are lowered into the above sequence of `GotoIfNot`s and `GotoNode`s
when converting `BBCode` back into `IRCode`, because `Switch` statements are not valid
nodes in regular Julia IR.
"""
struct Switch
    conds::Vector{Any}
    dests::Vector{ID}
    fallthrough_dest::ID
    function Switch(conds::Vector{Any}, dests::Vector{ID}, fallthrough_dest::ID)
        @assert length(conds) == length(dests)
        return new(conds, dests, fallthrough_dest)
    end
end

"""
    Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, ReturnNode}

A Union of the possible types of a terminator node.
"""
const Terminator = Union{Switch, IDGotoIfNot, IDGotoNode, ReturnNode}

"""
    const InstVector = Vector{NewInstruction}

Note: the `CC.NewInstruction` type is used to represent instructions because it has the
correct fields. While it is only used to represent new instrucdtions in `Core.Compiler`, it
is used to represent all instructions in `BBCode`.
"""
const InstVector = Vector{NewInstruction}

"""
    BBlock(id::ID, stmt_ids::Vector{ID}, stmts::InstVector)

A basic block data structure (not called `BasicBlock` to avoid accidental confusion with
`CC.BasicBlock`). Forms a single basic block.

Each `BBlock` has an `ID` (a unique name). This makes it possible to refer to blocks in a
way that does not change when additional `BBlocks` are inserted into a `BBCode`.
This differs from the positional block numbering found in `IRCode`, in which the number
associated to a basic block changes when new blocks are inserted.

The `n`th line of code in a `BBlock` is associated to `ID` `stmt_ids[n]`, and the `n`th
instruction from `stmts`.

Note that `PhiNode`s, `GotoIfNot`s, and `GotoNode`s should not appear in a `BBlock` --
instead an `IDPhiNode`, `IDGotoIfNot`, or `IDGotoNode` should be used.
"""
mutable struct BBlock
    id::ID
    inst_ids::Vector{ID}
    insts::InstVector
    function BBlock(id::ID, inst_ids::Vector{ID}, insts::InstVector)
        @assert length(inst_ids) == length(insts)
        return new(id, inst_ids, insts)
    end
end

"""
    BBlock(id::ID, inst_pairs::Vector{Tuple{ID, NewInstruction}})

Convenience constructor -- splits `inst_pairs` into a `Vector{ID}` and `InstVector` in order
to build a `BBlock`.
"""
function BBlock(id::ID, inst_pairs::Vector{Tuple{ID, NewInstruction}})
    return BBlock(id, first.(inst_pairs), last.(inst_pairs))
end

Base.length(bb::BBlock) = length(bb.inst_ids)

Base.copy(bb::BBlock) = BBlock(bb.id, copy(bb.inst_ids), copy(bb.insts))

"""
    phi_nodes(bb::BBlock)::Tuple{Vector{ID}, Vector{IDPhiNode}}

Returns all of the `IDPhiNode`s at the start of `bb`, along with their `ID`s. If there are
no `IDPhiNode`s at the start of `bb`, then both vectors will be empty.
"""
function phi_nodes(bb::BBlock)
    n_phi_nodes = findlast(x -> x.stmt isa IDPhiNode, bb.insts)
    if n_phi_nodes === nothing
        n_phi_nodes = 0
    end
    return bb.inst_ids[1:n_phi_nodes], bb.insts[1:n_phi_nodes]
end

"""
    Base.insert!(bb::BBlock, n::Int, id::ID, stmt::CC.NewInstruction)::Nothing

Inserts `stmt` and `id` into `bb` immediately before the `n`th instruction.
"""
function Base.insert!(bb::BBlock, n::Int, id::ID, inst::NewInstruction)::Nothing
    insert!(bb.inst_ids, n, id)
    insert!(bb.insts, n, inst)
    return nothing
end

"""
    terminator(bb::BBlock)

Returns the terminator associated to `bb`. If the last instruction in `bb` isa
`Terminator` then that is returned, otherwise `nothing` is returned.
"""
terminator(bb::BBlock) = isa(bb.insts[end].stmt, Terminator) ? bb.insts[end].stmt : nothing

"""
    collect_stmts(bb::BBlock)::Vector{Tuple{ID, NewInstruction}}

Returns a `Vector` containing the `ID`s and instructions associated to each line in `bb`.
These should be assumed to be ordered.
"""
function collect_stmts(bb::BBlock)::Vector{Tuple{ID, NewInstruction}}
    return collect(zip(bb.inst_ids, bb.insts))
end

"""
    BBCode(
        blocks::Vector{BBlock}
        argtypes::Vector{Any}
        sptypes::Vector{CC.VarState}
        linetable::Vector{Core.LineInfoNode}
        meta::Vector{Expr}
    )

A `BBCode` is a data structure which is similar to `IRCode`, but adds additional structure.

In particular, a `BBCode` comprises a sequence of basic blocks (`BBlock`s), each of which
comprise a sequence of statements. Moreover, each `BBlock` has its own unique `ID`, as does
each statment.

The consequence of this is that new basic blocks can be inserted into a `BBCode`. This is
distinct from `IRCode`, in which to create a new basic block, one must insert additional
statments which you know will create a new basic block -- this is generally quite an
unreliable process, while inserting a new `BBlock` into `BBCode` is entirely predictable.
Furthermore, inserting a new `BBlock` does not change the `ID` associated to the other
blocks, meaning that you can safely assume that references from existing basic block
terminators / phi nodes to other blocks will not be modified by inserting a new basic block.

Additionally, since each statment in each basic block has its own unique `ID`, new
statments can be inserted without changing references between other blocks. `IRCode` also
has some support for this via its `new_nodes` field, but eventually all statements will be
renamed upon `compact!`ing the `IRCode`, meaning that the name of any given statement will
eventually change.

Finally, note that the basic blocks in a `BBCode` support the custom `Switch` statement.
This statement is not valid in `IRCode`, and is therefore lowered into a collection of
`GotoIfNot`s and `GotoNode`s when a `BBCode` is converted back into an `IRCode`.
"""
struct BBCode
    blocks::Vector{BBlock}
    argtypes::Vector{Any}
    sptypes::Vector{CC.VarState}
    linetable::Vector{Core.LineInfoNode}
    meta::Vector{Expr}
end

"""
    BBCode(ir::Union{IRCode, BBCode}, new_blocks::Vector{Block})

Make a new `BBCode` whose `blocks` is given by `new_blocks`, and fresh copies are made of
all other fields from `ir`.
"""
function BBCode(ir::Union{IRCode, BBCode}, new_blocks::Vector{BBlock})
    return BBCode(
        new_blocks,
        CC.copy(ir.argtypes),
        CC.copy(ir.sptypes),
        CC.copy(ir.linetable),
        CC.copy(ir.meta),
    )
end

# Makes use of the above outer constructor for `BBCode`.
Base.copy(ir::BBCode) = BBCode(ir, copy(ir.blocks))

"""
    compute_all_successors(ir::BBCode)::Dict{ID, Vector{ID}}

Compute a map from the `ID of each `BBlock` in `ir` to its possible successors.
"""
function compute_all_successors(ir::BBCode)::Dict{ID, Vector{ID}}
    return _compute_all_successors(ir.blocks)
end

# Internal method. Just requires that a Vector of BBlocks are passed. This method is easier
# to construct test cases for because there is no need to construct all the other stuff that
# goes into a `BBCode`.
function _compute_all_successors(blks::Vector{BBlock})::Dict{ID, Vector{ID}}
    succs = map(enumerate(blks)) do (n, blk)
        return successors(terminator(blk), n, blks, n == length(blks))
    end
    return Dict{ID, Vector{ID}}(zip(map(b -> b.id, blks), succs))
end

function successors(::Nothing, n::Int, blks::Vector{BBlock}, is_final_block::Bool)
    return is_final_block ? ID[] : ID[blks[n+1].id]
end
successors(t::IDGotoNode, ::Int, ::Vector{BBlock}, ::Bool) = [t.label]
function successors(t::IDGotoIfNot, n::Int, blks::Vector{BBlock}, is_final_block::Bool)
    return is_final_block ? ID[t.dest] : ID[t.dest, blks[n + 1].id]
end
successors(::ReturnNode, ::Int, ::Vector{BBlock}, ::Bool) = ID[]
successors(t::Switch, ::Int, ::Vector{BBlock}, ::Bool) = vcat(t.dests, t.fallthrough_dest)

"""
    compute_all_predecessors(ir::BBCode)::Dict{ID, Vector{ID}}

Compute a map from the `ID of each `BBlock` in `ir` to its possible predecessors.
"""
function compute_all_predecessors(ir::BBCode)::Dict{ID, Vector{ID}}
    return _compute_all_predecessors(ir.blocks)
end

function _compute_all_predecessors(blks::Vector{BBlock})::Dict{ID, Vector{ID}}

    successor_map = _compute_all_successors(blks)

    # Initialise predecessor map to be empty.
    ks = collect(keys(successor_map))
    predecessor_map = Dict{ID, Vector{ID}}(zip(ks, map(_ -> ID[], ks)))

    # Find all predecessors by iterating through the successor map.
    for (k, succs) in successor_map
        for succ in succs
            push!(predecessor_map[succ], k)
        end
    end

    return predecessor_map
end

"""
    collect_stmts(ir::BBCode)::Vector{Tuple{ID, CC.NewInstruction}}

Produce a `Vector` containing all of the statements in `ir`. These are returned in
order, so it is safe to assume that element `n` refers to the `nth` element of the `IRCode`
associated to `ir`. 
"""
function collect_stmts(ir::BBCode)::Vector{Tuple{ID, NewInstruction}}
    return reduce(vcat, map(collect_stmts, ir.blocks))
end

"""
    id_to_line_map(ir::BBCode)

Produces a `Dict` mapping from each `ID` associated with a line in `ir` to its line number.
This is isomorphic to mapping to its `SSAValue` in `IRCode`. Terminators do not have `ID`s
associated to them, so not every line in the original `IRCode` is mapped to.
"""
function id_to_line_map(ir::BBCode)
    lines = collect_stmts(ir)
    lines_and_line_numbers = collect(zip(lines, eachindex(lines)))
    ids_and_line_numbers = map(x -> (x[1][1], x[2]), lines_and_line_numbers)
    return Dict(ids_and_line_numbers)
end

concatenate_ids(bb_code::BBCode) = reduce(vcat, map(b -> b.inst_ids, bb_code.blocks))
concatenate_stmts(bb_code::BBCode) = reduce(vcat, map(b -> b.insts, bb_code.blocks))

#
# Converting from IRCode to BBCode
#

"""
    BBCode(ir::IRCode)

Convert an `ir` into a `BBCode`. Creates a completely independent data structure, so
mutating the `BBCode` returned will not mutate `ir`.

All `PhiNode`s, `GotoIfNot`s, and `GotoNode`s will be replaced with the `IDPhiNode`s,
`IDGotoIfNot`s, and `IDGotoNode`s respectively.

See `IRCode` for conversion back to `IRCode`.

Note that `IRCode(BBCode(ir))` should be equal to the identity function.
"""
function BBCode(ir::IRCode)

    # Produce a new set of statements with `IDs` rather than `SSAValues` and block numbers.
    insts = new_inst_vec(ir.stmts)
    ssa_ids, stmts = _ssas_to_ids(insts)
    block_ids, stmts = _block_nums_to_ids(stmts, ir.cfg)

    # Chop up the new statements into `BBlocks`, according to the `CFG` in `ir`.
    blocks = map(zip(ir.cfg.blocks, block_ids)) do (bb, id)
        return BBlock(id, ssa_ids[bb.stmts], stmts[bb.stmts])
    end
    return BBCode(ir, blocks)
end

# Convert an InstructionStream into a list of `NewInstruction`s.
function new_inst_vec(x::CC.InstructionStream)
    return map((v..., ) -> NewInstruction(v...), x.inst, x.type, x.info, x.line, x.flag)
end

# Maps from positional names (SSAValues for nodes, Integers for basic blocks) to IDs.
const SSAToIdDict = Dict{SSAValue, ID}
const BlockNumToIdDict = Dict{Integer, ID}

# Assigns an ID to each line in `stmts`, and replaces each instance of an `SSAValue` in each
# line with the corresponding `ID`. For example, a call statement of the form
# `Expr(:call, :f, %4)` is be replaced with `Expr(:call, :f, id_assigned_to_%4)`.
function _ssas_to_ids(insts::InstVector)::Tuple{Vector{ID}, InstVector}
    ids = map(_ -> ID(), insts)
    val_id_map = SSAToIdDict(zip(SSAValue.(eachindex(insts)), ids))
    return ids, map(Base.Fix1(_ssa_to_ids, val_id_map), insts)
end

# Produce a new instance of `x` in which all instances of `SSAValue`s are replaced with
# the `ID`s prescribed by `d`, all basic block numbers are replaced with the `ID`s
# prescribed by `d`, and `GotoIfNot`, `GotoNode`, and `PhiNode` instances are replaced with
# the corresponding `ID` versions.
function _ssa_to_ids(d::SSAToIdDict, inst::NewInstruction)
    return NewInstruction(inst; stmt=_ssa_to_ids(d, inst.stmt))
end
function _ssa_to_ids(d::SSAToIdDict, x::ReturnNode)
    return isdefined(x, :val) ? ReturnNode(get(d, x.val, x.val)) : x
end
_ssa_to_ids(d::SSAToIdDict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_ssa_to_ids(d::SSAToIdDict, x::PiNode) = PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_ssa_to_ids(d::SSAToIdDict, x::QuoteNode) = x
_ssa_to_ids(d::SSAToIdDict, x) = x
function _ssa_to_ids(d::SSAToIdDict, x::PhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = get(d, x.values[n], x.values[n])
        end
    end
    return PhiNode(x.edges, new_values)
end
_ssa_to_ids(d::SSAToIdDict, x::GotoNode) = x
_ssa_to_ids(d::SSAToIdDict, x::GotoIfNot) = GotoIfNot(get(d, x.cond, x.cond), x.dest)

# Replace all integers corresponding to references to blocks with IDs.
function _block_nums_to_ids(insts::InstVector, cfg::CC.CFG)::Tuple{Vector{ID}, InstVector}
    ids = map(_ -> ID(), cfg.blocks)
    block_num_id_map = BlockNumToIdDict(zip(eachindex(cfg.blocks), ids))
    return ids, map(Base.Fix1(_block_num_to_ids, block_num_id_map), insts)
end

function _block_num_to_ids(d::BlockNumToIdDict, x::NewInstruction)
    return NewInstruction(x; stmt=_block_num_to_ids(d, x.stmt))
end
function _block_num_to_ids(d::BlockNumToIdDict, x::PhiNode)
    return IDPhiNode(ID[d[e] for e in x.edges], x.values)
end
_block_num_to_ids(d::BlockNumToIdDict, x::GotoNode) = IDGotoNode(d[x.label])
_block_num_to_ids(d::BlockNumToIdDict, x::GotoIfNot) = IDGotoIfNot(x.cond, d[x.dest])
_block_num_to_ids(d::BlockNumToIdDict, x) = x

#
# Converting from BBCode to IRCode
#

"""
    IRCode(bb_code::BBCode)

Produce an `IRCode` instance which is equivalent to `bb_code`. The resulting `IRCode`
shares no memory with `bb_code`, so can be safely mutated without modifying `bb_code`.

All `IDPhiNode`s, `IDGotoIfNot`s, and `IDGotoNode`s are converted into `PhiNode`s,
`GotoIfNot`s, and `GotoNode`s respectively.

In the resulting `bb_code`, any `Switch` nodes are lowered into a semantically-equivalent
collection of `GotoIfNot` nodes.
"""
function CC.IRCode(bb_code::BBCode)
    bb_code = _lower_switch_statements(bb_code)
    bb_code = _remove_double_edges(bb_code)
    insts = _ids_to_line_positions(bb_code)
    cfg = _compute_basic_blocks(insts)
    insts = _lines_to_blocks(insts, cfg)
    return IRCode(
        CC.InstructionStream(
            map(x -> x.stmt, insts),
            map(x -> x.type, insts),
            map(x -> x.info, insts),
            map(x -> x.line, insts),
            map(x -> x.flag, insts),
        ),
        cfg,
        CC.copy(bb_code.linetable),
        CC.copy(bb_code.argtypes),
        CC.copy(bb_code.meta),
        CC.copy(bb_code.sptypes),
    )
end

# Converts all `Switch`s into a semantically-equivalent collection of `GotoIfNot`s. See the
# `Switch` docstring for an explanation of what is going on here.
function _lower_switch_statements(bb_code::BBCode)
    new_blocks = Vector{BBlock}(undef, 0)
    for block in bb_code.blocks
        t = terminator(block)
        if t isa Switch

            # Create new block without the `Switch`.
            bb = BBlock(block.id, block.inst_ids[1:end-1], block.insts[1:end-1])
            push!(new_blocks, bb)

            # Create new blocks for each `GotoIfNot` from the `Switch`.
            foreach(t.conds, t.dests) do cond, dest
                blk = BBlock(ID(), [ID()], [new_inst(IDGotoIfNot(cond, dest), Any)])
                push!(new_blocks, blk)
            end

            # Create a new block for the fallthrough dest.
            fallthrough_inst = new_inst(IDGotoNode(t.fallthrough_dest), Any)
            push!(new_blocks, BBlock(ID(), [ID()], [fallthrough_inst]))
        else
            push!(new_blocks, block)
        end
    end
    return BBCode(bb_code, new_blocks)
end

# Returns a `Vector{Any}` of statements in which each `ID` has been replaced by either an
# `SSAValue`, or an `Int64` / `Int32` which refers to an `SSAValue`.
function _ids_to_line_positions(bb_code::BBCode)::InstVector

    # Construct map from `ID`s to `SSAValue`s.
    block_ids = [b.id for b in bb_code.blocks]
    block_lengths = map(length, bb_code.blocks)
    block_start_ssas = SSAValue.(vcat(1, cumsum(block_lengths)[1:end-1] .+ 1))
    line_ids = concatenate_ids(bb_code)
    line_ssas = SSAValue.(eachindex(line_ids))
    id_to_ssa_map = Dict(zip(vcat(block_ids, line_ids), vcat(block_start_ssas, line_ssas)))

    # Apply map.
    return [_to_ssas(id_to_ssa_map, stmt) for stmt in concatenate_stmts(bb_code)]
end

# Like `_to_ids`, but converts IDs to SSAValues / (integers corresponding to ssas).
_to_ssas(d::Dict, inst::NewInstruction) = NewInstruction(inst; stmt=_to_ssas(d, inst.stmt))
_to_ssas(d::Dict, x::ReturnNode) = isdefined(x, :val) ? ReturnNode(get(d, x.val, x.val)) : x
_to_ssas(d::Dict, x::Expr) = Expr(x.head, map(a -> get(d, a, a), x.args)...)
_to_ssas(d::Dict, x::PiNode) = PiNode(get(d, x.val, x.val), get(d, x.typ, x.typ))
_to_ssas(d::Dict, x::QuoteNode) = x
_to_ssas(d::Dict, x) = x
function _to_ssas(d::Dict, x::IDPhiNode)
    new_values = Vector{Any}(undef, length(x.values))
    for n in eachindex(x.values)
        if isassigned(x.values, n)
            new_values[n] = get(d, x.values[n], x.values[n])
        end
    end
    return PhiNode(map(e -> Int32(getindex(d, e).id), x.edges), new_values)
end
_to_ssas(d::Dict, x::IDGotoNode) = GotoNode(d[x.label].id)
_to_ssas(d::Dict, x::IDGotoIfNot) = GotoIfNot(get(d, x.cond, x.cond), d[x.dest].id)

# Compute the CFG associated to `insts`. All references to blocks must be references to
# the SSAValue associated to the first statement in the block.
function _compute_basic_blocks(insts::InstVector)
    return CC.compute_basic_blocks(Any[inst.stmt for inst in insts])
end

# Replaces references to blocks by line-number with references to block numbers.
function _lines_to_blocks(insts::InstVector, cfg::CC.CFG)
    return map(inst -> __lines_to_blocks(cfg, inst), insts)
end

function __lines_to_blocks(cfg::CC.CFG, inst::NewInstruction)
    return NewInstruction(inst; stmt=__lines_to_blocks(cfg, inst.stmt))
end
function __lines_to_blocks(cfg::CC.CFG, stmt::GotoNode)
    return GotoNode(CC.block_for_inst(cfg, stmt.label))
end
function __lines_to_blocks(cfg::CC.CFG, stmt::GotoIfNot)
    return GotoIfNot(stmt.cond, CC.block_for_inst(cfg, stmt.dest))
end
function __lines_to_blocks(cfg::CC.CFG, stmt::PhiNode)
    return PhiNode(Int32[CC.block_for_inst(cfg, Int(e)) for e in stmt.edges], stmt.values)
end
function __lines_to_blocks(cfg::CC.CFG, stmt::Expr)
    Meta.isexpr(stmt, :enter) && throw(error("Cannot handle enter yet"))
    return stmt
end
__lines_to_blocks(::CC.CFG, stmt) = stmt

# If the `dest` field of a `GotoIfNot` node points towards the next block, replace it with
# a `GotoNode`.
function _remove_double_edges(ir::BBCode)
    new_blks = map(enumerate(ir.blocks)) do (n, blk)
        t = terminator(blk)
        if t isa IDGotoIfNot && t.dest == ir.blocks[n+1].id
            new_insts = vcat(blk.insts[1:end-1], NewInstruction(t; stmt=IDGotoNode(t.dest)))
            return BBlock(blk.id, blk.inst_ids, new_insts)
        else
            return blk
        end
    end
    return BBCode(ir, new_blks)
end

#=
    _sort_blocks!(ir::BBCode)

Ensure that blocks appear in order of distance-from-entry-point, where distance the
distance from block b to the entry point is defined to be the minimum number of basic
blocks that must be passed through in order to reach b.

For reasons unknown (to me, Will), the compiler / optimiser needs this for inference to
succeed. Since we do quite a lot of re-ordering on the reverse-pass of AD, this is a problem
there.

WARNING: use with care. Only use if you are confident that arbitrary re-ordering of basic
blocks in `ir` is valid. Notably, this does not hold if you have any `IDGotoIfNot` nodes in
`ir`.
=#
function _sort_blocks!(ir::BBCode)

    node_ints = collect(eachindex(ir.blocks))
    id_to_int = Dict(zip(map(blk -> blk.id, ir.blocks), node_ints))
    ps = compute_all_predecessors(ir)
    direct_predecessors = map(ir.blocks) do blk
        return map(b -> Edge(id_to_int[b], id_to_int[blk.id]), ps[blk.id])
    end
    g = SimpleDiGraph(reduce(vcat, direct_predecessors))

    d = dijkstra_shortest_paths(g, id_to_int[ir.blocks[1].id]).dists
    I = sortperm(d)
    ir.blocks .= ir.blocks[I]
    return ir
end

#=
    characterise_unique_predecessor_blocks(blks::Vector{BBlock}) ->
        Tuple{Dict{ID, Bool}, Dict{ID, Bool}}

We call a block `b` a _unique_ _predecessor_ in the control flow graph associated to `blks`
if it is the only predecessor to all of its successors. Put differently we call `b` a unique
predecessor if, whenever control flow arrives in any of the successors of `b`, we know for
certain that the previous block must have been `b`.

Returns two `Dict`s. A value in the first `Dict` is `true` if the block associated to its
key is a unique precessor, and is `false` if not. A value in the second `Dict` is `true` if 
it has a single predecessor, and that predecessor is a unique predecessor.

*Context*:

This information is important for optimising AD because knowing that `b` is a unique
predecessor means that
1. on the forwards-pass, there is no need to push the ID of `b` to the block stack when
    passing through it, and
2. on the reverse-pass, there is no need to pop the block stack when passing through one of
    the successors to `b`.

Utilising this reduces the overhead associated to doing AD. It is quite important when
working with cheap loops -- loops where the operations performed at each iteration
are inexpensive -- for which minimising memory pressure is critical to performance. It is
also important for single-block functions, because it can be used to entirely avoid using a
block stack at all.
=#
function characterise_unique_predecessor_blocks(
    blks::Vector{BBlock}
)::Tuple{Dict{ID, Bool}, Dict{ID, Bool}}

    # Obtain the block IDs in order -- this ensures that we get the entry block first.
    blk_ids = ID[b.id for b in blks]
    preds = _compute_all_predecessors(blks)
    succs = _compute_all_successors(blks)

    # The bulk of blocks can be hanled by this general loop.
    is_unique_pred = Dict{ID, Bool}()
    for id in blk_ids
        ss = succs[id]
        is_unique_pred[id] = !isempty(ss) && all(s -> length(preds[s]) == 1, ss)
    end

    # If there is a single reachable return node, then that block is treated as a unique
    # pred, since control can only pass "out" of the function via this block. Conversely,
    # if there are multiple reachable return nodes, then execution can return to the calling
    # function via any of them, so they are not unique predecessors.
    # Note that the previous block sets is_unique_pred[id] to false for all nodes which
    # end with a reachable return node, so the value only needs changing if there is a
    # unique reachable return node.
    reachable_return_blocks = filter(blks) do blk
        is_reachable_return_node(terminator(blk))
    end
    if length(reachable_return_blocks) == 1
        is_unique_pred[only(reachable_return_blocks).id] = true
    end

    # pred_is_unique_pred is true if the unique predecessor to a block is a unique pred.
    pred_is_unique_pred = Dict{ID, Bool}()
    for id in blk_ids
        pred_is_unique_pred[id] = length(preds[id]) == 1 && is_unique_pred[only(preds[id])]
    end

    # If the entry block has no predecessors, then it can only be entered once, when the
    # function is first entered. In this case, we treat it as having a unique predecessor.
    entry_id = blk_ids[1]
    pred_is_unique_pred[entry_id] = isempty(preds[entry_id])

    return is_unique_pred, pred_is_unique_pred
end

"""
    characterise_used_ids(blks::Vector{BBlock})::Dict{ID, Bool}

For each line in `blks`, determine whether it is referenced anywhere else in the code.
Returns a dictionary containing the results. An element is `false` if the corresponding
`ID` is unused, and `true` if is used.
"""
function characterise_used_ids(stmts::Vector{Tuple{ID, NewInstruction}})::Dict{ID, Bool}
    ids = first.(stmts)
    insts = last.(stmts)

    # Initialise to false.
    is_used = Dict{ID, Bool}(zip(ids, fill(false, length(ids))))

    # Hunt through the instructions, flipping a value in is_used to true whenever an ID
    # is encountered which corresponds to an SSA.
    for inst in insts
        _find_id_uses!(is_used, inst.stmt)
    end
    return is_used
end

# Helper function used in characterise_used_ids.
function _find_id_uses!(d::Dict{ID, Bool}, x::Expr)
    for arg in x.args
        in(arg, keys(d)) && setindex!(d, true, arg)
    end
end
function _find_id_uses!(d::Dict{ID, Bool}, x::IDGotoIfNot)
    return in(x.cond, keys(d)) && setindex!(d, true, x.cond)
end
_find_id_uses!(::Dict{ID, Bool}, ::IDGotoNode) = nothing
function _find_id_uses!(d::Dict{ID, Bool}, x::PiNode)
    return in(x.val, keys(d)) && setindex!(d, true, x.val)
end
function _find_id_uses!(d::Dict{ID, Bool}, x::IDPhiNode)
    v = x.values
    for n in eachindex(v)
        isassigned(v, n) && in(v[n], keys(d)) && setindex!(d, true, v[n])
    end
end
function _find_id_uses!(d::Dict{ID, Bool}, x::ReturnNode)
    return isdefined(x, :val) && in(x.val, keys(d)) && setindex!(d, true, x.val)
end
_find_id_uses!(d::Dict{ID, Bool}, x::QuoteNode) = nothing
_find_id_uses!(d::Dict{ID, Bool}, x) = nothing
