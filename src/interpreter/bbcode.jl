"""
    module BasicBlockCode

See the docstring for the `BBCode` `struct` for info on this file.
"""
module BasicBlockCode

using Graphs

using Core.Compiler:
    ReturnNode,
    PhiNode,
    GotoIfNot,
    GotoNode,
    NewInstruction,
    IRCode,
    SSAValue,
    PiNode,
    Argument
const CC = Core.Compiler

export ID,
    seed_id!,
    IDPhiNode,
    IDGotoNode,
    IDGotoIfNot,
    Switch,
    BBlock,
    phi_nodes,
    terminator,
    insert_before_terminator!,
    collect_stmts,
    compute_all_predecessors,
    BBCode,
    remove_unreachable_blocks!,
    characterise_used_ids,
    characterise_unique_predecessor_blocks,
    sort_blocks!,
    InstVector,
    IDInstPair,
    __line_numbers_to_block_numbers!,
    is_reachable_return_node,
    new_inst

const _id_count::Dict{Int,Int32} = Dict{Int,Int32}()

"""
    new_inst(stmt, type=Any, flag=CC.IR_FLAG_REFINED)::NewInstruction

Create a `NewInstruction` with fields:
- `stmt` = `stmt`
- `type` = `type`
- `info` = `CC.NoCallInfo()`
- `line` = `Int32(1)`
- `flag` = `flag`
"""
function new_inst(@nospecialize(stmt), @nospecialize(type)=Any, flag=CC.IR_FLAG_REFINED)
    return NewInstruction(stmt, type, CC.NoCallInfo(), Int32(1), flag)
end

"""
    const InstVector = Vector{NewInstruction}

Note: the `CC.NewInstruction` type is used to represent instructions because it has the
correct fields. While it is only used to represent new instrucdtions in `Core.Compiler`, it
is used to represent all instructions in `BBCode`.
"""
const InstVector = Vector{NewInstruction}

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
        current_thread_id = Threads.threadid()
        id_count = get(_id_count, current_thread_id, Int32(0))
        _id_count[current_thread_id] = id_count + Int32(1)
        return new(id_count)
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
    return global _id_count[Threads.threadid()] = 0
end

"""
    IDPhiNode(edges::Vector{ID}, values::Vector{Any})

Like a `PhiNode`, but `edges` are `ID`s rather than `Int32`s.
"""
struct IDPhiNode
    edges::Vector{ID}
    values::Vector{Any}
end

Base.:(==)(x::IDPhiNode, y::IDPhiNode) = x.edges == y.edges && x.values == y.values

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
const Terminator = Union{Switch,IDGotoIfNot,IDGotoNode,ReturnNode}

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
    const IDInstPair = Tuple{ID, NewInstruction}
"""
const IDInstPair = Tuple{ID,NewInstruction}

"""
    BBlock(id::ID, inst_pairs::Vector{IDInstPair})

Convenience constructor -- splits `inst_pairs` into a `Vector{ID}` and `InstVector` in order
to build a `BBlock`.
"""
function BBlock(id::ID, inst_pairs::Vector{IDInstPair})
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
    insert_before_terminator!(bb::BBlock, id::ID, inst::NewInstruction)::Nothing

If the final instruction in `bb` is a `Terminator`, insert `inst` immediately before it.
Otherwise, insert `inst` at the end of the block.
"""
function insert_before_terminator!(bb::BBlock, id::ID, inst::NewInstruction)::Nothing
    insert!(bb, length(bb.insts) + (terminator(bb) === nothing ? 1 : 0), id, inst)
    return nothing
end

"""
    collect_stmts(bb::BBlock)::Vector{IDInstPair}

Returns a `Vector` containing the `ID`s and instructions associated to each line in `bb`.
These should be assumed to be ordered.
"""
collect_stmts(bb::BBlock)::Vector{IDInstPair} = collect(zip(bb.inst_ids, bb.insts))

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
function BBCode(ir::Union{IRCode,BBCode}, new_blocks::Vector{BBlock})
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

Compute a map from the `ID` of each `BBlock` in `ir` to its possible successors.
"""
compute_all_successors(ir::BBCode)::Dict{ID,Vector{ID}} = _compute_all_successors(ir.blocks)

"""
    _compute_all_successors(blks::Vector{BBlock})::Dict{ID, Vector{ID}}

Internal method implementing [`compute_all_successors`](@ref). This method is easier to
construct test cases for because it only requires the collection of `BBlocks`, not all of
the other stuff that goes into a `BBCode`.
"""
@noinline function _compute_all_successors(blks::Vector{BBlock})::Dict{ID,Vector{ID}}
    succs = map(enumerate(blks)) do (n, blk)
        is_final_block = n == length(blks)
        t = terminator(blk)
        if t === nothing
            return is_final_block ? ID[] : ID[blks[n + 1].id]
        elseif t isa IDGotoNode
            return [t.label]
        elseif t isa IDGotoIfNot
            return is_final_block ? ID[t.dest] : ID[t.dest, blks[n + 1].id]
        elseif t isa ReturnNode
            return ID[]
        elseif t isa Switch
            return vcat(t.dests, t.fallthrough_dest)
        else
            error("Unhandled terminator $t")
        end
    end
    return Dict{ID,Vector{ID}}((b.id, succ) for (b, succ) in zip(blks, succs))
end

"""
    compute_all_predecessors(ir::BBCode)::Dict{ID, Vector{ID}}

Compute a map from the `ID` of each `BBlock` in `ir` to its possible predecessors.
"""
function compute_all_predecessors(ir::BBCode)::Dict{ID,Vector{ID}}
    return _compute_all_predecessors(ir.blocks)
end

"""
    _compute_all_predecessors(blks::Vector{BBlock})::Dict{ID, Vector{ID}}

Internal method implementing [`compute_all_predecessors`](@ref). This method is easier to
construct test cases for because it only requires the collection of `BBlocks`, not all of
the other stuff that goes into a `BBCode`.
"""
function _compute_all_predecessors(blks::Vector{BBlock})::Dict{ID,Vector{ID}}
    successor_map = _compute_all_successors(blks)

    # Initialise predecessor map to be empty.
    ks = collect(keys(successor_map))
    predecessor_map = Dict{ID,Vector{ID}}(zip(ks, map(_ -> ID[], ks)))

    # Find all predecessors by iterating through the successor map.
    for (k, succs) in successor_map
        for succ in succs
            push!(predecessor_map[succ], k)
        end
    end

    return predecessor_map
end

"""
    collect_stmts(ir::BBCode)::Vector{IDInstPair}

Produce a `Vector` containing all of the statements in `ir`. These are returned in
order, so it is safe to assume that element `n` refers to the `nth` element of the `IRCode`
associated to `ir`. 
"""
collect_stmts(ir::BBCode)::Vector{IDInstPair} = reduce(vcat, map(collect_stmts, ir.blocks))

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

"""
    control_flow_graph(bb_code::BBCode)::Core.Compiler.CFG

Computes the `Core.Compiler.CFG` object associated to this `bb_code`.
"""
control_flow_graph(bb_code::BBCode)::Core.Compiler.CFG = _control_flow_graph(bb_code.blocks)

"""
    _control_flow_graph(blks::Vector{BBlock})::Core.Compiler.CFG

Internal function, used to implement [`control_flow_graph`](@ref). Easier to write test
cases for because there is no need to construct an ensure BBCode object, just the `BBlock`s.
"""
function _control_flow_graph(blks::Vector{BBlock})::Core.Compiler.CFG

    # Get IDs of predecessors and successors.
    preds_ids = _compute_all_predecessors(blks)
    succs_ids = _compute_all_successors(blks)

    # Construct map from block ID to block number.
    block_ids = map(b -> b.id, blks)
    id_to_num = Dict{ID,Int}(zip(block_ids, collect(eachindex(block_ids))))

    # Convert predecessor and successor IDs to numbers.
    preds = map(id -> sort(map(p -> id_to_num[p], preds_ids[id])), block_ids)
    succs = map(id -> sort(map(s -> id_to_num[s], succs_ids[id])), block_ids)

    # Predecessor of entry block is `0`. This needs to be added in manually.
    @static if VERSION >= v"1.11"
        push!(preds[1], 0)
    end

    # Compute the statement numbers associated to each basic block.
    index = vcat(0, cumsum(map(length, blks))) .+ 1
    basic_blocks = map(eachindex(blks)) do n
        stmt_range = Core.Compiler.StmtRange(index[n], index[n + 1] - 1)
        return Core.Compiler.BasicBlock(stmt_range, preds[n], succs[n])
    end
    return Core.Compiler.CFG(basic_blocks, index[2:(end - 1)])
end

"""
    _instructions_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector

Pulls out the instructions from `insts`, and calls `__line_numbers_to_block_numbers!`.
"""
function _lines_to_blocks(insts::InstVector, cfg::CC.CFG)::InstVector
    stmts = __line_numbers_to_block_numbers!(Any[x.stmt for x in insts], cfg)
    return map((inst, stmt) -> NewInstruction(inst; stmt), insts, stmts)
end

"""
    __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)

Converts any edges in `GotoNode`s, `GotoIfNot`s, `PhiNode`s, and `:enter` expressions which
refer to line numbers into references to block numbers. The `cfg` provides the information
required to perform this conversion.

For context, `CodeInfo` objects have references to line numbers, while `IRCode` uses
block numbers.

This code is copied over directly from the body of `Core.Compiler.inflate_ir!`.
"""
function __line_numbers_to_block_numbers!(insts::Vector{Any}, cfg::CC.CFG)
    for i in eachindex(insts)
        stmt = insts[i]
        if isa(stmt, GotoNode)
            insts[i] = GotoNode(CC.block_for_inst(cfg, stmt.label))
        elseif isa(stmt, GotoIfNot)
            insts[i] = GotoIfNot(stmt.cond, CC.block_for_inst(cfg, stmt.dest))
        elseif isa(stmt, PhiNode)
            insts[i] = PhiNode(
                Int32[CC.block_for_inst(cfg, Int(edge)) for edge in stmt.edges], stmt.values
            )
        elseif Meta.isexpr(stmt, :enter)
            stmt.args[1] = CC.block_for_inst(cfg, stmt.args[1]::Int)
            insts[i] = stmt
        end
    end
    return insts
end

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

"""
    new_inst_vec(x::CC.InstructionStream)

Convert an `Compiler.InstructionStream` into a list of `Compiler.NewInstruction`s.
"""
function new_inst_vec(x::CC.InstructionStream)
    stmt = @static VERSION < v"1.11.0-rc4" ? x.inst : x.stmt
    return map((v...,) -> NewInstruction(v...), stmt, x.type, x.info, x.line, x.flag)
end

# Maps from positional names (SSAValues for nodes, Integers for basic blocks) to IDs.
const SSAToIdDict = Dict{SSAValue,ID}
const BlockNumToIdDict = Dict{Integer,ID}

"""
    _ssas_to_ids(insts::InstVector)::Tuple{Vector{ID}, InstVector}

Assigns an ID to each line in `stmts`, and replaces each instance of an `SSAValue` in each
line with the corresponding `ID`. For example, a call statement of the form
`Expr(:call, :f, %4)` is be replaced with `Expr(:call, :f, id_assigned_to_%4)`.
"""
function _ssas_to_ids(insts::InstVector)::Tuple{Vector{ID},InstVector}
    ids = map(_ -> ID(), insts)
    val_id_map = SSAToIdDict(zip(SSAValue.(eachindex(insts)), ids))
    return ids, map(Base.Fix1(_ssa_to_ids, val_id_map), insts)
end

"""
    _ssa_to_ids(d::SSAToIdDict, inst::NewInstruction)

Produce a new instance of `inst` in which all instances of `SSAValue`s are replaced with
the `ID`s prescribed by `d`, all basic block numbers are replaced with the `ID`s
prescribed by `d`, and `GotoIfNot`, `GotoNode`, and `PhiNode` instances are replaced with
the corresponding `ID` versions.
"""
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

"""
    _block_nums_to_ids(insts::InstVector, cfg::CC.CFG)::Tuple{Vector{ID}, InstVector}

Assign to each basic block in `cfg` an `ID`. Replace all integers referencing block numbers
in `insts` with the corresponding `ID`. Return the `ID`s and the updated instructions.
"""
function _block_nums_to_ids(insts::InstVector, cfg::CC.CFG)::Tuple{Vector{ID},InstVector}
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
    insts = _ids_to_line_numbers(bb_code)
    cfg = control_flow_graph(bb_code)
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

"""
    _lower_switch_statements(bb_code::BBCode)

Converts all `Switch`s into a semantically-equivalent collection of `GotoIfNot`s. See the
`Switch` docstring for an explanation of what is going on here.
"""
function _lower_switch_statements(bb_code::BBCode)
    new_blocks = Vector{BBlock}(undef, 0)
    for block in bb_code.blocks
        t = terminator(block)
        if t isa Switch

            # Create new block without the `Switch`.
            bb = BBlock(block.id, block.inst_ids[1:(end - 1)], block.insts[1:(end - 1)])
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

"""
    _ids_to_line_numbers(bb_code::BBCode)::InstVector

For each statement in `bb_code`, returns a `NewInstruction` in which every `ID` is replaced
by either an `SSAValue`, or an `Int64` / `Int32` which refers to an `SSAValue`.
"""
function _ids_to_line_numbers(bb_code::BBCode)::InstVector

    # Construct map from `ID`s to `SSAValue`s.
    block_ids = [b.id for b in bb_code.blocks]
    block_lengths = map(length, bb_code.blocks)
    block_start_ssas = SSAValue.(vcat(1, cumsum(block_lengths)[1:(end - 1)] .+ 1))
    line_ids = concatenate_ids(bb_code)
    line_ssas = SSAValue.(eachindex(line_ids))
    id_to_ssa_map = Dict(zip(vcat(block_ids, line_ids), vcat(block_start_ssas, line_ssas)))

    # Apply map.
    return [_to_ssas(id_to_ssa_map, stmt) for stmt in concatenate_stmts(bb_code)]
end

"""
    _to_ssas(d::Dict, inst::NewInstruction)

Like `_ssas_to_ids`, but in reverse. Converts IDs to SSAValues / (integers corresponding
to ssas).
"""
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

"""
    _remove_double_edges(ir::BBCode)::BBCode

If the `dest` field of an `IDGotoIfNot` node in block `n` of `ir` points towards the `n+1`th
block then we have two edges from block `n` to block `n+1`. This transformation replaces all
such `IDGotoIfNot` nodes with unconditional `IDGotoNode`s pointing towards the `n+1`th block
in `ir`.
"""
function _remove_double_edges(ir::BBCode)
    new_blks = map(enumerate(ir.blocks)) do (n, blk)
        t = terminator(blk)
        if t isa IDGotoIfNot && t.dest == ir.blocks[n + 1].id
            new_insts = vcat(blk.insts[1:(end - 1)], NewInstruction(t; stmt=IDGotoNode(t.dest)))
            return BBlock(blk.id, blk.inst_ids, new_insts)
        else
            return blk
        end
    end
    return BBCode(ir, new_blks)
end

"""
    _build_graph_of_cfg(blks::Vector{BBlock})::Tuple{SimpleDiGraph, Dict{ID, Int}}

Builds a `SimpleDiGraph`, `g`, representing of the CFG associated to `blks`, where `blks`
comprises the collection of basic blocks associated to a `BBCode`.
This is a type from Graphs.jl, so constructing `g` makes it straightforward to analyse the
control flow structure of `ir` using algorithms from Graphs.jl.

Returns a 2-tuple, whose first element is `g`, and whose second element is a map from
the `ID` associated to each basic block in `ir`, to the `Int` corresponding to its node
index in `g`.
"""
function _build_graph_of_cfg(blks::Vector{BBlock})::Tuple{SimpleDiGraph,Dict{ID,Int}}
    node_ints = collect(eachindex(blks))
    id_to_int = Dict(zip(map(blk -> blk.id, blks), node_ints))
    successors = _compute_all_successors(blks)
    g = SimpleDiGraph(length(blks))
    for blk in blks, successor in successors[blk.id]
        add_edge!(g, id_to_int[blk.id], id_to_int[successor])
    end
    return g, id_to_int
end

"""
    _distance_to_entry(blks::Vector{BBlock})::Vector{Int}

For each basic block in `blks`, compute the distance from it to the entry point (the first
block. The distance is `typemax(Int)` if no path from the entry point to a given node.
"""
function _distance_to_entry(blks::Vector{BBlock})::Vector{Int}
    g, id_to_int = _build_graph_of_cfg(blks)
    return dijkstra_shortest_paths(g, id_to_int[blks[1].id]).dists
end

"""
    sort_blocks!(ir::BBCode)::BBCode

Ensure that blocks appear in order of distance-from-entry-point, where distance the
distance from block b to the entry point is defined to be the minimum number of basic
blocks that must be passed through in order to reach b.

For reasons unknown (to me, Will), the compiler / optimiser needs this for inference to
succeed. Since we do quite a lot of re-ordering on the reverse-pass of AD, this is a problem
there.

WARNING: use with care. Only use if you are confident that arbitrary re-ordering of basic
blocks in `ir` is valid. Notably, this does not hold if you have any `IDGotoIfNot` nodes in
`ir`.
"""
function sort_blocks!(ir::BBCode)::BBCode
    I = sortperm(_distance_to_entry(ir.blocks))
    ir.blocks .= ir.blocks[I]
    return ir
end

"""
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
"""
function characterise_unique_predecessor_blocks(
    blks::Vector{BBlock}
)::Tuple{Dict{ID,Bool},Dict{ID,Bool}}

    # Obtain the block IDs in order -- this ensures that we get the entry block first.
    blk_ids = ID[b.id for b in blks]
    preds = _compute_all_predecessors(blks)
    succs = _compute_all_successors(blks)

    # The bulk of blocks can be hanled by this general loop.
    is_unique_pred = Dict{ID,Bool}()
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
    pred_is_unique_pred = Dict{ID,Bool}()
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
    is_reachable_return_node(x::ReturnNode)

Determine whether `x` is a `ReturnNode`, and if it is, if it is also reachable. This is
purely a function of whether or not its `val` field is defined or not.
"""
is_reachable_return_node(x::ReturnNode) = isdefined(x, :val)
is_reachable_return_node(x) = false

"""
    characterise_used_ids(stmts::Vector{IDInstPair})::Dict{ID, Bool}

For each line in `stmts`, determine whether it is referenced anywhere else in the code.
Returns a dictionary containing the results. An element is `false` if the corresponding
`ID` is unused, and `true` if is used.
"""
function characterise_used_ids(stmts::Vector{IDInstPair})::Dict{ID,Bool}
    ids = first.(stmts)
    insts = last.(stmts)

    # Initialise to false.
    is_used = Dict{ID,Bool}(zip(ids, fill(false, length(ids))))

    # Hunt through the instructions, flipping a value in is_used to true whenever an ID
    # is encountered which corresponds to an SSA.
    for inst in insts
        _find_id_uses!(is_used, inst.stmt)
    end
    return is_used
end

"""
    _find_id_uses!(d::Dict{ID, Bool}, x)

Helper function used in [`characterise_used_ids`](@ref). For all uses of `ID`s in `x`, set
the corresponding value of `d` to `true`.

For example, if `x = ReturnNode(ID(5))`, then this function sets `d[ID(5)] = true`.
"""
function _find_id_uses!(d::Dict{ID,Bool}, x::Expr)
    for arg in x.args
        in(arg, keys(d)) && setindex!(d, true, arg)
    end
end
function _find_id_uses!(d::Dict{ID,Bool}, x::IDGotoIfNot)
    return in(x.cond, keys(d)) && setindex!(d, true, x.cond)
end
_find_id_uses!(::Dict{ID,Bool}, ::IDGotoNode) = nothing
function _find_id_uses!(d::Dict{ID,Bool}, x::PiNode)
    return in(x.val, keys(d)) && setindex!(d, true, x.val)
end
function _find_id_uses!(d::Dict{ID,Bool}, x::IDPhiNode)
    v = x.values
    for n in eachindex(v)
        isassigned(v, n) && in(v[n], keys(d)) && setindex!(d, true, v[n])
    end
end
function _find_id_uses!(d::Dict{ID,Bool}, x::ReturnNode)
    return isdefined(x, :val) && in(x.val, keys(d)) && setindex!(d, true, x.val)
end
_find_id_uses!(d::Dict{ID,Bool}, x::QuoteNode) = nothing
_find_id_uses!(d::Dict{ID,Bool}, x) = nothing

"""
    _is_reachable(blks::Vector{BBlock})::Vector{Bool}

Computes a `Vector` whose length is `length(blks)`. The `n`th element is `true` iff it is
possible for control flow to reach the `n`th block.
"""
_is_reachable(blks::Vector{BBlock})::Vector{Bool} = _distance_to_entry(blks) .< typemax(Int)

"""
    remove_unreachable_blocks!(ir::BBCode)::BBCode

If a basic block in `ir` cannot possibly be reached during execution, then it can be safely
removed from `ir` without changing its functionality.
A block is unreachable if either:
1. it has no predecessors _and_ it is not the first block, or
2. all of its predecessors are themselves unreachable.

For example, consider the following IR:
```jldoctest remove_unreachable_blocks
julia> ir = Mooncake.ircode(
           Any[Core.ReturnNode(nothing), Expr(:call, sin, 5), Core.ReturnNode(Core.SSAValue(2))],
           Any[Any, Any, Any],
       );
```
There is no possible way to reach the second basic block (lines 2 and 3). Applying this
function will therefore remove it, yielding the following:
```jldoctest remove_unreachable_blocks
julia> Mooncake.IRCode(Mooncake.remove_unreachable_blocks!(Mooncake.BBCode(ir)))
1 1 ─     return nothing
```

In the blocks which have not been removed, there may be references to blocks which have been
removed. For example, the `edge`s in a `PhiNode` may contain a reference to a removed block.
These references are removed in-place from these remaining blocks, so this function will (in
general) modify `ir`.
"""
remove_unreachable_blocks!(ir::BBCode) = BBCode(ir, _remove_unreachable_blocks!(ir.blocks))

function _remove_unreachable_blocks!(blks::Vector{BBlock})

    # Figure out which blocks are reachable.
    is_reachable = _is_reachable(blks)

    # Collect all blocks which are reachable.
    remaining_blks = blks[is_reachable]

    # For each reachable block, remove any references to removed blocks. These can appear in
    # `PhiNode`s with edges that come from remove blocks.
    removed_block_ids = map(idx -> blks[idx].id, findall(!, is_reachable))
    for blk in remaining_blks, inst in blk.insts
        stmt = inst.stmt
        stmt isa IDPhiNode || continue
        for n in reverse(1:length(stmt.edges))
            if stmt.edges[n] in removed_block_ids
                deleteat!(stmt.edges, n)
                deleteat!(stmt.values, n)
            end
        end
    end

    return remaining_blks
end

end
