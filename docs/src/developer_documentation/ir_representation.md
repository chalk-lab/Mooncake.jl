# IR Representations and Code Transformations

Mooncake.jl works by transforming Julia's SSA-form (static single assignment) Intermediate Representation (IR), so a good understanding of Julia's IR is needed to understand Mooncake.
Furthermore, Mooncake holds Julia's IR in a different data structure than the one usually used when producing code for reverse-mode AD.
We discuss both data structures below, and provide examples of the kinds of transformations which must be applied to Julia's IR in order to implement AD, contrasting the two different data structures.

Please note that Julia's SSA-form IR typically changes representation slightly between minor versions of Julia, as it's not part of the public interface of the language.
The information below is accurate on version 1.11.4, but you might well find that things are slightly different on different versions.

## Julia's SSA-form IR

### Straight-Line Code

You can find the IR associated to a given signature using `Base.code_ircode_by_type`:
```jldoctest
julia> function foo(x)
           y = sin(x)
           z = cos(y)
           return z
       end
foo (generic function with 1 method)

julia> signature = Tuple{typeof(foo), Float64}
Tuple{typeof(foo), Float64}

julia> Base.code_ircode_by_type(signature)[1][1]
2 1 ─ %1 = invoke sin(_2::Float64)::Float64
3 │   %2 = invoke cos(%1::Float64)::Float64
4 └──      return %2
```

What you can see here is that the calls to `sin` and `cos` in the original function are associated to a number, denoted `%1` and `%2`. We refers to these as the "ssa"s associated to each statement.
Each statement is associated to a single ssa, and this association is determined by where it appears in the list of statements -- the first statement is associated to `%1`, the second to `%2`, and so on.
You will also notice that the argument `x` has been replaced with a `_2` in the first statement -- in general, all uses of the `n`th argument are indicated by `_n` (the first argument is the function itself).
The final statement requires no explanation.

Note that this IR is obtained after both type inference and various Julia-level optimisation passes.
This means that the type information is available for each statement.
For example, the `::Float64` at the end of the first and second statements indicates that the type of `%1` and `%2` is always `Float64`.
The types are also displayed at uses -- the call to `sin` involves `_2::Float64`, not just `_2`.

Additionally notice that the statements are `invoke` statements, rather than just call statements.
In Julia's IR, an `invoke` statement represents static dispatch to a particular `MethodInstance` -- i.e. running type inference + optimisation passes has determined enough about the argument types to make it possible to know exactly which `MethodInstance` of `sin` and `cos` to call.
This is a very common occurrence in type-stable code.

### Control Flow

The above is straight-line code -- it does not involve any control flow.
Julia has several statements which are involved in handling control flow.
For example
```jldoctest bar
julia> function bar(x)
           if x > 0
               return x
           else
               return 5x
           end
       end
bar (generic function with 1 method)

julia> Base.code_ircode_by_type(Tuple{typeof(bar),Float64})[1][1]
2 1 ─ %1 = Base.lt_float(0.0, _2)::Bool
  │   %2 = Base.or_int(%1, false)::Bool
  └──      goto #3 if not %2
3 2 ─      return _2
5 3 ─ %5 = Base.mul_float(5.0, _2)::Float64
  └──      return %5
```
In this example we see the statement `goto #3 if not %2`.
This should be read as "jump to basic block 3 if %2 is `false`".
The second half of that statement should be clear, but to understand the first half requires knowing what a basic block is:
```julia
  1 ─
  │  
  └──
  2 ─
  3 ─
  └──
```
Here, everything is removed from the above example except for information about the basic block structure.
To first approximation, each basic block is a sequence of statements which _must_ always execute one after the other.
Once all statements in a basic block have run, we typically either jump to another basic block, or hit a `return` statement.
In this example, we have three basic blocks -- you can see this from the numbers `1`, `2`, and `3`.
The first basic block comprises three statements, the second only one statement, and the third two statements.
Another way to investigate this structure is to look at the control-flow graph associated to the IR:
```jldoctest bar
julia> Base.code_ircode_by_type(Tuple{typeof(bar),Float64})[1][1].cfg
CFG with 3 blocks:
  bb 1 (stmts 1:3) → bb 3, 2
  bb 2 (stmt 4)
  bb 3 (stmts 5:6)
```
For example, the above states that "bb" (basic block) 1 comprises statements 1 to 3, and has successor blocks 2 and 3 (ie. once the statements in basic block 1 have executed, we know for certain that either those in block 2 or block 3 will run next).
Blocks 2 and 3 have no successors, because they both end in a `return` statement.
The predecessors of each basic block (the blocks which could possibly have run immediately prior to a given block) are also stored in the blocks of the `CFG`, even though this is not printed -- you should have a play around with this data structure to see what is in there.

Additionally, note that `Base.lt_float` (used to check if one floating point number is less than another) and `Base.or_int` do not appear as `invoke` statements -- this is because they are not generic Julia functions.
Rather, they are Julia intrinsics:
```jldoctest
julia> Base.lt_float
lt_float (intrinsic function #31)
```
These intrinsics have special handling in the compiler.
Either way, the overall point is to be aware that these kinds of low-level intrinsics exist, and appear regularly in Julia IR.

### Simple Loops and Phi-Nodes

Finally, we shall consider a simple loop:
```jldoctest my_factorial
julia> function my_factorial(x::Int)
           n = 0
           s = 1
           while n < x
               n += 1
               s *= n
           end
           return s
       end
my_factorial (generic function with 1 method)

julia> ir = Base.code_ircode_by_type(Tuple{typeof(my_factorial), Int})[1][1]
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %7)::Int64
  │   %3 = φ (#1 => 0, #3 => %6)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─ %6 = Base.add_int(%3, 1)::Int64
6 │   %7 = Base.mul_int(%2, %6)::Int64
7 └──      goto #2
8 4 ─      return %2
```
There are a few new intrinsics that we have not seen previously (`Base.slt_int` (used to check whether one int is strictly less than another), `Base.add_int`, and `Base.mul_int`).
Additionally, there is the node `goto #2`, which simply states that control flow should jump to basic block 2 whenever it is hit.

The most interesting additional nodes, however, are the two `φ` (phi) nodes.
These are a defining feature of SSA-form IR. Consider the first `φ` node:
```julia
%2 = φ (#1 => 1, #3 => %7)
```
means ssa `%2` takes value `1` if the previous basic block was `#1`, and whatever value is currently associated to ssa `%7` if the previous basic block was `#3`.
It is helpful to step through this code in your head: upon calling `my_factorial` we enter basic block `#1`, and proceed directly to basic block `#2`.
Therefore, on the first iteration, `%2` takes value `1`. We never return to basic block `#1`, so all subsequent visits to this `φ` node will result in `%2` taking the value associated to `%7`.
You should convince yourself that `%2` corresponds to the value of `s` at each iteration, and `%3` corresponds to the value of `n` at each iteration.


### Summary

Julia's SSA-form IR comprises a sequence of statements, which can be broken down into a collection of basic blocks.
Each basic block begins with a (potentially empty) collection of phi nodes, followed by a sequence of statements, and potentially finished by a _terminator_ (goto, goto-if-not, return).
Control flow is dictated by the terminators at the end of basic blocks -- if there is no terminator then we "fall through" to the next basic block.

## Julia Compiler's IR Datastructure

The Julia compiler represents the IR associated to a signature via a `struct` called `Core.Compiler.IRCode`.
The statements are given by the `stmts` field, which is a `Core.Compiler.InstructionStream`.
An `InstructionStream` is a collection of 5 `Vector`s, each of which have the same length.
The properties of the `n`th statement in the `IR` are given by the `n`th element of each of these vectors.
For example, the `stmt` field contains the statement itself, the `type` field contains the inferred type associated to the statement.
We'll skip the rest for now.
For example, the statements associated to the `my_factorial` function above can be retrieved as follows:
```jldoctest my_factorial
julia> ir.stmts.stmt
9-element Vector{Any}:
 nothing
 :(φ (%1 => 1, %3 => %7))
 :(φ (%1 => 0, %3 => %6))
 :(Base.slt_int(%3, _2))
 :(goto %4 if not %4)
 :(Base.add_int(%3, 1))
 :(Base.mul_int(%2, %6))
 :(goto %2)
 :(return %2)
```
The types can be accessed in a similar way:
```jldoctest my_factorial
julia> ir.stmts.type
9-element Vector{Any}:
 Nothing
 Int64
 Int64
 Bool
 Any
 Int64
 Int64
 Any
 Any
```

As seen in [Control Flow](@ref), the control flow graph (CFG) is represented as a separate data structure, stored in the `cfg` field of the `IRCode`.
The argument types associated to the signature are stored in the `argtypes` field of the `IRCode`.

## An Alternative IR Datastructure

`IRCode` is a perfectly good way to represent Julia's IR the vast majority of the time.
For example, it suffices for the code transformations required for forwards-mode AD.
However, IR transformations involving multiple changes to the control flow structure of a programme are needed in reverse-mode, and are prohibitively awkward to undertake using `IRCode`.
Mooncake's implementation of reverse-mode AD instead makes use of a custom representation of Julia's IR, called `BBCode`.
We emphasise that `BBCode` represents the _same_ thing under the hood, it is just represented in memory in a slightly different way, such that certain kinds of transformations are straightforward to implement.

You can construct a `BBCode` from an `IRCode`, and vice versa:
```jldoctest my_factorial
julia> using Mooncake: BBCode

julia> bb_ir = BBCode(ir);

julia> bb_ir isa BBCode
true

julia> Core.Compiler.IRCode(bb_ir)
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %7)::Int64
  │   %3 = φ (#1 => 0, #3 => %6)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─ %6 = Base.add_int(%3, 1)::Int64
6 │   %7 = Base.mul_int(%2, %6)::Int64
7 └──      goto #2
8 4 ─      return %2
```
At present, `BBCode` does not display itself nicely, so to look at it we must either inspect its fields, or convert it back to an `IRCode` (which _does_ print nicely).

Instead of storing all of the statements in a single vector (and the types in their own vector, etc), `BBCode` stores all statements associated to a particular basic block in a `Mooncake.BBlock`, and stores these in a `Vector{Mooncake.BBlock}`.
```jldoctest my_factorial
julia> typeof(bb_ir.blocks)
Vector{BBlock} (alias for Array{Mooncake.BasicBlockCode.BBlock, 1})
```
Each `BBlock` has a field `insts`, containing the statements associated to that basic block.
This is stored as a `Vector{Core.Compiler.NewInstruction}`, because `Core.Compiler.NewInstruction` contains the 5 fields that define an instruction in `IRCode` (you should compare the fields of a `Core.Compiler.NewInstruction` with those of `Core.Compiler.InstructionStream` to see the correspondence).
For example, consider
```jldoctest my_factorial
julia> using Mooncake.BasicBlockCode: ID # to improve printing

julia> bb_ir.blocks[3].insts[1]
Core.Compiler.NewInstruction(:(Base.add_int(ID(97), 1)), Int64, Core.Compiler.NoCallInfo(), 9, 0x000012e0)
```
This is the first instruction of the third basic block.
The first field is a call to `Base.add_int`, the second field is `Int64` (we promise that the other fields are just copies of the corresponding data from the `Core.Compiler.InstructionStream` in the original `IRCode` representation of this IR).

The other structural difference is that `BBCode` has no field containing the control-flow graph.
Instead, the control-flow graph is represented implicitly as part of the `blocks` field.
The upside of this is that any transformations of `blocks` which modify the CFG are automatically reflected in the `blocks` -- there is no need to perform any book-keeping to ensure that the CFG is kept in sync with the instructions.
This saves both time and memory when inserting new basic blocks -- when basic block structure changes, a scan of the entire `IRCode` is required to modify any statements which refer to a given block, and yields code simplifications.
The downside is that the CFG must be computed whenever we need to know about it.
As a resut, neither `IRCode` nor `BBCode`'s representation of the CFG is strictly better than the other.
To extract CFG-related information from a `BBCode`, see [`Mooncake.BasicBlockCode.compute_all_successors`](@ref), [`Mooncake.BasicBlockCode.compute_all_predecessors`](@ref), and [`Mooncake.BasicBlockCode.control_flow_graph`](@ref).


The final major difference between `IRCode` and `BBCode` is that all ssa values in an `IRCode` (`%1`, `%2`, `%n`, etc) are replaced with unique `ID`s. The `ID` associated to a statement is stored separately from the statement in the `inst_ids` field of a `BBlock`:
```jldoctest my_factorial
julia> bb_ir.blocks[3].inst_ids
3-element Vector{ID}:
 ID(100)
 ID(101)
 ID(102)
```
There is exactly one `ID` per instruction, and it is an error to have the same `ID` associated to multiple instructions.
Similarly, while the number associated to a basic block in `IRCode` is a function of the number of basic blocks which preceed it, the `ID` of a basic block in `BBCode` is stored in its `id` field:
```jldoctest my_factorial
julia> bb_ir.blocks[3].id
ID(106)
```
As a result of this, all references to ssa values and basic block numbers in `IRCode` are replaced with `ID`s in `BBCode`.
The purpose of this is to guarantee that the "name" of a basic block and an instruction does not change when you insert new basic blocks and new instructions.
We shall see how this is useful in the examples below.

## Code Transformations

In what follows, we look at a few transformations of Julia's IR, and see how these can be undertaken using both `IRCode` and `BBCode`.
The purpose is two-fold:
1. to enable readers to understand the code used to implement Mooncake, and
2. to highlight the relative merits of `IRCode` vs `BBCode`.

### Replacing Instructions

This is a very simple code transformation.
It is used in both forwards-mode and reverse-mode in Mooncake to replace calls of the form
```julia
f(x, y, z)
```
with calls of the form
```julia
frule!!(f, x, y, z)
```
This kind of transformation is performed in basically the same way for both `IRCode` and `BBCode`.
For example, the `mul_int` statement associated to ssa `%7` can be replaced with an `add_int` statement as follows:
```jldoctest my_factorial
julia> using Core: SSAValue

julia> const CC = Core.Compiler;

julia> new_ir = Core.Compiler.copy(ir);

julia> old_stmt = new_ir.stmts.stmt[7]
:(Base.mul_int(%2, %6))

julia> new_stmt = Expr(:call, Base.add_int, old_stmt.args[2:end]...)
:((Core.Intrinsics.add_int)(%2, %6))

julia> # new_ir[SSAValue(7)][:stmt] = new_stmt
       CC.setindex!(CC.getindex(new_ir, SSAValue(7)), new_stmt, :stmt);

julia> new_ir
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %7)::Int64
  │   %3 = φ (#1 => 0, #3 => %6)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─ %6 = Base.add_int(%3, 1)::Int64
6 │   %7 = (Core.Intrinsics.add_int)(%2, %6)::Int64
7 └──      goto #2
8 4 ─      return %2
```
Observe that ssa `7` has been replaced with the new `:call` to `add_int`.
Unfortunately, in order to avoid committing type-piracy against `Core.Compiler`, we cannot currently write `new_ir[SSAValue(7)][:stmt]`. (`CC.getindex` is a different function from `Base.getindex` -- the same is true for `CC.setindex!` vs `Base.setindex!`).
In general, I would recommend defining helper functions to improve the DRYness of your code.

The same transformation can be performed on `BBCode`:
```jldoctest my_factorial
julia> bb_ir_copy = copy(bb_ir);

julia> old_inst = bb_ir_copy.blocks[3].insts[2]
Core.Compiler.NewInstruction(:(Base.mul_int(ID(96), ID(100))), Int64, Core.Compiler.NoCallInfo(), 10, 0x000012e0)

julia> new_stmt = Expr(:call, Base.add_int, old_inst.stmt.args[2:end]...)
:((Core.Intrinsics.add_int)(ID(96), ID(100)))

julia> bb_ir_copy.blocks[3].insts[2] = CC.NewInstruction(old_inst; stmt=new_stmt);

julia> CC.IRCode(bb_ir_copy)
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %7)::Int64
  │   %3 = φ (#1 => 0, #3 => %6)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─ %6 = Base.add_int(%3, 1)::Int64
6 │   %7 = (Core.Intrinsics.add_int)(%2, %6)::Int64
7 └──      goto #2
8 4 ─      return %2
```
As you can see, in both cases we wind up with the same `IRCode` at the end.

### Inserting New Instructions

Inserting entirely new instructions into the IR requires a little more thought, but is ultimately very straightforward using either `IRCode` or `BBCode`.

First, `IRCode`.
Suppose that we wish to insert another instruction immediately before the first `add_int` instruction which multiplies `%3` by 2 before adding `1` to it in `#3`.
In `IRCode`, this kind of modification requires some care, because naively inserting an instruction between the 5th and 6th line changes the name of all instructions from the 6th onwards.
Consequently, we need to replace all existing uses of e.g. `%6` with uses of `%7`, etc.
Happily, `IRCode` has a mechanism to achieve just this.
```jldoctest my_factorial
julia> ni = CC.NewInstruction(Expr(:call, Base.mul_int, SSAValue(3), 2), Int)
Core.Compiler.NewInstruction(:((Core.Intrinsics.mul_int)(%3, 2)), Int64, Core.Compiler.NoCallInfo(), nothing, nothing)

julia> new_ssa = CC.insert_node!(new_ir, SSAValue(6), ni)
:(%10)

julia> new_ir
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %7)::Int64
  │   %3 = φ (#1 => 0, #3 => %6)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─      (Core.Intrinsics.mul_int)(%3, 2)::Int64
  │   %6 = Base.add_int(%3, 1)::Int64
6 │   %7 = (Core.Intrinsics.add_int)(%2, %6)::Int64
7 └──      goto #2
8 4 ─      return %2
```
`CC.insert_node!(ir, ssa, new_inst)` inserts `new_inst` into `ir` immediately before `ssa`, and attaches it to the same basic block as `ssa` resides.
It returns an `SSAValue`, which is the "name" associated to the inserted instruction in the IR.
Here, we see it has inserted the instruction to multiply `%3` by `2` immediately before `%6`.
However, observe that the `IRCode` has not changed the name associated to the subsequent `add_int` instruction -- it still assigns to `%6`, despite not being the 6th statement in the IR anymore.
This is achieved via `IRCode`'s `new_nodes` field -- upon calling `CC.insert_node!`, rather than inserting the instruction directly into the `InstructionStream`, this list is appended to.
We can do this as many times as we like, and then call `CC.compact!` at the end to handle all of the book-keeping involved in inserting all of the statements, updating all ssa uses where required, and updating the `cfg` field of the IR.

Also observe that the inserted statement is printed without a `%10 =` at the start of it -- this is because there are not (yet) any uses of `%10`, so `IRCode` does not print it out (presumably in order to reduce visual noise).

To conclude this transformation, we replace the first argument of the `add_int` instruction with the new ssa returned by `insert_node!`, and then call `CC.compact!` to process all of the nodes currently in the `new_nodes` list, and produce a valid `IRCode`:
```jldoctest my_factorial
julia> stmt = CC.getindex(CC.getindex(new_ir, SSAValue(6)), :stmt)
:(Base.add_int(%3, 1))

julia> stmt.args[2] = new_ssa;

julia> new_ir
  1 ─       nothing::Nothing
4 2 ┄ %2  = φ (#1 => 1, #3 => %7)::Int64
  │   %3  = φ (#1 => 0, #3 => %6)::Int64
  │   %4  = Base.slt_int(%3, _2)::Bool
  └──       goto #4 if not %4
5 3 ─ %10 = (Core.Intrinsics.mul_int)(%3, 2)::Int64
  │   %6  = Base.add_int(%10, 1)::Int64
6 │   %7  = (Core.Intrinsics.add_int)(%2, %6)::Int64
7 └──       goto #2
8 4 ─       return %2

julia> new_ir = CC.compact!(new_ir)
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %8)::Int64
  │   %3 = φ (#1 => 0, #3 => %7)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
5 3 ─ %6 = (Core.Intrinsics.mul_int)(%3, 2)::Int64
  │   %7 = Base.add_int(%6, 1)::Int64
6 │   %8 = (Core.Intrinsics.add_int)(%2, %7)::Int64
7 └──      goto #2
8 4 ─      return %2
```
Observe that, before `compact!`-ing, the first instruction in basic block `#3` is still labelled as being `%10`.
After `compact!`-ing, we have standard sequentially-labelled IR again.
Note that the above is exactly the kind of thing that we do in our implementation of forwards-mode AD -- all insertions of nodes are performed in a single pass over the `IRCode`, and `CC.compact!` is called once at the end.

Performing this transformation using `BBCode` is similarly straightforward.
Since the name associated to instructions does not change when you insert another instruction, you really just need to insert an instruction + its `ID`, update the next instruction (as before), and you're done:
```jldoctest my_factorial
julia> using Mooncake.BasicBlockCode: ID, new_inst

julia> new_id = ID() # this produces a new unique `ID`.
ID(108)

julia> target_id = bb_ir_copy.blocks[3].insts[1].stmt.args[2] # find `ID` of argument to add_int.
ID(97)

julia> ni = new_inst(Expr(:call, Base.mul_int, target_id, 2), Int);

julia> insert!(bb_ir_copy.blocks[3], 1, new_id, ni)

julia> bb_ir_copy.blocks[3].insts[2].stmt.args[2] = new_id
ID(108)

julia> CC.IRCode(bb_ir_copy)
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %8)::Int64
  │   %3 = φ (#1 => 0, #3 => %7)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #4 if not %4
2 3 ─ %6 = (Core.Intrinsics.mul_int)(%3, 2)::Int64
5 │   %7 = Base.add_int(%6, 1)::Int64
6 │   %8 = (Core.Intrinsics.add_int)(%2, %7)::Int64
7 └──      goto #2
8 4 ─      return %2
```
We see here that `IRCode` and `BBCode` involve similar levels of complexity to insert an instruction.

### Inserting New Basic Blocks

This is the situation in which the design of `BBCode` shines vs `IRCode`.
`IRCode` does not, at present, really have much to say about transformations which change control flow.
It is, however, straightforward using `BBCode`.
Suppose that we wish to modify the above to display the value of `%2` if it is even on any given iteration.
Since this involves control flow, it necessarily requires at least one additional basic block.

We do this in two steps.
We first insert an additional basic block between blocks `#3` and `#4` which always prints out the value of `%2`, and then goes to block `#2`:
```jldoctest my_factorial
julia> using Mooncake.BasicBlockCode: BBlock, new_inst, IDGotoNode, IDGotoIfNot

julia> block_2_id = bb_ir_copy.blocks[2].id;

julia> new_bb_id = ID();

julia> new_bb = BBlock(
           new_bb_id,
           ID[ID(), ID()],
           CC.NewInstruction[
               new_inst(Expr(:call, println, CC.SSAValue(2))),
               new_inst(IDGotoNode(block_2_id)),
           ],
       );

julia> insert!(bb_ir_copy.blocks, 4, new_bb);

julia> CC.IRCode(bb_ir_copy)
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %8)::Int64
  │   %3 = φ (#1 => 0, #3 => %7)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #5 if not %4
2 3 ─ %6 = (Core.Intrinsics.mul_int)(%3, 2)::Int64
5 │   %7 = Base.add_int(%6, 1)::Int64
6 │   %8 = (Core.Intrinsics.add_int)(%2, %7)::Int64
7 └──      goto #2
2 4 ─      (println)(%2)::Any
  └──      goto #2
8 5 ─      return %2
```
Observe that, in this case, rather than creating `new_bb` and then inserting instructions into it, we simply create the block _with_ the instructions.
This programming style is often more convenient.
Additionally note that we create an `ID` for each statement in the new basic block.
These `ID`s are never actually used anywhere, but `BBCode` requires that each instruction be associated to an `ID`, so we must create them.

Additionally, note the usage of an [`Mooncake.BasicBlockCode.IDGotoNode`](@ref).
This is exactly the same thing as a `Core.Compiler.GotoNode`, except it contains an `ID` stating which basic block to jump to, rather than an `Int`.
Similarly, the [`Mooncake.BasicBlockCode.IDGotoIfNot`](@ref) is a direct translation of `Core.Compiler.GotoIfNot`, with the `dest` field being an `ID` rather than an `Int`.

Furthermore, note that the `goto if not` instruction at the end of basic block `#2` now (correctly) jumps to basic block `#5`, whereas before it jumped to block `#4`.
That is, by virtue of the fact that the `ID` associated to each basic block remains unchanged in `BBCode`, all pre-existing control flow relationships have remained the same.
Moreover, we did not have to write any book-keeping code to ensure that this update happened correctly.

Now that we've created the new basic block, we modify block `#3` to fall-through to the new block if `%2` is even, and to jump straight back to block `#2` if not:
```jldoctest my_factorial
julia> bb = bb_ir_copy.blocks[3];

julia> cond_id = ID();

julia> target_id = bb_ir_copy.blocks[2].inst_ids[1];

julia> insert!(bb, 4, cond_id, new_inst(Expr(:call, iseven, target_id)));

julia> bb.insts[end] = new_inst(IDGotoIfNot(cond_id, block_2_id));

julia> new_ir = CC.IRCode(bb_ir_copy)
  1 ─      nothing::Nothing
4 2 ┄ %2 = φ (#1 => 1, #3 => %8)::Int64
  │   %3 = φ (#1 => 0, #3 => %7)::Int64
  │   %4 = Base.slt_int(%3, _2)::Bool
  └──      goto #5 if not %4
2 3 ─ %6 = (Core.Intrinsics.mul_int)(%3, 2)::Int64
5 │   %7 = Base.add_int(%6, 1)::Int64
6 │   %8 = (Core.Intrinsics.add_int)(%2, %7)::Int64
2 │   %9 = (iseven)(%2)::Any
  └──      goto #2 if not %9
  4 ─      (println)(%2)::Any
  └──      goto #2
8 5 ─      return %2
```
Observe that in order to tie the conditional to the goto-if-not, we simply ensure that the `ID` associated to the instruction which computes the conditional appears in the `IDGotoIfNot` instruction.

### Run the new code

As ever, we can construct a `Core.OpaqueClosure` using `IRCode` in order to produce something runnable:
```jldoctest my_factorial
julia> oc = Core.OpaqueClosure(new_ir; do_compile=true)
(::Int64)::Int64->◌

julia> oc(1000)
2
12
58
248
1014
2037
```
Exactly what `oc` is computing is neither here nor there.
The point is that we've successfully inserted a new basic block into Julia's IR, and produced a callable from it.

## Summary

We have reviewed the two representations of Julia IR used in Mooncake.
Where possible, we always use `IRCode` -- as discussed, forwards-mode AD exclusively uses `IRCode`.
`BBCode` is basically only needed when undertaking transformations which involve changes to basic block structure -- the insertion of new basic blocks, and the modification of terminators in a way which changes the predecessors / successors of a given block being the primary sources of these kinds of changes.
Reverse-mode AD makes extensive use of such transformations, so `BBCode` is currently important there.

There are efforts such as [this PR](https://github.com/JuliaLang/julia/pull/45305) to augment `IRCode` with the capability to manipulate the CFG structure in a convenient manner.
Ideally these efforts will succeed, then we can do away with `BBCode`.

### Comparison with Alternative Approaches

It's worth noting that other automatic differentiation systems have taken different approaches to IR manipulation. For example, [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl) uses an "Optics" approach for IR transformations.

For readers interested in learning more about Julia's IR representation beyond what's covered here, the [Scientific Programming in Julia course materials](https://github.com/JuliaTeachingCTU/Scientific-Programming-in-Julia/blob/2023W/docs/src/lecture_09/ircode.md) provide additional valuable context.

## Docstrings

```@autodocs; canonical=true
Modules = [Mooncake.BasicBlockCode]
```
