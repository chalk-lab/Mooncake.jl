# IR Representations and Code Transformations

Mooncake works by transforming Julia's SSA-form Intermediate Representation (IR), so a good
working model of that IR is useful when touching the interpreter.

Please note that Julia's SSA-form IR changes slightly across minor versions, because it is
not a public language interface. The examples below are representative rather than
version-stable.

Before looking at the printed IR, keep three ideas in mind:

1. Each SSA statement produces one named value such as `%1` or `%2`.
2. Control flow is organized into basic blocks with branch or return terminators.
3. The compiler stores the statements and the control-flow graph separately, and Mooncake has
   to keep those two views coherent when transforming code.

## Julia's SSA-Form IR

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

The statements are associated to SSA names such as `%1` and `%2`. Each statement is
associated to a single SSA value, and uses of arguments are written as `_n`, where `_1` is
the function itself.

This IR is obtained after type inference and some optimisation passes, so each statement
already carries type information. In the example above, `%1` and `%2` are both known to be
`Float64`.

### Control Flow

Control flow is expressed via basic blocks and terminators:

```jldoctest bar
julia> function bar(x)
           if x > 0
               return x
           else
               return 5x
           end
       end
bar (generic function with 1 method)

julia> Base.code_ircode_by_type(Tuple{typeof(bar), Float64})[1][1]
2 1 ─ %1 = Base.lt_float(0.0, _2)::Bool
  │   %2 = Base.or_int(%1, false)::Bool
  └──      goto #3 if not %2
3 2 ─      return _2
5 3 ─ %5 = Base.mul_float(5.0, _2)::Float64
  └──      return %5
```

The corresponding control-flow graph is stored separately in the `cfg` field:

```jldoctest bar
julia> Base.code_ircode_by_type(Tuple{typeof(bar), Float64})[1][1].cfg
CFG with 3 blocks:
  bb 1 (stmts 1:3) → bb 3, 2
  bb 2 (stmt 4)
  bb 3 (stmts 5:6)
```

Each basic block is a straight-line region that ends either by falling through, branching,
or returning.

### Simple Loops and Phi Nodes

Loops introduce phi nodes:

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

For example,

```julia
%2 = φ (#1 => 1, #3 => %7)
```

means `%2` takes value `1` when control arrives from block `#1`, and the value of `%7`
when control arrives from block `#3`.

## Julia Compiler's IR Datastructure

The compiler represents inferred IR as `Core.Compiler.IRCode`. The statements live in the
`stmts` field, which is a `Core.Compiler.InstructionStream`. An `InstructionStream` is a
bundle of parallel vectors: the statement itself, its inferred type, call info, line data,
and flags.

For example:

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

The control-flow graph is stored separately in `ir.cfg`, and the argument types are stored
in `ir.argtypes`.

## Code Transformations

Mooncake uses two broad styles of transformation:

1. Straight-line edits on `IRCode`, especially in forward mode.
2. Reverse-mode assembly through a builder-local CFG in `reverse_mode.jl`, followed by a
   final lowering step back to coherent `IRCode`.

### Replacing Instructions in `IRCode`

Replacing one statement with another is straightforward:

```jldoctest my_factorial
julia> using Core: SSAValue

julia> const CC = Core.Compiler;

julia> new_ir = Core.Compiler.copy(ir);

julia> old_stmt = new_ir.stmts.stmt[7]
:(Base.mul_int(%2, %6))

julia> new_stmt = Expr(:call, Base.add_int, old_stmt.args[2:end]...)
:((Core.Intrinsics.add_int)(%2, %6))

julia> CC.setindex!(CC.getindex(new_ir, SSAValue(7)), new_stmt, :stmt);

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

This is the kind of local transformation that forward mode relies on heavily.

### Inserting New Instructions in `IRCode`

Insertion requires a little more care because later SSA names may need to shift. `IRCode`
handles this through `insert_node!` plus a later `compact!`:

```jldoctest my_factorial
julia> ni = CC.NewInstruction(Expr(:call, Base.mul_int, SSAValue(3), 2), Int)
Core.Compiler.NewInstruction(:((Core.Intrinsics.mul_int)(%3, 2)), Int64, Core.Compiler.NoCallInfo(), nothing, nothing)

julia> new_ssa = CC.insert_node!(new_ir, SSAValue(6), ni)
:(%10)

julia> stmt = CC.getindex(CC.getindex(new_ir, SSAValue(6)), :stmt)
:(Base.add_int(%3, 1))

julia> stmt.args[2] = new_ssa;

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

This is the right tool when the transformation stays local to existing basic blocks.

## Reverse-Mode CFG Assembly

Reverse mode needs more than local SSA insertion. It frequently has to:

1. create fresh blocks,
2. thread predecessor-sensitive phi handling,
3. insert reverse-only control flow, and
4. preserve a coherent CFG while doing so.

Mooncake now handles that in [`src/interpreter/reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl)
using a builder-local `CFGBlock` representation. The reverse transform first translates the
normalized primal `IRCode` into CFG blocks with stable internal `ID`s, assembles the
forwards and pullback control flow in that representation, and finally lowers the result
back to `IRCode`.

That split is deliberate:

1. `IRCode` remains the source of truth at the compiler boundary.
2. The builder provides a convenient place to manipulate reverse-mode control flow.
3. The final lowering step re-establishes standard compiler IR with a coherent CFG.

This page is mainly about the representations themselves. For the full reverse-mode pipeline,
including statement translation, control-flow replay, and forward-to-reverse communication, see
[`reverse_mode_design.md`](reverse_mode_design.md).

## Summary

`IRCode` is the main compiler-facing representation throughout Mooncake.
Forward mode mostly performs local statement rewrites on that representation.
Reverse mode still starts from normalized `IRCode`, but assembles its extra control flow in
a builder-local CFG before lowering back to `IRCode`.

If you are modifying interpreter internals, the most important invariants to preserve are:

1. SSA uses must stay coherent after insertion and compaction.
2. `ir.stmts` and `ir.cfg` must agree after lowering.
3. Phi-node edges and predecessor relationships must stay aligned when blocks are removed or reordered.
