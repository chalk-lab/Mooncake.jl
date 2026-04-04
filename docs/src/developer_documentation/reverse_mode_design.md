# Reverse-Mode Design

## Compilation process

Last checked: 04/04/2026, Julia v1.10 / v1.11 / v1.12.

This page gives a high-level map of how Mooncake's reverse-mode transform is structured.
It is aimed at readers who want to understand the main ideas before reading the implementation.

Rule building is done statically, based on types. Some methods accept values, e.g.
```julia
build_rrule(args...; debug_mode=false)
```
but these simply extract the types of all the arguments and call the main method (non Helper) for [`build_rrule`](@ref Mooncake.build_rrule).

The action happens in [`reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl), in particular the following method:
```julia
build_rrule(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false)
```
`sig_or_mi` is either a signature, such as `Tuple{typeof(foo), Float64}`, or a `Core.MethodInstance`.
Signatures are extracted from `Core.MethodInstance`s as necessary.

If a signature has a custom rule ([`Mooncake.is_primitive`](@ref) returns `true`), we take it, otherwise we generate the IR and differentiate it.

The forward and reverse pass IRs are created by the [`generate_ir`](@ref Mooncake.generate_ir) method.
The `OpaqueClosure` allows going back from the IR to a callable object. More precisely we use `MistyClosure` to store the associated IR.

The `Pullback` and `DerivedRule` structs are convenience wrappers for `MistyClosure`s with some bookkeeping.

Diving one level deeper, in the following method:

```julia
generate_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
```

The function [`lookup_ir`](@ref Mooncake.lookup_ir) calls `Core.Compiler.typeinf_ircode` on a method instance, which is a lower-level version of `Base.code_ircode`.

The IR considered is of type `Core.Compiler.IRCode`, which is different from the `CodeInfo` returned by `@code_typed`.
This format is obtained from `CodeInfo`, used to perform most optimizations in the Julia IR in the [evaluation pipeline](https://docs.julialang.org/en/v1/devdocs/eval/), then converted back to `CodeInfo`.

The function [`normalise!`](@ref Mooncake.normalise!) is a custom pass to modify `IRCode` and make some expressions nicer to work with.
The possible expressions one can encounter in lowered ASTs are documented [here](https://docs.julialang.org/en/v1/devdocs/ast/#Lowered-form).

Reverse-mode specific stuff: return type retrieval, `ADInfo`, the inline CFG builder in `reverse_mode.jl`, and `zero_like_rdata.jl`. Reverse mode now assembles through a builder-local CFG and lowers directly back to `IRCode`.

## High-Level Transform: `IRCode` to `IRCode`

The reverse-mode pipeline is easiest to understand as a four-step transform:

1. Start from normalized primal `IRCode`.
2. Convert that `IRCode` into Mooncake's builder-local CFG representation.
3. Assemble new forward and reverse CFGs, then lower each one back to compiler `IRCode`.

So reverse mode is no longer "edit compiler IR in place until it works". The compiler `IRCode`
is the input and output format, but most of the reverse-mode assembly happens in the middle on
`CFGBlock`s with Mooncake `ID`s.

### 1. Normalize and reinterpret the primal `IRCode`

[`generate_ir`](@ref Mooncake.generate_ir) begins by looking up the inferred primal `IRCode`,
running [`normalise!`](@ref Mooncake.normalise!), and converting the result into CFG blocks.
At that point the source of truth is still the primal `IRCode`, but reverse mode stops working
directly with compiler block numbers and SSA names.

The local CFG layer exists because reverse mode needs to introduce new control flow, not just
rewrite statements. In particular, the pullback has to reconstruct which predecessor edge was
taken through the primal CFG, including phi-sensitive cases.

### 2. Translate each primal statement into AD fragments

Each primal statement is translated by [`make_ad_stmts!`](@ref Mooncake.make_ad_stmts!) into an
`ADStmtInfo`. This is a small per-statement plan containing:

- forward-pass instructions
- reverse-pass instructions
- an optional communication value that must survive from the forward run to the pullback

`ADInfo` stores the global transform state used while assembling those
fragments: argument and SSA type tables, reverse-data references, shared-data bookkeeping,
block-stack information for control-flow reconstruction, and debug-mode configuration.

The important design point is that statement translation does not directly build the final
pullback `IRCode`. It produces forward and reverse fragments first, and whole CFGs are assembled
afterward.

### 3. Assemble forward and reverse CFGs

`forwards_pass_ir` builds the forward closure CFG. This CFG:

- loads shared captured state
- initializes lazy zero-rdata bookkeeping
- emits the translated forward statements for each primal block
- pushes communication values needed later by the pullback
- records block-stack information when the reverse pass must recover dynamic control flow

`pullback_ir` then builds the pullback CFG separately. This CFG:

- loads shared data and reverse-data references
- dispatches to the block that actually exited during the forward run
- walks primal statements in reverse order
- reconstructs predecessor-sensitive control flow
- handles phi nodes by routing through edge-specific reverse blocks when needed
- materializes the final cotangent tuple returned by the pullback closure

This is why the pullback is more than "run the statements backwards". It also has to replay the
control-flow structure needed to send cotangents back along the correct incoming edges.

### 4. Lower the builder CFG back to compiler `IRCode`

Once the forward and reverse CFGs are assembled, `lower_cfg_blocks_to_ir` turns them back into
coherent compiler `IRCode`. This step
handles the mechanical reconstruction work in one place:

- canonicalizing the local CFG
- pruning unreachable blocks
- lowering switch-style control flow into compiler-compatible terminators
- rebuilding SSA numbering and block numbering
- constructing a fresh `Core.Compiler.IRCode`

After that, [`generate_ir`](@ref Mooncake.generate_ir) can run the usual optimization pass on
both results and wrap them into opaque closures for the final derived reverse rule.

In short, the reverse-mode transform is:

```text
primal IRCode
  -> normalized primal IRCode
  -> builder-local CFG
  -> forward CFG + pullback CFG
  -> forward IRCode + pullback IRCode
```

That middle CFG stage is what keeps the reverse-mode implementation manageable: it isolates the
hard control-flow surgery from the compiler's concrete `IRCode` datastructure, and only lowers
back to compiler IR once the new program structure is complete.

## `CFGBlock`: The Reverse-Mode Working IR

`CFGBlock` is Mooncake's reverse-mode-local basic-block representation. It is the format used
while assembling the forward closure CFG and the pullback CFG inside
[`reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl).

At a high level, a `CFGBlock` is just:

- a stable internal block `ID`
- a vector of `(ID, NewInstruction)` pairs for the statements in that block

Those IDs are Mooncake-local IDs, not compiler SSA numbers or compiler block numbers. That is
deliberate: reverse mode needs a representation that can survive block insertion, block
splitting, and control-flow rewrites without constantly renumbering compiler SSA values.

### What `CFGBlock` is used for

The reverse transform uses `CFGBlock`s to do the hard structural work before lowering back to
compiler IR. In particular, the builder-local CFG is where Mooncake:

- translates primal SSA statements into forward and reverse fragments
- inserts extra entry and exit blocks
- groups and restores communication values for the pullback
- rewrites control flow into predecessor-sensitive reverse dispatch
- creates edge-specific reverse blocks for phi handling
- prunes unreachable blocks and canonicalizes the resulting CFG

So `CFGBlock` is not an alternative public IR for Mooncake as a whole. It is a local assembly
format used to construct reverse-mode programs safely before converting them back into
`Core.Compiler.IRCode`.

### Why `IRCode` is not sufficient as the working format

`IRCode` is sufficient as the source and target format. It is not a good format for the middle
of reverse-mode assembly.

The core problem is that reverse mode does more than local statement replacement. It often has
to:

- create fresh blocks
- insert control-flow that did not exist in the primal
- split one logical reverse step across several blocks
- route cotangents along predecessor-specific edges
- keep phi handling consistent with those edges
- rebuild SSA numbering and CFG numbering coherently at the end

Compiler `IRCode` stores statements and CFG structure in tightly related forms. If you edit it
mid-assembly, you have to keep the instruction stream, block numbering, terminators, phi edges,
and CFG metadata coherent at every intermediate step. That is possible for local edits, but it
becomes brittle once reverse mode starts inserting whole blocks and rethreading control flow.

`CFGBlock` avoids that problem by giving reverse mode a looser construction format:

- block identity is stable while the transform is running
- instructions can be inserted without immediate SSA renumbering
- predecessor and successor rewrites can happen before final lowering
- phi-edge handling can be expressed directly in terms of predecessor block IDs

Only once the new forward and reverse CFGs are structurally complete does Mooncake lower them
back to `IRCode`, rebuild block/SSA numbering, and hand the result back to the compiler.

### The right way to think about it

`IRCode` is the compiler-facing representation.
`CFGBlock` is the reverse-mode assembly representation.

Mooncake still begins with normalized primal `IRCode` and ends with compiler `IRCode`, but it
does the difficult control-flow surgery in `CFGBlock` form because that is the point where the
transform needs flexibility more than compiler-format exactness.

## Statement-Level MWEs

The reverse-mode transform is easiest to follow if you read `make_ad_stmts!` as a translator
from one primal SSA statement into:

- the forward statements that execute the primal computation and save anything the pullback
  will need later
- the reverse statements that consume cotangents and propagate them to arguments and earlier
  SSA values

These are only sketches, but they match the current implementation strategy closely.
They use pseudocode-style helper names such as `rule_for_sin`, `increment_ref!`, and
`switch_to_reverse_phi_edge(...)` to show the dataflow. They are not literal emitted APIs or
exact compiler IR.

### MWE 1: Constant literal

Primal statement:

```julia
%5 = 3.0
```

Forward fragment:

```julia
%5 = uninit_fcodual(3.0)
```

Reverse fragment:

```julia
nothing
```

Constants produce a `CoDual` on the forward pass, but there is nothing to propagate on the
reverse pass because Mooncake does not track cotangents for literals.

### MWE 2: Active branch condition

Primal statement:

```julia
goto #3 if not %2
```

If `%2` is active, then `%2` is a `CoDual` in the forward IR, so the condition must branch on
its primal:

Forward fragment:

```julia
%cond = primal(%2)
goto #3 if not %cond
```

Reverse fragment:

```julia
nothing
```

The branch itself carries no cotangent information. Its effect on the reverse pass is indirect:
the forward pass records enough control-flow information for the pullback to recover which edge
was taken.

### MWE 3: Active call

Primal statement:

```julia
%7 = sin(%4)
```

Very roughly, the forward fragment becomes:

```julia
%rule_result = rule_for_sin(%4)
%pb = getfield(%rule_result, 2)
%raw_y = getfield(%rule_result, 1)
%7 = typeassert(%raw_y, CoDual{Float64, ...})
```

and the reverse fragment becomes:

```julia
%dy = rdata_ref_for_%7[]
rdata_ref_for_%7[] = zero_like_rdata_from_type(typeof(%7))
%dx_tuple = %pb(%dy)
increment_ref!(rdata_ref_for_%4, getfield(%dx_tuple, 1))
```

The exact generated code is more explicit than this because Mooncake writes out the `getfield`,
`setfield!`, and `increment!!` operations directly. That keeps the reverse-data references
visible to the optimizer.

The important point is that the forward fragment computes both the output codual and a pullback
object, and the reverse fragment later consumes the stored pullback plus the output cotangent.

### MWE 4: Return of an active SSA value

Primal statement:

```julia
return %9
```

Forward fragment:

```julia
%checked = typeassert(%9, fwd_ret_type)
return %checked
```

Reverse fragment:

```julia
increment_ref!(rdata_ref_for_%9, dy)
```

The forward closure returns the codual value. The pullback closure receives the cotangent of
that return value and accumulates it into the reverse-data reference associated with `%9`.

### MWE 5: Phi nodes

Consider a primal block that starts with a phi node:

```julia
3 ┄ %6 = φ (#1 => _2, #2 => %5)
```

On the forward pass, this is mostly structural. The phi node is rebuilt so that its incoming
values are coduals rather than raw primal values:

```julia
3 ┄ %6 = φ (#1 => _3, #2 => %5)
```

The argument changes from `_2` to `_3` because the generated forward closure has an extra
leading shared-data argument, so the primal arguments are shifted by one position.

The important work happens on the reverse side, not at the phi statement itself. Suppose the
cotangent for `%6` is stored in `r%6`. When the pullback reaches the reverse counterpart of
this block, it first extracts and zeros the cotangent for `%6`, then dispatches to an
edge-specific reverse block:

```julia
%d6 = r%6[]
r%6[] = zero_like_rdata_from_type(typeof(%6))
switch_to_reverse_phi_edge(...)
```

If the primal arrived from block `#1`, the reverse phi-edge block behaves roughly like:

```julia
increment_ref!(r_2, %d6)
goto reverse_block_for_#1
```

If the primal arrived from block `#2`, it behaves roughly like:

```julia
increment_ref!(r%5, %d6)
goto reverse_block_for_#2
```

So the key point is that phi handling is split across two places:

- the forward pass rebuilds the phi with codual inputs
- the reverse pass routes cotangents back through predecessor-specific phi-edge blocks

This is why the pullback needs predecessor reconstruction, rather than just local statement
reversal.

## How Control Flow Is Handled

Reverse mode cannot treat control flow as a local statement rewrite problem. The forward pass
must preserve enough information for the pullback to know:

- which block actually executed next
- which predecessor edge reached a join block
- which reverse block should run after the current one finishes

That is why reverse mode assembles whole CFGs and uses the block stack when the predecessor is
not statically determined.

### MWE 1: Straight-line fallthrough

Suppose the primal CFG is just:

```text
bb1 -> bb2 -> return
```

and `bb2` has only one predecessor, namely `bb1`.

In this case there is no ambiguity. The forward pass does not need to log any predecessor ID
for `bb2`, and the reverse pass can jump directly from the reverse counterpart of `bb2` to the
reverse counterpart of `bb1`.

Very roughly:

```text
forward:   fwd_bb1 ; fwd_bb2 ; return
reverse:   rvs_bb2 ; goto rvs_bb1
```

This is the cheap path: no block-stack push, no block-stack pop, no predecessor switch.

### MWE 2: Simple conditional branch

Suppose the primal CFG is:

```text
bb1:
    if cond goto bb2 else bb3

bb2:
    ...
    goto bb4

bb3:
    ...
    goto bb4

bb4:
    return y
```

On the forward pass, execution may reach `bb4` from either `bb2` or `bb3`. That predecessor is
not statically known, so the forward pass records enough control-flow information to recover it
later.

Conceptually the forward pass does something like:

```text
fwd_bb1:
    branch on primal(cond)

fwd_bb2:
    push!(block_stack, bb2)
    ...
    goto fwd_bb4

fwd_bb3:
    push!(block_stack, bb3)
    ...
    goto fwd_bb4
```

Then, once the pullback reaches the reverse counterpart of `bb4`, it uses `make_switch_stmts`
to recover which predecessor actually led there:

```text
rvs_bb4:
    prev = pop!(block_stack)
    if prev == bb2
        goto rvs_phi_or_pred_for_bb2
    else
        goto rvs_phi_or_pred_for_bb3
```

So the pullback does not "re-run the branch condition". It replays the realized control-flow
path from the forward run.

### MWE 3: Join block with phi node

Now combine branching with a phi node:

```julia
bb2:
    %5 = ...
    goto bb4

bb3:
    goto bb4

bb4:
    %6 = φ (#2 => %5, #3 => _2)
    return %6
```

The forward pass computes `%6` in the usual SSA sense. The reverse pass has to do two things
at the join:

1. determine whether control came from `bb2` or `bb3`
2. send the cotangent of `%6` back to `%5` or `_2` accordingly

So the reverse CFG for `bb4` behaves roughly like:

```text
rvs_bb4:
    d6 = r%6[]
    r%6[] = zero(...)
    prev = pop!(block_stack)
    if prev == bb2
        goto rvs_phi_edge_bb2
    else
        goto rvs_phi_edge_bb3

rvs_phi_edge_bb2:
    increment_ref!(r%5, d6)
    goto rvs_bb2

rvs_phi_edge_bb3:
    increment_ref!(r_2, d6)
    goto rvs_bb3
```

This is the precise place where control-flow replay and phi handling meet.

### MWE 4: Unique-predecessor optimization

Mooncake does not always push block IDs. If a block's predecessor is uniquely determined, the
reverse pass can hard-code that predecessor instead of reading the block stack.

For example:

```text
bb1 -> bb2
bb2 -> bb3
bb3 -> return
```

If `bb3` can only ever be reached from `bb2`, then the reverse pass for `bb3` can go straight
to `bb2`. No stack traffic is needed for that edge.

This optimization matters because dynamic control-flow logging is only needed where the forward
execution path loses information that the pullback must recover later.

## Forward-to-Reverse Communication

Forward and reverse code do not communicate through a single channel. The current design uses
three distinct mechanisms, each for a different kind of information.

### 1. Shared captured data

Some values are known statically when the derived rule is built and are needed by both the
forward closure and the pullback closure. These go through `SharedDataPairs`.

Examples include:

- rule objects that are not singleton values
- constant coduals that are not safe to interpolate directly into IR
- the per-transform block stack object
- the lazy-zero-rdata reference used to materialize correctly typed zeros at pullback exit

Both generated closures receive the same captures tuple. At the start of each closure,
`shared_data_stmts` extracts those captures back into local IDs.

This is the "static shared state" channel: build-time data, not per-execution data.

### 2. Per-block communication stacks

Some values are only known during the forward execution and must be handed to the pullback for
the matching block. This is what the `comms_id` field of `ADStmtInfo` is for.

For a statement like an active call, the typical communicated value is the pullback object:

```julia
%rule_result = rule(...)
%pb = getfield(%rule_result, 2)
```

`%pb` is marked as the statement's `comms_id`. Later, `create_comms_insts!` groups the
`comms_id`s for a primal block, builds a tuple of them on the forward pass, and pushes that
tuple onto a block-local stack:

```julia
%tuple = tuple(%pb1, %pb2, ...)
push!(comms_stack, %tuple)
```

At the start of the corresponding reverse block, the pullback pops that tuple and restores the
saved values to the same IDs:

```julia
%tuple = pop!(comms_stack)
%pb1 = getfield(%tuple, 1)
%pb2 = getfield(%tuple, 2)
```

This is the "dynamic value" channel: values produced during the forward execution and consumed
later by the reverse execution.

### 3. The block stack for dynamic control flow

The pullback has to know which predecessor edge the primal actually took. When that predecessor
is not statically determined, the forward pass records block IDs on a dedicated `BlockStack`.

On the forward pass, selected blocks push their ID:

```julia
push!(block_stack, current_block_id)
```

On the reverse pass, `make_switch_stmts` pops the predecessor ID and dispatches accordingly:

```julia
%prev = pop!(block_stack)
switch(%prev == pred1 ? ..., %prev == pred2 ? ..., ...)
```

This is what allows the pullback to send cotangents along the correct incoming edge and to
handle phi nodes via predecessor-specific reverse blocks.

In short, the three communication mechanisms are:

1. captures shared by both closures for static build-time data
2. per-block comms stacks for dynamic forward values such as pullback objects
3. the block stack for replaying dynamic control flow on the reverse pass

## SSA Nodes Not Covered

Most common SSA-form node kinds are handled by `make_ad_stmts!`, but a few are explicitly out
of scope.

### `PhiCNode`

`PhiCNode`s are not currently supported. If reverse mode encounters one, it throws an
`unhandled_feature` error immediately.

These nodes are associated with exception-flow joins rather than ordinary CFG joins, so they
need more than the current phi-node machinery.

### `UpsilonNode`

`UpsilonNode`s are also not currently supported. Encountering one raises an `unhandled_feature`
error with guidance to avoid the construct or write a manual rule.

In practice these nodes arise from some `try` / `catch` / `finally` lowering patterns. The
current reverse-mode transform does not attempt to differentiate through that exception-state
machinery.

### Practical boundary

Ordinary `PhiNode`s are supported and lowered through predecessor-sensitive reverse CFG logic.
`PhiCNode`s and `UpsilonNode`s are not. So "control flow with standard SSA joins" is in scope,
while "exception SSA machinery" is still out of scope.

## Further Reading

If you want the supporting background after this page:

- [`ir_representation.md`](ir_representation.md) explains the compiler-facing `IRCode` representation.
- [`forwards_mode_design.md`](forwards_mode_design.md) covers the forward-mode side.
- [`src/interpreter/reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl) is the implementation described here.
