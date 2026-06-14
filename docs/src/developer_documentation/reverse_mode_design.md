# Reverse-Mode Design

Last checked: 04/04/2026, Julia v1.10 / v1.11 / v1.12.

This page gives a high-level map of how Mooncake's reverse-mode transform is structured.
It is aimed at readers who want to understand the main ideas before reading the implementation.

## High-Level Transform: `IRCode` to `IRCode`

The reverse-mode pipeline is easiest to understand as a four-step transform:

1. Start from normalized primal `IRCode`.
2. Convert that `IRCode` into Mooncake's builder-local CFG representation.
3. Assemble new forward and reverse CFGs.
4. Lower each assembled CFG back to compiler `IRCode`.

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

## A Worked Mini-Example

Here is the smallest useful mental model for the whole reverse-mode pipeline. Consider a primal
function with one active call followed by a return:

```julia
function f(x)
    y = sin(x)
    return y
end
```

Very roughly, the normalized primal `IRCode` looks like:

```text
bb1:
    %1 = sin(_2)
    return %1
```

### Step 1: statement translation

`make_ad_stmts!` translates the call and the return into forward and reverse fragments.

For the call, the important effect is:

```text
forward:
    %rule_result = rule_for_sin(x_arg)
    %pb = getfield(%rule_result, 2)
    %1 = getfield(%rule_result, 1)

reverse:
    %d1 = rdata_ref_for_%1[]
    rdata_ref_for_%1[] = zero(...)
    %dx = %pb(%d1)
    increment_ref!(rdata_ref_for_x, getfield(%dx, 1))
```

For the return, the important effect is:

```text
forward:
    return %1

reverse:
    increment_ref!(rdata_ref_for_%1, dy)
```

### Step 2: forward CFG assembly

`forwards_pass_ir` wraps those fragments in a forward CFG with an extra entry block.

Conceptually:

```text
fwd_entry:
    load shared captures
    initialize lazy zero-rdata state
    goto fwd_bb1

fwd_bb1:
    %rule_result = ...
    %pb = ...
    %1 = ...
    push!(comms_stack, tuple(%pb))
    return %1
```

The forward closure therefore does two things at once:

1. computes the primal/codual result
2. stores the information that the pullback will need later

### Step 3: pullback CFG assembly

`pullback_ir` builds a separate pullback CFG:

```text
rvs_entry:
    load shared captures
    create reverse-data refs
    goto rvs_bb1

rvs_bb1:
    %pb = getfield(pop!(comms_stack), 1)
    increment_ref!(rdata_ref_for_%1, dy)
    %d1 = rdata_ref_for_%1[]
    rdata_ref_for_%1[] = zero(...)
    %dx = %pb(%d1)
    increment_ref!(rdata_ref_for_x, getfield(%dx, 1))
    goto rvs_exit

rvs_exit:
    read argument rdata refs
    instantiate any lazy zero-rdata placeholders
    return argument cotangent tuple
```

This is the essential pattern repeated across larger examples: the pullback is a separate CFG
that consumes stored forward-pass data and sends cotangents backward through the primal
dependency structure.

### Step 4: lower back to compiler IR

Finally, `lower_cfg_blocks_to_ir` converts both CFGs back to ordinary compiler `IRCode`.

So even in this tiny example the real flow is:

```text
primal IRCode
  -> translated statement fragments
  -> forward CFG / pullback CFG
  -> forward IRCode / pullback IRCode
```

These examples use schematic names such as `x_arg` rather than exact lowered argument slots.
In the real generated IR, argument indices can shift because the generated closures carry extra
state in addition to the primal arguments.

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

## Data Structures That Matter Most

Two data structures carry most of the transform's state:

- `ADInfo`: global state shared across the whole derivation
- `ADStmtInfo`: the per-statement translation result

### `ADInfo`

When reading the implementation, the most important `ADInfo` fields are:

- `shared_data_pairs`: the values captured by both generated closures
- `block_stack_id` and `block_stack`: the control-flow replay channel
- `arg_rdata_ref_ids` and `ssa_rdata_ref_ids`: where reverse data is accumulated
- `ssa_insts` and `arg_types`: the primal type information used during translation
- `lazy_zero_rdata_ref_id`: the placeholder-zero mechanism used at pullback exit

The remaining fields mostly support those jobs rather than introducing separate ideas.

### `ADStmtInfo`

`ADStmtInfo` is simpler. Its central fields are:

- `fwds`: forward-pass instructions for the primal statement
- `rvs`: reverse-pass instructions for that statement
- `comms_id`: the optional value that must survive from forward execution to pullback execution

If you understand those three fields, you understand the role of `ADStmtInfo`.

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
3 ┄ %6 = φ (#1 => arg_x, #2 => %5)
```

On the forward pass, this is mostly structural. The phi node is rebuilt so that its incoming
values are coduals rather than raw primal values:

```julia
3 ┄ %6 = φ (#1 => codual_arg_x, #2 => %5)
```

Schematically, the incoming primal argument is replaced by the corresponding codual argument in
the generated forward closure. In the real lowered IR, the exact argument slot can shift because
the generated closure carries extra leading state.

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
increment_ref!(r_arg_x, %d6)
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
    %6 = φ (#2 => %5, #3 => arg_x)
    return %6
```

The forward pass computes `%6` in the usual SSA sense. The reverse pass has to do two things
at the join:

1. determine whether control came from `bb2` or `bb3`
2. send the cotangent of `%6` back to `%5` or `arg_x` accordingly

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
    increment_ref!(r_arg_x, d6)
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

## Entry and Exit Blocks

Both generated closures contain extra structural blocks that do not correspond directly to a
single primal block.

### Forward entry block

The forward entry block is responsible for:

- loading shared captured data
- initializing lazy zero-rdata placeholders for arguments
- optionally logging the synthetic entry block for later reverse dispatch

### Pullback entry block

The pullback entry block is responsible for:

- loading the same shared captured data
- creating reverse-data references for arguments and SSA values
- dispatching to the reverse block associated with the primal block that actually returned

### Pullback exit block

The pullback exit block is responsible for:

- reading argument reverse-data references
- materializing true zeros from lazy zero-rdata placeholders where needed
- packaging the final argument cotangent tuple
- returning that tuple with the expected type

These blocks are worth calling out explicitly because they explain why the generated forward and
reverse CFGs do not look like simple block-for-block reversals of the primal CFG.

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

## Concept-to-Helper Map

If you want to jump from the conceptual description in this page to the implementation, these
are the main landmarks:

- statement translation: `make_ad_stmts!`
- shared captured data: `SharedDataPairs`, `shared_data_tuple`, `shared_data_stmts`
- per-block forward-to-reverse values: `create_comms_insts!`
- forward CFG assembly: `forwards_pass_ir`
- pullback CFG assembly: `pullback_ir`
- control-flow replay: `__push_blk_stack!`, `__pop_blk_stack!`, `make_switch_stmts`
- phi-edge reverse routing: `conclude_rvs_block`, `rvs_phi_block`
- local CFG representation: `CFGBlock`
- lowering back to compiler IR: `lower_cfg_blocks_to_ir`

## Where This Lives in Code

If you want to connect the conceptual story above to the implementation, the main entry points
are:

```julia
build_rrule(interp::MooncakeInterpreter{C}, sig_or_mi; debug_mode=false)

generate_ir(
    interp::MooncakeInterpreter, sig_or_mi; debug_mode=false, do_inline=true
)
```

Here `sig_or_mi` is either a signature such as `Tuple{typeof(foo), Float64}` or a
`Core.MethodInstance`.

If the signature has a custom rule ([`Mooncake.is_primitive`](@ref) returns `true`), Mooncake
uses that rule. Otherwise it looks up the primal IR and differentiates it.

[`lookup_ir`](@ref Mooncake.lookup_ir) calls `Core.Compiler.typeinf_ircode` on a method
instance, which is a lower-level version of `Base.code_ircode`.

The transform works on `Core.Compiler.IRCode`, not the `CodeInfo` shown by `@code_typed`.
[`normalise!`](@ref Mooncake.normalise!) rewrites some `IRCode` expressions into forms that are
easier for the AD transform to handle, after which reverse mode assembles through the local CFG
builder in
[`reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl)
and lowers back to `IRCode`.

## Captures and Closure Construction

The generated forward and reverse `IRCode`s do not communicate by calling each other directly.
They communicate partly through ordinary runtime values and partly through a shared captures
tuple that is embedded into the generated closures.

### What gets captured

The shared captures tuple contains the values recorded in `SharedDataPairs`, such as:

- static rule objects that are not safe or convenient to inline directly
- stacks used for per-block communication
- the block stack used for control-flow replay
- the lazy-zero-rdata reference used to finish the pullback result

`generate_ir` builds this tuple with:

```julia
shared_data = shared_data_tuple(info.shared_data_pairs)
```

and both generated closures are later constructed with that same tuple:

```julia
fwd_oc = misty_closure(dri.fwd_ret_type, dri.fwd_ir, dri.shared_data...)
rvs_oc = misty_closure(dri.rvs_ret_type, dri.rvs_ir, dri.shared_data...)
```

Both generated closures therefore receive access to the same logical captured data. At the
start of each closure, `shared_data_stmts` lowers that tuple back into local IDs.

So the resulting `IRCode` does not contain the captured values directly as ordinary SSA
definitions. Instead, it contains loads from the closure's captures field near the entry block.

### MWE: one captured tuple shared by both closures

Suppose `SharedDataPairs` conceptually contains:

```text
[(id_rule, some_rule), (id_blk, block_stack), (id_zero, lazy_zero_ref)]
```

Then:

```text
shared_data_tuple(...) == (some_rule, block_stack, lazy_zero_ref)
```

and the entry blocks in the lowered `IRCode`s begin by reconstructing those bindings:

```text
fwd_entry:
    id_rule = getfield(_1, 1)
    id_blk  = getfield(_1, 2)
    id_zero = getfield(_1, 3)
    ...

rvs_entry:
    id_rule = getfield(_1, 1)
    id_blk  = getfield(_1, 2)
    id_zero = getfield(_1, 3)
    ...
```

Here `_1` is the generated closure object. The `getfield` operations shown above are the loads
that recover values from that closure's captured state. The important point is that the two
generated closures do not each get separate logical bindings. They are built against the same
captured state assembled during derivation.

### How forward-to-reverse sharing appears in the final IR

After lowering, the final forward `IRCode` still contains the machinery that writes dynamic data
needed by the pullback:

- pushes onto comms stacks for values such as pullback objects
- pushes onto the block stack when control-flow replay is needed
- initialization of lazy zero-rdata placeholders

The final pullback `IRCode` contains the matching reads:

- pops from comms stacks
- pops from the block stack
- loads from reverse-data references
- loads from the lazy-zero-rdata capture when finishing the returned cotangent tuple

So although the builder-local CFG disappears after lowering, the final `IRCode` still encodes
the forward-to-reverse contract explicitly through stack operations, ref operations, and capture
loads.

### MWE: dynamic value sharing in final `IRCode`

If an active call produces a pullback object `%pb`, the final forward `IRCode` contains code
equivalent to:

```text
%tuple = tuple(%pb)
push!(comms_stack, %tuple)
```

and the final pullback `IRCode` contains the matching restore:

```text
%tuple = pop!(comms_stack)
%pb = getfield(%tuple, 1)
```

That pair of writes and reads is how a value produced only during the forward run becomes
available later in the pullback.

### How the closures are built

`generate_ir` produces two compiler `IRCode`s:

- one for the forward closure
- one for the pullback closure

`build_derived_rrule` then packages those into closure objects. At a high level:

1. `generate_ir` returns `DerivedRuleInfo`, containing `fwd_ir`, `rvs_ir`, and `shared_data`
2. `build_derived_rrule` turns each `IRCode` plus `shared_data` into a `MistyClosure`
3. those closures are placed into a `DerivedRule`
4. when the derived rule is called, it returns the forward result plus a `Pullback` wrapper
   around the reverse closure

The end result is that users do not interact with raw `IRCode` directly. They call a derived
rule, which runs the generated forward closure and receives a callable pullback object whose
captures point at the same shared state prepared during derivation.

### MWE: final wrapper structure

Conceptually, the final derived rule looks like:

```text
DerivedRule(
    fwds_oc = MistyClosure(fwd_ir, shared_data),
    pb_oc_ref = Ref(MistyClosure(rvs_ir, shared_data)),
    ...
)
```

Calling that derived rule:

1. runs `fwds_oc`
2. returns the forward `CoDual`
3. returns a `Pullback` object that knows how to call `pb_oc_ref[]`

So the forward closure and pullback closure are separate pieces of generated code, but they are
stitched together by the shared captures tuple and the `DerivedRule` / `Pullback` wrappers.

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

## Known Boundaries

Beyond unsupported `PhiCNode` and `UpsilonNode` handling, a few broader boundaries are useful to
keep in mind when reading or extending reverse mode:

- many operations are only as good as the primitive or derived rules available for them
- debug mode wraps rule calls and changes the generated code shape slightly for diagnostics
- the transform depends on compiler IR conventions that can shift across Julia minor versions
- reverse mode assumes ordinary SSA/control-flow lowering patterns, not the full exception-state
  machinery generated for every possible language feature

## Further Reading

If you want the supporting background after this page:

- [`ir_representation.md`](ir_representation.md) explains the compiler-facing `IRCode` representation.
- [`forwards_mode_design.md`](forwards_mode_design.md) covers the forward-mode side.
- [`src/interpreter/reverse_mode.jl`](https://github.com/chalk-lab/Mooncake.jl/blob/main/src/interpreter/reverse_mode.jl) is the implementation described here.
