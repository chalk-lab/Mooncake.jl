---
name: mooncake-reference
description: >
  Explain Mooncake.jl internals, AD pipeline, IR transformations, tangent types,
  and rule system. Triggers on conceptual questions about how Mooncake works.
  Uses the ir-inspect skill to demonstrate concepts with real code examples.
---

# Mooncake.jl Reference & Teaching Guide

You are helping the user understand Mooncake.jl's internals. This skill combines conceptual explanations with live demonstrations.

## How to Use This Skill

1. Identify which topic the user is asking about (use the Topic Map below)
2. Give a concise conceptual explanation using the relevant Reference Section
3. Read the referenced doc/source files for deeper detail when needed
4. **Proactively offer to demonstrate the concept with a live example**:
   - Load the ir-inspect script and run relevant commands
   - Use simple functions (sin, cos, x->x*x, etc.) as teaching examples
   - Walk the user through the output, connecting it back to the concept
5. Let the user drive deeper — offer next topics they might want to explore

## Demonstration Recipes

When explaining a concept, offer to show it live. Load ir-inspect first:

```julia
using Revise
includet(".claude/skills/ir-inspect/scripts/ir_inspect.jl")
```

| Concept | Demo command | What to highlight |
|---------|-------------|-------------------|
| Julia IR / SSA form | `show_stage(inspect_ir(sin, 1.0), :raw)` | SSAValues (%n), Arguments (_n), :invoke |
| Normalization | `show_diff(inspect_ir(x->ccall(:jl_gc_collect, Cvoid, ()), nothing); from=:raw, to=:normalized)` | foreigncall→_foreigncall_, intrinsics→wrappers |
| BBCode vs IRCode | `show_stage(inspect_ir(sin, 1.0), :bbcode)` | ID-based naming, BBlock structure |
| Forward pass IR | `show_stage(inspect_ir(sin, 1.0), :fwd_ir)` | CoDual wrapping, rule calls, pullback capture |
| Reverse pass IR | `show_stage(inspect_ir(sin, 1.0), :rvs_ir)` | Switch dispatch, reverse order, rdata propagation |
| Forward mode (Dual) | `show_stage(inspect_ir(sin, 1.0; mode=:forward), :dual_ir)` | Dual types, frule!! calls |
| Full pipeline | `show_ir(inspect_ir(sin, 1.0))` | All stages, how each transforms the previous |
| Optimization | `show_diff(inspect_ir(sin, 1.0); from=:fwd_ir, to=:optimized_fwd)` | Inlining, SROA, dead code elimination |
| Tangent types | REPL: `Mooncake.tangent_type(Float64)`, `Mooncake.tangent_type(Vector{Float64})` | What types map to what tangents |
| FData/RData split | REPL: `Mooncake.fdata_type(...)`, `Mooncake.rdata_type(...)` | Address-identified vs value-identified |
| Primitives | REPL: `Mooncake.is_primitive(Mooncake.DefaultCtx(), Mooncake.ReverseMode(), Tuple{typeof(sin), Float64})` | What gets a hand-written rule |

For more complex examples (branching, loops, mutation), use:
- `f(x) = x > 0 ? sin(x) : cos(x)` — shows PhiNodes, multiple blocks, BBCode advantage
- `f(x) = (s = 0.0; for i in 1:3; s += x; end; s)` — shows loops, block stack
- `f!(y, x) = (y .= sin.(x); nothing)` — shows mutation handling

## Topic Map

| User asks about... | Section | Key source files |
|---|---|---|
| SSA, IRCode, basic blocks, PhiNode | §1 Julia Prerequisites | Julia `base/compiler/ssair/ir.jl` |
| Derivative, adjoint, chain rule, math | §2 Math Foundations | `docs/src/understanding_mooncake/algorithmic_differentiation.md` |
| Mutation model, closures, function model | §3 Function Model | `docs/src/understanding_mooncake/rule_system.md` |
| Tangent, NoTangent, fdata, rdata, CoDual, Dual | §4 Tangent Types | `src/tangents/` |
| rrule!!, frule!!, primitive, writing rules | §5 Rules | `docs/src/understanding_mooncake/rule_system.md`, `src/rules/` |
| BBCode, ID, BBlock, IRCode limitations | §6 BBCode | `src/interpreter/bbcode.jl`, `docs/src/developer_documentation/ir_representation.md` |
| Pipeline, compilation, how it all fits together | §7 Pipeline | `src/interpreter/` (multiple files) |
| Reverse mode, pullback, forward pass, make_ad_stmts | §8 Reverse Mode | `src/interpreter/reverse_mode.jl` |
| Forward mode, dual IR, DerivedFRule | §9 Forward Mode | `src/interpreter/forward_mode.jl` |
| Normalization, foreigncall, intrinsics | §7 step 2 | `src/interpreter/ir_normalisation.jl` |
| MooncakeInterpreter, inlining, abstract interp | §7 step 3 | `src/interpreter/abstract_interpretation.jl` |
| Optimization, compact, SROA, ADCE | §7 step 5 | `src/interpreter/ir_utils.jl` |

---

## Reference Sections

### §1. Julia Compiler Prerequisites

Julia compiles functions to SSA-form IR (Static Single Assignment) where each value is assigned exactly once. Understanding this IR is essential for following Mooncake's transformations.

**Key concepts:**
- **SSAValues** (`%1`, `%2`, ...): results of statements, referenced by position
- **Arguments** (`_1`, `_2`, ...): function parameters
- **Basic blocks**: sequences of statements ending in a terminator (return, goto, conditional branch)
- **CFG**: control flow graph connecting basic blocks via edges
- **PhiNode**: `phi(#1 => a, #2 => b)` — value depends on which predecessor block we came from. Essential for loops and branches.
- **GotoNode/GotoIfNot**: unconditional/conditional branches
- **ReturnNode**: function return
- **PiNode**: type assertion (narrows type after a branch)
- **`:invoke` vs `:call`**: `:invoke` includes the MethodInstance (statically dispatched); `:call` is dynamically dispatched

**Demo**: Show `inspect_ir(sin, 1.0)` at `:raw` stage and walk through each element.

**Read for depth**: `docs/src/developer_documentation/ir_representation.md` (extensive, with runnable doctests)

### §2. Mathematical Foundations

Mooncake models derivatives as linear operators (Frechet derivatives), not scalars or Jacobian matrices. This generalizes AD to arbitrary types.

**Key concepts:**
- **Frechet derivative** D f[x]: a linear map from the tangent space at x to the tangent space at f(x)
- **Chain rule**: D (g . f)[x] = D g[f(x)] . D f[x] — composition of linear maps
- **Adjoint** D f[x]*: the transpose/adjoint of the derivative — this is what reverse mode computes
- **Forward mode** computes D f[x](dot_x) — pushes tangents forward
- **Reverse mode** computes D f[x]*(bar_y) — pulls cotangents backward
- **Gradient**: nabla f(x) = D f[x]*(1) for scalar-valued f

**Read for depth**: `docs/src/understanding_mooncake/algorithmic_differentiation.md` (rigorous, with worked examples for scalar, tuple, and mutating functions)

### §3. Modelling Julia Functions for AD

Mooncake models a Julia function as f: X -> X x A, where X is the input state (arguments before and after), and A is newly allocated data. This handles mutation.

**Key concepts:**
- Arguments can be mutated (e.g., `f!(y, x)` modifies `y`)
- The derivative must account for both entry and exit states of mutable args
- Closures are supported (captured variables are part of the function object)
- Global mutable state is NOT supported (not modelled)

**Read for depth**: `docs/src/understanding_mooncake/rule_system.md` (first half)

### §4. Tangent Type System

Every primal type P has a unique tangent type `tangent_type(P)`. Tangents split into forward data (fdata) and reverse data (rdata).

**Key types:**
- `NoTangent`: for non-differentiable types (Int, String, Symbol)
- `Tangent{NamedTuple}`: for immutable structs
- `MutableTangent{NamedTuple}`: for mutable structs (shared by reference)
- `CoDual{Tx, Tdx}`: pairs primal + fdata (reverse mode)
- `Dual{P, T}`: pairs primal + tangent (forward mode)

**The fdata/rdata split** is crucial:
- **fdata** (forward data): address-identified components (mutable things like Array, MutableTangent). Passed forward, incremented in-place on reverse.
- **rdata** (reverse data): value-identified components (bits types like Float64). Propagated only on reverse pass.
- For any tangent t: `tangent(fdata(t), rdata(t)) === t`

**Demo**: Show `Mooncake.tangent_type(Float64)`, `Mooncake.tangent_type(Vector{Float64})`, `Mooncake.fdata_type(tangent_type(Vector{Float64}))` in the REPL.

**Key files**: `src/tangents/tangents.jl`, `src/tangents/dual.jl`, `src/tangents/codual.jl`, `src/tangents/fwds_rvs_data.jl`

**Read for depth**: `docs/src/developer_documentation/tangents.md`, `docs/src/developer_documentation/custom_tangent_type.md` (for recursive types)

### §5. Rule Interface

Rules tell Mooncake how to differentiate primitive operations.

**Reverse mode** (`rrule!!`):
```julia
function rrule!!(f::CoDual, x::CoDual...)
    y = primal(f)(primal.(x)...)
    function pb!!(dy_rdata)
        return (NoRData(), df_rdata, dx_rdata...)  # rdata for each arg
    end
    return CoDual(y, fdata_of_y), pb!!
end
```

**Forward mode** (`frule!!`):
```julia
function frule!!(f::Dual, x::Dual...)
    # Compute primal + propagate tangent
    return Dual(primal_result, tangent_result)
end
```

**Primitive declaration**: `@is_primitive DefaultCtx Tuple{typeof(sin), Float64}`
- `MinimalCtx`: only things that would break AD if not primitive
- `DefaultCtx`: includes performance-motivated primitives (inherits MinimalCtx)
- `is_primitive(ctx, mode, sig)` checks if a signature is primitive

**Demo**: Show `Mooncake.is_primitive(...)` for sin vs a custom function. Then show how ir-inspect handles each differently in the forward IR.

**Key files**: `src/interpreter/contexts.jl`, `src/rules/builtins.jl`, `src/rules/low_level_maths.jl`

**Read for depth**: `docs/src/understanding_mooncake/rule_system.md` (second half), `docs/src/utilities/defining_rules.md`

### §6. BBCode — Mooncake's IR Representation

BBCode is Mooncake's custom IR where blocks and statements have stable IDs (not position-dependent indices). Only used for reverse mode.

**Why it exists**: In IRCode, inserting a new basic block shifts all subsequent block numbers, requiring manual updates to every branch target. BBCode uses unique `ID` objects that never change, making CFG manipulation safe. This is critical for reverse mode where many new blocks are created (Switch dispatch for the reverse pass).

**Key types:**
- `ID`: wraps Int32, created via global counter — stable across insertions
- `BBlock`: has `id::ID`, `inst_ids::Vector{ID}`, `insts::InstVector`
- `BBCode`: collection of BBlocks + metadata (argtypes, sptypes, etc.)
- `IDPhiNode`, `IDGotoNode`, `IDGotoIfNot`: like Julia's nodes but with ID-based references
- `Switch`: custom multi-way branch (lowered to GotoIfNot chain when converting back to IRCode)
- Round-trip: `BBCode(ir::IRCode)` and `IRCode(bb::BBCode)` are invertible

**Demo**: Show `inspect_ir(x -> x > 0 ? sin(x) : cos(x), 1.0)` at `:bbcode` stage — multiple blocks with ID-based naming. Compare with `:raw` stage.

**Key files**: `src/interpreter/bbcode.jl`

**Read for depth**: `docs/src/developer_documentation/ir_representation.md` (with code transformation and insertion examples)

### §7. The Compilation Pipeline

End-to-end, what happens when you call `build_rrule(f, x...)`:

1. **`lookup_ir`** — Gets type-inferred IRCode via `Core.Compiler.typeinf_ircode` using MooncakeInterpreter
   - File: `src/interpreter/ir_utils.jl` (function at line ~244)

2. **`normalise!`** — Makes every node a `:call` for uniform differentiation:
   - `:foreigncall` → `_foreigncall_()`, `:new` → `_new_()`, `:splatnew` → `_splat_new_()`
   - `Core.IntrinsicFunction`s → dispatchable wrappers from `IntrinsicsWrappers`
   - `getfield` → `lgetfield` (type-stable via `Val`), `memoryrefget` → `lmemoryrefget`
   - File: `src/interpreter/ir_normalisation.jl`

3. **Abstract interpretation** — `MooncakeInterpreter <: CC.AbstractInterpreter` prevents primitives from being inlined away. Intercepts `abstract_call_gf_by_type` and wraps primitive call info in `NoInlineCallInfo`. Custom `inlining_policy` returns `nothing` for these.
   - File: `src/interpreter/abstract_interpretation.jl`

4a. **Reverse mode** — Convert to BBCode → `make_ad_stmts!` per statement → `forwards_pass_ir` + `pullback_ir`
   - File: `src/interpreter/reverse_mode.jl` (see §8)

4b. **Forward mode** — Transform IRCode directly via `modify_fwd_ad_stmts!`
   - File: `src/interpreter/forward_mode.jl` (see §9)

5. **`optimise_ir!`** — `compact!` → type inference → `ssa_inlining_pass!` → `sroa_pass!` → `adce_pass!` → `compact!`
   - File: `src/interpreter/ir_utils.jl` (function at line ~174)

6. **Wrap in `MistyClosure`** (from MistyClosures.jl) → `DerivedRule` (reverse) or `DerivedFRule` (forward). MistyClosure wraps `Core.OpaqueClosure` but keeps the IR accessible for higher-order differentiation.

**Demo**: `show_ir(inspect_ir(sin, 1.0))` shows all stages end-to-end. `show_all_diffs(inspect_ir(sin, 1.0))` shows what each step changes.

### §8. Reverse Mode Internals

The core of reverse mode is `generate_ir` in `src/interpreter/reverse_mode.jl` (line ~1204).

**Per-statement transform** (`make_ad_stmts!`, line ~418): Each primal statement produces an `ADStmtInfo` with:
- `fwds`: forward-pass instructions
- `rvs`: reverse-pass instructions
- `comms_id`: value communicated fwd→rvs (typically the pullback)

| Node | Forward | Reverse |
|------|---------|---------|
| `:call`/`:invoke` | `rule(codual_args...)` → output + pullback | `pullback(rdata)` → distribute rdata to args |
| `ReturnNode(val)` | Return CoDual | Increment rdata ref of val |
| `IDPhiNode` | Wrap values in CoDuals | Distribute rdata to predecessor edges |
| `IDGotoIfNot` | Extract primal(cond), branch | Nothing |
| GlobalRef/literal | Wrap in `uninit_fcodual` | Nothing |

**Forward pass IR** (`forwards_pass_ir`, line ~1353): Entry block extracts shared data from the closure's captures. Per block: execute fwds stmts, push comms (pullbacks, intermediate values) onto per-block `Stack`s, push block ID onto BlockStack.

**Pullback IR** (`pullback_ir`, line ~1453): Entry block inits rdata Refs for all SSAs and args, then Switch-dispatches to whichever block the forward pass terminated in (popping from BlockStack). Per block (reverse order): pop comms from Stacks, execute rvs stmts in reverse, Switch to predecessor. Exit block: dereference rdata Refs, return arg rdata tuple.

**Key types**: `ADInfo` (line ~123), `SharedDataPairs` (line ~13), `DerivedRule` (line ~959), `Pullback` (line ~943), `LazyDerivedRule` (line ~1899, for static dispatch), `DynamicDerivedRule` (line ~1809, for dynamic dispatch), `Stack{T}` (`src/stack.jl`)

**Demo**: Show `:fwd_ir` and `:rvs_ir` for a branching function and walk through the Switch dispatch.

**Read for depth**: `docs/src/developer_documentation/reverse_mode_design.md`

### §9. Forward Mode Internals

Forward mode is simpler — it operates directly on IRCode (no BBCode needed, no CFG modification).

Every primal value becomes a `Dual` (primal + tangent pair). Every call `f(x...)` becomes `frule!!(Dual(f), Dual(x)...)`.

**Statement transform** (`modify_fwd_ad_stmts!`, line ~251):
- `:call`/`:invoke`: if primitive → `frule!!`, else → `LazyFRule(mi)` (static) or `DynamicFRule()` (dynamic)
- `GotoIfNot`: extract `primal(cond)` from Dual, then branch
- `PhiNode`: convert constant values to Duals, increment argument indices
- `ReturnNode`: return the Dual directly
- Arguments shifted by +1 (captures tuple inserted as Argument(1))

**Key types**: `DerivedFRule` (line ~114), `LazyFRule` (line ~463), `DynamicFRule` (line ~512)

**Demo**: `show_stage(inspect_ir(sin, 1.0; mode=:forward), :dual_ir)` — compare with reverse mode's fwd_ir to see the structural differences.

**Read for depth**: `docs/src/developer_documentation/forwards_mode_design.md` (comprehensive, with statement-by-statement transformation rules)
