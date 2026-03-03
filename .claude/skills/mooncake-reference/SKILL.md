---
name: mooncake-reference
description: >
  Explain Mooncake.jl internals, AD pipeline, IR transformations, tangent types,
  and rule system. Triggers on conceptual questions about how Mooncake works.
  Uses the ir-inspect skill to demonstrate concepts with real code examples.
---

# Mooncake.jl Reference & Teaching Guide

## Ground Rules

1. **Do not guess.** The summaries below are routing hints, not answers. You MUST `read_file` the referenced docs or source before answering any technical question.
2. **Always demo.** If a demo command exists for a topic, run it via bash and walk the user through the actual output. Never explain a concept purely in prose when you can show real IR.
3. **Use ir-inspect.** Load it once per session, then reuse:
   ```julia
   using Revise; includet(".claude/skills/ir-inspect/scripts/ir_inspect.jl")
   ```
4. **Offer to go deeper.** After answering, suggest related topics or next steps the user might want to explore.

## Useful example functions for demos

- `sin` or `cos` — simplest single-call function
- `x -> sin(x) * cos(x)` — two calls, shows multiple rule invocations
- `x -> x > 0 ? sin(x) : cos(x)` — branching, PhiNodes, multiple basic blocks
- `x -> (s = 0.0; for i in 1:3; s += x; end; s)` — loops, block stack
- `f!(y, x) = (y .= sin.(x); nothing)` — mutation handling

---

## Topics

### Julia SSA IR (IRCode, basic blocks, PhiNode, CFG)

**Trigger:** User asks about SSA form, IRCode, basic blocks, PhiNode, GotoNode, GotoIfNot, ReturnNode, PiNode, `:invoke` vs `:call`, or how to read Julia IR output.

**Action:**
1. Read `docs/src/developer_documentation/ir_representation.md` — this is Mooncake's own comprehensive guide to Julia IR, with runnable doctests and worked examples.
2. For implementation details, read `src/interpreter/ir_utils.jl` (IRCode helpers, `lookup_ir` at ~line 244).

**Demo:**
```julia
show_stage(inspect_ir(sin, 1.0), :raw)
```
Walk through: SSAValues (`%n`), Arguments (`_n`), `:invoke` nodes, return statements. For PhiNodes, use a branching function:
```julia
show_stage(inspect_ir(x -> x > 0 ? sin(x) : cos(x), 1.0), :raw)
```

---

### Mathematical Foundations (derivatives, adjoints, chain rule)

**Trigger:** User asks about Frechet derivatives, adjoint operators, chain rule, forward vs reverse mode mathematically, or how Mooncake models differentiation.

**Action:**
1. Read `docs/src/understanding_mooncake/algorithmic_differentiation.md` — rigorous treatment with three worked examples (scalar, tuple input, mutating function). Covers Frechet derivative as linear operator, adjoint, forward/reverse mode, and gradients.

**Demo:** No ir-inspect demo — this is pure math. But after explaining, offer to show the *result* of AD:
```julia
using Mooncake; value_and_gradient!!(build_rrule(sin, 1.0), sin, 1.0)
```

---

### Function Model for AD (mutation, closures, the X → X×A model)

**Trigger:** User asks about how Mooncake handles mutation, closures, mutable arguments, or the mathematical model for Julia functions.

**Action:**
1. Read `docs/src/understanding_mooncake/rule_system.md` (first half, up to "Primitives") — explains the `f: X → X × A` model where X is input/output state and A is newly allocated data.

---

### Tangent Types (tangent_type, CoDual, Dual, fdata/rdata)

**Trigger:** User asks about `tangent_type`, `NoTangent`, `Tangent`, `MutableTangent`, `CoDual`, `Dual`, `fdata`, `rdata`, or how Mooncake represents derivatives at the type level.

**Action:**
1. Read `docs/src/developer_documentation/tangents.md` — overview of the tangent interface.
2. For the fdata/rdata split, read `docs/src/understanding_mooncake/rule_system.md` — search for "fdata" and "rdata" sections.
3. For implementation: `src/tangents/tangents.jl` (NoTangent, Tangent, MutableTangent), `src/tangents/codual.jl` (CoDual), `src/tangents/dual.jl` (Dual), `src/tangents/fwds_rvs_data.jl` (fdata/rdata split).
4. For custom/recursive tangent types: `docs/src/developer_documentation/custom_tangent_type.md` (extremely detailed, 35K).

**Demo:**
```julia
using Mooncake
Mooncake.tangent_type(Float64)           # → Float64
Mooncake.tangent_type(Vector{Float64})   # → Vector{Float64}
Mooncake.tangent_type(String)            # → NoTangent
Mooncake.fdata_type(Mooncake.tangent_type(Vector{Float64}))  # → Vector{Float64} (address-identified)
Mooncake.rdata_type(Mooncake.tangent_type(Vector{Float64}))  # → NoRData (nothing value-identified)
```

---

### Rule Interface (rrule!!, frule!!, primitives, writing rules)

**Trigger:** User asks about `rrule!!`, `frule!!`, `is_primitive`, `@is_primitive`, `MinimalCtx`, `DefaultCtx`, how to write a rule, or how Mooncake decides what to differentiate through vs treat as a primitive.

**Action:**
1. Read `docs/src/understanding_mooncake/rule_system.md` (second half, from "Primitives" onward) — explains the full rule interface with worked examples.
2. Read `docs/src/utilities/defining_rules.md` — practical guide for writing rules.
3. For real rule examples, read `src/rules/low_level_maths.jl` (sin, cos, exp via `@from_chainrules`) or `src/rules/builtins.jl`.
4. For the context/primitive system: `src/interpreter/contexts.jl` (`MinimalCtx`, `DefaultCtx`, `@is_primitive`, `is_primitive`).

**Demo:**
```julia
using Mooncake
# Check what's primitive
Mooncake.is_primitive(Mooncake.DefaultCtx(), Mooncake.ReverseMode(), Tuple{typeof(sin), Float64})  # true
Mooncake.is_primitive(Mooncake.DefaultCtx(), Mooncake.ReverseMode(), Tuple{typeof(x -> sin(x) * cos(x)), Float64})  # false
```
Then show how this affects IR — primitives become rule calls, non-primitives get recursively derived:
```julia
show_stage(inspect_ir(sin, 1.0), :fwd_ir)      # primitive: direct rrule!! call
show_stage(inspect_ir(x -> sin(x) * cos(x), 1.0), :fwd_ir)  # derived: nested rule calls
```

---

### BBCode (stable-ID IR, BBlock, why not just IRCode)

**Trigger:** User asks about `BBCode`, `BBlock`, `ID`, `IDPhiNode`, `IDGotoNode`, `Switch`, or why Mooncake doesn't just use `IRCode` directly.

**Action:**
1. Read `docs/src/developer_documentation/ir_representation.md` — covers both IRCode and BBCode with code examples showing why BBCode's stable IDs are necessary for reverse-mode CFG manipulation.
2. For implementation: `src/interpreter/bbcode.jl` — the `BasicBlockCode` module with `ID`, `BBlock`, `BBCode`, conversion functions.

**Demo:**
```julia
# Show BBCode for a branching function — note stable IDs vs positional indices
show_stage(inspect_ir(x -> x > 0 ? sin(x) : cos(x), 1.0), :raw)     # IRCode: blocks numbered 1,2,3
show_stage(inspect_ir(x -> x > 0 ? sin(x) : cos(x), 1.0), :bbcode)  # BBCode: blocks have ID objects
```
Point out how IDs remain stable even when the reverse pass inserts new blocks.

---

### Compilation Pipeline (end-to-end, build_rrule, how it all fits)

**Trigger:** User asks about the compilation pipeline, how `build_rrule` works, what happens end-to-end when differentiating a function, or how the stages connect.

**Action:**
1. Read `docs/src/developer_documentation/reverse_mode_design.md` — informal but accurate walkthrough of the pipeline from `build_rrule` entry to `DerivedRule` output.
2. For forward mode pipeline: `docs/src/developer_documentation/forwards_mode_design.md` — the 6-step pipeline (lookup, normalise, transform, optimise, wrap, return).
3. Key source files for each stage:
   - `src/interpreter/ir_utils.jl` — `lookup_ir` (~line 244), `optimise_ir!` (~line 174)
   - `src/interpreter/ir_normalisation.jl` — `normalise!` (~line 24)
   - `src/interpreter/abstract_interpretation.jl` — `MooncakeInterpreter`, inlining prevention
   - `src/interpreter/reverse_mode.jl` — `generate_ir` (~line 1204), `build_rrule` (~line 1083)
   - `src/interpreter/forward_mode.jl` — `generate_dual_ir` (~line 165), `build_frule`

**Demo:**
```julia
# Show all stages end-to-end
show_ir(inspect_ir(sin, 1.0))
# Show what each transformation step changes
show_all_diffs(inspect_ir(sin, 1.0))
```

---

### Normalization (foreigncall, intrinsics, getfield lifting)

**Trigger:** User asks about `normalise!`, `_foreigncall_`, `_new_`, `_splat_new_`, `IntrinsicsWrappers`, `lgetfield`, `lmemoryrefget`, or why IR looks different after normalization.

**Action:**
1. Read `src/interpreter/ir_normalisation.jl` — the `normalise!` function (~line 24) and each sub-transform: `foreigncall_to_call` (~line 152), `new_to_call` (~line 226), `intrinsic_to_function` (~line 298), `getfield` lifting (~line 321).
2. For why normalization exists: `docs/src/developer_documentation/forwards_mode_design.md` — search for "Standardisation" section.

**Demo:**
```julia
show_diff(inspect_ir(sin, 1.0); from=:raw, to=:normalized)
```

---

### Reverse Mode Internals (make_ad_stmts!, forward pass, pullback)

**Trigger:** User asks about `make_ad_stmts!`, `ADInfo`, `forwards_pass_ir`, `pullback_ir`, `DerivedRule`, `Pullback`, `LazyDerivedRule`, `DynamicDerivedRule`, `Stack`, `BlockStack`, `Switch` dispatch, or how the forward/reverse pass IR is constructed.

**Action:**
1. Read `docs/src/developer_documentation/reverse_mode_design.md` — orientation document for the reverse-mode pipeline.
2. For implementation details, read `src/interpreter/reverse_mode.jl`:
   - `ADInfo` struct (~line 123), `SharedDataPairs` (~line 13)
   - `make_ad_stmts!` (~line 418) — per-statement AD transform
   - `forwards_pass_ir` (~line 1353) — constructs forward pass BBCode
   - `pullback_ir` (~line 1453) — constructs reverse pass BBCode
   - `DerivedRule` (~line 959), `Pullback` (~line 943)
   - `LazyDerivedRule` (~line 1899), `DynamicDerivedRule` (~line 1809)
3. For the Stack data structure: `src/stack.jl`.

**Demo:**
```julia
# Show forward and reverse pass IR for a branching function
ins = inspect_ir(x -> x > 0 ? sin(x) * x : cos(x), 1.0)
show_stage(ins, :fwd_ir)   # Note: CoDual wrapping, rule calls, comms pushed to Stacks
show_stage(ins, :rvs_ir)   # Note: Switch dispatch, reverse-order execution, rdata propagation
```

---

### Forward Mode Internals (Dual IR, DerivedFRule)

**Trigger:** User asks about forward mode implementation, `generate_dual_ir`, `modify_fwd_ad_stmts!`, `DerivedFRule`, `LazyFRule`, `DynamicFRule`, or how forward mode differs from reverse.

**Action:**
1. Read `docs/src/developer_documentation/forwards_mode_design.md` — comprehensive design doc with statement-by-statement transformation rules and comparison with reverse mode.
2. For implementation: `src/interpreter/forward_mode.jl`:
   - `generate_dual_ir` (~line 165)
   - `modify_fwd_ad_stmts!` (~line 251)
   - `DerivedFRule` (~line 114), `LazyFRule` (~line 463), `DynamicFRule` (~line 512)

**Demo:**
```julia
# Compare forward mode vs reverse mode IR for the same function
show_stage(inspect_ir(sin, 1.0; mode=:forward), :dual_ir)
show_stage(inspect_ir(sin, 1.0), :fwd_ir)
```
Key difference: forward mode operates directly on IRCode (no BBCode), replaces values with Duals, and calls `frule!!` instead of `rrule!!`.

---

### Optimization (optimise_ir!, inlining, SROA, ADCE)

**Trigger:** User asks about what optimization Mooncake does, `optimise_ir!`, why the optimized IR looks different, or about compact/inlining/SROA/ADCE passes.

**Action:**
1. Read `src/interpreter/ir_utils.jl` — `optimise_ir!` function (~line 174). The pipeline is: `compact!` → `__infer_ir!` → `ssa_inlining_pass!` → `sroa_pass!` → `adce_pass!` → `compact!`.

**Demo:**
```julia
show_diff(inspect_ir(sin, 1.0); from=:fwd_ir, to=:optimized_fwd)
show_diff(inspect_ir(sin, 1.0); from=:rvs_ir, to=:optimized_rvs)
```

---

### Abstract Interpretation (MooncakeInterpreter, inlining prevention)

**Trigger:** User asks about `MooncakeInterpreter`, `AbstractInterpreter`, `NoInlineCallInfo`, why primitives aren't inlined, or how Mooncake hooks into Julia's compiler.

**Action:**
1. Read `src/interpreter/abstract_interpretation.jl` — `MooncakeInterpreter <: CC.AbstractInterpreter` (~line 27), `abstract_call_gf_by_type` (~line 130) intercepts calls and wraps primitives in `NoInlineCallInfo`, `inlining_policy` (~line 254) blocks inlining for these.
2. For context: `docs/src/developer_documentation/reverse_mode_design.md` — search for "AbstractInterpreter" section.

---

### Debugging & Troubleshooting

**Trigger:** User asks about errors during AD, "primitive not found", world age issues, how to debug Mooncake failures, or how to create minimal working examples.

**Action:**
1. Read `docs/src/utilities/debugging_and_mwes.md` — guide for debugging AD failures and creating MWEs.
2. Read `docs/src/utilities/debug_mode.md` — Mooncake's debug mode for verbose rule checking.
3. Read `docs/src/known_limitations.md` — known issues and unsupported patterns.
4. For world age debugging, use ir-inspect:
   ```julia
   ins = inspect_ir(f, args...); show_world_info(ins)
   ```

---

### User-Facing API (value_and_gradient!!, prepare_cache, etc.)

**Trigger:** User asks about how to use Mooncake, the public API, `value_and_gradient!!`, `value_and_pullback!!`, `prepare_gradient_cache`, or basic usage.

**Action:**
1. Read `docs/src/interface.md` — public API reference.
2. Read `docs/src/tutorial.md` — getting started guide.
3. For implementation: `src/interface.jl` — all public entry points.
