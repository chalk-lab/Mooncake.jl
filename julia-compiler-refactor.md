# Compiler refactor

Mooncake has a tight coupling with Julia compiler internals. Recent Julia
versions have changed field names, constructor layouts, pass return shapes, and
AbstractInterpreter hook signatures. The aim of this refactor is not to make the
current code prettier by wrapping every `Core.Compiler` use one-for-one. The aim
is to define a boundary that matches the kinds of interfaces a Julia compiler
team member might reasonably design, then adapt Mooncake to that shape.

In other words: think top-down from compiler services and IR invariants, not
bottom-up from today's `IRCode` fields.

## Goal

Meet the compiler in the middle:

- Julia compiler internals remain unstable and are implemented by whatever
  structs, fields, passes, and helper functions are convenient for Julia.
- Mooncake depends on higher-level compiler concepts: inferred IR, argument
  types, valid world ranges, CFG/statement traversal, optimization passes,
  custom interpreter behavior, and code-object construction.
- `src/compiler/` becomes Mooncake's local version of that high-level compiler
  interface. It may be implemented with private field access and version
  branches today, but the rest of Mooncake should be written against semantic
  operations rather than Julia's current struct layout.

## What not to do

Avoid designing the end state as a bag of accessors like:

```julia
Compiler.ir_argtypes(ir)
Compiler.ir_valid_worlds(ir)
Compiler.mi_specTypes(mi)
```

These are useful as tactical migration helpers because they centralize field
churn and make `rg`-based cleanup possible. But they still leak Julia's current
object model and even preserve unstable field names in Mooncake-facing APIs.

If a wrapper name is just a field name with a prefix, treat it as scaffolding,
not the final abstraction.

## Compiler-service view

A Julia compiler-facing interface would more naturally be organized around
services and invariants.

### Inference

Given a target method/signature, interpreter, and world, produce typed IR and an
inferred return type.

Conceptual API:

```julia
Compiler.infer_ir(interp, target; optimize_until=nothing)
Compiler.infer_ir_for_match(interp, match, target; optimize_until=nothing)
```

Mooncake should not encode whether this is currently implemented via
`typeinf_ircode`, a `MethodMatch`, `SpecInfo`, `MethodInfo`, or a version-specific
argument order.

### IR inspection

Read stable semantic properties of typed IR.

Conceptual API:

```julia
Compiler.argument_types(ir)
Compiler.statement_stream(ir)
Compiler.control_flow_graph(ir)
Compiler.debug_info(ir)
Compiler.valid_world_range(ir)
Compiler.inferred_return_type(ir)
```

These may be backed by field access, but the names describe compiler concepts,
not field spellings.

### IR mutation and invariants

Modify IR through operations that preserve compiler invariants.

Conceptual API:

```julia
Compiler.set_argument_type!(ir, i, T)
Compiler.replace_statement!(ir, ssa, stmt; type, info, flag)
Compiler.insert_before!(ir, ssa, inst)
Compiler.insert_after!(ir, ssa, inst)
Compiler.compact!(ir)
Compiler.verify(ir)
```

This matters more than simple read accessors. If callers still need to know the
`IRCode` constructor layout, `InstructionStream` field layout, or how statement
types/info/flags must stay aligned, the behavior has not really been isolated.

### Pass pipeline

Run compiler passes through Mooncake-level intent, not raw pass details.

Conceptual API:

```julia
Compiler.optimize_ir!(ir, interp; do_inline=true)
Compiler.run_inlining!(ir, interp)
Compiler.run_sroa!(ir, interp)
Compiler.run_adce!(ir, interp)
Compiler.verify_debug_info(ir)
```

Mooncake should not care that `adce_pass!` returns different shapes on different
Julia versions, or that debug-info verification takes `linetable` on one version
and `debuginfo` on another.

### World and method-table services

Represent the method-world assumptions under which IR is valid.

Conceptual API:

```julia
Compiler.current_world()
Compiler.valid_at_world(ir, world)
Compiler.restrict_to_world(ir, world)
Compiler.overlay_method_table(world, table)
```

World validity is not just a compiler detail for Mooncake; it is part of the
correctness story for MistyClosures, OpaqueClosures, cached rules, and nested AD.

### AbstractInterpreter behavior

Mooncake's custom interpreter behavior should be expressed as narrow hooks:

```julia
Compiler.should_inline(interp, callsite)
Compiler.method_table(interp)
Compiler.cache_policy(interp)
Compiler.override_call(interp, callsite)
```

The current reality requires method definitions on `Core.Compiler` functions and
some copied upstream logic. The design target is smaller: Mooncake wants to avoid
inlining primitives, use an overlay method table, customize caching/world age, and
possibly intercept calls. It does not want to own Julia's generic-function
abstract-call implementation.

### Code-object and closure construction

Mooncake should not populate raw `CodeInfo` fields outside the compiler boundary.

Conceptual API:

```julia
Compiler.codeinfo_from_ir(ir; nargs, slottypes, world)
Compiler.opaque_closure_from_ir(ir, env; isva)
```

This hides constructor layout, debug-info shape, slot metadata, and valid-world
fields.

## What stays raw

Mooncake can still traffic in raw compiler IR types where those are the domain
objects being transformed:

- `IRCode`
- `SSAValue`
- `NewInstruction`
- statement node types
- `BBCode` as Mooncake's own block representation
- developer tools returning raw IR for inspection

The boundary should isolate volatile behavior and representation assumptions,
not pretend Mooncake is independent of Julia IR.

## Current Migration Notes

New Mooncake internals should call `Mooncake.Compiler.*` directly. The old
top-level helper names `lookup_ir`, `optimise_ir!`, `set_valid_world!`,
`compute_ir_rettype`, and `compute_oc_signature` are compatibility shims only.
They remain for old internal snippets and downstream code that reached into
Mooncake internals, but they should not be used in new implementation code.

The compiler-boundary static gate lives in `test/compiler/compiler.jl`. It keeps
known compiler-internal names localized under `src/compiler/`, with a named
exception for the issue-319 compatibility patch in
`src/interpreter/patch_for_319.jl`.

Because this boundary absorbs Julia compiler churn, changes in this area should
be checked across supported Julia versions. The focused local target is the
compiler tests plus the interpreter IR-generation tests on Julia 1.12, 1.11,
and 1.10, followed by the `basic` test group on Julia 1.12.

## Migration rule

Use a two-stage migration:

1. Centralize current field accesses and version branches with simple helpers
   where needed. This gives mechanical safety and clear `rg` gates.
2. Collapse clusters of those helpers into semantic compiler services. Consumer
   code should mostly call operations like `argument_types`, `restrict_to_world`,
   `infer_ir`, `optimize_ir!`, and `opaque_closure_from_ir`.

Accessors are acceptable inside `src/compiler/` as private implementation
details. They should not become the conceptual API that AD code is written
against.

## Design test

For each proposed wrapper, ask:

- Would this name make sense if Julia exposed it as a compiler interface?
- Does it describe the operation Mooncake needs, or merely the field Julia
  happens to store today?
- Does it preserve an invariant or just return a mutable internal vector?
- If Julia changed the underlying struct layout, would callers outside
  `src/compiler/` still be unchanged?

The refactor is successful when most Julia-version churn can be absorbed inside
`src/compiler/`, while the rest of Mooncake reads like AD logic over a small
compiler-service API.
