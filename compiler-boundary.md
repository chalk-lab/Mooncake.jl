# Compiler Boundary

Mooncake deliberately transforms raw Julia IR, so types such as `Core.Compiler.IRCode`,
`Core.SSAValue`, and statement nodes are still part of the implementation. The boundary is
about where volatile compiler mechanics live: Julia-version-specific field names,
constructor layouts, pass return shapes, inference entry points, world ranges, and
`OpaqueClosure` construction belong under `Mooncake.Compiler`.

AD code outside `src/compiler/` should call compiler services such as:

```julia
Mooncake.Compiler.infer_ir(interp, target; optimize_until=nothing)
Mooncake.Compiler.restrict_to_world(ir, world)
Mooncake.Compiler.inferred_return_type(ir)
Mooncake.Compiler.optimize_ir!(ir; do_inline=true)
Mooncake.Compiler.opaque_closure_from_ir(ret_type, ir, env...)
```

Legacy helpers such as `Mooncake.lookup_ir`, `Mooncake.optimise_ir!`,
`Mooncake.set_valid_world!`, `Mooncake.compute_ir_rettype`, and
`Mooncake.compute_oc_signature` are compatibility shims. New Mooncake internals should not
use them.

## Abstraction Goal

`Mooncake.Compiler` is Mooncake's local interface layer between the Julia compiler and the
AD implementation. It is not only a folder for version guards. Its purpose is to make the
rest of Mooncake depend on compiler concepts and invariants rather than on the current
spelling of Julia internals.

The module is allowed to contain a mix of:

- services that are plausible future upstream compiler-extension APIs,
- Mooncake-specific IR operations used by the AD transform, and
- compatibility shims that keep old internal and downstream call sites working during the
  migration.

That mixing is intentional for now. The boundary is successful if caller code outside
`src/compiler/` can be written in terms of stable Mooncake needs: infer IR for this target,
restrict this IR to the interpreter world, replace this statement while preserving
instruction metadata, run the standard optimization pipeline, and materialize this IR as a
callable closure.

When adding or changing a compiler service, prefer names that describe a semantic operation
over names that mirror fields. For example, `restrict_to_world(ir, world)` is better than
`set_valid_world!`, and `replace_statement!` is better than setting the statement, type,
info, and flag vectors independently at each call site. Thin accessors are acceptable as
temporary scaffolding, but consumer code should move toward operations that preserve a
compiler invariant.

## Service Families

The current boundary is organized around the following service families. This list is a
design guide, not a promise that every function is final.

### Inference

`Compiler.infer_ir` owns the details of obtaining typed IR for signatures,
`MethodInstance`s, and `MistyClosure`s. Callers should not know whether this goes through
`typeinf_ircode`, a `MethodMatch`, a method-instance specialization, or a cached closure IR.

### IR Inspection And Mutation

Statement streams, argument types, statement types, instruction replacement, insertion, and
CFG edge repair belong behind compiler services. Mooncake can still transform raw Julia IR
nodes, but it should not repeatedly encode `IRCode` or `InstructionStream` layout outside
`src/compiler/`.

Mutation services should preserve the invariants they touch. For example,
`replace_statement!` updates the statement, type, call info, and flag together, and
`remove_control_flow_edge!` updates successors, predecessors, and leading `PhiNode`s
together.

### World And Method-Table Semantics

World-age validity is part of Mooncake's correctness model. Compiler services should own
world-range checks, world restriction, overlay method-table construction, and any
version-specific representation of valid worlds. AD code should ask whether IR is valid at
a world; it should not inspect compiler world fields directly.

### Interpreter Protocol

`MooncakeInterpreter` still implements Julia's `AbstractInterpreter` protocol methods, but
the implementation should be narrow and delegate into compiler services where useful. The
goal is to keep the protocol surface visible while avoiding scattered direct access to
interpreter caches, parameters, and world fields.

### Call Analysis And Inlining Policy

Mooncake needs primitive calls to remain visible in generated IR. The current implementation
still hooks Julia's `abstract_call_gf_by_type` and inlining-policy methods directly, but the
primitive-detection, return-type widening, and no-inline marking logic should live under
`Mooncake.Compiler`. If Julia later exposes a higher-level "do not inline this call pattern"
API, this is the family that should collapse onto it.

### Optimization Pipeline

`Compiler.optimize_ir!` is Mooncake's local "optimize synthesized IR" operation. It should
hide pass ordering, version-specific pass return values, debug-info verification, and
constant-propagation entry points. Callers should choose semantic options, such as whether
to allow inlining, rather than calling individual compiler passes.

### Materialization

`Compiler.opaque_closure_from_ir` owns the current path from synthesized IR to
`OpaqueClosure`. This includes signature computation, `CodeInfo` construction, world
metadata, and return-type assignment. Capture reuse is not fully behind this service yet:
`replace_captures` still depends on `OpaqueClosure` layout outside `src/compiler/`. Moving
that behind a materialization service is a concrete remaining boundary task.

## Next Steps

The next phase is not to make every wrapper perfect. It is to make the abstraction
consistent enough that future Julia compiler churn is absorbed in one place.

1. Section `src/compiler/Compiler.jl` by service family.
   Add comment headers matching the families above. This makes the mixed contents
   intentional and gives reviewers a place to put new compiler dependencies.

2. Move capture replacement behind the materialization family.
   `replace_captures(::OpaqueClosure, new_captures)` currently constructs a new closure by
   reaching into the `OpaqueClosure` layout from reverse-mode code. Add a compiler service
   for capture replacement, then make reverse mode call that service.

3. Tighten the static gate around materialization internals.
   After capture replacement moves, extend `test/compiler/compiler.jl` so direct
   `OpaqueClosure` layout construction and materialization internals stay out of the AD
   transform.

4. Replace remaining legacy helper use in implementation code.
   `lookup_ir`, `optimise_ir!`, `set_valid_world!`, `compute_ir_rettype`, and
   `compute_oc_signature` should remain available as compatibility shims, but ordinary
   Mooncake implementation code should call `Mooncake.Compiler.*` directly.

5. Classify thin accessors as scaffolding or stable local services.
   For each accessor-like function, decide whether it should stay because it preserves a
   useful invariant for callers, or whether it is only a migration helper. If it is only a
   helper, keep it private to `src/compiler/` call paths or replace call sites with a
   higher-level service.

6. Keep upstream candidates visible but do not optimize for upstreaming yet.
   The most likely future upstream discussions are inferred-IR lookup, synthesized-IR
   optimization, materialization from IR, interpreter protocol hooks, and no-inline call
   policy. For now, the code should first be a good Mooncake boundary; upstream API design
   can happen after the local abstraction has survived real use.

## Static Gate

`test/compiler/compiler.jl` includes a grep-style boundary test. The criterion: compiler
internal names that have either moved between Julia versions, that ccall directly into
Julia's C runtime, or that touch unstable binding/inference internals. The following names
should only appear under `src/compiler/`:

- `typeinf_ircode`
- `adce_pass!`
- `ssa_inlining_pass!`
- `sroa_pass!`
- `scan_leaf_partitions`
- `compute_ir_rettype`
- `compute_oc_signature`
- `jl_new_code_info_uninit`
- `jl_new_method_instance_uninit`
- `generate_opaque_closure`

`compute_ir_rettype` and `compute_oc_signature` have a documented exception for the legacy
compatibility shims at `src/utils.jl:479-480`.

The issue-319 compatibility patch in `src/interpreter/patch_for_319.jl` is the named
exception for `IRInterpretationState`. That file contains copied compiler bug-fix logic and
should remain isolated until the supported Julia patch releases include the upstream fix.

When adding a new compiler-internal dependency outside `src/compiler/`, either move the
operation behind a `Mooncake.Compiler` service or add a narrow, documented exception to the
static gate.

## Multi-Version Checks

Compiler-boundary changes should be checked on each supported Julia minor version. Use
version-specific manifests where available so local validation does not re-resolve one
shared environment:

```bash
julia +1.12 --project=temp/testenv -e 'using TestEnv; TestEnv.activate("Mooncake"); include("test/front_matter.jl"); include("test/compiler/compiler.jl"); include("test/interpreter/ir_utils.jl"); include("test/interpreter/forward_mode.jl"); include("test/interpreter/reverse_mode.jl")'
julia +1.12 --project=. -e 'import Pkg; Pkg.test(; test_args=["basic"])'
julia +1.11 --project=temp/testenv -e 'using TestEnv; TestEnv.activate("Mooncake"); include("test/front_matter.jl"); include("test/compiler/compiler.jl"); include("test/interpreter/ir_utils.jl"); include("test/interpreter/forward_mode.jl"); include("test/interpreter/reverse_mode.jl")'
julia +1.10 --project=temp/testenv -e 'using TestEnv; TestEnv.activate("Mooncake"); include("test/front_matter.jl"); include("test/compiler/compiler.jl"); include("test/interpreter/ir_utils.jl"); include("test/interpreter/forward_mode.jl"); include("test/interpreter/reverse_mode.jl")'
```

The local `./julia` checkout is source-only unless `./julia/julia` or
`./julia/usr/bin/julia` exists. Identify or build that binary before claiming validation
against it.
