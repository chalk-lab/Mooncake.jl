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
