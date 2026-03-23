# [Compiler Source Map](@id compiler_source_map)

This page maps each category of compiler boundary issue to the Julia source files
that implement the relevant decision logic. All paths are relative to the Julia
repository root.

!!! note
    Paths under `Compiler/src/` are for Julia 1.12+. On Julia < 1.12, the
    equivalent files are under `base/compiler/`. See
    [Version Handling](@ref compiler_version_policy) for the full mapping.

## 1. Specialization Widening

When Julia compiles a method, it may widen concrete argument types to more
abstract types in the compilation signature. This reduces the number of compiled
specializations but can prevent optimization.

| File | Key symbols | What it does |
|------|-------------|--------------|
| `src/gf.c` | `jl_compilation_sig`, `jl_isa_compileable_sig` | The main specialization decision function |
| `src/method.c` | `jl_method_set_source`, `jl_method_def` | Reads lowered code metadata and sets the `called` bitmask |
| `src/typemap.c` | `jl_typemap_insert` | Method cache for storing compiled specializations |
| `Compiler/src/typelimits.jl` | `tmerge`, `type_more_complex` | Type widening during abstract interpretation |

`jl_compilation_sig` contains multiple independent widening heuristics. Key
predicates to look for:
- `notcalled_func` -- widens Function subtypes when not in call position
- `very_general_type` -- widens `Type{T}` when declared type is too general
- `nospecialize` -- respects `@nospecialize` annotations

Common mitigations: `@inline` (eliminates the function boundary), `@eval`-generated
per-arity methods, wrapping callables in non-Function structs.

## 2. Inlining Failures

Julia's inlining pass decides which callees to inline based on a cost model,
effect analysis, and various heuristics.

| File | Key symbols | What it does |
|------|-------------|--------------|
| `Compiler/src/ssair/inlining.jl` | `resolve_todo`, `analyze_method!` | The inlining pass |
| `Compiler/src/optimize.jl` | `inlining_cost`, `is_inlineable` | Cost model and orchestration |
| `Compiler/src/effects.jl` | `Effects`, `is_nothrow` | Effect analysis constraints |
| `Compiler/src/stmtinfo.jl` | `MethodMatchInfo` | Dispatch resolution results |

Search terms: `inline_cost`, `MAX_INLINE_COST`, `ssa_inlining_pass!`,
`SRC_FLAG_DECLARED_NOINLINE`

## 3. SROA / Allocation Failures

Scalar Replacement of Aggregates (SROA) eliminates heap allocations by replacing
structs with their individual fields.

| File | Key symbols | What it does |
|------|-------------|--------------|
| `Compiler/src/ssair/passes.jl` | `sroa_pass!`, `getfield_elim_pass!` | High-level SROA |
| `Compiler/src/ssair/EscapeAnalysis.jl` | `analyze_escapes`, `EscapeState` | Escape analysis |
| `src/llvm-alloc-opt.cpp` | `AllocOptPass`, `tryPromote` | LLVM-level allocation optimization |

Search terms: `sroa_pass!`, `analyze_escapes`, `is_load_forwardable`, `gc_alloc_obj`

## 4. Type Instability / Dynamic Dispatch

When Julia can't infer a concrete return type, it falls back to dynamic
dispatch.

| File | Key symbols | What it does |
|------|-------------|--------------|
| `Compiler/src/abstractinterpretation.jl` | `abstract_call_gf_by_type` | Core type inference |
| `Compiler/src/typelattice.jl` | `widenconst`, `tmerge` | Type lattice and merging |
| `src/gf.c` | `jl_apply_generic` | Runtime dynamic dispatch entry point |

Search terms: `abstract_call_gf_by_type`, `jl_apply_generic`, `tmerge`

## 5. World Age / Method Invalidation

Julia uses world ages for consistency: code compiled at world N only sees
methods defined before world N.

| File | Key symbols | What it does |
|------|-------------|--------------|
| `src/method.c` | `jl_method_table_insert` | Method insertion and invalidation |
| `Compiler/src/cicache.jl` | `CodeInstance`, `WorldView` | Code instance cache |
| `Compiler/src/typeinfer.jl` | `typeinf`, `finish` | Type inference entry point |

!!! note
    On Julia < 1.12, `invalidate_method_instance` is in `base/compiler/typeinfer.jl`.
    On Julia 1.12+, it moved to `Compiler/src/bindinginvalidations.jl`.

Search terms: `world_age`, `min_world`, `max_world`, `invalidate`,
`valid_worlds`

## 6. Boxing

Boxing converts value types (e.g. `Float64`) into heap-allocated objects for
type-erased interfaces.

| File | Key symbols | What it does |
|------|-------------|--------------|
| `src/codegen.cpp` | `emit_expr`, `emit_invoke` | Main codegen |
| `src/cgutils.cpp` | `emit_box`, `emit_unbox` | Boxing/unboxing helpers |
| `src/llvm-alloc-opt.cpp` | `AllocOptPass` | Post-codegen allocation elimination |

Search terms: `emit_box`, `jl_box_`, `jl_gc_pool_alloc`, `gc_alloc_obj`

## Julia documentation (fast triage)

Before diving into source, check if the issue is a known pitfall:

| Doc file | Covers |
|----------|--------|
| `performance-tips.md` | All common performance pitfalls |
| `methods.md` | Method dispatch, parametric methods, Vararg |
| `devdocs/inference.md` | How type inference works |
| `devdocs/ssair.md` | SSA IR format |
| `devdocs/EscapeAnalysis.md` | Escape analysis internals |
