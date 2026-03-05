# Julia Compiler Source Map

Maps each category of compiler boundary issue to the Julia source files that
implement the relevant decision logic. All paths are relative to the Julia
repository root.

Use `{BASE}` as a placeholder for the version-pinned raw GitHub URL prefix, e.g.:
```
{BASE} = https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3
```

**Note on Julia versions:** Paths under `Compiler/src/` are for Julia 1.12+.
On Julia < 1.12, the equivalent files are under `base/compiler/`. See
`version-policy.md` for the full mapping.


## 1. Specialization Widening

When Julia compiles a method, it may choose to widen concrete argument types to
more abstract types in the compilation signature. This reduces the number of
compiled specializations but can prevent optimization.

### Files

| File | Key symbols | What it does |
|------|-------------|--------------|
| `src/gf.c` | `jl_compilation_sig`, `jl_isa_compileable_sig` | The main specialization decision function. Iterates over arguments and applies widening heuristics. |
| `src/method.c` | `jl_method_set_source`, `jl_method_def` | Reads lowered code metadata (slotflags) and sets method properties like the `called` bitmask. |
| `src/typemap.c` | `jl_typemap_insert`, `jl_typemap_assoc_by_type` | Method cache — how compiled specializations are stored and retrieved. |
| `Compiler/src/typelimits.jl` | `tmerge`, `type_more_complex` | Type widening heuristics during abstract interpretation (can compound with specialization widening). |
| `Compiler/src/methodtable.jl` | `findsup`, `getindex` | Julia-side method table operations. |

### What to look for

`jl_compilation_sig` contains multiple independent widening heuristics, each
guarded by its own predicate. When investigating, identify WHICH heuristic
fired and WHY. Key predicates to look for:
- `notcalled_func` — widens Function subtypes when the parameter isn't in call position
- `very_general_type` — widens Type{T} when the declared type is too general
- `nospecialize` — respects `@nospecialize` annotations
- Vararg collapsing — how `i_arg` maps argument indices to parameter indices

### Search terms

`jl_compilation_sig`, `notcalled_func`, `iscalled`, `definition->called`,
`nospecialize`, `i_arg`, `very_general_type`, `jl_function_type`,
`jl_is_type_type`, `jl_is_kind`

### Common mitigations

- `@inline` on the affected function (sidesteps the problem by eliminating the
  function boundary — no MethodInstance created, so widening never runs)
- `@eval`-generated per-arity methods (forces concrete specialization by
  expanding Vararg into explicit typed parameters)
- Wrapping callable arguments in a non-Function struct


## 2. Inlining Failures

Julia's inlining pass decides which callees to inline based on a cost model,
effect analysis, and various heuristics.

### Files

| File | Key symbols | What it does |
|------|-------------|--------------|
| `Compiler/src/ssair/inlining.jl` | `resolve_todo`, `analyze_method!`, `handle_single_case!` | The inlining pass. Decides per-callsite whether to inline. |
| `Compiler/src/optimize.jl` | `optimize`, `run_passes`, `inlining_cost`, `is_inlineable` | Orchestrates optimization passes. Contains cost model. |
| `Compiler/src/effects.jl` | `Effects`, `is_nothrow`, `is_consistent` | Effect analysis — constraints on what can be inlined. |
| `Compiler/src/stmtinfo.jl` | `MethodMatchInfo`, `UnionSplitInfo` | Carries dispatch resolution results from inference to the inliner. |

### Search terms

`inline_cost`, `isinlineable`, `MAX_INLINE_COST`, `InliningPolicy`,
`ssa_inlining_pass!`, `@inline`, `@noinline`, `SRC_FLAG_DECLARED_NOINLINE`


## 3. SROA / Allocation Failures

Scalar Replacement of Aggregates (SROA) eliminates heap allocations by replacing
structs with their individual fields. When SROA fails, values that should be
stack-allocated end up on the heap.

### Files

| File | Key symbols | What it does |
|------|-------------|--------------|
| `Compiler/src/ssair/passes.jl` | `sroa_pass!`, `getfield_elim_pass!`, `adce_pass!` | High-level SROA: replaces `getfield` on known-type allocations with direct SSA values. |
| `Compiler/src/ssair/EscapeAnalysis.jl` | `analyze_escapes`, `EscapeState` | Determines whether a value escapes its defining scope. |
| `src/llvm-alloc-opt.cpp` | `AllocOptPass`, `tryPromote` | LLVM-level allocation optimization — second chance after Julia-level SROA. |
| `src/llvm-late-gc-lowering.cpp` | `lowerGCFrame` | GC frame emission — relevant for GC root overhead. |

### Search terms

`sroa_pass!`, `getfield_elim_pass!`, `analyze_escapes`, `is_load_forwardable`,
`EscapeState`, `alloc_opt`, `gc_alloc_obj`


## 4. Type Instability / Dynamic Dispatch

When Julia can't infer a concrete return type, it falls back to dynamic dispatch,
which is orders of magnitude slower than static dispatch.

### Files

| File | Key symbols | What it does |
|------|-------------|--------------|
| `Compiler/src/abstractinterpretation.jl` | `abstract_call_gf_by_type`, `abstract_call_known`, `abstract_call` | Core of type inference. Resolves method calls to return types. |
| `Compiler/src/typelattice.jl` | `widenconst`, `tmerge` | The type lattice. `tmerge` joins types at control-flow merges — where precision is lost. |
| `Compiler/src/abstractlattice.jl` | `Conditional`, `PartialStruct` | Extended lattice elements that carry extra precision beyond plain types. |
| `Compiler/src/types.jl` | `MethodInstance`, `CodeInstance`, `InferenceResult` | Compiler type definitions. |
| `src/gf.c` | `jl_apply_generic` | The runtime dynamic dispatch entry point. |

### Search terms

`abstract_call_gf_by_type`, `abstract_call_known`, `jl_apply_generic`,
`widenconst`, `tmerge`, `MethodMatchInfo`, `UnionSplitApplicability`


## 5. World Age / Method Invalidation

Julia uses world ages for consistency: code compiled at world N only sees methods
defined before world N. New method definitions can invalidate existing compiled code.

### Files

| File | Key symbols | What it does |
|------|-------------|--------------|
| `src/method.c` | `jl_method_table_insert` | Where new methods are inserted and invalidation is triggered. |
| `Compiler/src/bindinginvalidations.jl` | `invalidate_method_instance` | Julia-side invalidation logic. **(1.12+ only; on < 1.12, this is in `base/compiler/typeinfer.jl`.)** |
| `Compiler/src/cicache.jl` | `CodeInstance`, `WorldView` | Code instance cache — how compiled code is stored and looked up by world range. |
| `Compiler/src/typeinfer.jl` | `typeinf`, `finish` | Type inference entry point — where inference is triggered. |

### Search terms

`world_age`, `min_world`, `max_world`, `invalidate`, `jl_method_table_insert`,
`BackedgePair`, `CodeInstance`, `valid_worlds`


## 6. Boxing / Unexpected Allocations

Boxing converts value types (e.g. `Float64`) into heap-allocated objects for
type-erased interfaces.

### Files

| File | Key symbols | What it does |
|------|-------------|--------------|
| `src/codegen.cpp` | `emit_expr`, `emit_invoke`, `emit_call` | Main codegen: translates Julia IR to LLVM IR. Boxing happens when a value must be passed as `jl_value_t*`. |
| `src/cgutils.cpp` | `emit_box`, `emit_unbox`, `emit_allocobj` | Helpers for boxing/unboxing values and allocating objects. |
| `src/llvm-alloc-opt.cpp` | `AllocOptPass` | Post-codegen optimization that can eliminate allocations that don't escape. |

### Search terms

`emit_box`, `emit_unbox`, `jl_box_`, `jl_gc_pool_alloc`, `emit_allocobj`,
`gc_alloc_obj`, `gc_small_alloc`


## Julia documentation (fast triage layer)

Before diving into source, check if the issue is a known pitfall:

| Doc file | Covers |
|----------|--------|
| `doc/src/manual/performance-tips.md` | All common performance pitfalls |
| `doc/src/manual/methods.md` | Method dispatch, parametric methods, Vararg |
| `doc/src/devdocs/inference.md` | How type inference works |
| `doc/src/devdocs/ssair.md` | SSA IR format |
| `doc/src/devdocs/EscapeAnalysis.md` | Escape analysis internals |
