# Julia Compiler Source Map

This file maps each category of compiler boundary issue to the Julia source files
that implement the relevant decision logic. All paths are relative to the Julia
repository root.

Use `{BASE}` as a placeholder for the version-pinned raw GitHub URL prefix, e.g.:
```
{BASE} = https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3
```

## 1. Specialization Widening

When Julia compiles a method, it may choose to widen concrete argument types to
more abstract types in the compilation signature. This reduces the number of
compiled specializations but can prevent optimization.

### Primary files

| File | Key functions/symbols | What it does |
|------|----------------------|--------------|
| `src/gf.c` | `jl_compilation_sig` | **The main decision function.** Iterates over arguments and decides whether to widen each one. Contains the `notcalled_func` heuristic that widens Function subtypes. |
| `src/gf.c` | `jl_isa_compileable_sig` | Mirror of `jl_compilation_sig` that validates whether a given signature could be a valid compilation target. |
| `src/method.c` | `jl_method_set_source` (called from `jl_method_def`) | Reads `slotflags` bit 6 (0x40 = "called") from lowered code and sets `m->called` bitmask. This bitmask is what `jl_compilation_sig` checks. |

### Secondary files

| File | What it adds |
|------|-------------|
| `src/typemap.c` | Method cache lookup — how compiled specializations are stored and retrieved. |
| `Compiler/src/typelimits.jl` | Type widening heuristics used during abstract interpretation (different from specialization widening but can compound). |
| `Compiler/src/methodtable.jl` | Julia-side method table operations. |
| `JuliaLowering/src/scope_analysis.jl` | **(Julia 1.12+ only, separate subproject)** Where `is_called` is set during scope analysis. A binding's `is_called` flag is set when a variable appears in call position. On Julia < 1.12, the equivalent logic is in `base/compiler/lowering.jl` or `src/julia-syntax.scm`. |
| `JuliaLowering/src/eval.jl` | **(Julia 1.12+ only)** Where `is_called` is encoded into `slotflags` as bit 6. |

### Key mechanism: the `notcalled_func` heuristic

```c
// src/gf.c, inside jl_compilation_sig, around line 1305
size_t i_arg = (i < nargs - 1 ? i : nargs - 1);  // line ~1188
// ^^^ ALL vararg positions collapse to the last declared parameter's index

int notcalled_func = (i_arg > 0 && i_arg <= 8 &&
    !(definition->called & (1 << (i_arg - 1))) &&  // not marked "called"
    !jl_has_free_typevars(decl_i) &&
    jl_subtype(elt, (jl_value_t*)jl_function_type));  // arg is a Function
if (notcalled_func && (jl_subtype((jl_value_t*)jl_function_type, type_i))) {
    // Widen typeof(sum) → Function
    jl_svecset(*newparams, i, (jl_value_t*)jl_function_type);
}
```

The chain:
1. Lowering sets `is_called=true` when a slot appears in call position `f(x)`
2. This becomes bit 6 of `slotflags`, read by `jl_method_set_source` into `m->called` bitmask (8 bits max)
3. `jl_compilation_sig` checks `called` per PARAMETER index (`i_arg`), not per ARGUMENT index
4. For Vararg methods, all vararg elements map to the same `i_arg` (the last parameter)
5. Vararg parameters can never syntactically appear in call position → the `called` bit is always 0
6. So `notcalled_func` fires for every Function-typed vararg element

### Standard mitigations for specialization widening

There are two well-known approaches. Both achieve equivalent performance in
benchmarks, but differ in mechanism and tradeoffs.

**Approach A: `@inline`** — Sidestep the problem entirely. When the function is
inlined into its caller, no `MethodInstance` is created for it, so
`jl_compilation_sig` never runs and widening never happens. The function body is
spliced into the caller's IR with fully concrete types.

- Pros: minimal code change (one keyword), zero maintenance burden
- Cons: `@inline` is a hint, not a guarantee — the compiler may decline for large
  functions. Does not fix the problem if the function is called from a context
  that prevents inlining. Does not help if the widened `MethodInstance` is created
  by other callers.

**Approach B: `@eval`-generated per-arity methods** — Generate explicit methods
for arities 1..N where each Vararg element becomes a typed parameter
(`x1::X1, x2::X2, ...`). Julia specializes normally on explicit type parameters,
so `typeof(sum)` stays concrete. The original Vararg method remains as a fallback
for arity > N.

- Pros: directly fixes the root cause, works regardless of inlining decisions,
  graceful degradation for high arity
- Cons: more code to maintain, increases method table size and precompilation time,
  requires a macro or manual `@eval` loop

In practice, benchmarks show no reliable performance difference between the two
(initial measurements suggesting one is faster than the other have not been
reproducible). Choose based on context: `@inline` for thin wrappers where
inlining is always desirable; per-arity expansion when the function is too large
to inline or when robustness against future inlining changes matters.

### Search terms

When investigating a specialization issue, search fetched source for:
`jl_compilation_sig`, `notcalled_func`, `iscalled`, `definition->called`,
`nospecialize`, `i_arg`, `very_general_type`, `jl_function_type`,
`jl_is_type_type`, `jl_is_kind`


## 2. Inlining Failures

Julia's inlining pass decides which callees to inline based on a cost model,
effect analysis, and various heuristics.

### Primary files

| File | Key functions/symbols | What it does |
|------|----------------------|--------------|
| `Compiler/src/ssair/inlining.jl` | `resolve_todo`, `analyze_method!`, `handle_single_case!` | The inlining pass. Decides per-callsite whether to inline. |
| `Compiler/src/optimize.jl` | `optimize` (entry point), `run_passes` | Orchestrates all optimization passes including inlining. |

### Secondary files

| File | What it adds |
|------|-------------|
| `Compiler/src/effects.jl` | Effect analysis — `nothrow`, `consistent`, `terminates` etc. Effects constrain what can be inlined. |
| `Compiler/src/stmtinfo.jl` | Statement info types (`MethodMatchInfo`, `UnionSplitInfo`, etc.) that carry dispatch resolution results to the inliner. |
| `Compiler/src/abstractinterpretation.jl` | Where `StmtInfo` is populated during abstract interpretation. The inliner consumes what inference resolved. |

### Search terms

`inline_cost`, `isinlineable`, `@inline`, `@noinline`, `InliningPolicy`,
`MAX_INLINE_COST`, `is_inlineable`, `inlining_policy`, `ssa_inlining_pass!`


## 3. SROA / Allocation Failures

Scalar Replacement of Aggregates (SROA) eliminates heap allocations of structs by
replacing them with their individual fields on the stack. When SROA fails, structs
that should be stack-allocated end up on the heap.

### Primary files

| File | Key functions/symbols | What it does |
|------|----------------------|--------------|
| `Compiler/src/ssair/passes.jl` | `sroa_pass!`, `getfield_elim_pass!`, `adce_pass!` | High-level SROA: replaces `getfield` on known-type allocations with direct SSA values. |
| `Compiler/src/ssair/EscapeAnalysis.jl` | `analyze_escapes`, `EscapeState` | Determines whether a value escapes its defining scope. Non-escaping values can be SROA'd or stack-allocated. |

### Secondary files

| File | What it adds |
|------|-------------|
| `src/llvm-alloc-opt.cpp` | LLVM-level allocation optimization — converts `gc_alloc_obj` to stack allocations when escape analysis proves safety. This is a second chance after Julia-level SROA. |
| `src/llvm-late-gc-lowering.cpp` | GC lowering — where GC roots and frames are emitted. Relevant when investigating GC frame overhead. |

### Search terms

`sroa_pass!`, `getfield_elim_pass!`, `analyze_escapes`, `is_load_forwardable`,
`EscapeState`, `AllocationOptimizer`, `alloc_opt`, `gc_alloc_obj`


## 4. Type Instability / Dynamic Dispatch

When Julia can't infer a concrete return type, it falls back to dynamic dispatch
(`jl_apply_generic`), which is orders of magnitude slower than static dispatch.

### Primary files

| File | Key functions/symbols | What it does |
|------|----------------------|--------------|
| `Compiler/src/abstractinterpretation.jl` | `abstract_call_gf_by_type`, `abstract_call_known`, `abstract_call` | Core of type inference. Resolves method calls to concrete return types (or fails to). |
| `Compiler/src/typelattice.jl` | `widenconst`, `tmerge`, lattice operations | The type lattice that inference operates on. `tmerge` joins types at control-flow merges — this is where precision is lost. |

### Secondary files

| File | What it adds |
|------|-------------|
| `Compiler/src/abstractlattice.jl` | Extended lattice elements (`Conditional`, `PartialStruct`, etc.) that carry extra precision beyond plain types. |
| `Compiler/src/types.jl` | Compiler type definitions: `MethodInstance`, `CodeInstance`, `InferenceResult`, etc. |
| `src/gf.c` | `jl_apply_generic` — the runtime dynamic dispatch entry point. |

### Search terms

`abstract_call_gf_by_type`, `abstract_call_known`, `jl_apply_generic`,
`widenconst`, `tmerge`, `@nospecialize`, `MethodMatchInfo`


## 5. World Age / Method Invalidation

Julia uses world ages to ensure consistency: code compiled at world N only sees
methods defined before world N. When new methods are defined, existing compiled
code may be invalidated.

### Primary files

| File | Key functions/symbols | What it does |
|------|----------------------|--------------|
| `src/method.c` | `jl_method_table_insert` | Where new methods are inserted and invalidation is triggered. |
| `Compiler/src/bindinginvalidations.jl` | `invalidate_method_instance`, `handle_invalidations` | Julia-side invalidation logic. **(Julia 1.12+ only; on < 1.12, invalidation logic is in `base/compiler/typeinfer.jl`.)** |

### Secondary files

| File | What it adds |
|------|-------------|
| `Compiler/src/cicache.jl` | `CodeInstance` cache — how compiled code is stored and looked up by world range. |
| `Compiler/src/typeinfer.jl` | `typeinf` entry point — where inference is triggered (possibly re-triggered after invalidation). |

### Search terms

`world_age`, `min_world`, `max_world`, `invalidate`, `jl_method_table_insert`,
`BackedgePair`, `CodeInstance`, `valid_worlds`


## 6. Boxing / Unexpected Allocations

Boxing occurs when the compiler must convert a value type (e.g. `Float64`) into a
heap-allocated object to pass it through a type-erased interface.

### Primary files

| File | Key functions/symbols | What it does |
|------|----------------------|--------------|
| `src/codegen.cpp` | `emit_expr`, `emit_invoke`, `emit_call` | Main codegen: translates Julia IR to LLVM IR. Boxing happens when a concrete value must be passed as `jl_value_t*`. |
| `src/cgutils.cpp` | `emit_box`, `emit_unbox`, `emit_allocobj` | Helper functions for boxing/unboxing values and allocating objects during codegen. |

### Secondary files

| File | What it adds |
|------|-------------|
| `src/llvm-alloc-opt.cpp` | Post-codegen optimization that can sometimes eliminate allocations (including boxes) that don't escape. |
| `src/llvm-gc-invariant-verifier.cpp` | Verifies GC invariants — useful for understanding GC root requirements that force boxing. |

### Search terms

`emit_box`, `emit_unbox`, `jl_box_float64`, `jl_gc_pool_alloc`,
`emit_allocobj`, `jl_apply_generic`, `gc_preserve`


## Julia documentation files (fast triage layer)

Before diving into source, check if the issue is a known pitfall:

| Doc file | Covers |
|----------|--------|
| `doc/src/manual/performance-tips.md` | All common performance pitfalls including type stability, specialization avoidance, allocation tips |
| `doc/src/manual/methods.md` | Method dispatch, parametric methods, Vararg |
| `doc/src/devdocs/inference.md` | How type inference works (developer-facing) |
| `doc/src/devdocs/ssair.md` | SSA IR format used by the compiler |
| `doc/src/devdocs/EscapeAnalysis.md` | Escape analysis internals |
