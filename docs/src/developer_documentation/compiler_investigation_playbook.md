# Compiler Investigation Playbook

This is an 8-step methodology for tracing a Julia compiler boundary performance
issue from symptom to root cause. Each step produces a concrete artifact that
feeds into the next.

The playbook is framework-agnostic and works for any Julia package or user code
that hits a compiler boundary issue.

## Step 1: Classify the symptom

Start with diagnostic output (e.g. `@code_typed`, `@code_llvm`, allocation counts,
benchmark results, or output from Mooncake's [`diagnose_perf`](@ref)) and map the
issue to a category using the decision table below.

Compiler boundary issues are often **cross-category** -- for instance, inference
imprecision can lead to an inlining failure, which causes an SROA failure,
which results in allocations. If the initial category doesn't explain
everything, check adjacent categories.

| Signal | Category |
|--------|----------|
| Concrete callable type widened to `Function` or `Any` in MethodInstance | Specialization widening |
| `Type{T}` widened to `Type` in MethodInstance | Specialization widening |
| Small callee survives optimization, invoke count stays high | Inlining failure |
| Heap allocs in `@code_llvm` but not expected from `@code_typed` | SROA failure |
| `jl_apply_generic` in `@code_llvm`, Union return types in `@code_typed` | Type instability |
| Recompilation on method redefinition, world range mismatch | World age |
| `jl_box_*` in `@code_llvm` without corresponding Julia-level allocation | Boxing |

The artifact from this step is a one-line classification, e.g.:

> "Specialization widening: `typeof(sum)` widened to `Function` in Vararg slot 4"

## Step 2: Pin the Julia version

Construct a source reference for the Julia version under investigation. See
[Version Handling](@ref compiler_version_policy) for details.

```
REF=v1.11.3
BASE=https://raw.githubusercontent.com/JuliaLang/julia/v1.11.3
```

## Step 3: Fast triage against documentation

Before reading compiler source, check if this is a known, documented pitfall.

| Category | Doc to check |
|----------|-------------|
| Specialization widening | `performance-tips.md` -- search for "specializing" |
| Inlining | `performance-tips.md` -- search for "@inline" |
| SROA / allocation | `devdocs/EscapeAnalysis.md` |
| Type instability | `performance-tips.md` -- search for "type stability" |
| World age | `methods.md` -- search for "world age" |
| Boxing | `performance-tips.md` -- search for "abstract containers" |

If the docs fully explain the issue, summarize and stop. Otherwise, proceed.

## Step 4: Fetch entry source files

Look up the category in the [Compiler Source Map](@ref compiler_source_map) and
read the primary files. For large C files (e.g. `gf.c` at ~4000 lines), search
for the specific function of interest rather than reading the whole file.

## Step 5: Find the decision point

In the fetched source, locate the exact predicate that controls the behavior --
the `if` statement or condition that determines whether widening, inlining, or
SROA happens.

What to look for by category:

- **Specialization**: widening predicates in `jl_compilation_sig`
- **Inlining**: cost comparison against `MAX_INLINE_COST`, effect requirements
- **SROA**: escape state checks, `is_load_forwardable`
- **Dispatch**: `MethodMatchInfo` resolution, union splitting thresholds
- **World age**: world range checks, `valid_worlds`, invalidation triggers
- **Boxing**: `emit_box` calls in codegen

!!! warning
    `@code_typed` can be misleading. Calling `@code_typed f(args...)` creates a
    fresh, fully-concrete specialization on demand -- one that runtime dispatch
    may never use. If the issue is specialization widening, `@code_typed` will
    show perfectly optimized code while the actual runtime path uses a widened
    `MethodInstance`. Always cross-check with `Base.specializations(method)` to
    see what the runtime actually compiled.

Document the decision point with its file path, line number, and the condition
that triggers it.

## Step 6: Trace upstream and downstream

From the decision point, trace in both directions:

- **Upstream:** What sets the inputs to this predicate? Follow the chain -- often
  the root cause is 2-3 hops upstream (e.g. lowering -> slotflags -> `called`
  bitmask -> compilation sig).
- **Downstream:** What does the decision affect? Trace through to the observable
  consequence (widened MethodInstance, failed inlining, heap allocation).

The chain should be deep enough that someone unfamiliar with Julia internals can
follow the causal path from user code to compiler behavior.

## Step 7: Construct MWE

Build a minimal Julia-only reproducer that preserves the same compiler predicate
firing. See [MWE Patterns](@ref compiler_mwe_patterns) for templates.

Requirements for a good MWE:

1. No package dependencies -- pure Julia
2. An A/B pair: one case where the predicate fires, one where it doesn't
3. Observable difference (timing, allocation count, `Base.specializations` output)
4. Removing any remaining line breaks the reproduction

## Step 8: Write the root cause report

Synthesize steps 1-7 into a structured writeup:

```
## Symptom
[What was observed]

## Classification
[Category and one-line summary]

## Source path
[The causal chain from Step 6, with file:line references]

## Decision predicate
[The exact condition from Step 5, quoted from source]

## Mechanism
[How the predicate interacts with the user's code]

## Minimal reproducer
[The MWE from Step 7]

## Mitigations
- Fix A: [description] -- Pros: ... Cons: ...
- Fix B: [description] -- Pros: ... Cons: ...

## References
- Julia version and source ref
- Files consulted with URLs
```
