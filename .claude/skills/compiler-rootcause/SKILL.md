---
name: compiler-rootcause
description: >
  Trace Julia compiler-boundary performance issues to an exact source-level
  predicate and causal chain. Use when runtime is unexpectedly slow or
  allocating even though package IR or algorithm looks correct, and the likely
  cause is specialization widening, inlining failure, SROA/allocation failure,
  dynamic dispatch, world age/invalidation, or boxing.
disable-model-invocation: true
---

# Julia Compiler Root Cause Investigation

Trace from symptom to exact compiler decision point. Prefer evidence over
intuition.

## Use this skill when

- `@code_typed` or package IR looks fine, but runtime is still slow or allocates.
- You suspect compiler-boundary behavior (specialization widening, inlining,
  SROA, dispatch, world age, boxing).
- You need a source-level explanation with reproducible evidence.
- You need a framework-free MWE and mitigation options.

If this is still first-pass triage, use `perf-diagnose` first.

## Required workflow

1. Pin Julia version first (`VERSION`, optional commit). See
   `references/version-policy.md`.
2. Classify the symptom and pick the initial category. See
   `references/source-map.md`.
3. Run the full method in `references/investigation-playbook.md`.
4. Build an A/B MWE with `references/mwe-patterns.md`.
5. Write the root-cause report with exact predicate and source path.

## Source access

- Prefer local clone for deep tracing:

```bash
git clone --depth 1 --branch <REF> https://github.com/JuliaLang/julia.git /tmp/julia-src
```

- Or read raw GitHub files using the same `<REF>`.

Never mix versions while tracing.

## Definition of done

Investigation is complete only when all are true:

- Julia `REF` is recorded.
- Exact decision predicate is identified.
- Upstream -> decision -> downstream causal chain is shown.
- MWE reproduces behavior with an A/B comparison.
- At least one mitigation with tradeoffs is given.
- References list exact files and URLs consulted.

## Report format

1. Symptom
2. Classification
3. Source path
4. Decision predicate
5. Mechanism
6. Minimal reproducer
7. Mitigations
8. References
