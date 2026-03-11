---
name: perf-diagnose
description: Diagnose Julia compiler boundary performance issues (specialization widening, allocation/SROA, inlining failures) in Mooncake.jl function calls. Use when a function is unexpectedly slow but Mooncake's internal IR (from ir-inspect) looks correct.
disable-model-invocation: true
---

# Performance Diagnostics

You are helping the user diagnose performance issues at the Julia compiler boundary in Mooncake.jl. These are problems invisible to Mooncake's internal IR — they occur in the entry-point functions (`value_and_gradient!!`, `value_and_pullback!!`, etc.) where Julia's compiler makes specialization and inlining decisions.

## When to use

- A Mooncake function call is slower than expected
- The Mooncake IR (from `ir-inspect`) looks identical between fast and slow variants
- Passing functions (like `sum`) as arguments causes unexpected overhead
- There are unexpected allocations in what should be allocation-free code

## Setup

The diagnostic script lives at `.claude/skills/perf-diagnose/scripts/perf_diagnose.jl`.

Start a Julia session (or reuse an existing one) and load the script:

```julia
using Revise
includet(".claude/skills/perf-diagnose/scripts/perf_diagnose.jl")
```

## Gathering user intent

Ask the user what they want to diagnose. Offer these choices:

1. **Function + arguments to diagnose** — e.g. `value_and_gradient!!(cache, foo, xs, sum)`
2. **What to check**:
   - Full diagnostic (all detectors)
   - Specialization widening only
   - Allocation/SROA analysis only
   - Inlining failure analysis only
   - Compare two calls (slow vs fast)

Do not assume — ask the user to pick.

## Detectors

| Detector | Function | Catches |
|----------|----------|---------|
| Specialization Widening | `check_specialization` | `typeof(sum)` widened to `Function` in Vararg |
| Allocation/SROA | `check_allocations` | Heap allocations that disappear with inlining |
| Inlining Failure | `check_inlining` | Small callees not inlined after optimization |
| Compare Mode | `compare_perf` | Side-by-side diff of two call variants |

## Commands reference

```julia
# Full diagnostic
report = diagnose_perf(f, args...)
show_report(report)

# Quick inspect + display
report = quick_diagnose(f, args...)

# Individual detectors
spec   = check_specialization(f, args...)
allocs = check_allocations(f, args...)
inl    = check_inlining(f, args...)

show_specialization(spec)
show_allocations(allocs)
show_inlining(inl)

# Compare two calls
comp = compare_perf(f, args_a, args_b)
show_comparison(comp)
```

## How to present results

- Run the Julia commands via Bash and capture output.
- Present the report in fenced code blocks, highlighting findings marked with `[!]`.
- For each finding, explain the severity and the suggested fix.
- When using compare mode, highlight which slots differ and the metrics delta.
- If errors occur, check that InteractiveUtils is loaded and the function signature is valid.
