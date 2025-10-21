# Agent Workflow Notes

This repo now includes a small amount of infrastructure to help automation agents collaborate on loop-optimisation work:

- `notes/loop_optimization_plan.md` — current focus, priority pullbacks to refactor, baseline benchmarks, and checklist for loop-safe pullbacks.
- `notes/motivation_and_discussion.md` — full copy of the Issue #156 discussion for deeper context.
- `notes/loop_analysis_insights.md` — summary of the retired loop matcher prototype so future work can reuse the ideas without resurrecting the file.
- `loop_optimization_profiling/` — benchmarking harness for the Issue #156 kernels:
  - `run.jl` to capture full benchmark/profiling snapshots,
  - `smoke_test.jl` for quick regression checks (gradient + microbenchmark),
  - `dev_version` / `release_version` projects to pin dependencies.

Agents should run `loop_optimization_profiling/smoke_test.jl` after significant refactors to catch correctness or performance regressions early. Keep notes in sync with the active scope so future agents know which areas are in or out of play.
