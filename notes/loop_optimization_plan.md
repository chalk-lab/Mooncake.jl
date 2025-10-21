# Loop Optimisation Plan (Oct 2025)

## Current Focus
- Singleton `call_pb` pullbacks in place; shift attention to designing a general loop/invariant analysis so we can cut bookkeeping without per-pattern hacks.

## Completed Pullback Refactors
- `src/rrules/misc.jl`: mutable `lgetfield`
- `src/rrules/array_legacy.jl`: `arrayref`, `arrayset`, `isbits_arrayset`, `copy`
- `src/rrules/builtins.jl`: immutable `getfield`
- `src/rrules/memory.jl`: `lmemoryrefget` / `memoryrefget`

## Baseline Snapshot (2025-10-21 11:12:17 UTC−07)
- `sum_1000`: 130.096 ns  
- `_sum_1000`: 3.568 μs  
- `map_sin_cos_exp`: 2.162 μs

## Pullback Requirements
1. Stateless across iterations; all dynamic data passed as arguments.
2. Mutate supplied tangent containers in place; avoid allocations.
3. Encode static metadata via argument types (`Val`, constants) instead of captured environment.
4. Return `NoRData()` placeholders for unused adjoints to match interpreter expectations.

## Working Notes
- Keep running `loop_optimization_profiling/smoke_test.jl` while iterating on loop-analysis experiments.
- Next steps: sketch loop detection, invariant propagation, and specialised reverse-loop emission before diving into code.
