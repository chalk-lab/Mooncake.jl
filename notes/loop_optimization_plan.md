# Loop Optimisation Plan (Oct 2025)

## Current Focus
- Standardise hot pullbacks on the `call_pb` singleton interface (no per-iteration closures).
- Keep loop-specialisation experiments parked until explicitly requested.

## High-Priority Conversions
- ✅ `src/rrules/misc.jl`: `lgetfield` mutable + ordered mutable variants (via `call_pb`)
- ✅ `src/rrules/array_legacy.jl`: `arrayset_pullback!!`, `isbits_arrayset_pullback!!`, `copy_pullback!!`
- `src/rrules/builtins.jl`: immutable `getfield` fast paths
- `src/rrules/memory.jl`: `lgetfield` / `getfield` for `Memory` / `MemoryRef`

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
- Run `loop_optimization_profiling/smoke_test.jl` after each conversion (gradient + benchmark).
- Update the “High-Priority Conversions” list as items land or new hotspots appear.
