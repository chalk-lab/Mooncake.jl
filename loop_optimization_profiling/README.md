# Loop Optimization Profiling

Reproducing benchmarks from [Issue #156](https://github.com/chalk-lab/Mooncake.jl/issues/156)

## Structure

```
loop_optimization_profiling/
├── benchmarks.jl        # Test function definitions
├── run.jl              # Runner script
├── release_version/    # Release Mooncake environment
├── dev_version/        # Local Mooncake environment
└── results/            # Timestamped results
```

## Usage

Run release version:
```bash
julia +1.11 --project=release_version run.jl
```

Run dev version:
```bash
julia +1.11 --project=dev_version run.jl
```

## Output

Results saved to `results/<env>_<timestamp>/`:
- `summary.txt` - Benchmark results
- `profile_<benchmark>.txt` - Profile data for each test

## Benchmarks

1. `sum_1000` - Built-in sum (expected: ~100% bookkeeping overhead)
2. `_sum_1000` - Manual loop sum (expected: high bookkeeping overhead)
3. `map_sin_cos_exp` - Map operation (expected: ~50% bookkeeping)
4. `turing` - Turing model (expected: ~43% bookkeeping)
