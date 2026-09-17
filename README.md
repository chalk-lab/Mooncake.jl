<div align="center">
  
<img src="https://github.com/user-attachments/assets/8b43b8d6-bff1-42bd-9e04-68b9ae8ff362" alt="Mooncake logo" width="300">

# Mooncake.jl

[![Build Status](https://github.com/chalk-lab/Mooncake.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chalk-lab/Mooncake.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/chalk-lab/Mooncake.jl/graph/badge.svg?token=NUPWTB4IAP)](https://codecov.io/github/chalk-lab/Mooncake.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://chalk-lab.github.io/Mooncake.jl/stable)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

</div>

The goal of the `Mooncake.jl` project is to produce an automatic differentiation (AD)
package written entirely in Julia that improves on `ForwardDiff.jl`, `ReverseDiff.jl`,
and `Zygote.jl` in several ways.
Applying AD to Julia's type-inferred, optimised intermediate representation helps
produce efficient derivative code.
Support for mutation allows Mooncake to differentiate a wide range of numerical Julia code
without hand-written rules.
See the [documentation](https://chalk-lab.github.io/Mooncake.jl/stable) for details.

## Performance

On one system, [Flux benchmarks](test/integration_testing/flux/README.md) found Mooncake gradient evaluations after warm-up were 2.03 times as fast as Zygote on CPU across 19 models, with comparable GPU performance. First evaluations were substantially slower. See also the [DynamicPPL benchmarks](https://github.com/TuringLang/DynamicPPL.jl/blob/ca32f3a05f8f866f51ee35dd1bc81ecd75876033/benchmarks/posteriordb.md). **Performance varies by workload.** 

## Getting started

Check whether Mooncake's [support policy](SUPPORT_POLICY.md) covers your Julia version.

Mooncake uses reusable caches for repeated gradient and Hessian evaluations:

```julia
import Mooncake as MC

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2  # Rosenbrock
x = [1.2, 1.2]

# Reverse mode
grad_cache = MC.prepare_gradient_cache(f, x);
value, (_, gradient) = MC.value_and_gradient!!(grad_cache, f, x)

# Forward mode
fwd_cache = MC.prepare_derivative_cache(f, x);
value_fwd, (_, gradient_fwd) = MC.value_and_gradient!!(fwd_cache, f, x)

# Hessian
hess_cache = MC.prepare_hessian_cache(f, x);
value, gradient, hessian = MC.value_gradient_and_hessian!!(hess_cache, f, x)
```

Cache preparation takes some time, but calls that reuse the cache are fast. Each cache
is tied to its inputs' types and sizes; passing a differently sized input raises an
error. See the
[tutorial](https://chalk-lab.github.io/Mooncake.jl/stable/tutorial/#Computing-gradients)
for a walkthrough and the
[interface](https://chalk-lab.github.io/Mooncake.jl/stable/interface/) for details.

## Contributing

The `Mooncake.jl` team has limited capacity for triage and review. In the spirit of
long-lived projects such as R and TeX, we favour correctness, stability,
and tightly scoped fixes over open-ended expansion.

Contributions are most welcome when they concern reproducible defects:
incorrect results, unexpected failures, or behaviour at odds with the
documented scope. See the [support policy](SUPPORT_POLICY.md) for details.

## Licensing

Mooncake is licensed under the [MIT License](LICENSE). Its required and optional
dependencies are licensed separately and may impose additional terms on redistributed
applications or binaries. See [`Project.toml`](Project.toml) for the dependency list.
