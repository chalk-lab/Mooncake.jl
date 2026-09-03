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

`Mooncake.jl` is an automatic differentiation (AD) package written entirely in
Julia. Support for mutation allows it to differentiate most numerical Julia code
without hand-written rules. Unsupported operations generally fail explicitly; the
[known limitations](https://chalk-lab.github.io/Mooncake.jl/stable/known_limitations/)
describe exceptions and validity boundaries.

See the [documentation](https://chalk-lab.github.io/Mooncake.jl/stable) for a fuller
introduction.

> [!NOTE]
> **Performance varies by workload.** On one system, [Flux
> benchmarks](test/integration_testing/flux/README.md) found that cached Mooncake
> gradient evaluations were 2.03 times faster than Zygote on CPU across 19 models and
> comparable on GPU. First evaluations were substantially slower.
> [DynamicPPL benchmarks](https://github.com/TuringLang/DynamicPPL.jl/blob/ca32f3a05f8f866f51ee35dd1bc81ecd75876033/benchmarks/posteriordb.md)
> covered all 147 PosteriorDB posteriors; Mooncake's geometric-mean runtime was 1.32
> times Stan's. See the reports for the methods and complete results.

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

## Project scope

Mooncake is maintained as research software, taking long-lived projects such as R and
TeX as models. It prioritises correctness and stability over broad feature coverage.

Reproducible cases of incorrect results or unexpected failures within the documented
scope guide further work. Rules for operations outside Julia Base, broad redesigns,
general debugging support, and the documented [known
limitations](https://chalk-lab.github.io/Mooncake.jl/stable/known_limitations/) are not
part of the current programme of work.

## Licensing

Mooncake is licensed under the [MIT License](LICENSE). Its required and optional
dependencies are licensed separately and may impose additional terms on redistributed
applications or binaries. See [`Project.toml`](Project.toml) for the dependency list.
