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
Julia. Its support for mutation allows Mooncake to differentiate most numerical
Julia code without hand-written rules. Unsupported operations fail explicitly
rather than silently returning an incorrect gradient.

See the [documentation](https://chalk-lab.github.io/Mooncake.jl/stable) for a fuller
introduction.

> [!NOTE]
> **Performance varies by workload.** On one system, [Flux
> benchmarks](test/integration_testing/flux/README.md) found cached Mooncake gradient
> evaluations 2.03 times faster than Zygote on CPU across 19 models and comparable on
> GPU, although first evaluations were substantially slower.
> [DynamicPPL benchmarks](https://github.com/TuringLang/DynamicPPL.jl/blob/main/benchmarks/posteriordb.md)
> covered all 147 PosteriorDB posteriors; Mooncake's geometric-mean runtime was 1.32
> times Stan's. See the reports for the methods and complete results.

## Getting started

Check that your Julia version is covered by Mooncake's [support
policy](SUPPORT_POLICY.md).

Mooncake prepares reusable caches for repeated gradient and Hessian evaluations:

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

Preparing a cache takes some time, but subsequent calls that reuse it are fast. A cache
is tied to each input's type and size; reuse with a differently sized input raises an
error. See the [tutorial](https://chalk-lab.github.io/Mooncake.jl/stable/tutorial/#Computing-gradients)
for a walkthrough and the [interface](https://chalk-lab.github.io/Mooncake.jl/stable/interface/)
for details.

## Contributions and support

In the spirit of long-lived projects such as R and TeX, we favour correctness,
stability, and tightly scoped fixes over open-ended expansion.

We welcome reproducible reports of incorrect results, unexpected failures, or behaviour
at odds with the documented scope. Feature requests, redesign proposals, and debugging
queries without a minimal reproducible example are generally outside the support we can
provide. The same applies to requests for rules beyond Julia Base and behaviour listed
under [known limitations](https://chalk-lab.github.io/Mooncake.jl/stable/known_limitations/).
Such issues will usually be closed.

Accounts involved in spam or abuse will be blocked and reported. Other moderation is
undertaken at our discretion, as capacity permits.
