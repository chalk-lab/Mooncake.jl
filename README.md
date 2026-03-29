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

The goal of the `Mooncake.jl` project is to produce an AD package written entirely in Julia that improves on `ForwardDiff.jl`, `ReverseDiff.jl`, and `Zygote.jl` in several ways.
Please refer to [the docs](https://chalk-lab.github.io/Mooncake.jl/dev) for more info.

> [!IMPORTANT]
> `Mooncake.jl` accepts issues and pull requests for reproducible defects only. Feature requests, enhancements, redesign proposals, support requests, and debugging requests without a
  minimal reproducible example are out of scope and will be closed. Although Mooncake currently supports a select subset of Julia standard libraries, mathematical libraries, and
  `CUDA.jl`, its intended rule-coverage scope is Julia Base, so requests for missing rules outside Julia Base are out of scope.

## Getting Started

Check that you're running a version of Julia that Mooncake.jl supports.
See the `SUPPORT_POLICY.md` file for more info.

There are several ways to interact with `Mooncake.jl`. To interact directly with `Mooncake.jl`, use Mooncake's native API, which allows reuse of prepared caches for repeated gradient and Hessian evaluations:

```julia
import Mooncake as MC

f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2  # Rosenbrock
x = [1.2, 1.2]

grad_cache = MC.prepare_gradient_cache(f, x);
val, grad = MC.value_and_gradient!!(grad_cache, f, x)

hess_cache = MC.prepare_hessian_cache(f, x);
val, grad, H = MC.value_gradient_and_hessian!!(hess_cache, f, x)
# val  : f(x)
# grad : ∇f(x)  (length-n vector)
# H    : ∇²f(x) (n×n matrix)
```

You should expect that `MC.prepare_gradient_cache` and `MC.prepare_hessian_cache` take a little time to run, but that subsequent calls using the prepared caches are fast.

For additional details, see the [interface docs](https://chalk-lab.github.io/Mooncake.jl/stable/interface/). You can also interact with `Mooncake.jl` via  [`DifferentiationInterface.jl`](https://github.com/gdalle/DifferentiationInterface.jl/), although this interface may lag behind Mooncake in supporting newly introduced features.
