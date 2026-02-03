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

The goal of the `Mooncake.jl` project is to produce an AD package which is written entirely in Julia, which improves over `ForwardDiff.jl`, `ReverseDiff.jl` and `Zygote.jl` in several ways, and is competitive with `Enzyme.jl`.
Please refer to [the docs](https://chalk-lab.github.io/Mooncake.jl/dev) for more info.

## Getting Started

Check that you're running a version of Julia that Mooncake.jl supports.
See the `SUPPORT_POLICY.md` file for more info.

There are several ways to interact with `Mooncake.jl`. To interact directly with `Mooncake.jl`, use `Mooncake.value_and_gradient!!`, which exposes the native API and allows reuse of a prepared gradient cache. For example, it can be used to compute the gradient of a function mapping a `Vector{ComplexF64}` to a `Float64`.

```julia
import Mooncake as MC

# f : ℂⁿ → ℝ
f(x) = sum(abs2, x)
x = [1.0 + 2.0im, 3.0 + 4.0im]

cache = MC.prepare_gradient_cache(f, x)
val, grad = MC.value_and_gradient!!(cache, f, x)
```

You should expect that `MC.prepare_gradient_cache` takes a little bit of time to run, but that `MC.value_and_gradient!!` is fast. For additional details, see the [interface docs](https://chalk-lab.github.io/Mooncake.jl/stable/interface/). You can also interact with `Mooncake.jl` via  [`DifferentiationInterface.jl`](https://github.com/gdalle/DifferentiationInterface.jl/).

```julia
import DifferentiationInterface as DI

# Gradient
backend = DI.AutoMooncake()
grad_cache = DI.prepare_gradient(f, backend, x)
g = DI.gradient(f, grad_cache, backend, x)

# Hessian (forward-over-reverse)
hess_backend = DI.SecondOrder(
    DI.AutoMooncakeForward(),
    DI.AutoMooncake()
)
hess_cache = DI.prepare_hessian(f, hess_backend, x);
H = DI.hessian(f, hess_cache, hess_backend, x)
```

We generally recommend interacting with `Mooncake.jl` through `DifferentiationInterface.jl`, although this interface may lag behind Mooncake in supporting newly introduced features.
