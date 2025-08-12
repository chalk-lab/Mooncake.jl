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

There are several ways to interact with `Mooncake.jl`.
We recommend that people interact with `Mooncake.jl` via  [`DifferentiationInterface.jl`](https://github.com/gdalle/DifferentiationInterface.jl/).
For example, use it as follows to compute the gradient of a function mapping a `Vector{Float64}` to `Float64`.
```julia
using DifferentiationInterface
import Mooncake

f(x) = sum(cos, x)
backend = AutoMooncake()
x = ones(1_000)
prep = prepare_gradient(f, backend, x)
gradient(f, prep, backend, x)
```
You should expect that `prepare_gradient` takes a little bit of time to run, but that `gradient` is fast.

We are committed to ensuring support for DifferentiationInterface, which is why we recommend using that.
If you are interested in interacting more directly with `Mooncake.jl`, you should consider `Mooncake.value_and_gradient!!`.
See its docstring for more info.
