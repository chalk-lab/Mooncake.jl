# Mooncake.jl

[![Build Status](https://github.com/compintell/Mooncake.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/compintell/Mooncake.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/github/compintell/Mooncake.jl/graph/badge.svg?token=NUPWTB4IAP)](https://codecov.io/github/compintell/Mooncake.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![](https://img.shields.io/badge/docs-blue.svg)](https://compintell.github.io/Mooncake.jl/dev)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

The goal of the `Mooncake.jl` project is to produce a reverse-mode AD package which is written entirely in Julia, which improves over both `ReverseDiff.jl` and `Zygote.jl` in several ways, and is competitive with `Enzyme.jl`.
Please refer to [the docs](https://compintell.github.io/Mooncake.jl/dev) for more info.

# Getting Started

Check that you're running a version of Julia that Mooncake.jl supports.
See the `SUPPORT_POLICY.md` file for more info.

There are several ways to interact with `Mooncake.jl`.
The way that we recommend people to interact with `Mooncake.jl` is via  [`DifferentiationInterface.jl`](https://github.com/gdalle/DifferentiationInterface.jl/).
For example, use it as follows to compute the gradient of a function mapping a `Vector{Float64}` to `Float64`.
```julia
using DifferentiationInterface
import Mooncake

f(x) = sum(cos, x)
backend = AutoMooncake(; config=nothing)
x = ones(1_000)
prep = prepare_gradient(f, backend, x)
gradient(f, prep, backend, x)
```
You should expect that `prepare_gradient` takes a little bit of time to run, but that `gradient` is fast.

We are committed to ensuring support for DifferentiationInterface, which is why we recommend using that.
If you are interested in interacting in a more direct fashion with `Mooncake.jl`, you should consider `Mooncake.value_and_gradient!!`.
See its docstring for more info.
