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

Mooncake.jl is an automatic differentiation (AD) package written entirely in Julia, with
both reverse and forward modes. Two features define its approach. First, it
differentiates Julia's type-inferred, optimised intermediate representation (IR), which
helps it produce efficient derivative code (see [Performance](#performance)) and gives
broad coverage of Julia's syntax. Second, it supports mutation, so most numerical Julia
code can be differentiated without user-written rules. The aim is a general-purpose AD
compiler for Julia. Some limits remain: foreign calls (`ccall`) may still need rules,
and reverse mode does not support threaded code. See the
[known limitations](https://chalk-lab.github.io/Mooncake.jl/stable/known_limitations/)
and the [documentation](https://chalk-lab.github.io/Mooncake.jl/stable) for details.

## Getting started

For repeated evaluations at inputs of fixed type and size, prepare a cache and reuse it:

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
value_h, gradient_h, hessian = MC.value_gradient_and_hessian!!(hess_cache, f, x)
```

Preparing a cache takes some time, but later calls that reuse it are fast. Each cache is
tied to the types and sizes of its inputs, and passing an input of a different size
raises an error. When input sizes vary, build a reverse-mode rule with
`Mooncake.build_rrule` instead; it depends only on the input types. The
[tutorial](https://chalk-lab.github.io/Mooncake.jl/stable/tutorial/#Computing-gradients)
walks through an example, and the
[interface documentation](https://chalk-lab.github.io/Mooncake.jl/stable/interface/)
gives details.

## Performance

On one machine, the [Flux benchmarks](test/integration_testing/flux/README.md) found Mooncake's warm
gradient evaluations to be 2.03 times as fast as Zygote's on CPU, by geometric mean over
19 models. On GPU the two were comparable: Mooncake was 1.05 times as fast by geometric
mean, taking between 0.73 and 1.10 times Zygote's time per model. First evaluations were
substantially slower. The
[DynamicPPL benchmarks](https://github.com/TuringLang/DynamicPPL.jl/blob/ca32f3a05f8f866f51ee35dd1bc81ecd75876033/benchmarks/posteriordb.md)
compare Mooncake's gradient times with Stan's across 147 PosteriorDB posteriors.
Performance varies by workload.

## Design

Mooncake has its own representation of derivatives. Each primal type has a unique
tangent type and, to support mutation, each piece of mutable memory has a single
location for its derivative, shared by every reference to it. This meant writing a new
set of differentiation rules, which Mooncake maintains itself. Tangent types are
computed from the types Julia infers, or from runtime types where inference is
imprecise. Types with nothing to differentiate, such as integers, symbols and structs
built only from them, get `NoTangent`.

Some AD tools reconstruct types from lower-level code, or analyse data flow to decide
which values are active. Exact versions of both problems are undecidable in general, as
is the may-alias analysis they rest on
([Landi, 1992](https://doi.org/10.1145/161494.161501)). Such tools therefore face a
trade-off: no analysis can, on every program, be both sound (never missing an active
value, which can silently give incorrect derivatives) and precise (never including an
inactive one, which degrades performance). Mooncake does not need to solve either
problem: its tangent types follow from Julia's own types, so every non-constant value of
a differentiable type carries a tangent. This is sound by construction, at the cost of
computing some derivatives that turn out to be zero.

GPU kernel launches are foreign calls, which Mooncake cannot trace into. For broadcasts
and mapped reductions such as `sum(f, x)` on CUDA arrays, it instead runs `f` on
forward-mode dual numbers ([`Nfwd`](src/nfwd/Nfwd.jl)) inside the kernel.

## Contributing

Supporting mutation, CUDA and as much of the language as possible makes the project
comparable in ambition to ForwardDiff, ChainRules, ChainRulesCore and Zygote combined.
To keep the project sustainable, the [support policy](SUPPORT_POLICY.md) sets the
supported Julia versions and the scope of differentiation rules; rules beyond Julia Base
are outside Mooncake's official scope.

We take a conservative approach to development, in the spirit of long-lived projects
such as R and TeX: correctness, stability and narrowly scoped fixes come before
expanding the package's scope.

Contributions are most welcome when they fix reproducible problems: incorrect results,
unexpected failures, or behaviour that contradicts the documented scope.

If you want to extend Mooncake's coverage, please do so in a package extension or
separate package; the [support policy](SUPPORT_POLICY.md#scope-of-differentiation-rules)
gives guidance and examples.

## Licensing

Mooncake is licensed under the [MIT License](LICENSE). Its required and optional
dependencies are licensed separately and may impose additional terms on redistributed
applications or binaries. See [`Project.toml`](Project.toml) for the dependency list.
