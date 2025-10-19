# Mooncake.jl

Documentation for Mooncake.jl is on its way!

!!! details "Documentation Updates"
    Note (03/10/2024): Various bits of utility functionality are now carefully documented. This
    includes how to change the code which Mooncake sees, declare that the derivative of a
    function is zero, make use of existing `ChainRules.rrule`s to quickly create new rules in
    Mooncake, and more.

    Note (02/07/2024): The first round of documentation has arrived.
    This is largely targeted at those who are interested in contributing to Mooncake.jl -- you can find this work in the "Understanding Mooncake.jl" section of the docs.
    There is more to do, but it should be sufficient to understand how AD works in principle, and the core abstractions underlying Mooncake.jl.

    Note (29/05/2024): I (Will) am currently actively working on the documentation.
    It will be merged in chunks over the next month or so as good first drafts of sections are completed.
    Please don't be alarmed that not all of it is here!

## Getting Started

Check that you're running a version of Julia that Mooncake.jl supports.
See the [`SUPPORT_POLICY.md`](https://github.com/chalk-lab/Mooncake.jl/blob/main/SUPPORT_POLICY.md) for more info.

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

## Project Goals

Below the four central objectives of this project are outlined, and the approaches that we take to achieve them.

### Coverage of more of the Language

The need for first-class support for mutation has been well understood in the Julia AD community for a number of years now.
Its primary benefit is the ability to differentiate through the truly vast quantity of mutating code on which users depend in `Base` / `Core`, the standard libraries, and packages in the general registry -- empowering users to AD through code which _they_ write in a mutating way is often of secondary importance.
Thus you should equate `rrule!!`'s support for mutation with good support for existing code.
Conversely, you should equate `Zygote.jl`'s / `ReverseDiff.jl`'s patchy support for mutation with patchy support for existing code.

`rrule!!`s impose no constraints on the types which can be operated on, as with `ChainRules`'s `rrule` and `Zygote`'s `_pullback`.
Consequently, there is in principle nothing to prevent `Mooncake.jl` from operating on any type, for example, structured arrays, GPU arrays, and complicated `struct`s / `mutable struct`s.


### Correctness and Testing

Mutation support enables writing `rrule!!`s at a low-level (`Core.InstrincFunction`s, `built-in function`s, `ccall`s to e.g. `BLAS` and `LAPACK` and the bits of Base Julia which are implemented in C).
The simplicity of this low-level functionality makes implementing correct `rrule!!`s for it a simple task.
In conjunction with being strict about the types used internally to represent (co)tangents, we have found this leads to `rrule!!`s composing very well, and AD being correct in practice as a consequence.

Furthermore, the interfaces for `rrule!!` and the (co)tangent type system have been designed to make a reliable `test_rule` function possible to create.
All of our testing is implemented via this (or via another function which calls this) which makes adding test-cases trivial, and has meant that we have produced a lot of test cases.

This contrasts with `Zygote.jl` / `ChainRules.jl`, where the permissive (co)tangent type system complicates both composition of `rrule`s and testing.

Additionally, our approach to AD naturally handles control flow which differs between multiple calls to the same function.
This contrasts with e.g. `ReverseDiff.jl`'s compiled tape, which can give silent numerical errors if control flow ought to differ between gradient evaluations at different arguments.

### Performance

Hand-written `rrule!!`s have excellent performance, provided that they have been written well (most of the hand-written rules in `Mooncake.jl` have excellent performance, but some require optimisation. Doing this just requires investing some time).
Consequently, whether or not the overall AD system has good performance is largely a question of how much overhead is associated to the mechanism by which hand-written `rrules!!`s are algorithmically composed.

At present (03/2024), we do this in a _reasonably_ performant way (but better than the previous way!)
See [Project Status](@ref) below for more info.

Additionally, the strategy of immediately incrementing (co)tangents resolves long-standing performance issues associated with indexing arrays.

### Written entirely in Julia

`Mooncake.jl` is written entirely in Julia.
This sits in contrast to `Enzyme.jl`, which targets LLVM and is primarily written in C++.
These two approaches entail different tradeoffs.

## Project Name

Before an initial release, this package was called `Taped.jl`, but that name ceased to be helpful when we stopped using a classic "Wengert list"-style type to implement AD.
For about 48 hours is was called `Phi.jl`, but the community guidelines state that the name of packages in the general registry should generally be at least 5 characters in length.

We then chose `Tapir.jl`, and didn't initially feel that other work [of the same name](https://github.com/wsmoses/Tapir-LLVM) presented a serious name clash, as it isn't AD-specific or a Julia project.
As it turns out, there has been significant work attempting to integrate the ideas from this work into the [Julia compiler](https://github.com/JuliaLang/julia/pull/39773), so the clash is something of a problem.

On 18/09/2024 this package was renamed from `Tapir.jl` to `Mooncake.jl`.
The last version while the package was called `Tapir.jl` was 0.2.51.
Upon renaming, the version was bumped to 0.3.0.
We finally settled on `Mooncake.jl`. Hopefully this name will stick.

## Project Status

The plan is to proceed in three phases:
1. design, correctness and testing
1. performance optimisation
1. maintenance

You should take this with a pinch of salt, as it seems highly likely that we will have to revisit some design choices when optimising performance -- we do not, however, anticipate requiring major re-writes to the design as part of performance optimisation.
We aim to reach the maintenance phase of the project before 01/06/2024.

!!! details "Updates"
    *Update: (07/02/2025)*
    We're largely in phase 3 now.
    We're largely working on documentation, and resolving existing issues.
    There are several ways that we _could_ improve the performance of Mooncake.jl on e.g. low-level loops, but our feeling is that the performance is generally good _enough_ to mean that it's more important to ensure that Mooncake is very stable.

    *Update: (30/04/2024)*
    Phase 2 continues!
    We are now finding that `Mooncake.jl` comfortably outperforms compiled `ReverseDiff.jl` on type-stable code in all of the situations we have tested.
    Optimising to get similar performance to `Enzyme.jl` is an on-going process.

    *Update: (22/03/2024)*
    Phase 2 is now further along.
    `Mooncake.jl` now uses something which could reasonably be described as a source-to-source system to perform AD.
    At present the performance of this system is not as good as that of Enzyme, but often beats compiled ReverseDiff, and comfortably beats Zygote in any situations involving dynamic control flow.
    The present focus is on dealing with some remaining performance limitations that should make `Mooncake.jl`'s performance much closer to that of Enzyme, and consistently beat ReverseDiff on a range of benchmarks.
    Fortunately, dealing with these performance limitations necessitates simplifying the internals substantially.

    *Update: (16/01/2024)*
    Phase 2 is now well underway. We now make use of a much faster approach to interpreting / executing Julia code, which yields performance that is comparable with ReverseDiff (when things go well). The current focus is on ironing out performance issues, and simplifying the implementation.

    *Update: (06/11/2023)*
    We are mostly through the first phase.
    Correctness testing is proceeding well, and we are ironing out known issues.
    Notably, our broad experience at present is that as we continue to increase the amount of Julia code on which the package is tested, things fail for known, predictable, straightforwardly-fixable reasons (largely missing rrules for `ccall`s), rather than unanticipated problems.
    Please note that, since we have yet to enter phase 2 of the project, we have spent _no_ time whatsoever optimising for performance.
    We strongly believe that there is nothing in principle preventing us from achieving excellent performance.
    However, currently, you should expect to experience _amazingly_ poor performance.

### What things should work well

Noteworthy things which should work and be performant include:
1. code containing control flow
1. value-dependent control flow
1. mutation of arrays and mutable structs

These are noteworthy in the sense that they are different from ReverseDiff / Zygote.
Enzyme is also able to do these things.

Please be aware that by "performant" we mean similar or better performance than ReverseDiff with a compiled tape, but not as good performance as Enzyme.

### What won't work

See [known limitations](known_limitations.md). 
