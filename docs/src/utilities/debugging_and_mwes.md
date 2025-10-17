# Debugging and MWEs

There's a reasonable chance that you'll run into an issue with Mooncake.jl at some point.
In order to debug what is going on when this happens, or to produce an MWE, it is helpful to have a convenient way to run Mooncake.jl on whatever function and arguments you have which are causing problems.

We recommend making use of Mooncake.jl's testing functionality to generate your test cases:

```@docs; canonical=false
Mooncake.TestUtils.test_rule
```

This approach is convenient because it can
1. check whether AD runs at all,
1. check whether AD produces the correct answers,
1. check whether AD is performant, and
1. can be used without having to manually generate tangents.

## Example

```@meta
DocTestSetup = quote
    using Random, Mooncake
end
```

For example
```julia
f(x) = Core.bitcast(Float64, x)
Mooncake.TestUtils.test_rule(Random.Xoshiro(123), f, 3; is_primitive=false)
```
will error.
(In this particular case, it is caused by Mooncake.jl preventing you from doing (potentially) unsafe casting. In this particular instance, Mooncake.jl just fails to compile, but in other instances other things can happen.)

In any case, the point here is that `Mooncake.TestUtils.test_rule` provides a convenient way to produce and report an error.

### Testing a Specific Set of Arguments

If you want to run the test suite on a single collection of arguments, you can do something like:
```julia
julia> using LinearAlgebra, Mooncake, Random

julia> fargs = (BLAS.gemm!, 'N', 'N', 1.1, randn(3, 2), randn(2, 1), 0.5, randn(3, 1));

julia> Mooncake.TestUtils.test_rule(Xoshiro(123), fargs...);
Test Summary:                                                                                                            | Pass  Total  Time
Tuple{typeof(LinearAlgebra.BLAS.gemm!), Char, Char, Float64, Matrix{Float64}, Matrix{Float64}, Float64, Matrix{Float64}} |   34     34  0.0s
```

When debugging, it might be helpful to set the `interface_only` kwarg to `test_rule` equal to `true`, in order to avoid running correctness tests:
```julia
julia> Mooncake.TestUtils.test_rule(Xoshiro(123), fargs...; interface_only=true);
```

## Manually Running a Rule

To run a rule manually at the command line, you might do something like the following:
```julia
julia> using LinearAlgebra, Mooncake, Random

julia> using Mooncake: rrule!!, zero_rdata, zero_fcodual, primal

julia> fargs = (BLAS.gemm!, 'N', 'N', 1.1, randn(3, 2), randn(2, 1), 0.5, randn(3, 1));

julia> fargs[1](fargs[2:end]...)
3×1 Matrix{Float64}:
  3.0675765934204846
  0.3241193013431162
 -0.10724921573866608

julia> # Run the rule
       y, rvs = rrule!!(map(zero_fcodual, fargs)...)
(Mooncake.CoDual{Matrix{Float64}, Matrix{Float64}}([3.097836282610327; 1.434872548283475; 0.36413373015673733;;], [0.0; 0.0; 0.0;;]), Mooncake.var"#gemm!_pb!!#344"{Float64, Base.RefValue{Matrix{Float64}}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Float64, Float64, Char, Char}(Base.RefValue{Matrix{Float64}}([1.4126922025789126; 0.8205107411804217; 0.23693699035842655;;]), [3.0877497195470465; 1.064621465970022; 0.20700608152493621;;], [0.0; 0.0; 0.0;;], [3.097836282610327; 1.434872548283475; 0.36413373015673733;;], [0.0; 0.0;;], [1.6096030714358025; 0.01575435958162854;;], [0.0 0.0; 0.0 0.0; 0.0 0.0], [0.8794851649178947 -0.18596885135024463; 0.5096859991842587 0.007527530712906747; 0.14416956239782056 0.3098329633805674], 0.5, 1.1, 'N', 'N'))

julia> # Run the reverse pass
       rvs(zero_rdata(primal(y)))
(Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(), 0.0, Mooncake.NoRData(), Mooncake.NoRData(), 0.0, Mooncake.NoRData())
```

## Segfaults

These are everyone's least favourite kind of problem, and they should be _extremely_ rare in Mooncake.jl.
However, if you are unfortunate enough to encounter one, please re-run your problem with the `debug_mode` kwarg set to `true`.
See [Debug Mode](@ref) for more info.
In general, this will catch problems before they become segfaults, at which point the above strategy for debugging and error reporting should work well.
