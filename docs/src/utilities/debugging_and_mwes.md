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

If you have a specific set of arguments that are causing issues, you can test them directly:
```julia
using Random
rng = Xoshiro(123)
Mooncake.TestUtils.test_rule(rng, sin, 5.0)
```

When debugging, it might be helpful to set the `interface_only=true` to skip the correctness tests and just check that the rule runs without error:
```julia
Mooncake.TestUtils.test_rule(rng, sin, 5.0; interface_only=true)
```

## Manually Running a Rule

For more fine-grained debugging, you can manually run `rrule!!` to inspect intermediate values.
Here's an example that differentiates a simple function:

```julia
using Mooncake: rrule!!, zero_fcodual, primal

x = 5.0

# Run the forward pass - returns output CoDual and pullback
# `zero_fcodual(x)` is equivalent to `CoDual(x, fdata(zero_tangent(x)))`.
y, pb!! = rrule!!(zero_fcodual(sin), zero_fcodual(x))

# Seed gradient for scalar output
dy = 1.0

# Run reverse pass - returns input cotangent/adjoint dx
# Note: for scalar outputs, the adjoint is a plain scalar and `pb!!(1.0)` is sufficient.
# More general patterns involving zero_rdata / increment!! apply to non-scalar outputs.
_, dx = pb!!(dy)

# The gradient should be cos(5.0) ≈ 0.28366
isapprox(dx, cos(5.0))
```

This approach lets you:
- Inspect the output of the forward pass `y` and `pb!!` before running the reverse pass
- Set custom seed gradients for the output `dy`
- Examine the computed gradient `dx` in detail

## Segfaults

These are everyone's least favourite kind of problem, and they should be _extremely_ rare in Mooncake.jl.
However, if you are unfortunate enough to encounter one, please re-run your problem with the `debug_mode` kwarg set to `true`.
See [Debug Mode](@ref) for more info.
In general, this will catch problems before they become segfaults, at which point the above strategy for debugging and error reporting should work well.
