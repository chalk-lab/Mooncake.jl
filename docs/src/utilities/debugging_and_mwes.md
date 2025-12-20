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
# Example with specific arguments
using Random
rng = Xoshiro(123)
Mooncake.TestUtils.test_rule(rng, sin, 5.0)
```

When debugging, it might be helpful to set the `interface_only` kwarg to `true` in order to skip the correctness tests and just check that the rule runs without error:
```julia
Mooncake.TestUtils.test_rule(rng, sin, 5.0; interface_only=true)
```

## Manually Running a Rule

For more fine-grained debugging, you can manually run `rrule!!` to inspect intermediate values.
Here's an example that differentiates a simple function:

```julia
using Mooncake: rrule!!, zero_rdata, increment!!

# A simple function to differentiate
x = 5.0

# Run the forward pass - returns output CoDual and pullback
# `zero_fcodual(x)` is equivalent to `CoDual(x, fdata(zero_tangent(x)))`.
y, pb!! = rrule!!(zero_fcodual(sin), zero_fcodual(x))

# Set seed gradient (output adjoint) to 1.0
dy = zero_rdata(y)
dy = increment!!(dy, 1.0)

# Run reverse pass - returns input cotangent/adjoint dx
_, dx = pb!!(dy)

# The gradient should be cos(5.0) ≈ 0.28366
isapprox(dx, cos(5.0))
```

This approach lets you:
- Inspect the output of the forward pass `y` and `pb!!` before running the reverse pass
- Set custom seed gradients for the output `dy`
- Examine the computed gradient `dx` in detail

### Multiple-argument example (vector inputs)

```julia
using LinearAlgebra
using Mooncake: build_rrule, zero_fcodual

# Function of two vector arguments
f(x1, x2) = dot(x1, x2)

x1 = [1.0, 2.0, 3.0]
x2 = [4.0, 5.0, 6.0]

# Build and capture the rrule
rule = build_rrule(f, x1, x2)

# Forward pass via the built rule
y, pb!! = rule(
    zero_fcodual(f),
    zero_fcodual(x1),
    zero_fcodual(x2),
)

# Scalar output ⇒ seed gradient directly
dy = 1.0

# Reverse pass: propagate adjoints
_, dx1, dx2 = pb!!(dy)

# Expected gradients:
# ∂y/∂x1 = x2
# ∂y/∂x2 = x1
dx1 == x2
dx2 == x1
```

## Segfaults

These are everyone's least favourite kind of problem, and they should be _extremely_ rare in Mooncake.jl.
However, if you are unfortunate enough to encounter one, please re-run your problem with the `debug_mode` kwarg set to `true`.
See [Debug Mode](@ref) for more info.
In general, this will catch problems before they become segfaults, at which point the above strategy for debugging and error reporting should work well.
