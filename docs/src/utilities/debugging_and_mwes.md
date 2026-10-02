# Debugging and MWEs

There's a reasonable chance that you'll run into an issue with Mooncake.jl at some point.
In order to debug what is going on when this happens, or to produce an MWE, it is helpful to have a convenient way to run Mooncake.jl on whatever function and arguments you have which are causing problems.

We recommend using Mooncake.jl’s built-in testing utility [`Mooncake.TestUtils.test_rule`](@ref) to generate such test cases.

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

## Pinned references

When finite differences cannot resolve the derivative, use `reference` to pin expected
results. Comparisons name the failing field. For example, pin a reverse array cotangent:

```julia
Mooncake.TestUtils.test_rule(
    rng, sum, [2.0, 3.0]; mode=Mooncake.ReverseMode, is_primitive=false,
    output_tangent=1.0,
    reference=(
        value=5.0,
        deriv=(rvs=(Mooncake.NoRData(), Mooncake.NoRData()), fdata=(nothing, [1.0, 1.0])),
    ),
)
```

A NamedTuple `deriv` must use only `fwd`, `rvs`, and `fdata` keys. Put a
NamedTuple-valued expected derivative under the appropriate mode key. References are
validated when a `TestCase` is constructed; expected-value functions run at test time.
An `fdata` function also runs during validation to check which positions it pins.
Fields excluded by the test case's `mode` are refused, while `TEST_MODE` may filter checks.

`deriv.fwd` pins the output JVP; `deriv.rvs` pins the pullback's returned tuple;
`deriv.fdata` pins argument cotangent storage after the pullback, including the function
position. A reverse reference must pin the entire cotangent: `rvs` and every argument
with fdata. Use `nothing` only at positions without fdata; scalar-only arguments need
no `fdata` entry. Expected values can be zero-argument
functions, for example `value=() -> sum([2.0, 3.0])`.

The default comparator is `isequal`. Use `cmp=Mooncake.TestUtils.isequal_ignoring_signed_zero`
to equate signed zeros while retaining NaN equality, or `cmp=isapprox` with `rtol`/`atol`
inside `reference`. These tolerances are independent of finite-difference tolerances.

For chunked forward rules, `reference.lanes=:inactive_zero` uses the test case's seed in
lane 1 and checks exact zeros in all other output and argument partials. To pin individual
lanes, use `reference=(lanes=((seed=(Mooncake.NoTangent(), 1.0), value=2.0),
(seed=(Mooncake.NoTangent(), 3.0), value=6.0)),)` for `x -> 2x`, with `chunk_size=2`.
Each seed includes the function position. The default width is 8; unsupported lane
representations and mismatched lane counts are refused. Comparisons with width-1 results
use the harness's partials precision tolerance; `cmp` applies to pinned expected values.
With an explicit `chunk_size`, only the pinned reference checks run at width 1, not
the full width-1 battery.

## Second-order registry checks

The internal, unexported `Mooncake.TestUtils.TestCase` also lets a registry test case
exercise forward-over-reverse HVPs and Hessians through `test_rule`:

```julia
tc = Mooncake.TestUtils.TestCase(x -> sum(abs2, x), [1.0, 2.0]; hvp=true)
Mooncake.TestUtils.test_rule(rng, tc)
```

The function must take one argument and return a real scalar. `hvp=true` compares
HVPs with finite differences of the reverse gradient, reuses the cache across two
directions, and checks Hessian assembly for real floating-point vectors.
Use an `hvp` NamedTuple for explicit `directions`, `reference`, `cmp`, `rtol`, or
`atol`; `check=:hvp` omits implicit Hessian assembly, and `check=:reference` checks
only the supplied reference fields. These settings are separate from first-order
options. See the `TestCase` docstring for the complete internal contract.

An HVP test case runs only second order unless `hvp.first_order=true`. Registry
runners execute second order once, in the reverse runner. `TEST_MODE=hvp` selects
only second-order checks; `TEST_MODE=forward` or `reverse` selects only first order.

## Manually Running a Rule

For more fine-grained debugging, you can manually run `rrule!!` to inspect intermediate values.
Here's an example that differentiates a simple function:

```julia
using Mooncake: rrule!!, zero_fcodual

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
