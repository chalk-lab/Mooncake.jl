# Interface

This is the public interface that day-to-day users of AD are expected to interact with if
for some reason DifferentiationInterface.jl does not suffice.
If you have not tried using Mooncake.jl via DifferentiationInterface.jl, please do so.
See [Tutorial](@ref) for more info.

## Example

Here's a simple example demonstrating how to use Mooncake.jl's native API:

```@example interface
import Mooncake as MC

struct SimplePair
    x1::Float64
    x2::Float64
end

# Define a simple function
g(x::SimplePair) = x.x1^2 + x.x2^2

# Where to evaluate the derivative
x_eval = SimplePair(1.0, 2.0)
```

With `friendly_tangents = false` (the default), gradients for custom structures use a representation based on `Mooncake.Tangent` types.
See [Mooncake.jl's Rule System](@ref) for more information.

```@example interface
cache = MC.prepare_gradient_cache(g, x_eval)
val, grad = MC.value_and_gradient!!(cache, g, x_eval)
```
This produces a tuple containing the value of the function (here `5.0`) and the gradient.
The first part of the gradient is the gradient wrt. `g` itself, here `NoTangent()` since `g` is not differentiable.
The second part of the gradient is the gradient wrt. `x`;
for the type `SimpePair`, its gradient is represented using
a `@NamedTuple{x1::Float64, x2::Float64}` wrapped in a `Tangent` object.
The gradient wrt. `x1` can for example be retrieved with `grad[2].fields.x1`.

With `friendly_tangents=true`, gradients use the same types as the original function:

```@example interface
cache = MC.prepare_gradient_cache(g, x_eval; config=MC.Config(friendly_tangents=true))
val, grad = MC.value_and_gradient!!(cache, g, x_eval)
```
The gradient wrt. `x` is now `SimplePair(2.0, 4.0)`.

In addition, there is an optional tuple-typed argument `args_to_zero` that specifies
a true/false value for each argument (e.g., `g`, `x_eval`), allowing tangent
zeroing to be skipped on a per-argument basis when the value is constant. 
Note that the first true/false entry specifies whether to zero the tangent of `g`;
zeroing `g`'s tangent is not always necessary, but is sometimes required for
non-constant callable objects.

```@example interface
cache = MC.prepare_gradient_cache(g, x_eval; config=MC.Config(friendly_tangents=true))
val, grad = MC.value_and_gradient!!(
    cache,
    g,
    x_eval;
    args_to_zero = (false, true),
)
```

Aside: Any performance impact from using `friendly_tangents = true` should be very minor.
If it is noticeable, something is likely wrong—please open an issue.

## API Reference

```@docs; canonical=true
Mooncake.Config
Mooncake.value_and_derivative!!
Mooncake.value_and_gradient!!(::Mooncake.Cache, f::F, x::Vararg{Any, N}) where {F, N}
Mooncake.value_and_pullback!!(::Mooncake.Cache, ȳ, f::F, x::Vararg{Any, N}) where {F, N}
Mooncake.prepare_derivative_cache
Mooncake.prepare_gradient_cache
Mooncake.prepare_pullback_cache
```
