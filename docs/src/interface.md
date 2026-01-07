# Interface

This is the public interface that day-to-day users of AD are expected to interact with if
for some reason DifferentiationInterface.jl does not suffice.
If you have not tried using Mooncake.jl via DifferentiationInterface.jl, please do so.
See [Tutorial](@ref) for more info.

## Example

Here's a simple example demonstrating how to use Mooncake.jl's native API:

```@example interface
import Mooncake as MC

# Define a simple function
g(x) = sum(abs2, x)

# Function with complex numbers
x_complex = [1.0 + 2.0im, 3.0 + 4.0im]
```

With `friendly_tangents=true`, gradients use the same types as inputs:

```@example interface
cache = MC.prepare_gradient_cache(g, x_complex; friendly_tangents=true)
val, grad = MC.value_and_gradient!!(cache, g, x_complex)
```

With `friendly_tangents=false`, gradients use the same types as tangents:

```@example interface
cache = MC.prepare_gradient_cache(g, x_complex; friendly_tangents=false)
val, grad = MC.value_and_gradient!!(cache, g, x_complex)
```

`args_to_zero` is optional; when provided, it specifies a true/false value
for each argument (e.g., `g`, `x_complex`), allowing tangent zeroing to be skipped
per argument when the value is constant:

```@example interface
cache = MC.prepare_gradient_cache(g, x_complex; friendly_tangents = true)
val, grad = MC.value_and_gradient!!(
    cache,
    g,
    x_complex;
    args_to_zero = (false, true),
)
```

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
