# Interface

This is the public interface that day-to-day users of AD are expected to interact with if
for some reason DifferentiationInterface.jl does not suffice.
If you have not tried using Mooncake.jl via DifferentiationInterface.jl, please do so.
See [Tutorial](@ref) for more info.

```@example interface
import Mooncake as MC
```

## Example

Here's a simple example demonstrating how to use Mooncake.jl's native API:

```@example interface
# Define a simple function
g(x) = sum(abs2, x)

# Function with complex numbers
x_complex = [1.0 + 2.0im, 3.0 + 4.0im]

# With friendly_tangents=true, gradients use the same types as inputs
cache_friendly = MC.prepare_gradient_cache(g, x_complex; friendly_tangents=true)
val, grad = MC.value_and_gradient!!(cache_friendly, g, x_complex)
```

```julia
# `args_to_zero` is optional; when provided, it specifies a true/false value
# for each argument (e.g., loss, model), allowing tangent zeroing to be skipped
# per argument when the value is constant.
val, grad = MC.value_and_gradient!!(cache_friendly, loss, model, args_to_zero=(false, true))
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
