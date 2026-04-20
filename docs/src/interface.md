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
The second part of the gradient is the gradient wrt. `x`; for the type `SimplePair`, its gradient is represented using a `@NamedTuple{x1::Float64, x2::Float64}` wrapped in a `Tangent` object.
The gradient wrt. `x1` can for example be retrieved with `grad[2].fields.x1`.

With `friendly_tangents=true`, gradients are returned in a more readable form:

```@example interface
cache = MC.prepare_gradient_cache(g, x_eval; config=MC.Config(friendly_tangents=true))
val, grad = MC.value_and_gradient!!(cache, g, x_eval)
```
The gradient wrt. `x` is now the NamedTuple `(x1 = 2.0, x2 = 4.0)`.

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

If you want to use forward mode explicitly, the cache from `prepare_derivative_cache` can now
also drive `value_and_gradient!!` for scalar outputs. Mooncake seeds standard-basis directions
internally and evaluates them in chunks:

```@example interface
fcache = MC.prepare_derivative_cache(g, x_eval; config=MC.Config(chunk_size=2))
val, grad = MC.value_and_gradient!!(fcache, g, x_eval)
```

Passing `Config(chunk_size=2)` caps the forward chunk width used by this public cache path
when it dispatches to `NfwdMooncake`. If `Nfwd` is not used, changing `chunk_size` is not
useful. Leaving `chunk_size=nothing` keeps Mooncake's default heuristic. Cache
construction stays passive, but a later `value_and_gradient!!` or
`value_and_derivative!!` call may still fail at runtime if `nfwd` turns out not to
support the function. In that case, rebuild the cache with `Config(enable_nfwd=false)` to
force the `frule!!` (aka ir-based forward) path instead. `show(cache)` / `repr(cache)`
also report whether the prepared `ForwardCache` is currently using `nfwd`.

When a public cache path dispatches to `NfwdMooncake`, `value_and_gradient!!` remains the
higher-level Mooncake interface. It may need to bridge richer user-facing inputs, such as
custom structs, to the scalar/array/tuple nfwd signatures used internally, and it also
does the usual cache checks and tangent zeroing. That extra interface work adds some
overhead relative to calling `NfwdMooncake.build_rrule(...)(...)` directly on a supported
nfwd signature over `IEEEFloat` / `Complex{<:IEEEFloat}` scalars, dense arrays with those
element types, and tuples thereof.

Separately, the Hessian path exposed by `prepare_hessian_cache` /
`value_gradient_and_hessian!!` uses forward-over-reverse AD over a captured gradient
closure. It does not currently use the public `NfwdMooncake` fast path, even though the
outer layer is forward mode.

## API Reference

```@docs; canonical=true
Mooncake.Config
Mooncake.value_and_derivative!!
Mooncake.value_and_gradient!!(::Mooncake.Cache, f::F, x::Vararg{Any, N}) where {F, N}
Mooncake.value_and_gradient!!(::Mooncake.ForwardCache, f::F, x::Vararg{Any, N}) where {F, N}
Mooncake.value_and_jacobian!!
Mooncake.value_and_pullback!!(::Mooncake.Cache, ȳ, f::F, x::Vararg{Any, N}) where {F, N}
Mooncake.prepare_derivative_cache
Mooncake.prepare_gradient_cache
Mooncake.prepare_pullback_cache
Mooncake.prepare_hvp_cache
Mooncake.value_and_hvp!!
Mooncake.prepare_hessian_cache
Mooncake.value_gradient_and_hessian!!
```
