# Interface

This page covers Mooncake.jl's public API in detail, including options that go
beyond the [Tutorial](@ref): friendly tangents for `struct`s, per-argument tangent zeroing
via `args_to_zero`, and the full set of prepare/run docstrings.

## Example

Here's a simple example demonstrating how to use Mooncake.jl's API:

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
zeroing to be skipped on a per-argument basis. A `false` entry means that argument's
cotangent is not reset when the cache is reused, so stale values from the previous
call can silently corrupt gradients — including those of *other* arguments, since
reverse-mode rules propagate cotangents between them. Passing `false` is therefore
only guaranteed safe for arguments that carry no differentiable data
(`tangent_type(typeof(arg)) === NoTangent`); an argument being conceptually constant
is *not* sufficient (see
[issue #1238](https://github.com/chalk-lab/Mooncake.jl/issues/1238)).
The first entry corresponds to `g` itself: `false` is safe below because `g` is a
plain function with no fields, but a closure capturing differentiable data must use
`true`.

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

## Jacobian example

For a vector-valued function of a single dense vector input, `value_and_jacobian!!`
returns the primal output together with a dense Jacobian whose columns correspond to
input coordinates.

```jldoctest
julia> using Mooncake

julia> f(x) = [x[1]^2 + x[2], x[1] * x[2]]
f (generic function with 1 method)

julia> x = [2.0, 3.0];

julia> cache = Mooncake.prepare_derivative_cache(f, x);

julia> Mooncake.value_and_jacobian!!(cache, f, x)
([7.0, 6.0], [4.0 1.0; 3.0 2.0])
```

## Reusing a cache, and varying input sizes

A prepared cache preallocates its gradient buffers, and `value_and_gradient!!` writes into them
in place on every call. That in-place reuse is what makes repeated calls fast — the returned
gradient *is* the cache's own buffer, so a later call overwrites it; take a copy if you need to
keep a result:

```@meta
DocTestSetup = quote
    using Mooncake: NoTangent
end
```

```jldoctest interface-varying
julia> using Mooncake

julia> f(x) = sum(abs2, x);

julia> cache = Mooncake.prepare_gradient_cache(f, [1.0, 2.0, 3.0]);

julia> val, grad = Mooncake.value_and_gradient!!(cache, f, [1.0, 2.0, 3.0])
(14.0, (NoTangent(), [2.0, 4.0, 6.0]))
```

Calling again on a new same-shaped input overwrites that earlier `grad`:

```jldoctest interface-varying
julia> _, grad2 = Mooncake.value_and_gradient!!(cache, f, [4.0, 5.0, 6.0]);

julia> grad2[2], grad[2] === grad2[2]   # new gradient; same buffer, mutated in place
([8.0, 10.0, 12.0], true)
```

Because a cache is bound to the type and size of each input, reusing it with a differently sized
input raises an error. When your input sizes vary, skip the cache and build a reusable `rule` once;
the rule depends only on the input *types*, so one rule handles every size:

```jldoctest interface-varying
julia> rule = Mooncake.build_rrule(f, [1.0, 2.0, 3.0]);   # depends on input type, not size

julia> Mooncake.value_and_gradient!!(rule, f, [1.0, 2.0])   # one rule, any length
(5.0, (NoTangent(), [2.0, 4.0]))
```

```@meta
DocTestSetup = nothing
```

Reusing just the rule allocates fresh gradient buffers on each call, which a prepared cache avoids.

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
