# Scalar And Low-Dimensional Rules Via `NDual`

For many scalar and low-dimensional primitives, Mooncake uses a two-part strategy:

1. define the local *forward* derivative behavior once on `NDual`, and expose it through `nfwd`, and
1. give reverse mode a small, direct **native** analytic pullback.

Forward mode reuses the `NDual` scalar semantics; reverse mode does **not** run `NDual` at all — it
has zero dependence on the forward-mode machinery.

## Core Idea

If a primitive is fundamentally "a few scalar inputs in, a few scalar outputs out", it is often
better to teach `NDual` how that primitive behaves (for forward mode) and to write a one-line
closed-form pullback (for reverse mode) than to run a general AD engine over it.

In this setup:

- `src/nfwd/Nfwd.jl` owns the scalar **forward** derivative semantics (the `f(::NDual)` overloads), and
- `src/rules/low_level_maths.jl` holds the primitive registrations: a thin `frule!!` that runs the
  `NDual` overload, and a `rrule!!` that applies a native closed-form derivative factor.

Each reverse `rrule!!` writes its closed-form derivative factor inline in a small pullback closure and
applies it to the output cotangent with `_rvs_guarded_scale`, which keeps an inactive (zero-cotangent)
lane exactly zero even where the local derivative is `±Inf` — the reverse analogue of the forward
`_fwd_guarded_scale`. This covers ordinary derivatives, strong-zero behavior, and awkward points such
as discontinuities or removable singularities.

## Concrete MWE

Here is the full pattern for a simple scalar primitive such as `exp(x)`.

The `NDual` method owns the local **forward** derivative behavior. Outside `src/nfwd/Nfwd.jl`,
the internal helper names need to be imported or qualified explicitly:

```julia
const NDual = Mooncake.Nfwd.NDual
const _fwd_scale = Mooncake.Nfwd._fwd_scale

@inline function Base.exp(a::NDual{T,N}) where {T,N}
    return (ev=exp(a.value); NDual{T,N}(ev, _fwd_scale(a.partials, ev)))
end
```

Key details:

- `a.value` is the primal scalar value.
- `a.partials` is the `N`-lane tuple of tangent directions carried by `NDual`.
- `_fwd_scale(a.partials, s)` multiplies every tangent lane by the same local scalar derivative `s`.
- The returned `NDual` therefore contains both the primal `exp(a)` value and the propagated tangent
  lanes. Here the value *is* the derivative factor, so it is computed once and used twice.

The forward `frule!!` stays thin — it just runs that overload. The reverse `rrule!!` is a direct
native pullback that writes the closed-form factor inline, with no `NDual` seeding:

```julia
@is_primitive MinimalCtx Tuple{typeof(exp),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(exp),N}, x::Lifted{P,N,NDual{P,N}}) where {N,P<:IEEEFloat}
    dy = exp(tangent(x))    # the NDual overload runs the primal once, storing it in dy.value
    y = dy.value            # read the primal back — do NOT recompute exp(primal(x))
    return Lifted{_typeof(y),N}(y, dy)
end

function rrule!!(::CoDual{typeof(exp)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = exp(_x)
    exp_pb(ȳ::P) = (NoRData(), _rvs_guarded_scale(ȳ, y))
    return zero_fcodual(y), exp_pb
end
```

The real registrations live in `src/rules/low_level_maths.jl`.

The forward rule runs the primal on the inner `NDual` — the input slot already carries the `N` seeded
tangent lanes, so the `NDual` overload propagates all of them in one evaluation.

The reverse rule is independent: it evaluates the primal directly, writes the closed-form derivative
factor inline, and applies it to the output cotangent through `_rvs_guarded_scale`. It never constructs
an `NDual`, so reverse mode does not depend on the forward-mode `Nfwd` submodule.

`nfwd` only supports scalar leaves it can lift to `NDual` directly, so the forward side of this pattern
fits primitives whose inputs and outputs are a few `IEEEFloat` scalars (or small tuples of them, e.g.
`sincos`); the reverse factors are written by hand for the same signatures.

### The fused exception

A handful of primitives get the value and the derivative factor from one call, and those write the
arithmetic inline in both rules rather than delegating to the `NDual` overload. The clusters are
`sin`/`cos`/`tan`, `sind`/`cosd`/`tand` and `sinpi`/`cospi`, each backed by one `sincos`-family call:

```julia
@inline function Base.cospi(a::NDual{T,N}) where {T,N}
    sv, cv = sincospi(a.value)
    return NDual{T,N}(cv, _fwd_scale(a.partials, -T(π) * sv))
end

function frule!!(
    ::Lifted{typeof(cospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    nd = tangent(x)
    v = nd.value
    s, c = sincospi(v)
    y = c
    return Lifted{P,N}(y, NDual{P,N}(y, _fwd_guarded_scale(nd.partials, -oftype(v, π) * s)))
end
```

`sincospi` yields `sinpi` and `cospi` together, so `cospi`'s value and its derivative factor
`-π·sinpi(x)` cost one transcendental call between them instead of two. Follow the thin `exp` shape
by default; reach for this one only where a fused primitive genuinely supplies both halves.

## Why This Is Useful

This approach keeps the local numerical semantics close to the scalar arithmetic. That usually gives:

- one clear place (`Nfwd.jl` + `low_level_maths.jl`) to handle edge cases such as `log`, `sqrt`,
  `hypot`, `^`, `mod`, or `mod2pi`,
- forward rules that reuse the shared `NDual` arithmetic, and
- reverse rules that are small, allocation-free, and free of any forward-mode dependency.

## Where It Is A Good Fit

This approach is a good fit when:

- the primitive is scalar or low-dimensional,
- the derivative behavior is local and numerical, and
- the output is already something `nfwd` can lift and extract cleanly (forward), and has a simple
  closed-form derivative (reverse).

Typical examples are unary scalar functions, binary scalar functions, small tuple-output functions, and a few carefully chosen low-arity vararg cases.

## Where It Is Not A Good Fit

It is usually not the right abstraction when:

- mutation or alias restoration is the main difficulty,
- the rule depends on array canonicalisation such as `arrayify` or `matrixify`,
- the tangent structure matters more than the scalar arithmetic, or
- performance depends on a custom reverse implementation that should not be reconstructed from a scalar derivative factor.

In those cases, a hand-written Mooncake rule is usually clearer.

## Practical Rule Of Thumb

If a primitive's AD behavior can be described as "small numerical semantics on a few scalar slots", start by asking whether `NDual` should own the forward behavior and whether the reverse derivative has a simple closed form.

If yes, implement the `NDual` forward overload and the native reverse factor in `low_level_maths.jl`.
If not, write the Mooncake rule directly.
