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

The reverse factors are dispatched through small tables (`_unary_deriv` / `_binary_deriv`, plus
`_pow_grad_x`/`_pow_grad_p` for the pow family and `_pick_first_max`/`_pick_first_min` for `max`/`min`)
and contracted against the output cotangent by `_rev_contract`, which keeps an inactive
(zero-cotangent) lane exactly zero even where the local derivative is `±Inf` — the reverse analogue of
the forward `_pt_guarded_scale`. This covers ordinary derivatives, strong-zero behavior, and awkward
points such as discontinuities or removable singularities.

## Concrete MWE

Here is the full pattern for a simple scalar primitive such as `cospi(x)`.

The `NDual` method owns the local **forward** derivative behavior. Outside `src/nfwd/Nfwd.jl`,
the internal helper names need to be imported or qualified explicitly:

```julia
const NDual = Mooncake.Nfwd.NDual
const _pt_scale = Mooncake.Nfwd._pt_scale

@inline function Base.cospi(x::NDual{T,N}) where {T,N}
    return NDual{T,N}(cospi(x.value), _pt_scale(x.partials, -T(π) * sinpi(x.value)))
end
```

Key details:

- `x.value` is the primal scalar value.
- `x.partials` is the `N`-lane tuple of tangent directions carried by `NDual`.
- `_pt_scale(x.partials, s)` multiplies every tangent lane by the same local scalar derivative `s`.
- The returned `NDual` therefore contains both the primal `cospi(x)` value and the propagated tangent lanes.

The forward `frule!!` stays thin — it just runs that overload. The reverse `rrule!!` is a direct
native pullback using the closed-form factor `d(cospi)/dx = -π·sinpi(x)` (registered in the
`_unary_deriv` table), with no `NDual` seeding:

```julia
@is_primitive MinimalCtx Tuple{typeof(cospi),P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(cospi),N}, x::Lifted{P,N,NDual{P,N}}
) where {N,P<:IEEEFloat}
    dy = cospi(tangent(x))      # the NDual overload runs the primal once, storing it in dy.value
    y = _nfwd_out_value(dy)     # read the primal back — do NOT recompute cospi(primal(x))
    return Lifted{_typeof(y),N}(y, dy)
end

# _unary_deriv(::typeof(cospi), x, y) = -oftype(x, π) * sinpi(x)
function rrule!!(::CoDual{typeof(cospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    _x = primal(x)
    y = cospi(_x)
    cospi_pb!!(ȳ::P) = (NoRData(), _rev_contract(ȳ, _unary_deriv(cospi, _x, y)))
    return zero_fcodual(y), cospi_pb!!
end
```

The real registrations live in `src/rules/low_level_maths.jl`.

The forward rule runs the primal on the inner `NDual` — the input slot already carries the `N` seeded
tangent lanes, so the `NDual` overload propagates all of them in one evaluation.

The reverse rule is independent: it evaluates the primal directly, looks up the closed-form derivative
factor, and contracts it against the output cotangent through `_rev_contract`. It never constructs an
`NDual`, so reverse mode does not depend on the forward-mode `Nfwd` submodule.

`nfwd` only supports scalar leaves it can lift to `NDual` directly, so the forward side of this pattern
fits primitives whose inputs and outputs are a few `IEEEFloat` scalars (or small tuples of them, e.g.
`sincos`); the reverse factors are written by hand for the same signatures.

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
