# Scalar And Low-Dimensional Rules Via `NDual`

For many scalar and low-dimensional primitives, the simplest strategy in Mooncake is:

1. define the local derivative behavior once on `NDual`, and then
1. expose that behavior to Mooncake through `nfwd`.

This keeps the scalar semantics in one place and lets lifted forward rules reuse them. Reverse rules use analytic pullbacks.

## Core Idea

If a primitive is fundamentally "a few scalar inputs in, a few scalar outputs out", it is often better to teach `NDual` how that primitive behaves than to hand-write separate Mooncake rules for it.

In this setup:

- `src/nfwd/Nfwd.jl` owns the scalar derivative semantics,
- `src/tangents/lifted.jl` defines the `Lifted` forward representation, and
- `src/rules/low_level_maths.jl` registers scalar primitive rules.

That gives Mooncake one source of truth for:

- ordinary derivatives,
- strong-zero behavior, and
- awkward points such as discontinuities or removable singularities.

## Concrete MWE

Here is the full pattern for a simple scalar primitive such as `cospi(x)`.

The `NDual` method owns the local derivative behavior. Outside `src/nfwd/Nfwd.jl`,
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

Once that exists, the Mooncake primitive wrapper can stay thin:

```julia
@is_primitive MinimalCtx ForwardMode Tuple{typeof(cospi),P} where {P<:IEEEFloat}
function frule!!(::Lifted{typeof(cospi),N}, x::Lifted{P,N}) where {P<:IEEEFloat,N}
    v = cospi(tangent(x))
    return Lifted{P,N}(v.value, v)
end
```

Here `N` is the number of tangent lanes, not the arity. The `NDual` implementation
propagates them together. Register a reverse rule too when claiming the primitive in
both modes; the analytic pullbacks in `src/rules/low_level_maths.jl` show the pattern.
`NfwdMooncake` and its primitive-wrapper helpers have been removed.

## Why This Is Useful

This approach works well because it keeps the local numerical semantics close to the scalar arithmetic.

That usually gives:

- consistent scalar and chunked forward behavior,
- less duplicated rule code,
- one place to handle edge cases such as `log`, `sqrt`, `hypot`, `^`, `mod`, or `mod2pi`, and
- thinner primitive wrappers in `low_level_maths.jl`.

The forward wrappers in `low_level_maths.jl` reuse these scalar methods.

## Where It Is A Good Fit

This approach is a good fit when:

- the primitive is scalar or low-dimensional,
- the derivative behavior is local and numerical,
- the same behavior should be shared by scalar and chunked forward mode, and
- the output is already something `nfwd` can lift and extract cleanly.

Typical examples are unary scalar functions, binary scalar functions, small tuple-output functions, and a few carefully chosen low-arity vararg cases.

## Where It Is Not A Good Fit

It is usually not the right abstraction when:

- mutation or alias restoration is the main difficulty,
- the rule depends on array canonicalisation such as `arrayify` or `matrixify`,
- the tangent structure matters more than the scalar arithmetic, or
- performance depends on a custom reverse implementation that should not be reconstructed from scalar forward propagation.

In those cases, a hand-written Mooncake rule is usually clearer.

## Practical Rule Of Thumb

If a primitive's AD behavior can be described as "small numerical semantics on a few scalar slots", start by asking whether `NDual` should own that behavior.

If yes, implement it there first and expose it through `nfwd`.
If not, write the Mooncake rule directly.
