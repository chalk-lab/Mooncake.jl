# Scalar And Low-Dimensional Rules Via `NDual`

For many scalar and low-dimensional primitives, the simplest strategy in Mooncake is:

1. define the local derivative behavior once on `NDual`, and then
1. expose that behavior to Mooncake through `nfwd`.

This keeps the scalar semantics in one place and lets both forward and reverse mode reuse them.

## Core Idea

If a primitive is fundamentally "a few scalar inputs in, a few scalar outputs out", it is often better to teach `NDual` how that primitive behaves than to hand-write separate Mooncake rules for it.

In this setup:

- `src/nfwd/Nfwd.jl` owns the scalar derivative semantics,
- `src/nfwd/NfwdMooncake.jl` lifts those semantics into Mooncake's `Dual` / `CoDual` interface, and
- `src/rules/rules_via_nfwd.jl` decides which primitive signatures should use that path.

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
@is_primitive MinimalCtx Tuple{typeof(cospi),P} where {P<:IEEEFloat}
function frule!!(f::Dual{typeof(cospi)}, x::Dual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_frule_call(Val(1), f, x)
end

function rrule!!(f::CoDual{typeof(cospi)}, x::CoDual{P}) where {P<:IEEEFloat}
    return NfwdMooncake._nfwd_primitive_rrule_call(Val(1), f, x)
end
```

The real registrations live in `src/rules/rules_via_nfwd.jl`.

Here `Val(1)` means "run the shared `nfwd` path with chunk size 1".
In other words, this primitive wrapper asks `nfwd` to propagate one tangent direction at a time through the `NDual` implementation of `cospi`.

More generally, `Val(N)` is how these helpers receive the chunk size as a compile-time constant.
Use:

- `Val(1)` for the usual scalar primitive wrappers in `rules_via_nfwd.jl`,
- `Val(N)` with `N > 1` when you are deliberately calling the lower-level `nfwd` machinery in chunked mode.

The key point is that `N` is not an arity marker here.
It is the number of tangent lanes carried by the `NDual` evaluation.

`NfwdMooncake._nfwd_primitive_rrule_call`/`NfwdMooncake._nfwd_primitive_frule_call` are internal helpers for primitive wrappers, not
a general public rule interface. They expect a stateless callable tangent, i.e. `NoTangent` or `NoFData`.
More generally, `nfwd` only supports scalar leaves it can lift to `NDual` directly, and
arrays or tuples only when their element types and tangent layouts are supported by the
same lift/extract path.

The important part is that the Mooncake-level rule does not re-encode the derivative.
It just routes the primitive through the shared `nfwd` path.

## Why This Is Useful

This approach works well because it keeps the local numerical semantics close to the scalar arithmetic.

That usually gives:

- better alignment between forward and reverse mode,
- less duplicated rule code,
- one place to handle edge cases such as `log`, `sqrt`, `hypot`, `^`, `mod`, or `mod2pi`, and
- thinner primitive wrappers in `rules_via_nfwd.jl`.

`rules_via_nfwd.jl` then becomes mostly a dispatch table, not a second implementation of the derivative logic.

## Where It Is A Good Fit

This approach is a good fit when:

- the primitive is scalar or low-dimensional,
- the derivative behavior is local and numerical,
- the same behavior should be shared by forward and reverse mode, and
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
