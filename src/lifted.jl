# Forward-mode slot wrapper `Lifted{P, N, V}` and its associated width-N
# `dual_type` / `lifted_type` queries and seed factories.
#
# This file is the entry point for the new forward-mode design. The legacy
# width-1 `Dual{P, T}` path in `src/tangents/dual.jl` is preserved as the
# backward-compatibility boundary; new forward-mode rules dispatching on
# `Lifted` use the width-N machinery defined here.
#
# Loaded after `nfwd/Nfwd.jl` so the `NDual{T, N}` IEEEFloat carrier is in
# scope. See AGENTS.md and the design notes
# (`~/notes/mooncake/{lifted,dual,primal}-types.md`) for the specification.

"""
    Lifted{P, N, V}

Forward-mode slot wrapper for a primal value of type `P` and its canonical
`N`-width forward representation `V`. Two fields:

- `primal::P` — slot-level back-reference to the user's primal value. For
  mutable struct primals this aliases user storage; for immutable primals
  it carries the same value.
- `value::V` — the canonical `N`-width forward representation. For
  concrete runtime wrappers `V === dual_type(Val(N), P)`.

Rules dispatch on `Lifted{P, N}` (V left abstract) and use `primal`,
`tangent`, and per-lane extractors to access tangent data. Outputs are
constructed via `Lifted{P_out, N}(primal_out, value_out)`.

Width `N == 1` is ordinary forward mode; `N >= 2` is chunked forward
mode. `Lifted` never nests inside another `Lifted`'s `V`; only `Val(0)`
slots are unwrapped (primal passthrough).
"""
struct Lifted{P,N,V}
    primal::P
    value::V
end

# Two-argument constructor with V inferred from typeof(value). The
# canonical wrapping path: `value` is already a built inner V of the
# correct shape; this overload just wraps the (primal, value) pair.
@inline function Lifted{P,N}(primal::P, value::V) where {P,N,V}
    return Lifted{P,N,V}(primal, value)
end

# Accessors — mirror the existing `Dual` / `CoDual` API.
primal(d::Lifted) = d.primal
tangent(d::Lifted) = d.value

"""
    extract(d::Lifted) -> (primal, value)

Helper that returns the `(primal(d), tangent(d))` tuple. Mirrors
`extract(::Dual)` and `extract(::CoDual)` for symmetric ergonomics.
"""
extract(d::Lifted) = (primal(d), tangent(d))

function Base.copy(d::Lifted{P,N,V}) where {P,N,V}
    return Lifted{P,N,V}(copy(primal(d)), copy(tangent(d)))
end

# Lifted slots are safe to share by reference within a forward pass —
# slot-local tangent storage rules out cross-slot aliasing hazards.
_copy(d::Lifted) = d

function Base.:(==)(a::Lifted, b::Lifted)
    return primal(a) == primal(b) && tangent(a) == tangent(b)
end

# ──────────────────────────────────────────────────────────────────────────
# Width-N `dual_type` and `lifted_type` queries.
#
# `dual_type(Val(N), P)` returns the canonical inner V for a primal of
# type `P` at width `N` — i.e. the type of `tangent(d::Lifted{P, N})`'s
# payload, equal to the type of the slot's `value::V` field.
#
# `lifted_type(Val(N), P)` returns the corresponding wrapped slot type.
# For concrete `P`, `lifted_type(Val(N), P) === Lifted{P, N, dual_type(Val(N), P)}`.
#
# This file defines the IEEEFloat scalar case; container shapes (Array,
# Complex, Tuple, NamedTuple, struct lifts) are added in follow-up
# commits.
#
# Note: the legacy width-1 `dual_type(::Type{P})` defined in
# `src/tangents/dual.jl` remains untouched and continues to drive the
# bare-`Dual{P,T}` forward-mode path. The width-N variant below is a
# distinct function (different arity), so the two do not interfere.
# ──────────────────────────────────────────────────────────────────────────

"""
    dual_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `N`-width forward-mode value type for a primal of
type `P`. This is the type of the inner `V` carried by a
`Lifted{P, N, V}` slot.

Shapes defined so far:

- `P <: IEEEFloat`: `NDual{P, N}` — the packed scalar forward value.
- `Complex{R}` with `R <: IEEEFloat`: `Complex{NDual{R, N}}` — element-wise
  recursion through the complex real/imag parts.

Other primal shapes are added in follow-up commits.
"""
@inline dual_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat} = NDual{P,N}
@inline function dual_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}
end

"""
    lifted_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `Lifted{P, N, V}` slot type for a primal of type `P`
at width `N`. For concrete `P`, equals `Lifted{P, N, dual_type(Val(N), P)}`.

Shapes defined so far:

- `P <: IEEEFloat`: `Lifted{P, N, NDual{P, N}}`.
- `Complex{R}` with `R <: IEEEFloat`: `Lifted{Complex{R}, N, Complex{NDual{R, N}}}`.
"""
@inline lifted_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat} = Lifted{P,N,NDual{P,N}}
@inline function lifted_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Lifted{Complex{R},N,Complex{NDual{R,N}}}
end

# ──────────────────────────────────────────────────────────────────────────
# Seed factories.
#
# Layer 2 — bare inner V (the slot's `value::V` field content):
#   `zero_dual(Val(N), x)`     — `dual_type(Val(N), typeof(x))` with zero partials.
#   `uninit_dual(Val(N), x)`   — same shape; tangent payload semantically uninitialized.
#   `randn_dual(Val(N), rng, x)` — random partials sampled from `randn`.
#
# Layer 3 — wrapped Lifted slot:
#   `zero_lifted(Val(N), x)`     — `Lifted{typeof(x), N}` wrapping `zero_dual`.
#   `uninit_lifted(Val(N), x)`   — wrapping `uninit_dual`.
#   `randn_lifted(Val(N), rng, x)` — wrapping `randn_dual`.
# ──────────────────────────────────────────────────────────────────────────

@inline function zero_dual(::Val{N}, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> zero(T), Val(N)))
end

@inline function uninit_dual(::Val{N}, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> zero(T), Val(N)))
end

@inline function randn_dual(::Val{N}, rng::AbstractRNG, x::T) where {N,T<:IEEEFloat}
    return NDual{T,N}(x, ntuple(_ -> randn(rng, T), Val(N)))
end

@inline function zero_lifted(w::Val{N}, x::T) where {N,T<:IEEEFloat}
    return Lifted{T,N}(x, zero_dual(w, x))
end

@inline function uninit_lifted(w::Val{N}, x::T) where {N,T<:IEEEFloat}
    return Lifted{T,N}(x, uninit_dual(w, x))
end

@inline function randn_lifted(w::Val{N}, rng::AbstractRNG, x::T) where {N,T<:IEEEFloat}
    return Lifted{T,N}(x, randn_dual(w, rng, x))
end
