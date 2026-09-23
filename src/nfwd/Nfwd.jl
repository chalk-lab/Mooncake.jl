"""
    Nfwd

Low-level N-wide dual arithmetic used by Mooncake's `nfwd` machinery.

## High-level examples

Construct an `NDual` directly and inspect its primal value and partials:

```julia
julia> d = NDual{Float64,2}(1.5, (1.0, 0.0))
NDual{Float64, 2}(1.5, (1.0, 0.0))

julia> ndual_value(d)
1.5

julia> ndual_partial(d, 1)
1.0
```

Propagate multiple directions through ordinary scalar code in one pass:

```julia
julia> x = NDual{Float64,2}(2.0, (1.0, 0.0));  # seed dx

julia> y = NDual{Float64,2}(3.0, (0.0, 1.0));  # seed dy

julia> z = x * y + sin(x);

julia> ndual_value(z)
6.909297426825682

julia> ndual_partials(z)  # (dz/dx, dz/dy)
(2.5838531634528574, 2.0)
```

Use `NDual` arrays with reductions such as `sum(abs2, xs)`:

```julia
julia> xs = [
           NDual{Float64,2}(1.0, (1.0, 0.0)),
           NDual{Float64,2}(2.0, (0.0, 1.0)),
       ];

julia> y = sum(abs2, xs);

julia> ndual_value(y)
5.0

julia> ndual_partials(y)  # (d/dx₁, d/dx₂)
(2.0, 4.0)
```

`Nfwd.jl` provides the N-wide dual arithmetic and signature helpers used by the scalar
primitive **forward** `frule!!`s in `src/rules/low_level_maths.jl` (which run the `f(::NDual)`
overloads). The matching reverse `rrule!!`s there are direct native analytic pullbacks and do
not use this module.
"""
module Nfwd

using Base: IEEEFloat
using LinearAlgebra

export NDual,
    NDualArray,
    NDualEltype,
    NDualUnsupportedError,
    ndual_value,
    ndual_partial,
    ndual_partials,
    primal_dim,
    static_primal_dim,
    UnsupportedError,
    UnsupportedInputError,
    UnsupportedOutputError,
    _NFWD_PREFERRED_CHUNK_SIZE,
    _nfwd_check_chunk_size,
    _nfwd_default_chunk_size,
    _nfwd_fold_slots,
    _nfwd_input_error,
    _nfwd_is_supported_primal,
    _nfwd_is_supported_scalar,
    _nfwd_output_error,
    _nfwd_unfold_slots

#
# ── Role of `ntuple` ──────────────────────────────────────────────────────────────
# `ntuple(f, Val(N))` is the workhorse for constructing and transforming NDual
# partials.  Its role differs by context:
#
#   On the CPU (rule setup, before kernel launch):
#     ntuple(f, Val(N)) also unrolls at compile time — Julia's Base implementation
#     is @generated and emits N independent expressions, which LLVM then sees as a
#     fixed-size tuple and may vectorise (e.g. a single <N x double> select for the
#     standard-basis seed).  So seed construction:
#       NDual{T,N}(x, ntuple(i -> i == k ? one(T) : zero(T), Val(N)))
#     is branchless on CPU too.  Performance is not critical here because this runs
#     once per input slot (host code), not once per array element.
#
#   Inside GPU kernels (arithmetic rules):
#     ntuple(f, Val(N)) with a statically-known N unrolls to N independent PTX
#     instructions at compile time — no loop, no heap allocation, no runtime
#     dispatch.  LLVM/NVVM sees a fixed-size tuple and vectorises each partial
#     slot independently, keeping everything in registers.  This is the key reason
#     N is a *type parameter* and not a runtime integer: the unrolling requires N
#     to be a compile-time constant.
#
# !! GPU KERNEL ARITHMETIC — PREFER BRANCHLESS OPERATIONS !!
# NDual arithmetic executes inside GPU kernels. Prefer `ifelse(cond, a, b)` over
# `cond ? a : b` or `if/else` blocks: `ifelse` evaluates both branches
# unconditionally and reliably lowers to a single PTX `selp` instruction.
# `?:` may also be optimised to `selp` by LLVM for simple scalar expressions,
# but this is not guaranteed — for data-dependent conditions (values that differ
# across threads) an unoptimised branch causes warp divergence.

"""
    NDual{T<:IEEEFloat, N} <: AbstractFloat

An N-wide dual number: carries one primal `value::T` and `N` partial derivatives
`partials::NTuple{N,T}`.  It is a plain `isbits` type — lives in GPU registers and
compiles to PTX without heap allocation.

## Analogy to ForwardDiff chunk mode

ForwardDiff's chunk mode computes N directional derivatives simultaneously by using
`ForwardDiff.Dual{Tag,T,N}` — a dual number with N partial slots.  `NDual{T,N}` is
the same idea, stripped of the tag parameter and defined entirely within Mooncake:

| Type                         | Tangent width | Tag parameter | Use case                        |
|------------------------------|---------------|---------------|---------------------------------|
| `NDual{T,1}`                 | 1             | n/a           | Standard width-1 `frule!!`      |
| `ForwardDiff.Dual{Tag,T,N}`  | N             | yes           | ForwardDiff chunk mode          |
| `NDual{T,N}`                 | N             | no            | GPU kernel widening (this type) |

`NDual` is a drop-in replacement for `ForwardDiff.Dual` in GPU broadcast kernels.
Removing the tag simplifies the type signature and eliminates the ForwardDiff
dependency from GPU AD.  The arithmetic rules are identical: each operation applies
the chain rule to all N slots at once.

## NDual vs the canonical Lifted V: scalar leaves and flattening

The canonical forward representation `Lifted{P, N, V}` wraps any differentiable value `P` —
it threads through Mooncake's tangent system and handles arbitrary structs transparently
(its `V` recurses structurally).

`NDual{T,N}` only wraps **scalar IEEEFloat (or Complex{IEEEFloat}) leaves**.
For a complex input type (e.g. a struct with several float fields), you must
**flatten** it to its scalar leaves before wrapping:

```
struct S; a::Float64; b::ComplexF64; end   # dimension = 3 slots

S(a, b) → flatten → [a, re(b), im(b)]
             ↓ wrap each leaf as NDual{Float64,3}(x, eₖ)
             ↓ kernel runs
             ↓ extract partials
             ↓ unflatten → Tangent{S}(∂a, Complex(∂re_b, ∂im_b))
```

GPU kernels cannot receive a Dict or arbitrary struct; flattening to scalars
must happen on the CPU before launch, and gradient reassembly happens on the
CPU after.  The broadcast rule in `MooncakeCUDAExt.jl` implements this for the
specific node types that appear in a `Broadcasted` tree
(`_gpu_bcast_leaves` / `_gpu_fill_args_rdata`).

## Complex support

For complex inputs the kernel uses `Complex{NDual{T,N}}` where each component
(`re`, `im`) carries its own N partials.  Julia's generic `Complex` arithmetic
(`+`, `*`, `sin`, etc.) composes with `NDual` naturally because `NDual <: AbstractFloat`.

## Usage in GPU kernels

```julia
# Wrap input scalar at slot k (1-indexed) out of N total slots
d = NDual{T,N}(x, ntuple(j -> T(j == k), Val(N)))

# After kernel: extract primal and k-th partial
v  = ndual_value(d)
dk = ndual_partial(d, k)
```

To extend to a new scalar type S (non-IEEEFloat): define `_broadcast_elem_dof_type(::Type{S})`
and handle the wrapping / gradient extraction in `_leaf_effective_tangent`,
`materialize_pb!!`, and `_gpu_fill_args_rdata` in `MooncakeCUDAExt.jl`.

"""
struct NDual{T<:IEEEFloat,N} <: AbstractFloat
    value::T
    partials::NTuple{N,T}
end

# ── Constructors ─────────────────────────────────────────────────────────────────

# Promote a plain scalar to a NDual with zero partials (acts as a constant).
NDual{T,N}(x::Real) where {T<:IEEEFloat,N} = NDual{T,N}(T(x), ntuple(_ -> zero(T), Val(N)))
# NDual{T,N}(::Rational{S}) is ambiguous between the NDual{T,N}(::Real) method above and
# Base's `(::Type{T})(x::Rational{S}) where {S, T<:AbstractFloat}` (rational.jl).
# Resolve by making the Rational case concrete.
function NDual{T,N}(x::Rational{S}) where {T<:IEEEFloat,N,S}
    NDual{T,N}(T(x), ntuple(_ -> zero(T), Val(N)))
end
# Identity / same-precision constructor: NDual{T,N}(d::NDual{T,N}) must not call T(d).
NDual{T,N}(d::NDual{T,N}) where {T<:IEEEFloat,N} = d
# Cross-precision constructor: NDual{T,N}(d::NDual{S,N}) where S ≠ T.
function NDual{T,N}(d::NDual{S,N}) where {T<:IEEEFloat,N,S<:IEEEFloat}
    NDual{T,N}(T(d.value), ntuple(i -> T(d.partials[i]), Val(N)))
end

# ── Accessors ────────────────────────────────────────────────────────────────────

@inline ndual_value(d::NDual) = d.value
@inline ndual_partial(d::NDual, k::Int) = d.partials[k]
@inline ndual_partials(d::NDual) = d.partials

# Internal NDual decode helpers shared by nfwd and the CUDA extension.
@inline _nfwd_dual_value(d::NDual) = ndual_value(d)
@inline _nfwd_dual_value(z::Complex{<:NDual}) = complex(
    ndual_value(real(z)), ndual_value(imag(z))
)
@inline _nfwd_dual_value(x) = x

@inline _nfwd_dual_partial(d::NDual, k::Int) = ndual_partial(d, k)
@inline _nfwd_dual_partial(z::Complex{<:NDual}, k::Int) = complex(
    ndual_partial(real(z), k), ndual_partial(imag(z), k)
)
@inline _nfwd_dual_partial(x, ::Int) = false

@inline _nfwd_dual_primal_type(::Type{<:NDual{T}}) where {T} = T
@inline _nfwd_dual_primal_type(::Type{Complex{NDual{T,N}}}) where {T,N} = Complex{T}
@inline _nfwd_dual_primal_type(::Type{T}) where {T} = T

@inline _nfwd_dual_has_partials(::Type{<:NDual}) = true
@inline _nfwd_dual_has_partials(::Type{<:Complex{<:NDual}}) = true
@inline _nfwd_dual_has_partials(::Type) = false

# Assemble canonical scalar V from primal-typed lanes; shared by element-read frules.
@inline _scalar_ndual(y::T, parts::NTuple{N,T}) where {T<:IEEEFloat,N} = NDual{T,N}(
    y, parts
)
@inline function _scalar_ndual(
    y::Complex{R}, parts::NTuple{N,Complex{R}}
) where {R<:IEEEFloat,N}
    return Complex(
        NDual{R,N}(real(y), ntuple(k -> real(parts[k]), Val(N))),
        NDual{R,N}(imag(y), ntuple(k -> imag(parts[k]), Val(N))),
    )
end

# ── NTuple arithmetic helpers ─────────────────────────────────────────────────────
# All fully unrolled at compile time via Val(N) — safe for GPU registers.

@inline _fwd_scale(p::NTuple{N,T}, s::T) where {N,T} = ntuple(i -> s * p[i], Val(N))
# N=1 specializations avoid closure heap-allocation on the scalar (chunk_size=1) hot path.
@inline _fwd_scale(p::NTuple{1,T}, s::T) where {T} = (s * p[1],)
# `_nfwd_zero_mask` plays the same role as `nan_tangent_guard` for scalar NDual algebra:
# when the local seed / upstream factor `a` is zero, replace `b` by zero(b) before the
# multiply so `0 * Inf` and `0 * NaN` collapse to zero instead of poisoning the tangent.
# nfwd uses this in forward mode through `_fwd_guarded_scale`, which masks zero NDual lanes
# in singular formulas such as `log`, `sqrt`, `cbrt`, and `hypot`, and in reverse mode
# through `_nfwd_real_dot`, which masks zero upstream cotangents before contracting them
# against nfwd output tangents. This is the same strong-zero idea used in other AD systems,
# including ForwardDiff, to keep inactive directions from turning into NaNs.
@inline _nfwd_zero_mask(a, b) = ifelse(iszero(a), zero(b), b)
@inline function _fwd_guarded_scale(p::NTuple{N,T}, s::T) where {N,T}
    return ntuple(i -> begin
        pi = p[i]
        pi * _nfwd_zero_mask(pi, s)
    end, Val(N))
end
@inline _fwd_add(p::NTuple{N}, q::NTuple{N}) where {N} = ntuple(i -> p[i] + q[i], Val(N))
@inline _fwd_sub(p::NTuple{N}, q::NTuple{N}) where {N} = ntuple(i -> p[i] - q[i], Val(N))
@inline _fwd_neg(p::NTuple{N}) where {N} = ntuple(i -> -p[i], Val(N))
@inline _fwd_zero(::Val{N}, ::Type{T}) where {N,T} = ntuple(_ -> zero(T), Val(N))
@inline _fwd_add(p::NTuple{1,T}, q::NTuple{1,T}) where {T} = (p[1] + q[1],)
@inline _fwd_sub(p::NTuple{1,T}, q::NTuple{1,T}) where {T} = (p[1] - q[1],)
@inline _fwd_neg(p::NTuple{1,T}) where {T} = (-p[1],)
@inline _fwd_zero(::Val{1}, ::Type{T}) where {T} = (zero(T),)

# These helpers define the scalar edge-case behavior used by nfwd for non-smooth
# primitives: `^` keeps the removable-singularity cases at x == 0, while `mod` and
# `mod2pi` return NaN coefficients at their discontinuity points.
@inline _nfwd_pow_grad_x(x, p, y) = p * y / x
@inline function _nfwd_pow_grad_x(x::P, p::P, y) where {P<:IEEEFloat}
    return ifelse(
        !iszero(x) || p < zero(P),
        p * y / x,
        ifelse(isone(p), one(y), ifelse(iszero(p) || p > one(P), zero(y), oftype(y, Inf))),
    )
end

@inline _nfwd_pow_grad_p(x, p, y) = y * log(complex(x))
@inline function _nfwd_pow_grad_p(x::P, p::P, y) where {P<:IEEEFloat}
    return ifelse(
        !iszero(x), y * real(log(complex(x))), ifelse(p > zero(P), zero(y), oftype(y, NaN))
    )
end

@inline function _nfwd_mod_grad_coeffs(x::P, y::P) where {P<:IEEEFloat}
    u = x / y
    nan = oftype(u, NaN)
    isint = isinteger(u)
    return ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))
end

@inline _nfwd_mod2pi_grad(x::P) where {P<:IEEEFloat} = ifelse(
    isinteger(x / P(2π)), P(NaN), one(P)
)

# ── AbstractFloat traits (needed for promote_rule with Complex etc.) ──────────────

Base.float(a::NDual) = a
Base.AbstractFloat(a::NDual) = a
Base.floatmin(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(floatmin(T))
Base.floatmax(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(floatmax(T))
Base.typemin(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(typemin(T))
Base.typemax(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(typemax(T))
# Instance dispatch: floatmin(x::NDual) and floatmax(x::NDual) forward to the type method.
Base.floatmin(x::NDual{T,N}) where {T,N} = NDual{T,N}(floatmin(T))
Base.floatmax(x::NDual{T,N}) where {T,N} = NDual{T,N}(floatmax(T))
Base.precision(::Type{NDual{T,N}}) where {T<:AbstractFloat,N} = precision(T)
Base.precision(::NDual{T,N}) where {T<:AbstractFloat,N} = precision(T)
# nextfloat / prevfloat are treated as identity maps for differentiation, so preserve the
# partials while advancing or retreating the scalar value by one representable step.
Base.nextfloat(a::NDual{T,N}) where {T,N} = NDual{T,N}(nextfloat(a.value), a.partials)
Base.prevfloat(a::NDual{T,N}) where {T,N} = NDual{T,N}(prevfloat(a.value), a.partials)
# exponent / significand / frexp: scalar operations; return scalar value (integer / NDual).
# `significand` and `frexp` rescale `x` by a power of two that is CONSTANT within a binade, so the
# derivative is that scale. Apply it directly to each partial: the scale itself may overflow
# even when the scaled partial is finite.
Base.exponent(a::NDual) = exponent(a.value)
@inline function Base.significand(a::NDual{T,N}) where {T,N}
    # At zero and non-finite values, use the same unit-scale convention as `frexp`.
    e = (iszero(a.value) || !isfinite(a.value)) ? 0 : -exponent(a.value)
    return NDual{T,N}(significand(a.value), map(p -> ldexp(p, e), a.partials))
end
@inline function Base.frexp(a::NDual{T,N}) where {T,N}
    v, e = frexp(a.value)
    return NDual{T,N}(v, map(p -> ldexp(p, -e), a.partials)), e
end

# ── Zero / One ────────────────────────────────────────────────────────────────────

Base.zero(::NDual{T,N}) where {T,N} = NDual{T,N}(zero(T), _fwd_zero(Val(N), T))
Base.one(::NDual{T,N}) where {T,N} = NDual{T,N}(one(T), _fwd_zero(Val(N), T))
Base.zero(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(zero(T), _fwd_zero(Val(N), T))
Base.one(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(one(T), _fwd_zero(Val(N), T))
# Default oneunit(T) = T(one(T)) would call NDual{T,N}(::NDual) → Float64(::NDual) → error.
# Override to use the scalar constructor directly.
Base.oneunit(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(oneunit(T))
Base.oneunit(::NDual{T,N}) where {T,N} = NDual{T,N}(oneunit(T))

# ── Promotion / Conversion ────────────────────────────────────────────────────────

@inline function Base.convert(::Type{NDual{T,N}}, x::Real) where {T,N}
    return NDual{T,N}(T(x), _fwd_zero(Val(N), T))
end
Base.convert(::Type{NDual{T,N}}, d::NDual{T,N}) where {T,N} = d

@inline function Base.promote_rule(::Type{NDual{T,N}}, ::Type{S}) where {T,N,S<:Real}
    return NDual{promote_type(T, S),N}
end
Base.promote_rule(::Type{NDual{T,N}}, ::Type{NDual{T,N}}) where {T,N} = NDual{T,N}
# Cross-precision: NDual{Float32,N} op NDual{Float64,N} → NDual{Float64,N}
@inline function Base.promote_rule(::Type{NDual{T1,N}}, ::Type{NDual{T2,N}}) where {T1,T2,N}
    return NDual{promote_type(T1, T2),N}
end
# Different widths cannot nest: diagnose at promotion so max/min and containers inherit it.
# Residue: hand-built mixed-width muladd/fma/clamp still raise TypeError when rewrapping.
function Base.promote_rule(::Type{NDual{T1,N1}}, ::Type{NDual{T2,N2}}) where {T1,T2,N1,N2}
    _throw_ndual_lane_mismatch(:promote_type, N1, N2)
end
@inline function Base.convert(::Type{NDual{T,N}}, d::NDual{S,N}) where {T,N,S<:IEEEFloat}
    return NDual{T,N}(T(d.value), ntuple(i -> T(d.partials[i]), Val(N)))
end

# Integer conversion cannot propagate derivatives; refuse rather than silently discard them.
@noinline _throw_ndual_to_int(::Type{I}) where {I<:Integer} = throw(
    ArgumentError(
        "cannot convert a forward-mode dual (NDual) to $I: an integer conversion is not " *
        "differentiable, so the enclosing function is not forward-mode differentiable here.",
    ),
)
(::Type{I})(::NDual) where {I<:Integer} = _throw_ndual_to_int(I)
Base.convert(::Type{I}, ::NDual) where {I<:Integer} = _throw_ndual_to_int(I)
# Disambiguate against `Bool(x::Real)`: the `Integer` method above is more specific in the type
# argument but less specific in the value, so `Bool(::NDual)` would otherwise be an ambiguity.
Base.Bool(::NDual) = _throw_ndual_to_int(Bool)
@inline function NDual{T,N}(x::Real, r::RoundingMode) where {T<:IEEEFloat,N}
    return NDual{T,N}(T(x, r), _fwd_zero(Val(N), T))
end

@noinline function _throw_ndual_lane_mismatch(op::Symbol, n1::Int, n2::Int)
    throw(
        DimensionMismatch(
            "NDual lane count mismatch in `$op`: left operand has $n1 lanes, right operand has " *
            "$n2 lanes. Duals of different widths come from two forward passes, and `NDual` " *
            "carries no perturbation tag, so it cannot nest. Mooncake supports " *
            "forward-over-reverse for second-order derivatives.",
        ),
    )
end

@inline function _promote_matching_nduals(
    op::Symbol, a::NDual{T1,N1}, b::NDual{T2,N2}
) where {T1,T2,N1,N2}
    N1 == N2 || _throw_ndual_lane_mismatch(op, N1, N2)
    return promote(a, b)
end

# ── Arithmetic ────────────────────────────────────────────────────────────────────

@inline function Base.:+(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(a.value + b.value, _fwd_add(a.partials, b.partials))
end
@inline function Base.:-(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(a.value - b.value, _fwd_sub(a.partials, b.partials))
end
@inline Base.:-(a::NDual{T,N}) where {T,N} = NDual{T,N}(-a.value, _fwd_neg(a.partials))

# Real ± NDual: skip promotion — partials are unchanged for add, negated for sub.
# Without these, `c + x` promotes c to NDual(c, zeros) then adds zero partials,
# generating `fadd 0.0, p[k]` per slot that LLVM cannot fold (signed-zero corner case).
# Output type is promote_type(T, R) so that Float64 + NDual{Float32} → NDual{Float64}.
@inline function Base.:+(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(c) + S(x.value), ntuple(i -> S(x.partials[i]), Val(N)))
end
@inline function Base.:+(x::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(x.value) + S(c), ntuple(i -> S(x.partials[i]), Val(N)))
end
@inline function Base.:-(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(c) - S(x.value), ntuple(i -> -S(x.partials[i]), Val(N)))
end
@inline function Base.:-(x::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(x.value) - S(c), ntuple(i -> S(x.partials[i]), Val(N)))
end

# Disambiguate cross-precision NDual ± NDual: the Real ± NDual methods above match
# NDual{T2,N} as the "Real" argument when T1≠T2, creating ambiguity.  Explicit
# cross-precision methods resolve this by promoting to a common precision first.
@inline Base.:+(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = +(promote(a, b)...)
@inline Base.:-(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = -(promote(a, b)...)
@inline Base.:*(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = *(promote(a, b)...)
@inline Base.:/(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = /(promote(a, b)...)
@inline Base.:+(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = +(_promote_matching_nduals(
    :+, a, b
)...)
@inline Base.:-(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = -(_promote_matching_nduals(
    :-, a, b
)...)
@inline Base.:*(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = *(_promote_matching_nduals(
    :*, a, b
)...)
@inline Base.:/(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = /(_promote_matching_nduals(
    :/, a, b
)...)

# Product rule: d(a*b) = a*db + b*da
@inline function Base.:*(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        a.value * b.value,
        _fwd_add(_fwd_scale(a.partials, b.value), _fwd_scale(b.partials, a.value)),
    )
end

# Mixed Real*NDual: skip promotion and product rule — just scale the partials.
# Without these, `c * x` where c::Real promotes c to NDual(c, zeros) and runs
# the full product rule, generating a useless `x.value * 0.0` fmul per partial
# slot that IEEE semantics prevent LLVM from folding (-0 * NaN ≠ 0).
# Output type is promote_type(T, R) so that Float64 * NDual{Float32} → NDual{Float64}.
@inline function Base.:*(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    s = S(c)
    return NDual{S,N}(s * S(x.value), ntuple(i -> s * S(x.partials[i]), Val(N)))
end
@inline function Base.:*(x::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    s = S(c)
    return NDual{S,N}(S(x.value) * s, ntuple(i -> S(x.partials[i]) * s, Val(N)))
end

# Bool ± NDual and Bool * NDual: Base defines concrete overloads for (Bool, AbstractFloat)
# in bool.jl (+(::Bool, ::T), +(::T, ::Bool), *(::Bool, ::T), *(::T, ::Bool) where T<:AbstractFloat).
# Since NDual <: AbstractFloat these are now ambiguous with our (Real, NDual) methods.
# Resolve with concrete Bool overloads:
#   + : Bool acts as its numeric value (false=0, true=1) — same as T(b) + x.
#   * : preserves Base's "strong zero" contract (false*NaN == 0.0 via ifelse).
@inline Base.:+(b::Bool, x::NDual{T,N}) where {T,N} = NDual{T,N}(T(b) + x.value, x.partials)
@inline Base.:+(x::NDual{T,N}, b::Bool) where {T,N} = b + x
@inline Base.:*(b::Bool, x::NDual{T,N}) where {T,N} = ifelse(b, x, copysign(zero(x), x))
@inline Base.:*(x::NDual{T,N}, b::Bool) where {T,N} = b * x

# Guard both quotient-rule scales: at b.value == 0, inactive lanes must not become 0*Inf.
@inline function Base.:/(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    v = a.value / b.value
    return NDual{T,N}(
        v,
        _fwd_guarded_scale(
            _fwd_sub(a.partials, _fwd_guarded_scale(b.partials, v)), inv(b.value)
        ),
    )
end

# NDual / Real: the partials multiply by the reciprocal, avoiding promoting c to NDual.
@inline function Base.:/(x::NDual{T,N}, c::R) where {T,N,R<:Real}
    S = promote_type(T, R)
    cS = S(c)
    # The value must equal the primal's, so it divides; `x.value * inv(cS)` differs in the
    # last ulp. The `N` partials keep the reciprocal, one division instead of `N`.
    sp = ntuple(i -> S(x.partials[i]), Val(N))
    return NDual{S,N}(S(x.value) / cS, _fwd_guarded_scale(sp, inv(cS)))
end

# Real / NDual: d(c/b) = -(c/b²) db.  Without this, c::Real is promoted to
# NDual(c, zeros) and the quotient rule runs with a zero-partial numerator,
# producing a fneg(partial) chain that cancels with inv's -r² scaling but forces
# LLVM to emit a `fadd x, 0.0` canonicalization per partial slot (IEEE -0 rule).
# Defining this explicitly computes the scale as -(c*vi²) — a single scalar fneg —
# which pairs with the fneg already in the partial to give fmul(neg,neg)=pos,
# eliminating the fsub/fadd artifact.  Mirrors ForwardDiff's /(::Real,::Dual).
@inline function Base.:/(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    vi = inv(S(x.value))
    # Promote the partials to `S` too (the value is already `S`), so the guarded scale sees
    # matching types — mirrors the `+`/`-`/`*` Real/NDual methods.
    sp = ntuple(i -> S(x.partials[i]), Val(N))
    # The value is the DIVISION, not `c * inv(x)`: `inv` overflows for a subnormal divisor where
    # `c / x` is finite. The `inv` form stays in the scale, which is what it was written for.
    v = S(c) / S(x.value)
    return NDual{S,N}(v, _fwd_guarded_scale(sp, -(v * vi)))
end

# Avoid the zero numerator partials of the quotient rule; guard inactive lanes at zero.
# Literal x^-1 delegates here too.
@inline function Base.inv(a::NDual{T,N}) where {T,N}
    vi = inv(a.value)
    return NDual{T,N}(vi, _fwd_guarded_scale(a.partials, -(vi * vi)))
end

# FMA (Fused Multiply-Add) based muladd: a single CPU instruction computes a*b+c
# in one step instead of separate fmul+fadd.  The default `a*b+c` would compute
# the product rule in two passes, emitting separate fmul+fadd per partial slot.
# Using nested muladd fuses both into two FMA instructions per slot:
#   value:   muladd(va, vb, vc)
#   partial: muladd(va, pb[i], muladd(vb, pa[i], pc[i]))
# This halves the instruction count for the matmul inner loop and triangular-
# solve back-substitution, which are the dominant cost in sum_matmat / sum_linsolve.
@inline function Base.muladd(a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        muladd(a.value, b.value, c.value),
        ntuple(
            i -> muladd(
                a.value, b.partials[i], muladd(b.value, a.partials[i], c.partials[i])
            ),
            Val(N),
        ),
    )
end
# Base.fma guarantees a single hardware FMA instruction (no intermediate rounding),
# whereas muladd may or may not fuse depending on platform/compiler flags.
@inline function Base.fma(a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        fma(a.value, b.value, c.value),
        ntuple(
            i -> fma(a.value, b.partials[i], fma(b.value, a.partials[i], c.partials[i])),
            Val(N),
        ),
    )
end

# Real*NDual+NDual and NDual*Real+NDual: the mixed cases arise in triangular solves
# where the factor matrix is Float64 and the rhs is NDual.  One FMA per partial slot.
@inline function Base.muladd(a::R, b::NDual{T,N}, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        muladd(S(a), S(b.value), S(c.value)),
        ntuple(i -> muladd(S(a), S(b.partials[i]), S(c.partials[i])), Val(N)),
    )
end
@inline function Base.fma(a::R, b::NDual{T,N}, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        fma(S(a), S(b.value), S(c.value)),
        ntuple(i -> fma(S(a), S(b.partials[i]), S(c.partials[i])), Val(N)),
    )
end
@inline function Base.muladd(a::NDual{T,N}, b::R, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        muladd(S(a.value), S(b), S(c.value)),
        ntuple(i -> muladd(S(a.partials[i]), S(b), S(c.partials[i])), Val(N)),
    )
end
@inline function Base.fma(a::NDual{T,N}, b::R, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        fma(S(a.value), S(b), S(c.value)),
        ntuple(i -> fma(S(a.partials[i]), S(b), S(c.partials[i])), Val(N)),
    )
end

# NDual*NDual+Real: product rule with a scalar addend.  Without this, c::Real is promoted
# to NDual(c, zeros) and the inner muladd becomes muladd(bv, ap, 0.0) per partial slot,
# emitting a wasted `fadd 0.0` (IEEE -0 semantics prevent LLVM from folding it).
# Specialising drops the zero addend: partial_i = muladd(av, bp[i], bv * ap[i]).
@inline function Base.muladd(a::NDual{T,N}, b::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        muladd(S(a.value), S(b.value), S(c)),
        ntuple(
            i -> muladd(S(a.value), S(b.partials[i]), S(b.value) * S(a.partials[i])), Val(N)
        ),
    )
end

# ── Integer and real power ────────────────────────────────────────────────────────

# Use Base.literal_pow so literal powers retain compile-time scalar specialisation.
@inline function Base.literal_pow(::typeof(^), a::NDual{T,N}, ::Val{n}) where {T,N,n}
    v = Base.literal_pow(^, a.value, Val(n))
    dv = ifelse(iszero(n), zero(T), T(n) * Base.literal_pow(^, a.value, Val(n - 1)))
    # Guard inactive lanes when negative powers have infinite coefficients at zero.
    return NDual{T,N}(v, _fwd_guarded_scale(a.partials, dv))
end
# Base defines literal_pow(^, ::AbstractFloat, ::Val{-1}) = inv(x) as a concrete
# specialisation.  Since NDual <: AbstractFloat, this creates an ambiguity with the
# general Val{n} method above (NDual wins on arg 2, Base wins on the concrete Val).
# Resolve with a concrete override that delegates to our inv rule.
@inline Base.literal_pow(::typeof(^), a::NDual{T,N}, ::Val{-1}) where {T,N} = inv(a)

# d(x^n) = n * x^(n-1) * dx  (ifelse keeps this branchless; see file header)
@inline function Base.:^(a::NDual{T,N}, n::Integer) where {T,N}
    v = a.value^n
    dv = ifelse(iszero(n), zero(T), T(n) * a.value^(n - 1))
    # Guard inactive lanes at singularities, as for real exponents.
    return NDual{T,N}(v, _fwd_guarded_scale(a.partials, dv))
end

@inline Base.:^(a::NDual{T,N}, b::Rational) where {T,N} = a ^ T(b)

@inline function Base.:^(a::NDual{T,N}, b::R) where {T,N,R<:Real}
    S = promote_type(T, R)
    av, bS = S(a.value), S(b)
    v = av^bS
    ap = ntuple(i -> S(a.partials[i]), Val(N))
    return NDual{S,N}(v, _fwd_guarded_scale(ap, _nfwd_pow_grad_x(av, bS, v)))
end

@inline function Base.:^(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    v = a.value^b.value
    coeff_a = _nfwd_pow_grad_x(a.value, b.value, float(v))
    coeff_b = _nfwd_pow_grad_p(a.value, b.value, float(v))
    return NDual{T,N}(
        v,
        _fwd_add(
            _fwd_guarded_scale(a.partials, coeff_a), _fwd_guarded_scale(b.partials, coeff_b)
        ),
    )
end
@inline Base.:^(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = ^(_promote_matching_nduals(
    :^, a, b
)...)

# d(b^a)/da = b^a * log(b)  (b a plain Real, a the NDual)
@inline function Base.:^(b::R, a::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    v = S(b)^S(a.value)
    # For b<0, use v*log|b|, matching the reverse rule despite no real derivative.
    # At b==0, positive exponents have limit 0; nonpositive ones give NaN on active lanes.
    ap = ntuple(i -> S(a.partials[i]), Val(N))
    return NDual{S,N}(v, _fwd_guarded_scale(ap, _nfwd_pow_grad_p(S(b), S(a.value), v)))
end
@inline Base.:^(::Irrational{:ℯ}, a::NDual{T,N}) where {T,N} = exp(a)

@inline function Base.FastMath.pow_fast(a::NDual{T,N}, n::Integer) where {T,N}
    v = Base.FastMath.pow_fast(a.value, n)
    return NDual{T,N}(v, _fwd_guarded_scale(a.partials, _nfwd_pow_grad_x(a.value, T(n), v)))
end
@inline function Base.FastMath.pow_fast(a::NDual{T,N}, ::Val{p}) where {T,N,p}
    v = Base.FastMath.pow_fast(a.value, Val(p))
    return NDual{T,N}(v, _fwd_guarded_scale(a.partials, _nfwd_pow_grad_x(a.value, T(p), v)))
end

# ── Math functions ─────────────────────────────────────────────────────────────────
# Each follows: f(NDual(v,p)) = NDual(f(v), f'(v)*p)

# Trig
# Use sincos / sincosd to share the cordic/libm computation between sin and cos.
@inline function Base.sin(a::NDual{T,N}) where {T,N}
    s, c = sincos(a.value)
    return NDual{T,N}(s, _fwd_scale(a.partials, c))
end
@inline function Base.cos(a::NDual{T,N}) where {T,N}
    s, c = sincos(a.value)
    return NDual{T,N}(c, _fwd_scale(a.partials, -s))
end
@inline function Base.tan(a::NDual{T,N}) where {T,N}
    v = tan(a.value)
    return NDual{T,N}(v, _fwd_guarded_scale(a.partials, one(T) + v^2))
end
# Coefficients can be singular even at finite primal values (e.g. asin at its boundary).
# Guard inactive lanes, matching reverse mode.
for (f, coefficient) in (
    (:asin, :(inv(sqrt(one(T) - a.value^2)))),
    (:acos, :(-inv(sqrt(one(T) - a.value^2)))),
    (:atanh, :(inv(one(T) - a.value^2))),
    (:asech, :(-inv(a.value * sqrt(one(T) - a.value^2)))),
    (:acsch, :(-inv(abs(a.value) * sqrt(one(T) + a.value^2)))),
    (:acoth, :(inv(one(T) - a.value^2))),
    (:log, :(inv(a.value))),
    (:log1p, :(inv(one(T) + a.value))),
    (:log2, :(inv(a.value * T(log(2))))),
    (:log10, :(inv(a.value * T(log(10))))),
    (:asec, :(inv(abs(a.value) * sqrt(a.value^2 - one(T))))),
    (:acsc, :(-inv(abs(a.value) * sqrt(a.value^2 - one(T))))),
    (:asind, :(inv(T(deg2rad(sqrt(one(T) - a.value^2)))))),
    (:acosd, :(-inv(T(deg2rad(sqrt(one(T) - a.value^2)))))),
    (:asecd, :(inv(T(deg2rad(abs(a.value) * sqrt(a.value^2 - one(T))))))),
    (:acscd, :(-inv(T(deg2rad(abs(a.value) * sqrt(a.value^2 - one(T))))))),
)
    @eval @inline function Base.$f(a::NDual{T,N}) where {T,N}
        return NDual{T,N}($f(a.value), _fwd_guarded_scale(a.partials, $coefficient))
    end
end

@inline function Base.atan(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(atan(a.value), _fwd_scale(a.partials, inv(one(T) + a.value^2)))
end
@inline function Base.atan(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    r2 = a.value^2 + b.value^2
    regular = isfinite(r2) && !iszero(r2)
    h = regular ? zero(T) : hypot(a.value, b.value)
    c1 = regular ? b.value / r2 : (b.value / h) / h
    c2 = regular ? -a.value / r2 : (-a.value / h) / h
    # Scale original seeds separately: at (0, 0), x*dy-y*dx is zero even on active lanes,
    # so guarding that intermediate would suppress the undefined derivative.
    return NDual{T,N}(
        atan(a.value, b.value),
        _fwd_add(_fwd_guarded_scale(a.partials, c1), _fwd_guarded_scale(b.partials, c2)),
    )
end
@inline Base.atan(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = atan(
    _promote_matching_nduals(:atan, a, b)...
)

# NDual*Real atan: d/dy[atan(y,x)] = x/(y²+x²).  Without this, x::Real is promoted to
# NDual(x, zeros), and _fwd_scale(x.partials, y.value) generates a fmul(partial, 0.0) per
# slot (zero-partial scale), followed by a wasted subtraction of that zero from the result.
@inline function Base.atan(y::NDual{T,N}, x::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    r2 = S(y.value)^2 + S(x)^2
    regular = isfinite(r2) && !iszero(r2)
    h = regular ? zero(S) : hypot(S(y.value), S(x))
    c = regular ? S(x) / r2 : (S(x) / h) / h
    sp = ntuple(i -> S(y.partials[i]), Val(N))
    return NDual{S,N}(atan(S(y.value), S(x)), _fwd_guarded_scale(sp, c))
end

# Real*NDual atan: d/dx[atan(y,x)] = -y/(y²+x²).  Without this, y::Real is promoted to
# NDual(y, zeros), and _fwd_scale(y.partials, x.value) = 0 per slot, then fsub(0, partial)
# hits the same IEEE -0 canonicalization that the existing Real/NDual division rule has.
@inline function Base.atan(y::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    r2 = S(y)^2 + S(x.value)^2
    regular = isfinite(r2) && !iszero(r2)
    h = regular ? zero(S) : hypot(S(y), S(x.value))
    c = regular ? -S(y) / r2 : (-S(y) / h) / h
    sp = ntuple(i -> S(x.partials[i]), Val(N))
    return NDual{S,N}(atan(S(y), S(x.value)), _fwd_guarded_scale(sp, c))
end

# Hyperbolic
@inline function Base.sinh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sinh(a.value), _fwd_scale(a.partials, cosh(a.value)))
end
@inline function Base.cosh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(cosh(a.value), _fwd_scale(a.partials, sinh(a.value)))
end
# `4u / (1 + u)^2` with `u = exp(-2|x|)`: `1 - tanh(x)^2` is exactly `0` once `tanh(x)` rounds
# to `1.0`, while the true derivative is still normal (Float64 `|x| ≳ 19.5`, Float32 `≳ 9`).
@inline function Base.tanh(a::NDual{T,N}) where {T,N}
    u = exp(-2 * abs(a.value))
    return NDual{T,N}(tanh(a.value), _fwd_scale(a.partials, 4u / (one(T) + u)^2))
end
@inline function Base.asinh(a::NDual{T,N}) where {T,N}
    c = if abs(a.value) > sqrt(floatmax(T))
        inv(hypot(a.value, one(T)))
    else
        inv(sqrt(a.value^2 + one(T)))
    end
    return NDual{T,N}(asinh(a.value), _fwd_scale(a.partials, c))
end
@inline function Base.acosh(a::NDual{T,N}) where {T,N}
    c = if abs(a.value) > sqrt(floatmax(T))
        inv(abs(a.value))
    else
        inv(sqrt(a.value^2 - one(T)))
    end
    return NDual{T,N}(acosh(a.value), _fwd_guarded_scale(a.partials, c))
end

# Reciprocal hyperbolic: sech, csch, coth and their inverses.
@inline function Base.sech(a::NDual{T,N}) where {T,N}
    sv = sech(a.value)
    return NDual{T,N}(sv, _fwd_scale(a.partials, -tanh(a.value) * sv))
end
@inline function Base.csch(a::NDual{T,N}) where {T,N}
    cv = csch(a.value)
    return NDual{T,N}(cv, _fwd_guarded_scale(a.partials, -coth(a.value) * cv))
end
@inline function Base.coth(a::NDual{T,N}) where {T,N}
    sv = csch(a.value)
    return NDual{T,N}(coth(a.value), _fwd_guarded_scale(a.partials, -(sv^2)))
end

# Exp / Log
@inline function Base.exp(a::NDual{T,N}) where {T,N}
    return (ev=exp(a.value); NDual{T,N}(ev, _fwd_scale(a.partials, ev)))
end
@inline function Base.exp2(a::NDual{T,N}) where {T,N}
    return (ev=exp2(a.value); NDual{T,N}(ev, _fwd_scale(a.partials, ev * T(log(2)))))
end
# exp10 needs guarding: value*log(10) can overflow while the value stays finite
# (e.g. x=308). exp/exp2 coefficients cannot overflow first.
@inline function Base.exp10(a::NDual{T,N}) where {T,N}
    return (
        ev=exp10(a.value); NDual{T,N}(ev, _fwd_guarded_scale(a.partials, ev * T(log(10))))
    )
end

@inline function Base.expm1(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(expm1(a.value), _fwd_scale(a.partials, exp(a.value)))
end

# Two-argument log: log(b, x) = log(x)/log(b); d/dx = inv(x * log(b)),
# d/db = -log(x) / (b * log(b)^2) = -log(b, x) / (b * log(b)).
@inline function Base.log(b::R, a::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    av = S(a.value)
    ap = ntuple(i -> S(a.partials[i]), Val(N))
    return NDual{S,N}(log(S(b), av), _fwd_guarded_scale(ap, inv(av * S(log(b)))))
end
@inline function Base.log(b::NDual{T,N}, a::NDual{T,N}) where {T,N}
    log_b = log(b.value)
    y = log(b.value, a.value)
    return NDual{T,N}(
        y,
        _fwd_add(
            _fwd_guarded_scale(b.partials, -y / (b.value * log_b)),
            _fwd_guarded_scale(a.partials, inv(a.value * log_b)),
        ),
    )
end
@inline Base.log(b::NDual{T1,N1}, a::NDual{T2,N2}) where {T1,T2,N1,N2} = log(
    _promote_matching_nduals(:log, b, a)...
)
@inline Base.log(::Irrational{:ℯ}, a::NDual{T,N}) where {T,N} = log(a)

# Scale each partial directly to avoid overflow or underflow of the coefficient `2^n`.
@inline function Base.ldexp(a::NDual{T,N}, n::Integer) where {T,N}
    return NDual{T,N}(ldexp(a.value, n), map(p -> ldexp(p, n), a.partials))
end

# Roots
@inline function Base.sqrt(a::NDual{T,N}) where {T,N}
    return (sv=sqrt(a.value); NDual{T,N}(sv, _fwd_guarded_scale(a.partials, inv(2 * sv))))
end
@inline function Base.cbrt(a::NDual{T,N}) where {T,N}
    return (cv=cbrt(a.value); NDual{T,N}(cv, _fwd_guarded_scale(a.partials, inv(3 * cv^2))))
end

# Absolute value and sign
@inline function Base.abs(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(abs(a.value), _fwd_scale(a.partials, sign(a.value)))
end
@inline function Base.abs2(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(abs2(a.value), _fwd_scale(a.partials, 2 * a.value))
end
Base.sign(a::NDual{T,N}) where {T,N} = NDual{T,N}(sign(a.value), _fwd_zero(Val(N), T))

# sincos — fused sin+cos; returns (sin(a), cos(a)) as a tuple of 
@inline function Base.sincos(a::NDual{T,N}) where {T,N}
    sv, cv = sincos(a.value)
    return NDual{T,N}(sv, _fwd_scale(a.partials, cv)),
    NDual{T,N}(cv, _fwd_scale(a.partials, -sv))
end

# sincospi shares the value and derivative factor, halving transcendental work.
@inline function Base.sinpi(a::NDual{T,N}) where {T,N}
    sv, cv = sincospi(a.value)
    return NDual{T,N}(sv, _fwd_scale(a.partials, T(π) * cv))
end
@inline function Base.cospi(a::NDual{T,N}) where {T,N}
    sv, cv = sincospi(a.value)
    return NDual{T,N}(cv, _fwd_scale(a.partials, -T(π) * sv))
end

# tanpi(x) = tan(π·x); derivative = π·sec²(π·x) = π·(1 + tan²(π·x)).
@inline function Base.tanpi(a::NDual{T,N}) where {T<:IEEEFloat,N}
    v = tanpi(a.value)
    return NDual{T,N}(v, _fwd_guarded_scale(a.partials, T(π) * (one(T) + v^2)))
end

# sincospi — fused sin(π·x)+cos(π·x); each derivative gains a π factor.
@inline function Base.sincospi(a::NDual{T,N}) where {T<:IEEEFloat,N}
    sv, cv = sincospi(a.value)
    return NDual{T,N}(sv, _fwd_scale(a.partials, T(π) * cv)),
    NDual{T,N}(cv, _fwd_scale(a.partials, -T(π) * sv))
end

# Reciprocal trigonometric: sec, csc, cot and their inverses.
@inline function Base.sec(a::NDual{T,N}) where {T,N}
    sv = sec(a.value)
    return NDual{T,N}(sv, _fwd_guarded_scale(a.partials, sv * tan(a.value)))
end
@inline function Base.csc(a::NDual{T,N}) where {T,N}
    cv = csc(a.value)
    return NDual{T,N}(cv, _fwd_guarded_scale(a.partials, -cv * cot(a.value)))
end
@inline function Base.cot(a::NDual{T,N}) where {T,N}
    cv = cot(a.value)
    return NDual{T,N}(cv, _fwd_guarded_scale(a.partials, -(one(T) + cv^2)))
end

@inline function Base.acot(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(acot(a.value), _fwd_scale(a.partials, -inv(one(T) + a.value^2)))
end

# Degree-based trigonometric functions — argument in degrees, derivative gains π/180.
@inline function Base.sind(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sind(a.value), _fwd_scale(a.partials, T(deg2rad(cosd(a.value)))))
end
@inline function Base.cosd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(cosd(a.value), _fwd_scale(a.partials, T(-deg2rad(sind(a.value)))))
end
@inline function Base.tand(a::NDual{T,N}) where {T,N}
    tv = tand(a.value)
    return NDual{T,N}(tv, _fwd_guarded_scale(a.partials, T(deg2rad(one(T) + tv^2))))
end
@inline function Base.secd(a::NDual{T,N}) where {T,N}
    sv = secd(a.value)
    return NDual{T,N}(sv, _fwd_guarded_scale(a.partials, T(deg2rad(sv * tand(a.value)))))
end
@inline function Base.cscd(a::NDual{T,N}) where {T,N}
    cv = cscd(a.value)
    return NDual{T,N}(cv, _fwd_guarded_scale(a.partials, T(-deg2rad(cv * cotd(a.value)))))
end
@inline function Base.cotd(a::NDual{T,N}) where {T,N}
    cv = cotd(a.value)
    return NDual{T,N}(cv, _fwd_guarded_scale(a.partials, T(-deg2rad(one(T) + cv^2))))
end

@inline function Base.atand(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        atand(a.value), _fwd_scale(a.partials, inv(T(deg2rad(one(T) + a.value^2))))
    )
end

@inline function Base.acotd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        acotd(a.value), _fwd_scale(a.partials, -inv(T(deg2rad(one(T) + a.value^2))))
    )
end

# Angle unit conversions — linear transforms; derivative is the constant scale factor.
@inline function Base.deg2rad(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(deg2rad(a.value), _fwd_scale(a.partials, T(deg2rad(one(T)))))
end
@inline function Base.rad2deg(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(rad2deg(a.value), _fwd_scale(a.partials, T(rad2deg(one(T)))))
end

# sinc(x) = sin(πx)/(πx) for x≠0, 1 at x=0; derivative = cosc(x).
@inline function Base.sinc(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sinc(a.value), _fwd_scale(a.partials, T(cosc(a.value))))
end

# hypot — d/da hypot(a,b) = a / hypot(a,b), d/db = b / hypot(a,b).
@inline function Base.hypot(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    h = hypot(a.value, b.value)
    coeff_a = _nfwd_zero_mask(a.value, a.value / h)
    coeff_b = _nfwd_zero_mask(b.value, b.value / h)
    return NDual{T,N}(
        h, _fwd_add(_fwd_scale(a.partials, coeff_a), _fwd_scale(b.partials, coeff_b))
    )
end
@inline Base.hypot(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = hypot(
    _promote_matching_nduals(:hypot, a, b)...
)
@inline Base.hypot(a::NDual{T,N}) where {T,N} = abs(a)
@inline function Base.hypot(
    a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}, xs::Vararg{NDual{T,N},M}
) where {T,N,M}
    return hypot(hypot(a, b), c, xs...)
end

# True to take `a`. `isequal` against the already-computed result, never an ordering test:
# `min(-0.0, 0.0)` is `-0.0` while `-0.0 < 0.0` is false and `-0.0 == 0.0` is true. The two differ
# only on a tie, where `max` keeps `b` and `min` keeps `a`.
@inline function _ndual_pick_max(a, b)
    v = max(a, b)
    return isequal(v, a) & !isequal(v, b)
end

@inline function _ndual_pick_min(a, b)
    v = min(a, b)
    return isequal(v, a) | !isequal(v, b)
end

# Preserve Base's NaN/signed-zero result; ties credit b for max, a for min.
# Selecting the whole dual is safe only because _ndual_pick_* uses isequal against
# the primal result; a bare ordering comparison would break the inner-value invariant.
@inline function Base.max(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(_ndual_pick_max(a.value, b.value), a, b)
end
@inline function Base.min(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(_ndual_pick_min(a.value, b.value), a, b)
end

# FastMath primitives have different tie semantics from min/max. Evaluate them directly
# to preserve platform-dependent signed zeros, selecting only partials. Equal-value ties
# credit b except for max_fast before Julia 1.12. NaN is outside FastMath's contract.
@inline function Base.FastMath.min_fast(a::NDual{T,N}, b::NDual{T,N}) where {T<:IEEEFloat,N}
    v = Base.FastMath.min_fast(a.value, b.value)
    pick_a = isequal(v, a.value) & !isequal(v, b.value)
    return NDual{T,N}(v, ifelse(pick_a, a.partials, b.partials))
end
@inline function Base.FastMath.max_fast(a::NDual{T,N}, b::NDual{T,N}) where {T<:IEEEFloat,N}
    v = Base.FastMath.max_fast(a.value, b.value)
    @static if VERSION >= v"1.12-"
        pick_a = isequal(v, a.value) & !isequal(v, b.value)
    else
        pick_a = isequal(v, a.value) | !isequal(v, b.value)
    end
    return NDual{T,N}(v, ifelse(pick_a, a.partials, b.partials))
end
@inline function Base.FastMath.minmax_fast(
    a::NDual{T,N}, b::NDual{T,N}
) where {T<:IEEEFloat,N}
    return (Base.FastMath.min_fast(a, b), Base.FastMath.max_fast(a, b))
end
# rem_fast uses rem_internal on absolute values: take its primal directly (including
# DivideError at zero divisors), and guard the rem coefficient when trunc(a/b) is infinite.
@inline function Base.FastMath.rem_fast(a::NDual{T,N}, b::NDual{T,N}) where {T<:IEEEFloat,N}
    v = Base.FastMath.rem_fast(a.value, b.value)
    c = trunc(a.value / b.value)
    return NDual{T,N}(v, _fwd_add(a.partials, _fwd_guarded_scale(b.partials, -c)))
end

# Use Base's primal for promotion, signed zeros and crossed bounds; non-strict tangent
# comparisons match the reverse rule at endpoints. ifelse avoids GPU warp divergence.
@inline function Base.clamp(a::NDual{T,N}, lo::NDual{T,N}, hi::NDual{T,N}) where {T,N}
    # Test the upper bound first to match Base for crossed bounds; non-strict comparisons
    # retain the reverse rule's endpoint convention.
    above = a.value >= hi.value
    below = (a.value <= lo.value) & !above
    src = ifelse(above, hi, ifelse(below, lo, a))
    return NDual{T,N}(clamp(a.value, lo.value, hi.value), src.partials)
end
# Explicit mixed bounds avoid promoting T with NDual (which would nest duals).
# Promote the plain bound rather than narrowing it, and retain the dual bound's tangent.
@inline function Base.clamp(a::NDual{T,N}, lo::NDual{T,N}, hi::H) where {T,N,H<:Real}
    S = promote_type(T, H)
    return clamp(convert(NDual{S,N}, a), convert(NDual{S,N}, lo), NDual{S,N}(S(hi)))
end
@inline function Base.clamp(a::NDual{T,N}, lo::L, hi::NDual{T,N}) where {T,N,L<:Real}
    S = promote_type(T, L)
    return clamp(convert(NDual{S,N}, a), NDual{S,N}(S(lo)), convert(NDual{S,N}, hi))
end
@inline function Base.clamp(a::NDual{T,N}, lo::L, hi::H) where {T,N,L<:Real,H<:Real}
    S = promote_type(T, L, H)
    av, loS, hiS = S(a.value), S(lo), S(hi)
    interior = (av > loS) & (av < hiS)
    p = ntuple(i -> ifelse(interior, S(a.partials[i]), zero(S)), Val(N))
    return NDual{S,N}(clamp(av, loS, hiS), p)
end

# flipsign / copysign — sign of result determined by primal; tangent follows.
@inline function Base.flipsign(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(signbit(b.value), -a, a)
end
@inline function Base.copysign(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(signbit(a.value) == signbit(b.value), a, -a)
end

# ── Real / imag / conj — for Complex{NDual} to compose generically ────────────────
# A NDual is always the "real part" of itself; conj is the identity for reals.

@inline Base.real(a::NDual) = a
Base.imag(a::NDual{T,N}) where {T,N} = zero(NDual{T,N})
@inline Base.conj(a::NDual) = a
Base.reim(a::NDual{T,N}) where {T,N} = (a, zero(NDual{T,N}))
Base.isreal(::NDual) = true

# ── LinearAlgebra.dot specialisation ──────────────────────────────────────────────
# The generic AbstractArray fallback calls dot(x[i], y[i]) per element via an
# out-of-line function (sret convention for large structs), which prevents LLVM from
# fusing the inner loop.  For NDual{T,8} each element costs 2×72-byte memcpys plus
# an external call.  This specialisation keeps the loop body inlinable so LLVM can
# vectorise the partials accumulation.
@inline function LinearAlgebra.dot(
    x::StridedVector{NDual{T,N}}, y::StridedVector{NDual{T,N}}
) where {T,N}
    lx = length(x)
    lx == length(y) || throw(
        DimensionMismatch(
            lazy"first array has length $(lx) which does not match the length of the second, $(length(y)).",
        ),
    )
    lx == 0 && return NDual{T,N}(zero(T))
    @inbounds s = x[1] * y[1]
    @inbounds for i in 2:lx
        s = muladd(x[i], y[i], s)
    end
    return s
end

# ── LinearAlgebra.ldiv for LU{T} with NDual rhs ──────────────────────────────────
# The generic ldiv(F::Factorization, B) (non-mutating) converts the factorization to
# LU{NDual} before calling ldiv!, allocating a full Matrix{NDual} for no reason.
# The generic ldiv!(A::LU, B::AbstractVecOrMat) already handles mixed element types
# (Float64 factors, NDual rhs) correctly via _apply_ipiv_rows! + triangular solves.
# Override ldiv to bypass the conversion and call ldiv! directly.
@inline function LinearAlgebra.ldiv(
    F::LinearAlgebra.LU{T,<:AbstractMatrix{T}}, b::AbstractVector{<:NDual{T}}
) where {T}
    bb = copy(b)
    LinearAlgebra.ldiv!(F, bb)
    return bb
end

# ── Comparisons (on value only — for control flow in kernels) ──────────────────────

Base.:<(a::NDual, b::NDual) = a.value < b.value
Base.:>(a::NDual, b::NDual) = a.value > b.value
Base.:<=(a::NDual, b::NDual) = a.value <= b.value
Base.:>=(a::NDual, b::NDual) = a.value >= b.value
Base.:(==)(a::NDual, b::NDual) = a.value == b.value
Base.isless(a::NDual, b::NDual) = isless(a.value, b.value)

# NDual vs plain Real: compare value directly, avoiding zero-partial NDual construction
# via promotion.  The NDual×NDual methods above are more specific and still win when
# both sides are NDual.
Base.:<(a::NDual, b::Real) = a.value < b
Base.:>(a::NDual, b::Real) = a.value > b
Base.:<=(a::NDual, b::Real) = a.value <= b
Base.:>=(a::NDual, b::Real) = a.value >= b
Base.:<(a::Real, b::NDual) = a < b.value
Base.:>(a::Real, b::NDual) = a > b.value
Base.:<=(a::Real, b::NDual) = a <= b.value
Base.:>=(a::Real, b::NDual) = a >= b.value
Base.:<(a::NDual, b::Rational) = a.value < b
Base.:<(a::Rational, b::NDual) = a < b.value
Base.:<=(a::NDual, b::Rational) = a.value <= b
Base.:<=(a::Rational, b::NDual) = a <= b.value
Base.:<=(a::NDual, b::AbstractIrrational) = a.value <= b
Base.:<=(a::AbstractIrrational, b::NDual) = a <= b.value
Base.isnan(a::NDual) = isnan(a.value)
Base.isinf(a::NDual) = isinf(a.value)
Base.isfinite(a::NDual) = isfinite(a.value)
Base.signbit(a::NDual) = signbit(a.value)

# ── Utility ───────────────────────────────────────────────────────────────────────
Base.eps(d::NDual) = eps(d.value)
Base.eps(::Type{NDual{T,N}}) where {T,N} = eps(T)
# Value-only, like every other predicate on an `NDual` and like `Base.iszero` on every other
# `Number`, where it means `x == zero(x)`. Consulting the partials made it disagree both with this
# type's own `==`/`isequal`/`hash` (all value-only) and with the primal, so a body branching on it
# took a different branch under nfwd than the primal did: `h(t) = iszero(t) ? one(t) : sin(t)/t`
# returned NaN at `t == 0`, where the primal is 1.0.
@inline Base.iszero(d::NDual) = iszero(d.value)
Base.hash(d::NDual, hsh::UInt) = hash(d.value, hsh)

# ── ifelse ────────────────────────────────────────────────────────────────────────
# Standard subgradient convention: branch on primal, propagate selected tangent.

Base.ifelse(c::Bool, a::NDual{T,N}, b::NDual{T,N}) where {T,N} = c ? a : b
Base.complex(re::NDual{T,N}, im::NDual{T,N}) where {T,N} = Complex{NDual{T,N}}(re, im)

# ── Complex{NDual} math — explicit GPU-safe implementations ───────────────────────
# Julia's generic Complex math (sin, cos, exp, log, sqrt) calls float(T::Type) and
# has isnan-guard branches that do not compile cleanly to PTX for custom T.
# Explicit implementations use only NDual scalar ops and compile without issues.

@inline function Base.abs(z::Complex{NDual{T,N}}) where {T,N}
    return hypot(real(z), imag(z))
end
@inline function Base.abs2(z::Complex{NDual{T,N}}) where {T,N}
    return real(z)^2 + imag(z)^2
end
@inline function Base.conj(z::Complex{NDual{T,N}}) where {T,N}
    return Complex(real(z), -imag(z))
end

# sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
@inline function Base.sin(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    sa, ca = sincos(a)
    return Complex(sa * cosh(b), ca * sinh(b))
end

# cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
@inline function Base.cos(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    sa, ca = sincos(a)
    return Complex(ca * cosh(b), -(sa * sinh(b)))
end

# exp(a + bi) = exp(a)·(cos(b) + i·sin(b))
@inline function Base.exp(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    er = exp(a)
    sb, cb = sincos(b)
    return Complex(er * cb, er * sb)
end

# log(a + bi) = log(|z|) + i·atan(b, a)
@inline function Base.log(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    return Complex(log(hypot(a, b)), atan(b, a))
end

# sqrt(a + bi) = sqrt((|z|+a)/2) + i·sign(b)·sqrt((|z|-a)/2)
# Construct the NDual arguments to sqrt directly with _fwd_scale to avoid an
# unnecessary NDual*NDual product-rule evaluation (the factor 0.5 has zero partials).
@inline function Base.sqrt(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    r = hypot(a, b)
    half = T(0.5)
    re = sqrt(
        NDual{T,N}(
            (r.value + a.value) * half, _fwd_scale(_fwd_add(r.partials, a.partials), half)
        ),
    )
    im =
        copysign(one(NDual{T,N}), b) * sqrt(
            NDual{T,N}(
                (r.value - a.value) * half,
                _fwd_scale(_fwd_sub(r.partials, a.partials), half),
            ),
        )
    return Complex(re, im)
end

# tan(z) = sin(z)/cos(z)
@inline function Base.tan(z::Complex{NDual{T,N}}) where {T,N}
    return sin(z) / cos(z)
end

# ── Unsupported-operation error ───────────────────────────────────────────────────
# Operations that would silently destroy partial information (integer/rounding ops,
# integer division, modulo) throw a clear error instead of falling through to a
# confusing MethodError or, worse, silently dropping gradients.
#
# If you hit this, the function you are differentiating calls one of these
# non-differentiable operations on a floating-point argument.
# Options:
#   • Replace the operation with a differentiable approximation.
#   • Mark that argument as non-differentiable so NDual wrapping is skipped.
#   • Open an issue if you believe the operation should have a subgradient rule.

struct NDualUnsupportedError <: Exception
    op::Symbol
end
@inline function Base.showerror(io::IO, e::NDualUnsupportedError)
    return _nfwd_print_boxed_error(
        io,
        [
            "NDual does not support `$(e.op)`.",
            "This operation cannot propagate partial derivatives.",
            "Use a differentiable alternative, or open an issue if a subgradient rule makes sense.",
        ],
    )
end

# Keep the integer-conversion entrypoints explicit as well. These are the user-facing
# typed rounding paths (`floor(Int, x)`, `round(Int, x)`, etc.) and should fail with the
# same NDual-specific error instead of falling through to AbstractFloat methods.
for _op in (:floor, :ceil, :round, :trunc)
    @eval Base.$_op(::Type{I}, ::NDual{T,N}) where {I<:Union{Signed,Unsigned},T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
end

# Rounding ops have zero partial derivatives (piecewise constant). Define specific methods
# so that functions like `modf` (which calls `trunc`) work through NDual on the CPU.
for _op in (:floor, :ceil, :trunc)
    @eval function Base.$_op(x::NDual{T,N}) where {T<:IEEEFloat,N}
        return NDual{T,N}(Base.$_op(ndual_value(x)), ntuple(_ -> zero(T), Val(N)))
    end
end
@inline Base.round(x::NDual{T,N}) where {T<:IEEEFloat,N} = NDual{T,N}(
    round(ndual_value(x)), ntuple(_ -> zero(T), Val(N))
)
for _r in (
    RoundNearest,
    RoundNearestTiesAway,
    RoundNearestTiesUp,
    RoundToZero,
    RoundFromZero,
    RoundUp,
    RoundDown,
)
    @eval @inline function Base.round(x::NDual{T,N}, ::typeof($_r)) where {T<:IEEEFloat,N}
        return NDual{T,N}(round(ndual_value(x), $_r), ntuple(_ -> zero(T), Val(N)))
    end
    @eval @inline function Base.round(
        ::Type{I}, x::NDual{T,N}, ::typeof($_r)
    ) where {I<:Union{Signed,Unsigned},T<:IEEEFloat,N}
        throw(NDualUnsupportedError(:round))
    end
end

for _op in (:div, :fld, :cld, :gcd, :lcm)
    @eval Base.$_op(x::NDual{T,N}) where {T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
    @eval Base.$_op(x::NDual{T,N}, y::Real) where {T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
    @eval Base.$_op(x::Real, y::NDual{T,N}) where {T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
    @eval Base.$_op(x::NDual{T,N}, y::NDual{S,M}) where {T<:IEEEFloat,S<:IEEEFloat,N,M} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
end

# rem uses -trunc(x/y), not mod's -floor(x/y), and keeps the finite one-sided
# subgradient at integer ratios for modf. Two NDual arguments also disambiguate Base.
@inline function Base.rem(x::NDual{T,N}, y::NDual{T,N}) where {T<:IEEEFloat,N}
    pv, yv = ndual_value(x), ndual_value(y)
    c = trunc(pv / yv)
    # Guard inactive lanes if pv/yv overflows while the primal remains finite.
    return NDual{T,N}(rem(pv, yv), _fwd_add(x.partials, _fwd_guarded_scale(y.partials, -c)))
end

@inline function Base.mod(x::NDual{T,N}, y::NDual{T,N}) where {T<:IEEEFloat,N}
    coeff_x, coeff_y = _nfwd_mod_grad_coeffs(x.value, y.value)
    return NDual{T,N}(
        mod(x.value, y.value),
        _fwd_add(
            _fwd_guarded_scale(x.partials, coeff_x), _fwd_guarded_scale(y.partials, coeff_y)
        ),
    )
end
@inline Base.mod(x::NDual{T1,N1}, y::NDual{T2,N2}) where {T1<:IEEEFloat,T2<:IEEEFloat,N1,N2} = mod(
    _promote_matching_nduals(:mod, x, y)...
)
@inline Base.mod(x::NDual{T,N}) where {T<:IEEEFloat,N} = throw(NDualUnsupportedError(:mod))
@inline Base.mod(x::NDual{T,N}, y::Real) where {T<:IEEEFloat,N} = throw(
    NDualUnsupportedError(:mod)
)
@inline Base.mod(x::Real, y::NDual{T,N}) where {T<:IEEEFloat,N} = throw(
    NDualUnsupportedError(:mod)
)

@inline function Base.mod2pi(x::NDual{T,N}) where {T<:IEEEFloat,N}
    coeff = _nfwd_mod2pi_grad(x.value)
    return NDual{T,N}(mod2pi(x.value), _fwd_guarded_scale(x.partials, coeff))
end

# ── Future: tiled GPU kernels with NDual ──────────────────────────────────────────
#
# The current broadcast AD uses one NDual{T,N} per thread: every thread computes
# the primal and all N partials in registers in a single kernel pass.  This is
# already efficient for small N and element-wise functions.  For larger N or
# functions with cross-element data reuse (reductions, softmax, layer norm),
# *tiled* kernels offer further gains:
#
# ── Conceptual note: tiling applied to the NDual itself ─────────────────────
# An NDual{T,N} is a tile in the partial-derivative dimension.  Just as spatial
# tiling partitions an M-element array into ceil(M/K) tiles of width K — each
# processed in one pass with data reuse in shared memory — slot-tiling partitions
# the N-wide NDual into ceil(N/K) tiles of width K, each processed in one kernel
# launch:
#
#   Jf = [∂f/∂x₁  ∂f/∂x₂  …  ∂f/∂xₙ]          (1×N Jacobian row per element)
#
#   tile b covers columns  [(b-1)K+1, min(bK, N)]
#   each thread carries   NDual{T,K}  with those K slots live, rest zero
#
# The primal f(x) is recomputed in each of the ceil(N/K) launches (cost), but
# register usage per thread drops from O(N) to O(K), restoring warp occupancy.
# This is the GPU spatial analogue of ForwardDiff's CPU chunk mode, where N is
# the Jacobian width and K is the chunk size.
#
# ── N vs D ───────────────────────────────────────────────────────────────────
# With D differentiable input parameters, the total slot count is
#   N = Σᵢ dimension(inputᵢ),   dimension = 1 (real),  dimension = 2 (complex)
# so N ≥ D in general.  For all-real inputs dimension = 1 for every input and N = D
# exactly — this is a consequence of the slot definition, not a separate choice.
# The tiling logic is uniform over N regardless; the real/complex distinction
# only affects how N is computed from D (via _broadcast_elem_dof_type).
#
# ── Slot-tiled execution (reduce register pressure for large N) ───────────────
#    Background: with D differentiable inputs, the total slot count is
#    N = Σᵢ dimension(inputᵢ) where dimension = 1 (real) or 2 (complex).  Currently every
#    thread carries ONE NDual{T,N} whose N partials cover ALL D inputs at once.
#
#    Slot-tiling partitions those N slots across ceil(N/K) kernel launches:
#      batch b → slots (b-1)K+1 .. bK:  only these inputs wrapped as NDual{T,K},
#                                        all others passed as plain T.
#    Each thread carries NDual{T,K} instead of NDual{T,N}, using (K+1)·(sizeof T/4)
#    registers instead of (N+1)·(sizeof T/4).  Partial results from each batch are
#    assembled into the full gradient vector after all ceil(N/K) launches complete.
#
#    Cost: ceil(N/K) re-evaluations of f on the same input data.
#    Useful when N > ~8 and register pressure is reducing warp occupancy.
#    Ref: CUDA occupancy calculator —
#    https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy
#
# ── Memory complexity: forward (NDual) vs reverse mode ───────────────────────
# Let M = number of output elements in the broadcast (length of y in y .= f.(args...)).
# Each of the M output elements is computed by one GPU thread carrying NDual{T,N},
# so the output dual array has M·(N+1) scalars.  For N total slots (one pass, K=N):
#
#   Forward (NDual):   O(M·N·sizeof T)   — write N gradient arrays of length M;
#                                          no tape, sequential coalesced access.
#   Reverse mode:      O(M·N·sizeof T)   — same gradient storage,
#                      + O(M·depth)       — forward tape for backward pass
#                                          (random-access reads, cache-unfriendly).
#
# Both are O(M·N) in gradient storage, but reverse mode carries an additional
# tape term proportional to the computation graph depth.  For shallow element-wise
# broadcasts (depth ~ constant) this is negligible; for deep networks it dominates.
# NDual avoids the tape entirely at the cost of recomputing the primal ceil(N/K)
# times when tiling is used:
#
#   Tiled forward:     O(M·K·sizeof T)   peak memory per launch (K < N),
#                      ceil(N/K) passes  over input data.

# ── Cholesky factorization for matrices of NDuals ─────────────────────────────────

function LinearAlgebra.cholesky(
    A::AbstractMatrix{NDual{T,N}},
    (::LinearAlgebra.NoPivot)=LinearAlgebra.NoPivot();
    check::Bool=true,
) where {T<:IEEEFloat,N}
    # Generic Cholesky on NDuals matches the analytic pushforward to machine precision
    # and is faster for small matrices than primal factorisation plus per-lane solves.
    return @invoke LinearAlgebra.cholesky(A::AbstractMatrix, LinearAlgebra.NoPivot(); check)
end

# Hermitian and Symmetric wrappers: materialise the symmetric view, then defer to
# the Matrix{NDual} method above.
#
# Critically, we call `copytri!` to mirror the active triangle before passing the
# matrix to `cholesky(::Matrix{NDual})`.  Without this, the `_cholesky_ndual_fwd`
# helper would see raw (unmirrored) NDual values for the inactive triangle, producing
# wrong partial derivatives for those positions.  For real NDual, Hermitian and
# Symmetric are equivalent (conj is identity), so a single `copytri!` suffices.
for _WrapType in (:Hermitian, :Symmetric)
    @eval function LinearAlgebra.cholesky(
        A::LinearAlgebra.$_WrapType{NDual{T,N},<:Matrix{NDual{T,N}}},
        (::LinearAlgebra.NoPivot)=LinearAlgebra.NoPivot();
        check::Bool=true,
    ) where {T<:IEEEFloat,N}
        data = LinearAlgebra.copytri!(copy(A.data), A.uplo)
        return LinearAlgebra.cholesky(data, LinearAlgebra.NoPivot(); check)
    end
end

# ── Symmetric / Hermitian matrix operations for NDual ────────────────────────────────
#
# LinearAlgebra's BLAS-backed `mul!` specialisations don't accept NDual elements.
# Materialise the lazy symmetric/hermitian wrapper to a plain Matrix{NDual} before
# dispatching, so the generic (non-BLAS) matrix multiply is used.
for _WrapType in (:Symmetric, :Hermitian)
    @eval begin
        # AbstractVecOrMat covers vectors; the AbstractMatrix overloads below resolve
        # the ambiguity with LinearAlgebra's *(AbstractMatrix, AbstractMatrix) when B
        # is a plain Matrix (LinearAlgebra wins on B, we win on A — ambiguous without
        # a more specific method that wins on both).
        # `Matrix`, not `AbstractMatrix`: a wider bound claimed device- and sparse-backed
        # parents these methods cannot serve, colliding with cuSPARSE and cuSOLVER.
        function Base.:*(
            A::LinearAlgebra.$_WrapType{NDual{T,N},<:Matrix{NDual{T,N}}},
            B::Union{StridedVector,StridedMatrix},
        ) where {T<:IEEEFloat,N}
            return Matrix(A) * B
        end

        function Base.:*(
            A::Union{StridedVector,StridedMatrix},
            B::LinearAlgebra.$_WrapType{NDual{T,N},<:Matrix{NDual{T,N}}},
        ) where {T<:IEEEFloat,N}
            return A * Matrix(B)
        end
    end
end

# logdet(Cholesky{NDual}): 2·∑ᵢ log(Lᵢᵢ).
# The generic LinearAlgebra path reaches this formula via `sum(log, diag(C.L))`, but
# `diag(LowerTriangular{NDual})` may trigger a BLAS-adjacent specialisation.  Spelling
# it out explicitly avoids that ambiguity.
function LinearAlgebra.logdet(
    C::LinearAlgebra.Cholesky{NDual{T,N},Matrix{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    L = C.L
    n = size(L, 1)
    s = log(L[1, 1])
    for i in 2:n
        s = s + log(L[i, i])
    end
    return 2 * s
end

# ── Array reductions for NDual ─────────────────────────────────────────────────
# Base's generic mapreduce_impl is @noinline, which prevents the compiler from
# fusing the inner reduction loop with surrounding code.  For NDual{T,N} compound
# types this is particularly costly: the noinline barrier defeats register-level
# accumulation of all N partial slots simultaneously.  These inlineable overrides
# replace the barrier with a simple sequential left-fold that LLVM can optimise.
@inline function _ndual_mapreduce_impl(
    f::F, op::O, A::AbstractArray{<:NDual{T,N}}, ifirst::Integer, ilast::Integer
) where {F,O,T,N}
    ifirst > ilast && return Base.mapreduce_empty(f, op, eltype(A))
    @inbounds acc = f(A[ifirst])
    @inbounds for i in (ifirst + 1):ilast
        acc = op(acc, f(A[i]))
    end
    return acc
end

@inline function Base.mapreduce_impl(
    f::F, op::O, A::AbstractArray{<:NDual{T,N}}, ifirst::Integer, ilast::Integer
) where {F,O,T,N}
    return _ndual_mapreduce_impl(f, op, A, ifirst, ilast)
end

@inline function Base.mapreduce_impl(
    f::F,
    op::Union{typeof(max),typeof(min)},
    A::AbstractArray{<:NDual{T,N}},
    ifirst::Int,
    ilast::Int,
) where {F,T,N}
    return _ndual_mapreduce_impl(f, op, A, ifirst, ilast)
end

# 6-arg form (blksize is unused; pairwise recursion is never beneficial for NDual).
@inline function Base.mapreduce_impl(
    f::F, op::O, A::AbstractArray{<:NDual{T,N}}, ifirst::Integer, ilast::Integer, ::Int
) where {F,O,T,N}
    return Base.mapreduce_impl(f, op, A, ifirst, ilast)
end

@inline function _nfwd_check_chunk_size(chunk_size::Integer)
    chunk_size > 0 && return Int(chunk_size)
    throw(ArgumentError("`chunk_size` must be a positive integer, got $chunk_size."))
end

# Conservative SIMD-friendly default: 8 lanes covers one AVX-512 register (8×Float64)
# and two AVX2 registers. Chunk sizes beyond 8 add register pressure without
# proportional throughput gains on most hardware.
const _NFWD_PREFERRED_CHUNK_SIZE = 8

@inline function _nfwd_default_chunk_size(x::Tuple)
    # init=0 gives zero-argument callables chunk size 1, matching primal_dim(::Tuple).
    return max(1, min(sum(primal_dim, x; init=0), _NFWD_PREFERRED_CHUNK_SIZE))
end

# Type-level dimension: returns the number of differentiable scalar components for a
# concrete type, or `nothing` when the size cannot be determined from the type
# alone (e.g. heap-allocated Array whose length is a runtime value).
@inline static_primal_dim(::Type{<:IEEEFloat}) = 1
@inline static_primal_dim(::Type{<:Complex{<:IEEEFloat}}) = 2
# Propagate `nothing` (not `0 + nothing`, which would throw) when any element's size is not
# type-determinable — e.g. a tuple containing an Array.
@inline function static_primal_dim(T::Type{<:Tuple})
    total = 0
    for P in T.parameters
        d = static_primal_dim(P)
        d === nothing && return nothing
        total += d
    end
    return total
end
@inline static_primal_dim(::Type{<:AbstractArray}) = nothing
@inline static_primal_dim(::Type) = 0

@inline _nfwd_is_supported_scalar(::Type{<:IEEEFloat}) = true
@inline _nfwd_is_supported_scalar(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nfwd_is_supported_scalar(::Type) = false

@inline _nfwd_tuple_primal_supported(::Tuple{}) = true
@inline function _nfwd_tuple_primal_supported(x::Tuple)
    return _nfwd_is_supported_primal(first(x)) && _nfwd_tuple_primal_supported(Base.tail(x))
end

@inline _nfwd_is_supported_primal(::IEEEFloat) = true
@inline _nfwd_is_supported_primal(::Complex{<:IEEEFloat}) = true
@inline function _nfwd_is_supported_primal(x::Array{ET}) where {ET}
    _nfwd_is_supported_scalar(ET)
end
@inline _nfwd_is_supported_primal(x::Tuple) = _nfwd_tuple_primal_supported(x)
@inline _nfwd_is_supported_primal(::Any) = false

abstract type UnsupportedError <: Exception end

struct UnsupportedInputError <: UnsupportedError
    msg::String
end

struct UnsupportedOutputError <: UnsupportedError
    msg::String
end

# Keep diagnostics local so Nfwd can load without Mooncake.
@inline function _nfwd_boxed_message_width(io::IO, prefix::AbstractString)
    return max(20, displaysize(io)[2] - textwidth(prefix))
end

function _nfwd_wrap_boxed_line(line, width::Int)
    text = string(line)
    (isempty(text) || width < 1 || textwidth(text) <= width) && return [text]

    wrapped = String[]
    remaining = SubString(text)
    while textwidth(remaining) > width
        split_idx = nothing
        for idx in eachindex(remaining)
            textwidth(SubString(remaining, 1, idx)) > width && break
            remaining[idx] == ' ' && (split_idx = idx)
        end
        if isnothing(split_idx)
            split_idx = firstindex(remaining)
            for idx in eachindex(remaining)
                textwidth(SubString(remaining, firstindex(remaining), idx)) > width && break
                split_idx = idx
            end
        end
        push!(wrapped, rstrip(SubString(remaining, firstindex(remaining), split_idx)))
        remaining = lstrip(SubString(remaining, nextind(remaining, split_idx)))
        isempty(remaining) && break
    end
    isempty(remaining) || push!(wrapped, remaining)
    return wrapped
end

function _nfwd_print_boxed_error(
    io::IO, lines; indent::AbstractString="  ", footer=nothing, newline::Bool=false
)
    first_item = iterate(lines)
    isnothing(first_item) && return nothing
    line, state = first_item
    rest_prefix = indent * "│ "
    first_width = _nfwd_boxed_message_width(io, "")
    rest_width = _nfwd_boxed_message_width(io, rest_prefix)
    first_wrapped = _nfwd_wrap_boxed_line(line, first_width)
    println(io, first(first_wrapped))
    for wrapped_line in @view first_wrapped[2:end]
        println(io, rest_prefix, wrapped_line)
    end
    while true
        item = iterate(lines, state)
        isnothing(item) && break
        line, state = item
        for wrapped_line in _nfwd_wrap_boxed_line(line, rest_width)
            println(io, rest_prefix, wrapped_line)
        end
    end
    print(io, indent, "└")
    isnothing(footer) || print(io, ' ', footer)
    newline && println(io)
    return nothing
end

@inline function Base.showerror(
    io::IO, err::Union{UnsupportedInputError,UnsupportedOutputError}
)
    return _nfwd_print_boxed_error(io, split(err.msg, '\n'))
end

@inline _nfwd_supported_input_summary() = "IEEEFloat scalars, Complex{<:IEEEFloat} scalars, and dense Arrays with those element types"

@inline _nfwd_supported_output_summary() = "IEEEFloat scalars, Complex{<:IEEEFloat} scalars, dense Arrays with those element types, and tuples thereof"

@inline _nfwd_shape_summary(::IEEEFloat) = "scalar"
@inline _nfwd_shape_summary(::Complex{<:IEEEFloat}) = "scalar"
@inline _nfwd_shape_summary(x::AbstractArray) = "size $(size(x))"
@inline _nfwd_shape_summary(x::Tuple) = "tuple length $(length(x))"
@inline _nfwd_shape_summary(::Any) = "not size-bearing"

@inline _nfwd_value_summary(x) = "$(typeof(x)) ($(_nfwd_shape_summary(x)))"

@inline function _nfwd_inputs_summary(xs::Tuple)
    isempty(xs) && return "  (none)"
    return join(ntuple(i -> "  $i. $(_nfwd_value_summary(xs[i]))", Val(length(xs))), '\n')
end

@inline _nfwd_input_error(x) = throw(
    UnsupportedInputError(
        "nfwd input unsupported.\n" *
        "Supported nfwd inputs: $(_nfwd_supported_input_summary()).\n" *
        "Input:\n" *
        "  $(_nfwd_value_summary(x))",
    ),
)

@inline function _nfwd_output_error(y)
    throw(
        UnsupportedOutputError(
            "nfwd output unsupported.\n" *
            "Supported nfwd inputs: $(_nfwd_supported_input_summary()).\n" *
            "Supported nfwd outputs: $(_nfwd_supported_output_summary()).\n" *
            "Output:\n" *
            "  $(_nfwd_value_summary(y))",
        ),
    )
end

@inline function _nfwd_output_error(xs::Tuple, y)
    throw(
        UnsupportedOutputError(
            "nfwd output unsupported.\n" *
            "Supported nfwd inputs: $(_nfwd_supported_input_summary()).\n" *
            "Supported nfwd outputs: $(_nfwd_supported_output_summary()).\n" *
            "Inputs:\n" *
            "$(_nfwd_inputs_summary(xs))\n" *
            "Output:\n" *
            "  $(_nfwd_value_summary(y))",
        ),
    )
end

#
# ── Canonical slot traversal ──────────────────────────────────────────────────────
#
# Every supported nfwd primal decomposes into a fixed number of scalar "slots" in
# a canonical order:
#   • IEEEFloat              → 1 slot  (the value itself)
#   • Complex{<:IEEEFloat}   → 2 slots (real, imag)
#   • AbstractArray{<:above} → one slot per scalar component, in eachindex order
#   • Tuple of the above     → concatenation, left to right
#
# `_nfwd_fold_slots` and `_nfwd_unfold_slots` define this order exactly once.
# All dimension counting, basis seeding, and gradient scatter must use these helpers so
# that the slot order is guaranteed to agree everywhere.

"""
    _nfwd_fold_slots(f, init, x, state) -> (acc, state)

Left-fold over the scalar slots of `x` in canonical order.  Each slot corresponds
to one differentiable scalar degree of freedom.  Real IEEE-float values contribute
one floating-point slot.  Complex IEEE-float values contribute two scalar slots,
visited as real then imaginary.  Tuples are visited left to right, and arrays are
visited in `eachindex` order.

Each slot visit calls `(acc, state) = f(acc, x_leaf, slot_index_within_leaf, state)`
and returns the updated accumulator and state.  The slot cursor should be threaded
through `state` by the caller.
"""
@inline function _nfwd_fold_slots(f::F, init, x::IEEEFloat, state) where {F}
    return f(init, x, 1, state)
end

@inline function _nfwd_fold_slots(f::F, init, x::Complex{<:IEEEFloat}, state) where {F}
    acc, state = f(init, x, 1, state)  # real part
    return f(acc, x, 2, state)          # imag part
end

@inline function _nfwd_fold_slots(
    f::F, init, x::AbstractArray{T}, state
) where {F,T<:IEEEFloat}
    acc = init
    @inbounds for i in eachindex(x)
        acc, state = f(acc, x, i, state)
    end
    return acc, state
end

@inline function _nfwd_fold_slots(
    f::F, init, x::AbstractArray{Complex{T}}, state
) where {F,T<:IEEEFloat}
    acc = init
    @inbounds for i in eachindex(x)
        acc, state = f(acc, x, 2i - 1, state)  # real part of element i
        acc, state = f(acc, x, 2i, state)       # imag part of element i
    end
    return acc, state
end

@inline _nfwd_fold_slots(f::F, init, x::Tuple{}, state) where {F} = (init, state)
@inline function _nfwd_fold_slots(f::F, init, x::Tuple, state) where {F}
    acc, state = _nfwd_fold_slots(f, init, first(x), state)
    return _nfwd_fold_slots(f, acc, Base.tail(x), state)
end

"""
    _nfwd_unfold_slots(f, x, state) -> (rebuilt, state)

Map-like structural rebuild over the primitive leaves of `x`.  Each slot
corresponds to one differentiable scalar degree of freedom (same semantics as
`_nfwd_fold_slots`).  Each leaf visit calls `(result, state) = f(x_leaf, state)` and
returns the rebuilt value for that leaf position.  For tuples, the unfold recurses
left to right and collects per-leaf results into a new tuple.

The returned value at each leaf position may have a different type from the input
leaf (e.g. seeding produces NTuples from scalar inputs).  The slot cursor should
be threaded through `state` by the caller.

`_nfwd_fold_slots` and `_nfwd_unfold_slots` agree on traversal order: tuples left to right,
arrays in `eachindex` order.  Within each leaf, the number of slots consumed equals
`primal_dim(leaf)`.
"""
@inline function _nfwd_unfold_slots(
    f::F,
    x::Union{
        IEEEFloat,
        Complex{<:IEEEFloat},
        AbstractArray{<:IEEEFloat},
        AbstractArray{<:Complex{<:IEEEFloat}},
    },
    state,
) where {F}
    return f(x, state)
end

@inline _nfwd_unfold_slots(f::F, x::Tuple{}, state) where {F} = ((), state)
@inline function _nfwd_unfold_slots(f::F, x::Tuple, state) where {F}
    head, state = _nfwd_unfold_slots(f, first(x), state)
    tail, state = _nfwd_unfold_slots(f, Base.tail(x), state)
    return (head, tail...), state
end

# ── dimension counting ─────────────────────────────────────────────────────────────────

@inline primal_dim(x::IEEEFloat) = 1
@inline primal_dim(x::Complex{<:IEEEFloat}) = 2
@inline primal_dim(x::AbstractArray{<:IEEEFloat}) = length(x)
@inline primal_dim(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 2 * length(x)
@inline primal_dim(x::Tuple) = sum(primal_dim, x; init=0)

# ──────────────────────────────────────────────────────────────────────────
# `NDualArray{Element,N,D,A,Wrapped,B}` is the canonical array V.
# The primal aliases user storage and remains usable by BLAS/LAPACK; partials are slot-local.
# B = _block_type(A) holds lanes in element-major shape (N, size(primal)...), so each
# element's lanes are contiguous. Memory primals use rank-2 blocks.
# Wrapped is NDual{T,N} for real T, Complex{NDual{T,N}} for Complex{T}; element access
# constructs it lazily while the AbstractArray supertype preserves generic dispatch.
# Legacy a.partials consumers receive strided lane views. Consumers needing dense arrays
# (ccall/pointer/setfield!) must use the block. Mooncake methods live in tangents/lifted.jl.

const NDualEltype = Union{IEEEFloat,Complex{<:IEEEFloat}}

# The coherent wrapped-element type for an `NDualArray` of `Element` at width `N` — what
# `getindex` actually returns. Used by the inner constructor to reject an incoherent `Wrapped`.
@inline _wrapped_eltype(::Type{Element}, ::Val{N}) where {Element<:IEEEFloat,N} = NDual{
    Element,N
}
@inline function _wrapped_eltype(
    ::Type{Element}, ::Val{N}
) where {T<:IEEEFloat,Element<:Complex{T},N}
    return Complex{NDual{T,N}}
end

# ──────────────────────────────────────────────────────────────────────────
# `NDualBlock{T, D}` — the element-major partials block, one type on every supported Julia.
#
# A header over a flat Vector keeps storage resizable on Julia 1.10; Base.reshape
# would mark it shared and make resize throw. The immutable header lets field loads
# hoist; _derived_dims recomputes the trailing dimension from the parent length,
# so Vector primal growth needs no mutable shape field.
# DenseArray membership is essential for StridedArray/BLAS dispatch (otherwise
# mul! uses the slower generic kernel). Storage is contiguous with unit first stride.

struct NDualBlock{T,D} <: DenseArray{T,D}
    parent::Vector{T}
    dims::NTuple{D,Int}
end
@inline function NDualBlock{T,D}(::UndefInitializer, dims::Vararg{Int,D}) where {T,D}
    return NDualBlock{T,D}(Vector{T}(undef, prod(dims)), dims)
end

# `dims` with the trailing entry recomputed from the parent's current length. A zero leading
# product means an empty block, which never grows, so the stored entry stands.
@inline function _derived_dims(dims::NTuple{D,Int}, len::Int) where {D}
    lead = 1
    @inbounds for i in 1:(D - 1)
        lead *= dims[i]
    end
    lead == 0 && return dims
    return (Base.front(dims)..., len ÷ lead)
end

@inline function Base.size(b::NDualBlock)
    return _derived_dims(getfield(b, :dims), length(getfield(b, :parent)))
end
Base.length(b::NDualBlock) = length(getfield(b, :parent))
Base.IndexStyle(::Type{<:NDualBlock}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(b::NDualBlock, i::Int) = getfield(b, :parent)[i]
Base.@propagate_inbounds function Base.setindex!(b::NDualBlock, v, i::Int)
    getfield(b, :parent)[i] = v
    return b
end
Base.strides(b::NDualBlock) = Base.size_to_strides(1, size(b)...)
Base.elsize(::Type{<:NDualBlock{T}}) where {T} = sizeof(T)
# `cconvert` hands a `ccall` the parent `Array` rather than the block, so the argument the GC roots
# is a real heap object and the block header itself need never be boxed.
Base.cconvert(::Type{Ptr{T}}, b::NDualBlock{T}) where {T} = getfield(b, :parent)
function Base.unsafe_convert(::Type{Ptr{T}}, b::NDualBlock{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, getfield(b, :parent))
end
Base.dataids(b::NDualBlock) = Base.dataids(getfield(b, :parent))
Base.copy(b::NDualBlock) = NDualBlock(copy(getfield(b, :parent)), getfield(b, :dims))
Base.fill!(b::NDualBlock, x) = (fill!(getfield(b, :parent), x); b)
function Base.convert(::Type{NDualBlock{S,D}}, b::NDualBlock{T,D}) where {S,T,D}
    return NDualBlock{S,D}(convert(Vector{S}, getfield(b, :parent)), getfield(b, :dims))
end
# Reshaping a block is a new header over the same parent, so it neither copies nor marks the
# parent shared (which would break the resize primitives on 1.10).
function Base.reshape(b::NDualBlock{T}, dims::NTuple{D,Int}) where {T,D}
    return NDualBlock{T,D}(getfield(b, :parent), dims)
end

# Access the flat parent to keep hot stores on one array; the fallback serves CUDA blocks.
@inline _block_storage(b::NDualBlock) = getfield(b, :parent)
@inline _block_storage(b) = b

# Each primal element owns N block entries; resizing the parent updates the trailing shape.
@inline function _resize_block!(resize!!, b::NDualBlock, N::Int, d::Integer)
    resize!!(getfield(b, :parent), N * d)
    return b
end

@static if VERSION >= v"1.11-rc4"  # `Memory` does not exist on Julia 1.10.
    # A window into `memblock` covering the `prod(dims) ÷ N` columns starting at the backing
    # `Memory`'s slot `off`. The block is element-major, so slot j owns flat entries
    # `(j-1)*N+1 : j*N`; sharing that storage is what makes a lane written through an `Array`'s V
    # visible through its backing `Memory`'s V, mirroring the primal aliasing.
    #
    # Growth needs nothing extra. The block's `Memory` is the primal's scaled by `N`, so the slack
    # on each side of the window is `N` times the primal's, and `d ≤ slack` holds for the primal
    # exactly when `N*d ≤ N*slack` holds for the block: `_growend!`/`_growbeg!` reallocate on
    # precisely the same calls, and a reallocated primal has no lifted `Memory` left to share with.
    # An empty window has no slot to address: offsetting to `(off-1)*N+1` would build a
    # one-past-end ref, which `--check-bounds=yes` rejects however the call is annotated. `wrap`
    # never dereferences it at length 0, so the parent's own ref serves.
    @inline function _window_block(
        memblock::NDualBlock{E,2}, ::Val{N}, off::Int, dims::NTuple{D,Int}
    ) where {E,N,D}
        parent_ref = getfield(getfield(memblock, :parent), :ref)
        len = prod(dims)
        ref = if iszero(len)
            parent_ref
        else
            Core.memoryrefnew(parent_ref, (off - 1) * N + 1, false)
        end
        return NDualBlock{E,D}(Base.wrap(Array, ref, (len,))::Vector{E}, dims)
    end
end

# Unknown containers must define _block_type rather than silently getting CPU storage.
# Backends with different orientation override _lane_index, tangent_view and _lane_views.
# Memory is one-dimensional and uses a rank-2 block.
@inline _block_type(::Type{Array{Element,D}}) where {Element,D} = NDualBlock{Element,D + 1}
@static if VERSION >= v"1.11-rc4"  # `Memory` does not exist on Julia 1.10.
    @inline _block_type(::Type{Memory{Element}}) where {Element} = NDualBlock{Element,2}
end
@inline _block_dims(N::Int, p) = (N, size(p)...)
@inline _block_shape_ok(block, N::Int, p) = size(block) == (N, size(p)...)

@noinline function _throw_block_shape_error(block_size, N, primal_size)
    return throw(
        DimensionMismatch(
            "NDualArray partials block has size $block_size; expected $N lanes over a " *
            "primal of size $primal_size.",
        ),
    )
end

struct NDualArray{
    Element<:NDualEltype,N,D,A<:AbstractArray{Element,D},Wrapped,B<:AbstractArray{Element}
} <: AbstractArray{Wrapped,D}
    primal::A
    partials_block::B
    # Enforce coherent element and block types; static checks constant-fold, shape is O(1).
    function NDualArray{Element,N,D,A,Wrapped,B}(
        primal::A, partials_block::B
    ) where {
        Element<:NDualEltype,
        N,
        D,
        A<:AbstractArray{Element,D},
        Wrapped,
        B<:AbstractArray{Element},
    }
        Wrapped === _wrapped_eltype(Element, Val(N)) || throw(
            ArgumentError(
                "NDualArray Wrapped parameter $Wrapped is incoherent: expected " *
                "$(_wrapped_eltype(Element, Val(N))) for Element=$Element, N=$N.",
            ),
        )
        B === _block_type(A) || throw(
            ArgumentError(
                "NDualArray block parameter $B is incoherent: expected " *
                "$(_block_type(A)) for A=$A.",
            ),
        )
        # `_block_shape_ok` encodes the block orientation per backend: element-major
        # `(N, dims...)` on the host, and lane-major `(dims..., N)` for `CuArray` (overridden
        # in the CUDA extension, where lanes must be contiguous). The `B === _block_type(A)`
        # check above already pins the block type.
        # `_throw_block_shape_error` takes the block's size, never the block: a call taking a
        # freshly built block escapes it, and the caller's array header then survives -- one
        # allocation per `.mem` projection, which element-wise access performs per element.
        _block_shape_ok(partials_block, N, primal) ||
            _throw_block_shape_error(size(partials_block), N, size(primal))
        return new{Element,N,D,A,Wrapped,B}(primal, partials_block)
    end
end

# Pack per-lane same-shape arrays into a fresh element-major block: lane `k` of element
# `j` lands at linear block index `(j - 1) * N + k` (the block's leading dim is the lane).
@inline function _pack_block(
    p::A, ts::NTuple{N,<:AbstractArray{Element}}
) where {Element,N,A<:AbstractArray{Element}}
    block = _block_type(A)(undef, _block_dims(N, p)...)
    @inbounds for k in 1:N
        tk = ts[k]
        for j in 1:length(p)
            block[(j - 1) * N + k] = tk[j]
        end
    end
    return block
end

# Tuple-accepting outer constructors pack the incoming per-lane arrays into the block, so
# pre-existing construction sites (including `typeof(v)(p, ts)`) keep working unchanged.
# The 4-parameter form fills in `Wrapped` via `_wrapped_eltype` (`NDual{Element,N}` for real
# `Element`, `Complex{NDual{T,N}}` for `Element === Complex{T}`) and `B` via `_block_type`;
# the 5-/6-parameter forms keep the caller's `Wrapped` (and `B`) so the inner constructor
# still rejects incoherent parameters.
@inline function NDualArray{Element,N,D,A}(
    p::A, ts::NTuple{N,<:AbstractArray{Element}}
) where {Element<:NDualEltype,N,D,A<:AbstractArray{Element,D}}
    return NDualArray{Element,N,D,A,_wrapped_eltype(Element, Val(N)),_block_type(A)}(
        p, _pack_block(p, ts)
    )
end
@inline function NDualArray{Element,N,D,A,Wrapped}(
    p::A, ts::NTuple{N,<:AbstractArray{Element}}
) where {Element<:NDualEltype,N,D,A<:AbstractArray{Element,D},Wrapped}
    return NDualArray{Element,N,D,A,Wrapped,_block_type(A)}(p, _pack_block(p, ts))
end
@inline function NDualArray{Element,N,D,A,Wrapped,B}(
    p::A, ts::NTuple{N,<:AbstractArray{Element}}
) where {Element<:NDualEltype,N,D,A<:AbstractArray{Element,D},Wrapped,B}
    return NDualArray{Element,N,D,A,Wrapped,B}(p, _pack_block(p, ts)::B)
end

# Block-accepting outer constructor: adopt an existing element-major block AS the partials
# storage (no copy — the caller passes a block deliberately, usually to share storage with
# another V so mutations through either are visible through both).
@inline function NDualArray{Element,N,D,A}(
    p::A, block::AbstractArray{Element}
) where {Element<:NDualEltype,N,D,A<:AbstractArray{Element,D}}
    return NDualArray{Element,N,D,A,_wrapped_eltype(Element, Val(N)),_block_type(A)}(
        p, block
    )
end

# Zero-init seed: allocate a fresh slot-local zero block matching the primal.
@inline function NDualArray{Element,N,D,A}(
    p::A
) where {Element<:NDualEltype,N,D,A<:AbstractArray{Element,D}}
    block = fill!(_block_type(A)(undef, _block_dims(N, p)...), zero(Element))
    return NDualArray{Element,N,D,A,_wrapped_eltype(Element, Val(N)),_block_type(A)}(
        p, block
    )
end

# The canonical NDualArray V-type for a primal array type `A` at width `N` — the single source
# of the type's parameter arity, so `dual_type`/`lifted_type` don't spell out the block param.
@inline _ndual_array_V(::Type{A}, ::Val{N}) where {Element<:NDualEltype,D,A<:AbstractArray{Element,D},N} = NDualArray{
    Element,N,D,A,_wrapped_eltype(Element, Val(N)),_block_type(A)
}

# Centralise element-major lane addressing here; CUDA overrides it for lane-major
# (dims..., N) storage. Duplicating the formula would silently address the wrong lane.
@inline _lane_index(::NDualArray{Element,N}, elem::Int, lane::Int) where {Element,N} =
    (elem - 1) * N + lane

# Inlined seed access lets getfield hoist out of gradient/Jacobian loops.
@inline function _zero_seed!(a::NDualArray{Element}) where {Element}
    fill!(getfield(a, :partials_block), zero(Element))
    return a
end
@inline _get_partial(a::NDualArray, elem::Int, lane::Int) = @inbounds _block_storage(
    getfield(a, :partials_block)
)[_lane_index(a, elem, lane)]
@inline function _set_partial!(a::NDualArray, elem::Int, lane::Int, v)
    @inbounds _block_storage(getfield(a, :partials_block))[_lane_index(a, elem, lane)] = v
    return a
end

# Per-lane strided views write through to the block; tangent_view builds just one lane.
@inline function _lane_views(a::NDualArray{Element,N,D}) where {Element,N,D}
    block = getfield(a, :partials_block)
    colons = ntuple(_ -> Colon(), Val(D))
    return ntuple(k -> view(block, k, colons...), Val(N))
end

# Write-through lane view, unlike tangent(x, lane)'s dense reverse-shaped copy.
@inline tangent_view(a::NDualArray{Element,N,D}, k::Integer) where {Element,N,D} = view(
    getfield(a, :partials_block), k, ntuple(_ -> Colon(), Val(D))...
)

# Shape follows the primal; resize mutates primal and block together while preserving
# the immutable wrapper's field identities.
Base.size(a::NDualArray) = size(a.primal)
function Base.IndexStyle(::Type{<:NDualArray{<:Any,<:Any,<:Any,A}}) where {A}
    return IndexStyle(A)
end

# The primal's linear index for `i`, which `_lane_index` turns into a block address.
# `LinearIndices` accepts both the linear and the cartesian index forms and bounds-checks them,
# so the block access itself can be `@inbounds`.
@inline _linear_elem(a::NDualArray, i::Vararg{Int}) = LinearIndices(getfield(a, :primal))[i...]

# Element-wise reduction over a flat block. Reading each element's lane column as one
# `NTuple{N,Element}` load matters twice over: `getindex` would reload the block's parent pointer
# per element, and `N` separate scalar loads do not vectorise. Together those cost ~25% at width
# 8. Wrapper primals and GPU blocks keep the generic `getindex` path.
@inline function _ndual_mapreduce_impl(
    f::F,
    op::O,
    A::NDualArray{Element,N,D,<:Array,W,<:NDualBlock},
    ifirst::Integer,
    ilast::Integer,
) where {F,O,Element,N,D,W}
    ifirst > ilast && return Base.mapreduce_empty(f, op, eltype(A))
    i0, i1 = Int(ifirst), Int(ilast)
    p = getfield(A, :primal)
    cols = reinterpret(NTuple{N,Element}, getfield(getfield(A, :partials_block), :parent))
    @inbounds acc = f(_scalar_ndual(p[i0], cols[i0]))
    @inbounds for i in (i0 + 1):i1
        acc = op(acc, f(_scalar_ndual(p[i], cols[i])))
    end
    return acc
end

# _scalar_ndual handles real/complex elements; setindex! dispatches on their distinct V types.
@inline function Base.getindex(a::NDualArray{Element,N}, i::Vararg{Int}) where {Element,N}
    block = _block_storage(getfield(a, :partials_block))
    e = _linear_elem(a, i...)
    return _scalar_ndual(
        a.primal[i...], ntuple(k -> @inbounds(block[_lane_index(a, e, k)]), Val(N))
    )
end
@inline function Base.setindex!(
    a::NDualArray{Element,N}, x::NDual{Element,N}, i::Vararg{Int}
) where {Element<:IEEEFloat,N}
    a.primal[i...] = x.value
    block = _block_storage(getfield(a, :partials_block))
    e = _linear_elem(a, i...)
    @inbounds for k in 1:N
        block[_lane_index(a, e, k)] = x.partials[k]
    end
    return a
end
# Complex eltype: the V element is `Complex{NDual{T,N}}` (real/imag each an `NDual`). The primal
# and the block hold `Complex{T}`, so split/recombine the real and imaginary parts.
@inline function Base.setindex!(
    a::NDualArray{Element,N}, x::Complex{NDual{T,N}}, i::Vararg{Int}
) where {T<:IEEEFloat,Element<:Complex{T},N}
    a.primal[i...] = Complex(x.re.value, x.im.value)
    block = _block_storage(getfield(a, :partials_block))
    e = _linear_elem(a, i...)
    @inbounds for k in 1:N
        block[_lane_index(a, e, k)] = Complex(x.re.partials[k], x.im.partials[k])
    end
    return a
end
# A real dual stored into a complex array (e.g. `copyto!(::complex, ::real)`) promotes: the
# imaginary part of the value and of every partial is zero.
@inline function Base.setindex!(
    a::NDualArray{Element,N}, x::NDual{T,N}, i::Vararg{Int}
) where {T<:IEEEFloat,Element<:Complex{T},N}
    a.primal[i...] = Complex(x.value)
    block = _block_storage(getfield(a, :partials_block))
    e = _linear_elem(a, i...)
    @inbounds for k in 1:N
        block[_lane_index(a, e, k)] = Complex(x.partials[k])
    end
    return a
end
# No element-type convert: it would copy the primal and sever user-storage aliasing.
# Converting stores must rebuild V over the destination (as IdDict setindex! does),
# or fail with MethodError. Scalar NDual conversion is safe because storage is inline.

# Scan the real primal then index one dual (~10x faster than folding NDual elements).
# max ties credit the LAST element, unlike argmax. Use isequal to distinguish signed zeros.
function Base.maximum(nda::NDualArray{E}) where {E<:IEEEFloat}
    p = getfield(nda, :primal)
    @inbounds nda[findlast(isequal(maximum(p)), p)]
end
# min ties credit the FIRST element; findfirst's vectorised scan beats argmin/findmin.
function Base.minimum(nda::NDualArray{E}) where {E<:IEEEFloat}
    p = getfield(nda, :primal)
    @inbounds nda[findfirst(isequal(minimum(p)), p)]
end

# ──────────────────────────────────────────────────────────────────────────
# `NDualRef{P,N}` keeps partials in a separate Ref for parallel raw-pointer access.
# Its distinct type prevents generic struct re-lifting; seeds, materialisation, field
# access and pointer rules handle it explicitly. The primal Ref lives in Lifted.
export NDualRef
struct NDualRef{P<:NDualEltype,N}
    partials::Base.RefValue{NTuple{N,P}}
    # Suppress the implicit constructor: at N=0 its unbound P would fail Aqua.
    function NDualRef{P,N}(partials::Base.RefValue{NTuple{N,P}}) where {P<:NDualEltype,N}
        return new{P,N}(partials)
    end
end
# Zero-init seed: fresh slot-local partials.
@inline function NDualRef{P,N}() where {P<:NDualEltype,N}
    return NDualRef{P,N}(Base.RefValue{NTuple{N,P}}(ntuple(_ -> zero(P), Val(N))))
end

# ──────────────────────────────────────────────────────────────────────────
# `NDualMemoryRef{Element,N,M}` covers MemoryRef, which is not an AbstractArray.
# Keep the primal ref usable by ccall and the shared (N,ncols) partials block as a flat
# backing ref at column 1, lane 1. col selects the element; memoryrefnew advances it by i-1.
# Sharing with Memory/Array V preserves mutation aliasing. Column col+j pairs with primal
# slot memoryrefoffset(primal)+j: Memory factories cover the full backing, whereas
# Array-projected refs start at that array's first element.

@static if VERSION >= v"1.11-rc4"
    export NDualMemoryRef

    # A flat MemoryRef keeps the V type rank-independent and Array.ref projection alloc-free.
    # Reconstruct shaped blocks only for bulk operations, off the hot path.
    struct NDualMemoryRef{Element<:NDualEltype,N,M<:Memory{Element}}
        primal::MemoryRef{Element}
        partials_ref::MemoryRef{Element}
        ncols::Int
        col::Int
        function NDualMemoryRef{Element,N,M}(
            primal::MemoryRef{Element},
            partials_ref::MemoryRef{Element},
            ncols::Int,
            col::Int,
        ) where {Element<:NDualEltype,N,M<:Memory{Element}}
            # Validate against seeded columns, not backing capacity: Array refs may have slack.
            # Column ncols+1 is formable but not dereferenceable, including an empty array's ref.
            1 <= col <= ncols + 1 || throw(
                ArgumentError(
                    "NDualMemoryRef column $col is past the $ncols partials columns. The " *
                    "offset is inside the backing `Memory` but past the seeded array's " *
                    "length, so there is no partials column for it.",
                ),
            )
            return new{Element,N,M}(primal, partials_ref, ncols, col)
        end
    end

    # Convenience for the block-shaped construction sites and the Memory-seed factory:
    # validate the block's lane rows, then adopt its backing ref and column count.
    @inline function NDualMemoryRef{Element,N,M}(
        primal::MemoryRef{Element}, block::NDualBlock{Element,2}, col::Int
    ) where {Element<:NDualEltype,N,M<:Memory{Element}}
        size(block, 1) == N || throw(
            DimensionMismatch(
                "NDualMemoryRef partials block has $(size(block, 1)) rows; " *
                "expected the chunk width N = $N (lane-leading layout).",
            ),
        )
        return NDualMemoryRef{Element,N,M}(
            primal, getfield(getfield(block, :parent), :ref), size(block, 2), col
        )
    end

    # Zero-init seed: a fresh zero block covering the whole backing `Memory` (column j ↔ mem
    # slot j), so the referenced element's column is the ref's offset. Element types in
    # `NDualEltype` are bits types, so undef iteration is not needed.
    @inline function NDualMemoryRef{Element,N,M}(
        p::MemoryRef{Element}
    ) where {Element<:NDualEltype,N,M<:Memory{Element}}
        block = fill!(NDualBlock{Element,2}(undef, N, length(p.mem)), zero(Element))
        return NDualMemoryRef{Element,N,M}(p, block, Core.memoryrefoffset(p))
    end

    # Column col starts at lane 1; n columns occupy n*N contiguous entries.
    # Skip bounds checks: construction confines col to 1:ncols+1 and lanes stay in 1:N.
    # Dereferencing the end ref is primal UB; checking every lane would tax scalar access.
    @inline function _block_column_ref(partials_ref::MemoryRef, col::Int, N::Int)
        col == 1 && return partials_ref
        return Core.memoryrefnew(partials_ref, (col - 1) * N + 1, false)
    end

    # Read the `N` contiguous lanes starting at `colref` (position 1 == `colref` itself), and
    # write them. Alloc-free (`memoryref` ops + an isbits `ntuple`), for the hot get/set frules.
    @inline function _read_lanes(colref::MemoryRef, ::Val{N}) where {N}
        return ntuple(
            k -> Core.memoryrefget(Core.memoryrefnew(colref, k, false), :not_atomic, false),
            Val(N),
        )
    end
    @inline function _write_lanes!(colref::MemoryRef, vals, ::Val{N}) where {N}
        for k in 1:N
            Core.memoryrefset!(
                Core.memoryrefnew(colref, k, false), vals[k], :not_atomic, false
            )
        end
        return nothing
    end

    # Reconstruct the whole `(N, ncols)` block sharing the ref's backing (bulk/interface ops
    # only — off the hot path). `partials_ref` is col-1 lane-1, so wrapping `N * ncols` entries
    # from it reproduces the original block exactly.
    @inline function _reconstruct_block(v::NDualMemoryRef{E,N}) where {E,N}
        ncols = getfield(v, :ncols)
        flat = Base.wrap(Array, getfield(v, :partials_ref), (N * ncols,))::Vector{E}
        return NDualBlock{E,2}(flat, (N, ncols))
    end
end

end
