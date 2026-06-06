# Let x be a TwicePrecision{<:IEEEFloat}, then the `Float64` associated to x is roughly
# `x.hi + x.lo`.

#
# Implementation of tangent for TwicePrecision. Let `x` be the number that a given
# TwicePrecision represents, and its fields be `hi` and `lo`. Since `x = hi + lo`, we should
# not think of `TwicePrecision` as a struct with two fields, but as a single number. As
# such, we need to be careful to ensure that tangents of `TwicePrecision`s do not depend on
# the values of `hi` and `lo`, but on their sum.
#

const TwicePrecisionFloat{P<:IEEEFloat} = TwicePrecision{P}
const TWP{P} = TwicePrecisionFloat{P}

@foldable tangent_type(P::Type{<:TWP}) = P

zero_tangent_internal(::TWP{F}, ::MaybeCache) where {F} = TWP{F}(zero(F), zero(F))

function randn_tangent_internal(rng::AbstractRNG, p::TWP{F}, ::MaybeCache) where {F}
    return TWP{F}(randn(rng, F), randn(rng, F))
end

import .TestUtils: has_equal_data_internal
function has_equal_data_internal(
    p::P, q::P, ::Bool, ::Dict{Tuple{UInt,UInt},Bool}
) where {P<:TWP}
    return Float64(p) ≈ Float64(q)
end

increment_internal!!(::IncCache, t::T, s::T) where {T<:TWP} = t + s

set_to_zero_internal!!(::SetToZeroCache, t::TWP) = zero_tangent_internal(t, NoCache())

_add_to_primal_internal(::MaybeCache, p::P, t::P, ::Bool) where {P<:TWP} = p + t

tangent_to_primal_internal!!(x::P, t::P, ::MaybeCache) where {P<:TWP} = t
primal_to_tangent_internal!!(t::P, x::P, ::MaybeCache) where {P<:TWP} = x

_dot_internal(::MaybeCache, t::P, s::P) where {P<:TWP} = Float64(t) * Float64(s)

_scale_internal(::MaybeCache, a::Float64, t::TWP) = a * t

populate_address_map_internal(m::AddressMap, ::P, ::P) where {P<:TWP} = m
# Forward V `NTuple{N,TWP}` carries no mutable aliasing (TWP is immutable) — no-op, like the
# reverse `(::P, ::P)` above.
populate_address_map_internal(m::AddressMap, ::TWP, ::Tuple{Vararg{TWP}}) = m

fdata_type(::Type{<:TWP}) = NoFData

rdata_type(P::Type{<:TWP}) = P

__verify_fdata_value(::IdDict{Any,Nothing}, ::P, ::P) where {P<:TWP} = nothing

_verify_rdata_value(::P, ::P) where {P<:TWP} = nothing

@foldable tangent_type(::Type{NoFData}, T::Type{<:TWP}) = T

tangent(::NoFData, t::TWP) = t

zero_rdata(p::TWP) = zero_tangent(p)

zero_rdata_from_type(P::Type{<:TWP{F}}) where {F} = P(zero(F), zero(F))

#
# Forward-mode canonical V for `TwicePrecision{P}` — `NTuple{N, TWP{P}}`,
# i.e. `N` parallel TWP-typed partials, one per lane. Mirrors the
# reverse-mode `tangent_type(TWP) === TWP` shape at the per-lane level,
# echoing the Ptr V pattern (a single primitive-leaf type carrying N
# parallel copies). The structural-lift fallback doesn't apply here
# because TWP is conceptually a single number, not a struct with
# differentiable fields.
#

@foldable @inline function dual_type(
    ::Val{N}, ::Type{TwicePrecision{P}}
) where {N,P<:IEEEFloat}
    return NTuple{N,TwicePrecision{P}}
end
@foldable @inline function lifted_type(
    ::Val{N}, ::Type{TwicePrecision{P}}
) where {N,P<:IEEEFloat}
    return Lifted{TwicePrecision{P},N,NTuple{N,TwicePrecision{P}}}
end

# Forward seed/lift/unlift for the custom V `NTuple{N, TWP{P}}` (one TWP partial per lane),
# mirroring reverse `zero_tangent_internal` / `randn_tangent_internal`. The generic
# `@generated` struct-lift factory would build `TWP((NDual, NDual))` and hit the TWP
# constructor's `Float64(::NDual)` (NDual has no `Float64` conversion), so seed the per-lane
# tuple directly.
for f in (:_zero_dual_internal, :_uninit_dual_internal)
    @eval @inline function $f(::Val{N}, ::TWP{F}, ::MaybeCache) where {N,F}
        return ntuple(_ -> TWP{F}(zero(F), zero(F)), Val(N))
    end
end
@inline function _randn_dual_internal(
    ::Val{N}, rng::AbstractRNG, ::TWP{F}, ::MaybeCache
) where {N,F}
    return ntuple(_ -> TWP{F}(randn(rng, F), randn(rng, F)), Val(N))
end
@inline lift(x::TWP{F}, ẋ::TWP{F}) where {F} = Lifted{TWP{F},1}(x, (ẋ,))
# A TWP is conceptually a single number, so its width-1 V `Tuple{TWP}` is a leaf, not a
# structural tuple. Override `_unlift_seed` (which both the top-level `unlift` and nested
# fields — e.g. a `StepRangeLen`'s `ref`/`step` — route through) to read the lane directly,
# bypassing the per-field tuple path that would index the scalar primal.
@inline _unlift_seed(x::Lifted{P,1,Tuple{S}}, ::IdDict) where {P<:TWP,S<:TWP} = tangent(
    x, 1
)

#
# Rules. These are required for a lot of functionality in this case.
#

@is_primitive MinimalCtx Tuple{typeof(_new_),<:TWP,IEEEFloat,IEEEFloat}
function frule!!(
    ::Lifted{typeof(_new_),N},
    ::Lifted{Type{TWP{P}},N},
    hi::Lifted{P,N,NDual{P,N}},
    lo::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    x = _new_(TWP{P}, primal(hi), primal(lo))
    hi_parts = tangent(hi).partials
    lo_parts = tangent(lo).partials
    dx = ntuple(k -> _new_(TWP{P}, hi_parts[k], lo_parts[k]), Val(N))
    return Lifted{TWP{P},N}(x, dx)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{TWP{P}}}, hi::CoDual{P}, lo::CoDual{P}
) where {P<:IEEEFloat}
    _new_twice_precision_pb(dy::TWP{P}) = NoRData(), NoRData(), P(dy), P(dy)
    return zero_fcodual(_new_(TWP{P}, hi.x, lo.x)), _new_twice_precision_pb
end

@is_primitive MinimalCtx Tuple{typeof(twiceprecision),IEEEFloat,Integer}
function frule!!(
    ::Lifted{typeof(twiceprecision),N}, val::Lifted{P,N,NDual{P,N}}, nb::Lifted{<:Integer}
) where {N,P<:IEEEFloat}
    _nb = primal(nb)
    x = twiceprecision(primal(val), _nb)
    val_parts = tangent(val).partials
    dx = ntuple(k -> twiceprecision(val_parts[k], _nb), Val(N))
    return Lifted{TWP{P},N}(x, dx)
end
function rrule!!(
    ::CoDual{typeof(twiceprecision)}, val::CoDual{P}, nb::CoDual{<:Integer}
) where {P<:IEEEFloat}
    twiceprecision_float_pb(dy::TWP{P}) = NoRData(), P(dy), NoRData()
    return zero_fcodual(twiceprecision(val.x, nb.x)), twiceprecision_float_pb
end

@is_primitive MinimalCtx Tuple{typeof(twiceprecision),TWP,Integer}
function frule!!(
    ::Lifted{typeof(twiceprecision),N}, val::Lifted{P,N,NTuple{N,P}}, nb::Lifted{<:Integer}
) where {N,P<:TWP}
    _nb = primal(nb)
    x = twiceprecision(primal(val), _nb)
    val_parts = tangent(val)
    dx = ntuple(k -> twiceprecision(val_parts[k], _nb), Val(N))
    return Lifted{P,N}(x, dx)
end
function rrule!!(
    ::CoDual{typeof(twiceprecision)}, val::CoDual{P}, nb::CoDual{<:Integer}
) where {P<:TWP}
    twiceprecision_pb(dy::P) = NoRData(), dy, NoRData()
    return zero_fcodual(twiceprecision(val.x, nb.x)), twiceprecision_pb
end

@is_primitive MinimalCtx Tuple{Type{<:IEEEFloat},TWP}
function frule!!(
    ::Lifted{Type{P},N}, x::Lifted{S,N,NTuple{N,S}}
) where {N,P<:IEEEFloat,S<:TWP}
    y = P(primal(x))
    x_parts = tangent(x)
    dy = ntuple(k -> P(x_parts[k]), Val(N))
    return Lifted{P,N}(y, NDual{P,N}(y, dy))
end
function rrule!!(::CoDual{Type{P}}, x::CoDual{S}) where {P<:IEEEFloat,S<:TWP}
    float_from_twice_precision_pb(dy::P) = NoRData(), S(dy)
    return zero_fcodual(P(x.x)), float_from_twice_precision_pb
end

@is_primitive MinimalCtx Tuple{typeof(-),TWP}
function frule!!(::Lifted{typeof(-),N}, x::Lifted{P,N,NTuple{N,P}}) where {N,P<:TWP}
    y = -primal(x)
    x_parts = tangent(x)
    dy = ntuple(k -> -x_parts[k], Val(N))
    return Lifted{P,N}(y, dy)
end
function rrule!!(::CoDual{typeof(-)}, x::CoDual{P}) where {P<:TWP}
    negate_twice_precision_pb(dy::P) = NoRData(), -dy
    return zero_fcodual(-(x.x)), negate_twice_precision_pb
end

@is_primitive MinimalCtx Tuple{typeof(+),TWP,IEEEFloat}
function frule!!(
    ::Lifted{typeof(+),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{S,N,NDual{S,N}}
) where {N,P<:TWP,S<:IEEEFloat}
    z = primal(x) + primal(y)
    x_parts = tangent(x)
    y_parts = tangent(y).partials
    dz = ntuple(k -> x_parts[k] + y_parts[k], Val(N))
    return Lifted{P,N}(z, dz)
end
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{S}
) where {P<:TWP,S<:IEEEFloat}
    plus_pullback(dz::P) = NoRData(), dz, S(dz)
    return zero_fcodual(x.x + y.x), plus_pullback
end

@is_primitive(MinimalCtx, Tuple{typeof(+),P,P} where {P<:TWP})
function frule!!(
    ::Lifted{typeof(+),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{P,N,NTuple{N,P}}
) where {N,P<:TWP}
    z = primal(x) + primal(y)
    x_parts = tangent(x)
    y_parts = tangent(y)
    dz = ntuple(k -> x_parts[k] + y_parts[k], Val(N))
    return Lifted{P,N}(z, dz)
end
function rrule!!(::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{P}) where {P<:TWP}
    plus_pullback(dz::P) = NoRData(), dz, dz
    return zero_fcodual(x.x + y.x), plus_pullback
end

@is_primitive MinimalCtx Tuple{typeof(+),TWP,Integer}
function frule!!(
    ::Lifted{typeof(+),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{<:Integer}
) where {N,P<:TWP}
    z = primal(x) + primal(y)
    return Lifted{P,N}(z, tangent(x))
end
function rrule!!(::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{<:Integer}) where {P<:TWP}
    plus_twice_precision_integer_pb(dz::P) = NoRData(), dz, NoRData()
    return zero_fcodual(x.x + primal(y)), plus_twice_precision_integer_pb
end

@is_primitive MinimalCtx Tuple{typeof(*),TWP,IEEEFloat}
function frule!!(
    ::Lifted{typeof(*),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{S,N,NDual{S,N}}
) where {N,P<:TWP,S<:IEEEFloat}
    xp = primal(x)
    yp = primal(y)
    z = xp * yp
    x_parts = tangent(x)
    y_parts = tangent(y).partials
    dz = ntuple(k -> xp * y_parts[k] + x_parts[k] * yp, Val(N))
    return Lifted{P,N}(z, dz)
end
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{P}, y::CoDual{S}
) where {P<:TWP,S<:IEEEFloat}
    _x, _y = x.x, y.x
    mul_twice_precision_and_float_pb(dz::P) = NoRData(), dz * _y, S(dz * _x)
    return zero_fcodual(_x * _y), mul_twice_precision_and_float_pb
end

@is_primitive MinimalCtx Tuple{typeof(*),TWP,Integer}
function frule!!(
    ::Lifted{typeof(*),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{<:Integer}
) where {N,P<:TWP}
    yp = primal(y)
    z = primal(x) * yp
    x_parts = tangent(x)
    dz = ntuple(k -> x_parts[k] * yp, Val(N))
    return Lifted{P,N}(z, dz)
end
function rrule!!(::CoDual{typeof(*)}, x::CoDual{P}, y::CoDual{<:Integer}) where {P<:TWP}
    _y = y.x
    mul_twice_precision_and_int_pb(dz::P) = NoRData(), dz * _y, NoRData()
    return zero_fcodual(x.x * _y), mul_twice_precision_and_int_pb
end

@is_primitive MinimalCtx Tuple{typeof(/),TWP,IEEEFloat}
function frule!!(
    ::Lifted{typeof(/),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{S,N,NDual{S,N}}
) where {N,P<:TWP,S<:IEEEFloat}
    xp = primal(x)
    yp = primal(y)
    z = xp / yp
    x_parts = tangent(x)
    y_parts = tangent(y).partials
    dz = ntuple(k -> x_parts[k] / yp - y_parts[k] * xp / yp^2, Val(N))
    return Lifted{P,N}(z, dz)
end
function rrule!!(
    ::CoDual{typeof(/)}, x::CoDual{P}, y::CoDual{S}
) where {P<:TWP,S<:IEEEFloat}
    _x, _y = x.x, y.x
    div_twice_precision_and_float_pb(dz::P) = NoRData(), dz / _y, S(-dz * _x / _y^2)
    return zero_fcodual(_x / _y), div_twice_precision_and_float_pb
end

@is_primitive MinimalCtx Tuple{typeof(/),TWP,Integer}
function frule!!(
    ::Lifted{typeof(/),N}, x::Lifted{P,N,NTuple{N,P}}, y::Lifted{<:Integer}
) where {N,P<:TWP}
    yp = primal(y)
    z = primal(x) / yp
    x_parts = tangent(x)
    dz = ntuple(k -> x_parts[k] / yp, Val(N))
    return Lifted{P,N}(z, dz)
end
function rrule!!(::CoDual{typeof(/)}, x::CoDual{P}, y::CoDual{<:Integer}) where {P<:TWP}
    _y = y.x
    div_twice_precision_and_int_pb(dz::P) = NoRData(), dz / _y, NoRData()
    return zero_fcodual(x.x / _y), div_twice_precision_and_int_pb
end

# Primitives

@zero_derivative MinimalCtx Tuple{Type{<:TwicePrecision},Tuple{Integer,Integer},Integer}
@zero_derivative MinimalCtx Tuple{typeof(Base.splitprec),Type,Integer}
@zero_derivative(
    MinimalCtx,
    Tuple{typeof(Base.floatrange),Type{<:IEEEFloat},Integer,Integer,Integer,Integer},
)
@zero_derivative(
    MinimalCtx,
    Tuple{typeof(Base._linspace),Type{<:IEEEFloat},Integer,Integer,Integer,Integer},
)

using Base: range_start_step_length
@is_primitive(
    MinimalCtx, Tuple{typeof(range_start_step_length),T,T,Integer} where {T<:IEEEFloat}
)
function frule!!(
    ::Lifted{typeof(range_start_step_length),N},
    a::Lifted{T,N,NDual{T,N}},
    st::Lifted{T,N,NDual{T,N}},
    len::Lifted{<:Integer},
) where {N,T<:IEEEFloat}
    y = range_start_step_length(primal(a), primal(st), primal(len))
    a_parts = tangent(a).partials
    st_parts = tangent(st).partials
    ref_v = ntuple(k -> TWP{T}(a_parts[k], zero(T)), Val(N))
    step_v = ntuple(k -> TWP{T}(st_parts[k], zero(T)), Val(N))
    nt = (ref=ref_v, step=step_v, len=NoDual(), offset=NoDual())
    return Lifted{typeof(y),N}(y, ImmutableDual(nt))
end
function rrule!!(
    ::CoDual{typeof(range_start_step_length)},
    a::CoDual{T},
    st::CoDual{T},
    len::CoDual{<:Integer},
) where {T<:IEEEFloat}
    pb(dz) = NoRData(), T(dz.data.ref), T(dz.data.step), NoRData()
    return zero_fcodual(range_start_step_length(a.x, st.x, len.x)), pb
end

using Base: unsafe_getindex
const TWPStepRangeLen = StepRangeLen{<:Any,<:TWP,<:TWP}
@is_primitive(MinimalCtx, Tuple{typeof(unsafe_getindex),TWPStepRangeLen,Integer})
function frule!!(
    ::Lifted{typeof(unsafe_getindex),N},
    r::Lifted{P,N,<:ImmutableDual},
    i::Lifted{<:Integer},
) where {N,P<:TWPStepRangeLen}
    _r = primal(r)
    _i = primal(i)
    x = unsafe_getindex(_r, _i)
    Eout = eltype(P)
    ref_v = tangent(r).value.ref  # NTuple{N, TWP{T}}
    step_v = tangent(r).value.step  # NTuple{N, TWP{T}}
    offset = _r.offset
    dy_lanes = ntuple(k -> Eout(ref_v[k] + step_v[k] * (_i - offset)), Val(N))
    return Lifted{Eout,N}(x, NDual{Eout,N}(x, dy_lanes))
end
function rrule!!(
    ::CoDual{typeof(unsafe_getindex)}, r::CoDual{P}, i::CoDual{<:Integer}
) where {P<:TWPStepRangeLen}
    offset = r.x.offset
    function unsafe_getindex_pb(dy)
        T = rdata_type(tangent_type(P))
        dy_twice_precision = TwicePrecision(dy)
        dref = dy_twice_precision
        dstep = dy_twice_precision * (i.x - offset)
        dr = T((ref=dref, step=dstep, len=NoRData(), offset=NoRData()))
        return NoRData(), dr, NoRData()
    end
    return zero_fcodual(unsafe_getindex(r.x, i.x)), unsafe_getindex_pb
end

using Base: _getindex_hiprec
@is_primitive(MinimalCtx, Tuple{typeof(_getindex_hiprec),TWPStepRangeLen,Integer})
function frule!!(
    ::Lifted{typeof(_getindex_hiprec),N},
    r::Lifted{P,N,<:ImmutableDual},
    i::Lifted{<:Integer},
) where {N,P<:TWPStepRangeLen}
    _r = primal(r)
    _i = primal(i)
    x = _getindex_hiprec(_r, _i)
    Pout = typeof(x)
    ref_v = tangent(r).value.ref
    step_v = tangent(r).value.step
    offset = _r.offset
    dy_lanes = ntuple(k -> (_i - offset) * step_v[k] + ref_v[k], Val(N))
    return Lifted{Pout,N}(x, dy_lanes)
end
function rrule!!(
    ::CoDual{typeof(_getindex_hiprec)}, r::CoDual{P}, i::CoDual{<:Integer}
) where {P<:TWPStepRangeLen}
    offset = r.x.offset
    function unsafe_getindex_pb(dy)
        T = rdata_type(tangent_type(P))
        dref = dy
        dstep = dy * (i.x - offset)
        dr = T((ref=dref, step=dstep, len=NoRData(), offset=NoRData()))
        return NoRData(), dr, NoRData()
    end
    return zero_fcodual(_getindex_hiprec(r.x, i.x)), unsafe_getindex_pb
end

@is_primitive MinimalCtx Tuple{typeof(:),P,P,P} where {P<:IEEEFloat}
function frule!!(
    ::Lifted{typeof(:),N},
    start::Lifted{P,N,NDual{P,N}},
    step::Lifted{P,N,NDual{P,N}},
    stop::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    y = (:)(primal(start), primal(step), primal(stop))
    start_parts = tangent(start).partials
    step_parts = tangent(step).partials
    ref_v = ntuple(k -> TWP{P}(start_parts[k], zero(P)), Val(N))
    step_v = ntuple(k -> TWP{P}(step_parts[k], zero(P)), Val(N))
    nt = (ref=ref_v, step=step_v, len=NoDual(), offset=NoDual())
    return Lifted{typeof(y),N}(y, ImmutableDual(nt))
end
function rrule!!(
    ::CoDual{typeof(:)}, start::CoDual{P}, step::CoDual{P}, stop::CoDual{P}
) where {P<:IEEEFloat}
    colon_pb(dy::RData) = NoRData(), P(dy.data.ref), P(dy.data.step), zero(P)
    return zero_fcodual((:)(start.x, step.x, stop.x)), colon_pb
end

@is_primitive MinimalCtx Tuple{typeof(sum),TWPStepRangeLen}
function frule!!(
    ::Lifted{typeof(sum),N}, x::Lifted{P,N,<:ImmutableDual}
) where {N,P<:TWPStepRangeLen}
    _x = primal(x)
    y = sum(_x)
    l = _x.len
    offset = _x.offset
    ref_v = tangent(x).value.ref
    step_v = tangent(x).value.step
    Yout = typeof(y)
    dy_lanes = ntuple(
        k -> Yout(ref_v[k] * l + step_v[k] * (0.5 * l * (l + 1) - l * offset)), Val(N)
    )
    return Lifted{Yout,N}(y, NDual{Yout,N}(y, dy_lanes))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{P}) where {P<:TWPStepRangeLen}
    l = x.x.len
    offset = x.x.offset
    function sum_pb(dy::Float64)
        R = rdata_type(tangent_type(P))
        dref = TwicePrecision(l * dy)
        dstep = TwicePrecision(dy * (0.5 * l * (l + 1) - l * offset))
        dx = R((ref=dref, step=dstep, len=NoRData(), offset=NoRData()))
        return NoRData(), dx
    end
    return zero_fcodual(sum(x.x)), sum_pb
end

@is_primitive(
    MinimalCtx,
    Tuple{typeof(Base.range_start_stop_length),P,P,Integer} where {P<:IEEEFloat},
)
function frule!!(
    ::Lifted{typeof(Base.range_start_stop_length),N},
    start::Lifted{P,N,NDual{P,N}},
    stop::Lifted{P,N,NDual{P,N}},
    length::Lifted{<:Integer},
) where {N,P<:IEEEFloat}
    _len = primal(length)
    l = _len - 1
    y = Base.range_start_stop_length(primal(start), primal(stop), _len)
    start_parts = tangent(start).partials
    stop_parts = tangent(stop).partials
    ref_v = ntuple(k -> TWP{P}(start_parts[k], zero(P)), Val(N))
    step_v = ntuple(k -> TWP{P}((stop_parts[k] - start_parts[k]) / l, zero(P)), Val(N))
    nt = (ref=ref_v, step=step_v, len=NoDual(), offset=NoDual())
    return Lifted{typeof(y),N}(y, ImmutableDual(nt))
end
function rrule!!(
    ::CoDual{typeof(Base.range_start_stop_length)},
    start::CoDual{P},
    stop::CoDual{P},
    length::CoDual{<:Integer},
) where {P<:IEEEFloat}
    l = (length.x - 1)
    function range_start_stop_length_pb(dy::RData)
        dstart = P(dy.data.ref) - P(dy.data.step) / l
        dstop = P(dy.data.step) / l
        return NoRData(), dstart, dstop, NoRData()
    end
    y = zero_fcodual(Base.range_start_stop_length(start.x, stop.x, length.x))
    return y, range_start_stop_length_pb
end

@static if VERSION >= v"1.11"
    @is_primitive MinimalCtx Tuple{
        typeof(Base._exp_allowing_twice64),TwicePrecision{Float64}
    }
    function frule!!(
        ::Lifted{typeof(Base._exp_allowing_twice64),N},
        x::Lifted{TwicePrecision{Float64},N,NTuple{N,TwicePrecision{Float64}}},
    ) where {N}
        _x = primal(x)
        # `_exp_allowing_twice64` returns a `Float64` (not a `TwicePrecision`), so the output
        # slot is a scalar `NDual`; the per-lane JVP `y * dx` (a `TwicePrecision`) projects to
        # the `Float64` output tangent. Mirrors the reverse rule's `Float64` output cotangent.
        y = Base._exp_allowing_twice64(_x)
        x_parts = tangent(x)
        dy = ntuple(k -> Float64(y * x_parts[k]), Val(N))
        return Lifted{Float64,N}(y, NDual{Float64,N}(y, dy))
    end
    function rrule!!(
        ::CoDual{typeof(Base._exp_allowing_twice64)}, x::CoDual{TwicePrecision{Float64}}
    )
        y = Base._exp_allowing_twice64(x.x)
        _exp_allowing_twice64_pb(dy::Float64) = NoRData(), TwicePrecision(dy * y)
        return zero_fcodual(y), _exp_allowing_twice64_pb
    end

    @is_primitive(MinimalCtx, Tuple{typeof(Base._log_twice64_unchecked),Float64})
    function frule!!(
        ::Lifted{typeof(Base._log_twice64_unchecked),N},
        x::Lifted{Float64,N,NDual{Float64,N}},
    ) where {N}
        _x = primal(x)
        y = Base._log_twice64_unchecked(_x)
        x_parts = tangent(x).partials
        dy = ntuple(k -> TwicePrecision{Float64}(x_parts[k] / _x), Val(N))
        return Lifted{TwicePrecision{Float64},N}(y, dy)
    end
    function rrule!!(::CoDual{typeof(Base._log_twice64_unchecked)}, x::CoDual{Float64})
        _x = x.x
        _log_twice64_pb(dy::TwicePrecision{Float64}) = NoRData(), Float64(dy) / _x
        return zero_fcodual(Base._log_twice64_unchecked(_x)), _log_twice64_pb
    end
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:twice_precision})
    test_cases = Any[
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            TwicePrecisionFloat{Float64},
            5.0,
            4.0,
        ),
        (false, :stability_and_allocs, nothing, twiceprecision, 5.0, 4),
        (false, :stability_and_allocs, nothing, twiceprecision, TwicePrecision(5.0), 4),
        (false, :stability_and_allocs, nothing, Float64, TwicePrecision(5.0, 3.0)),
        (false, :stability_and_allocs, nothing, -, TwicePrecision(5.0, 3.0)),
        (false, :stability_and_allocs, nothing, +, TwicePrecision(5.0, 3.0), 4.0),
        (
            false,
            :stability_and_allocs,
            nothing,
            +,
            TwicePrecision(5.0, 3.0),
            TwicePrecision(4.0, 5.0),
        ),
        (false, :stability_and_allocs, nothing, +, TwicePrecision(5.0, 3.0), 4),
        (false, :stability_and_allocs, nothing, *, TwicePrecision(5.0, 1e-12), 3.0),
        (false, :stability_and_allocs, nothing, *, TwicePrecision(5.0, 1e-12), 3),
        (false, :stability_and_allocs, nothing, /, TwicePrecision(5.0, 1e-12), 3.0),
        (false, :stability_and_allocs, nothing, /, TwicePrecision(5.0, 1e-12), 3),
        (false, :stability_and_allocs, nothing, Base.splitprec, Float64, 5),
        (false, :stability_and_allocs, nothing, Base.splitprec, Float32, 5),
        (false, :stability_and_allocs, nothing, Base.splitprec, Float16, 5),
        (false, :stability_and_allocs, nothing, Base.floatrange, Float64, 5, 6, 7, 8),
        (false, :stability_and_allocs, nothing, Base._linspace, Float64, 5, 6, 7, 8),
        (false, :allocs, nothing, Base.range_start_step_length, 5.0, 6.0, 10),
        (false, :allocs, nothing, Base.range_start_step_length, 5.0, Float64(π), 10),
        (
            false,
            :stability_and_allocs,
            nothing,
            unsafe_getindex,
            StepRangeLen(TwicePrecision(-0.45), TwicePrecision(0.98), 10, 3),
            5,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _getindex_hiprec,
            StepRangeLen(TwicePrecision(-0.45), TwicePrecision(0.98), 10, 3),
            5,
        ),
        (false, :allocs, nothing, (:), -0.1, 0.99, 5.1),
        (false, :stability_and_allocs, nothing, sum, range(-0.1, 9.9; length=51)),
        (false, :allocs, nothing, Base.range_start_stop_length, -0.5, 11.7, 7),
        (false, :allocs, nothing, Base.range_start_stop_length, -0.5, -11.7, 11),
    ]
    @static if VERSION >= v"1.11"
        extra_test_cases = Any[
            (
                false,
                :stability_and_allocs,
                nothing,
                Base._exp_allowing_twice64,
                TwicePrecision(2.0),
            ),
            (false, :stability_and_allocs, nothing, Base._log_twice64_unchecked, 3.0),
        ]
        test_cases = vcat(test_cases, extra_test_cases)
    end
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:twice_precision})
    test_cases = Any[

        # Functionality in base/twiceprecision.jl
        (false, :allocs, nothing, TwicePrecision{Float64}, 5.0, 0.3),
        (
            false,
            :allocs,
            nothing,
            (x, y) -> Float64(TwicePrecision{Float64}(x, y)),
            5.0,
            0.3,
        ),
        (false, :allocs, nothing, TwicePrecision, 5.0, 0.3),
        (false, :allocs, nothing, (x, y) -> Float64(TwicePrecision(x, y)), 5.0, 0.3),
        (false, :allocs, nothing, TwicePrecision{Float64}, 5.0),
        (false, :allocs, nothing, x -> Float64(TwicePrecision{Float64}(x)), 5.0),
        (false, :allocs, nothing, TwicePrecision, 5.0),
        (false, :allocs, nothing, x -> Float64(TwicePrecision(x)), 5.0),
        (false, :allocs, nothing, TwicePrecision{Float64}, 5),
        (false, :allocs, nothing, x -> Float64(TwicePrecision{Float64}(x)), 5),
        (false, :none, nothing, TwicePrecision{Float64}, (5, 4)),
        (false, :none, nothing, x -> Float64(TwicePrecision{Float64}(x)), (5, 4)),
        (false, :none, nothing, TwicePrecision{Float64}, (5, 4), 3),
        (
            false,
            :none,
            nothing,
            (x, y) -> Float64(TwicePrecision{Float64}(x, y)),
            (5, 4),
            3,
        ),
        (false, :allocs, nothing, +, TwicePrecision(5.0), TwicePrecision(4.0)),
        (false, :allocs, nothing, +, 5.0, TwicePrecision(4.0)),
        (false, :allocs, nothing, +, TwicePrecision(5.0), 4.0),
        (false, :allocs, nothing, -, TwicePrecision(5.0), TwicePrecision(4.0)),
        (false, :allocs, nothing, -, 5.0, TwicePrecision(4.0)),
        (false, :allocs, nothing, -, TwicePrecision(5.0), 4.0),
        (false, :allocs, nothing, *, 3.0, TwicePrecision(5.0, 1e-12)),
        (false, :allocs, nothing, *, 3, TwicePrecision(5.0, 1e-12)),
        (
            false,
            :allocs,
            nothing,
            getindex,
            StepRangeLen(TwicePrecision(-0.45), TwicePrecision(0.98), 10, 3),
            2:2:6,
        ),
        (
            false,
            :allocs,
            nothing,
            +,
            range(0.0, 5.0; length=44),
            range(-33.0, 4.5; length=44),
        ),

        # Functionality in base/range.jl
        (false, :allocs, nothing, range, 0.0, 5.6),
        (false, :allocs, nothing, (lb, ub) -> range(lb, ub; length=10), -0.45, 9.5),
    ]
    @static if VERSION >= v"1.11"
        push!(test_cases, (false, :allocs, nothing, Base._logrange_extra, 1.1, 3.5, 5))
        push!(test_cases, (false, :allocs, nothing, logrange, 5.0, 10.0, 11))
    end
    return test_cases, Any[]
end
