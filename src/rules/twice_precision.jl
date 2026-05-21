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

fdata_type(::Type{<:TWP}) = NoFData

rdata_type(P::Type{<:TWP}) = P

__verify_fdata_value(::IdDict{Any,Nothing}, ::P, ::P) where {P<:TWP} = nothing

_verify_rdata_value(::P, ::P) where {P<:TWP} = nothing

@foldable tangent_type(::Type{NoFData}, T::Type{<:TWP}) = T

tangent(::NoFData, t::TWP) = t

zero_rdata(p::TWP) = zero_tangent(p)

zero_rdata_from_type(P::Type{<:TWP{F}}) where {F} = P(zero(F), zero(F))

#
# Rules. These are required for a lot of functionality in this case.
#

@is_primitive MinimalCtx Tuple{typeof(_new_),<:TWP,IEEEFloat,IEEEFloat}
# Implementation kernels: dispatch on Dual/NDual inner V to extract value/tangent.
# `tangent(d::Dual{P, NTangent{Tuple{T}}})` returns the NTangent wrapper.
# These kernels and downstream rule bodies need the bare lane value so
# arithmetic / conversion / `twiceprecision(t, nb)` dispatches correctly.
# Unwrap singleton NTangent at every `tangent(d)` boundary in this file.
@inline _twp_tangent(d::Dual) = Mooncake._ntangent_unwrap_singleton(tangent(d))
@inline _twp_val(d::Dual{P}) where {P<:IEEEFloat} = primal(d), _twp_tangent(d)
@inline _twp_val(d::Mooncake.Nfwd.NDual{P,1}) where {P<:IEEEFloat} = d.value, d.partials[1]
# Width-N NDual: return per-lane partials tuple. Callers above (the
# `twiceprecision` and `_new_(TWP)` rule bodies) detect the tuple shape
# and apply the op per-lane to build a canonical
# `Dual{TWP{P}, NTangent{NTuple{N, TWP{P}}}}` result.
@inline _twp_val(d::Mooncake.Nfwd.NDual{P,N}) where {P<:IEEEFloat,N} = d.value, d.partials
@inline function frule!!(
    ::Mooncake.Lifted{typeof(_new_),N},
    ::Mooncake.Lifted{Type{TWP{P}}},
    hi::Mooncake.Lifted{P},
    lo::Mooncake.Lifted{P},
) where {N,P<:IEEEFloat}
    hv, ht = _twp_val(Mooncake._unlift(hi))
    lv, lt = _twp_val(Mooncake._unlift(lo))
    # Width-N: `ht` / `lt` are `NTuple{N, P}` (per-lane partials), so
    # `_new_(TWP{P}, ht, lt)` would error. Build per-lane TWP tangents
    # and wrap in NTangent (matches canonical V `Dual{TWP{P},
    # NTangent{NTuple{N, TWP{P}}}}`).
    if ht isa NTuple{N,P} && lt isa NTuple{N,P} && N >= 2
        primal_twp = _new_(TWP{P}, hv, lv)
        tangent_twps = ntuple(lane -> _new_(TWP{P}, ht[lane], lt[lane]), Val(N))
        return Mooncake.Lifted{TWP{P},N}(primal_twp, Mooncake.NTangent(tangent_twps))
    end
    bare_result = Dual(_new_(TWP{P}, hv, lv), _new_(TWP{P}, ht, lt))
    return _wrap_rule_result(Val(N), bare_result)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{TWP{P}}}, hi::CoDual{P}, lo::CoDual{P}
) where {P<:IEEEFloat}
    _new_twice_precision_pb(dy::TWP{P}) = NoRData(), NoRData(), P(dy), P(dy)
    return zero_fcodual(_new_(TWP{P}, hi.x, lo.x)), _new_twice_precision_pb
end

@is_primitive MinimalCtx Tuple{typeof(twiceprecision),IEEEFloat,Integer}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(twiceprecision),N},
    val::Mooncake.Lifted,
    nb::Mooncake.Lifted{<:Integer},
) where {N}
    vv, vt = _twp_val(Mooncake._unlift(val))
    nb_p = primal(Mooncake._unlift(nb))
    # Width-N: `vt` is `NTuple{N, P}` (per-lane partials). Build per-lane
    # `TWP{P}` tangents and wrap in NTangent.
    if vt isa Tuple && N >= 2
        primal_twp = twiceprecision(vv, nb_p)
        tangent_twps = ntuple(lane -> twiceprecision(vt[lane], nb_p), Val(N))
        return Mooncake.Lifted{typeof(primal_twp),N}(
            primal_twp, Mooncake.NTangent(tangent_twps)
        )
    end
    bare_result = Dual(twiceprecision(vv, nb_p), twiceprecision(vt, nb_p))
    return _wrap_rule_result(Val(N), bare_result)
end
function rrule!!(
    ::CoDual{typeof(twiceprecision)}, val::CoDual{P}, nb::CoDual{<:Integer}
) where {P<:IEEEFloat}
    twiceprecision_float_pb(dy::TWP{P}) = NoRData(), P(dy), NoRData()
    return zero_fcodual(twiceprecision(val.x, nb.x)), twiceprecision_float_pb
end

@is_primitive MinimalCtx Tuple{typeof(twiceprecision),TWP,Integer}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(twiceprecision),N},
    val::Mooncake.Lifted{P},
    nb::Mooncake.Lifted{<:Integer},
) where {N,P<:TWP}
    inner_val = Mooncake._unlift(val)
    nb_p = primal(Mooncake._unlift(nb))
    bare_result = Dual(
        twiceprecision(primal(inner_val), nb_p),
        twiceprecision(_twp_tangent(inner_val), nb_p),
    )
    return _wrap_rule_result(Val(N), bare_result)
end
function rrule!!(
    ::CoDual{typeof(twiceprecision)}, val::CoDual{P}, nb::CoDual{<:Integer}
) where {P<:TWP}
    twiceprecision_pb(dy::P) = NoRData(), dy, NoRData()
    return zero_fcodual(twiceprecision(val.x, nb.x)), twiceprecision_pb
end

@is_primitive MinimalCtx Tuple{Type{<:IEEEFloat},TWP}
@inline function frule!!(
    ::Mooncake.Lifted{Type{P},N}, x::Mooncake.Lifted{S}
) where {N,P<:IEEEFloat,S<:TWP}
    inner_x = Mooncake._unlift(x)
    bare_result = Dual(P(primal(inner_x)), P(_twp_tangent(inner_x)))
    return _wrap_rule_result(Val(N), bare_result)
end
function rrule!!(::CoDual{Type{P}}, x::CoDual{S}) where {P<:IEEEFloat,S<:TWP}
    float_from_twice_precision_pb(dy::P) = NoRData(), S(dy)
    return zero_fcodual(P(x.x)), float_from_twice_precision_pb
end

@is_primitive MinimalCtx Tuple{typeof(-),TWP}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(-),N}, x::Mooncake.Lifted{P}
) where {N,P<:TWP}
    # Per-lane assembly.
    p_out = -primal(Mooncake._unlift(x))
    tangents = ntuple(n -> -Mooncake.tangent(x, n), Val(N))
    return Mooncake.Lifted{_typeof(p_out),N}(p_out, Mooncake.NTangent(tangents))
end
function rrule!!(::CoDual{typeof(-)}, x::CoDual{P}) where {P<:TWP}
    negate_twice_precision_pb(dy::P) = NoRData(), -dy
    return zero_fcodual(-(x.x)), negate_twice_precision_pb
end

@is_primitive MinimalCtx Tuple{typeof(+),TWP,IEEEFloat}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(+),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted
) where {N,P<:TWP}
    # Per-lane assembly so each lane's tangent is computed independently
    # (was: single-lane via `_twp_val` then broadcast at wrap, silently
    # duplicating across N lanes for N >= 2).
    px = primal(Mooncake._unlift(x))
    py = primal(Mooncake._unlift(y))
    p_out = px + py
    tangents = ntuple(Val(N)) do n
        Mooncake.tangent(x, n) + Mooncake.tangent(y, n)
    end
    return Mooncake.Lifted{_typeof(p_out),N}(p_out, Mooncake.NTangent(tangents))
end
function rrule!!(
    ::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{S}
) where {P<:TWP,S<:IEEEFloat}
    plus_pullback(dz::P) = NoRData(), dz, S(dz)
    return zero_fcodual(x.x + y.x), plus_pullback
end

@is_primitive(MinimalCtx, Tuple{typeof(+),P,P} where {P<:TWP})
@inline function frule!!(
    ::Mooncake.Lifted{typeof(+),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted{P}
) where {N,P<:TWP}
    p_out = primal(Mooncake._unlift(x)) + primal(Mooncake._unlift(y))
    tangents = ntuple(n -> Mooncake.tangent(x, n) + Mooncake.tangent(y, n), Val(N))
    return Mooncake.Lifted{_typeof(p_out),N}(p_out, Mooncake.NTangent(tangents))
end
function rrule!!(::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{P}) where {P<:TWP}
    plus_pullback(dz::P) = NoRData(), dz, dz
    return zero_fcodual(x.x + y.x), plus_pullback
end

@is_primitive MinimalCtx Tuple{typeof(+),TWP,Integer}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(+),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted{<:Integer}
) where {N,P<:TWP}
    p_out = primal(Mooncake._unlift(x)) + primal(Mooncake._unlift(y))
    tangents = ntuple(n -> Mooncake.tangent(x, n), Val(N))
    return Mooncake.Lifted{_typeof(p_out),N}(p_out, Mooncake.NTangent(tangents))
end
function rrule!!(::CoDual{typeof(+)}, x::CoDual{P}, y::CoDual{<:Integer}) where {P<:TWP}
    plus_twice_precision_integer_pb(dz::P) = NoRData(), dz, NoRData()
    return zero_fcodual(x.x + primal(y)), plus_twice_precision_integer_pb
end

@is_primitive MinimalCtx Tuple{typeof(*),TWP,IEEEFloat}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(*),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted
) where {N,P<:TWP}
    # Per-lane assembly. d(xy)/dx = y, d(xy)/dy = x.
    px = primal(Mooncake._unlift(x))
    py = primal(Mooncake._unlift(y))
    z = px * py
    tangents = ntuple(Val(N)) do n
        px * Mooncake.tangent(y, n) + Mooncake.tangent(x, n) * py
    end
    return Mooncake.Lifted{_typeof(z),N}(z, Mooncake.NTangent(tangents))
end
function rrule!!(
    ::CoDual{typeof(*)}, x::CoDual{P}, y::CoDual{S}
) where {P<:TWP,S<:IEEEFloat}
    _x, _y = x.x, y.x
    mul_twice_precision_and_float_pb(dz::P) = NoRData(), dz * _y, S(dz * _x)
    return zero_fcodual(_x * _y), mul_twice_precision_and_float_pb
end

@is_primitive MinimalCtx Tuple{typeof(*),TWP,Integer}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(*),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted{<:Integer}
) where {N,P<:TWP}
    yp = primal(Mooncake._unlift(y))
    z = primal(Mooncake._unlift(x)) * yp
    tangents = ntuple(n -> Mooncake.tangent(x, n) * yp, Val(N))
    return Mooncake.Lifted{_typeof(z),N}(z, Mooncake.NTangent(tangents))
end
function rrule!!(::CoDual{typeof(*)}, x::CoDual{P}, y::CoDual{<:Integer}) where {P<:TWP}
    _y = y.x
    mul_twice_precision_and_int_pb(dz::P) = NoRData(), dz * _y, NoRData()
    return zero_fcodual(x.x * _y), mul_twice_precision_and_int_pb
end

@is_primitive MinimalCtx Tuple{typeof(/),TWP,IEEEFloat}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(/),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted
) where {N,P<:TWP}
    # Per-lane assembly. d(x/y)/dx = 1/y, d(x/y)/dy = -x/y^2.
    px = primal(Mooncake._unlift(x))
    py = primal(Mooncake._unlift(y))
    z = px / py
    tangents = ntuple(Val(N)) do n
        Mooncake.tangent(x, n) / py - Mooncake.tangent(y, n) * px / py^2
    end
    return Mooncake.Lifted{_typeof(z),N}(z, Mooncake.NTangent(tangents))
end
function rrule!!(
    ::CoDual{typeof(/)}, x::CoDual{P}, y::CoDual{S}
) where {P<:TWP,S<:IEEEFloat}
    _x, _y = x.x, y.x
    div_twice_precision_and_float_pb(dz::P) = NoRData(), dz / _y, S(-dz * _x / _y^2)
    return zero_fcodual(_x / _y), div_twice_precision_and_float_pb
end

@is_primitive MinimalCtx Tuple{typeof(/),TWP,Integer}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(/),N}, x::Mooncake.Lifted{P}, y::Mooncake.Lifted{<:Integer}
) where {N,P<:TWP}
    yp = primal(Mooncake._unlift(y))
    z = primal(Mooncake._unlift(x)) / yp
    tangents = ntuple(n -> Mooncake.tangent(x, n) / yp, Val(N))
    return Mooncake.Lifted{_typeof(z),N}(z, Mooncake.NTangent(tangents))
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(range_start_step_length),N},
    a::Mooncake.Lifted,
    st::Mooncake.Lifted,
    len::Mooncake.Lifted{<:Integer},
) where {N}
    av, at = _twp_val(Mooncake._unlift(a))
    sv, stt = _twp_val(Mooncake._unlift(st))
    lp = primal(Mooncake._unlift(len))
    x = range_start_step_length(av, sv, lp)
    Tx = tangent_type(typeof(x))
    dx = Tx((ref=at, step=stt, len=NoTangent(), offset=NoTangent()))
    bare_result = Dual(x, dx)
    return _wrap_rule_result(Val(N), bare_result)
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(unsafe_getindex),N},
    r::Mooncake.Lifted{P},
    i::Mooncake.Lifted{<:Integer},
) where {N,P<:TWPStepRangeLen}
    inner_r = Mooncake._unlift(r)
    inner_i = Mooncake._unlift(i)
    x = unsafe_getindex(primal(inner_r), primal(inner_i))
    raw_t = tangent(inner_r)
    # Multi-lane NTangent (width N≥2): compute per-lane derivative. The
    # singleton path's NTangent arithmetic (`dstep * (i - offset)`) errors
    # because `*(::NTangent, ::Int)` is undefined.
    if raw_t isa Mooncake.NTangent && length(raw_t.lanes) >= 2
        i_p = primal(inner_i)
        offset = primal(inner_r).offset
        dxs = ntuple(Val(length(raw_t.lanes))) do lane
            lane_t = raw_t.lanes[lane]
            dref_lane = _get_tangent_field(lane_t, :ref)
            dstep_lane = _get_tangent_field(lane_t, :step)
            eltype(P)(dref_lane + dstep_lane * (i_p - offset))
        end
        return Mooncake.Lifted{eltype(P),N}(x, Mooncake.NTangent(dxs))
    end
    dref = _get_tangent_field(_twp_tangent(inner_r), :ref)
    dstep = _get_tangent_field(_twp_tangent(inner_r), :step)
    dx = eltype(P)(dref + dstep * (primal(inner_i) - primal(inner_r).offset))
    bare_result = Dual(x, dx)
    return _wrap_rule_result(Val(N), bare_result)
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(_getindex_hiprec),N},
    r::Mooncake.Lifted{<:TWPStepRangeLen},
    i::Mooncake.Lifted{<:Integer},
) where {N}
    inner_r = Mooncake._unlift(r)
    inner_i = Mooncake._unlift(i)
    x = _getindex_hiprec(primal(inner_r), primal(inner_i))
    offset = primal(inner_r).offset
    raw_t = tangent(inner_r)
    # Multi-lane NTangent: per-lane derivative.
    if raw_t isa Mooncake.NTangent && length(raw_t.lanes) >= 2
        i_p = primal(inner_i)
        dxs = ntuple(Val(length(raw_t.lanes))) do lane
            lane_t = raw_t.lanes[lane]
            dref_lane = _get_tangent_field(lane_t, :ref)
            dstep_lane = _get_tangent_field(lane_t, :step)
            (i_p - offset) * dstep_lane + dref_lane
        end
        return Mooncake.Lifted{_typeof(x),N}(x, Mooncake.NTangent(dxs))
    end
    dstep = _get_tangent_field(_twp_tangent(inner_r), :step)
    dref = _get_tangent_field(_twp_tangent(inner_r), :ref)
    dx = (primal(inner_i) - offset) * dstep + dref
    bare_result = Dual(x, dx)
    return _wrap_rule_result(Val(N), bare_result)
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(:),N},
    start::Mooncake.Lifted{P},
    step::Mooncake.Lifted{P},
    stop::Mooncake.Lifted{P},
) where {N,P<:IEEEFloat}
    sav, sat = _twp_val(Mooncake._unlift(start))
    sv, st = _twp_val(Mooncake._unlift(step))
    sopv, _ = _twp_val(Mooncake._unlift(stop))
    x = (:)(sav, sv, sopv)
    T = tangent_type(typeof(x))
    dx = T((ref=sat, step=st, len=NoTangent(), offset=NoTangent()))
    bare_result = Dual(x, dx)
    return _wrap_rule_result(Val(N), bare_result)
end
function rrule!!(
    ::CoDual{typeof(:)}, start::CoDual{P}, step::CoDual{P}, stop::CoDual{P}
) where {P<:IEEEFloat}
    colon_pb(dy::RData) = NoRData(), P(dy.data.ref), P(dy.data.step), zero(P)
    return zero_fcodual((:)(start.x, step.x, stop.x)), colon_pb
end

@is_primitive MinimalCtx Tuple{typeof(sum),TWPStepRangeLen}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(sum),N}, x::Mooncake.Lifted{<:TWPStepRangeLen}
) where {N}
    inner_x = Mooncake._unlift(x)
    y = sum(primal(inner_x))
    l = primal(inner_x).len
    offset = primal(inner_x).offset
    dref = _get_tangent_field(_twp_tangent(inner_x), :ref)
    dstep = _get_tangent_field(_twp_tangent(inner_x), :step)
    dy = dref * l + dstep * (0.5 * l * (l + 1) - l * offset)
    bare_result = Dual(y, typeof(y)(dy))
    return _wrap_rule_result(Val(N), bare_result)
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.range_start_stop_length),N},
    start::Mooncake.Lifted,
    stop::Mooncake.Lifted,
    length::Mooncake.Lifted{<:Integer},
) where {N}
    sav, sat = _twp_val(Mooncake._unlift(start))
    spv, spt = _twp_val(Mooncake._unlift(stop))
    lp = primal(Mooncake._unlift(length))
    l = lp - 1
    y = Base.range_start_stop_length(sav, spv, lp)
    T = tangent_type(typeof(y))
    dy = T((ref=sat, step=(spt - sat) / l, len=NoTangent(), offset=NoTangent()))
    bare_result = Dual(y, dy)
    return _wrap_rule_result(Val(N), bare_result)
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
    @inline function frule!!(
        ::Mooncake.Lifted{typeof(Base._exp_allowing_twice64),N},
        x::Mooncake.Lifted{TwicePrecision{Float64}},
    ) where {N}
        inner_x = Mooncake._unlift(x)
        y = Base._exp_allowing_twice64(primal(inner_x))
        bare_result = Dual(y, typeof(y)(y * _twp_tangent(inner_x)))
        return _wrap_rule_result(Val(N), bare_result)
    end
    function rrule!!(
        ::CoDual{typeof(Base._exp_allowing_twice64)}, x::CoDual{TwicePrecision{Float64}}
    )
        y = Base._exp_allowing_twice64(x.x)
        _exp_allowing_twice64_pb(dy::Float64) = NoRData(), TwicePrecision(dy * y)
        return zero_fcodual(y), _exp_allowing_twice64_pb
    end

    @is_primitive(MinimalCtx, Tuple{typeof(Base._log_twice64_unchecked),Float64})
    @inline function frule!!(
        ::Mooncake.Lifted{typeof(Base._log_twice64_unchecked),N},
        x::Mooncake.Lifted{Float64},
    ) where {N}
        xv, xt = _twp_val(Mooncake._unlift(x))
        y = Base._log_twice64_unchecked(xv)
        bare_result = Dual(y, typeof(y)(xt / xv))
        return _wrap_rule_result(Val(N), bare_result)
    end
    function rrule!!(::CoDual{typeof(Base._log_twice64_unchecked)}, x::CoDual{Float64})
        _x = x.x
        _log_twice64_pb(dy::TwicePrecision{Float64}) = NoRData(), Float64(dy) / _x
        return zero_fcodual(Base._log_twice64_unchecked(_x)), _log_twice64_pb
    end
end

# Lifted-aware trait registrations: each rule's body works against bare
# `Dual{TWP, TWP}` slot values, and the generic Lifted-aware adapter
# (`primal_mode.jl`) handles the wrap/unwrap. Registering the trait skips
# the IR-emit scaffold and lets the adapter dispatch directly.
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(twiceprecision),Vararg}}) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{Type{<:IEEEFloat},<:TwicePrecision}}) =
    true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(-),<:TwicePrecision}}) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(+),<:TwicePrecision,Any}}) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(*),<:TwicePrecision,Any}}) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(/),<:TwicePrecision,Any}}) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base.range_start_step_length),Any,Any,Any}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(unsafe_getindex),<:TWPStepRangeLen,Any}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(_getindex_hiprec),<:TWPStepRangeLen,Any}}
) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(:),Any,Any,Any}}) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sum),<:TWPStepRangeLen}}) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base.range_start_stop_length),Any,Any,Any}}
) = true
@static if VERSION < v"1.11"
    @inline Mooncake._is_lifted_aware(
        ::Type{<:Tuple{typeof(Base._log_twice64_unchecked),Any}}
    ) = true
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
