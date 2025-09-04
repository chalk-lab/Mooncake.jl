struct CoDual{Tx,Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function CoDual(x::Type{P}, dx::NoFData) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoFData}(P, dx)
end

function CoDual(x::Type{P}, dx::NoTangent) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(P, dx)
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))
_copy(x::P) where {P<:CoDual} = x

"""
    extract(x::CoDual)

Helper function. Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::CoDual) = primal(x), tangent(x)

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))

"""
    uninit_codual(x)

Equivalent to `CoDual(x, uninit_tangent(x))`.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

function _codual_internal(::Type{P}, f::F, extractor::E) where {P,F,E}
    P == Union{} && return Union{}
    P == DataType && return CoDual
    P isa Union && return Union{f(P.a),f(P.b)}

    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{f(P_split.a),f(P_split.b)}
        end
    end

    P <: UnionAll && return CoDual # P is abstract, so we don't know its tangent type.
    return isconcretetype(P) ? CoDual{P,extractor(P)} : CoDual
end

"""
    codual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and associated tangents.
"""
codual_type(::Type{P}) where {P} = _codual_internal(P, codual_type, tangent_type)

@unstable function codual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoTangent} : CoDual{_typeof(p),NoTangent}
end

"""
    fcodual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and its fdata.
"""
function fcodual_type(::Type{P}) where {P}
    return _codual_internal(P, fcodual_type, P -> fdata_type(tangent_type(P)))
end

@unstable function fcodual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoFData} : CoDual{_typeof(p),NoFData}
end

to_fwds(x::CoDual) = CoDual(primal(x), fdata(tangent(x)))

to_fwds(x::CoDual{Type{P}}) where {P} = CoDual{Type{P},NoFData}(primal(x), NoFData())

zero_fcodual(p) = to_fwds(zero_codual(p))

"""
    uninit_fcodual(x)

Like `zero_fcodual`, but doesn't guarantee that the value of the fdata is initialised.
See implementation for details, as this function is subject to change.
"""
@inline uninit_fcodual(x::P) where {P} = CoDual(x, uninit_fdata(x))

struct NoPullback{R<:Tuple}
    r::R
end

_copy(x::P) where {P<:NoPullback} = P(_copy(x.r))

"""
    NoPullback(args::CoDual...)

Construct a `NoPullback` from the arguments passed to an `rrule!!`. For each argument,
extracts the primal value, and constructs a `LazyZeroRData`. These are stored in a
`NoPullback` which, in the reverse-pass of AD, instantiates these `LazyZeroRData`s and
returns them in order to perform the reverse-pass of AD.

The advantage of this approach is that if it is possible to construct the zero rdata element
for each of the arguments lazily, the `NoPullback` generated will be a singleton type. This
means that AD can avoid generating a stack to store this pullback, which can result in
significant performance improvements.
"""
function NoPullback(args::Vararg{CoDual,N}) where {N}
    return NoPullback(tuple_map(lazy_zero_rdata ∘ primal, args))
end

@inline (pb::NoPullback)(_) = tuple_map(instantiate, pb.r)
