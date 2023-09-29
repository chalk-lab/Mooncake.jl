#
# Performance-only rules. These should be able to be removed, and everything still works,
# just a bit slower. The effect of these is typically to remove many nodes from the tape.
# Ideally, it would be the case that acitivty analysis eliminates any run-time improvements
# that these rules provide. Possibly they would still be useful in order to avoid having to
# deduce that these bits of code are inactive though.
#

for name in [
    :size,
    :(LinearAlgebra.lapack_size),
    :(Base.require_one_based_indexing),
    :in,
    :iszero,
    :isempty,
    :isbitstype,
    :sizeof,
    :promote_type,
    :(Base.elsize),
    :(Core.Compiler.sizeof_nothrow),
    :(Base.datatype_haspadding),
    :(Base.datatype_nfields),
    :(Base.datatype_pointerfree),
    :(Base.datatype_alignment),
    :(Base.datatype_fielddesc_type),
]
    @eval isprimitive(::RMC, ::Core.Typeof($name), args...) = true
    @eval function rrule!!(::CoDual{Core.Typeof($name)}, args::CoDual...)
        v = $name(map(primal, args)...)
        return CoDual(v, zero_tangent(v)), NoPullback()
    end
end

lgetfield(x, ::SSym{f}) where {f} = getfield(x, f)

lgetfield(x::Tuple, ::SInt{i}) where {i} = getfield(x, i)

function rrule!!(::CoDual{typeof(lgetfield)}, x::CoDual, ::CoDual{T}) where {f, T<:Union{SSym{f}, SInt{f}}}
    lgetfield_pb!!(dy, df, dx, dsym) = df, increment_field!!(dx, dy, T()), dsym
    y = CoDual(getfield(primal(x), f), _get_shadow_field(primal(x), shadow(x), f))
    return y, lgetfield_pb!!
end

Umlaut.isprimitive(::RMC, ::typeof(lgetfield), args...) = true
