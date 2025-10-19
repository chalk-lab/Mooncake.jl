# We're going to use `IdDict`s to represent tangents for `IdDict`s.

@foldable tangent_type(::Type{<:IdDict{K,V}}) where {K,V} = IdDict{K,tangent_type(V)}

function zero_tangent_internal(d::P, dict::MaybeCache) where {P<:IdDict}
    T = tangent_type(P)
    if haskey(dict, d)
        return dict[d]::T
    else
        t = T([k => zero_tangent_internal(v, dict) for (k, v) in d])
        dict[d] = t
        return t
    end
end

function randn_tangent_internal(rng::AbstractRNG, d::P, dict::MaybeCache) where {P<:IdDict}
    T = tangent_type(P)
    if haskey(dict, d)
        return dict[d]::T
    else
        t = T([k => randn_tangent_internal(rng, v, dict) for (k, v) in d])
        dict[d] = t
        return t
    end
end

function increment_internal!!(c::IncCache, p::T, q::T) where {T<:IdDict}
    haskey(c, p) && return p
    for k in keys(p)
        p[k] = increment_internal!!(c, p[k], q[k])
    end
    return p
end
function set_to_zero_internal!!(c::SetToZeroCache, t::IdDict)
    _already_tracked!(c, t) && return t
    foreach(keys(t)) do k
        t[k] = set_to_zero_internal!!(c, t[k])
    end
    return t
end
function _scale_internal(c::MaybeCache, a::Float64, t::IdDict{K,V}) where {K,V}
    haskey(c, t) && return c[t]::IdDict{K,V}
    t′ = IdDict{K,V}()
    c[t] = t′
    for (k, v) in t
        t′[k] = _scale_internal(c, a, v)
    end
    return t′
end
function _dot_internal(c::MaybeCache, p::T, q::T) where {T<:IdDict}
    key = (p, q)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return sum(keys(p); init=0.0) do k
        _dot_internal(c, p[k], q[k])::Float64
    end
end
function _add_to_primal_internal(
    c::MaybeCache, p::IdDict{K,V}, t::IdDict{K}, unsafe::Bool
) where {K,V}
    key = (p, t, unsafe)
    haskey(c, key) && return c[key]::IdDict{K,V}
    p′ = IdDict{K,V}()
    c[key] = p′
    ks = intersect(keys(p), keys(t))
    for k in ks
        p′[k] = _add_to_primal_internal(c, p[k], t[k], unsafe)
    end
    return p′
end
function _diff_internal(c::MaybeCache, p::P, q::P) where {K,V,P<:IdDict{K,V}}
    @assert union(keys(p), keys(q)) == keys(p)
    key = (p, q)
    haskey(c, key) && return c[key]::tangent_type(P)
    t = IdDict{K,tangent_type(V)}()
    c[key] = t
    for k in keys(p)
        t[k] = _diff_internal(c, p[k], q[k])
    end
    return t
end
function TestUtils.populate_address_map_internal(
    m::TestUtils.AddressMap, p::IdDict, t::IdDict
)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    if haskey(m, k)
        @assert m[k] == v
        return m
    end
    m[k] = v
    foreach(n -> TestUtils.populate_address_map_internal(m, p[n], t[n]), keys(p))
    return m
end
function TestUtils.has_equal_data_internal(
    p::P, q::P, equal_undefs::Bool, d::Dict{Tuple{UInt,UInt},Bool}
) where {P<:IdDict}
    ks = union(keys(p), keys(q))
    ks != keys(p) && return false
    return all([TestUtils.has_equal_data_internal(p[k], q[k], equal_undefs, d) for k in ks])
end

fdata_type(::Type{T}) where {T<:IdDict} = T
fdata(t::IdDict) = t
rdata_type(::Type{<:IdDict}) = NoRData
rdata(t::IdDict) = NoRData()

__verify_fdata_value(::IdDict{Any,Nothing}, p::IdDict, f::IdDict) = nothing

@foldable tangent_type(::Type{T}, ::Type{NoRData}) where {T<:IdDict} = T
tangent(f::IdDict, ::NoRData) = f

# All of the rules in here are provided in order to avoid nasty `:ccall`s, and to support
# standard built-in functionality on `IdDict`s.

@is_primitive MinimalCtx Tuple{typeof(Base.rehash!),IdDict,Any}
function frule!!(::Dual{typeof(Base.rehash!)}, d::Dual{<:IdDict}, newsz::Dual)
    Base.rehash!(primal(d), primal(newsz))
    Base.rehash!(tangent(d), primal(newsz))
    return d
end
function rrule!!(::CoDual{typeof(Base.rehash!)}, d::CoDual{<:IdDict}, newsz::CoDual)
    Base.rehash!(primal(d), primal(newsz))
    Base.rehash!(tangent(d), primal(newsz))
    return d, NoPullback((NoRData(), NoRData(), NoRData()))
end

@is_primitive MinimalCtx Tuple{typeof(setindex!),IdDict,Any,Any}
function frule!!(::Dual{typeof(setindex!)}, d::Dual{IdDict{K,V}}, val, key) where {K,V}
    setindex!(primal(d), primal(val), primal(key))
    setindex!(tangent(d), tangent(val), primal(key))
    return d
end
function rrule!!(::CoDual{typeof(setindex!)}, d::CoDual{IdDict{K,V}}, val, key) where {K,V}
    k = primal(key)
    restore_state = in(k, keys(primal(d)))
    if restore_state
        old_primal_val = primal(d)[k]
        old_tangent_val = tangent(d)[k]
    end

    setindex!(primal(d), primal(val), k)
    setindex!(tangent(d), zero_tangent(primal(val), tangent(val)), k)

    dval = lazy_zero_rdata(primal(val))
    dkey = lazy_zero_rdata(primal(key))
    function setindex_pb!!(::NoRData)

        # Increment tangent.
        _dval = increment!!(instantiate(dval), rdata(tangent(d)[k]))

        # Restore previous state if necessary.
        if restore_state
            primal(d)[k] = old_primal_val
            tangent(d)[k] = old_tangent_val
        else
            delete!(primal(d), k)
            delete!(tangent(d), k)
        end

        return NoRData(), NoRData(), _dval, instantiate(dkey)
    end
    return d, setindex_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(get),IdDict,Any,Any}
function frule!!(
    ::Dual{typeof(get)}, d::Dual{IdDict{K,V}}, key::Dual, default::Dual
) where {K,V}
    x = get(primal(d), primal(key), primal(default))
    dx = get(tangent(d), primal(key), tangent(default))
    return Dual(x, dx)
end
function rrule!!(
    ::CoDual{typeof(get)}, d::CoDual{IdDict{K,V}}, key::CoDual, default::CoDual
) where {K,V}
    k = primal(key)
    has_key = in(k, keys(primal(d)))
    y = has_key ? CoDual(primal(d)[k], fdata(tangent(d)[k])) : default

    dd = tangent(d)
    dkey = lazy_zero_rdata(primal(key))
    rdefault = lazy_zero_rdata(primal(default))
    function get_pb!!(dy)
        if has_key
            dd[k] = increment_rdata!!(dd[k], dy)
            _rdefault = instantiate(rdefault)
        else
            _rdefault = increment_rdata!!(instantiate(rdefault), dy)
        end
        return NoRData(), NoRData(), instantiate(dkey), _rdefault
    end
    return y, get_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(getindex),IdDict,Any}
function frule!!(::Dual{typeof(getindex)}, d::Dual{IdDict{K,V}}, key::Dual) where {K,V}
    return Dual(getindex(primal(d), primal(key)), getindex(tangent(d), primal(key)))
end
function rrule!!(
    ::CoDual{typeof(getindex)}, d::CoDual{IdDict{K,V}}, key::CoDual
) where {K,V}
    k = primal(key)
    y = CoDual(getindex(primal(d), k), fdata(getindex(tangent(d), k)))
    dkey = lazy_zero_rdata(primal(key))
    dd = tangent(d)
    function getindex_pb!!(dy)
        dd[k] = increment_rdata!!(dd[k], dy)
        return NoRData(), NoRData(), instantiate(dkey)
    end
    return y, getindex_pb!!
end

for name in
    [:(:jl_idtable_rehash), :(:jl_eqtable_put), :(:jl_eqtable_get), :(:jl_eqtable_nextind)]
    @eval function frule!!(::Dual{typeof(_foreigncall_)}, ::Dual{Val{$name}}, args...)
        return unexpected_foreigncall_error($name)
    end
    @eval function rrule!!(::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{$name}}, args...)
        return unexpected_foreigncall_error($name)
    end
end

@is_primitive MinimalCtx Tuple{Type{IdDict{K,V}} where {K,V}}
function frule!!(::Dual{Type{IdDict{K,V}}}) where {K,V}
    return Dual(IdDict{K,V}(), IdDict{K,tangent_type(V)}())
end
function rrule!!(f::CoDual{Type{IdDict{K,V}}}) where {K,V}
    return CoDual(IdDict{K,V}(), IdDict{K,tangent_type(V)}()), NoPullback(f)
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:iddict})
    test_cases = Any[
        (false, :stability, nothing, Base.rehash!, IdDict(true => 5.0, false => 4.0), 10),
        (false, :none, nothing, setindex!, IdDict(true => 5.0, false => 4.0), 3.0, false),
        (false, :none, nothing, setindex!, IdDict(true => 5.0), 3.0, false),
        (false, :none, nothing, get, IdDict(true => 5.0, false => 4.0), false, 2.0),
        (false, :none, nothing, get, IdDict(true => 5.0), false, 2.0),
        (false, :none, nothing, getindex, IdDict(true => 5.0, false => 4.0), true),
        (false, :none, nothing, IdDict{Any,Any}),
    ]
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:iddict}) = Any[], Any[]
