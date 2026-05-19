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
function tangent_to_primal_internal!!(x::P, t, c::MaybeCache) where {P<:IdDict}
    haskey(c, x) && return c[x]::P
    @assert union(keys(x), keys(t)) == keys(x)
    c[x] = x
    for k in keys(x)
        x[k] = tangent_to_primal_internal!!(x[k], t[k], c)
    end
    return x
end
function primal_to_tangent_internal!!(t, x::P, c::MaybeCache) where {P<:IdDict}
    haskey(c, x) && return c[x]::tangent_type(P)
    @assert union(keys(t), keys(x)) == keys(t)
    c[x] = t
    for k in keys(t)
        t[k] = primal_to_tangent_internal!!(t[k], x[k], c)
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(Base.rehash!),<:IdDict,Any}}) = true
# `tangent(d::Dual{P, NTangent{Tuple{T}}})` returns the NTangent wrapper
# rather than the bare T. For mutation-semantic rules (rehash!, setindex!,
# getindex on IdDict), unwrap the singleton lane so the underlying mutable
# IdDict is operated on rather than the wrapper.
@inline _iddict_tangent(d::Dual{<:IdDict}) = Mooncake._ntangent_unwrap_singleton(tangent(d))
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.rehash!),N},
    d::Mooncake.Lifted{<:IdDict},
    newsz::Mooncake.Lifted,
) where {N}
    inner_d = Mooncake._unlift(d)
    inner_newsz = Mooncake._unlift(newsz)
    sz = primal(inner_newsz)
    Base.rehash!(primal(inner_d), sz)
    # rehash! all N lane IDdicts. Width-1 unwraps to a single bare IdDict;
    # width-N has NTangent{NTuple{N, IdDict}} — rehash each.
    _rehash_iddict_tangent!(tangent(inner_d), sz)
    return d
end
@inline _rehash_iddict_tangent!(t::Mooncake.NTangent, sz) = foreach(
    td -> Base.rehash!(td, sz), t.lanes
)
@inline _rehash_iddict_tangent!(t, sz) = Base.rehash!(t, sz)
function rrule!!(::CoDual{typeof(Base.rehash!)}, d::CoDual{<:IdDict}, newsz::CoDual)
    Base.rehash!(primal(d), primal(newsz))
    Base.rehash!(tangent(d), primal(newsz))
    return d, NoPullback((NoRData(), NoRData(), NoRData()))
end

@is_primitive MinimalCtx Tuple{typeof(setindex!),IdDict,Any,Any}
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(setindex!),<:IdDict,Any,Any}}) =
    true
# Implementation kernels for `setindex!(::IdDict, val, key)`. The Lifted body
# below dispatches on the runtime val V (`Dual` or `NDual` for IEEEFloat).
@inline function _setindex_iddict!(d::Dual{IdDict{K,V}}, val::Dual, key) where {K,V}
    setindex!(primal(d), primal(val), primal(key))
    setindex!(
        _iddict_tangent(d), Mooncake._ntangent_unwrap_singleton(tangent(val)), primal(key)
    )
    return nothing
end
@inline function _setindex_iddict!(
    d::Dual{IdDict{K,V}}, val::Mooncake.Nfwd.NDual{P,1}, key
) where {K,V<:IEEEFloat,P<:IEEEFloat}
    setindex!(primal(d), val.value, primal(key))
    setindex!(_iddict_tangent(d), val.partials[1], primal(key))
    return nothing
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(setindex!),N},
    d::Mooncake.Lifted{<:IdDict},
    val::Mooncake.Lifted,
    key::Mooncake.Lifted,
) where {N}
    _setindex_iddict!(Mooncake._unlift(d), Mooncake._unlift(val), Mooncake._unlift(key))
    return d
end
# Width-N setindex! for `Dual{<:IdDict, NTangent{...}}` slot. Each lane has
# its own IDdict tangent — set the per-lane value from `tangent(val, n)`.
@inline function frule!!(
    f::Mooncake.Lifted{typeof(setindex!),N},
    d::Mooncake.Lifted{<:IdDict,N,<:Dual{<:IdDict,<:Mooncake.NTangent}},
    val::Mooncake.Lifted,
    key::Mooncake.Lifted,
) where {N}
    N == 1 && return @invoke frule!!(
        f::Mooncake.Lifted{typeof(setindex!),N},
        d::Mooncake.Lifted{<:IdDict},
        val::Mooncake.Lifted,
        key::Mooncake.Lifted,
    )
    bare_d = Mooncake._unlift(d)
    bare_v = Mooncake._unlift(val)
    bare_k = primal(Mooncake._unlift(key))
    setindex!(primal(bare_d), primal(bare_v), bare_k)
    # Per-lane tangent write.
    for n in 1:N
        setindex!(tangent(bare_d).lanes[n], Mooncake.tangent(val, n), bare_k)
    end
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(get),<:IdDict,Any,Any}}) = true
# Implementation kernels for `get(::IdDict, key, default)`. Lifted body
# dispatches on the runtime default V.
@inline function _get_iddict(d::Dual{IdDict{K,V}}, key::Dual, default::Dual) where {K,V}
    x = get(primal(d), primal(key), primal(default))
    dx = get(
        _iddict_tangent(d),
        primal(key),
        Mooncake._ntangent_unwrap_singleton(tangent(default)),
    )
    return Dual(x, dx)
end
@inline function _get_iddict(
    d::Dual{IdDict{K,V}}, key::Dual, default::Mooncake.Nfwd.NDual{P,1}
) where {K,V<:IEEEFloat,P<:IEEEFloat}
    x = get(primal(d), primal(key), default.value)
    dx = get(_iddict_tangent(d), primal(key), default.partials[1])
    return Dual(x, dx)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(get),N},
    d::Mooncake.Lifted{<:IdDict},
    key::Mooncake.Lifted,
    default::Mooncake.Lifted,
) where {N}
    bare_result = _get_iddict(
        Mooncake._unlift(d), Mooncake._unlift(key), Mooncake._unlift(default)
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
# Chunk-size-N correctness: width-N IDdict has tangent
# `NTangent{NTuple{N, IdDict{K, tangent_type(V)}}}` (one IDdict per lane).
# The width-1 `_get_iddict` above only services lane 1. For N >= 2, loop
# per-lane: for each lane `n`, look up `tangent(d, n)[key]` and use
# `tangent(default, n)` as the fallback, then assemble all lanes into an
# `NTangent` result. Mirrors the canonical `frule_wrapper` pattern.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(get),N},
    d::Mooncake.Lifted{<:IdDict,N,<:Dual{<:IdDict,<:Mooncake.NTangent}},
    key::Mooncake.Lifted,
    default::Mooncake.Lifted,
) where {N}
    N == 1 && return @invoke frule!!(
        Mooncake.zero_lifted(Val(N), get)::Mooncake.Lifted{typeof(get),N},
        d::Mooncake.Lifted{<:IdDict},
        key::Mooncake.Lifted,
        default::Mooncake.Lifted,
    )
    bare_d = Mooncake._unlift(d)
    bare_k = primal(Mooncake._unlift(key))
    P_d = primal(bare_d)
    has_key = in(bare_k, keys(P_d))
    P_v = has_key ? P_d[bare_k] : primal(Mooncake._unlift(default))
    # Per-lane tangent lookup.
    lane_tangents = ntuple(Val(N)) do n
        td = tangent(bare_d).lanes[n]
        in(bare_k, keys(td)) ? td[bare_k] : Mooncake.tangent(default, n)
    end
    return Mooncake.Lifted{_typeof(P_v),N}(P_v, Mooncake.NTangent(lane_tangents))
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(getindex),<:IdDict,Any}}) = true
@inline function frule!!(
    ::Mooncake.Lifted{typeof(getindex),N},
    d::Mooncake.Lifted{<:IdDict},
    key::Mooncake.Lifted,
) where {N}
    inner_d = Mooncake._unlift(d)
    inner_key = Mooncake._unlift(key)
    bare_result = Dual(
        getindex(primal(inner_d), primal(inner_key)),
        getindex(_iddict_tangent(inner_d), primal(inner_key)),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
# Width-N getindex on NTangent-wrapped IdDict (parallel to the `get` rule's
# width-N overload). The generic delegator above calls `_iddict_tangent` →
# `Mooncake._ntangent_unwrap_singleton` (singleton-only), then `getindex(::NTangent, key::Int)`
# resolves to `Base.getindex(::NTangent, ::Int)` returning LANE `key` — not
# the value at that key in any of the N per-lane IdDicts. Silent
# correctness bug (returns the wrong shape entirely).
@inline function frule!!(
    ::Mooncake.Lifted{typeof(getindex),N},
    d::Mooncake.Lifted{<:IdDict,N,<:Dual{<:IdDict,<:Mooncake.NTangent}},
    key::Mooncake.Lifted,
) where {N}
    N == 1 && return @invoke frule!!(
        Mooncake.zero_lifted(Val(N), getindex)::Mooncake.Lifted{typeof(getindex),N},
        d::Mooncake.Lifted{<:IdDict},
        key::Mooncake.Lifted,
    )
    bare_d = Mooncake._unlift(d)
    bare_k = primal(Mooncake._unlift(key))
    P_v = getindex(primal(bare_d), bare_k)
    lane_tangents = ntuple(Val(N)) do n
        getindex(tangent(bare_d).lanes[n], bare_k)
    end
    return Mooncake.Lifted{_typeof(P_v),N}(P_v, Mooncake.NTangent(lane_tangents))
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
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{Type{<:IdDict}}}) = true
# `IdDict{K,V}()`: the Lifted body below computes the result independently
# — no kernel or bare-Dual body needed.
@inline function frule!!(::Mooncake.Lifted{Type{IdDict{K,V}},N}) where {K,V,N}
    # Per-lane independent tangent IdDicts. The naive
    # `Lifted{IdDict, N}(IdDict_primal, single_tangent_IdDict)` would
    # broadcast a single IdDict across all N NTangent lanes (silent
    # aliasing — writes to one lane corrupt all). At N==1 the broadcast
    # is a no-op (singleton); at N≥2 we allocate N independent tangent
    # IdDicts explicitly.
    primal_dict = IdDict{K,V}()
    if N >= 2
        InnerT = Mooncake.dual_type(Val(N), IdDict{K,V})
        if InnerT isa DataType &&
            InnerT <: Dual &&
            InnerT.parameters[2] <: Mooncake.NTangent
            tangents = ntuple(_ -> IdDict{K,tangent_type(V)}(), Val(N))
            return Mooncake.Lifted{IdDict{K,V},N,InnerT}(
                InnerT(primal_dict, Mooncake.NTangent(tangents))
            )
        end
    end
    return Mooncake.Lifted{IdDict{K,V},N}(primal_dict, IdDict{K,tangent_type(V)}())
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
