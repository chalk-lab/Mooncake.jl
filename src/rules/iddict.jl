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

# Unlift rebuilds a reverse tangent, so each value must be unlifted rather than taken from the
# lane accessor, which yields a `MutableDualTangentView` for a mutable-struct value type. An
# `IdDict` is mutable, so register the shell before recursing.
@inline unlift(x::Lifted{P,1,<:IdDict}) where {P<:IdDict} = (
    primal(x), _unlift_seed(x, IdDict{Any,Any}())
)
function _unlift_seed(x::Lifted{P,1,<:IdDict}, cache::IdDict) where {P<:IdDict}
    p = primal(x)
    haskey(cache, p) && return cache[p]
    t = tangent_type(P)()
    cache[p] = t
    for (k, v) in tangent(x)
        pk = p[k]
        t[k] = _unlift_seed(Lifted{typeof(pk),1}(pk, v), cache)
    end
    return t
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
# An `IdDict`'s forward V is an `IdDict` over the same keys holding each value's V. It is not an
# `AbstractArray`, so the harness's element-wise method does not match it.
function TestUtils._chunked_v_invariant(p::IdDict, v::IdDict, c::IdDict)
    haskey(c, v) && return true
    c[v] = nothing
    length(p) == length(v) || return false
    return all(k -> haskey(v, k) && TestUtils._chunked_v_invariant(p[k], v[k], c), keys(p))
end

function TestUtils.has_equal_data_internal(
    p::P, q::P, equal_undefs::Bool, d::IdDict{Any,Bool}
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

# Forward-mode canonical V for `IdDict{K, V}` — one dict mapping K to the
# value type's canonical N-width V. Matches reverse-mode `tangent_type` shape
# (one dict, K → tangent_type(V)) but with V replaced by `dual_type(Val(N), V)`.
@foldable @inline function dual_type(::Val{N}, ::Type{IdDict{K,V}}) where {N,K,V}
    return IdDict{K,dual_type(Val(N), V)}
end
# No `lifted_type(::IdDict)` method needed: the generic concrete-struct `lifted_type` returns
# `Lifted{P,N,dual_type(Val(N),P)}`, which for a concrete `IdDict{K,V}` uses the `dual_type` above
# and yields exactly `Lifted{IdDict{K,V},N,IdDict{K,dual_type(Val(N),V)}}`.

# Forward seed / lift / lane-accessor for the custom V `IdDict{K, dual_type(V)}`. Without these the
# generic struct-lift fallback fires on `IdDict`'s `ht::Memory{Any}` field and builds an invalid
# `MutableDual{Memory{Any}}`. Mirror the reverse `*_tangent_internal` per-value recursion (with the
# same aliasing/cycle cache), the `lift` boundary, and the AbstractArray lane accessor.
for f in (:_zero_dual_internal, :_uninit_dual_internal)
    @eval function $f(w::Val{N}, x::IdDict{K,V}, c::MaybeCache) where {N,K,V}
        DV = dual_type(Val(N), V)
        haskey(c, x) && return c[x]::IdDict{K,DV}
        out = IdDict{K,DV}()
        c[x] = out
        for (k, v) in x
            out[k] = $f(w, v, c)
        end
        return out
    end
end
function _randn_dual_internal(
    w::Val{N}, rng::AbstractRNG, x::IdDict{K,V}, c::MaybeCache
) where {N,K,V}
    DV = dual_type(Val(N), V)
    haskey(c, x) && return c[x]::IdDict{K,DV}
    out = IdDict{K,DV}()
    c[x] = out
    for (k, v) in x
        out[k] = _randn_dual_internal(w, rng, v, c)
    end
    return out
end
# Width-1 boundary: pair each primal value with its reverse tangent to build the forward V.
@inline lift(x::IdDict, ẋ::IdDict) = lift(x, ẋ, nothing)
# Cache-threading form mirroring the reverse `_zero_dual_internal(::IdDict)` factory above and the
# struct/array `lift` boundaries: register the (empty) `out` V in the aliasing cache `c` BEFORE
# recursing into the values, so aliased values share one V and a self-referential / cyclic IdDict
# terminates instead of overflowing the stack (the reverse oracle's IdDict factories all guard).
function lift(x::IdDict{K,V}, ẋ::IdDict, c::Union{Nothing,IdDict}) where {K,V}
    d = c === nothing ? IdDict() : c
    haskey(d, x) && return d[x]::Lifted{IdDict{K,V},1}
    DV = dual_type(Val(1), V)
    out = IdDict{K,DV}()
    lifted = Lifted{IdDict{K,V},1}(x, out)
    d[x] = lifted
    for (k, v) in x
        out[k] = tangent(lift(v, ẋ[k], d))
    end
    return lifted
end
# Lane accessor: extract lane `l` from each value's V, producing the reverse `tangent_type` dict.
@inline function tangent(
    x::Lifted{IdDict{K,V},N,IdDict{K,DV}}, lane::Integer
) where {K,V,N,DV}
    p = primal(x)
    v = tangent(x)
    # Concrete `typeof(pe)`, not the declared `V`: an `IdDict{K,Any}` would otherwise build
    # `Lifted{Any,N,...}` children, which the lane methods dispatch on and mishandle.
    entries = [k => tangent(Lifted{typeof(pe),N}(pe, v[k]), lane) for (k, pe) in p]
    # The value type comes from the reads, not from `tangent_type(V)`: a mutable value's lane
    # tangent is a live write-through view rather than a materialised `MutableTangent`, so the
    # reverse-shaped type does not hold it. Empty keeps the reverse shape, having nothing to read.
    isempty(entries) && return IdDict{K,tangent_type(V)}()
    return IdDict(entries)
end

function frule!!(
    ::Lifted{typeof(Base.rehash!),N}, d::Lifted{<:IdDict,N}, newsz::Lifted
) where {N}
    Base.rehash!(primal(d), primal(newsz))
    Base.rehash!(tangent(d), primal(newsz))
    return d
end
function rrule!!(::CoDual{typeof(Base.rehash!)}, d::CoDual{<:IdDict}, newsz::CoDual)
    Base.rehash!(primal(d), primal(newsz))
    Base.rehash!(tangent(d), primal(newsz))
    return d, NoPullback((NoRData(), NoRData(), NoRData()))
end

# Rebuild `dv` over `stored` so the primal and its dual share storage. Converting an
# `NDualArray`'s element type must allocate a new primal, which severs it from the stored object.
@inline function _fwd_dual_over_stored(
    ::Val{N}, stored::Array{E}, dv::NDualArray
) where {N,E<:NDualEltype}
    out = zero_dual(Val(N), stored)
    copyto!(getfield(out, :partials_block), getfield(dv, :partials_block))
    return out
end
@inline _fwd_dual_over_stored(::Val, stored, _) = throw(
    ArgumentError(
        "forward mode cannot store into an `IdDict` with value type $(typeof(stored)) when the " *
        "conversion allocates: the dual cannot be rebuilt over the stored object, so a later " *
        "mutation through the dict would be lost. Convert the value before storing it.",
    ),
)

@is_primitive MinimalCtx Tuple{typeof(setindex!),IdDict,Any,Any}
function frule!!(
    ::Lifted{typeof(setindex!),N},
    d::Lifted{IdDict{K,V},N,IdDict{K,Vdv}},
    val::Lifted,
    key::Lifted,
) where {N,K,V,Vdv}
    setindex!(primal(d), primal(val), primal(key))
    # `setindex!` above stored `convert(V, val)`, so the dual slot is `dual_type(Val(N), V)` (= `Vdv`),
    # not the dual type of `val`'s own type.
    dslot = if Vdv == NoDual
        NoDual()
    elseif dual_type(Val(N), typeof(primal(val))) == NoDual
        zero_dual(Val(N), primal(d)[primal(key)])
    else
        # A conversion that allocated a fresh mutable value leaves `val`'s dual over an object the
        # dict does not hold, so a mutation through the dict would be invisible to it.
        stored = primal(d)[primal(key)]
        if stored === primal(val) || !ismutable(stored)
            tangent(val)
        else
            _fwd_dual_over_stored(Val(N), stored, tangent(val))
        end
    end
    setindex!(tangent(d), dslot, primal(key))
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
    # `setindex!` above stored `convert(V, val)`, so the slot is `tangent_type(V)`. The
    # two-arg `zero_tangent` reuses `val`'s fdata to preserve aliasing for mutable values.
    dslot = if tangent_type(V) == NoTangent
        NoTangent()
    elseif tangent_type(typeof(primal(val))) == NoTangent
        zero_tangent(primal(d)[k])
    else
        zero_tangent(primal(val), tangent(val))
    end
    setindex!(tangent(d), dslot, k)

    dval = lazy_zero_rdata(primal(val))
    dkey = lazy_zero_rdata(primal(key))
    function setindex_pb!!(::NoRData)

        # Map the slot cotangent back to `val`: zero if either side is non-diff, a direct
        # increment for matched/abstract `V`, else undo the widening (`fptrunc` for floats).
        S = tangent_type(typeof(primal(val)))
        _dval = if tangent_type(V) == NoTangent || S == NoTangent
            instantiate(dval)
        elseif S <: tangent_type(V)
            increment!!(instantiate(dval), rdata(tangent(d)[k]))
        else
            # The slot holds `convert(V, val)`, a fresh object, so what accumulates there
            # reaches `val`'s fdata only by an explicit copy-back. The result is discarded
            # because every gradient carrier in fdata is mutable, so this lands in place.
            increment!!(tangent(val), convert(fdata_type(S), fdata(tangent(d)[k])))
            increment!!(instantiate(dval), convert(rdata_type(S), rdata(tangent(d)[k])))
        end

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
    ::Lifted{typeof(get),N}, d::Lifted{IdDict{K,V},N}, key::Lifted, default::Lifted
) where {N,K,V}
    _key = primal(key)
    # Key absent ⇒ return the `default` slot unchanged, mirroring the reverse rrule's
    # `has_key ? ... : default`. Building `Lifted{V,N}(default, ...)` would mis-type a
    # default whose type differs from the dict value type `V` (the ctor requires `primal::V`).
    haskey(primal(d), _key) || return default
    # Typed from the STORED VALUE, not from the dict's declared `V`: for `V === Any` the latter
    # gives a `Lifted{Any,…}` slot that downstream frule dispatch has no method for. Mirrors the
    # reverse rrule, which derives the `CoDual`'s primal type from the value.
    y = primal(d)[_key]
    return Lifted{typeof(y),N}(y, tangent(d)[_key])
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
            # Key absent: `y === default`, so `dy` is exactly `default`'s cotangent.
            _rdefault = dy
        end
        return NoRData(), NoRData(), instantiate(dkey), _rdefault
    end
    return y, get_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(getindex),IdDict,Any}
function frule!!(
    ::Lifted{typeof(getindex),N}, d::Lifted{IdDict{K,V},N}, key::Lifted
) where {N,K,V}
    y = getindex(primal(d), primal(key))
    return Lifted{typeof(y),N}(y, getindex(tangent(d), primal(key)))
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
    @eval function frule!!(
        ::Lifted{typeof(_foreigncall_),N}, ::Lifted{Val{$name},N}, args...
    ) where {N}
        return unexpected_foreigncall_error($name)
    end
    @eval function rrule!!(::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{$name}}, args...)
        return unexpected_foreigncall_error($name)
    end
end

@is_primitive MinimalCtx Tuple{Type{IdDict{K,V}} where {K,V}}
function frule!!(::Lifted{Type{IdDict{K,V}},N}) where {N,K,V}
    return Lifted{IdDict{K,V},N}(IdDict{K,V}(), IdDict{K,dual_type(Val(N), V)}())
end
function rrule!!(f::CoDual{Type{IdDict{K,V}}}) where {K,V}
    return CoDual(IdDict{K,V}(), IdDict{K,tangent_type(V)}()), NoPullback(f)
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:iddict})
    test_cases = Any[
        (false, :stability, nothing, Base.rehash!, IdDict(true => 5.0, false => 4.0), 10),
        (false, :none, nothing, setindex!, IdDict(true => 5.0, false => 4.0), 3.0, false),
        (false, :none, nothing, setindex!, IdDict(true => 5.0), 3.0, false),
        # type-mismatched stores (typeof(val) ≠ V): non-diff/diff slots, mutable & abstract
        # values (aliasing), and a floating-point width change (fpext/fptrunc).
        (false, :none, nothing, setindex!, IdDict(:a => 1.0), 2, :b),
        # interface_only: the finite-difference perturbation would break `convert(Int, ...)`.
        (true, :none, nothing, setindex!, IdDict(:a => 1), 3.0, :b),
        (
            false,
            :none,
            nothing,
            setindex!,
            IdDict{Symbol,Vector{Float64}}(:a => [1.0, 2.0]),
            [3.0, 4.0],
            :b,
        ),
        (false, :none, nothing, setindex!, IdDict{Symbol,Any}(:a => 1.0), 2.0, :b),
        (false, :none, nothing, setindex!, IdDict{Symbol,Any}(:a => [1.0]), [2.0, 3.0], :b),
        (false, :none, nothing, setindex!, IdDict{Symbol,Float64}(:a => 1.0), 2.0f0, :b),
        # the same width change for an array, where the gradient rides fdata, not rdata
        (
            false,
            :none,
            nothing,
            setindex!,
            IdDict{Symbol,Vector{Float64}}(:a => [1.0, 2.0]),
            Float32[3.0, 4.0],
            :b,
        ),
        (false, :none, nothing, get, IdDict(true => 5.0, false => 4.0), false, 2.0),
        (false, :none, nothing, get, IdDict(true => 5.0), false, 2.0),
        # Absent key with a default whose type differs from the dict value type V:
        # the frule must return the `default` slot, not force `Lifted{V}` (regression).
        (false, :none, nothing, get, IdDict(true => 5.0), false, 2.0f0),
        # `get` returning a default (absent key) whose rdata type differs from its tangent type.
        (
            false,
            :none,
            nothing,
            get,
            IdDict{Symbol,Vector{Float64}}(:a => [1.0]),
            :b,
            [2.0, 3.0],
        ),
        # interface_only: the non-differentiable default carries no derivative.
        (true, :none, nothing, get, IdDict{Symbol,Float64}(:a => 1.0), :b, 2),
        (false, :none, nothing, getindex, IdDict(true => 5.0, false => 4.0), true),
        # `V === Any`: the slot must be typed from the STORED VALUE. Typing it from the
        # declared `V` gives a `Lifted{Any,…}` that downstream frule dispatch has no method for,
        # while every concrete-`V` case above passes because there the two agree.
        (false, :none, nothing, getindex, IdDict{Symbol,Any}(:a => 2.0), :a),
        (false, :none, nothing, get, IdDict{Symbol,Any}(:a => 2.0), :a, 0.0),
        # A MUTABLE-STRUCT value type: unlifting the dict argument has to rebuild a reverse
        # tangent per value, since the lane accessor gives a `MutableDualTangentView` that
        # `IdDict{Symbol,MutableTangent}` storage cannot hold. Array and scalar value types
        # are both leaves and so miss this.
        (
            false,
            :none,
            nothing,
            getindex,
            IdDict{Symbol,TestResources.TypeStableMutableStruct{Float64}}(
                :a => TestResources.TypeStableMutableStruct{Float64}(5.0, 4.0)
            ),
            :a,
        ),
        (false, :none, nothing, IdDict{Any,Any}),
    ]
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:iddict})
    # A store whose `convert` allocates, then a mutation THROUGH the dict. Storing alone does not
    # catch it: an unreachable primal reads the same as the right one until something mutates.
    function converting_store_then_mutate(x::Vector{Float32})
        d = IdDict{Int,Vector{Float64}}()
        d[1] = x
        d[1][1] += 1.0
        return sum(d[1])
    end
    test_cases = Any[(
        false, :none, nothing, converting_store_then_mutate, Float32[3.0, 4.0]
    )]
    return test_cases, Any[]
end
