for D in (Dict, IdDict)
    @eval @foldable tangent_type(::Type{<:$D{K,V}}) where {K,V} = $D{K,tangent_type(V)}
end

function zero_tangent_internal(d::P, dict::MaybeCache) where {P<:KeyedDict}
    T = tangent_type(P)
    if haskey(dict, d)
        return dict[d]::T
    else
        t = T()
        dict[d] = t
        for (k, v) in d
            t[k] = zero_tangent_internal(v, dict)
        end
        return t
    end
end

function randn_tangent_internal(
    rng::AbstractRNG, d::P, dict::MaybeCache
) where {P<:KeyedDict}
    T = tangent_type(P)
    if haskey(dict, d)
        return dict[d]::T
    else
        t = T()
        dict[d] = t
        for (k, v) in d
            t[k] = randn_tangent_internal(rng, v, dict)
        end
        return t
    end
end

# A keyed dictionary is mutable, so register the shell before recursing into the values.
@inline tangent(x::Lifted{P,N,<:KeyedDict}, lane::Integer) where {P<:KeyedDict,N} = _materialise_lane(
    x, lane, IdDict{Any,Any}()
)
function _materialise_lane(
    x::Lifted{P,N,<:KeyedDict}, lane::Integer, cache::IdDict
) where {P<:KeyedDict,N}
    p = primal(x)
    haskey(cache, p) && return cache[p]
    t = tangent_type(P)()
    cache[p] = t
    for (k, v) in tangent(x)
        pk = p[k]
        t[k] = _materialise_lane(Lifted{typeof(pk),N}(pk, v), lane, cache)
    end
    return t
end

@inline _replace_dict_value!(d::IdDict, v, k) = setindex!(d, v, k)
@inline function _replace_dict_value!(d::Dict, v, k)
    i = Base.ht_keyindex(d, k)
    i > 0 || throw(KeyError(k))
    @inbounds d.vals[i] = v
    return d
end

# Existing-value writes have setindex!'s derivative, without a possible table growth in
# tangent algebra. Keeping a primitive boundary also permits forward-over-reverse AD.
@is_primitive MinimalCtx Tuple{typeof(_replace_dict_value!),KeyedDict,Any,Any}

# NoTangent values need no arithmetic. Keep the keyed holder for the rules for now;
# collapsing it to NoTangent requires NoFData variants of the dictionary rules.
# Zero construction still needs O(length(d)) storage to retain the key map.
function increment_internal!!(c::IncCache, p::T, q::T) where {T<:KeyedDict}
    _check_direction_keys(p, q, "cotangent")
    valtype(T) === NoTangent && return p
    (p === q || haskey(c, p)) && return p
    c[p] = true
    for k in keys(p)
        _replace_dict_value!(p, increment_internal!!(c, p[k], q[k]), k)
    end
    return p
end
function set_to_zero_internal!!(c::SetToZeroCache, t::KeyedDict)
    valtype(typeof(t)) === NoTangent && return t
    _already_tracked!(c, t) && return t
    foreach(keys(t)) do k
        _replace_dict_value!(t, set_to_zero_internal!!(c, t[k]), k)
    end
    return t
end
function _scale_internal(c::MaybeCache, a::Float64, t::T) where {T<:KeyedDict}
    valtype(T) === NoTangent && return t
    haskey(c, t) && return c[t]::T
    t′ = T()
    c[t] = t′
    for (k, v) in t
        t′[k] = _scale_internal(c, a, v)
    end
    return t′
end
function _dot_internal(c::MaybeCache, p::T, q::T) where {T<:KeyedDict}
    valtype(T) === NoTangent && return 0.0
    key = (p, q)
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    return sum(keys(p); init=0.0) do k
        _dot_internal(c, p[k], q[k])::Float64
    end
end
function _add_to_primal_internal(
    c::MaybeCache, p::P, t::KeyedDict, unsafe::Bool
) where {P<:KeyedDict}
    key = (p, t, unsafe)
    haskey(c, key) && return c[key]::P
    p′ = P()
    c[key] = p′
    _check_direction_keys(p, t, "native dictionary")
    ks = keys(p)
    for k in ks
        p′[k] = _add_to_primal_internal(c, p[k], t[k], unsafe)
    end
    return p′
end
function tangent_to_primal_internal!!(x::P, t, c::MaybeCache) where {P<:KeyedDict}
    haskey(c, x) && return c[x]::P
    _check_direction_keys(x, t, "tangent")
    c[x] = x
    for k in keys(x)
        x[k] = tangent_to_primal_internal!!(x[k], t[k], c)
    end
    return x
end
function primal_to_tangent_internal!!(t, x::P, c::MaybeCache) where {P<:KeyedDict}
    haskey(c, x) && return c[x]::tangent_type(P)
    _check_direction_keys(t, x, "cotangent")
    c[x] = t
    for k in keys(t)
        t[k] = primal_to_tangent_internal!!(t[k], x[k], c)
    end
    return t
end
function TestUtils.populate_address_map_internal(
    m::TestUtils.AddressMap, p::KeyedDict, t::KeyedDict
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
# A dictionary's forward V holds each value's V at the same key. It is not an
# `AbstractArray`, so the harness's element-wise method does not match it.
function TestUtils._chunked_v_invariant(p::KeyedDict, v::KeyedDict, c::IdDict)
    haskey(c, v) && return true
    c[v] = nothing
    length(p) == length(v) || return false
    return all(k -> haskey(v, k) && TestUtils._chunked_v_invariant(p[k], v[k], c), keys(p))
end

function TestUtils.has_equal_data_internal(
    p::P, q::P, equal_undefs::Bool, d::IdDict{Any,Bool}
) where {P<:KeyedDict}
    haskey(d, (p, q)) && return true
    d[(p, q)] = true
    length(p) == length(q) || return false
    return all(keys(p)) do k
        haskey(q, k) && TestUtils.has_equal_data_internal(p[k], q[k], equal_undefs, d)
    end
end

fdata_type(::Type{T}) where {T<:KeyedDict} = T
fdata(t::KeyedDict) = t
rdata_type(::Type{<:KeyedDict}) = NoRData
rdata(t::KeyedDict) = NoRData()

__verify_fdata_value(::IdDict{Any,Nothing}, p::KeyedDict, f::KeyedDict) = nothing

@foldable tangent_type(::Type{T}, ::Type{NoRData}) where {T<:KeyedDict} = T
tangent(f::KeyedDict, ::NoRData) = f

# All of the rules in here are provided in order to avoid nasty `:ccall`s, and to support
# standard built-in functionality on `IdDict`s.

@is_primitive MinimalCtx Tuple{typeof(Base.rehash!),IdDict,Any}

for D in (Dict, IdDict)
    @eval @foldable @inline function dual_type(::Val{N}, ::Type{$D{K,V}}) where {N,K,V}
        return $D{K,dual_type(Val(N), V)}
    end
end
# Recurse by value with an aliasing/cycle cache; the structural fallback would lift
# the internal hash table instead of constructing the canonical dictionary V.
for f in (:_zero_dual_internal, :_uninit_dual_internal, :_randn_dual_internal)
    rng_args = f === :_randn_dual_internal ? (:(rng::AbstractRNG),) : ()
    rng_vals = f === :_randn_dual_internal ? (:rng,) : ()
    @eval function $f(w::Val{N}, $(rng_args...), x::P, c::MaybeCache) where {N,P<:KeyedDict}
        DV = dual_type(w, P)
        haskey(c, x) && return c[x]::DV
        out = DV()
        c[x] = out
        for (k, v) in x
            out[k] = $f(w, $(rng_vals...), v, c)
        end
        return out
    end
end
# Direct factories also need the custom V. A real cache preserves aliased values
# and terminates cycles; NoCache would create independent partial stores.
for (f, internal) in
    ((:zero_dual, :_zero_dual_internal), (:uninit_dual, :_uninit_dual_internal))
    @eval @inline $f(w::Val{N}, x::KeyedDict) where {N} = $internal(w, x, IdDict{Any,Any}())
end
@inline function randn_dual(w::Val{N}, rng::AbstractRNG, x::KeyedDict) where {N}
    return _randn_dual_internal(w, rng, x, IdDict{Any,Any}())
end
# Width-1 boundary: pair each primal value with its reverse tangent to build the forward V.
@inline lift(x::KeyedDict, ẋ::KeyedDict) = lift(x, ẋ, nothing)
# Register the shell before recursion to preserve aliases and terminate cycles.
function lift(x::P, ẋ::KeyedDict, c::Union{Nothing,IdDict}) where {P<:KeyedDict}
    _check_direction_keys(x, ẋ, "native dictionary")
    d = c === nothing ? IdDict() : c
    haskey(d, x) && return d[x]::Lifted{P,1}
    out = dual_type(Val(1), P)()
    lifted = Lifted{P,1}(x, out)
    d[x] = lifted
    for (k, v) in x
        out[k] = tangent(lift(v, ẋ[k], d))
    end
    return lifted
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

# Rebuild `val`'s dual over `stored` so the primal and its dual share storage. Converting a
# dual's element type would allocate a new primal, severing it from the stored object.
@inline function _fwd_dual_over_stored(
    w::Val{N}, stored::Array{E}, val::Lifted{<:Array{T},N}
) where {N,E<:NDualEltype,T<:NDualEltype}
    return _fwd_dual_over_stored(_storage_kind(E), _storage_kind(T), w, stored, val)
end
@inline function _fwd_dual_over_stored(
    ::NumericStorage, ::NumericStorage, ::Val{N}, stored, val
) where {N}
    out = zero_dual(Val(N), stored)
    block, _ = _partials_block(val)
    copyto!(getfield(out, :partials_block), block)
    return out
end
# Otherwise a numeric union holds one dual per element: rebuild each from `val`'s lanes.
function _fwd_dual_over_stored(::Any, ::Any, ::Val{N}, stored, val) where {N}
    out = zero_dual(Val(N), stored)
    lanes = ntuple(k -> tangent(val, k), Val(N))
    for i in eachindex(stored)
        s = stored[i]
        out[i] = tangent(_lift_from_lanes(s, map(t -> convert(typeof(s), t[i]), lanes)))
    end
    return out
end
@inline _fwd_dual_over_stored(::Val, stored, _) = throw(
    ArgumentError(
        "forward mode cannot store into a dictionary with value type $(typeof(stored)) when the " *
        "conversion allocates: the dual cannot be rebuilt over the stored object, so a later " *
        "mutation through the dict would be lost. Convert the value before storing it.",
    ),
)

@is_primitive MinimalCtx Tuple{typeof(setindex!),KeyedDict,Any,Any}
function frule!!(
    store::Lifted{<:Union{typeof(setindex!),typeof(_replace_dict_value!)},N},
    d::Lifted{P,N,DV},
    val::Lifted,
    key::Lifted,
) where {N,K,V,P<:KeyedDict{K,V},DV}
    primal(store)(primal(d), primal(val), primal(key))
    # The write above stored `convert(V, val)`, so the dual slot is `dual_type(Val(N), V)` (= `Vdv`),
    # not the dual type of `val`'s own type.
    Vdv = dual_type(Val(N), V)
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
            _fwd_dual_over_stored(Val(N), stored, val)
        end
    end
    primal(store)(tangent(d), dslot, primal(key))
    return d
end
# An absent entry has no saved value. A concrete holder keeps the pullback's
# captured state type-stable without using a tangent wrapper for primal data.
struct SavedDictValue{V,T}
    value::V
    tangent::T
    SavedDictValue{V,T}() where {V,T} = new{V,T}()
    SavedDictValue{V,T}(v, t) where {V,T} = new{V,T}(v, t)
end

function rrule!!(
    store::CoDual{<:Union{typeof(setindex!),typeof(_replace_dict_value!)}},
    d::CoDual{<:KeyedDict{K,V}},
    val,
    key,
) where {K,V}
    k = primal(key)
    restore_state = in(k, keys(primal(d)))
    stored_key = primal(d) isa Dict ? getkey(primal(d), k, k) : k
    Saved = SavedDictValue{V,tangent_type(V)}
    old = if restore_state
        Saved(primal(d)[k], tangent(d)[k])
    else
        Saved()
    end

    primal(store)(primal(d), primal(val), k)
    # The write above stored `convert(V, val)`, so the slot is `tangent_type(V)`. The
    # two-arg `zero_tangent` reuses `val`'s fdata to preserve aliasing for mutable values.
    dslot = if tangent_type(V) == NoTangent
        NoTangent()
    elseif tangent_type(typeof(primal(val))) == NoTangent
        zero_tangent(primal(d)[k])
    else
        zero_tangent(primal(val), tangent(val))
    end
    primal(store)(tangent(d), dslot, k)

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
            primal(d)[stored_key] = old.value
            tangent(d)[stored_key] = old.tangent
        else
            delete!(primal(d), k)
            delete!(tangent(d), k)
        end

        return NoRData(), NoRData(), _dval, instantiate(dkey)
    end
    return d, setindex_pb!!
end

# A key added to the global after build has no tangent entry, so the read rebuilds and stores
# one. A `Dict` is probed by index: `get` with a `nothing` default boxed its V on Julia 1.11.
@inline function _read_dict_entry(::Val{N}, y, d::Lifted{<:Dict}, key) where {N}
    td = tangent(d)
    i = Base.ht_keyindex(td, key)
    i > 0 || return _rebuild_lifted!(Val(N), d, key)::lifted_type(Val(N), typeof(y))
    return _read_lifted(Val(N), y, (@inbounds td.vals[i]), d, key)
end
@inline function _read_dict_entry(::Val{N}, y, d::Lifted{<:IdDict}, key) where {N}
    return _read_lifted(Val(N), y, get(tangent(d), key, nothing), d, key)
end

@is_primitive MinimalCtx Tuple{typeof(get),KeyedDict,Any,Any}
function frule!!(
    ::Lifted{typeof(get),N}, d::Lifted{<:KeyedDict{K,V},N}, key::Lifted, default::Lifted
) where {N,K,V}
    _key = primal(key)
    # An absent key returns default unchanged, even when its type differs from V.
    haskey(primal(d), _key) || return default
    # Use the stored type: V may be Any, which would break downstream slot dispatch.
    y = primal(d)[_key]
    return _read_dict_entry(Val(N), y, d, _key)
end
function rrule!!(
    ::CoDual{typeof(get)}, d::CoDual{<:KeyedDict{K,V}}, key::CoDual, default::CoDual
) where {K,V}
    k = primal(key)
    has_key = in(k, keys(primal(d)))
    y = has_key ? CoDual(primal(d)[k], fdata(tangent(d)[k])) : default

    dd = tangent(d)
    dkey = lazy_zero_rdata(primal(key))
    rdefault = lazy_zero_rdata(primal(default))
    function get_pb!!(dy)
        if has_key
            _replace_dict_value!(dd, increment_rdata!!(dd[k], dy), k)
            _rdefault = instantiate(rdefault)
        else
            # Key absent: `y === default`, so `dy` is exactly `default`'s cotangent.
            _rdefault = dy
        end
        return NoRData(), NoRData(), instantiate(dkey), _rdefault
    end
    return y, get_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(getindex),KeyedDict,Any}
function frule!!(
    ::Lifted{typeof(getindex),N}, d::Lifted{<:KeyedDict{K,V},N}, key::Lifted
) where {N,K,V}
    y = getindex(primal(d), primal(key))
    return _read_dict_entry(Val(N), y, d, primal(key))
end
function rrule!!(
    ::CoDual{typeof(getindex)}, d::CoDual{<:KeyedDict{K,V}}, key::CoDual
) where {K,V}
    k = primal(key)
    y = CoDual(getindex(primal(d), k), fdata(getindex(tangent(d), k)))
    dkey = lazy_zero_rdata(primal(key))
    dd = tangent(d)
    function getindex_pb!!(dy)
        _replace_dict_value!(dd, increment_rdata!!(dd[k], dy), k)
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

for D in (Dict, IdDict)
    @eval begin
        @is_primitive MinimalCtx Tuple{Type{$D{K,V}} where {K,V}}
        function frule!!(::Lifted{Type{$D{K,V}},N}) where {N,K,V}
            return Lifted{$D{K,V},N}($D{K,V}(), $D{K,dual_type(Val(N), V)}())
        end
        function rrule!!(f::CoDual{Type{$D{K,V}}}) where {K,V}
            return CoDual($D{K,V}(), $D{K,tangent_type(V)}()), NoPullback(f)
        end
    end
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:iddict})
    _keyed_dict_rule_cases(rng_ctor, IdDict)
end

function _keyed_dict_rule_cases(rng_ctor, D)
    test_cases = Any[
        TestCase(Base.rehash!, D(true => 5.0, false => 4.0), 10; perf_flag=:stability),
        TestCase(setindex!, D(true => 5.0, false => 4.0), 3.0, false),
        TestCase(setindex!, D(true => 5.0), 3.0, false),
        # type-mismatched stores (typeof(val) ≠ V): non-diff/diff slots, mutable & abstract
        # values (aliasing), and a floating-point width change (fpext/fptrunc).
        TestCase(setindex!, D(:a => 1.0), 2, :b),
        # interface_only: the finite-difference perturbation would break `convert(Int, ...)`.
        TestCase(setindex!, D(:a => 1), 3.0, :b; interface_only=true),
        TestCase(setindex!, D{Symbol,Vector{Float64}}(:a => [1.0, 2.0]), [3.0, 4.0], :b),
        TestCase(setindex!, D{Symbol,Any}(:a => 1.0), 2.0, :b),
        TestCase(setindex!, D{Symbol,Any}(:a => [1.0]), [2.0, 3.0], :b),
        TestCase(setindex!, D{Symbol,Float64}(:a => 1.0), 2.0f0, :b),
        # the same width change for an array, where the gradient rides fdata, not rdata
        TestCase(
            setindex!, D{Symbol,Vector{Float64}}(:a => [1.0, 2.0]), Float32[3.0, 4.0], :b
        ),
        # a numeric-union eltype has structural storage
        TestCase(
            setindex!,
            D{Symbol,Vector{Union{Float32,Float64}}}(:a => [1.0, 2.0f0]),
            [3.0, 4.0],
            :b,
        ),
        TestCase(get, D(true => 5.0, false => 4.0), false, 2.0),
        TestCase(get, D(true => 5.0), false, 2.0),
        # Absent key with a default whose type differs from the dict value type V:
        # the frule must return the `default` slot, not force `Lifted{V}` (regression).
        TestCase(get, D(true => 5.0), false, 2.0f0),
        # `get` returning a default (absent key) whose rdata type differs from its tangent type.
        TestCase(get, D{Symbol,Vector{Float64}}(:a => [1.0]), :b, [2.0, 3.0]),
        # interface_only: the non-differentiable default carries no derivative.
        TestCase(get, D{Symbol,Float64}(:a => 1.0), :b, 2; interface_only=true),
        TestCase(getindex, D(true => 5.0, false => 4.0), true),
        # Abstract V must still return a concretely typed slot.
        TestCase(getindex, D{Symbol,Any}(:a => 2.0), :a),
        TestCase(get, D{Symbol,Any}(:a => 2.0), :a, 0.0),
        # Mutable structs exercise recursive lane materialisation, unlike leaf values.
        TestCase(
            getindex,
            D{Symbol,TestResources.TypeStableMutableStruct{Float64}}(
                :a => TestResources.TypeStableMutableStruct{Float64}(5.0, 4.0)
            ),
            :a,
        ),
        TestCase(D{Any,Any}),
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
    function union_store_then_mutate(x::Vector{Float64})
        d = IdDict{Int,Vector{Union{Float32,Float64}}}()
        d[1] = x
        d[1][1] *= 3
        return sum(d[1])
    end
    test_cases = Any[
        TestCase(converting_store_then_mutate, Float32[3.0, 4.0]),
        TestCase(union_store_then_mutate, [3.0, 4.0]),
    ]
    return test_cases, Any[]
end
