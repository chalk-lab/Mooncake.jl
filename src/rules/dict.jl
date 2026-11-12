# Dict's hash table is private to its primal. No AD route may read or mutate those fields.
function _dict_field_error()
    throw(
        ArgumentError(
            "Dict uses a keyed tangent; differentiation through Dict fields or unsupported " *
            "hash-table internals is not supported. This operation needs a keyed Dict rule or overlay.",
        ),
    )
end
for (f, arity) in ((getfield, 1), (getfield, 2), (setfield!, 2), (setfield!, 3))
    args = [Symbol(:arg, i) for i in 1:arity]
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),Dict,Vararg{Any,$arity}}
        function frule!!(
            ::Lifted{typeof($f),N},
            ::Lifted{<:Dict,N,<:Dict},
            $(map(a -> :($a::Lifted), args)...),
        ) where {N}
            return _dict_field_error()
        end
        function rrule!!(
            ::CoDual{typeof($f)},
            ::CoDual{<:Dict,<:Dict},
            $(map(a -> :($a::CoDual), args)...),
        )
            return _dict_field_error()
        end
    end
end
for order in ((), (:order,))
    @eval begin
        @is_primitive MinimalCtx Tuple{
            typeof(lgetfield),Dict,Val,$(map(_ -> :Val, order)...)
        }
        function frule!!(
            ::Lifted{typeof(lgetfield),N},
            ::Lifted{<:Dict,N,<:Dict},
            ::Lifted{Val{f}},
            $(map(o -> :(::Lifted{Val{$o}}), order)...),
        ) where {N,f,$(order...)}
            return _dict_field_error()
        end
        function rrule!!(
            ::CoDual{typeof(lgetfield)},
            ::CoDual{<:Dict,<:Dict},
            ::CoDual{Val{f}},
            $(map(o -> :(::CoDual{Val{$o}}), order)...),
        ) where {f,$(order...)}
            return _dict_field_error()
        end
    end
end
@is_primitive MinimalCtx Tuple{typeof(lsetfield!),Dict,Val,Any}
function frule!!(
    ::Lifted{typeof(lsetfield!),N}, ::Lifted{<:Dict,N,<:Dict}, ::Lifted{Val{f}}, ::Lifted
) where {N,f}
    _dict_field_error()
end
function rrule!!(
    ::CoDual{typeof(lsetfield!)}, ::CoDual{<:Dict,<:Dict}, ::CoDual{Val{f}}, ::CoDual
) where {f}
    _dict_field_error()
end

@zero_derivative MinimalCtx Tuple{typeof(haskey),Dict,Any}
@zero_derivative MinimalCtx Tuple{typeof(length),Dict}
@zero_derivative MinimalCtx Tuple{typeof(isempty),Dict}

# Table rearrangement changes neither the key set nor any value's tangent. Return the same
# slot, preserving aliasing; capacity/order need not be restored by the pullback.
for f in (sizehint!, Base.rehash!)
    @eval begin
        @is_primitive MinimalCtx Tuple{typeof($f),Dict,Integer}
        function frule!!(::Lifted{typeof($f),N}, d::Lifted{<:Dict,N}, n::Lifted) where {N}
            $f(primal(d), primal(n))
            return d
        end
        function rrule!!(::CoDual{typeof($f)}, d::CoDual{<:Dict}, n::CoDual)
            $f(primal(d), primal(n))
            return d, NoPullback((NoRData(), NoRData(), NoRData()))
        end
    end
end

@static if VERSION >= v"1.11"
    @is_primitive MinimalCtx Tuple{
        typeof(Core.kwcall),
        NamedTuple{(:shrink,),Tuple{Bool}},
        typeof(sizehint!),
        Dict,
        Integer,
    }
    function frule!!(
        ::Lifted{typeof(Core.kwcall),N},
        kw::Lifted{NamedTuple{(:shrink,),Tuple{Bool}}},
        ::Lifted{typeof(sizehint!)},
        d::Lifted{<:Dict,N},
        n::Lifted,
    ) where {N}
        sizehint!(primal(d), primal(n); primal(kw)...)
        return d
    end
    function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kw::CoDual{NamedTuple{(:shrink,),Tuple{Bool}}},
        ::CoDual{typeof(sizehint!)},
        d::CoDual{<:Dict},
        n::CoDual,
    )
        sizehint!(primal(d), primal(n); primal(kw)...)
        return d, NoPullback(ntuple(_ -> NoRData(), 5))
    end
end

@is_primitive MinimalCtx Tuple{typeof(delete!),Dict,Any}
function frule!!(::Lifted{typeof(delete!),N}, d::Lifted{<:Dict,N}, k::Lifted) where {N}
    delete!(primal(d), primal(k))
    delete!(tangent(d), primal(k))
    return d
end
function rrule!!(::CoDual{typeof(delete!)}, d::CoDual{Dict{K,V}}, k::CoDual) where {K,V}
    key = primal(k)
    present = haskey(primal(d), key)
    stored_key = getkey(primal(d), key, key)
    Saved = SavedDictValue{V,tangent_type(V)}
    old = if present
        Saved(primal(d)[key], tangent(d)[key])
    else
        Saved()
    end
    if present
        delete!(primal(d), key)
        delete!(tangent(d), key)
    end
    dk = lazy_zero_rdata(key)
    function delete_pb!!(::NoRData)
        if present
            primal(d)[stored_key] = old.value
            tangent(d)[stored_key] = old.tangent
        end
        return NoRData(), NoRData(), instantiate(dk)
    end
    return d, delete_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(empty!),Dict}
function frule!!(::Lifted{typeof(empty!),N}, d::Lifted{<:Dict,N}) where {N}
    empty!(primal(d))
    empty!(tangent(d))
    return d
end
function rrule!!(::CoDual{typeof(empty!)}, d::CoDual{<:Dict})
    p, t = copy(primal(d)), copy(tangent(d))
    empty!(primal(d))
    empty!(tangent(d))
    function empty_pb!!(::NoRData)
        merge!(primal(d), p)
        merge!(tangent(d), t)
        return NoRData(), NoRData()
    end
    return d, empty_pb!!
end

# Traverse the PRIMAL table to choose a key; derivatives are always fetched by that key.
# The cursor is Base's cursor, including its end sentinel; the tangent's table may differ.
# Inspect tangent types, not values: derivative-free mutable keys are safe to expose.
function _check_dict_key(k)
    !_holds_derivatives(tangent_type(typeof(k))) || throw(
        ArgumentError(
            "Dict and Set tangents cannot expose differentiable keys; use non-differentiable keys.",
        ),
    )
    return k
end
function _dict_next_value(d::Dict, state)
    item = iterate(d, state)
    item === nothing && return nothing
    return item[1].first, item[2]
end
function _dict_next(d::Dict, state)
    item = _dict_next_value(d, state)
    item === nothing && return nothing
    return _check_dict_key(item[1]), item[2]
end
@zero_derivative MinimalCtx Tuple{typeof(_dict_next_value),Dict,Any}
@zero_derivative MinimalCtx Tuple{typeof(_dict_next),Dict,Any}
function _dict_getkey(d::Dict, k, default)
    return _check_dict_key(getkey(d, k, default))
end
@zero_derivative MinimalCtx Tuple{typeof(_dict_getkey),Dict,Any,Any}
@mooncake_overlay Base.getkey(d::Dict, k, default) = _dict_getkey(d, k, default)
@mooncake_overlay function Base.iterate(d::Dict{K,V}, state=1) where {K,V}
    item = _dict_next(d, state)
    item === nothing && return nothing
    k, next = item
    return Pair{K,V}(k, d[k]), next
end
@mooncake_overlay function Base.iterate(v::Base.KeySet{K,D}, state=1) where {K,D<:Dict}
    return _dict_next(v.dict, state)
end
@mooncake_overlay function Base.iterate(v::Base.ValueIterator{D}, state=1) where {D<:Dict}
    item = _dict_next_value(v.dict, state)
    item === nothing && return nothing
    k, next = item
    return v.dict[k], next
end

# Express compound operations in terms of the primitive boundaries above. These overlays
# are differentiated in both modes, including the lazy default's closure and conversions.
@mooncake_overlay function Base.get(default::Union{Function,Type}, d::Dict, k)
    return haskey(d, k) ? d[k] : default()
end
@mooncake_overlay function Base.get!(
    default::Union{Function,Type}, d::Dict{K,V}, key
) where {K,V}
    k = convert(K, key)
    if !haskey(d, k)
        d[k] = default()
    end
    return d[k]
end
@mooncake_overlay function Base.pop!(d::Dict, k)
    v = d[k]
    delete!(d, k)
    return v
end
@mooncake_overlay function Base.pop!(d::Dict, k, default)
    haskey(d, k) || return default
    return pop!(d, k)
end
@mooncake_overlay function Base.pop!(d::Dict)
    isempty(d) && throw(ArgumentError("dict must be non-empty"))
    p = first(d)
    delete!(d, p.first)
    return p.first => p.second
end
@static if isdefined(Base, :in!)
    @mooncake_overlay function Base.in!(x, s::Set)
        key = convert(eltype(s), x)
        key in s && return true
        push!(s, key)
        return false
    end
end
@mooncake_overlay function Base.pop!(s::Set, x)
    x in s || throw(KeyError(x))
    k = VERSION < v"1.11" ? x : getkey(s.dict, x, x)
    delete!(s.dict, x)
    return k
end
@mooncake_overlay function Base.pop!(s::Set, x, default)
    return x in s ? pop!(s, x) : default
end
@mooncake_overlay function (::Type{Set{T}})(ks::Base.KeySet{T,D}) where {T,D<:Dict}
    s = Set{T}()
    for k in ks
        push!(s, k)
    end
    return s
end
@mooncake_overlay function Base.filter!(pred, d::Dict)
    for p in d
        pred(p) || delete!(d, p.first)
    end
    return d
end
@mooncake_overlay function Base.map!(f, vs::Base.ValueIterator{D}) where {D<:Dict}
    d = vs.dict
    item = _dict_next_value(d, 1)
    while item !== nothing
        k, state = item
        d[k] = f(d[k])
        item = _dict_next_value(d, state)
    end
    return vs
end
@mooncake_overlay function Base.mergewith!(combine, d::Dict, other::AbstractDict)
    for (k, v) in other
        if haskey(d, k)
            _replace_dict_value!(d, combine(d[k], v), k)
        else
            d[k] = v
        end
    end
    return d
end

# Shallow-copy the container while sharing mutable VALUE derivatives. In particular, do
# not differentiate through key iteration: keys may themselves be opaque mutable objects.
@is_primitive MinimalCtx Tuple{typeof(copy),Dict}
function frule!!(::Lifted{typeof(copy),N}, d::Lifted{P,N}) where {P<:Dict,N}
    return Lifted{P,N}(copy(primal(d)), copy(tangent(d)))
end
function rrule!!(::CoDual{typeof(copy)}, d::CoDual{P}) where {P<:Dict}
    p, dd = primal(d), tangent(d)
    dt = tangent_type(P)()
    for (k, v) in p
        dt[k] = zero_tangent(v, fdata(dd[k]))
    end
    function copy_pb!!(::NoRData)
        for k in keys(dd)
            dd[k] = increment_rdata!!(dd[k], rdata(dt[k]))
        end
        return NoRData(), NoRData()
    end
    return CoDual(copy(p), dt), copy_pb!!
end
@mooncake_overlay (::Type{Dict{K,V}})(d::Dict{K,V}) where {K,V} = copy(d)
@mooncake_overlay (::Type{Dict})(d::Dict) = copy(d)

function hand_written_rule_test_cases(rng_ctor, ::Val{:dict})
    cases, memory = _keyed_dict_rule_cases(rng_ctor, Dict)
    d = Dict(1 => 2.0, 23 => 3.0)
    append!(
        cases,
        Any[
            (false, :none, nothing, haskey, copy(d), 1),
            (false, :none, nothing, length, copy(d)),
            (false, :none, nothing, isempty, copy(d)),
            (false, :none, nothing, sizehint!, copy(d), 100),
            (false, :none, nothing, _replace_dict_value!, copy(d), 4.0, 1),
            (false, :none, nothing, delete!, copy(d), 1),
            (false, :none, nothing, delete!, copy(d), 2),
            (false, :none, nothing, empty!, copy(d)),
            (false, :none, nothing, copy, copy(d)),
            (false, :none, nothing, copy, Dict{Any,Any}("A" => [5.0], [3.0] => 5.0)),
            (
                false,
                :none,
                (throws=(ArgumentError, "differentiable keys"),),
                _dict_next,
                Dict([1.0] => 2.0),
                1,
            ),
            (
                false,
                :none,
                (throws=(ArgumentError, "differentiable keys"),),
                _dict_next,
                Dict(1.0 => 2.0),
                1,
            ),
            (false, :none, nothing, _dict_next, copy(d), 1),
        ],
    )
    if VERSION >= v"1.11"
        push!(
            cases,
            (false, :none, nothing, Core.kwcall, (; shrink=false), sizehint!, copy(d), 100),
        )
    end
    for width in (1, 8),
        (f, args) in (
            (getfield, (copy(d), :vals)),
            (getfield, (copy(d), :vals, :not_atomic)),
            (lgetfield, (copy(d), Val(:vals))),
            (lgetfield, (copy(d), Val(:vals), Val(:not_atomic))),
            (setfield!, (copy(d), :count, length(d))),
            (lsetfield!, (copy(d), Val(:count), length(d))),
            (setfield!, (copy(d), :count, length(d), :not_atomic)),
            (_new_, (typeof(d), map(n -> getfield(d, n), fieldnames(typeof(d)))...)),
        )

        push!(
            cases,
            (
                false,
                :none,
                (throws=(ArgumentError, "Dict uses a keyed tangent"), chunk_size=width),
                f,
                args...,
            ),
        )
    end
    for (f, args) in
        ((Base.ht_keyindex, (copy(d), 1)), (Base.ht_keyindex2_shorthash!, (copy(d), 1)))
        push!(
            cases,
            (
                false,
                :none,
                (throws=(ArgumentError, "Dict uses a keyed tangent"),),
                f,
                args...,
            ),
        )
    end
    return cases, memory
end
function derived_rule_test_cases(rng_ctor, ::Val{:dict})
    d = Dict(1 => 2.0, 23 => 3.0)
    cases = Any[
        (false, :none, nothing, f, copy(d), args...) for (f, args) in (
            (get!, (1, 7.0)),
            (get!, (2, 7.0)),
            (pop!, (1,)),
            (pop!, (2, 7.0)),
            (pop!, ()),
            (iterate, ()),
            (iterate, (1,)),
            (keys, ()),
            (values, ()),
            (pairs, ()),
            (Dict, ()),
            (Dict{Int,Float64}, ()),
            (x -> sum(values(x)), ()),
            (x -> sum(keys(x)), ()),
            (x -> sum(p.second for p in pairs(x)), ()),
            (x -> get(() -> 7.0, x, 2), ()),
            (x -> get!(() -> 7.0, x, 2), ()),
            (x -> Dict(1 => x[1], 23 => x[23]), ()),
            (x -> merge(x, Dict(2 => 7.0)), ()),
            (x -> merge!(x, Dict(2 => 7.0)), ()),
        )
    ]
    append!(
        cases,
        Any[
            (
                false,
                :none,
                nothing,
                x -> x.b[1]^2,
                TestResources.TypeStableMutableStruct{typeof(d)}(1.0, copy(d)),
            ),
            (false, :none, nothing, x -> x[1][1]^2, [copy(d)]),
            (false, :none, nothing, x -> (g=()->x[1]^2; g()), copy(d)),
            (
                false,
                :none,
                nothing,
                x -> (x[1][1] += 1; sum(x[2])),
                let a=[2.0];
                    Dict(1=>a, 2=>a);
                end,
            ),
            (
                false,
                :none,
                nothing,
                x -> (t=Dict{Int,Vector{Float64}}(); t[1]=x; t[1][1]+=1; sum(t[1])),
                Float32[2, 3],
            ),
        ],
    )
    # Derived public routes exercise the overlays against finite differences in both modes.
    x = [1.3, 0.7, 2.1]
    for f in (
        x -> sum(x[unique([1, 2, 2, 3])]),
        x -> sum(unique([x; x[1]])),
        x -> allunique(collect(1:100)) ? x[1]^2 : 0.0,
        x -> allunique(x) ? x[1]^2 : 0.0,
        x -> pop!(Set([1, 2]), 1) * x[1],
        x -> pop!(Set([1, 2]), 1, 0) * x[1],
        x -> pop!(Set([1, 2]), 3, x[1]),
        x -> length(Set(keys(Dict(1 => x[1], 2 => x[2])))) * x[1],
        x -> (s=Set(x); x[2] in s ? x[2]^2 : 0.0),
        x -> (local d = Dict(0.5 => x[1], 1.5 => x[2]); d[0.5] * d[1.5]),
        x -> sum(values(Dict(0.5 => x[1], 1.5 => x[2]))),
        x -> (local d = Dict(1 => x[1], 2 => x[2]); filter!(p -> p.first == 1, d); d[1]^2),
        x -> (local d = Dict(0.5 => x[1]); map!(v -> v^2, values(d)); d[0.5]),
        x -> getkey(Dict(1 => x[1]), 1, 0) * x[1],
        x -> mergewith(+, Dict(1 => x[1]), Dict(1 => x[2], 2 => x[3]))[1]^2,
        x -> merge!(+, Dict(1 => x[1]), Dict(1 => x[2]))[1]^2,
    )
        push!(cases, (false, :none, nothing, f, copy(x)))
    end
    for f in (
            x -> (
                local d=Dict{Any,Float64}(1=>x[1]);
                first(d) isa Pair{Any,Float64} ? x[1] : -x[1]
            ),
            x -> (
                local d=Dict{Int,Real}(1=>x[1]);
                first(d) isa Pair{Int,Real} ? x[1] : -x[1]
            ),
            x -> (
                local d=Dict{Any,Float64}(1=>x[1]);
                pop!(d) isa Pair{Int,Float64} ? x[1] : -x[1]
            ),
            x -> (
                local d=Dict{Any,Float64}(1=>x[1]);
                mergewith!(+, d, Dict{Any,Float64}(0x01=>x[2]));
                first(keys(d)) isa Int ? d[1] : -d[1]
            ),
            x -> (
                local d=Dict{Any,Float64}(1=>x[1]);
                merge!(+, d, Dict{Any,Float64}(0x01=>x[2]));
                first(keys(d)) isa Int ? d[1] : -d[1]
            ),
            x -> (local s=Set{Any}([1]); pop!(s, 0x01) isa UInt8 ? x[1] : -x[1]),
            x -> (local s=Set{Any}([1]); pop!(s, 0x01, 0) isa UInt8 ? x[1] : -x[1]),
            x -> sum(length(k) * v for (k, v) in Dict([1]=>x[1], [2, 3]=>x[2])),
            x -> sum(length, keys(Dict([:a]=>x[1], [:b, :c]=>x[2]))) * x[3],
            x -> sum(x[first(k)] for k in Set([[1], [2, 3]])),
            x -> (
                local d=Dict([1]=>x[1], [2, 3]=>x[2]);
                filter!(p -> length(p.first)==1, d);
                d[[1]]^2
            ),
        ),
        width in (1, 8)

        push!(cases, (false, :none, (; chunk_size=width), f, copy(x)))
    end
    if VERSION < v"1.11"
        for f in (x -> pop!(Set(x), x[1]), x -> pop!(Set(x), x[1], 0.0)), width in (1, 8)
            push!(cases, (false, :none, (; chunk_size=width), f, copy(x)))
        end
    end
    push!(cases, (false, :none, nothing, d -> sum(values(d)), Dict(0.5 => 1.0, 1.5 => 2.0)))
    for f in (
            x -> sum(Set(x)),
            x -> sum(k * v for (k, v) in Dict(x[1] => x[2])),
            x -> getkey(Dict(x[1] => x[2]), x[1], 0.0),
            (
                if VERSION < v"1.11"
                    ()
                else
                    (x -> pop!(Set(x), x[1]), x -> pop!(Set(x), x[1], 0.0))
                end
            )...,
            x -> Set(keys(Dict(x[1] => x[2]))),
        ),
        width in (1, 8)

        push!(
            cases,
            (
                false,
                :none,
                (throws=(ArgumentError, "differentiable keys"), chunk_size=width),
                f,
                copy(x),
            ),
        )
    end
    for f in (
            x -> (local d = Dict(1 => x[1]); replace!(p -> p.first => p.second^2, d); d[1]),
            x -> replace(p -> p.first => p.second^2, Dict(1 => x[1]))[1],
            x -> (s=Set([1]); replace!(i -> i + 1, s); x[first(s)]),
            x -> x[first(replace(i -> i + 1, Set([1])))],
            x -> rand(Random.Xoshiro(1), Dict(1 => x[1])).second,
            x -> x[rand(Random.Xoshiro(1), Set([1]))],
        ),
        width in (1, 8)

        push!(
            cases,
            (
                false,
                :none,
                (throws=(ArgumentError, "Dict uses a keyed tangent"), chunk_size=width),
                f,
                copy(x),
            ),
        )
    end
    sample_key = x -> rand(Random.Xoshiro(1), keys(Dict(1 => x[1]))) * x[1]
    sample_value = x -> rand(Random.Xoshiro(1), values(Dict(1 => x[1])))
    if VERSION < v"1.11"
        # 1.10 collects keys and has no value-iterator sampler; 1.11+ samples Dict slots.
        push!(cases, (false, :none, nothing, sample_key, copy(x)))
        for width in (1, 8)
            push!(
                cases,
                (
                    false,
                    :none,
                    (throws=(MethodError, "Sampler"), primal_throws=true, chunk_size=width),
                    sample_value,
                    copy(x),
                ),
            )
        end
    else
        for f in (sample_key, sample_value), width in (1, 8)
            push!(
                cases,
                (
                    false,
                    :none,
                    (throws=(ArgumentError, "Dict uses a keyed tangent"), chunk_size=width),
                    f,
                    copy(x),
                ),
            )
        end
    end
    return cases, Any[]
end

TestUtils.supports_field_access_interactions(::Type{<:Dict}) = false

@is_primitive MinimalCtx Tuple{typeof(_new_),Type{<:Dict},Vararg}
function frule!!(
    ::Lifted{typeof(_new_),N}, ::Lifted{Type{P},N}, ::Vararg{Lifted}
) where {N,P<:Dict}
    _dict_field_error()
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{P}}, ::Vararg{CoDual}
) where {P<:Dict}
    _dict_field_error()
end
