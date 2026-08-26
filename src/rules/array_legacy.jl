# Julia 1.10 has no `Memory`, so two `Array`s over one buffer (`a` and `reshape(a)`) are distinct
# objects with nothing in common to key a cache on — the 1.11+ path keys on the backing `Memory`
# object. `Base.dataids` is what Base's own aliasing machinery uses for exactly this question, and
# the pair with `length` distinguishes different extents over one start. Valid as a cache key only
# WITHIN one call, which is all these caches live for: every array walked is rooted by the caller
# and so cannot be freed and its address reused mid-walk.
@inline _legacy_storage(x::Array) = (Base.dataids(x), length(x))
# A cache hit may have been stored under a different shape over the same buffer, so it is reshaped
# back — sharing data, not copying. The ELTYPE is asserted at retrieval and the dimensionality comes
# from the caller's own `N`, which keeps the result concrete: asserting only the reshaped result
# leaves the cached value `Any` and every use of it dispatches at runtime.
@inline function _legacy_reshape(cached, ::Type{T}, sz::NTuple{N,Int}) where {T,N}
    return reshape(cached::Vector{T}, sz)::Array{T,N}
end
# Cached as a `vec`, which shares data, so a hit knows both the eltype and the dimensionality of
# what it is reshaping. Storing the caller's own shape instead leaves the source `ndims` unknown and
# `reshape` dispatches on `size` at runtime.
@inline _legacy_cached(x::Array) = vec(x)

@inline function zero_tangent_internal(x::Array{P,N}, dict::MaybeCache) where {P,N}
    k = _legacy_storage(x)
    haskey(dict, k) && return _legacy_reshape(dict[k], tangent_type(P), size(x))

    zt = Array{tangent_type(P),N}(undef, size(x)...)
    dict[k] = _legacy_cached(zt)
    return _map_if_assigned!(
        Base.Fix2(zero_tangent_internal, dict), zt, x
    )::Array{tangent_type(P),N}
end

function randn_tangent_internal(
    rng::AbstractRNG, x::Array{T,N}, dict::MaybeCache
) where {T,N}
    k = _legacy_storage(x)
    haskey(dict, k) && return _legacy_reshape(dict[k], tangent_type(T), size(x))

    dx = Array{tangent_type(T),N}(undef, size(x)...)
    dict[k] = _legacy_cached(dx)
    return _map_if_assigned!(x -> randn_tangent_internal(rng, x, dict), dx, x)
end

function increment_internal!!(c::IncCache, x::T, y::T) where {P,N,T<:Array{P,N}}
    k = _legacy_storage(x)
    (haskey(c, k) || x === y) && return x
    c[k] = true
    return _map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y)
end

function set_to_zero_internal!!(c::SetToZeroCache, x::Array)
    _already_tracked!(c, x) && return x
    return _map_if_assigned!(Base.Fix1(set_to_zero_internal!!, c), x, x)
end

function _scale_internal(c::MaybeCache, a::Float64, t::Array{T,N}) where {T,N}
    k = _legacy_storage(t)
    haskey(c, k) && return _legacy_reshape(c[k], T, size(t))
    t′ = Array{T,N}(undef, size(t)...)
    c[k] = _legacy_cached(t′)
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Array}
    key = (_legacy_storage(t), _legacy_storage(s))
    haskey(c, key) && return c[key]::Float64
    c[key] = 0.0
    bitstype = Val(isbitstype(eltype(T)))
    return sum(eachindex(t, s); init=0.0) do i
        if bitstype isa Val{true} || (isassigned(t, i) && isassigned(s, i))
            _dot_internal(c, t[i], s[i])::Float64
        else
            0.0
        end
    end
end

function _add_to_primal_internal(
    c::MaybeCache, x::Array{P,N}, t::Array{<:Any,N}, unsafe::Bool
) where {P,N}
    key = (_legacy_storage(x), _legacy_storage(t), unsafe)
    haskey(c, key) && return _legacy_reshape(c[key], P, size(x))
    x′ = Array{P,N}(undef, size(x)...)
    c[key] = _legacy_cached(x′)
    return _map_if_assigned!((x, t) -> _add_to_primal_internal(c, x, t, unsafe), x′, x, t)
end

function tangent_to_primal_internal!!(
    x::Array{P,N}, t::Array{<:Any,N}, c::MaybeCache
) where {P,N}
    haskey(c, x) && return c[x]::Array{P,N}
    c[x] = x
    return _map_if_assigned!(x, x, t) do xn, tn
        return tangent_to_primal_internal!!(xn, tn, c)
    end
end
function primal_to_tangent_internal!!(
    t::Array{<:Any,N}, x::Array{P,N}, c::MaybeCache
) where {P,N}
    haskey(c, x) && return c[x]::Array{tangent_type(P),N}
    c[x] = t
    return _map_if_assigned!(t, t, x) do txn, xn
        return primal_to_tangent_internal!!(txn, xn, c)
    end
end

@zero_derivative MinimalCtx Tuple{Type{<:Array{T,N}},typeof(undef),Vararg} where {T,N}
@zero_derivative MinimalCtx Tuple{Type{<:Array{T,N}},typeof(undef),Tuple{}} where {T,N}
@zero_derivative MinimalCtx Tuple{Type{<:Array{T,N}},typeof(undef),NTuple{N}} where {T,N}

# `Base.dataids` is broadcasting's aliasing token, and on Julia 1.10 it is the array's raw address
# (`(UInt(pointer(A)),)`) — which the `jl_array_ptr` frule cannot serve above chunk width 1, since a
# lane of the element-major block is stride-`N`. The address is only compared, never dereferenced,
# and a `Tuple{UInt}` carries no derivative, so intercept here rather than descending to the
# pointer. Both modes: a `dx .+= ...` inside a reverse pullback only reaches the pointer under
# chunked forward-over-reverse, but the descent happens while the REVERSE rule is built, so a
# forward-only rule comes too late. Julia 1.11+ keys `dataids` on the backing `Memory`'s
# `objectid` and never takes this path.
@zero_derivative MinimalCtx Tuple{typeof(Base.dataids),Array}

@is_primitive MinimalCtx Tuple{typeof(Base._deletebeg!),Vector,Integer}
# Mutate the user's Vector and the partials block in sync. The block is element-major, so `d`
# primal elements from the front are the leading `N * d` block entries. `T<:NDualEltype` with the
# 4-param V prefix so complex `NDualArray`s (`Complex{NDual}` inner) match too; the body is
# element-type-agnostic. The plain-`Array`-V overload below covers non-`NDualArray`
# element-wise Vs.
function frule!!(
    ::Lifted{typeof(Base._deletebeg!),N},
    a::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    d::Lifted,
) where {N,T<:NDualEltype}
    d_p = primal(d)
    Base._deletebeg!(primal(a), d_p)
    Nfwd._resize_block!(Base._deletebeg!, getfield(tangent(a), :partials_block), N, d_p)
    return zero_lifted(Val(N), nothing)
end
# Plain-`Array` V: delete primal and the element-wise tangent `Array` in lockstep. Covers both
# differentiable non-float elements and non-differentiable element vectors (`Array{NoDual}` V, e.g.
# `Vector{Int}`) — `Array{NoDual} <: Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Base._deletebeg!),N}, a::Lifted{<:Vector,N,<:Array}, d::Lifted
) where {N}
    d_p = primal(d)
    Base._deletebeg!(primal(a), d_p)
    Base._deletebeg!(tangent(a), d_p)
    return zero_lifted(Val(N), nothing)
end
function rrule!!(
    ::CoDual{typeof(Base._deletebeg!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer}
)
    delta = primal(_delta)
    a = primal(_a)
    da = tangent(_a)

    a_beg = a[1:delta]
    da_beg = da[1:delta]

    Base._deletebeg!(a, delta)
    Base._deletebeg!(da, delta)

    function _deletebeg!_pb!!(::NoRData)
        splice!(a, 1:0, a_beg)
        splice!(da, 1:0, da_beg)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _deletebeg!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deleteend!),Vector,Integer}
function frule!!(
    ::Lifted{typeof(Base._deleteend!),N},
    a::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    d::Lifted,
) where {N,T<:NDualEltype}
    d_p = primal(d)
    Base._deleteend!(primal(a), d_p)
    Nfwd._resize_block!(Base._deleteend!, getfield(tangent(a), :partials_block), N, d_p)
    return zero_lifted(Val(N), nothing)
end
# Plain-`Array` V: an `Array` of per-element Vs, deleted in lockstep. Covers both differentiable
# non-float elements and non-differentiable element vectors (`Array{NoDual}` V, e.g. `Vector{Int}`
# reached via `filter`) — `Array{NoDual} <: Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Base._deleteend!),N}, a::Lifted{<:Vector,N,<:Array}, d::Lifted
) where {N}
    d_p = primal(d)
    Base._deleteend!(primal(a), d_p)
    Base._deleteend!(tangent(a), d_p)
    return zero_lifted(Val(N), nothing)
end
function rrule!!(
    ::CoDual{typeof(Base._deleteend!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer}
)
    # Extract data.
    a = primal(_a)
    da = tangent(_a)
    delta = primal(_delta)

    # Store the section to be cut for later.
    primal_tail = a[(end - delta + 1):end]
    tangent_tail = da[(end - delta + 1):end]

    # Cut the end off the primal and tangent.
    Base._deleteend!(a, delta)
    Base._deleteend!(da, delta)

    function _deleteend!_pb!!(::NoRData)
        Base._growend!(a, delta)
        a[(end - delta + 1):end] .= primal_tail

        Base._growend!(da, delta)
        da[(end - delta + 1):end] .= tangent_tail

        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _deleteend!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._deleteat!),Vector,Integer,Integer}
function frule!!(
    ::Lifted{typeof(Base._deleteat!),N},
    a::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    i::Lifted,
    delta::Lifted,
) where {N,T<:NDualEltype}
    i_p = primal(i)
    d_p = primal(delta)
    Base._deleteat!(primal(a), i_p, d_p)
    # Element `i_p` starts at block entry `(i_p - 1) * N + 1`, and `d_p` elements span `d_p * N`.
    Base._deleteat!(
        getfield(getfield(tangent(a), :partials_block), :parent), (i_p - 1) * N + 1, d_p * N
    )
    return zero_lifted(Val(N), nothing)
end
# Plain-`Array` V: an `Array` of per-element Vs, deleted in lockstep. Covers both differentiable
# non-float elements and non-differentiable element vectors (`Array{NoDual}` V) — `Array{NoDual} <:
# Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Base._deleteat!),N},
    a::Lifted{<:Vector,N,<:Array},
    i::Lifted,
    delta::Lifted,
) where {N}
    i_p = primal(i)
    d_p = primal(delta)
    Base._deleteat!(primal(a), i_p, d_p)
    Base._deleteat!(tangent(a), i_p, d_p)
    return zero_lifted(Val(N), nothing)
end
function rrule!!(
    ::CoDual{typeof(Base._deleteat!)},
    _a::CoDual{<:Vector},
    _i::CoDual{<:Integer},
    _delta::CoDual{<:Integer},
)
    # Extract data.
    a, i, delta = map(primal, (_a, _i, _delta))
    da = tangent(_a)

    # Store the cut section for later.
    primal_mem = a[i:(i + delta - 1)]
    tangent_mem = da[i:(i + delta - 1)]

    # Run the primal.
    Base._deleteat!(a, i, delta)
    Base._deleteat!(da, i, delta)

    function _deleteat!_pb!!(::NoRData)
        splice!(a, i:(i - 1), primal_mem)
        splice!(da, i:(i - 1), tangent_mem)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(nothing), _deleteat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growbeg!),Vector,Integer}
function frule!!(
    ::Lifted{typeof(Base._growbeg!),N},
    a::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    d::Lifted,
) where {N,T<:NDualEltype}
    d_p = primal(d)
    Base._growbeg!(primal(a), d_p)
    Nfwd._resize_block!(Base._growbeg!, getfield(tangent(a), :partials_block), N, d_p)
    return zero_lifted(Val(N), nothing)
end
# Plain-`Array` V: an `Array` of per-element Vs, grown in lockstep. Covers both differentiable
# non-float elements and non-differentiable element vectors (`Array{NoDual}` V) — `Array{NoDual} <:
# Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Base._growbeg!),N}, a::Lifted{<:Vector,N,<:Array}, d::Lifted
) where {N}
    d_p = primal(d)
    Base._growbeg!(primal(a), d_p)
    Base._growbeg!(tangent(a), d_p)
    return zero_lifted(Val(N), nothing)
end
function rrule!!(
    ::CoDual{typeof(Base._growbeg!)}, _a::CoDual{<:Vector{T}}, _delta::CoDual{<:Integer}
) where {T}
    d = primal(_delta)
    a = primal(_a)
    da = tangent(_a)
    Base._growbeg!(a, d)
    Base._growbeg!(da, d)
    function _growbeg!_pb!!(::NoRData)
        Base._deletebeg!(a, d)
        Base._deletebeg!(da, d)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _growbeg!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growend!),Vector,Integer}
function frule!!(
    ::Lifted{typeof(Base._growend!),N},
    a::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    d::Lifted,
) where {N,T<:NDualEltype}
    d_p = primal(d)
    Base._growend!(primal(a), d_p)
    Nfwd._resize_block!(Base._growend!, getfield(tangent(a), :partials_block), N, d_p)
    return zero_lifted(Val(N), nothing)
end
# Plain-`Array` V: an `Array` of per-element Vs, grown in lockstep. Covers vectors of differentiable
# non-float elements (e.g. the `Vector{Tuple{pullback}}` grown by reverse rules under
# forward-over-reverse on Julia 1.10) AND non-differentiable element vectors (`Array{NoDual}` V) —
# `Array{NoDual} <: Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Base._growend!),N}, a::Lifted{<:Vector,N,<:Array}, d::Lifted
) where {N}
    d_p = primal(d)
    Base._growend!(primal(a), d_p)
    Base._growend!(tangent(a), d_p)
    return zero_lifted(Val(N), nothing)
end
function rrule!!(
    ::CoDual{typeof(Base._growend!)}, _a::CoDual{<:Vector}, _delta::CoDual{<:Integer}
)
    d = primal(_delta)
    a = primal(_a)
    da = tangent(_a)
    Base._growend!(a, d)
    Base._growend!(da, d)
    function _growend!_pullback!!(::NoRData)
        Base._deleteend!(a, d)
        Base._deleteend!(da, d)
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _growend!_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(Base._growat!),Vector,Integer,Integer}
function frule!!(
    ::Lifted{typeof(Base._growat!),N},
    a::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    i::Lifted,
    d::Lifted,
) where {N,T<:NDualEltype}
    i_p = primal(i)
    d_p = primal(d)
    Base._growat!(primal(a), i_p, d_p)
    Base._growat!(
        getfield(getfield(tangent(a), :partials_block), :parent), (i_p - 1) * N + 1, d_p * N
    )
    return zero_lifted(Val(N), nothing)
end
# Plain-`Array` V: an `Array` of per-element Vs, grown in lockstep. Covers both differentiable
# non-float elements and non-differentiable element vectors (`Array{NoDual}` V) — `Array{NoDual} <:
# Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Base._growat!),N}, a::Lifted{<:Vector,N,<:Array}, i::Lifted, d::Lifted
) where {N}
    i_p = primal(i)
    d_p = primal(d)
    Base._growat!(primal(a), i_p, d_p)
    Base._growat!(tangent(a), i_p, d_p)
    return zero_lifted(Val(N), nothing)
end
function rrule!!(
    ::CoDual{typeof(Base._growat!)},
    _a::CoDual{<:Vector},
    _i::CoDual{<:Integer},
    _delta::CoDual{<:Integer},
)
    # Extract data.
    a, i, delta = map(primal, (_a, _i, _delta))
    da = tangent(_a)

    # Run the primal.
    Base._growat!(a, i, delta)
    Base._growat!(da, i, delta)

    function _growat!_pb!!(::NoRData)
        deleteat!(a, i:(i + delta - 1))
        deleteat!(da, i:(i + delta - 1))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(nothing), _growat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(sizehint!),Vector,Integer}
function frule!!(
    ::Lifted{typeof(sizehint!),N},
    x::Lifted{Vector{T},N,<:NDualArray{T,N,1,Vector{T}}},
    sz::Lifted,
) where {N,T<:NDualEltype}
    sz_p = primal(sz)
    sizehint!(primal(x), sz_p)
    sizehint!(getfield(getfield(tangent(x), :partials_block), :parent), N * sz_p)
    return x
end
# Plain-`Array` V: an `Array` of per-element Vs, hinted in lockstep. Covers both differentiable
# non-float elements and non-differentiable element vectors (`Array{NoDual}` V) — `Array{NoDual} <:
# Array`, so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(sizehint!),N}, x::Lifted{<:Vector,N,<:Array}, sz::Lifted
) where {N}
    sz_p = primal(sz)
    sizehint!(primal(x), sz_p)
    sizehint!(tangent(x), sz_p)
    return x
end
function rrule!!(f::CoDual{typeof(sizehint!)}, x::CoDual{<:Vector}, sz::CoDual{<:Integer})
    sizehint!(primal(x), primal(sz))
    sizehint!(tangent(x), primal(sz))
    return x, NoPullback(f, x, sz)
end

# `Lifted{Ptr{T},N}(y, dy_partials)` is the canonical Ptr V — `dy_partials` is the
# `NTuple{N,Ptr{T}}` that `dual_type(Val(N), Ptr{T})` expects. In the element-major block a lane
# is stride-`N`, so only width 1 has a dense per-lane buffer a raw pointer could address; wider
# chunks fail loudly rather than hand out a pointer that silently reads lane 1 only. Real and
# complex element types share this body.
function frule!!(
    ::Lifted{typeof(_foreigncall_),N},
    ::Lifted{Val{:jl_array_ptr},N},
    ::Lifted{Val{Ptr{T}},N},
    ::Lifted{Tuple{Val{Any}},N},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    a::Lifted{Array{T,D},N,<:NDualArray{T,N,D,Array{T,D}}},
) where {N,T<:Union{IEEEFloat,Complex{<:IEEEFloat}},D}
    N == 1 || throw(
        ArgumentError(
            "Forward-mode raw pointer (`jl_array_ptr`) of a lifted `Array{$T}` is unsupported " *
            "at chunk width $N > 1: the element-major partials block stores each lane with " *
            "stride $N, so there is no dense per-lane buffer a raw pointer could address. " *
            "Differentiate at chunk width 1.",
        ),
    )
    y = ccall(:jl_array_ptr, Ptr{T}, (Any,), primal(a))
    block_parent = getfield(getfield(tangent(a), :partials_block), :parent)
    return Lifted{Ptr{T},N}(y, (ccall(:jl_array_ptr, Ptr{T}, (Any,), block_parent),))
end
# Element-wise V (abstract/non-float-element array, e.g. `Matrix{Real}` reached via
# `Base._unsafe_copyto!`): the tangent is a single element-wise `Array` `tangent(a)`,
# not the `NDualArray`'s parallel per-lane partials. Its pointer is the single element-wise
# partial pointer; the canonical V for `Ptr{P}` is `NTuple{1, Ptr{E}}` (E = element-wise
# dual element), coherent with `dual_type(Val(1), Ptr{P})`. Width-1 only: width-N over an
# abstract element type would need N distinct pointers from one interleaved element-wise array.
function frule!!(
    ::Lifted{typeof(_foreigncall_),1},
    ::Lifted{Val{:jl_array_ptr},1},
    ::Lifted{Val{Ptr{P}},1},
    ::Lifted{Tuple{Val{Any}},1},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    a::Lifted{<:Array,1,<:Array{E}},
) where {P,E}
    y = ccall(:jl_array_ptr, Ptr{P}, (Any,), primal(a))
    dy = ccall(:jl_array_ptr, Ptr{E}, (Any,), tangent(a))
    return Lifted{Ptr{P},1}(y, (dy,))
end
# An all-`NoDual` element-wise V carries no derivative, so neither does its pointer: `NoDual` IS the
# canonical V here, since a non-differentiable element type makes `dual_type(Val(1), Ptr{P})` `NoDual`.
# The method above would instead hand out `pointer(::Array{NoDual})` — a real address into a buffer of
# zero-size elements — which contradicts the slot's own declared type and which a re-typing `bitcast`
# then reads as `Float64` partials, giving a derivative that varies with unrelated heap contents.
# Mirrors the 1.11+ `_get_lifted_field(::MemoryRef, :ptr_or_offset)` exclusion for a `NoDual` element.
function frule!!(
    ::Lifted{typeof(_foreigncall_),1},
    ::Lifted{Val{:jl_array_ptr},1},
    ::Lifted{Val{Ptr{P}},1},
    ::Lifted{Tuple{Val{Any}},1},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    a::Lifted{<:Array,1,<:Array{NoDual}},
) where {P}
    return Lifted{Ptr{P},1}(ccall(:jl_array_ptr, Ptr{P}, (Any,), primal(a)), NoDual())
end
function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
    ::CoDual{Val{:jl_array_ptr}},
    ::CoDual{Val{Ptr{T}}},
    ::CoDual{Tuple{Val{Any}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    a::CoDual{<:Array{T},<:Array{V}},
) where {T,V}
    y = CoDual(
        ccall(:jl_array_ptr, Ptr{T}, (Any,), primal(a)),
        ccall(:jl_array_ptr, Ptr{V}, (Any,), tangent(a)),
    )
    return y, NoPullback(ntuple(_ -> NoRData(), 7))
end

@is_primitive MinimalCtx Tuple{
    typeof(unsafe_copyto!),Array{T},Any,Array{T},Any,Any
} where {T}
# `unsafe_copyto!` copies `n` elements linearly, so dest and src dimensionalities need not
# match (e.g. copying a 0-dim `Array{T,0}` into a `Vector{T}`); bind them separately.
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),N},
    dest::Lifted{Array{T,Dd},N,<:NDualArray{T,N,Dd,Array{T,Dd}}},
    doffs::Lifted,
    src::Lifted{Array{T,Ds},N,<:NDualArray{T,N,Ds,Array{T,Ds}}},
    soffs::Lifted,
    n::Lifted,
) where {N,T<:NDualEltype,Dd,Ds}
    _n = primal(n)
    _doffs = primal(doffs)
    _soffs = primal(soffs)
    Base.unsafe_copyto!(primal(dest), _doffs, primal(src), _soffs, _n)
    # Element-major: the `_n` elements' lanes are one contiguous run of `_n * N` block entries,
    # so all lanes move in a single copy.
    Base.unsafe_copyto!(
        getfield(getfield(tangent(dest), :partials_block), :parent),
        (_doffs - 1) * N + 1,
        getfield(getfield(tangent(src), :partials_block), :parent),
        (_soffs - 1) * N + 1,
        _n * N,
    )
    return dest
end
# Element-wise V: copy primals and the parallel per-element-V arrays (e.g. `Vector{Any}`
# pullback buffers in the public forward-over-reverse HVP interface on 1.10).
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),N},
    dest::Lifted{<:Array,N,<:Array},
    doffs::Lifted,
    src::Lifted{<:Array,N,<:Array},
    soffs::Lifted,
    n::Lifted,
) where {N}
    _n = primal(n)
    _doffs = primal(doffs)
    _soffs = primal(soffs)
    Base.unsafe_copyto!(primal(dest), _doffs, primal(src), _soffs, _n)
    Base.unsafe_copyto!(tangent(dest), _doffs, tangent(src), _soffs, _n)
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{<:Array{T}},
    doffs::CoDual,
    src::CoDual{<:Array{T}},
    soffs::CoDual,
    n::CoDual,
) where {T}
    _n = primal(n)

    # Record values that will be overwritten.
    _doffs = primal(doffs)
    dest_idx = _doffs:(_doffs + _n - 1)
    _soffs = primal(soffs)
    pdest = primal(dest)
    ddest = tangent(dest)
    dest_copy = pdest[dest_idx]
    ddest_copy = ddest[dest_idx]

    # Run primal computation.
    dsrc = tangent(src)
    unsafe_copyto!(primal(dest), _doffs, primal(src), _soffs, _n)
    unsafe_copyto!(tangent(dest), _doffs, dsrc, _soffs, _n)

    function unsafe_copyto_pb!!(::NoRData)

        # Increment dsrc.
        src_idx = _soffs:(_soffs + _n - 1)
        @inbounds for (s, d) in zip(src_idx, dest_idx)
            if isassigned(dsrc, s)
                dsrc[s] = increment!!(dsrc[s], ddest[d])
            end
        end

        # Restore initial state.
        @inbounds for n in eachindex(dest_copy)
            isassigned(dest_copy, n) || continue
            pdest[dest_idx[n]] = dest_copy[n]
            ddest[dest_idx[n]] = ddest_copy[n]
        end

        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end

    return dest, unsafe_copyto_pb!!
end

# Primitive forward rule for `complex(::Array{<:IEEEFloat})`. Its derived copy chain
# (`Vector{Complex}` constructor → `_copyto_impl!` → `_unsafe_copyto!`) reaches the `jl_array_ptr`
# foreigncall and element-wise `Core.arrayset`, which are uncovered for complex `NDualArray` on
# legacy-array Julia (1.11-rc4+ lowers array copies through the covered `MemoryRef` path); a
# primitive short-circuits the chain. ForwardMode-scoped so reverse mode keeps its derived rule;
# the JVP is exact (complex(x) = x + 0im, so d(complex(x)) = complex(dx)). No version guard — this
# file is only included on `VERSION < 1.11-rc4` (see `Mooncake.jl`).
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(complex),Array{P,D}
} where {P<:IEEEFloat,D}
function frule!!(
    ::Lifted{typeof(complex),N}, x::Lifted{Array{P,D},N,<:NDualArray}
) where {N,P<:IEEEFloat,D}
    y = complex(primal(x))  # y = x + 0im
    # d(complex(x)) = complex(dx): complexify the block, which keeps its element-major layout.
    blk = getfield(tangent(x), :partials_block)
    new_block = NDualBlock(complex.(getfield(blk, :parent)), getfield(blk, :dims))
    return Lifted{typeof(y),N}(y, NDualArray{Complex{P},N,D,typeof(y)}(y, new_block))
end

Base.@propagate_inbounds function frule!!(
    ::Lifted{typeof(Core.arrayref),Nw},
    inbounds::Lifted{Bool,Nw},
    x::Lifted{Array{T,D},Nw,<:NDualArray{T,Nw,D,Array{T,D}}},
    inds::Vararg{Lifted{Int,Nw},M},
) where {Nw,T<:NDualEltype,D,M}
    _inds = tuple_map(primal, inds)
    _inb = primal(inbounds)
    y = arrayref(_inb, primal(x), _inds...)
    # Element `_inds`'s `Nw` lanes are the contiguous block column at linear offset `off`.
    v = tangent(x)
    blk = getfield(v, :partials_block)
    off = Nfwd._lane_offset(v, _inds...)
    dy_partials = ntuple(k -> @inbounds(blk[off + k]), Val(Nw))
    return Lifted{T,Nw}(y, _scalar_ndual(y, dy_partials))
end
# Element-wise V: read the element primal and its per-element V from the parallel arrays.
# Covers non-differentiable element vectors too (`Array{NoDual} <: Array`; the read yields
# the element's `NoDual`), so no separate `NoDual` method is needed.
Base.@propagate_inbounds function frule!!(
    ::Lifted{typeof(Core.arrayref),Nw},
    inbounds::Lifted{Bool,Nw},
    x::Lifted{<:Array,Nw,<:Array},
    inds::Vararg{Lifted{Int,Nw},M},
) where {Nw,M}
    _inds = tuple_map(primal, inds)
    _inb = primal(inbounds)
    y = arrayref(_inb, primal(x), _inds...)
    return Lifted{typeof(y),Nw}(y, arrayref(_inb, tangent(x), _inds...))
end
Base.@propagate_inbounds function rrule!!(
    ::CoDual{typeof(Core.arrayref)},
    checkbounds::CoDual{Bool},
    x::CoDual{<:Array},
    inds::Vararg{CoDual{Int},N},
) where {N}

    # Convert to linear indices to reduce amount of data required on the reverse-pass, to
    # avoid converting from cartesian to linear indices multiple times, and to perform a
    # bounds check if required by the calling context.
    lin_inds = LinearIndices(size(primal(x)))[tuple_map(primal, inds)...]

    dx = tangent(x)
    function arrayref_pullback!!(dy)
        new_tangent = increment_rdata!!(arrayref(false, dx, lin_inds), dy)
        arrayset(false, dx, new_tangent, lin_inds)
        return NoRData(), NoRData(), NoRData(), ntuple(_ -> NoRData(), N)...
    end
    _y = arrayref(false, primal(x), lin_inds)
    dy = fdata(arrayref(false, tangent(x), lin_inds))
    return CoDual(_y, dy), arrayref_pullback!!
end

function frule!!(
    ::Lifted{typeof(Core.arrayset),Nw},
    inbounds::Lifted{Bool,Nw},
    A::Lifted{Array{T,D},Nw,<:NDualArray{T,Nw,D,Array{T,D}}},
    v::Lifted{T,Nw},
    inds::Vararg{Lifted{Int,Nw},M},
) where {Nw,T<:NDualEltype,D,M}
    _inds = tuple_map(primal, inds)
    _inb = primal(inbounds)
    Core.arrayset(_inb, primal(A), primal(v), _inds...)
    dA = tangent(A)
    blk = getfield(dA, :partials_block)
    off = Nfwd._lane_offset(dA, _inds...)
    @inbounds for lane in 1:Nw
        blk[off + lane] = tangent(v, lane)
    end
    return A
end
# Element-wise V: set the element primal and its per-element V into the parallel arrays.
# Covers non-differentiable element vectors too (`Array{NoDual} <: Array`; writing the
# scalar's `NoDual` into the V is a typed no-op), so no separate `NoDual` method is needed.
function frule!!(
    ::Lifted{typeof(Core.arrayset),Nw},
    inbounds::Lifted{Bool,Nw},
    A::Lifted{<:Array,Nw,<:Array},
    v::Lifted,
    inds::Vararg{Lifted{Int,Nw},M},
) where {Nw,M}
    _inds = tuple_map(primal, inds)
    _inb = primal(inbounds)
    Core.arrayset(_inb, primal(A), primal(v), _inds...)
    Core.arrayset(_inb, tangent(A), tangent(v), _inds...)
    return A
end
function rrule!!(
    ::CoDual{typeof(Core.arrayset)},
    inbounds::CoDual{Bool},
    A::CoDual{<:Array{P},TdA},
    v::CoDual,
    inds::CoDual{Int}...,
) where {P,V,TdA<:Array{V}}
    _inbounds = primal(inbounds)
    _inds = map(primal, inds)

    if isbitstype(P)
        return isbits_arrayset_rrule(_inbounds, _inds, A, v)
    end

    to_save = isassigned(primal(A), _inds...)
    old_A = Ref{Tuple{P,V}}()
    if to_save
        old_A[] = (
            arrayref(_inbounds, primal(A), _inds...),
            arrayref(_inbounds, tangent(A), _inds...),
        )
    end

    arrayset(_inbounds, primal(A), primal(v), _inds...)
    dA = tangent(A)
    arrayset(_inbounds, dA, tangent(tangent(v), zero_rdata(primal(v))), _inds...)
    function arrayset_pullback!!(::NoRData)
        dv = rdata(arrayref(_inbounds, dA, _inds...))
        if to_save
            arrayset(_inbounds, primal(A), old_A[][1], _inds...)
            arrayset(_inbounds, dA, old_A[][2], _inds...)
        end
        return NoRData(), NoRData(), NoRData(), dv, tuple_map(_ -> NoRData(), _inds)...
    end
    return A, arrayset_pullback!!
end

function isbits_arrayset_rrule(
    boundscheck, _inds, A::CoDual{<:Array{P},TdA}, v::CoDual{P}
) where {P,V,TdA<:Array{V}}

    # Convert to linear indices
    lin_inds = LinearIndices(size(primal(A)))[_inds...]

    old_A = (arrayref(false, primal(A), lin_inds), arrayref(false, tangent(A), lin_inds))
    arrayset(false, primal(A), primal(v), lin_inds)

    _A = primal(A)
    dA = tangent(A)
    arrayset(false, dA, zero_tangent(primal(v), tangent(v)), lin_inds)
    ninds = Val(length(_inds))
    function isbits_arrayset_pullback!!(::NoRData)
        dv = rdata(arrayref(false, dA, lin_inds))
        arrayset(false, _A, old_A[1], lin_inds)
        arrayset(false, dA, old_A[2], lin_inds)
        return NoRData(), NoRData(), NoRData(), dv, tuple_fill(NoRData(), ninds)...
    end
    return A, isbits_arrayset_pullback!!
end

function frule!!(::Lifted{typeof(Core.arraysize),N}, X::Lifted, dim::Lifted) where {N}
    return zero_lifted(Val(N), Core.arraysize(primal(X), primal(dim)))
end
function rrule!!(f::CoDual{typeof(Core.arraysize)}, X, dim)
    return zero_fcodual(Core.arraysize(primal(X), primal(dim))), NoPullback(f, X, dim)
end

@is_primitive MinimalCtx Tuple{typeof(copy),Array}
# `T<:NDualEltype` (not just `IEEEFloat`) with the 4-param V prefix so complex `NDualArray`s
# (`Wrapped === Complex{NDual}`) match too — the `rrule!!` already handles complex.
function frule!!(
    ::Lifted{typeof(copy),N}, a::Lifted{Array{T,D},N,<:NDualArray{T,N,D,Array{T,D}}}
) where {N,T<:NDualEltype,D}
    new_primal = copy(primal(a))
    new_block = copy(getfield(tangent(a), :partials_block))
    return Lifted{Array{T,D},N}(
        new_primal, NDualArray{T,N,D,Array{T,D}}(new_primal, new_block)
    )
end
# Element-wise V (non-differentiable / element-wise array, e.g. a `Vector{UInt8}` → `Vector{NoDual}`
# reached via `copy(::Set)`/`copy(::Dict)` internals): copy the primal and the element-wise V
# array. Mirrors the 1.11+ `Memory` path's general overload and the `rrule!!`'s `<:Array`
# breadth; the more-specific `NDualArray` overload above wins for float parallel-arrays.
@inline function frule!!(::Lifted{typeof(copy),N}, a::Lifted{<:Array,N,<:Array}) where {N}
    return Lifted{typeof(primal(a)),N}(copy(primal(a)), copy(tangent(a)))
end
function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Array})
    dx = tangent(a)
    dy = copy(dx)
    y = CoDual(copy(primal(a)), dy)
    function copy_pullback!!(::NoRData)
        increment!!(dx, dy)
        return NoRData(), NoRData()
    end
    return y, copy_pullback!!
end

function _copy_dict_tangent(mt::MutableTangent)
    t = mt.fields
    new_fields = typeof(t)((
        copy(t.slots), copy(t.keys), copy(t.vals), tuple_fill(NoTangent(), Val(5))...
    ))
    return MutableTangent(new_fields)
end

@is_primitive MinimalCtx Tuple{typeof(fill!),Array{<:Union{UInt8,Int8}},Integer}
# UInt8/Int8 element arrays are non-differentiable — no per-lane tangent
# update needed; mutate the primal and return the slot unchanged.
function frule!!(
    ::Lifted{typeof(fill!),N}, a::Lifted{<:Array{<:Union{UInt8,Int8}},N}, x::Lifted
) where {N}
    fill!(primal(a), primal(x))
    return a
end
function rrule!!(
    ::CoDual{typeof(fill!)}, a::CoDual{T}, x::CoDual{<:Integer}
) where {V<:Union{UInt8,Int8},T<:Array{V}}
    pa = primal(a)
    old_value = copy(pa)
    fill!(pa, primal(x))
    function fill!_pullback!!(::NoRData)
        pa .= old_value
        return NoRData(), NoRData(), NoRData()
    end
    return a, fill!_pullback!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:array_legacy})
    _x = Ref(5.0)
    _dx = randn_tangent(Xoshiro(123456), _x)

    _a, _da = randn(5), randn(5)
    _b, _db = randn(4), randn(4)

    test_cases = Any[

        # Old foreigncall wrappers.
        (true, :stability, nothing, Array{Float64,0}, undef),
        (true, :stability, nothing, Array{Float64,1}, undef, 5),
        (true, :stability, nothing, Array{Float64,2}, undef, 5, 4),
        (true, :stability, nothing, Array{Float64,3}, undef, 5, 4, 3),
        (true, :stability, nothing, Array{Float64,4}, undef, 5, 4, 3, 2),
        (true, :stability, nothing, Array{Float64,5}, undef, 5, 4, 3, 2, 1),
        (true, :stability, nothing, Array{Float64,0}, undef, ()),
        (true, :stability, nothing, Array{Float64,4}, undef, (2, 3, 4, 5)),
        (true, :stability, nothing, Array{Float64,5}, undef, (2, 3, 4, 5, 6)),
        (false, :stability, nothing, copy, randn(5, 4)),
        (false, :stability, nothing, copy, randn(Xoshiro(123456), ComplexF64, 5)),
        (false, :stability, nothing, Base._deletebeg!, randn(5), 0),
        (false, :stability, nothing, Base._deletebeg!, randn(5), 2),
        (false, :stability, nothing, Base._deletebeg!, randn(5), 5),
        (false, :stability, nothing, Base._deleteend!, randn(5), 2),
        (false, :stability, nothing, Base._deleteend!, randn(5), 5),
        (false, :stability, nothing, Base._deleteend!, randn(5), 0),
        (false, :stability, nothing, Base._deleteat!, randn(5), 2, 2),
        (false, :stability, nothing, Base._deleteat!, randn(5), 1, 5),
        (false, :stability, nothing, Base._deleteat!, randn(5), 5, 1),
        (false, :stability, nothing, fill!, rand(Int8, 5), Int8(2)),
        (false, :stability, nothing, fill!, rand(UInt8, 5), UInt8(2)),
        (true, :stability, nothing, Base._growbeg!, randn(5), 3),
        (true, :stability, nothing, Base._growend!, randn(5), 3),
        (true, :stability, nothing, Base._growat!, randn(5), 2, 2),
        (false, :stability, nothing, sizehint!, randn(5), 10),
        # Complex vectors (`NDualArray` V `Complex{NDual}` inner) exercise the broad `@is_primitive`
        # against a matching frule; a `T<:IEEEFloat`-only frule would leave it uncovered -> MethodError.
        (false, :stability, nothing, Base._deletebeg!, randn(ComplexF64, 5), 2),
        (false, :stability, nothing, Base._deleteend!, randn(ComplexF64, 5), 2),
        (false, :stability, nothing, Base._deleteat!, randn(ComplexF64, 5), 2, 2),
        (true, :stability, nothing, Base._growbeg!, randn(ComplexF64, 5), 3),
        (true, :stability, nothing, Base._growend!, randn(ComplexF64, 5), 3),
        (true, :stability, nothing, Base._growat!, randn(ComplexF64, 5), 2, 2),
        (false, :stability, nothing, sizehint!, randn(ComplexF64, 5), 10),
        (false, :stability, nothing, unsafe_copyto!, randn(4), 2, randn(3), 1, 2),
        # Mismatched dest/src dimensionality (0-dim source into a Vector).
        (false, :none, nothing, unsafe_copyto!, [0.0], 1, fill(2.0), 1, 1),
        (
            false,
            :stability,
            nothing,
            unsafe_copyto!,
            [rand(3) for _ in 1:5],
            2,
            [rand(4) for _ in 1:4],
            1,
            3,
        ),
        (
            false,
            :none,
            nothing,
            unsafe_copyto!,
            Vector{Any}(undef, 5),
            2,
            Any[rand() for _ in 1:4],
            1,
            3,
        ),
        (
            false,
            :none,
            nothing,
            unsafe_copyto!,
            fill!(Vector{Any}(undef, 3), 4.0),
            1,
            Vector{Any}(undef, 2),
            1,
            2,
        ),
        (
            # A lane of the element-major block is stride-`N`, so no dense per-lane buffer exists
            # for a raw pointer to address above width 1; the frule throws there (covered by
            # `throwing_rule_test_cases(::Val{:foreigncall})`). The width-1 path is correct.
            true,
            :none,
            (skip_chunked=true,),
            _foreigncall_,
            Val(:jl_array_ptr),
            Val(Ptr{Float64}),
            (Val(Any),),
            Val(0), # nreq
            Val(:ccall), # calling convention
            randn(5),
        ),

        # Old builtins.
        (false, :stability, nothing, IntrinsicsWrappers.arraylen, randn(10)),
        (false, :stability, nothing, IntrinsicsWrappers.arraylen, randn(10, 7)),
        (false, :stability, nothing, Base.arrayref, true, randn(5), 1),
        (false, :stability, nothing, Base.arrayref, false, randn(4), 1),
        (false, :stability, nothing, Base.arrayref, true, randn(5, 4), 1, 1),
        (false, :stability, nothing, Base.arrayref, false, randn(5, 4), 5, 4),
        (false, :stability, nothing, Base.arrayref, true, randn(5, 4), 1),
        (false, :stability, nothing, Base.arrayref, false, randn(5, 4), 5),
        (false, :stability, nothing, Base.arrayref, false, [1, 2, 3], 1),
        (false, :stability, nothing, Base.arrayset, false, [1, 2, 3], 4, 2),
        (false, :stability, nothing, Base.arrayset, false, randn(5), 4.0, 3),
        (false, :stability, nothing, Base.arrayset, false, randn(5, 4), 3.0, 1, 3),
        (false, :stability, nothing, Base.arrayset, true, randn(5), 4.0, 3),
        (false, :stability, nothing, Base.arrayset, true, randn(5, 4), 3.0, 1, 3),
        (
            false,
            :stability,
            nothing,
            Base.arrayset,
            false,
            [randn(3) for _ in 1:5],
            randn(4),
            1,
        ),
        (
            false,
            :stability,
            nothing,
            Base.arrayset,
            true,
            [(5.0, rand(1))],
            (4.0, rand(1)),
            1,
        ),
        (
            false,
            :stability,
            nothing,
            Base.arrayset,
            false,
            setindex!(Vector{Vector{Float64}}(undef, 3), randn(3), 1),
            randn(4),
            1,
        ),
        (
            false,
            :stability,
            nothing,
            Base.arrayset,
            false,
            setindex!(Vector{Vector{Float64}}(undef, 3), randn(3), 2),
            randn(4),
            1,
        ),
        (false, :stability, nothing, Core.arraysize, randn(5, 4, 3), 2),
        (false, :stability, nothing, Core.arraysize, randn(5, 4, 3, 2, 1), 100),
    ]
    memory = Any[_x, _dx, _a, _da, _b, _db]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:array_legacy})
    test_cases = Any[(
        false,
        :none,
        nothing,
        Base._unsafe_copyto!,
        fill!(Matrix{Real}(undef, 5, 4), 1.0),
        3,
        randn(10),
        2,
        4,
    ),]
    return test_cases, Any[]
end
