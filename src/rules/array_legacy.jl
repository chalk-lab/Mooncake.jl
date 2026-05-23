@inline function zero_tangent_internal(x::Array{P,N}, dict::MaybeCache) where {P,N}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))

    zt = Array{tangent_type(P),N}(undef, size(x)...)
    dict[x] = zt
    return _map_if_assigned!(
        Base.Fix2(zero_tangent_internal, dict), zt, x
    )::Array{tangent_type(P),N}
end

function randn_tangent_internal(
    rng::AbstractRNG, x::Array{T,N}, dict::MaybeCache
) where {T,N}
    haskey(dict, x) && return dict[x]::tangent_type(typeof(x))

    dx = Array{tangent_type(T),N}(undef, size(x)...)
    dict[x] = dx
    return _map_if_assigned!(x -> randn_tangent_internal(rng, x, dict), dx, x)
end

function increment_internal!!(c::IncCache, x::T, y::T) where {P,N,T<:Array{P,N}}
    (haskey(c, x) || x === y) && return x
    c[x] = true
    return _map_if_assigned!((x, y) -> increment_internal!!(c, x, y), x, x, y)
end

function set_to_zero_internal!!(c::SetToZeroCache, x::Array)
    _already_tracked!(c, x) && return x
    return _map_if_assigned!(Base.Fix1(set_to_zero_internal!!, c), x, x)
end

function _scale_internal(c::MaybeCache, a::Float64, t::Array{T,N}) where {T,N}
    haskey(c, t) && return c[t]::Array{T,N}
    t′ = Array{T,N}(undef, size(t)...)
    c[t] = t′
    return _map_if_assigned!(t -> _scale_internal(c, a, t), t′, t)
end

function _dot_internal(c::MaybeCache, t::T, s::T) where {T<:Array}
    key = (t, s)
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
    key = (x, t, unsafe)
    haskey(c, key) && return c[key]::Array{P,N}
    x′ = Array{P,N}(undef, size(x)...)
    c[key] = x′
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

@is_primitive MinimalCtx Tuple{typeof(Base._deletebeg!),Vector,Integer}
# Direct Lifted bodies per inner V shape. Wrapper-exception V: delete from
# primal and tangent separately. Canonical NDual V: NDual elements pack
# primal+tangent so a single `_deletebeg!` mutates both at once (uses
# `_unlift(a)` to access the bare `AbstractVector{<:NDual}` storage that
# `primal(a)` wouldn't expose — the canonical accessor returns the bare
# primal Vector).
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deletebeg!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:Dual{<:Vector}}
    pd = primal(d)
    Base._deletebeg!(primal(a), pd)
    Base._deletebeg!(tangent(a), pd)
    return zero_lifted(Val(N), nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deletebeg!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:AbstractVector{<:NDual}}
    Base._deletebeg!(Mooncake._unlift(a), primal(d))
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deleteend!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:Dual{<:Vector}}
    pd = primal(d)
    Base._deleteend!(primal(a), pd)
    Base._deleteend!(tangent(a), pd)
    return zero_lifted(Val(N), nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deleteend!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:AbstractVector{<:NDual}}
    Base._deleteend!(Mooncake._unlift(a), primal(d))
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deleteat!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    i::Mooncake.Lifted{<:Integer},
    delta::Mooncake.Lifted{<:Integer},
) where {N,V_a<:Dual{<:Vector}}
    pi, pd = primal(i), primal(delta)
    Base._deleteat!(primal(a), pi, pd)
    Base._deleteat!(tangent(a), pi, pd)
    return zero_lifted(Val(N), nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deleteat!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    i::Mooncake.Lifted{<:Integer},
    delta::Mooncake.Lifted{<:Integer},
) where {N,V_a<:AbstractVector{<:NDual}}
    Base._deleteat!(Mooncake._unlift(a), primal(i), primal(delta))
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growbeg!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:Dual{<:Vector}}
    pd = primal(d)
    Base._growbeg!(primal(a), pd)
    Base._growbeg!(tangent(a), pd)
    return zero_lifted(Val(N), nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growbeg!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:AbstractVector{<:NDual}}
    Base._growbeg!(Mooncake._unlift(a), primal(d))
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growend!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:Dual{<:Vector}}
    pd = primal(d)
    Base._growend!(primal(a), pd)
    Base._growend!(tangent(a), pd)
    return zero_lifted(Val(N), nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growend!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:AbstractVector{<:NDual}}
    Base._growend!(Mooncake._unlift(a), primal(d))
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
# `Base._growat!` returns `nothing` on Julia 1.10/1.11 and a `Tuple{Int}` on
# 1.12+; `zero_lifted` produces the canonical width-`N` form for either shape.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growat!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    i::Mooncake.Lifted{<:Integer},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:Dual{<:Vector}}
    pi, pd = primal(i), primal(d)
    r = Base._growat!(primal(a), pi, pd)
    Base._growat!(tangent(a), pi, pd)
    return zero_lifted(Val(N), r)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growat!),N},
    a::Mooncake.Lifted{<:Vector,N,V_a},
    i::Mooncake.Lifted{<:Integer},
    d::Mooncake.Lifted{<:Integer},
) where {N,V_a<:AbstractVector{<:NDual}}
    r = Base._growat!(Mooncake._unlift(a), primal(i), primal(d))
    return zero_lifted(Val(N), r)
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

    # Run the primal and capture the actual return value (Julia 1.12+ returns
    # the new-length tuple `(::Int,)` rather than `nothing`).
    r = Base._growat!(a, i, delta)
    Base._growat!(da, i, delta)

    function _growat!_pb!!(::NoRData)
        deleteat!(a, i:(i + delta - 1))
        deleteat!(da, i:(i + delta - 1))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(r), _growat!_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(sizehint!),Vector,Integer}
@inline function frule!!(
    ::Mooncake.Lifted{typeof(sizehint!),N},
    x::Mooncake.Lifted{<:Vector,N,V_x},
    sz::Mooncake.Lifted{<:Integer},
) where {N,V_x<:Dual{<:Vector}}
    psz = primal(sz)
    sizehint!(primal(x), psz)
    sizehint!(tangent(x), psz)
    return x
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(sizehint!),N},
    x::Mooncake.Lifted{<:Vector,N,V_x},
    sz::Mooncake.Lifted{<:Integer},
) where {N,V_x<:AbstractVector{<:NDual}}
    sizehint!(Mooncake._unlift(x), primal(sz))
    return x
end
function rrule!!(f::CoDual{typeof(sizehint!)}, x::CoDual{<:Vector}, sz::CoDual{<:Integer})
    sizehint!(primal(x), primal(sz))
    sizehint!(tangent(x), primal(sz))
    return x, NoPullback(f, x, sz)
end

@inline function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N},
    ::Mooncake.Lifted{Val{:jl_array_ptr}},
    ::Mooncake.Lifted{Val{Ptr{T}}},
    ::Mooncake.Lifted{Tuple{Val{Any}}},
    ::Mooncake.Lifted, # nreq
    ::Mooncake.Lifted, # calling convention
    a::Mooncake.Lifted{<:Array{T},N,V_a},
) where {N,T,V,V_a<:Dual{<:Array{T},<:Array{V}}}
    bare_a = Mooncake._unlift(a)
    y = ccall(:jl_array_ptr, Ptr{T}, (Any,), primal(bare_a))
    dy = ccall(:jl_array_ptr, Ptr{V}, (Any,), tangent(bare_a))
    return Mooncake.Lifted{Ptr{T},N}(y, dy)
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
# Wrapper-exception slot V (Dual{Array, Array}): copy primal and tangent
# separately.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(unsafe_copyto!),N},
    dest::Mooncake.Lifted{<:Array{T},N,V_d},
    doffs::Mooncake.Lifted,
    src::Mooncake.Lifted{<:Array{T},N,V_s},
    soffs::Mooncake.Lifted,
    n::Mooncake.Lifted,
) where {N,T,V_d<:Dual,V_s<:Dual}
    bare_dest = Mooncake._unlift(dest)
    bare_src = Mooncake._unlift(src)
    _n = primal(n)
    _doffs = primal(doffs)
    _soffs = primal(soffs)
    Base.unsafe_copyto!(primal(bare_dest), _doffs, primal(bare_src), _soffs, _n)
    Base.unsafe_copyto!(tangent(bare_dest), _doffs, tangent(bare_src), _soffs, _n)
    return dest
end
# Canonical NDual slots: NDual elements pack primal+tangent so a single
# `unsafe_copyto!` mutates both at once.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(unsafe_copyto!),N},
    dest::Mooncake.Lifted{<:Array{T},N,V_d},
    doffs::Mooncake.Lifted,
    src::Mooncake.Lifted{<:Array{T},N,V_s},
    soffs::Mooncake.Lifted,
    n::Mooncake.Lifted,
) where {N,T,V_d<:AbstractArray{<:NDual},V_s<:AbstractArray{<:NDual}}
    bare_dest = Mooncake._unlift(dest)
    bare_src = Mooncake._unlift(src)
    Base.unsafe_copyto!(bare_dest, primal(doffs), bare_src, primal(soffs), primal(n))
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

# Wrapper-exception slot V (Dual{Array, Array}): separate primal+tangent
# arrayref calls, wrap result via 2-arg Lifted ctor.
Base.@propagate_inbounds @inline function frule!!(
    ::Mooncake.Lifted{typeof(Core.arrayref),N},
    inbounds::Mooncake.Lifted{Bool},
    x::Mooncake.Lifted{<:Array{T},N,V_x},
    inds::Vararg{Mooncake.Lifted{Int},M},
) where {N,T,M,V_x<:Dual}
    bare_x = Mooncake._unlift(x)
    ib = primal(inbounds)
    _inds = ntuple(i -> primal(inds[i]), Val(M))
    y = arrayref(ib, primal(bare_x), _inds...)
    dy = arrayref(ib, tangent(bare_x), _inds...)
    return Mooncake.Lifted{T,N}(y, dy)
end
# Canonical NDual slot V: tangent info lives in elements; one arrayref
# returns the lifted element directly.
Base.@propagate_inbounds @inline function frule!!(
    ::Mooncake.Lifted{typeof(Core.arrayref),N},
    inbounds::Mooncake.Lifted{Bool},
    x::Mooncake.Lifted{<:Array{T},N,V_x},
    inds::Vararg{Mooncake.Lifted{Int},M},
) where {N,T,M,V_x<:Array{<:_HasNDual}}
    bare_x = Mooncake._unlift(x)
    result = arrayref(primal(inbounds), bare_x, ntuple(i -> primal(inds[i]), Val(M))...)
    return Mooncake.Lifted{T,N,typeof(result)}(result)
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

@inline function frule!!(
    ::Mooncake.Lifted{typeof(Core.arrayset),N},
    inbounds::Mooncake.Lifted{Bool},
    A::Mooncake.Lifted{<:Array{T},N,V_A},
    v::Mooncake.Lifted,
    inds::Vararg{Mooncake.Lifted{Int},M},
) where {N,T,M,V_A<:Dual}
    bare_A = Mooncake._unlift(A)
    bare_v = Mooncake._unlift(v)
    ib = primal(inbounds)
    _inds = ntuple(i -> primal(inds[i]), Val(M))
    Core.arrayset(ib, primal(bare_A), primal(bare_v), _inds...)
    Core.arrayset(ib, tangent(bare_A), tangent(bare_v), _inds...)
    return A
end
# Canonical NDual slot V: tangent info lives in elements; one arrayset
# stores the lifted element directly.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Core.arrayset),N},
    inbounds::Mooncake.Lifted{Bool},
    A::Mooncake.Lifted{<:Array{T},N,V_A},
    v::Mooncake.Lifted,
    inds::Vararg{Mooncake.Lifted{Int},M},
) where {N,T,M,V_A<:Array{<:_HasNDual}}
    bare_A = Mooncake._unlift(A)
    bare_v = Mooncake._unlift(v)
    Core.arrayset(primal(inbounds), bare_A, bare_v, ntuple(i -> primal(inds[i]), Val(M))...)
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

# `Core.arraysize`: the Lifted-typed body below computes the result
# independently (no `_unlift` delegation), so no kernel function or
# bare-Dual body is needed.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Core.arraysize),N}, X::Mooncake.Lifted, dim::Mooncake.Lifted
) where {N}
    return zero_lifted(Val(N), Core.arraysize(primal(X), primal(dim)))
end
function rrule!!(f::CoDual{typeof(Core.arraysize)}, X, dim)
    return zero_fcodual(Core.arraysize(primal(X), primal(dim))), NoPullback(f, X, dim)
end

@is_primitive MinimalCtx Tuple{typeof(copy),Array}
# Wrapper-exception V (Dual{Array, NTangent}): per-lane copy.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(copy),N}, a::Mooncake.Lifted{<:Array,N,V_a}
) where {N,V_a<:Dual{<:Array,<:Mooncake.NTangent}}
    y = copy(primal(a))
    dys = ntuple(k -> copy(Mooncake.tangent(a, k)), Val(N))
    return Mooncake.Lifted{_typeof(y),N}(y, Mooncake.NTangent(dys))
end
# Canonical NDual-element Array V: single copy.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(copy),N}, a::Mooncake.Lifted{<:Array,N,V_a}
) where {N,V_a<:AbstractArray{<:NDual}}
    return _wrap_rule_result(Val(N), copy(Mooncake._unlift(a)))
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
# Element type is non-differentiable (NoTangent V); `fill!` mutates the
# primal in place and the same Lifted slot is returned unchanged.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(fill!),N},
    a::Mooncake.Lifted{<:Array{<:Union{UInt8,Int8}}},
    x::Mooncake.Lifted{<:Integer},
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

# Lifted-aware trait registrations for the rules above. Each rule's body
# accepts the unwrapped slot V and the generic `frule!!(::Lifted{F,N},
# args::Vararg{Lifted,M})` adapter handles the wrap/unwrap.

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
        (false, :stability, nothing, unsafe_copyto!, randn(4), 2, randn(3), 1, 2),
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
            true,
            :none,
            nothing,
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
