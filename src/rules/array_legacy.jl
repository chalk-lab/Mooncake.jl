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
# Implementation kernels for `_deletebeg!`. Lifted body below dispatches the
# inner V into either the legacy `Dual{<:Vector}` path or the canonical
# `AbstractVector{<:NDual}` path (NDual elements pack primal+tangent so a
# single `_deletebeg!` mutates both at once).
@inline function _deletebeg_kernel!(a::Dual{<:Vector}, d::Dual{<:Integer})
    Base._deletebeg!(primal(a), primal(d))
    Base._deletebeg!(tangent(a), primal(d))
    return zero_dual(nothing)
end
@inline function _deletebeg_kernel!(a::AbstractVector{<:NDual}, d::Dual{<:Integer})
    Base._deletebeg!(a, primal(d))
    return zero_dual(nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deletebeg!),N},
    a::Mooncake.Lifted{<:Vector},
    d::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _deletebeg_kernel!(Mooncake._unlift(a), Mooncake._unlift(d))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline function _deleteend_kernel!(a::Dual{<:Vector}, d::Dual{<:Integer})
    Base._deleteend!(primal(a), primal(d))
    Base._deleteend!(tangent(a), primal(d))
    return zero_dual(nothing)
end
@inline function _deleteend_kernel!(a::AbstractVector{<:NDual}, d::Dual{<:Integer})
    Base._deleteend!(a, primal(d))
    return zero_dual(nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deleteend!),N},
    a::Mooncake.Lifted{<:Vector},
    d::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _deleteend_kernel!(Mooncake._unlift(a), Mooncake._unlift(d))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline function _deleteat_kernel!(
    a::Dual{<:Vector}, i::Dual{<:Integer}, delta::Dual{<:Integer}
)
    Base._deleteat!(primal(a), primal(i), primal(delta))
    Base._deleteat!(tangent(a), primal(i), primal(delta))
    return zero_dual(nothing)
end
@inline function _deleteat_kernel!(
    a::AbstractVector{<:NDual}, i::Dual{<:Integer}, delta::Dual{<:Integer}
)
    Base._deleteat!(a, primal(i), primal(delta))
    return zero_dual(nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._deleteat!),N},
    a::Mooncake.Lifted{<:Vector},
    i::Mooncake.Lifted{<:Integer},
    delta::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _deleteat_kernel!(
        Mooncake._unlift(a), Mooncake._unlift(i), Mooncake._unlift(delta)
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline function _growbeg_kernel!(a::Dual{<:Vector}, d::Dual{<:Integer})
    Base._growbeg!(primal(a), primal(d))
    Base._growbeg!(tangent(a), primal(d))
    return zero_dual(nothing)
end
@inline function _growbeg_kernel!(a::AbstractVector{<:NDual}, d::Dual{<:Integer})
    Base._growbeg!(a, primal(d))
    return zero_dual(nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growbeg!),N},
    a::Mooncake.Lifted{<:Vector},
    d::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _growbeg_kernel!(Mooncake._unlift(a), Mooncake._unlift(d))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline function _growend_kernel!(a::Dual{<:Vector}, d::Dual{<:Integer})
    Base._growend!(primal(a), primal(d))
    Base._growend!(tangent(a), primal(d))
    return zero_dual(nothing)
end
@inline function _growend_kernel!(a::AbstractVector{<:NDual}, d::Dual{<:Integer})
    Base._growend!(a, primal(d))
    return zero_dual(nothing)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growend!),N},
    a::Mooncake.Lifted{<:Vector},
    d::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _growend_kernel!(Mooncake._unlift(a), Mooncake._unlift(d))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
# `Base._growat!` returns the new length as a `Tuple{Int}` on Julia 1.12+;
# wrap with `zero_dual` so the rule output type matches the primal's. (Earlier
# Julia versions returned `nothing` and the previous `zero_dual(nothing)` was
# correct; `zero_dual(::Tuple{Int})` returns the canonical width-1 form for the
# tuple, so the rule output now matches `rrule_output_type(Tuple{Int})`.)
@inline function _growat_kernel!(a::Dual{<:Vector}, i::Dual{<:Integer}, d::Dual{<:Integer})
    r = Base._growat!(primal(a), primal(i), primal(d))
    Base._growat!(tangent(a), primal(i), primal(d))
    return zero_dual(r)
end
@inline function _growat_kernel!(
    a::AbstractVector{<:NDual}, i::Dual{<:Integer}, d::Dual{<:Integer}
)
    r = Base._growat!(a, primal(i), primal(d))
    return zero_dual(r)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base._growat!),N},
    a::Mooncake.Lifted{<:Vector},
    i::Mooncake.Lifted{<:Integer},
    d::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _growat_kernel!(
        Mooncake._unlift(a), Mooncake._unlift(i), Mooncake._unlift(d)
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline function _sizehint_kernel!(x::Dual{<:Vector}, sz::Dual{<:Integer})
    sizehint!(primal(x), primal(sz))
    sizehint!(tangent(x), primal(sz))
    return x
end
@inline function _sizehint_kernel!(x::AbstractVector{<:NDual}, sz::Dual{<:Integer})
    sizehint!(x, primal(sz))
    return x
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(sizehint!),N},
    x::Mooncake.Lifted{<:Vector},
    sz::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = _sizehint_kernel!(Mooncake._unlift(x), Mooncake._unlift(sz))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
function rrule!!(f::CoDual{typeof(sizehint!)}, x::CoDual{<:Vector}, sz::CoDual{<:Integer})
    sizehint!(primal(x), primal(sz))
    sizehint!(tangent(x), primal(sz))
    return x, NoPullback(f, x, sz)
end

function frule!!(
    ::Dual{typeof(_foreigncall_)},
    ::Dual{Val{:jl_array_ptr}},
    ::Dual{Val{Ptr{T}}},
    ::Dual{Tuple{Val{Any}}},
    ::Dual, # nreq
    ::Dual, # calling convention
    a::Dual{<:Array{T},<:Array{V}},
) where {T,V}
    y = ccall(:jl_array_ptr, Ptr{T}, (Any,), primal(a))
    dy = ccall(:jl_array_ptr, Ptr{V}, (Any,), tangent(a))
    return Dual(y, dy)
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(_foreigncall_),N},
    a1::Mooncake.Lifted{Val{:jl_array_ptr}},
    a2::Mooncake.Lifted{Val{Ptr{T}}},
    a3::Mooncake.Lifted{Tuple{Val{Any}}},
    a4::Mooncake.Lifted,
    a5::Mooncake.Lifted,
    a::Mooncake.Lifted{<:Array{T}},
) where {N,T}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(a1),
        Mooncake._unlift(a2),
        Mooncake._unlift(a3),
        Mooncake._unlift(a4),
        Mooncake._unlift(a5),
        Mooncake._unlift(a),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(_foreigncall_),Val{:jl_array_ptr},Vararg}}
) = true
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
function frule!!(
    ::Dual{typeof(unsafe_copyto!)},
    dest::Dual{<:Array{T}},
    doffs::Dual,
    src::Dual{<:Array{T}},
    soffs::Dual,
    n::Dual,
) where {T}
    _n = primal(n)
    Base.unsafe_copyto!(primal(dest), primal(doffs), primal(src), primal(soffs), _n)
    Base.unsafe_copyto!(tangent(dest), primal(doffs), tangent(src), primal(soffs), _n)
    return dest
end
# Bare NDual-array overload: V at width 1 for Array{<:IEEEFloat} is
# Array{NDual{T,1}}; the NDual elements pack primal+tangent so a single
# `unsafe_copyto!` mutates both at once.
@inline function frule!!(
    ::Dual{typeof(unsafe_copyto!)},
    dest::AbstractArray{<:NDual},
    doffs::Dual,
    src::AbstractArray{<:NDual},
    soffs::Dual,
    n::Dual,
)
    Base.unsafe_copyto!(dest, primal(doffs), src, primal(soffs), primal(n))
    return dest
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(unsafe_copyto!),N},
    dest::Mooncake.Lifted{<:Array{T}},
    doffs::Mooncake.Lifted,
    src::Mooncake.Lifted{<:Array{T}},
    soffs::Mooncake.Lifted,
    n::Mooncake.Lifted,
) where {N,T}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(dest),
        Mooncake._unlift(doffs),
        Mooncake._unlift(src),
        Mooncake._unlift(soffs),
        Mooncake._unlift(n),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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

Base.@propagate_inbounds function frule!!(
    ::Dual{typeof(Core.arrayref)},
    inbounds::Dual{Bool},
    x::Dual{<:Array},
    inds::Vararg{Dual{Int},N},
) where {N}
    _inds = tuple_map(primal, inds)
    y = arrayref(primal(inbounds), primal(x), _inds...)
    dy = arrayref(primal(inbounds), tangent(x), _inds...)
    return Dual(y, dy)
end
# NDual (chunked forward) path: tangent info lives inside elements, not in a Dual wrapper.
Base.@propagate_inbounds function frule!!(
    ::Dual{typeof(Core.arrayref)},
    inbounds::Dual{Bool},
    x::Array{<:_HasNDual},
    inds::Vararg{Dual{Int},N},
) where {N}
    return arrayref(primal(inbounds), x, tuple_map(primal, inds)...)
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(Core.arrayref),N},
    inbounds::Mooncake.Lifted{Bool},
    x::Mooncake.Lifted{<:Array},
    inds::Vararg{Mooncake.Lifted{Int},M},
) where {N,M}
    bare_inds = ntuple(i -> Mooncake._unlift(inds[i]), Val(M))
    bare_result = frule!!(
        Mooncake._unlift(f), Mooncake._unlift(inbounds), Mooncake._unlift(x), bare_inds...
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
    ::Dual{typeof(Core.arrayset)},
    inbounds::Dual{Bool},
    A::Dual{<:Array},
    v::Dual,
    inds::Dual{Int}...,
)
    _inds = tuple_map(primal, inds)
    Core.arrayset(primal(inbounds), primal(A), primal(v), _inds...)
    Core.arrayset(primal(inbounds), tangent(A), tangent(v), _inds...)
    return A
end
# NDual (chunked forward) path: tangent info lives inside elements.
function frule!!(
    ::Dual{typeof(Core.arrayset)},
    inbounds::Dual{Bool},
    A::Array{<:_HasNDual},
    v,
    inds::Dual{Int}...,
)
    Core.arrayset(primal(inbounds), A, v, tuple_map(primal, inds)...)
    return A
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(Core.arrayset),N},
    inbounds::Mooncake.Lifted{Bool},
    A::Mooncake.Lifted{<:Array},
    v::Mooncake.Lifted,
    inds::Vararg{Mooncake.Lifted{Int},M},
) where {N,M}
    bare_inds = ntuple(i -> Mooncake._unlift(inds[i]), Val(M))
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(inbounds),
        Mooncake._unlift(A),
        Mooncake._unlift(v),
        bare_inds...,
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline _copy_array_legacy_kernel(a::Dual{<:Array}) = Dual(
    copy(primal(a)), copy(tangent(a))
)
@inline _copy_array_legacy_kernel(a::AbstractArray{<:NDual}) = copy(a)
@inline function frule!!(
    ::Mooncake.Lifted{typeof(copy),N}, a::Mooncake.Lifted{<:Array}
) where {N}
    bare_result = _copy_array_legacy_kernel(Mooncake._unlift(a))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(copy),<:Array}}) = true
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
function frule!!(
    ::Dual{typeof(fill!)}, a::Dual{<:Array{<:Union{UInt8,Int8}}}, x::Dual{<:Integer}
)
    fill!(primal(a), primal(x))
    return a
end
@inline function frule!!(
    f::Mooncake.Lifted{typeof(fill!),N},
    a::Mooncake.Lifted{<:Array{<:Union{UInt8,Int8}}},
    x::Mooncake.Lifted{<:Integer},
) where {N}
    bare_result = frule!!(Mooncake._unlift(f), Mooncake._unlift(a), Mooncake._unlift(x))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
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
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base._deletebeg!),<:Vector,<:Integer}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base._deleteend!),<:Vector,<:Integer}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base._deleteat!),<:Vector,<:Integer,<:Integer}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base._growbeg!),<:Vector,<:Integer}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base._growend!),<:Vector,<:Integer}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Base._growat!),<:Vector,<:Integer,<:Integer}}
) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(sizehint!),<:Vector,<:Integer}}) =
    true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(unsafe_copyto!),<:Array,Any,<:Array,Any,Any}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Core.arrayref),Bool,<:Array,Vararg{<:Integer}}}
) = true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(Core.arrayset),Bool,<:Array,Any,Vararg{<:Integer}}}
) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(Core.arraysize),<:Array,Any}}) =
    true
@inline Mooncake._is_lifted_aware(
    ::Type{<:Tuple{typeof(fill!),<:Array{<:Union{UInt8,Int8}},<:Integer}}
) = true

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
