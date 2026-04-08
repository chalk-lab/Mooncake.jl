#
# Direct nfwd-backed primitive wrapper entrypoints used by `rules_via_nfwd.jl` and a few
# low-level scalar rules. This file is intentionally self-contained: it owns the NDual
# dual-type bridge plus the scalar/tuple primitive lowering that still uses nfwd directly.
#
# These entrypoints are still needed even though forward execution can run on width-aware
# dual types directly. Mooncake's public AD surface remains rule-based, so primitive
# `frule!!` calls still need a thin integration boundary, and reverse-mode primitive
# `rrule!!` calls still need explicit pullback entrypoints.
#

function hand_written_rule_test_cases(rng_ctor, ::Val{:primitive_wrappers})
    # Keep wrapper-specific coverage here so the test file can stay a thin
    # `test_dual` / `test_rule` loop over source-owned cases.
    test_cases = (
        (
            name="real scalar width-aware dual bridge",
            kind=:dual,
            primal=2.0,
            dual_type=Nfwd.NDual{Float64,2},
            width=2,
        ),
        (
            name="complex scalar width-aware dual bridge",
            kind=:dual,
            primal=1.0 + 2.0im,
            dual_type=Complex{Nfwd.NDual{Float64,2}},
            width=2,
        ),
        (
            name="exp primitive rule",
            kind=:rule,
            f=exp,
            args=(1.5,),
            chunk_sizes=(1, 2),
            output_tangent=nothing,
            is_primitive=true,
        ),
        (
            name="sincos primitive rule",
            kind=:rule,
            f=sincos,
            args=(1.5,),
            chunk_sizes=(1, 2),
            output_tangent=(2.0, 3.0),
            is_primitive=true,
        ),
    )
    return test_cases, Any[]
end

function derived_rule_test_cases(rng_ctor, ::Val{:primitive_wrappers})
    # These exercise the Mooncake-facing nfwd wrapper path, not pure `Nfwd`
    # internals. Keep the pure `Nfwd` corpus in `test/nfwd/nfwd.jl`.
    test_cases = (
        (
            name="array sum(abs2) derived rule",
            f=x -> sum(abs2, x),
            args=([1.0, 2.0, 3.0],),
            chunk_sizes=(1,),
            output_tangent=nothing,
        ),
        (
            name="complex scalar derived rule",
            f=z -> real(z * z + cos(z)),
            args=(1.2 - 0.3im,),
            chunk_sizes=(1, 2),
            output_tangent=nothing,
        ),
    )
    return test_cases, Any[]
end

@inline function dual_type(::Val{N}, ::Type{T}) where {N,T<:IEEEFloat}
    return N == 1 ? Dual{T,tangent_type(Val(1), T)} : Nfwd.NDual{T,N}
end

@inline function dual_type(::Val{N}, ::Type{Complex{T}}) where {N,T<:IEEEFloat}
    return if N == 1
        Dual{Complex{T},tangent_type(Val(1), Complex{T})}
    else
        Complex{Nfwd.NDual{T,N}}
    end
end

@inline primal(x::Nfwd.NDual) = Nfwd.ndual_value(x)
@inline tangent(x::Nfwd.NDual{T,N}) where {T,N} = NTangent(Nfwd.ndual_partials(x))

@inline primal(z::Complex{<:Nfwd.NDual}) = Nfwd._nfwd_dual_value(z)
@inline tangent(z::Complex{<:Nfwd.NDual{T,N}}) where {T,N} = NTangent(
    ntuple(i -> Nfwd._nfwd_dual_partial(z, i), Val(N))
)

# NTangent → NDual / Complex{NDual} constructors.  These bridge the public tangent
# representation (NTangent) with the internal NDual representation.  Used by `randn_dual`,
# `_nfwd_extract_primitive_output` (tuple path), and any code that constructs a dual via
# `dual_type(Val(N), T)(primal, ntangent)`.

function (::Type{Nfwd.NDual{T,N}})(
    x::T, dx::NTangent{NTuple{N,T}}
) where {T<:Union{Float16,Float32,Float64},N}
    return invoke(Nfwd.NDual{T,N}, Tuple{T,NTuple{N,T}}, x, ntuple(i -> T(dx[i]), Val(N)))
end

function (::Type{Complex{Nfwd.NDual{T,N}}})(
    z::Complex{T}, dz::NTangent{NTuple{N,Complex{T}}}
) where {T<:Union{Float16,Float32,Float64},N}
    re = Nfwd.NDual{T,N}(real(z), ntuple(i -> T(real(dz[i])), Val(N)))
    im = Nfwd.NDual{T,N}(imag(z), ntuple(i -> T(imag(dz[i])), Val(N)))
    return invoke(Complex{Nfwd.NDual{T,N}}, Tuple{Nfwd.NDual{T,N},Nfwd.NDual{T,N}}, re, im)
end

@inline function _nfwd_check_function_tangent(df)
    df isa Union{NoTangent,NoFData} && return nothing
    df isa NTangent && all(t -> t isa NoTangent, df) && return nothing
    throw(ArgumentError("nfwd does not support differentiating with respect to `f`."))
end

# ── Scalar / complex lifting ────────────────────────────────────────────────────
#
# Convert (primal, tangent) pairs into NDual / Complex{NDual}.
# Width-1 with a plain scalar/complex tangent (interface derivative path):

@inline function _nfwd_lift_primitive_arg(x::T, dx::Real, ::Val{1}) where {T<:IEEEFloat}
    return Nfwd.NDual{T,1}(x, (T(dx),))
end

@inline function _nfwd_lift_primitive_arg(
    x::Complex{T}, dx::Complex, ::Val{1}
) where {T<:IEEEFloat}
    return Complex(
        Nfwd.NDual{T,1}(real(x), (T(real(dx)),)), Nfwd.NDual{T,1}(imag(x), (T(imag(dx)),))
    )
end

# Width-N with an indexable tangent (NTangent from frule path, NTuple from seeding):

@inline function _nfwd_lift_primitive_arg(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    return Nfwd.NDual{T,N}(x, ntuple(i -> T(dx[i]), Val(N)))
end

@inline function _nfwd_lift_primitive_arg(
    x::Complex{T}, dx, ::Val{N}
) where {T<:IEEEFloat,N}
    return Complex(
        Nfwd.NDual{T,N}(real(x), ntuple(i -> T(real(dx[i])), Val(N))),
        Nfwd.NDual{T,N}(imag(x), ntuple(i -> T(imag(dx[i])), Val(N))),
    )
end

# ── Dense array lifting ──────────────────────────────────────────────────────

# Pass-through: Array{NDual} is already lifted (used by seeded tangent path).
@inline _nfwd_lift_primitive_arg(
    ::Array{T}, dx::Array{Nfwd.NDual{T,N}}, ::Val{N}
) where {T<:IEEEFloat,N} = dx
@inline _nfwd_lift_primitive_arg(
    ::Array{Complex{T}}, dx::Array{Complex{Nfwd.NDual{T,N}}}, ::Val{N}
) where {T<:IEEEFloat,N} = dx

# Width-1: plain Array{T} tangent (user-provided direction, frule path).
@inline function _nfwd_lift_primitive_arg(
    x::Array{T}, dx::Array{T}, ::Val{1}
) where {T<:IEEEFloat}
    return map((xi, dxi) -> Nfwd.NDual{T,1}(xi, (dxi,)), x, dx)
end

@inline function _nfwd_lift_primitive_arg(
    x::Array{Complex{T}}, dx::Array{Complex{T}}, ::Val{1}
) where {T<:IEEEFloat}
    return map(
        (xi, dxi) -> Complex(
            Nfwd.NDual{T,1}(real(xi), (T(real(dxi)),)),
            Nfwd.NDual{T,1}(imag(xi), (T(imag(dxi)),)),
        ),
        x,
        dx,
    )
end

@inline function _nfwd_lift_primitive_args(
    ::Val{N}, primals::Tuple, tangents::Tuple
) where {N}
    return map(
        (x, dx) -> if dx isa NoTangent
            x
        else
            _nfwd_lift_primitive_arg(Nfwd._nfwd_check_primal(x), dx, Val(N))
        end,
        primals,
        tangents,
    )
end

@inline _nfwd_unpack_output_lane(::T, dy::T, ::Val{1}) where {T<:IEEEFloat} = dy
@inline _nfwd_unpack_output_lane(
    ::Complex{T}, dy::Complex{T}, ::Val{1}
) where {T<:IEEEFloat} = dy
@inline _nfwd_unpack_output_lane(::T, dy::NTuple{N,T}, ::Val{k}) where {T<:IEEEFloat,N,k} = dy[k]
@inline _nfwd_unpack_output_lane(::Complex{T}, dy::NTuple{N,Complex{T}}, ::Val{k}) where {T<:IEEEFloat,N,k} = dy[k]
@inline function _nfwd_unpack_output_lane(y::Tuple, dy::Tuple, ::Val{k}) where {k}
    return tuple_map((yi, dyi) -> _nfwd_unpack_output_lane(yi, dyi, Val(k)), y, dy)
end

# Dense array unpack: width-1 tangent is the array itself; width-N tangent is NTuple of arrays.
@inline _nfwd_unpack_output_lane(
    ::Array{T}, dy::Array{T}, ::Val{1}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}}} = dy
@inline _nfwd_unpack_output_lane(::Array{T}, dy::NTuple{N,Array{T}}, ::Val{k}) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},N,k} = dy[k]

@inline function _nfwd_extract_primitive_parts(
    y::Nfwd.NDual{T,N}, ::Val{N}
) where {T<:IEEEFloat,N}
    p = Nfwd.ndual_value(y)
    t = N == 1 ? Nfwd.ndual_partial(y, 1) : Nfwd.ndual_partials(y)
    return p, t
end

@inline function _nfwd_extract_primitive_parts(
    y::Complex{Nfwd.NDual{T,N}}, ::Val{N}
) where {T<:IEEEFloat,N}
    p = Nfwd._nfwd_dual_value(y)
    t = if N == 1
        Nfwd._nfwd_dual_partial(y, 1)
    else
        ntuple(i -> Nfwd._nfwd_dual_partial(y, i), Val(N))
    end
    return p, t
end

@inline function _nfwd_extract_primitive_parts(y::Tuple, ::Val{N}) where {N}
    parts = map(yi -> _nfwd_extract_primitive_parts(yi, Val(N)), y)
    return map(first, parts), map(last, parts)
end

# ── Dense array extraction ───────────────────────────────────────────────────

function _nfwd_extract_primitive_parts(
    y::Array{Nfwd.NDual{T,N}}, ::Val{N}
) where {T<:IEEEFloat,N}
    p = similar(y, T)
    if N == 1
        t = similar(y, T)
        @inbounds for i in eachindex(y)
            p[i] = Nfwd.ndual_value(y[i])
            t[i] = Nfwd.ndual_partial(y[i], 1)
        end
    else
        t = ntuple(Val(N)) do k
            tk = similar(y, T)
            @inbounds for i in eachindex(y)
                tk[i] = Nfwd.ndual_partial(y[i], k)
            end
            tk
        end
        @inbounds for i in eachindex(y)
            p[i] = Nfwd.ndual_value(y[i])
        end
    end
    return p, t
end

function _nfwd_extract_primitive_parts(
    y::Array{Complex{Nfwd.NDual{T,N}}}, ::Val{N}
) where {T<:IEEEFloat,N}
    CT = Complex{T}
    p = similar(y, CT)
    @inbounds for i in eachindex(y)
        p[i] = Nfwd._nfwd_dual_value(y[i])
    end
    if N == 1
        t = similar(y, CT)
        @inbounds for i in eachindex(y)
            t[i] = Nfwd._nfwd_dual_partial(y[i], 1)
        end
    else
        t = ntuple(Val(N)) do k
            tk = similar(y, CT)
            @inbounds for i in eachindex(y)
                tk[i] = Nfwd._nfwd_dual_partial(y[i], k)
            end
            tk
        end
    end
    return p, t
end

@inline _nfwd_public_output_tangent(::Any, t, ::Val{1}) = t
@inline _nfwd_public_output_tangent(::Any, t, ::Val{N}) where {N} = NTangent(t)
@inline function _nfwd_public_output_tangent(p::Tuple, t::Tuple, ::Val{1})
    return t
end
@inline function _nfwd_public_output_tangent(p::Tuple, t::Tuple, ::Val{N}) where {N}
    return NTangent(ntuple(lane -> _nfwd_unpack_output_lane(p, t, Val(lane)), Val(N)))
end

# Scalar / complex NDual: short-circuit the extract → wrap → unwrap round-trip.
# N>1: the result is already an NDual / Complex{NDual}, return as-is.
# N=1: convert to the width-1 Dual type expected by callers.
@inline _nfwd_extract_primitive_output(
    y::Nfwd.NDual{T,N}, ::Val{N}
) where {T<:IEEEFloat,N} = y
@inline function _nfwd_extract_primitive_output(
    y::Nfwd.NDual{T,1}, ::Val{1}
) where {T<:IEEEFloat}
    return Dual{T,NTangent{Tuple{T}}}(
        Nfwd.ndual_value(y), NTangent((Nfwd.ndual_partial(y, 1),))
    )
end
@inline _nfwd_extract_primitive_output(
    y::Complex{Nfwd.NDual{T,N}}, ::Val{N}
) where {T<:IEEEFloat,N} = y
@inline function _nfwd_extract_primitive_output(
    y::Complex{Nfwd.NDual{T,1}}, ::Val{1}
) where {T<:IEEEFloat}
    CT = Complex{T}
    v = Nfwd._nfwd_dual_value(y)
    d = Nfwd._nfwd_dual_partial(y, 1)
    return Dual{CT,NTangent{Tuple{CT}}}(v, NTangent((d,)))
end

# Tuple / other fallback: extract, wrap as public tangent, reconstruct Dual.
@inline function _nfwd_extract_primitive_output(y, ::Val{N}) where {N}
    p, t = _nfwd_extract_primitive_parts(y, Val(N))
    public_tangent = _nfwd_public_output_tangent(p, t, Val(N))
    return dual_type(Val(N), typeof(p))(p, public_tangent)
end

@inline function _nfwd_contract_output(ȳ::T, dy::T) where {T<:IEEEFloat}
    return (ȳ * Nfwd._nfwd_zero_mask(ȳ, dy),)
end

@inline function _nfwd_contract_output(ȳ::Complex{T}, dy::Complex{T}) where {T<:IEEEFloat}
    return (real(conj(ȳ) * Nfwd._nfwd_zero_mask(ȳ, dy)),)
end

@inline function _nfwd_contract_output(ȳ::T, dy::NTuple{N,T}) where {T<:IEEEFloat,N}
    return ntuple(k -> ȳ * Nfwd._nfwd_zero_mask(ȳ, dy[k]), Val(N))
end

@inline function _nfwd_contract_output(
    ȳ::Complex{T}, dy::NTuple{N,Complex{T}}
) where {T<:IEEEFloat,N}
    return ntuple(k -> real(conj(ȳ) * Nfwd._nfwd_zero_mask(ȳ, dy[k])), Val(N))
end

@inline function _nfwd_contract_output(ȳ::Tuple, dy::Tuple)
    contributions = map(_nfwd_contract_output, ȳ, dy)
    return foldl((a, b) -> map(+, a, b), contributions)
end

@inline function _nfwd_seed_scalar_tangent(
    ::Type{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    return ntuple(
        k -> (offset + 1 == start_slot + k - 1 ? one(T) : zero(T)), Val(chunk_size)
    )
end

@inline function _nfwd_seed_complex_tangent(
    ::Type{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    return ntuple(k -> begin
        slot = start_slot + k - 1
        if offset + 1 == slot
            complex(one(T), zero(T))
        elseif offset + 2 == slot
            complex(zero(T), one(T))
        else
            zero(Complex{T})
        end
    end, Val(chunk_size))
end

@inline function _nfwd_seed_primitive_tangent(
    x::T, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    return if chunk_size == 1
        first(_nfwd_seed_scalar_tangent(T, 1, start_slot, offset))
    else
        _nfwd_seed_scalar_tangent(T, chunk_size, start_slot, offset)
    end
end

@inline function _nfwd_seed_primitive_tangent(
    x::Complex{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    return if chunk_size == 1
        first(_nfwd_seed_complex_tangent(T, 1, start_slot, offset))
    else
        _nfwd_seed_complex_tangent(T, chunk_size, start_slot, offset)
    end
end

# ── Dense array seeding ──────────────────────────────────────────────────────

# Returns Array{NDual{T,N}} directly — primal values embedded, partials seeded.
@inline function _nfwd_seed_primitive_tangent(
    x::Array{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    out = similar(x, Nfwd.NDual{T,chunk_size})
    @inbounds for i in eachindex(x)
        partials = _nfwd_seed_scalar_tangent(T, chunk_size, start_slot, offset + i - 1)
        out[i] = Nfwd.NDual{T,chunk_size}(x[i], partials)
    end
    return out
end

@inline function _nfwd_seed_primitive_tangent(
    x::Array{Complex{T}}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    out = similar(x, Complex{Nfwd.NDual{T,chunk_size}})
    @inbounds for i in eachindex(x)
        ct = _nfwd_seed_complex_tangent(T, chunk_size, start_slot, offset + 2 * (i - 1))
        out[i] = Complex(
            Nfwd.NDual{T,chunk_size}(
                real(x[i]), ntuple(k -> T(real(ct[k])), Val(chunk_size))
            ),
            Nfwd.NDual{T,chunk_size}(
                imag(x[i]), ntuple(k -> T(imag(ct[k])), Val(chunk_size))
            ),
        )
    end
    return out
end

@inline _nfwd_seed_primitive_tangent(x, chunk_size::Int, start_slot::Int, offset::Int) = NoTangent()

@inline _nfwd_primitive_input_dof(x) =
    Nfwd._nfwd_is_supported_primal(x) ? Nfwd._nfwd_input_dof(x) : 0
@inline _nfwd_seed_primitive_tangents(::Tuple{}, ::Val{N}, start_slot::Int, offset::Int=0) where {N} = ()
@inline function _nfwd_seed_primitive_tangents(
    primals::Tuple, ::Val{N}, start_slot::Int, offset::Int=0
) where {N}
    x = first(primals)
    dof = _nfwd_primitive_input_dof(x)
    return (
        _nfwd_seed_primitive_tangent(x, N, start_slot, offset),
        _nfwd_seed_primitive_tangents(
            Base.tail(primals), Val(N), start_slot, offset + dof
        )...,
    )
end

@inline function _nfwd_accumulate_scalar_gradient(g::T, slot::Int, v) where {T<:IEEEFloat}
    return slot == 1 ? g + v : g
end

@inline function _nfwd_accumulate_scalar_gradient(
    g::Complex{T}, slot::Int, v
) where {T<:IEEEFloat}
    if slot == 1
        return g + complex(v, zero(T))
    elseif slot == 2
        return g + complex(zero(T), v)
    end
    return g
end

# ── Dense array gradient accumulation ────────────────────────────────────────

@inline function _nfwd_accumulate_scalar_gradient(
    g::Array{T}, slot::Int, v
) where {T<:IEEEFloat}
    @inbounds g[slot] += v
    return g
end

@inline function _nfwd_accumulate_scalar_gradient(
    g::Array{Complex{T}}, slot::Int, v
) where {T<:IEEEFloat}
    local_idx = (slot + 1) >> 1  # ceil(slot / 2)
    if isodd(slot)
        @inbounds g[local_idx] += complex(v, zero(T))
    else
        @inbounds g[local_idx] += complex(zero(T), v)
    end
    return g
end

# Base case: global_slot exceeds total DOF (chunk padding overshoot).
@inline _nfwd_update_scalar_grad(grads::Tuple{}, ::Tuple{}, global_slot::Int, lane_val, offset::Int=0) = ()

@inline function _nfwd_update_scalar_grad(
    grads::Tuple, primals::Tuple, global_slot::Int, lane_val, offset::Int=0
)
    x = first(primals)
    dof = _nfwd_primitive_input_dof(x)
    if offset < global_slot <= offset + dof
        local_slot = global_slot - offset
        return (
            _nfwd_accumulate_scalar_gradient(first(grads), local_slot, lane_val),
            Base.tail(grads)...,
        )
    end
    return (
        first(grads),
        _nfwd_update_scalar_grad(
            Base.tail(grads), Base.tail(primals), global_slot, lane_val, offset + dof
        )...,
    )
end

@inline function _nfwd_scatter_scalar_chunk(
    grads::Tuple, primals::Tuple, dy::Tuple, start_slot::Int
)
    global_slot = start_slot
    for lane_val in dy
        grads = _nfwd_update_scalar_grad(grads, primals, global_slot, lane_val)
        global_slot += 1
    end
    return grads
end

@inline function _nfwd_primitive_width(x::Any, xs::Vararg{Any,M}) where {M}
    N = _dual_width(x)
    all(y -> _dual_width(y) == N, xs) || throw(
        ArgumentError(
            "nfwd primitive rules expect all width-aware dual-type inputs to have the same width.",
        ),
    )
    return Val(N)
end

@inline function _nfwd_primitive_frule_call(::Val{N}, f::Dual, x::Vararg{Any,M}) where {M,N}
    all(xi -> try
        primal(xi)
        tangent(xi)
        true
    catch
        false
    end, x) || throw(
        ArgumentError(
            "nfwd primitive rules expect forward inputs exposing `primal` and `tangent`.",
        ),
    )
    _nfwd_check_function_tangent(tangent(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    lifted = _nfwd_lift_primitive_args(Val(N), primals, tangents)
    return _nfwd_extract_primitive_output(primal(f)(lifted...), Val(N))
end

@inline function _nfwd_primitive_rrule_call(
    ::Val{N}, f::CoDual, x::Vararg{CoDual,M}
) where {M,N}
    _nfwd_check_function_tangent(tangent(f))
    primals = map(primal, x)
    y_primal = primal(f)(primals...)
    Nfwd._nfwd_is_supported_primal(y_primal) || Nfwd._nfwd_output_error(primals, y_primal)
    y_fdata = fdata(zero_tangent(y_primal))
    y = CoDual(y_primal, y_fdata)
    function primitive_pb!!(y_rdata)
        ȳ = tangent(y_fdata, y_rdata)
        grads = map(zero_tangent, primals)
        total_dof = Nfwd._nfwd_input_dof(primals)
        for start_slot in 1:N:total_dof
            seeded = _nfwd_seed_primitive_tangents(primals, Val(N), start_slot)
            lifted = _nfwd_lift_primitive_args(Val(N), primals, seeded)
            _, dy = _nfwd_extract_primitive_parts(primal(f)(lifted...), Val(N))
            lane_vals = _nfwd_contract_output(ȳ, dy)
            grads = _nfwd_scatter_scalar_chunk(grads, primals, lane_vals, start_slot)
        end
        return (NoRData(), map(rdata, grads)...)
    end
    return y, primitive_pb!!
end
