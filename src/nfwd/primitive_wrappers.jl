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

# ── Type bridge ───────────────────────────────────────────────────────────────

@inline dual_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
@inline dual_type(::Val{0}, ::Type{T}) where {T<:IEEEFloat} = T

@inline function dual_type(::Val{N}, ::Type{T}) where {N,T<:IEEEFloat}
    return N == 1 ? Dual{T,tangent_type(Val(1), T)} : Nfwd.NDual{T,N}
end

@inline dual_type(::Val{0}, ::Type{Complex{T}}) where {T<:IEEEFloat} = Complex{T}

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

# ── Input lifting ─────────────────────────────────────────────────────────────
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

# Width-1: plain Array{T} tangent (user-provided basis_dir, frule path).
@inline function _nfwd_lift_primitive_arg(
    x::Array{T}, dx::Array{T}, ::Val{1}
) where {T<:IEEEFloat}
    return map((xi, dxi) -> Nfwd.NDual{T,1}(xi, (dxi,)), x, dx)
end

@inline function _nfwd_lift_primitive_arg(
    x::Array{T}, dx::NTangent{<:Tuple}, ::Val{N}
) where {T<:IEEEFloat,N}
    out = similar(x, Nfwd.NDual{T,N})
    @inbounds for i in eachindex(x)
        out[i] = Nfwd.NDual{T,N}(x[i], ntuple(k -> T(dx[k][i]), Val(N)))
    end
    return out
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

@inline function _nfwd_lift_primitive_arg(
    x::Array{Complex{T}}, dx::NTangent{<:Tuple}, ::Val{N}
) where {T<:IEEEFloat,N}
    out = similar(x, Complex{Nfwd.NDual{T,N}})
    @inbounds for i in eachindex(x)
        out[i] = Complex(
            Nfwd.NDual{T,N}(real(x[i]), ntuple(k -> T(real(dx[k][i])), Val(N))),
            Nfwd.NDual{T,N}(imag(x[i]), ntuple(k -> T(imag(dx[k][i])), Val(N))),
        )
    end
    return out
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

# ── Output extraction ────────────────────────────────────────────────────────

@inline _nfwd_unpack_output_basis_dir(::T, dy::T, ::Val{1}) where {T<:IEEEFloat} = dy
@inline _nfwd_unpack_output_basis_dir(
    ::Complex{T}, dy::Complex{T}, ::Val{1}
) where {T<:IEEEFloat} = dy
@inline _nfwd_unpack_output_basis_dir(::T, dy::NTuple{N,T}, ::Val{k}) where {T<:IEEEFloat,N,k} = dy[k]
@inline _nfwd_unpack_output_basis_dir(::Complex{T}, dy::NTuple{N,Complex{T}}, ::Val{k}) where {T<:IEEEFloat,N,k} = dy[k]
@inline function _nfwd_unpack_output_basis_dir(y::Tuple, dy::Tuple, ::Val{k}) where {k}
    return tuple_map((yi, dyi) -> _nfwd_unpack_output_basis_dir(yi, dyi, Val(k)), y, dy)
end

# Dense array unpack: width-1 tangent is the array itself; width-N tangent is NTuple of arrays.
@inline _nfwd_unpack_output_basis_dir(
    ::Array{T}, dy::Array{T}, ::Val{1}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}}} = dy
@inline _nfwd_unpack_output_basis_dir(::Array{T}, dy::NTuple{N,Array{T}}, ::Val{k}) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},N,k} = dy[k]

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

@inline function _nfwd_extract_primitive_parts(y, ::Val{N}) where {N}
    tangent_type(typeof(y)) == NoTangent && return y, NoTangent()
    # Plain scalar returned from a function that strips dual info (e.g. eps, nextfloat).
    if y isa IEEEFloat
        return y, N == 1 ? zero(y) : ntuple(_ -> zero(y), Val(N))
    elseif y isa Complex{<:IEEEFloat}
        z = zero(y)
        return y, N == 1 ? z : ntuple(_ -> z, Val(N))
    end
    throw(MethodError(_nfwd_extract_primitive_parts, (y, Val(N))))
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
    return NTangent(
        ntuple(basis_dir -> _nfwd_unpack_output_basis_dir(p, t, Val(basis_dir)), Val(N))
    )
end

# Short-circuit: avoid the extract → wrap → unwrap round-trip for the common scalar case.
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

# ── Output contraction ────────────────────────────────────────────────────────

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

# ── Primitive-slot traversal ─────────────────────────────────────────────────

@inline _nfwd_primitive_input_dof(x) =
    if Nfwd._nfwd_is_supported_primal(x)
        _fold_slots((acc, _, _) -> acc + 1, 0, x, nothing)
    else
        0
    end

# ── Primitive entry points ────────────────────────────────────────────────────

@inline function _nfwd_check_function_tangent(df)
    df isa Union{NoTangent,NoFData} && return nothing
    df isa NTangent && all(t -> t isa NoTangent, df) && return nothing
    throw(ArgumentError("nfwd does not support differentiating with respect to `f`."))
end

@inline _nfwd_supports_public_forward_input(x) =
    applicable(primal, x) && applicable(tangent, x)

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
    all(_nfwd_supports_public_forward_input, x) || throw(
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
    total_dof = sum(_nfwd_primitive_input_dof, primals; init=0)
    function primitive_pb!!(y_rdata)
        ȳ = tangent(y_fdata, y_rdata)
        total_dof == 0 && return (NoRData(), map(rdata, map(zero_tangent, primals))...)
        if M == 1
            x1 = primals[1]
            if x1 isa IEEEFloat
                lifted = _nfwd_lift_primitive_arg(
                    x1,
                    N == 1 ? one(x1) : ntuple(k -> k == 1 ? one(x1) : zero(x1), Val(N)),
                    Val(N),
                )
                _, dy = _nfwd_extract_primitive_parts(primal(f)(lifted), Val(N))
                # Avoid capturing a return-type DataType in the closure; on Julia 1.11 that
                # alone was enough to reintroduce a scalar pullback allocation.
                return (NoRData(), rdata(first(_nfwd_contract_output(ȳ, dy))))
            elseif x1 isa Complex{<:IEEEFloat}
                T = typeof(real(x1))
                if N == 1
                    lifted_re = _nfwd_lift_primitive_arg(
                        x1, complex(one(T), zero(T)), Val(1)
                    )
                    _, dy_re = _nfwd_extract_primitive_parts(primal(f)(lifted_re), Val(1))
                    lifted_im = _nfwd_lift_primitive_arg(
                        x1, complex(zero(T), one(T)), Val(1)
                    )
                    _, dy_im = _nfwd_extract_primitive_parts(primal(f)(lifted_im), Val(1))
                    return (
                        NoRData(),
                        rdata(
                            complex(
                                first(_nfwd_contract_output(ȳ, dy_re)),
                                first(_nfwd_contract_output(ȳ, dy_im)),
                            ),
                        ),
                    )
                else
                    lifted = _nfwd_lift_primitive_arg(
                        x1, ntuple(k -> if k == 1
                            complex(one(T), zero(T))
                        elseif k == 2
                            complex(zero(T), one(T))
                        else
                            zero(x1)
                        end, Val(N)), Val(N)
                    )
                    _, dy = _nfwd_extract_primitive_parts(primal(f)(lifted), Val(N))
                    basis_dir_vals = _nfwd_contract_output(ȳ, dy)
                    return (
                        NoRData(),
                        rdata(
                            complex(
                                basis_dir_vals[1],
                                if N >= 2
                                    basis_dir_vals[2]
                                else
                                    zero(typeof(basis_dir_vals[1]))
                                end,
                            )
                        ),
                    )
                end
            end
        end
        seed_slots = let primals = primals
            function seed_slots(start_slot)
                seed_state = (start_slot=start_slot, next_slot=Ref(1))
                let seed_state = seed_state
                    local seed_one
                    seed_one =
                        primal_x ->
                            if primal_x isa IEEEFloat || primal_x isa Complex{<:IEEEFloat}
                                _unfold_slots(
                                    (slot_primal, st) -> begin
                                        slot = st.next_slot[]
                                        st.next_slot[] = slot + 1
                                        return if N == 1
                                            if slot == st.start_slot
                                                one(slot_primal)
                                            else
                                                zero(slot_primal)
                                            end
                                        else
                                            ntuple(
                                                k -> if slot == st.start_slot + k - 1
                                                        one(slot_primal)
                                                    else
                                                        zero(slot_primal)
                                                    end,
                                                Val(N),
                                            )
                                        end
                                    end,
                                    primal_x,
                                    seed_state,
                                )
                            elseif primal_x isa Array{<:IEEEFloat} ||
                                primal_x isa Array{<:Complex{<:IEEEFloat}}
                                _unfold_slots(
                                    (slot_primal, st) -> begin
                                        slot = st.next_slot[]
                                        st.next_slot[] = slot + 1
                                        Nfwd.NDual{typeof(slot_primal),N}(
                                            slot_primal,
                                            ntuple(
                                                k -> if slot == st.start_slot + k - 1
                                                        one(slot_primal)
                                                    else
                                                        zero(slot_primal)
                                                    end,
                                                Val(N),
                                            ),
                                        )
                                    end,
                                    primal_x,
                                    seed_state,
                                )
                            elseif primal_x isa Tuple
                                tuple_map(seed_one, primal_x)
                            else
                                NoTangent()
                            end
                    map(seed_one, primals)
                end
            end
        end
        slot_grads = if total_dof == 1
            slot_grad_1 = nothing
            for start_slot in 1:N:total_dof
                lifted = _nfwd_lift_primitive_args(Val(N), primals, seed_slots(start_slot))
                _, dy = _nfwd_extract_primitive_parts(primal(f)(lifted...), Val(N))
                basis_dir_vals = _nfwd_contract_output(ȳ, dy)
                if isnothing(slot_grad_1)
                    slot_grad_1 = zero(typeof(first(basis_dir_vals)))
                end
                slot_grad_1 += first(basis_dir_vals)
            end
            (slot_grad_1,)
        elseif total_dof == 2
            # Peel the first iteration to establish concrete element type,
            # avoiding Union{Nothing,T} on v1.10.
            _first_lifted = _nfwd_lift_primitive_args(Val(N), primals, seed_slots(1))
            _, _first_dy = _nfwd_extract_primitive_parts(
                primal(f)(_first_lifted...), Val(N)
            )
            _first_basis_dir_vals = _nfwd_contract_output(ȳ, _first_dy)
            Tbasis_dir = typeof(first(_first_basis_dir_vals))
            slot_grad_1 = zero(Tbasis_dir) + _first_basis_dir_vals[1]
            slot_grad_2 = zero(Tbasis_dir)
            if length(_first_basis_dir_vals) > 1
                slot_grad_2 += _first_basis_dir_vals[2]
            end
            for start_slot in (1 + N):N:total_dof
                lifted = _nfwd_lift_primitive_args(Val(N), primals, seed_slots(start_slot))
                _, dy = _nfwd_extract_primitive_parts(primal(f)(lifted...), Val(N))
                basis_dir_vals = _nfwd_contract_output(ȳ, dy)
                slot_grad_1 += basis_dir_vals[1]
                if length(basis_dir_vals) > 1
                    slot_grad_2 += basis_dir_vals[2]
                end
            end
            (slot_grad_1, slot_grad_2)
        else
            slot_grads = nothing
            for start_slot in 1:N:total_dof
                lifted = _nfwd_lift_primitive_args(Val(N), primals, seed_slots(start_slot))
                _, dy = _nfwd_extract_primitive_parts(primal(f)(lifted...), Val(N))
                basis_dir_vals = _nfwd_contract_output(ȳ, dy)
                if isnothing(slot_grads)
                    slot_grads = zeros(typeof(first(basis_dir_vals)), total_dof)
                end
                for (basis_dir, basis_dir_val) in enumerate(basis_dir_vals)
                    slot = start_slot + basis_dir - 1
                    slot > total_dof && break
                    @inbounds slot_grads[slot] += basis_dir_val
                end
            end
            slot_grads
        end
        grad_state = (vals=slot_grads, next_slot=Ref(1))
        grads = let grad_state = grad_state
            local rebuild_grad
            rebuild_grad =
                primal_x ->
                    if primal_x isa IEEEFloat ||
                        primal_x isa Complex{<:IEEEFloat} ||
                        primal_x isa Array{<:IEEEFloat} ||
                        primal_x isa Array{<:Complex{<:IEEEFloat}}
                        _unfold_slots(
                            (_, st) -> begin
                                y_slot = st.vals[st.next_slot[]]
                                st.next_slot[] = st.next_slot[] + 1
                                y_slot
                            end,
                            primal_x,
                            grad_state,
                        )
                    elseif primal_x isa Tuple
                        tuple_map(rebuild_grad, primal_x)
                    else
                        zero_tangent(primal_x)
                    end
            map(rebuild_grad, primals)
        end
        return (NoRData(), map(rdata, grads)...)
    end
    return y, primitive_pb!!
end

# ── Test resources ────────────────────────────────────────────────────────────

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
