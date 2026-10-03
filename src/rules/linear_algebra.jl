@is_primitive MinimalCtx ForwardMode Tuple{
    LinearAlgebra.MulAddMul{true,b,A,B},P
} where {
    b,
    A<:Union{Bool,Base.BitInteger,BlasFloat},
    B<:Union{Bool,Base.BitInteger,BlasFloat},
    P<:BlasFloat,
}
@is_primitive MinimalCtx ForwardMode Tuple{
    Union{LinearAlgebra.MulAddMul{true,b,A,B},LinearAlgebra.MulAddMul{false,true,A,B}},P,P
} where {
    b,
    A<:Union{Bool,Base.BitInteger,BlasFloat},
    B<:Union{Bool,Base.BitInteger,BlasFloat},
    P<:BlasFloat,
}

# MulAddMul's `alpha == 1` / `beta == 0` shortcuts drop the coefficient's direction, so
# differentiate through the coefficients explicitly.
@inline function frule!!(
    p::Lifted{<:LinearAlgebra.MulAddMul,N}, x::Lifted{P,N}, ys::Vararg{Lifted{P,N},K}
) where {P<:BlasFloat,N,K}
    z = primal(p)(primal(x), map(primal, ys)...)
    coef(f) = _fwd_blas_alpha(
        typeof(z), frule!!(zero_lifted(Val(N), lgetfield), p, zero_lifted(Val(N), f))
    )
    # Lane `k` of d(c·v): a strong zero on either factor's zero direction.
    function term(c, v, k)
        dv = oftype(z, tangent(v, k))
        return (iszero(dv) ? zero(z) : _rvs_mul(dv, primal(c))) +
               _rvs_mul(oftype(z, primal(v)), tangent(c, k))
    end
    α = coef(Val(:alpha))
    dx = ntuple(k -> term(α, x, k), Val(N))
    K == 0 && return Lifted{typeof(z),N}(z, _scalar_ndual(z, dx))
    β, y = coef(Val(:beta)), only(ys)
    ds = ntuple(k -> dx[k] + term(β, y, k), Val(N))
    return Lifted{typeof(z),N}(z, _scalar_ndual(z, ds))
end

@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(LinearAlgebra._modify!),
    Union{LinearAlgebra.MulAddMul{true,b,A,B},LinearAlgebra.MulAddMul{false,true,A,B}},
    P,
    Array{P},
    Union{Integer,Tuple,CartesianIndex},
} where {
    b,
    A<:Union{Bool,Base.BitInteger,BlasFloat},
    B<:Union{Bool,Base.BitInteger,BlasFloat},
    P<:BlasFloat,
}
@inline function frule!!(
    ::Lifted{typeof(LinearAlgebra._modify!),N},
    p::Lifted{<:LinearAlgebra.MulAddMul,N},
    x::Lifted{P,N},
    C::Lifted{<:Array{P},N},
    idx::Lifted,
) where {P<:BlasFloat,N}
    i = CartesianIndex(primal(idx))
    y = Lifted{P,N}(primal(C)[i], tangent(C)[i])
    out = frule!!(p, x, y)
    tangent(C)[i] = tangent(out)
    return zero_lifted(Val(N), nothing)
end

# friendly_tangent_cache and tangent_to_friendly_internal!! for structured matrix types.
#
# Symmetric, Hermitian, and SymTridiagonal store only part of the matrix internally but
# represent a full symmetric/Hermitian matrix. The user-facing gradient is a plain Matrix{T}.
#
# Because we do not track which elements were getindex'ed, we cannot assume the tangent
# retains the original structure — it must be treated as a dense matrix. The original
# Symmetric/Hermitian/SymTridiagonal structure is therefore lost in the friendly gradient.
#
# friendly_tangent_cache pre-allocates the Matrix{T} output buffer at prepare time.
# tangent_to_friendly_internal!! copies the stored tangent fields directly into dest.
# The stored triangle (for Symmetric/Hermitian) or diagonals (for SymTridiagonal) hold the
# accumulated chain-rule gradient; all other entries are zero-initialised by Mooncake and
# are left zero by fill! (SymTridiagonal) or implicit via copyto! (Symmetric/Hermitian).
#
# For Hermitian{T} where T is complex: the stored triangle of .data accumulates the
# chain-rule gradient for both logical positions it represents (via Mooncake's usual
# tangent accumulation), and the non-stored triangle is zero-initialised. copyto! copies
# the full data matrix (including complex entries) to dest, which is a plain Matrix{T}.

# Adjoint and Transpose lose nothing — each entry is one parent entry, relabelled — so AsPrimal
# can rebuild the wrapper around the parent's gradient, giving an `AbstractMatrix` of `size(x)`
# rather than a raw `Tangent`. Reconstruction recurses, so `Adjoint(Symmetric(A))` works too.
#
# The conjugation in `Adjoint(dparent)` is right: Mooncake cotangents are real gradients
# (∂L/∂re + i ∂L/∂im) and x[i, j] == conj(parent[j, i]), so d/dx[i, j] == conj(dparent[j, i]).
#
# Test the parent's eltype, not `eltype(x)`, which is `Union{}` for e.g. `Matrix{Symbol}`. A
# non-differentiable one stays AsRaw; AsPrimal would return primal entries as a gradient.
function Mooncake.friendly_tangent_cache(
    x::Union{LinearAlgebra.Adjoint,LinearAlgebra.Transpose}
)
    tangent_type(eltype(parent(x))) == NoTangent &&
        return FriendlyTangentCache{AsRaw}(nothing)
    return FriendlyTangentCache{AsPrimal}(_copy_output(x))
end

function Mooncake.friendly_tangent_cache(x::LinearAlgebra.Symmetric{T}) where {T}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, size(x)...))
end
function Mooncake.friendly_tangent_cache(x::LinearAlgebra.Hermitian{T}) where {T}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, size(x)...))
end
function Mooncake.friendly_tangent_cache(x::LinearAlgebra.SymTridiagonal{T}) where {T}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, length(x.dv), length(x.dv)))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, ::LinearAlgebra.Symmetric{T}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.data))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, ::LinearAlgebra.Hermitian{T}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.data))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, ::LinearAlgebra.SymTridiagonal{T}, tangent
) where {T}
    dv = val(tangent.fields.dv)
    ev = val(tangent.fields.ev)
    fill!(tangent_as_friendly, zero(T))
    @inbounds for i in eachindex(dv)
        tangent_as_friendly[i, i] = dv[i]
    end
    @inbounds for i in eachindex(ev)
        tangent_as_friendly[i, i + 1] = ev[i]
        tangent_as_friendly[i + 1, i] = ev[i]
    end
    return tangent_as_friendly
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:linear_algebra})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    test_cases = if Base.get_extension(Mooncake, :MooncakeChainRulesExt) === nothing
        Any[]
    else
        vcat(
            map_prod([3, 7], Ps) do (N, P)
                return (false, :none, nothing, exp, randn(rng, P, N, N))
            end,
        )
    end
    test_cases = Any[test_cases...]
    # Pinned struct seeds cannot be replicated by the chunked registry harness.
    for P in (Float64, ComplexF64), a in (1, 2, Inf), b in (0, 2), unary in (false, true)
        (a == 1 || (!unary && b == 0)) || continue
        p = LinearAlgebra.MulAddMul(P(a), P(b))
        dp = Tangent((; alpha=P(3), beta=P(5)))
        args = if unary
            (CoDual(P(7), zero(P)),)
        else
            (CoDual(P(7), zero(P)), CoDual(P(11), zero(P)))
        end
        expected = P(unary ? 21 : 76)
        opts = (mode=ForwardMode, skip_chunked=true, oracle=(deriv=expected,))
        push!(test_cases, (false, :allocs, opts, CoDual(p, dp), args...))
    end
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:linear_algebra})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    test_cases = vcat(
        map_prod([3, 7], Ps) do (N, P)
            flags = (false, :none, nothing)
            Any[
                (flags..., inv, randn(rng, P, N, N)), (flags..., det, randn(rng, P, N, N))
            ]
        end...,
    )
    # A mutating MulAddMul returns nothing, so its derivative oracle needs a returned array.
    test_cases = Any[test_cases...]
    for P in (Float64, ComplexF64), integer_beta in (false, true)
        f = function (a, b, x, C)
            LinearAlgebra._modify!(
                LinearAlgebra.MulAddMul{true,true,typeof(a),typeof(b)}(a, b), x, C, 1
            )
            return C
        end
        opts = (mode=ForwardMode, oracle=(deriv=fill(P(integer_beta ? 21 : 76), 1),))
        push!(
            test_cases,
            (
                false,
                :allocs,
                opts,
                f,
                CoDual(one(P), P(3)),
                integer_beta ? 0 : CoDual(zero(P), P(5)),
                CoDual(P(7), zero(P)),
                fill(P(11), 1),
            ),
        )
    end
    memory = Any[]
    return test_cases, memory
end
