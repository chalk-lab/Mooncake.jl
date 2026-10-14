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

# `inv`, `det`, `logabsdet` and `\` on a dense matrix check its values and, if it is
# triangular or diagonal, take a path that never reads the entries outside the triangle.
# Differentiating that path gives those entries a zero gradient, so these rules use the matrix
# formulas instead, which hold whichever path computes the primal (#1372).
#
# The rules only cover arrays whose entries each map to one entry of their storage, so that a
# dense gradient can be written into their tangent. Structured matrices (`Symmetric`,
# `UpperTriangular`, `Diagonal`, ...) have their own methods, which don't branch on values.
const DenseMat{P} = Union{
    Matrix{P},SubArray{P,2,<:Array{P}},Base.ReshapedArray{P,2,<:SubArray{P,1,<:Array{P}}}
}
const DenseMatOrAdj{P} = Union{
    DenseMat{P},Adjoint{P,<:DenseMat{P}},Transpose{P,<:DenseMat{P}}
}
const DenseVecOrMatArg{P} = Union{Vector{P},SubArray{P,1,<:Array{P}},DenseMatOrAdj{P}}

@is_primitive MinimalCtx Tuple{typeof(inv),DenseMat{P}} where {P<:BlasRealFloat}
function frule!!(::Dual{typeof(inv)}, A_dA::Dual{<:DenseMat{P}}) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    Y = inv(A)
    return Dual(Y, -Y * dA * Y)
end
function rrule!!(
    ::CoDual{typeof(inv)}, A_dA::CoDual{<:DenseMat{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    Y = inv(A)
    dY = zero(Y)
    function inv_pb!!(::NoRData)
        dA .-= Y' * dY * Y'
        return NoRData(), NoRData()
    end
    return CoDual(Y, dY), inv_pb!!
end

# For a singular matrix the gradient is set to zero, as in the `Symmetric` rules in lapack.jl.
# The true gradient there is the adjugate, which is nonzero at rank n - 1.
@is_primitive MinimalCtx Tuple{typeof(det),DenseMatOrAdj{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Dual{typeof(det)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    d = det(A)
    iszero(d) && return Dual(d, zero(P))
    return Dual(d, d * tr(A \ dA))
end
function rrule!!(
    ::CoDual{typeof(det)}, A_dA::CoDual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    d = det(A)
    Ainv = iszero(d) ? nothing : inv(A)
    function det_pb!!(ȳ::P)
        isnothing(Ainv) || (dA .+= (ȳ * d) .* Ainv')
        return NoRData(), NoRData()
    end
    return CoDual(d, NoFData()), det_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(logabsdet),DenseMatOrAdj{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Dual{typeof(logabsdet)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s = logabsdet(A)
    iszero(s) && return Dual((ld, s), (zero(P), zero(P)))
    return Dual((ld, s), (tr(A \ dA), zero(P)))
end
function rrule!!(
    ::CoDual{typeof(logabsdet)}, A_dA::CoDual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s = logabsdet(A)
    Ainv = iszero(s) ? nothing : inv(A)
    function logabsdet_pb!!(ȳ::Tuple{P,P})
        isnothing(Ainv) || (dA .+= ȳ[1] .* Ainv')
        return NoRData(), NoRData()
    end
    return CoDual((ld, s), NoFData()), logabsdet_pb!!
end

# `X = A \ B` is `pinv(A) * B` when `A` is not square (least squares or minimum norm), so the
# rules use the derivative of the pseudo-inverse, which assumes `A` has full rank. For square
# `A` the extra terms vanish.
@is_primitive(
    MinimalCtx,
    Tuple{typeof(\),DenseMatOrAdj{P},DenseVecOrMatArg{P}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(\)}, A_dA::Dual{<:DenseMatOrAdj{P}}, B_dB::Dual{<:DenseVecOrMatArg{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    X = A \ B
    dX = A \ (dB - dA * X)
    if size(A, 1) != size(A, 2)
        Z = dA' * (A' \ X)
        dX .+= A \ (A' \ (dA' * (B - A * X))) .+ Z .- A \ (A * Z)
    end
    return Dual(X, dX)
end
function rrule!!(
    ::CoDual{typeof(\)},
    A_dA::CoDual{<:DenseMatOrAdj{P}},
    B_dB::CoDual{<:DenseVecOrMatArg{P}},
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    X = A \ B
    dX = zero(X)
    function ldiv_pb!!(::NoRData)
        B̄ = A' \ dX
        dB .+= B̄
        dA .-= B̄ * X'
        if size(A, 1) != size(A, 2)
            dA .+= (B - A * X) * B̄' / A' .+ (A' \ X) * (dX' - B̄' * A)
        end
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(X, dX), ldiv_pb!!
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
    # inv, det, logabsdet and \ on dense matrices, including ones whose values are triangular
    # or diagonal (#1372).
    dense_cases = map_prod([1, 3], Ps) do (N, P)
        U = triu(randn(rng, P, N, N)) + 3I
        As = vcat(
            invertible_blas_matrices(rng, P, N),
            [U, collect(U'), Matrix{P}(Diagonal(U))],
        )
        # For Float32 det, finite differences are too inaccurate (see the Symmetric cases).
        cases = Any[]
        for A in As
            push!(cases, (false, :none, nothing, inv, A))
            push!(cases, (P == Float32, :none, nothing, det, A))
            push!(cases, (false, :none, nothing, logabsdet, A))
            push!(cases, (false, :none, nothing, \, A, randn(rng, P, N)))
            push!(cases, (false, :none, nothing, \, A, randn(rng, P, N, 2)))
        end
        A = U + randn(rng, P, N, N)
        push!(cases, (P == Float32, :none, nothing, det, A'))
        push!(cases, (false, :none, nothing, logabsdet, transpose(A)))
        push!(cases, (false, :none, nothing, \, A', randn(rng, P, 2, N)'))
        return cases
    end
    # Non-square \ (least squares and minimum norm).
    ldiv_cases = map_prod([(5, 3), (3, 5)], Ps) do ((m, n), P)
        return Any[
            (false, :none, nothing, \, randn(rng, P, m, n), randn(rng, P, m)),
            (false, :none, nothing, \, randn(rng, P, m, n), randn(rng, P, m, 2)),
        ]
    end
    test_cases = vcat(
        test_cases,
        dense_cases...,
        ldiv_cases...,
        # det of a matrix of rank n - 2, whose gradient is zero.
        Any[(false, :none, nothing, det, zeros(Float64, 3, 3))],
    )
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
        # logdet and / reach the logabsdet and \ rules (#1372).
        map_prod([3], Ps) do (N, P)
            U = triu(randn(rng, P, N, N)) + 3I
            return Any[
                (false, :none, nothing, logdet, U),
                (false, :none, nothing, /, randn(rng, P, 2, N), U),
            ]
        end...,
    )
    memory = Any[]
    return test_cases, memory
end
