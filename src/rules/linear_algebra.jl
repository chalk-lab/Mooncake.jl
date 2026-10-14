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

# `inv`, `det`, `logabsdet` and `\` on a dense matrix check whether its values are triangular
# and, if so, take a path that never reads the zeros, which would get a zero gradient. These
# rules use the matrix formulas instead (#1372). They only take arrays whose entries each map
# to one stored entry, so a dense gradient can be written into the tangent.
const DenseMat{P} = Union{
    Matrix{P},
    SubArray{P,2,<:Array{P}},
    Base.ReshapedArray{P,2,<:SubArray{P,<:Any,<:Array{P}}},
}
const DenseMatOrAdj{P} = Union{
    DenseMat{P},Adjoint{P,<:DenseMat{P}},Transpose{P,<:DenseMat{P}}
}
const DenseVecOrMatArg{P} = Union{Vector{P},SubArray{P,1,<:Array{P}},DenseMatOrAdj{P}}

# The factorization LinearAlgebra uses for a square `A`, so that the value and the gradient
# need only one.
function _square_factor(A; check)
    istriu(A) && return UpperTriangular(A)
    istril(A) && return LowerTriangular(A)
    return lu(A; check)
end

@is_primitive MinimalCtx Tuple{typeof(inv),DenseMat{P}} where {P<:BlasRealFloat}
function frule!!(::Dual{typeof(inv)}, A_dA::Dual{<:DenseMat{P}}) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    Y = inv(A)
    return Dual(Y, mul!(similar(Y), Y, dA * Y, -one(P), zero(P)))
end
function rrule!!(
    ::CoDual{typeof(inv)}, A_dA::CoDual{<:DenseMat{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    Y = inv(A)
    dY = zero(Y)
    function inv_pb!!(::NoRData)
        mul!(dA, Y', dY * Y', -one(P), one(P))
        return NoRData(), NoRData()
    end
    return CoDual(Y, dY), inv_pb!!
end

# `det(A)` and its gradient, the transposed adjugate `det(A) * inv(A)'`. When `det(A)` is
# zero, subnormal or infinite that product is undefined, overflows or loses precision, though
# the adjugate is finite (and nonzero at rank n - 1), so it then comes from the SVD.
function _det_and_grad(A::AbstractMatrix{P}) where {P}
    F = _square_factor(A; check=false)
    d = det(F)
    if isfinite(d) && !iszero(d) && !issubnormal(d)
        return d, rmul!(ldiv!(F', Matrix{P}(I, size(A))), d)
    end
    U, σ, V = svd(A)
    p = [prod(σ[j] for j in eachindex(σ) if j != i; init=one(P)) for i in eachindex(σ)]
    return d, (det(U) * det(V)) .* (U * Diagonal(p) * V')
end

@is_primitive MinimalCtx Tuple{typeof(det),DenseMatOrAdj{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Dual{typeof(det)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    d, G = _det_and_grad(A)
    return Dual(d, dot(G, dA))
end
function rrule!!(
    ::CoDual{typeof(det)}, A_dA::CoDual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    d, G = _det_and_grad(A)
    function det_pb!!(ȳ::P)
        dA .+= ȳ .* G
        return NoRData(), NoRData()
    end
    return CoDual(d, NoFData()), det_pb!!
end

# `logabsdet(A)` and the gradient of its first output, `inv(A)'`, which is unbounded at a
# singular `A`.
function _logabsdet_and_grad(A::AbstractMatrix{P}) where {P}
    F = _square_factor(A; check=false)
    ld, s = logabsdet(F)
    if iszero(s)
        msg = "Mooncake cannot differentiate `logabsdet` of a singular matrix."
        throw(ArgumentError(msg))
    end
    return ld, s, ldiv!(F', Matrix{P}(I, size(A)))
end

@is_primitive MinimalCtx Tuple{typeof(logabsdet),DenseMatOrAdj{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Dual{typeof(logabsdet)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s, G = _logabsdet_and_grad(A)
    return Dual((ld, s), (dot(G, dA), zero(P)))
end
function rrule!!(
    ::CoDual{typeof(logabsdet)}, A_dA::CoDual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s, G = _logabsdet_and_grad(A)
    function logabsdet_pb!!(ȳ::Tuple{P,P})
        dA .+= ȳ[1] .* G
        return NoRData(), NoRData()
    end
    return CoDual((ld, s), NoFData()), logabsdet_pb!!
end

# For a non-square `A`, `A \ B` is `pinv(A) * B` (least squares or minimum norm). The rules
# work with whichever of `A` and `A'` is tall, `M`, through `(M'M) \ x`, computed from the R
# factor of its pivoted QR. At a rank-deficient `A` the solution, which LinearAlgebra
# truncates by rank, is not differentiable, so they throw.
function _tall_qr(M::AbstractMatrix{P}) where {P}
    F = qr(M, ColumnNorm())
    # The rank as LinearAlgebra's least-squares solve estimates it, with the same tolerance.
    rank = LinearAlgebra.ldiv!(F, zeros(P, size(M, 1), 1), min(size(M)...) * eps(P))[2]
    if rank < size(M, 2)
        msg =
            "Mooncake cannot differentiate `A \\ B` for a non-square `A` without full rank, " *
            "where the solution is not differentiable."
        throw(ArgumentError(msg))
    end
    return F
end

# `(M'M) \ x` for the pivoted QR `F` of a tall `M` with full rank.
function _gram_ldiv(F, x)
    R = UpperTriangular(F.R)
    y = R \ (R' \ _rows(x, F.p))
    out = similar(y)
    _rows(out, F.p) .= y
    return out
end
_rows(x::AbstractVector, p) = view(x, p)
_rows(x::AbstractMatrix, p) = view(x, p, :)

@is_primitive(
    MinimalCtx,
    Tuple{typeof(\),DenseMatOrAdj{P},DenseVecOrMatArg{P}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(\)}, A_dA::Dual{<:DenseMatOrAdj{P}}, B_dB::Dual{<:DenseVecOrMatArg{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    m, n = size(A)
    if m == n
        F = _square_factor(A; check=true)
        X = F \ B
        return Dual(X, ldiv!(F, mul!(copy(dB), dA, X, -one(P), one(P))))
    elseif m > n
        F = _tall_qr(A)
        X = F \ B
        T = A' * mul!(copy(dB), dA, X, -one(P), one(P))
        mul!(T, dA', mul!(copy(B), A, X, -one(P), one(P)), one(P), one(P))
        return Dual(X, _gram_ldiv(F, T))
    else
        F = _tall_qr(A')
        Y = _gram_ldiv(F, B)
        X = A' * Y
        dX = dA' * Y
        T = mul!(mul!(copy(dB), dA, X, -one(P), one(P)), A, dX, -one(P), one(P))
        return Dual(X, mul!(dX, A', _gram_ldiv(F, T), one(P), one(P)))
    end
end
function rrule!!(
    ::CoDual{typeof(\)},
    A_dA::CoDual{<:DenseMatOrAdj{P}},
    B_dB::CoDual{<:DenseVecOrMatArg{P}},
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    m, n = size(A)
    # Which factorizations exist depends on the shape of `A`. Holding them in `Ref`s of fixed
    # type keeps the type of the pullback the same for every shape.
    BI = LinearAlgebra.BlasInt
    TA = typeof(A)
    Sq = Union{UpperTriangular{P,TA},LowerTriangular{P,TA},LU{P,Matrix{P},Vector{BI}}}
    S = Ref{Union{Nothing,Sq}}(m == n ? _square_factor(A; check=true) : nothing)
    Q = Ref{Union{Nothing,QRPivoted{P,Matrix{P},Vector{P},Vector{BI}}}}(
        m == n ? nothing : _tall_qr(m > n ? A : A')
    )
    Y = Ref{Union{Nothing,Array{P,ndims(B)}}}(m < n ? _gram_ldiv(Q[], B) : nothing)
    X = m == n ? S[] \ B : (m > n ? Q[] \ B : A' * Y[])
    dX = zero(X)
    function ldiv_pb!!(::NoRData)
        if m == n
            B̄ = S[]' \ dX
            mul!(dA, B̄, X', -one(P), one(P))
        elseif m > n
            Z = _gram_ldiv(Q[], dX)
            B̄ = A * Z
            mul!(dA, B̄, X', -one(P), one(P))
            mul!(dA, mul!(copy(B), A, X, -one(P), one(P)), Z', one(P), one(P))
        else
            B̄ = _gram_ldiv(Q[], A * dX)
            mul!(dA, B̄, X', -one(P), one(P))
            mul!(dA, Y[], dX' - B̄' * A, one(P), one(P))
        end
        dB .+= B̄
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
            push!(cases, (false, :stability, nothing, inv, A))
            push!(cases, (P == Float32, :stability, nothing, det, A))
            push!(cases, (false, :stability, nothing, logabsdet, A))
            push!(cases, (false, :stability, nothing, \, A, randn(rng, P, N)))
            push!(cases, (false, :stability, nothing, \, A, randn(rng, P, N, 2)))
        end
        A = U + randn(rng, P, N, N)
        push!(cases, (P == Float32, :stability, nothing, det, A'))
        # A reshaped non-contiguous 2-D view.
        R = reshape(view(randn(rng, P, 6, 8), 1:2, 1:8), 4, 4)
        foreach(i -> R[i, i] += 4, 1:4)
        # `inv` of a non-strided array goes through `factorize`, which isn't type-stable.
        push!(cases, (false, :none, nothing, inv, R))
        push!(cases, (false, :stability, nothing, \, R, randn(rng, P, 4)))
        push!(cases, (false, :stability, nothing, logabsdet, transpose(A)))
        push!(cases, (false, :stability, nothing, \, A', randn(rng, P, 2, N)'))
        return cases
    end
    # Non-square \ (least squares and minimum norm).
    ldiv_cases = map_prod([(5, 3), (3, 5)], Ps) do ((m, n), P)
        return Any[
            (false, :stability, nothing, \, randn(rng, P, m, n), randn(rng, P, m)),
            (false, :stability, nothing, \, randn(rng, P, m, n), randn(rng, P, m, 2)),
        ]
    end
    test_cases = vcat(
        test_cases,
        dense_cases...,
        ldiv_cases...,
        # det of singular matrices: rank n - 1, where the gradient is the adjugate, and rank 0.
        Any[
            (false, :stability, nothing, det, [0.0 2.0 1.0; 0.0 1.0 3.0; 0.0 0.0 4.0]),
            (false, :stability, nothing, det, [1.0 2.0; 2.0 4.0]),
            (false, :stability, nothing, det, zeros(3, 3)),
        ],
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
