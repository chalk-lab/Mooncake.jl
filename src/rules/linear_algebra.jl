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
function _square_factor(A)
    istriu(A) && return UpperTriangular(A)
    istril(A) && return LowerTriangular(A)
    return lu(A; check=false)
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
        _addmul!(dA, Y', dY * Y', -one(P))
        return NoRData(), NoRData()
    end
    return CoDual(Y, dY), inv_pb!!
end

# `det(A)` and its gradient, the transposed adjugate `det(A) * inv(A)'`. When `det(A)` is
# zero, subnormal or infinite that product is undefined, overflows or loses precision, though
# the adjugate is finite (and nonzero at rank n - 1), so it then comes from the SVD.
function _det_and_grad(A::AbstractMatrix{P}) where {P}
    F = _square_factor(A)
    d = det(F)
    if (isfinite(d) && !iszero(d) && !issubnormal(d)) || !all(isfinite, A)
        return d, rmul!(ldiv!(F', Matrix{P}(I, size(A))), d)
    end
    U, σ, V = svd(A)
    p = [prod(σ[j] for j in eachindex(σ) if j != i; init=one(P)) for i in eachindex(σ)]
    return d, (det(U) * det(V)) .* (U * Diagonal(p) * V')
end

# For second derivatives, the forward-mode derivative of `_det_and_grad` uses the formula
# rather than its code, whose triangular case would drop the dependence on the other entries:
# with `G = det(A) * W` and `W = inv(A)'`, `dG = dot(G, dA) * W - det(A) * W * dA' * W`. At a
# singular `A` it comes from the entries of `G` instead, see `_cofactor_derivative`.
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{typeof(_det_and_grad),DenseMatOrAdj{P}} where {P<:BlasRealFloat}
)
function frule!!(
    ::Dual{typeof(_det_and_grad)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    d, G = _det_and_grad(A)
    dd = dot(G, dA)
    F = lu(Matrix{P}(A); check=false)
    issuccess(F) || return Dual((d, G), (dd, _cofactor_derivative(A, dA)))
    W = ldiv!(F', Matrix{P}(I, size(A)))
    return Dual((d, G), (dd, dd .* W .- d .* (W * dA' * W)))
end

# The derivative of `G`, the transposed adjugate, along `dA`. Its entries are the cofactors
# `(-1)^(i + j) * det(M)`, `M` being `A` without row `i` and column `j`, so their derivatives
# are `(-1)^(i + j) * dot(∇det(M), dM)`.
function _cofactor_derivative(A::AbstractMatrix{P}, dA::AbstractMatrix{P}) where {P}
    n = size(A, 1)
    dG = zeros(P, n, n)
    n == 1 && return dG
    for j in 1:n, i in 1:n
        rows, cols = [1:(i - 1); (i + 1):n], [1:(j - 1); (j + 1):n]
        _, GM = _det_and_grad(A[rows, cols])
        dG[i, j] = (-1)^(i + j) * dot(GM, view(dA, rows, cols))
    end
    return dG
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

# `logabsdet(A)` and the gradient of its first output, `inv(A)'`. That gradient is unbounded
# at a singular `A`, so the rules throw there if a derivative is needed.
function _logabsdet_and_grad(A::AbstractMatrix{P}) where {P}
    F = _square_factor(A)
    ld, s = logabsdet(F)
    return ld, s, iszero(s) ? zeros(P, size(A)) : ldiv!(F', Matrix{P}(I, size(A)))
end
function _check_logabsdet(s, dy)
    (iszero(s) && !iszero(dy)) || return nothing
    throw(ArgumentError("Mooncake cannot differentiate `logabsdet` of a singular matrix."))
end

# As for `_det_and_grad`: with `G = inv(A)'`, `dG = -G * dA' * G`.
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{typeof(_logabsdet_and_grad),DenseMatOrAdj{P}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(_logabsdet_and_grad)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s, G = _logabsdet_and_grad(A)
    return Dual((ld, s, G), (dot(G, dA), zero(P), -G * dA' * G))
end

@is_primitive MinimalCtx Tuple{typeof(logabsdet),DenseMatOrAdj{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Dual{typeof(logabsdet)}, A_dA::Dual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s, G = _logabsdet_and_grad(A)
    _check_logabsdet(s, dA)
    return Dual((ld, s), (dot(G, dA), zero(P)))
end
function rrule!!(
    ::CoDual{typeof(logabsdet)}, A_dA::CoDual{<:DenseMatOrAdj{P}}
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    ld, s, G = _logabsdet_and_grad(A)
    function logabsdet_pb!!(ȳ::Tuple{P,P})
        _check_logabsdet(s, ȳ[1])
        dA .+= ȳ[1] .* G
        return NoRData(), NoRData()
    end
    return CoDual((ld, s), NoFData()), logabsdet_pb!!
end

# For a non-square `A`, `A \ B` is `pinv(A) * B` (least squares or minimum norm). The rules
# work with whichever of `A` and `A'` is tall, `M`, through the pivoted QR of `M`. At a
# rank-deficient `A` the solution, which LinearAlgebra truncates by rank, is not
# differentiable, so they throw.
function _tall_qr(M::AbstractMatrix{P}) where {P}
    F = qr!(Matrix{P}(M), ColumnNorm())
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
    y = R \ (R' \ view(x, F.p, :))
    out = similar(y)
    out[F.p, :] .= y
    return out
end

# `M⁺ x` and `M⁺' y` (`M⁺ = (M'M) \ M'`) for the pivoted QR `F` of a tall `M` with full rank,
# as `R \ (Q'x)` and `Q (R' \ y)`, which don't square the condition number of `M`.
function _pinv_ldiv(F, x)
    R = UpperTriangular(F.R)
    y = R \ view(_q_mul!('T', F, copy(x)), axes(R, 1), :)
    out = similar(y)
    out[F.p, :] .= y
    return out
end
function _pinv_adj_ldiv(F, y::AbstractMatrix{P}) where {P}
    R = UpperTriangular(F.R)
    z = zeros(P, size(F.factors, 1), size(y, 2))
    z[axes(R, 1), :] .= R' \ y[F.p, :]
    return _q_mul!('N', F, z)
end
# `Q * x` or `Q' * x`, in place: `F.Q` isn't type-stable on Julia 1.10.
_q_mul!(trans, F, x) = LAPACK.ormqr!('L', trans, getfield(F, :factors), getfield(F, :τ), x)

@is_primitive(
    MinimalCtx,
    Tuple{typeof(\),DenseMatOrAdj{P},DenseVecOrMatArg{P}} where {P<:BlasRealFloat},
)
function frule!!(
    ::Dual{typeof(\)}, A_dA::Dual{<:DenseMatOrAdj{P}}, B_dB::Dual{<:DenseVecOrMatArg{P}}
) where {P<:BlasRealFloat}
    A, dA = map(_dense, arrayify(A_dA))
    B, dB = matrixify(B_dB)
    m, n = size(A)
    size(B, 1) == m ||
        throw(DimensionMismatch("`A` and `B` must have the same number of rows."))
    if m == n
        F = lu!(Matrix{P}(A))
        X = F \ B
        dX = F \ (dB - dA * X)
    elseif m > n
        F = _tall_qr(A)
        X = F \ B
        dX = _pinv_ldiv(F, dB - dA * X) + _gram_ldiv(F, dA' * (B - A * X))
    else
        F = _tall_qr(A')
        X = _pinv_adj_ldiv(F, B)
        W = dA' * _gram_ldiv(F, B)
        dX = W + _pinv_adj_ldiv(F, dB - dA * X - A * W)
    end
    return Dual(_like(primal(B_dB), X), _like(primal(B_dB), dX))
end

# Undo `matrixify(B_dB)`, which avoids type-unstable matrix-vector products on Julia 1.10.
_like(::AbstractVector, X::Matrix) = vec(X)
_like(::AbstractMatrix, X::Matrix) = X
function rrule!!(
    ::CoDual{typeof(\)},
    A_dA::CoDual{<:DenseMatOrAdj{P}},
    B_dB::CoDual{<:DenseVecOrMatArg{P}},
) where {P<:BlasRealFloat}
    A, dA = arrayify(A_dA)
    A = _dense(A)
    B, dB = matrixify(B_dB)
    m, n = size(A)
    size(B, 1) == m ||
        throw(DimensionMismatch("`A` and `B` must have the same number of rows."))
    X, dX, pb = m == n ? _ldiv_square(A, dA, B, dB) : _ldiv_nonsquare(A, dA, B, dB)
    return CoDual(_like(primal(B_dB), X), _like(primal(B_dB), dX)), pb
end

# `C .+= α .* (U * V)`. For an `Adjoint` or `Transpose` `C` (real here), this writes
# `α .* (V' * U')` into its parent: 5-argument `mul!` into one isn't type-stable on Julia 1.10.
_addmul!(C, U, V, α) = mul!(C, U, V, α, true)
_addmul!(C::Union{Adjoint,Transpose}, U, V, α) = (_addmul!(parent(C), V', U', α); C)
_addmul!(C::Base.ReshapedArray, U, V, α) = (C .+= α .* (U * V); C)

# Products with a reshaped non-strided array aren't type-stable on Julia 1.10, so such an
# array is copied before it is multiplied.
_dense(x::Base.ReshapedArray) = collect(x)
_dense(x::Adjoint) = adjoint(_dense(parent(x)))
_dense(x::Transpose) = transpose(_dense(parent(x)))
_dense(x) = x

# Separate functions, so that each pullback captures concretely typed values: an LU for
# square `A`, a pivoted QR of the tall one of `A` and `A'` otherwise.
function _ldiv_square(A, dA, B, dB::AbstractVecOrMat{P}) where {P}
    F = lu!(Matrix{P}(A))
    X = F \ B
    dX = zero(X)
    function ldiv_square_pb!!(::NoRData)
        B̄ = F' \ dX
        _addmul!(dA, B̄, X', -one(P))
        dB .+= B̄
        return NoRData(), NoRData(), NoRData()
    end
    return X, dX, ldiv_square_pb!!
end
function _ldiv_nonsquare(A, dA, B, dB::AbstractVecOrMat{P}) where {P}
    m, n = size(A)
    F = _tall_qr(m > n ? A : A')
    Y = m < n ? _gram_ldiv(F, B) : Matrix{P}(undef, 0, 0)
    X = m > n ? F \ B : _pinv_adj_ldiv(F, B)
    dX = zero(X)
    function ldiv_nonsquare_pb!!(::NoRData)
        if m > n
            Z = _gram_ldiv(F, dX)
            B̄ = _pinv_adj_ldiv(F, dX)
            _addmul!(dA, B̄, X', -one(P))
            _addmul!(dA, B - A * X, Z', one(P))
        else
            B̄ = _pinv_ldiv(F, dX)
            _addmul!(dA, B̄, X', -one(P))
            _addmul!(dA, Y, dX' - B̄' * A, one(P))
        end
        dB .+= B̄
        return NoRData(), NoRData(), NoRData()
    end
    return X, dX, ldiv_nonsquare_pb!!
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
        # Float32 det finite differences are inaccurate;
        # see the Symmetric `det` cases in src/rules/lapack.jl.
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
        push!(cases, (false, :stability, nothing, \, R', randn(rng, P, 4)))
        push!(cases, (false, :stability, nothing, \, R, view(randn(rng, P, 6), 2:5)))
        push!(cases, (false, :stability, nothing, \, R, transpose(randn(rng, P, 2, 4))))
        push!(cases, (false, :stability, nothing, logabsdet, transpose(A)))
        push!(cases, (false, :stability, nothing, \, A', randn(rng, P, 2, N)'))
        return cases
    end
    # Non-square \ (least squares and minimum norm). Finite differences are too inaccurate for
    # Float32 here, so those cases only check the interface.
    ldiv_cases = map_prod([(5, 3), (3, 5)], Ps) do ((m, n), P)
        flags = (P == Float32, :stability, nothing)
        return Any[
            (flags..., \, randn(rng, P, m, n), randn(rng, P, m, k...)) for k in ((), (2,))
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
