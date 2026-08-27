module MooncakeDistancesExt

using Distances, LinearAlgebra, Mooncake
import Mooncake:
    BlasRealFloat,
    CoDual,
    DefaultCtx,
    Dual,
    NoRData,
    @is_primitive,
    arrayify,
    frule!!,
    primal,
    rrule!!,
    zero_rdata

# Distances funnels every `pairwise` and `pairwise!` call for these metrics through
# `_pairwise!`, which always takes observations as columns: `dims == 1` is discharged by a
# `permutedims` one level up. Writing the rules here therefore covers both entry points,
# both metrics, and both `dims` without any `dims` branching.
#
# Distances does ship `ChainRulesCore.rrule`s for `pairwise` with these metrics, but each
# takes a `RuleConfig` first argument and Mooncake has no `RuleConfig` support, so
# `@from_rrule` cannot reach them.
const PairwiseMetric = Union{SqEuclidean,Euclidean}

# The rules differentiate `V[i, j] == dots[i] + dots[j] - 2 * ai'aj`, where `dots` holds
# the squared column norms. `Euclidean` is that composed with `sqrt`.
column_dots(A, dA) = vec(sum(A .* dA; dims=1))

function sqdist_pushforward!(dV, pA, dA, Adots, pB, dB, Bdots)
    P = eltype(dA)
    mul!(dV, transpose(dA), pB)
    mul!(dV, transpose(pA), dB, one(P), one(P))
    dV .= 2 .* (Adots .+ transpose(Bdots) .- dV)
    return dV
end

# Accumulate into `dA` the cotangent reaching the columns of `pA` through the weights `W`,
# which pair them against the columns of `pB`. Passing `transpose(W)` swaps the roles.
# `scales` is the corresponding row sum of `W`; the caller reduces the untransposed matrix
# because on Julia 1.10 `sum(::Transpose; dims)` is type-unstable, routing through
# `LinearAlgebra.switch_dim12` and `PermutedDimsArray`.
function sqdist_accumulate!(dA, pA, pB, W, scales)
    P = eltype(dA)
    dA .+= 2 .* transpose(scales) .* pA
    mul!(dA, pB, transpose(W), -P(2), one(P))
    return nothing
end

# `Euclidean` is `sqrt` of the squared distance, so its derivative carries a `1 / 2R`
# factor that is singular where two observations coincide. Distances' own convention there
# is to use 1 in place of `1 / 0`; on the diagonal, and wherever else `R == 0`, the
# resulting contribution cancels between the scaling and the matrix-multiply terms.
normalise(x::Real, nrm::Real) = iszero(nrm) && !isnan(x) ? one(x / nrm) : x / nrm

# Convert a squared-distance pushforward into a Euclidean one. `dV == 0` wherever `R == 0`,
# so a zero derivative is the right reading of the cusp for the forward direction.
euclidean_pushforward!(::SqEuclidean, dV, R) = dV
function euclidean_pushforward!(::Euclidean, dV, R)
    dV .= ifelse.(iszero.(R), zero(eltype(dV)), dV ./ (2 .* R))
    return dV
end

@is_primitive DefaultCtx Tuple{
    typeof(Distances._pairwise!),M,StridedMatrix{P},StridedMatrix{P}
} where {M<:PairwiseMetric,P<:BlasRealFloat}
@is_primitive DefaultCtx Tuple{
    typeof(Distances._pairwise!),M,StridedMatrix{P},StridedMatrix{P},StridedMatrix{P}
} where {M<:PairwiseMetric,P<:BlasRealFloat}

function frule!!(
    ::Dual{typeof(Distances._pairwise!)},
    dist::Dual{M},
    r::Dual{<:StridedMatrix{P}},
    A::Dual{<:StridedMatrix{P}},
) where {M<:PairwiseMetric,P<:BlasRealFloat}
    pdist = primal(dist)
    pr, dr = arrayify(r)
    pA, dA = arrayify(A)
    Distances._pairwise!(pdist, pr, pA)
    dots = column_dots(pA, dA)

    # `dA' * pA + pA' * dA` is symmetric, so one `syr2k!` over a single triangle does the
    # work of the two `gemm!`s the two-argument form needs.
    BLAS.syr2k!('U', 'T', one(P), dA, pA, zero(P), dr)

    # Mirror that triangle into the full matrix. `_pairwise!` zeroes the diagonal of the
    # one-argument form exactly, so its derivative there is zero rather than the
    # cancelling-but-inexact value the identity gives.
    @inbounds for j in axes(dr, 2)
        dr[j, j] = zero(P)
        for i in 1:(j - 1)
            v = 2 * (dots[i] + dots[j] - dr[i, j])
            dr[i, j] = v
            dr[j, i] = v
        end
    end
    euclidean_pushforward!(pdist, dr, pr)
    return r
end

function frule!!(
    ::Dual{typeof(Distances._pairwise!)},
    dist::Dual{M},
    r::Dual{<:StridedMatrix{P}},
    A::Dual{<:StridedMatrix{P}},
    B::Dual{<:StridedMatrix{P}},
) where {M<:PairwiseMetric,P<:BlasRealFloat}
    pdist = primal(dist)
    pr, dr = arrayify(r)
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    Distances._pairwise!(pdist, pr, pA, pB)
    sqdist_pushforward!(dr, pA, dA, column_dots(pA, dA), pB, dB, column_dots(pB, dB))
    euclidean_pushforward!(pdist, dr, pr)
    return r
end

# Cotangent of the squared distances, given the cotangent `dr` of the metric's own output.
# `SqEuclidean` needs no conversion, so it also needs no temporary; `Euclidean`'s
# elementwise `1 / 2R` cannot fold into the matrix multiplies and forces one.
sqdist_cotangent(::SqEuclidean, dr, R) = dr
sqdist_cotangent(::Euclidean, dr, R) = normalise.(dr, 2 .* R)

function rrule!!(
    ::CoDual{typeof(Distances._pairwise!)},
    dist::CoDual{M},
    r::CoDual{<:StridedMatrix{P}},
    A::CoDual{<:StridedMatrix{P}},
) where {M<:PairwiseMetric,P<:BlasRealFloat}
    pdist = primal(dist)
    pr, dr = arrayify(r)
    pA, dA = arrayify(A)
    old_pr = copy(pr)
    Distances._pairwise!(pdist, pr, pA)
    R = pdist isa Euclidean ? copy(pr) : pr
    function _pairwise!_pb!!(::NoRData)
        # The primal's diagonal is a structural zero, so its cotangent must not reach `A`.
        # This has to happen after the conversion: `normalise(0, 0)` is 1, not 0.
        W = sqdist_cotangent(pdist, dr, R)
        W[diagind(W)] .= zero(P)
        sqdist_accumulate!(dA, pA, pA, W, vec(sum(W; dims=2)))
        sqdist_accumulate!(dA, pA, pA, transpose(W), vec(sum(W; dims=1)))
        copyto!(pr, old_pr)
        fill!(dr, zero(P))
        return NoRData(), zero_rdata(pdist), NoRData(), NoRData()
    end
    return r, _pairwise!_pb!!
end

function rrule!!(
    ::CoDual{typeof(Distances._pairwise!)},
    dist::CoDual{M},
    r::CoDual{<:StridedMatrix{P}},
    A::CoDual{<:StridedMatrix{P}},
    B::CoDual{<:StridedMatrix{P}},
) where {M<:PairwiseMetric,P<:BlasRealFloat}
    pdist = primal(dist)
    pr, dr = arrayify(r)
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    old_pr = copy(pr)
    Distances._pairwise!(pdist, pr, pA, pB)
    R = pdist isa Euclidean ? copy(pr) : pr
    function _pairwise!_pb!!(::NoRData)
        W = sqdist_cotangent(pdist, dr, R)
        sqdist_accumulate!(dA, pA, pB, W, vec(sum(W; dims=2)))
        sqdist_accumulate!(dB, pB, pA, transpose(W), vec(sum(W; dims=1)))
        copyto!(pr, old_pr)
        fill!(dr, zero(P))
        return NoRData(), zero_rdata(pdist), NoRData(), NoRData(), NoRData()
    end
    return r, _pairwise!_pb!!
end

end
