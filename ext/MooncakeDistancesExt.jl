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
column_dots(A, dA) = map(dot, eachcol(A), eachcol(dA))

# Both weight sums in one pass over `W`. Reducing `W` itself rather than its transpose also
# avoids `sum(::Transpose; dims)`, which on Julia 1.10 is type-unstable, routing through
# `LinearAlgebra.switch_dim12` and `PermutedDimsArray`.
function row_and_column_sums(W::AbstractMatrix{P}) where {P}
    rows = zeros(P, size(W, 1))
    cols = Vector{P}(undef, size(W, 2))
    @inbounds for j in axes(W, 2)
        total = zero(P)
        @simd for i in axes(W, 1)
            w = W[i, j]
            rows[i] += w
            total += w
        end
        cols[j] = total
    end
    return rows, cols
end

# Accumulate the cotangent reaching the columns of `pA` and of `pB` through the weights `W`,
# which pair the two sets against each other. `dA === dB` recovers the one-argument case.
function sqdist_accumulate!(dA, dB, pA, pB, W)
    P = eltype(dA)
    rows, cols = row_and_column_sums(W)
    dA .+= 2 .* transpose(rows) .* pA
    mul!(dA, pB, transpose(W), -P(2), one(P))
    dB .+= 2 .* transpose(cols) .* pB
    mul!(dB, pA, W, -P(2), one(P))
    return nothing
end

# `Euclidean` is `sqrt` of the squared distance, so its derivative carries a `1 / 2R`
# factor that is singular where two observations coincide. Distances' own convention is to
# use 1 in place of `1 / 0`, whose contribution then cancels between the scaling and the
# matrix-multiply terms. The one-argument diagonal is zeroed outright rather than left to
# cancel, so only coincident off-diagonal pairs rely on that.
normalise(x::Real, nrm::Real) = iszero(nrm) && !isnan(x) ? one(x / nrm) : x / nrm

# Convert a squared-distance pushforward into a Euclidean one. `dV == 0` wherever `R == 0`,
# so zero is the natural reading of the cusp here, rather than the 1 that `normalise` takes:
# the two directions disagree at coincident observations, where no derivative exists.
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

# A lane of a width-`N` slot is a stride-`N` view, which the pointer-based BLAS wrappers misread
# above width 1, so each lane is gathered into dense scratch and written back. The scratch is
# hoisted, and the in-place primal runs once above the loop.
function frule!!(
    ::Lifted{typeof(Distances._pairwise!),N},
    dist::Lifted{M,N},
    r::Lifted{<:StridedMatrix{P},N},
    A::Lifted{<:StridedMatrix{P},N},
) where {N,M<:PairwiseMetric,P<:BlasRealFloat}
    pdist = primal(dist)
    pr, drs = arrayify(r)
    pA, dAs = arrayify(A)
    Distances._pairwise!(pdist, pr, pA)
    sA, sr = similar(pA), similar(pr)
    for k in 1:N
        copyto!(sA, dAs[k])
        dots = column_dots(pA, sA)

        # `dA' * pA + pA' * dA` is symmetric, so one `syr2k!` over a single triangle does the
        # work of the two `gemm!`s the two-argument form needs.
        BLAS.syr2k!('U', 'T', one(P), sA, pA, zero(P), sr)

        # Mirror that triangle into the full matrix. `_pairwise!` zeroes the diagonal of the
        # one-argument form exactly, so its derivative there is zero rather than the
        # cancelling-but-inexact value the identity gives.
        @inbounds for j in axes(sr, 2)
            sr[j, j] = zero(P)
            for i in 1:(j - 1)
                v = 2 * (dots[i] + dots[j] - sr[i, j])
                sr[i, j] = v
                sr[j, i] = v
            end
        end
        euclidean_pushforward!(pdist, sr, pr)
        copyto!(drs[k], sr)
    end
    return r
end

function frule!!(
    ::Lifted{typeof(Distances._pairwise!),N},
    dist::Lifted{M,N},
    r::Lifted{<:StridedMatrix{P},N},
    A::Lifted{<:StridedMatrix{P},N},
    B::Lifted{<:StridedMatrix{P},N},
) where {N,M<:PairwiseMetric,P<:BlasRealFloat}
    pdist = primal(dist)
    pr, drs = arrayify(r)
    pA, dAs = arrayify(A)
    pB, dBs = arrayify(B)
    Distances._pairwise!(pdist, pr, pA, pB)
    sA, sB, sr = similar(pA), similar(pB), similar(pr)
    for k in 1:N
        copyto!(sA, dAs[k])
        copyto!(sB, dBs[k])
        mul!(sr, transpose(sA), pB)
        mul!(sr, transpose(pA), sB, one(P), one(P))
        sr .= 2 .* (column_dots(pA, sA) .+ transpose(column_dots(pB, sB)) .- sr)
        euclidean_pushforward!(pdist, sr, pr)
        copyto!(drs[k], sr)
    end
    return r
end

# Cotangent of the squared distances, given the cotangent `dr` of the metric's own output.
# `SqEuclidean` needs no conversion, so it also needs no temporary; `Euclidean`'s
# elementwise `1 / 2R` cannot fold into the matrix multiplies and forces one. `R` is the
# rule's own output buffer, read before the pullback restores it.
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
    function _pairwise!_pb!!(::NoRData)
        # The primal's diagonal is a structural zero, so its cotangent must not reach `A`.
        # This has to happen after the conversion: `normalise(0, 0)` is 1, not 0.
        W = sqdist_cotangent(pdist, dr, pr)
        W[diagind(W)] .= zero(P)
        sqdist_accumulate!(dA, dA, pA, pA, W)
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
    function _pairwise!_pb!!(::NoRData)
        W = sqdist_cotangent(pdist, dr, pr)
        sqdist_accumulate!(dA, dB, pA, pB, W)
        copyto!(pr, old_pr)
        fill!(dr, zero(P))
        return NoRData(), zero_rdata(pdist), NoRData(), NoRData(), NoRData()
    end
    return r, _pairwise!_pb!!
end

end
