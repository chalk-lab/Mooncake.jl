# All of the code here purely exists to work around current performance limitations of
# Mooncake.jl. In order to prevent this from getting out of hand, there are several
# conventions to which we adhere when writing these rules:
# 1. for each rule, a comment is added containing a link to the issue or issues that are
#   believed to describe the deficiencies of Mooncake.jl which cause the rule to be needed.
# 2. the number of concrete types for which the signature is valid is finite, and all are
#   tested. For example, `Array{<:IEEEFloat}` is a permissible type. The only exception to
#   this is the dimension of an `Array` argument. For example, it is fine to write rules for
#   `Array{Float64}`, despite the fact that this technically includes `Array{Float64,1}`,
#   `Array{Float64,2}`, `Array{Float64,3}`, etc.
#   `Diagonal{<:IEEEFloat}` is not, on the other hand, permissible. This is because we do
#   not know what the type of its `diag` field is, and it _could_ be any `AbstractVector`.
#   Something more precise like `Diagonal{P, Vector{P}} where {P<:IEEEFloat}` is fine.
#   This convention ensures that we are confident the rules here provide a strict
#   improvement over what we currently have, and helps to prevent the addition of flakey
#   rules which cause robustness or correctness problems.

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),Array{<:IEEEFloat}})
function frule!!(::Dual{typeof(sum)}, x::Dual{<:Array{P}}) where {P<:IEEEFloat}
    return Dual(sum(primal(x)), sum(tangent(x)))
end
function rrule!!(::CoDual{typeof(sum)}, x::CoDual{<:Array{P}}) where {P<:IEEEFloat}
    dx = x.dx
    function sum_pb!!(dz::P)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(identity, x.x)), sum_pb!!
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),ContiguousSubVector{<:IEEEFloat}})
function frule!!(::Dual{typeof(sum)}, x::Dual{ContiguousSubVector{P}}) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    return Dual(sum(px), sum(dx))
end
function rrule!!(
    ::CoDual{typeof(sum)}, x::CoDual{ContiguousSubVector{P}}
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    function sum_view_pb!!(dz::P)
        dx .+= dz
        return NoRData(), NoRData()
    end
    return zero_fcodual(sum(px)), sum_view_pb!!
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
@is_primitive(DefaultCtx, Tuple{typeof(sum),typeof(abs2),Array{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(sum)}, ::Dual{typeof(abs2)}, x::Dual{<:Array{P}}
) where {P<:IEEEFloat}
    return Dual(sum(abs2, primal(x)), 2 * dot(primal(x), tangent(x)))
end
function rrule!!(
    ::CoDual{typeof(sum)}, ::CoDual{typeof(abs2)}, x::CoDual{<:Array{P}}
) where {P<:IEEEFloat}
    function sum_abs2_pb!!(dz::P)
        x.dx .+= 2 .* x.x .* dz
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(sum(abs2, x.x)), sum_abs2_pb!!
end

# Without this, `A * B` differentiates through the in-place `gemm!` rule, which copies the
# output buffer so that the pullback can restore it. `*` allocates that buffer fresh, so
# the copy is pure overhead.
@is_primitive DefaultCtx Tuple{typeof(*),Matrix{P},Matrix{P}} where {P<:BlasRealFloat}
function frule!!(
    ::Dual{typeof(*)}, A::Dual{<:Matrix{P}}, B::Dual{<:Matrix{P}}
) where {P<:BlasRealFloat}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    C = pA * pB
    dC = dA * pB
    mul!(dC, pA, dB, one(P), one(P))
    return Dual(C, dC)
end
function rrule!!(
    ::CoDual{typeof(*)}, A::CoDual{<:Matrix{P}}, B::CoDual{<:Matrix{P}}
) where {P<:BlasRealFloat}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    C = pA * pB
    dC = zero(C)
    function matmul_pb!!(::NoRData)
        mul!(dA, dC, transpose(pB), one(P), one(P))
        mul!(dB, transpose(pA), dC, one(P), one(P))
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(C, dC), matmul_pb!!
end

# Differentiating the generic implementation element by element costs more than the rules
# either side of it; `Distances.pairwise(...; dims=1)` reaches it on every call.
@is_primitive DefaultCtx Tuple{typeof(permutedims),Matrix{P}} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(permutedims)}, x::Dual{<:Matrix{P}}) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    return Dual(permutedims(px), permutedims(dx))
end
function rrule!!(::CoDual{typeof(permutedims)}, x::CoDual{<:Matrix{P}}) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    y = permutedims(px)
    dy = zero(y)
    function permutedims_pb!!(::NoRData)
        dx .+= transpose(dy)
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), permutedims_pb!!
end

@is_primitive DefaultCtx Tuple{
    typeof(permutedims),Array{P,N},NTuple{N,Int}
} where {P<:IEEEFloat,N}
function frule!!(
    ::Dual{typeof(permutedims)}, x::Dual{<:Array{P,N}}, perm::Dual{<:NTuple{N,Int}}
) where {P<:IEEEFloat,N}
    px, dx = arrayify(x)
    pperm = primal(perm)
    return Dual(permutedims(px, pperm), permutedims(dx, pperm))
end
function rrule!!(
    ::CoDual{typeof(permutedims)}, x::CoDual{<:Array{P,N}}, perm::CoDual{<:NTuple{N,Int}}
) where {P<:IEEEFloat,N}
    px, dx = arrayify(x)
    pperm = primal(perm)
    y = permutedims(px, pperm)
    dy = zero(y)
    iperm = invperm(pperm)
    function permutedims_pb!!(::NoRData)
        dx .+= permutedims(dy, iperm)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), permutedims_pb!!
end

# Matrices only: `kron(::Vector, ::Vector)` returns a `Vector`, which these rules would
# widen to a `Matrix` by matrixifying their factors.
const KronFactor{T} = Union{
    StridedMatrix{T},
    UpperTriangular{T,<:StridedMatrix{T}},
    LowerTriangular{T,<:StridedMatrix{T}},
}

# Both `kron` pullbacks contract `dy` against one factor to accumulate into the other. The
# contraction is dense, so it goes through `densify` / `accumulate_densified!`. Read
# as `P x M x Q x N`, both contractions read the same element of `dy`, so a single pass in
# memory order serves both. A `gemv` per `(q, n)` block is slower at every shape measured:
# the blocks are small enough that BLAS call overhead dominates.
function _kron_pb!(dx1, dx2, dy, px1, px2)
    T = eltype(px1)
    M, N = size(px1)
    P, Q = size(px2)
    W = reshape(dy, P, M, Q, N)
    t1 = densify(dx1)
    t2 = densify(dx2)
    @inbounds for n in 1:N, q in 1:Q, i in 1:M
        acc = zero(T)
        x1 = px1[i, n]
        @simd for k in 1:P
            w = W[k, i, q, n]
            acc += w * px2[k, q]
            t2[k, q] += w * x1
        end
        t1[i, n] += acc
    end
    accumulate_densified!(dx1, t1)
    accumulate_densified!(dx2, t2)
    return nothing
end

# https://github.com/chalk-lab/Mooncake.jl/issues/526
@is_primitive DefaultCtx Tuple{
    typeof(LinearAlgebra._kron!),StridedMatrix{T},KronFactor{T},KronFactor{T}
} where {T<:IEEEFloat}
function Mooncake.frule!!(
    ::Dual{typeof(LinearAlgebra._kron!)},
    out::Dual{<:StridedMatrix{<:T}},
    x1::Dual{<:KronFactor{<:T}},
    x2::Dual{<:KronFactor{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
    LinearAlgebra._kron!(pout, px1, px2)
    # manually compute dout .= kron(dx1, px2) .+ kron(px1, dx2), otherwise performance
    # suffers
    m = firstindex(dout)
    for j in axes(px1, 2), l in axes(px2, 2), i in axes(px1, 1)
        x1ij = px1[i, j]
        dx1ij = dx1[i, j]
        for k in axes(px2, 1)
            dout[m] = (x1ij * dx2[k, l]) + (dx1ij * px2[k, l])
            m += 1
        end
    end
    return out
end
function Mooncake.rrule!!(
    ::CoDual{typeof(LinearAlgebra._kron!)},
    out::CoDual{<:StridedMatrix{<:T}},
    x1::CoDual{<:KronFactor{<:T}},
    x2::CoDual{<:KronFactor{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
    old_pout = copy(pout)
    LinearAlgebra._kron!(pout, px1, px2)
    function _kron!_pb!!(::NoRData)
        _kron_pb!(dx1, dx2, dout, px1, px2)
        copyto!(pout, old_pout)
        fill!(dout, zero(T))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return out, _kron!_pb!!
end

# Using the rule for `_kron!` above makes performance on `kron` better, but still not as
# good as it _could_ be. To maximise performance we need a rule specifically for `kron`
# itself. See https://github.com/chalk-lab/Mooncake.jl/pull/886
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),KronFactor{T},KronFactor{T}
} where {T<:IEEEFloat}
function Mooncake.rrule!!(
    ::CoDual{typeof(kron)}, x1::CoDual{<:KronFactor{<:T}}, x2::CoDual{<:KronFactor{<:T}}
) where {T<:Base.IEEEFloat}
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
    y = kron(px1, px2)
    dy = zero(y)
    function kron_pb!!(::NoRData)
        _kron_pb!(dx1, dx2, dy, px1, px2)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), kron_pb!!
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    sum_sizes = [(11,), (11, 3)]
    precisions = [Float64, Float32, Float16]
    test_cases = vcat(

        # sum(x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, randn(rng, P, sz...))
        end,

        # sum(view(x, a:b))
        map(precisions) do P
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, view(randn(rng, P, 11), 2:9))
        end,

        # sum(abs2, x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, abs2, randn(rng, P, sz...))
        end,

        # _kron!(x, y)
        map(precisions) do (P)
            return (
                true,
                :none,
                nothing,
                LinearAlgebra._kron!,
                zeros(P, 50, 50),
                randn(rng, P, 5, 5),
                randn(rng, P, 10, 10),
            )
        end,

        # permutedims(x)
        map([Float64, Float32]) do P
            return (false, :stability, nothing, permutedims, randn(rng, P, 7, 11))
        end,
        map([Float64, Float32]) do P
            return (
                false,
                :stability,
                nothing,
                permutedims,
                randn(rng, P, 2, 3, 4, 5),
                (3, 1, 2, 4),
            )
        end,

        # x * y
        map([Float64, Float32]) do P
            return (
                false, :stability, nothing, *, randn(rng, P, 7, 11), randn(rng, P, 11, 5)
            )
        end,
    )
    memory = Any[]
    return test_cases, memory
end

_square_matmul(x) = x * x

function derived_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    precisions = [Float64, Float32]
    test_cases = vcat(
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                UpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                UpperTriangular(randn(rng, P, 5, 5)),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                view(randn(rng, P, 5, 5), 1:5, 1:5),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                false,
                :none,
                nothing,
                LinearAlgebra.kron,
                view(randn(rng, P, 5, 5), 1:5, 1:5),
                UpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
        # `A * A` aliases the rule's arguments, so `dA === dB` and the pullback must
        # accumulate both terms into the one array.
        map(precisions) do (P)
            return (false, :none, nothing, _square_matmul, randn(rng, P, 5, 5))
        end,
    )
    memory = Any[]
    return test_cases, memory
end
