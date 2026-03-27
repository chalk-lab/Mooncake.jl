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
# Stateful rule (issue #403): stores dx reference at call time so the rule struct itself
# acts as the pullback, avoiding a per-call closure allocation.
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

mutable struct SumRuleCallable{P<:IEEEFloat,N}
    dx::Array{P,N}
    SumRuleCallable{P,N}() where {P<:IEEEFloat,N} = new{P,N}()
end
_copy(::SumRuleCallable{P,N}) where {P<:IEEEFloat,N} = SumRuleCallable{P,N}()
@inline function (rule::SumRuleCallable{P,N})(
    ::CoDual{typeof(sum)}, x::CoDual{<:Array{P,N}}
) where {P<:IEEEFloat,N}
    rule.dx = x.dx
    return zero_fcodual(sum(identity, x.x)), rule
end
@inline function (pb::SumRuleCallable{P})(dz::P) where {P<:IEEEFloat}
    pb.dx .+= dz
    return NoRData(), NoRData()
end
function build_primitive_rrule(
    ::Type{T}
) where {P<:IEEEFloat,N,T<:Tuple{typeof(sum),Array{P,N}}}
    return SumRuleCallable{P,N}()
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
# Stateful rule (issue #403): same pattern for sum(abs2, x).
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

mutable struct SumAbs2RuleCallable{P<:IEEEFloat,N}
    x_primal::Array{P,N}
    dx::Array{P,N}
    SumAbs2RuleCallable{P,N}() where {P<:IEEEFloat,N} = new{P,N}()
end
_copy(::SumAbs2RuleCallable{P,N}) where {P<:IEEEFloat,N} = SumAbs2RuleCallable{P,N}()
@inline function (rule::SumAbs2RuleCallable{P,N})(
    ::CoDual{typeof(sum)}, ::CoDual{typeof(abs2)}, x::CoDual{<:Array{P,N}}
) where {P<:IEEEFloat,N}
    rule.x_primal = x.x
    rule.dx = x.dx
    return zero_fcodual(sum(abs2, x.x)), rule
end
@inline function (pb::SumAbs2RuleCallable{P})(dz::P) where {P<:IEEEFloat}
    pb.dx .+= 2 .* pb.x_primal .* dz
    return NoRData(), NoRData(), NoRData()
end
function build_primitive_rrule(
    ::Type{T}
) where {P<:IEEEFloat,N,T<:Tuple{typeof(sum),typeof(abs2),Array{P,N}}}
    return SumAbs2RuleCallable{P,N}()
end

# https://github.com/chalk-lab/Mooncake.jl/issues/526
@is_primitive DefaultCtx Tuple{
    typeof(LinearAlgebra._kron!),AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
function Mooncake.frule!!(
    ::Dual{typeof(LinearAlgebra._kron!)},
    out::Dual{<:AbstractMatrix{<:T}},
    x1::Dual{<:AbstractVecOrMat{<:T}},
    x2::Dual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = matrixify(x1)
    px2, dx2 = matrixify(x2)
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
    out::CoDual{<:AbstractMatrix{<:T}},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = matrixify(x1)
    px2, dx2 = matrixify(x2)
    old_pout = copy(pout)
    LinearAlgebra._kron!(pout, px1, px2)
    function _kron!_pb!!(::NoRData)
        P, Q = size(px2)
        for m in axes(px1, 1), n in axes(px1, 2)
            dx1[m, n] += dot(
                (@view dout[((m - 1) * P + 1):(m * P), ((n - 1) * Q + 1):(n * Q)]), px2
            )
        end
        for p in axes(px2, 1), q in axes(px2, 2)
            dx2[p, q] += dot((@view dout[p:P:end, q:Q:end]), px1)
        end
        copyto!(pout, old_pout)
        fill!(dout, zero(T))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return out, _kron!_pb!!
end

# Using the rule for `_kron!` above makes performance on `kron` better, but still not as
# good as it _could_ be. To maximise performance we need a rule specifically for `kron`
# itself. See https://github.com/chalk-lab/Mooncake.jl/pull/886
# Stateful rule (issue #403): all backward logic lives in KronRuleCallable; rrule!!
# delegates to a fresh instance so there is a single implementation.
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),AbstractMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
mutable struct KronRuleCallable{T<:IEEEFloat}
    px1::Matrix{T}  # collected to plain Matrix so dot() is contiguous-fast
    px2::Matrix{T}
    dx1::AbstractMatrix{T}
    dx2::AbstractMatrix{T}
    dy::Matrix{T}
    KronRuleCallable{T}() where {T<:IEEEFloat} = new{T}()
end
_copy(::KronRuleCallable{T}) where {T} = KronRuleCallable{T}()
@inline function (rule::KronRuleCallable{T})(
    ::CoDual{typeof(kron)},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:IEEEFloat}
    px1, dx1 = matrixify(x1)
    px2, dx2 = matrixify(x2)
    y = kron(px1, px2)
    dy = zero(y)
    rule.px1 = px1 isa Matrix{T} ? px1 : Matrix{T}(px1)
    rule.px2 = px2 isa Matrix{T} ? px2 : Matrix{T}(px2)
    rule.dx1 = dx1
    rule.dx2 = dx2
    rule.dy = dy
    return CoDual(y, dy), rule
end
@inline function (pb::KronRuleCallable{T})(::NoRData) where {T<:IEEEFloat}
    M, N = size(pb.dx1)
    P, Q = size(pb.dx2)
    # dx1[m,n] = ⟨dC_block_mn, B⟩  where dC_block_mn = dy[(m-1)P+1:mP, (n-1)Q+1:nQ]
    # dx2[p,q] = ⟨dy[p:P:end, q:Q:end], A⟩  (strided view; structurally-zero positions
    # receive zero gradient since those entries of dy are zero by construction)
    for m in 1:M, n in 1:N
        pb.dx1[m, n] += dot(
            @view(pb.dy[((m - 1) * P + 1):(m * P), ((n - 1) * Q + 1):(n * Q)]), pb.px2
        )
    end
    for p in 1:P, q in 1:Q
        pb.dx2[p, q] += dot(@view(pb.dy[p:P:end, q:Q:end]), pb.px1)
    end
    return NoRData(), NoRData(), NoRData()
end
# rrule!! delegates to a fresh callable — single implementation of backward logic.
function Mooncake.rrule!!(
    f::CoDual{typeof(kron)},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:IEEEFloat}
    return KronRuleCallable{T}()(f, x1, x2)
end
function build_primitive_rrule(
    ::Type{S}
) where {T<:IEEEFloat,A<:AbstractMatrix{T},B<:AbstractMatrix{T},S<:Tuple{typeof(kron),A,B}}
    return KronRuleCallable{T}()
end

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/156
# Matrix multiply `*` is not a primitive, so Mooncake differentiates through the Julia
# call chain: * → mul! → _mul! → gemm_wrapper! → BLAS.gemm!. This adds block-stack
# overhead for each intermediate call and causes an unnecessary copy of the output matrix
# in the BLAS.gemm! rule (needed for in-place restore semantics but wasteful here).
# A direct rule short-circuits to BLAS and skips both costs.
# Stateful rule (issue #403): stores array references at call time so the struct itself
# acts as the pullback, avoiding a per-call closure allocation.
@is_primitive DefaultCtx Tuple{
    typeof(*),Matrix{P},Matrix{P}
} where {P<:Union{Float32,Float64}}
function frule!!(
    ::Dual{typeof(*)}, A::Dual{<:Matrix{P}}, B::Dual{<:Matrix{P}}
) where {P<:Union{Float32,Float64}}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    C = pA * pB
    dC = dA * pB
    BLAS.gemm!('N', 'N', one(P), pA, dB, one(P), dC)
    return Dual(C, dC)
end
function rrule!!(
    f::CoDual{typeof(*)}, A::CoDual{<:Matrix{P}}, B::CoDual{<:Matrix{P}}
) where {P<:Union{Float32,Float64}}
    return MatMulRuleCallable{P}()(f, A, B)
end

mutable struct MatMulRuleCallable{P<:Union{Float32,Float64}}
    pA::Matrix{P}
    pB::Matrix{P}
    dA::Matrix{P}
    dB::Matrix{P}
    dC::Matrix{P}
    MatMulRuleCallable{P}() where {P<:Union{Float32,Float64}} = new{P}()
end
_copy(::MatMulRuleCallable{P}) where {P} = MatMulRuleCallable{P}()
@inline function (rule::MatMulRuleCallable{P})(
    ::CoDual{typeof(*)}, A::CoDual{<:Matrix{P}}, B::CoDual{<:Matrix{P}}
) where {P<:Union{Float32,Float64}}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    C = pA * pB
    dC = zero(C)
    rule.pA = pA
    rule.pB = pB
    rule.dA = dA
    rule.dB = dB
    rule.dC = dC
    return CoDual(C, dC), rule
end
@inline function (pb::MatMulRuleCallable{P})(::NoRData) where {P<:Union{Float32,Float64}}
    BLAS.gemm!('N', 'T', one(P), pb.dC, pb.pB, one(P), pb.dA)
    BLAS.gemm!('T', 'N', one(P), pb.pA, pb.dC, one(P), pb.dB)
    return NoRData(), NoRData(), NoRData()
end
function build_primitive_rrule(
    ::Type{T}
) where {P<:Union{Float32,Float64},T<:Tuple{typeof(*),Matrix{P},Matrix{P}}}
    return MatMulRuleCallable{P}()
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

        # sum(abs2, x)
        map_prod(sum_sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, abs2, randn(rng, P, sz...))
        end,

        # *(A, B) — matrix multiply: primal allocates a new matrix so only check stability
        map_prod([(5, 7), (7, 5)], [Float64, Float32]) do ((m, k), P)
            n = 4
            return (
                false, :stability, nothing, *, randn(rng, P, m, k), randn(rng, P, k, n)
            )
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
    )
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    precisions = [Float64, Float32]
    test_cases = vcat(
        map(precisions) do (P)
            return (
                true,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                UpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                true,
                :none,
                nothing,
                LinearAlgebra.kron,
                randn(rng, P, 5, 5),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                true,
                :none,
                nothing,
                LinearAlgebra.kron,
                UpperTriangular(randn(rng, P, 5, 5)),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                true,
                :none,
                nothing,
                LinearAlgebra.kron,
                view(randn(rng, P, 5, 5), 1:5, 1:5),
                LowerTriangular(randn(rng, P, 10, 10)),
            )
        end,
        map(precisions) do (P)
            return (
                true,
                :none,
                nothing,
                LinearAlgebra.kron,
                view(randn(rng, P, 5, 5), 1:5, 1:5),
                UpperTriangular(randn(rng, P, 10, 10)),
            )
        end,
    )
    memory = Any[]
    return test_cases, memory
end
