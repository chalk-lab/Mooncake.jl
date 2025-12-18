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

# https://github.com/chalk-lab/Mooncake.jl/issues/526
@is_primitive DefaultCtx Tuple{
    typeof(LinearAlgebra._kron!),AbstractMatrix{T},AbstractMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
function Mooncake.frule!!(
    ::Dual{typeof(LinearAlgebra._kron!)},
    out::Dual{<:AbstractMatrix{<:T}},
    x1::Dual{<:AbstractMatrix{<:T}},
    x2::Dual{<:AbstractMatrix{<:T}},
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
    out::CoDual{<:AbstractMatrix{<:T}},
    x1::CoDual{<:AbstractMatrix{<:T}},
    x2::CoDual{<:AbstractMatrix{<:T}},
) where {T<:Base.IEEEFloat}
    pout, dout = arrayify(out)
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
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
@mooncake_overlay function LinearAlgebra._kron!(
    out::AbstractMatrix{T}, a::AbstractVector{T}, b::AbstractMatrix{T}
) where {T<:IEEEFloat}
    return LinearAlgebra._kron!(out, reshape(a, :, 1), b)::typeof(out)
end
@mooncake_overlay function LinearAlgebra._kron!(
    out::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractVector{T}
) where {T<:IEEEFloat}
    return LinearAlgebra._kron!(out, a, reshape(b, :, 1))::typeof(out)
end

# TODO: move to blas.jl
arrayify(x::UpperTriangular{<:IEEEFloat, <:Matrix{<:IEEEFloat}}, dx::TangentOrFData) = arrayify(x.data, _fields(dx).data)

# Using the rule for `_kron!` above makes performance on `kron` better, but still not as
# good as it _could_ be. To maximise performance we need a rule specifically for `kron`
# itself. See https://github.com/chalk-lab/Mooncake.jl/pull/886
@is_primitive DefaultCtx ReverseMode Tuple{
    typeof(kron),AbstractMatrix{T},AbstractMatrix{T}
} where {T<:IEEEFloat}
function Mooncake.rrule!!(
    ::CoDual{typeof(kron)}, x1::CoDual{<:AbstractMatrix{<:T}}, x2::CoDual{<:AbstractMatrix{<:T}}
) where {T<:Base.IEEEFloat}
    px1, dx1 = arrayify(x1)
    px2, dx2 = arrayify(x2)
    y = kron(px1, px2)
    dy = zero(y)
    function kron_pb!!(::NoRData)
        M, N = size(px1)
        P, Q = size(px2)
        for m in 1:M, n in 1:N
            dx1[m, n] += dot(
                (@view dy[((m - 1) * P + 1):(m * P), ((n - 1) * Q + 1):(n * Q)]), px2
            )
        end
        for p in 1:P, q in 1:Q
            dx2[p, q] += dot((@view dy[p:P:end, q:Q:end]), px1)
        end
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
    )
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:performance_patches}) = Any[], Any[]
