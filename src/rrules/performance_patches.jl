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
# Matrix × Matrix kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Matrix{<:IEEEFloat},Matrix{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, A_dual::Dual{<:Matrix{P}}, B_dual::Dual{<:Matrix{P}}
) where {P<:IEEEFloat}
    A, dA = primal(A_dual), tangent(A_dual)
    B, dB = primal(B_dual), tangent(B_dual)

    y = kron(A, B)
    dy = kron(dA, B) + kron(A, dB)

    return Dual(y, dy)
end

function rrule!!(
    ::CoDual{typeof(kron)}, A_codual::CoDual{<:Matrix{P}}, B_codual::CoDual{<:Matrix{P}}
) where {P<:IEEEFloat}
    A, B = A_codual.x, B_codual.x
    ∇A, ∇B = A_codual.dx, B_codual.dx # Use ∇ to denote gradients (cotangents)

    function kron_pb!!(dy)
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1, n1 = size(A)
        m2, n2 = size(B)

        dy_tensor = permutedims(reshape(dy, (m2, m1, n2, n1)), (2, 4, 1, 3))
        dy_flat_blocks = reshape(dy_tensor, (m1 * n1, m2 * n2))

        LinearAlgebra.mul!(vec(∇A), dy_flat_blocks, vec(B), 1, 1)
        LinearAlgebra.mul!(vec(∇B), dy_flat_blocks', vec(A), 1, 1)

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(A, B)), kron_pb!!
end

# Vector × Vector kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Vector{<:IEEEFloat},Vector{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, a_dual::Dual{<:Vector{P}}, b_dual::Dual{<:Vector{P}}
) where {P<:IEEEFloat}
    a, da = primal(a_dual), tangent(a_dual)
    b, db = primal(b_dual), tangent(b_dual)
    
    y = kron(a, b)
    dy = kron(da, b) + kron(a, db)
    
    return Dual(y, dy)
end

function rrule!!(
    ::CoDual{typeof(kron)}, a_codual::CoDual{<:Vector{P}}, b_codual::CoDual{<:Vector{P}}
) where {P<:IEEEFloat}
    a, b = a_codual.x, b_codual.x
    ∇a, ∇b = a_codual.dx, b_codual.dx # Use ∇ for gradients

    function kron_vec_pb!!(dy)
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1 = length(a)
        m2 = length(b)
        
        dy_reshaped = reshape(dy, (m2, m1))

        LinearAlgebra.mul!(∇a, dy_reshaped', b, 1, 1)
        LinearAlgebra.mul!(∇b, dy_reshaped, a, 1, 1)

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(a, b)), kron_vec_pb!!
end

# Vector × Matrix kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Vector{<:IEEEFloat},Matrix{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, a_dual::Dual{<:Vector{P}}, B_dual::Dual{<:Matrix{P}}
) where {P<:IEEEFloat}
    a, da = primal(a_dual), tangent(a_dual)
    B, dB = primal(B_dual), tangent(B_dual)
    
    y = kron(a, B)
    dy = kron(da, B) + kron(a, dB)
    
    return Dual(y, dy)
end

function rrule!!(
    ::CoDual{typeof(kron)}, a_codual::CoDual{<:Vector{P}}, B_codual::CoDual{<:Matrix{P}}
) where {P<:IEEEFloat}
    a, B = a_codual.x, B_codual.x
    ∇a, ∇B = a_codual.dx, B_codual.dx # Use ∇ for gradients

    function kron_vec_mat_pb!!(dy)
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1 = length(a)
        m2, n2 = size(B)

        dy_flat_blocks = reshape(permutedims(reshape(dy, (m2, m1, n2)), (2, 1, 3)), (m1, m2 * n2))
        LinearAlgebra.mul!(∇a, dy_flat_blocks, vec(B), 1, 1)
        
        dy_flat_cols = reshape(permutedims(reshape(dy, (m2, m1, n2)), (1, 3, 2)), (m2 * n2, m1))
        LinearAlgebra.mul!(vec(∇B), dy_flat_cols, a, 1, 1)

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(a, B)), kron_vec_mat_pb!!
end

# Matrix × Vector kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Matrix{<:IEEEFloat},Vector{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, A_dual::Dual{<:Matrix{P}}, b_dual::Dual{<:Vector{P}}
) where {P<:IEEEFloat}
    A, dA = primal(A_dual), tangent(A_dual)
    b, db = primal(b_dual), tangent(b_dual)
    
    y = kron(A, b)
    dy = kron(dA, b) + kron(A, db)
    
    return Dual(y, dy)
end

function rrule!!(
    ::CoDual{typeof(kron)}, A_codual::CoDual{<:Matrix{P}}, b_codual::CoDual{<:Vector{P}}
) where {P<:IEEEFloat}
    A, b = A_codual.x, b_codual.x
    ∇A, ∇b = A_codual.dx, b_codual.dx # Use ∇ for gradients

    function kron_mat_vec_pb!!(dy)
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1, n1 = size(A)
        m2 = length(b)

        dy_flat_rows = reshape(permutedims(reshape(dy, (m2, m1, n1)), (2, 3, 1)), (m1 * n1, m2))
        LinearAlgebra.mul!(vec(∇A), dy_flat_rows, b, 1, 1)
        
        dy_flat_cols = reshape(dy, (m2, m1 * n1))
        LinearAlgebra.mul!(∇b, dy_flat_cols, vec(A), 1, 1)

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(A, b)), kron_mat_vec_pb!!
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:performance_patches})
    rng = rng_ctor(123)
    sizes = [(11,), (11, 3)]
    precisions = [Float64, Float32, Float16]
    test_cases = vcat(
        # sum(x)
        map_prod(sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, randn(rng, P, sz...))
        end,

        # sum(abs2, x)
        map_prod(sizes, precisions) do (sz, P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., sum, abs2, randn(rng, P, sz...))
        end,

        # kron(A, B) - Matrix × Matrix
        map_prod(
            [((2, 2), (3, 3)), ((3, 2), (2, 4)), ((4, 3), (2, 2))], precisions
        ) do ((sz_A, sz_B), P)
            flags = (P == Float16 ? true : false, :stability, nothing)
            return (flags..., kron, randn(rng, P, sz_A...), randn(rng, P, sz_B...))
        end,

        # kron(a, b) - Vector × Vector
        map_prod(
            [(3, 4), (2, 5), (4, 3)], precisions
        ) do ((sz_a, sz_b), P)
            flags = (P == Float16 ? true : false, :stability, nothing)
            return (flags..., kron, randn(rng, P, sz_a), randn(rng, P, sz_b))
        end,

        # kron(a, B) - Vector × Matrix
        map_prod(
            [((3,), (2, 3)), ((4,), (3, 2)), ((2,), (4, 2))], precisions
        ) do ((sz_a, sz_B), P)
            flags = (P == Float16 ? true : false, :stability, nothing)
            return (flags..., kron, randn(rng, P, sz_a...), randn(rng, P, sz_B...))
        end,

        # kron(A, b) - Matrix × Vector  
        map_prod(
            [((2, 2), (3,)), ((2, 3), (3,)), ((3, 2), (4,)), ((4, 2), (2,))], precisions
        ) do ((sz_A, sz_b), P)
            flags = (P == Float16 ? true : false, :stability, nothing)
            return (flags..., kron, randn(rng, P, sz_A...), randn(rng, P, sz_b...))
        end,
    )
    memory = Any[]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:performance_patches}) = Any[], Any[]
