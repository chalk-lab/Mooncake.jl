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

# Performance issue: https://github.com/chalk-lab/Mooncake.jl/issues/526

# Matrix × Matrix kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Matrix{<:IEEEFloat},Matrix{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, A::Dual{<:Matrix{P}}, B::Dual{<:Matrix{P}}
) where {P<:IEEEFloat}
    # Forward rule: d(kron(A, B)) = kron(dA, B) + kron(A, dB)
    primal_A, tangent_A = primal(A), tangent(A)
    primal_B, tangent_B = primal(B), tangent(B)

    primal_result = kron(primal_A, primal_B)
    tangent_result = kron(tangent_A, primal_B) + kron(primal_A, tangent_B)

    return Dual(primal_result, tangent_result)
end

function rrule!!(
    ::CoDual{typeof(kron)}, A::CoDual{<:Matrix{P}}, B::CoDual{<:Matrix{P}}
) where {P<:IEEEFloat}
    primal_A, primal_B = A.x, B.x
    dA, dB = A.dx, B.dx

    function kron_pb!!(dy)
        # Handle NoRData case
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1, n1 = size(primal_A)
        m2, n2 = size(primal_B)

        # For dA: each element A[i,j] affects block dy[(i-1)*m2+1:i*m2, (j-1)*n2+1:j*n2] 
        # The contribution is A[i,j] * B, so dA[i,j] += sum(dy[block] .* B)
        for i in 1:m1, j in 1:n1
            block_rows = ((i - 1) * m2 + 1):(i * m2)
            block_cols = ((j - 1) * n2 + 1):(j * n2)
            dA[i, j] += sum(view(dy, block_rows, block_cols) .* primal_B)
        end

        # For dB: each element B[k,l] appears in all blocks, multiplied by A[i,j]
        # So dB[k,l] += sum over all i,j: A[i,j] * dy[block_i_j][k,l]
        for k in 1:m2, l in 1:n2
            for i in 1:m1, j in 1:n1
                block_row = (i-1)*m2 + k
                block_col = (j-1)*n2 + l
                dB[k, l] += primal_A[i, j] * dy[block_row, block_col]
            end
        end

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(primal_A, primal_B)), kron_pb!!
end

# Vector × Vector kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Vector{<:IEEEFloat},Vector{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, a::Dual{<:Vector{P}}, b::Dual{<:Vector{P}}
) where {P<:IEEEFloat}
    # Convert to matrices and use matrix kron, then vectorize
    primal_a, tangent_a = primal(a), tangent(a)
    primal_b, tangent_b = primal(b), tangent(b)
    
    # kron(a, b) = vec(kron(reshape(a, :, 1), reshape(b, :, 1)))
    a_mat = reshape(primal_a, :, 1)
    b_mat = reshape(primal_b, :, 1)
    da_mat = reshape(tangent_a, :, 1)
    db_mat = reshape(tangent_b, :, 1)
    
    primal_result = vec(kron(a_mat, b_mat))
    tangent_result = vec(kron(da_mat, b_mat) + kron(a_mat, db_mat))
    
    return Dual(primal_result, tangent_result)
end

function rrule!!(
    ::CoDual{typeof(kron)}, a::CoDual{<:Vector{P}}, b::CoDual{<:Vector{P}}
) where {P<:IEEEFloat}
    primal_a, primal_b = a.x, b.x
    da, db = a.dx, b.dx

    function kron_vec_pb!!(dy)
        # Handle NoRData case
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        # Convert vectors to column matrices
        m1, m2 = length(primal_a), length(primal_b)
        # dy is a vector of length m1*m2, reshape to matrix form
        dy_mat = reshape(dy, m1 * m2, 1)
        
        # For da: each element a[i] contributes to dy[(i-1)*m2+1:i*m2]
        for i in 1:m1
            block_rows = ((i - 1) * m2 + 1):(i * m2)
            da[i] += sum(dy_mat[block_rows, 1] .* primal_b)
        end
        
        # For db: each element b[j] appears at positions j, m2+j, 2*m2+j, etc.
        for j in 1:m2
            for i in 1:m1
                index = (i-1)*m2 + j  # Position where a[i]*b[j] appears
                db[j] += primal_a[i] * dy[index]
            end
        end

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(vec(kron(reshape(primal_a, :, 1), reshape(primal_b, :, 1)))), kron_vec_pb!!
end

# Vector × Matrix kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Vector{<:IEEEFloat},Matrix{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, a::Dual{<:Vector{P}}, B::Dual{<:Matrix{P}}
) where {P<:IEEEFloat}
    # kron(a, B) = kron(reshape(a, :, 1), B)
    primal_a, tangent_a = primal(a), tangent(a)
    primal_B, tangent_B = primal(B), tangent(B)
    
    a_mat = reshape(primal_a, :, 1)
    da_mat = reshape(tangent_a, :, 1)
    
    primal_result = kron(a_mat, primal_B)
    tangent_result = kron(da_mat, primal_B) + kron(a_mat, tangent_B)
    
    return Dual(primal_result, tangent_result)
end

function rrule!!(
    ::CoDual{typeof(kron)}, a::CoDual{<:Vector{P}}, B::CoDual{<:Matrix{P}}
) where {P<:IEEEFloat}
    primal_a, primal_B = a.x, B.x
    da, dB = a.dx, B.dx

    function kron_vec_mat_pb!!(dy)
        # Handle NoRData case
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1 = length(primal_a)
        m2, n2 = size(primal_B)
        
        # For da: each element a[i] affects block dy[(i-1)*m2+1:i*m2, :]
        for i in 1:m1
            block_rows = ((i - 1) * m2 + 1):(i * m2)
            da[i] += sum(view(dy, block_rows, :) .* primal_B)
        end
        
        # For dB: each element B[k,l] appears in all blocks, multiplied by a[i]
        for k in 1:m2, l in 1:n2
            for i in 1:m1
                block_row = (i-1)*m2 + k
                dB[k, l] += primal_a[i] * dy[block_row, l]
            end
        end

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(reshape(primal_a, :, 1), primal_B)), kron_vec_mat_pb!!
end

# Matrix × Vector kron
@is_primitive(DefaultCtx, Tuple{typeof(kron),Matrix{<:IEEEFloat},Vector{<:IEEEFloat}})
function frule!!(
    ::Dual{typeof(kron)}, A::Dual{<:Matrix{P}}, b::Dual{<:Vector{P}}
) where {P<:IEEEFloat}
    # kron(A, b) = kron(A, reshape(b, :, 1))
    primal_A, tangent_A = primal(A), tangent(A)
    primal_b, tangent_b = primal(b), tangent(b)
    
    b_mat = reshape(primal_b, :, 1)
    db_mat = reshape(tangent_b, :, 1)
    
    primal_result = kron(primal_A, b_mat)
    tangent_result = kron(tangent_A, b_mat) + kron(primal_A, db_mat)
    
    return Dual(primal_result, tangent_result)
end

function rrule!!(
    ::CoDual{typeof(kron)}, A::CoDual{<:Matrix{P}}, b::CoDual{<:Vector{P}}
) where {P<:IEEEFloat}
    primal_A, primal_b = A.x, b.x
    dA, db = A.dx, b.dx

    function kron_mat_vec_pb!!(dy)
        # Handle NoRData case
        dy isa NoRData && return NoRData(), NoRData(), NoRData()
        
        m1, n1 = size(primal_A)
        m2 = length(primal_b)
        
        # For dA: each element A[i,j] affects block dy[(i-1)*m2+1:i*m2, j]
        for i in 1:m1, j in 1:n1
            block_rows = ((i - 1) * m2 + 1):(i * m2)
            dA[i, j] += sum(view(dy, block_rows, j) .* primal_b)
        end
        
        # For db: each element b[k] appears in all blocks, multiplied by A[i,j]
        for k in 1:m2
            for i in 1:m1, j in 1:n1
                block_row = (i-1)*m2 + k
                db[k] += primal_A[i, j] * dy[block_row, j]
            end
        end

        return NoRData(), NoRData(), NoRData()
    end

    return zero_fcodual(kron(primal_A, reshape(primal_b, :, 1))), kron_mat_vec_pb!!
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
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., kron, randn(rng, P, sz_A...), randn(rng, P, sz_B...))
        end,

        # kron(a, b) - Vector × Vector
        map_prod(
            [(3, 4), (2, 5), (4, 3)], precisions
        ) do ((sz_a, sz_b), P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., kron, randn(rng, P, sz_a), randn(rng, P, sz_b))
        end,

        # kron(a, B) - Vector × Matrix
        map_prod(
            [((3,), (2, 3)), ((4,), (3, 2)), ((2,), (4, 2))], precisions
        ) do ((sz_a, sz_B), P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., kron, randn(rng, P, sz_a...), randn(rng, P, sz_B...))
        end,

        # kron(A, b) - Matrix × Vector  
        map_prod(
            [((2, 3), (3,)), ((3, 2), (4,)), ((4, 2), (2,))], precisions
        ) do ((sz_A, sz_b), P)
            flags = (P == Float16 ? true : false, :stability_and_allocs, nothing)
            return (flags..., kron, randn(rng, P, sz_A...), randn(rng, P, sz_b...))
        end,
    )
    memory = Any[]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:performance_patches}) = Any[], Any[]
