@is_primitive MinimalCtx Tuple{typeof(exp),Matrix{<:IEEEFloat}}

struct ExpPullback{P}
    pb
    Ȳ::Matrix{P}
    X̄::Matrix{P}
end

function (pb::ExpPullback)(::NoRData)
    _, X̄_inc = pb.pb(pb.Ȳ)
    pb.X̄ .+= X̄_inc
    return NoRData(), NoRData()
end

function frule!!(::Dual{typeof(exp)}, X_dX::Dual{Matrix{P}}) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    return Dual(ChainRules.frule((ChainRules.NoTangent(), dX), LinearAlgebra.exp!, X)...)
end
function rrule!!(::CoDual{typeof(exp)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    Y, pb = ChainRules.rrule(exp, X.x)
    Ȳ = zero(Y)
    return CoDual(Y, Ȳ), ExpPullback{P}(pb, Ȳ, X.dx)
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:linear_algebra})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    test_cases = vcat(
        map_prod([3, 7], Ps) do (N, P)
            return (false, :none, nothing, exp, randn(rng, P, N, N))
        end,
    )
    memory = Any[]
    return test_cases, memory
end

# Symmetric: off-diagonal elements in the stored triangle are doubled in the internal
# tangent representation (each off-diagonal element appears twice in the symmetric matrix).
# Divide by 2 to recover the user-friendly gradient w.r.t. the symmetric matrix elements.
function Mooncake.tangent_to_primal_internal!!(
    x::LinearAlgebra.Symmetric{T,S}, tx, c::MaybeCache
) where {T,S}
    tx isa Mooncake.NoTangent && return x
    data_tangent = tx.fields.data
    n = size(x.data, 1)
    for j in 1:n, i in 1:n
        if i == j
            x.data[i, j] = data_tangent[i, j]
        elseif (x.uplo == 'U' && i < j) || (x.uplo == 'L' && i > j)
            x.data[i, j] = data_tangent[i, j] / 2
        end
    end
    return x
end

# Hermitian: same doubling issue as Symmetric for off-diagonal elements.
function Mooncake.tangent_to_primal_internal!!(
    x::LinearAlgebra.Hermitian{T,S}, tx, c::MaybeCache
) where {T,S}
    tx isa Mooncake.NoTangent && return x
    data_tangent = tx.fields.data
    n = size(x.data, 1)
    for j in 1:n, i in 1:n
        if i == j
            x.data[i, j] = real(data_tangent[i, j])
        elseif (x.uplo == 'U' && i < j) || (x.uplo == 'L' && i > j)
            x.data[i, j] = data_tangent[i, j] / 2
        end
    end
    return x
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
    )
    memory = Any[]
    return test_cases, memory
end
