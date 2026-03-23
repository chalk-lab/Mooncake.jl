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

# Symmetric's tangent is a plain Tangent wrapping a full Matrix, with no knowledge of
# symmetric structure. Both S[i,j] and S[j,i] route to the same stored slot S.data[i,j]
# (the tangent entry S.data[j,i] accumulates no gradient and stays zero), so the chain
# rule accumulates all gradient into S.data[i,j] without accounting for the fact that it
# represents two matrix entries. A perturbation of ε to S.data[i,j] changes both S[i,j]
# and S[j,i] by ε, so the full-matrix gradient G must satisfy ⟨G, δS⟩ = δf, which
# requires G[i,j] = ∂f/∂(S.data[i,j]) / 2. Without this, ⟨G, δS⟩ would equal 2·δf.
#
# The cleaner fix would be a custom tangent type for Symmetric that encodes this factor
# structurally, so no correction is needed at conversion time. This specialisation is a
# pragmatic workaround at the tangent_to_primal!! boundary instead.
function Mooncake.tangent_to_primal_internal!!(
    x::LinearAlgebra.Symmetric{T,S}, tx, c::MaybeCache
) where {T,S}
    tx isa Mooncake.NoTangent && return x
    data_tangent = val(tx.fields.data)
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

# Hermitian's tangent has the same structure as Symmetric's: H[i,j] and H[j,i]
# (= conj(H.data[i,j])) both route to H.data[i,j], while the tangent entry H.data[j,i]
# accumulates no gradient and stays zero. By the same ⟨G, δH⟩ = δf argument, the
# full-matrix gradient requires G[i,j] = ∂f/∂(H.data[i,j]) / 2.
# Additionally, diagonal entries of a Hermitian matrix are constrained to be real, so we
# take real() to drop any imaginary noise in the accumulated diagonal gradient.
function Mooncake.tangent_to_primal_internal!!(
    x::LinearAlgebra.Hermitian{T,S}, tx, c::MaybeCache
) where {T,S}
    tx isa Mooncake.NoTangent && return x
    data_tangent = val(tx.fields.data)
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
