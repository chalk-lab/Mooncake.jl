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

function frule!!(
    ::Dual{typeof(LinearAlgebra.det)}, X_dX::Dual{Matrix{P}}
) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    C = det(X)
    return Dual(C, C * sum(diag(inv(X)*dX)))
end
function rrule!!(
    ::CoDual{typeof(LinearAlgebra.det)}, X::CoDual{Matrix{P}}
) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    Y = det(X)
    function det_pb(dY)
        dX .= dY * Y .* inv(adjoint(X))
        return NoRData(), NoRData()
    end
    Ȳ = zero(Y)
    return CoDual(Y, Ȳ), det_pb
end

function frule!!(::Dual{typeof(inv)}, X_dX::Dual{Matrix{P}}) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    Xi = inv(X)
    return Dual(Xi, -Xi * dX * Xi)
end
function rrule!!(::CoDual{typeof(inv)}, X::CoDual{Matrix{P}}) where {P<:IEEEFloat}
    X = copy(primal(X_dX))
    dX = copy(tangent(X_dX))
    Y = inv(X)
    function inv_pb(::NoRData)
        dX .= -adjoint(Y) * dX * adjoint(Y)
        return NoRData(), NoRData()
    end
    Ȳ = zero(Y)
    return CoDual(Y, Ȳ), inv_pb
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

derived_rule_test_cases(rng_ctor, ::Val{:linear_algebra}) = Any[], Any[]
