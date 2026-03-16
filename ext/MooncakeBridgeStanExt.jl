module MooncakeBridgeStanExt

using BridgeStan, Mooncake
import Mooncake:
    DefaultCtx,
    rrule!!,
    @is_primitive,
    CoDual,
    zero_fcodual,
    NoRData,
    primal,
    tangent,
    tangent_type,
    NoTangent

# StanModel contains C pointers (Ptr{Nothing}) which Mooncake can't recurse into.
# Declaring its tangent type as NoTangent short-circuits the generic struct machinery.
Mooncake.tangent_type(::Type{BridgeStan.StanModel}) = NoTangent

# Shared forward pass: compute log density and gradient in a single Stan call.
# BridgeStan's C interface requires Vector{Float64}; convert is a no-op when the
# type already matches, so there is no unnecessary copy for the common case.
function _log_density_fwd(sm, q, propto, jacobian)
    q_vec = convert(Vector{Float64}, q)
    grad = Vector{Float64}(undef, length(q_vec))
    ld, _ = BridgeStan.log_density_gradient!(sm, q_vec, grad; propto, jacobian)
    return ld, grad
end

# Rule for the direct call: log_density(sm, q) — no kwargs, defaults apply.
@is_primitive DefaultCtx Tuple{
    typeof(BridgeStan.log_density),BridgeStan.StanModel,AbstractVector{<:Real}
}

function rrule!!(
    ::CoDual{typeof(BridgeStan.log_density)},
    sm::CoDual{BridgeStan.StanModel},
    q::CoDual{<:AbstractVector{<:Real}},
)
    ld, grad = _log_density_fwd(primal(sm), primal(q), true, true)
    function log_density_pb(dld::Float64)
        tangent(q) .+= dld .* grad
        return NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(ld), log_density_pb
end

# Rule for the kwarg call: log_density(sm, q; propto=..., jacobian=...).
@is_primitive DefaultCtx Tuple{
    typeof(Core.kwcall),
    NamedTuple,
    typeof(BridgeStan.log_density),
    BridgeStan.StanModel,
    AbstractVector{<:Real},
}

function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs::CoDual{<:NamedTuple},
    ::CoDual{typeof(BridgeStan.log_density)},
    sm::CoDual{BridgeStan.StanModel},
    q::CoDual{<:AbstractVector{<:Real}},
)
    kw = primal(kwargs)
    propto = get(kw, :propto, true)::Bool
    jacobian = get(kw, :jacobian, true)::Bool
    ld, grad = _log_density_fwd(primal(sm), primal(q), propto, jacobian)
    function log_density_kw_pb(dld::Float64)
        tangent(q) .+= dld .* grad
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData()
    end
    return zero_fcodual(ld), log_density_kw_pb
end

end
