# friendly_tangent_dest and tangent_to_friendly_internal!! for structured matrix types.
#
# Symmetric, Hermitian, and SymTridiagonal store only part of the matrix internally but
# represent a full symmetric/Hermitian matrix. The user-facing gradient is a plain Matrix{T}.
#
# Because we do not track which elements were getindex'ed, we cannot assume the tangent
# retains the original structure — it must be treated as a dense matrix. The original
# Symmetric/Hermitian/SymTridiagonal structure is therefore lost in the friendly gradient.
#
# friendly_tangent_dest pre-allocates the Matrix{T} output buffer at prepare time.
# tangent_to_friendly_internal!! copies the raw tangent fields directly into the dest.
# This is safe because tangents are always initialised to zero, so the unused triangle
# (for Symmetric/Hermitian) or off-tridiagonal entries (for SymTridiagonal) are zero.

function Mooncake.friendly_tangent_dest(x::LinearAlgebra.Symmetric{T}) where {T}
    Matrix{T}(undef, size(x.data)...)
end
function Mooncake.friendly_tangent_dest(x::LinearAlgebra.Hermitian{T}) where {T}
    Matrix{T}(undef, size(x.data)...)
end
function Mooncake.friendly_tangent_dest(x::LinearAlgebra.SymTridiagonal{T}) where {T}
    Matrix{T}(undef, length(x.dv), length(x.dv))
end

function Mooncake.tangent_to_friendly_internal!!(
    ::LinearAlgebra.Symmetric{T}, tangent_as_friendly::Matrix{T}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.data))
end

function Mooncake.tangent_to_friendly_internal!!(
    ::LinearAlgebra.Hermitian{T}, tangent_as_friendly::Matrix{T}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.data))
end

function Mooncake.tangent_to_friendly_internal!!(
    ::LinearAlgebra.SymTridiagonal{T}, tangent_as_friendly::Matrix{T}, tangent
) where {T}
    dv = val(tangent.fields.dv)
    ev = val(tangent.fields.ev)
    fill!(tangent_as_friendly, zero(T))
    @inbounds for i in eachindex(dv)
        tangent_as_friendly[i, i] = dv[i]
    end
    @inbounds for i in eachindex(ev)
        tangent_as_friendly[i, i + 1] = ev[i]
        tangent_as_friendly[i + 1, i] = ev[i]
    end
    return tangent_as_friendly
end

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
