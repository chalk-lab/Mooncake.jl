# friendly_tangent_cache and tangent_to_friendly_internal!! for structured matrix types.
#
# Symmetric, Hermitian, and SymTridiagonal store only part of the matrix internally but
# represent a full symmetric/Hermitian matrix. The user-facing gradient is a plain Matrix{T}.
#
# Because we do not track which elements were getindex'ed, we cannot assume the tangent
# retains the original structure — it must be treated as a dense matrix. The original
# Symmetric/Hermitian/SymTridiagonal structure is therefore lost in the friendly gradient.
#
# friendly_tangent_cache pre-allocates the Matrix{T} output buffer at prepare time.
# tangent_to_friendly_internal!! copies the stored tangent fields directly into dest.
# The stored triangle (for Symmetric/Hermitian) or diagonals (for SymTridiagonal) hold the
# accumulated chain-rule gradient; all other entries are zero-initialised by Mooncake and
# are left zero by fill! (SymTridiagonal) or implicit via copyto! (Symmetric/Hermitian).
#
# For Hermitian{T} where T is complex: the stored triangle of .data accumulates the
# chain-rule gradient for both logical positions it represents (via Mooncake's usual
# tangent accumulation), and the non-stored triangle is zero-initialised. copyto! copies
# the full data matrix (including complex entries) to dest, which is a plain Matrix{T}.

# Adjoint and Transpose lose nothing — each entry is one parent entry, relabelled — so AsPrimal
# can rebuild the wrapper around the parent's gradient, giving an `AbstractMatrix` of `size(x)`
# rather than a raw `Tangent`. Reconstruction recurses, so `Adjoint(Symmetric(A))` works too.
#
# The conjugation in `Adjoint(dparent)` is right: Mooncake cotangents are real gradients
# (∂L/∂re + i ∂L/∂im) and x[i, j] == conj(parent[j, i]), so d/dx[i, j] == conj(dparent[j, i]).
#
# Test the parent's eltype, not `eltype(x)`, which is `Union{}` for e.g. `Matrix{Symbol}`. A
# non-differentiable one stays AsRaw; AsPrimal would return primal entries as a gradient.
function Mooncake.friendly_tangent_cache(
    x::Union{LinearAlgebra.Adjoint,LinearAlgebra.Transpose}
)
    tangent_type(eltype(parent(x))) == NoTangent &&
        return FriendlyTangentCache{AsRaw}(nothing)
    return FriendlyTangentCache{AsPrimal}(_copy_output(x))
end

function Mooncake.friendly_tangent_cache(x::LinearAlgebra.Symmetric{T}) where {T}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, size(x)...))
end
function Mooncake.friendly_tangent_cache(x::LinearAlgebra.Hermitian{T}) where {T}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, size(x)...))
end
function Mooncake.friendly_tangent_cache(x::LinearAlgebra.SymTridiagonal{T}) where {T}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, length(x.dv), length(x.dv)))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, ::LinearAlgebra.Symmetric{T}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.data))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, ::LinearAlgebra.Hermitian{T}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.data))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, ::LinearAlgebra.SymTridiagonal{T}, tangent
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

function hand_written_rule_test_cases(rng_ctor, ::Val{:linear_algebra})
    rng = rng_ctor(123)
    Ps = [Float64, Float32]
    test_cases = if Base.get_extension(Mooncake, :MooncakeChainRulesExt) === nothing
        Any[]
    else
        vcat(
            map_prod([3, 7], Ps) do (N, P)
                return (false, :none, nothing, exp, randn(rng, P, N, N))
            end,
        )
    end
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
