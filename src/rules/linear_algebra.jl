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

# Adjoint/Transpose aren't lossy like Symmetric/Hermitian/SymTridiagonal: every entry maps
# to exactly one parent entry, just transposed. The friendly tangent is a plain matrix
# shaped like the wrapper itself, filled by permuting the parent's tangent.
#
# The buffer uses similar(x.parent, ...) instead of a hardcoded Matrix, so it stays on the
# same device as the primal. A CPU buffer can't be written to from a GPU tangent.
#
# Both also wrap Vectors, giving a (1, N) row shape. There the parent's tangent is a Vector,
# not a matrix, so permutedims! doesn't apply; copyto! is used instead, since a (1, N)
# buffer's memory layout matches the underlying vector's exactly.

function Mooncake.friendly_tangent_cache(x::LinearAlgebra.Adjoint{T}) where {T}
    FriendlyTangentCache{AsCustomised}(similar(x.parent, T, size(x)))
end
function Mooncake.friendly_tangent_cache(x::LinearAlgebra.Transpose{T}) where {T}
    FriendlyTangentCache{AsCustomised}(similar(x.parent, T, size(x)))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::AbstractMatrix{T}, ::LinearAlgebra.Adjoint{T,<:AbstractVector}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.parent))
end
@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::AbstractMatrix{T}, ::LinearAlgebra.Adjoint{T,<:AbstractMatrix}, tangent
) where {T}
    return permutedims!(tangent_as_friendly, val(tangent.fields.parent), (2, 1))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::AbstractMatrix{T}, ::LinearAlgebra.Transpose{T,<:AbstractVector}, tangent
) where {T}
    return copyto!(tangent_as_friendly, val(tangent.fields.parent))
end
@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::AbstractMatrix{T}, ::LinearAlgebra.Transpose{T,<:AbstractMatrix}, tangent
) where {T}
    return permutedims!(tangent_as_friendly, val(tangent.fields.parent), (2, 1))
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
