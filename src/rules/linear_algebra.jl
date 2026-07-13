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

# Symmetric/Hermitian/SymTridiagonal only store one triangle (or the diagonals) of the
# full matrix, so their tangent has no slot for the un-stored half. Adjoint/Transpose have
# no such gap: every logical entry maps to exactly one parent entry, just transposed. The
# friendly tangent is a plain matrix shaped like the wrapper itself, reconstructed via
# `arrayify` (src/rules/blas.jl, the same canonicalisation the differentiation rules use),
# so parent types it already recurses through (SubArray, Diagonal, nested Adjoint/Transpose,
# ...) are handled automatically rather than reimplemented here.
#
# The buffer uses similar(x.parent, ...) instead of a hardcoded Matrix, so it stays on the
# same device as the primal. A CPU buffer can't be written to from a GPU tangent.
#
# `.=` (not permutedims!/copyto!) broadcasts from the lazy adjoint/transpose view arrayify
# returns, so vector parents ((1, N) row shape) and matrix parents are both handled by the
# same method, and Adjoint's conjugation for complex T falls out of the broadcast for free
# (matching adjoint(_dx) in arrayify) instead of needing a separate conj! call.
#
# Constrained to T<:Union{IEEEFloat,BlasFloat}: without this, e.g. Transpose{Int} would
# also match here and return AsCustomised instead of AsRaw, breaking the existing
# regression test for #1149 (non-differentiable element types have no gradient to
# reconstruct, so they should fall through to the default AsRaw path).

function Mooncake.friendly_tangent_cache(
    x::LinearAlgebra.AdjOrTrans{T}
) where {T<:Union{IEEEFloat,BlasFloat}}
    FriendlyTangentCache{AsCustomised}(similar(x.parent, T, size(x)))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::AbstractMatrix{T}, x::LinearAlgebra.AdjOrTrans{T}, tangent
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, dx = arrayify(x, tangent)
    return tangent_as_friendly .= dx
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
