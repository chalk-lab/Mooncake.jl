# friendly_tangent_cache and tangent_to_friendly_internal!! for Symmetric, Hermitian,
# SymTridiagonal, Adjoint, and Transpose. All five reuse arrayify (src/rules/blas.jl), the
# same canonicalisation utility the BLAS rules use to turn a tangent into a real array for
# actual matrix computations. Reusing it here means a type gets the same dense gradient
# whether it appears on its own or nested inside one of the other wrappers, rather than
# having a second, separately maintained presentation just for the "on its own" case.
#
# Adjoint and Transpose are exact relabellings: every entry of the wrapper corresponds to
# exactly one entry of the parent, just moved (and conjugated, for Adjoint). Nesting them
# is still an exact relabelling however deep, so a plain array wrapped in any number of
# Adjoint/Transpose layers reconstructs correctly.
#
# Symmetric, Hermitian, and SymTridiagonal only store one triangle (or, for SymTridiagonal,
# one off-diagonal band), so one real, unambiguous gradient number can end up needing to be
# shown at two positions in the dense matrix, for example both (1, 2) and (2, 1). We show
# the same number at both. This matches the standard matrix calculus convention for the
# gradient of a symmetric or Hermitian matrix, the symmetrised G + Gᵀ form (G + Gᴴ for
# Hermitian): it is the form you want if you then use the gradient in an update step and
# need the result to stay symmetric. arrayify already implements this for Symmetric by
# wrapping the reconstructed tangent in a fresh Symmetric and letting its own indexing
# mirror the value; Hermitian and SymTridiagonal do the same, reusing each type's own
# indexing instead of writing the mirroring by hand.
#
# Two structural cases genuinely cannot be reconstructed this way, and stay AsRaw:
#   - A SubArray with repeated indices: two output positions read the same, already summed
#     parent tangent value, and there is no way to recover what each occurrence originally
#     contributed. A SubArray with no repeated indices has no such problem and is exactly
#     as safe as Adjoint/Transpose.
#   - A triangular wrapper's implicit diagonal (UnitUpperTriangular's unit diagonal, for
#     example) is a constant baked into the primal's shape, not a real tangent value.
#     Reading it back through the same wrapper type shows that constant as if it were a
#     gradient, which is wrong. Diagonal has an implicit off-diagonal too, but that one
#     really is 0, which happens to be the correct tangent there, so Diagonal is fine.
#
# _arrayify_roundtrip_safe checks these properties by walking down the parent chain.
# _implicit_positions_are_zero covers the "implicit constant" case in general: it builds
# the all-zero tangent for a type and checks that arrayify's dense presentation of it is
# also all zero, rather than hardcoding which types pass. It doubles as the "does arrayify
# even support this type" check: arrayify falls back to a method that always throws for a
# type it has never seen, so an unrecognised parent is caught here and reported through
# @debug, instead of surfacing as a crash the next time a real gradient is computed. AsRaw
# is always a safe fallback, just a less friendly one.
#
# Every friendly_tangent_cache method below constrains T to Union{IEEEFloat,BlasFloat},
# matching arrayify's own bound. Without this, a non-differentiable eltype (Symmetric{Int},
# say) would also match and return AsCustomised instead of AsRaw, the same regression
# #1149 fixed for Transpose{Int}.

function Mooncake.friendly_tangent_cache(
    x::LinearAlgebra.Symmetric{T}
) where {T<:Union{IEEEFloat,BlasFloat}}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, size(x)...))
end
function Mooncake.friendly_tangent_cache(
    x::LinearAlgebra.Hermitian{T}
) where {T<:Union{IEEEFloat,BlasFloat}}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, size(x)...))
end
function Mooncake.friendly_tangent_cache(
    x::LinearAlgebra.SymTridiagonal{T}
) where {T<:Union{IEEEFloat,BlasFloat}}
    FriendlyTangentCache{AsCustomised}(Matrix{T}(undef, length(x.dv), length(x.dv)))
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, x::LinearAlgebra.Symmetric{T}, tangent
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, dx = arrayify(x, tangent)
    return tangent_as_friendly .= dx
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, x::LinearAlgebra.Hermitian{T}, tangent
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, dx = arrayify(x, tangent)
    return tangent_as_friendly .= dx
end

@unstable function Mooncake.tangent_to_friendly_internal!!(
    tangent_as_friendly::Matrix{T}, x::LinearAlgebra.SymTridiagonal{T}, tangent
) where {T<:Union{IEEEFloat,BlasFloat}}
    _, dx = arrayify(x, tangent)
    return tangent_as_friendly .= dx
end

_unaliased_index(::AbstractRange) = true
_unaliased_index(::Integer) = true
_unaliased_index(idx::AbstractArray) = allunique(idx)
_unaliased_index(::Any) = false

_has_unaliased_indices(x::SubArray) = all(_unaliased_index, x.indices)

function _implicit_positions_are_zero(x)
    local dz
    try
        _, dz = arrayify(x, zero_tangent(x))
    catch e
        e isa ErrorException || rethrow()
        @debug(
            "friendly_tangent_cache: could not verify that `arrayify` supports parent type $(typeof(x)) (see error below); if it's a new array type, add a method in src/rules/blas.jl to enable a friendly (AsCustomised) presentation. Falling back to AsRaw.",
            exception = e,
        )
        return false
    end
    return all(iszero, dz)
end

function _arrayify_roundtrip_safe(x)
    tangent_type(typeof(x)) <: AbstractArray && return true
    x isa LinearAlgebra.AdjOrTrans && return _arrayify_roundtrip_safe(x.parent)
    if x isa SubArray
        return _has_unaliased_indices(x) && _arrayify_roundtrip_safe(x.parent)
    end
    return _implicit_positions_are_zero(x)
end

# The buffer uses similar(x.parent, ...) instead of a hardcoded Matrix, so it stays on the
# same device as the primal. A CPU buffer can't be written to from a GPU tangent.
function Mooncake.friendly_tangent_cache(
    x::LinearAlgebra.AdjOrTrans{T}
) where {T<:Union{IEEEFloat,BlasFloat}}
    _arrayify_roundtrip_safe(x.parent) || return FriendlyTangentCache{AsRaw}(nothing)
    FriendlyTangentCache{AsCustomised}(similar(x.parent, T, size(x)))
end

# .= (not permutedims!/copyto!) broadcasts from the lazy view arrayify returns, so vector
# parents ((1, N) row shape) and matrix parents are both handled by the same method, and
# Adjoint's conjugation for complex T falls out of the broadcast for free (matching
# adjoint(_dx) in arrayify) instead of needing a separate conj! call.
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
