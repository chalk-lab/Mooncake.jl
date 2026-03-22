module MooncakeSparseArraysExt

using Mooncake, SparseArrays

import Mooncake: tangent_type, zero_tangent_internal, MaybeCache, NoTangent

# CHOLMOD.Factor{T,Ti} wraps a Ptr{cholmod_factor_struct} — an opaque C handle
# to CHOLMOD-managed memory. Registered as NoTangent so that structs containing
# a Factor can be differentiated through without crashing. See issue #698.
#
# When rrules for sparse Cholesky operations are added in the future, this will
# be upgraded to a custom tangent type (e.g., CholmodFactorTangent wrapping a
# SparseMatrixCSC). The upgrade path is:
#   1. Define a custom tangent type for Factor
#   2. Change tangent_type to return it (instead of NoTangent)
#   3. Implement the full tangent interface (~16 methods)
#   4. Write rrules for cholesky, logdet, \, etc.
# The dense Cholesky analogues (potrf!, potrs!) are in src/rules/lapack.jl.

const _cholmod_factor_struct = SparseArrays.LibSuiteSparse.cholmod_factor_struct

# Register the pointer type. This blocks the default Ptr{P} recursion into
# cholmod_factor_struct, which has many Ptr{Nothing} fields that would trigger
# the "zero_tangent not available for pointers" error.
tangent_type(::Type{Ptr{_cholmod_factor_struct}}) = NoTangent
zero_tangent_internal(::Ptr{_cholmod_factor_struct}, ::MaybeCache) = NoTangent()

# Register the Factor wrapper itself.
tangent_type(::Type{<:SparseArrays.CHOLMOD.Factor}) = NoTangent

end # module MooncakeSparseArraysExt
