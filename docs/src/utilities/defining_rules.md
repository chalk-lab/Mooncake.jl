# Defining Rules

Most of the time, Mooncake.jl can just differentiate your code, but you will need to intervene if you make use of a language feature which is unsupported.
However, this does not always necessitate writing your own `rrule!!` from scratch.
In this section, we detail some useful strategies which can help you avoid having to write `rrule!!`s in many situations, which we discuss before discussing the more involved process of actually writing rules.

## Simplifying Code via Overlays

```@docs; canonical=false
Mooncake.@mooncake_overlay
```

## Functions with Zero Adjoint

If the above strategy does not work, but you find yourself in the surprisingly common
situation that the adjoint of the derivative of your function is always zero, you can very
straightforwardly write a rule by making use of the following:
```@docs; canonical=false
Mooncake.@zero_adjoint
Mooncake.zero_adjoint
```

## Using ChainRules.jl

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides a large number of rules for differentiating functions in reverse-mode.
These rules are methods of the `ChainRulesCore.rrule` function.
There are some instances where it is most convenient to implement a `Mooncake.rrule!!` by wrapping an existing `ChainRulesCore.rrule`.

There is enough similarity between these two systems that most of the boilerplate code can be avoided.

```@docs; canonical=false
Mooncake.@from_rrule
```

## Adding Methods To `rrule!!` And `build_primitive_rrule`

If the above strategies do not work for you, you should first implement a method of [`Mooncake.is_primitive`](@ref) for the signature of interest:
```@docs; canonical=false
Mooncake.is_primitive
```
Then implement a method of one of the following:
```@docs; canonical=false
Mooncake.rrule!!
Mooncake.build_primitive_rrule
```

## Canonicalising Tangent Types

For several GEMM-like BLAS rules, Mooncake uses an explicit normalisation step inside `rrule!!` to collapse different array and tangent types into a smaller set of canonical representations. This allows a single rule to handle many argument-type combinations without duplicating logic.

### Example: `rrule!!` with `CoDual` types

In Mooncake, `rrule!!` methods receive `CoDual`-wrapped arguments, including the function itself. Each `CoDual` carries both a primal value and an associated tangent (or fdata). Consider a simplified BLAS-like function rule:

```julia
function rrule!!(
    ::CoDual{typeof(my_blas_op)},
    A_dA::CoDual{<:AbstractMatrix{T}},
    x_dx::CoDual{<:AbstractVector{T}},
) where {T<:BlasFloat}
    # Normalise inputs: convert tangents to canonical array representations.
    # arrayify returns a tuple (primal, tangent_array).
    A, dA = arrayify(A_dA)
    x, dx = arrayify(x_dx)

    # Run the primal computation.
    y = my_blas_op(A, x)

    function pullback!!(dy)
        # Work with normalised tangent arrays.
        # Increment dA and dx based on the reverse-mode computation.
        dA .+= dy * x'
        dx .+= A' * dy
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(y, zero_tangent(y)), pullback!!
end
```

The key insight is that `arrayify`, `matrixify`, and `numberify` convert potentially heterogeneous tangent representations (e.g., `Tangent{ComplexF64}` for complex numbers, `SubArray` tangents, or wrapped array types) into simple, uniform array types that BLAS routines can consume directly.

Without this normalisation step, the rule would need separate methods or complex dispatch logic to handle every combination of input tangent types (e.g., `Array` vs `SubArray` vs `ReshapedArray`, or real vs complex element tangents). By explicitly converting at the boundary, the remainder of the rule can assume a single, well-defined tangent representation.

### Connection to Julia's Promotion System

The tangent normalisation utilities (`arrayify`, `matrixify`, `numberify`) play a conceptual role similar to Julia's numeric promotion system:

```julia
Base.promote_rule(::Type{Type1}, ::Type{Type2}) = CommonType
```

Just as `promote_rule` reconciles heterogeneous numeric types into a common representation, these utilities reconcile heterogeneous tangent types into canonical forms. This approach:

- **Centralises** conversion logic in testable, well-defined functions
- **Makes explicit** what would otherwise be implicit dispatch complexity
- **Fails loudly** when unsupported type combinations are encountered (see the error message in `arrayify` for unhandled types)
- **Simplifies maintenance**: adding support for a new array type requires only a new `arrayify` method, not modifications to every BLAS rule

This pattern is particularly valuable for BLAS/LAPACK rules where performance-critical code must work with many array wrapper types (views, transposes, diagonals, etc.) while maintaining type stability and avoiding allocations.