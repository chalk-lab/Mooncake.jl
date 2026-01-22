# Defining Rules

Most of the time, Mooncake.jl can just differentiate your code, but you will need to intervene if you make use of a language feature which is unsupported.
However, this does not always necessitate writing your own `rrule!!` from scratch.
In this section, we detail some useful strategies which can help you avoid having to write `rrule!!`s in many situations, which we discuss before discussing the more involved process of actually writing rules.

## Simplfiying Code via Overlays

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

For several `gemm`-like BLAS rules, Mooncake uses an explicit normalisation step inside `rrule!!` to collapse different arrays and tangent types into a smaller set of canonical representations. This allows a single rule to handle many argument-type combinations without duplicating logic.

### Example: `rrule!!` with `CoDual` types

Consider an rrule!! invoked on CoDual inputs, each carrying a primal value and an associated tangent:

```julia
y, pullback!! = rrule!!(
    CoDual(f, Δf),
    CoDual(x₁, Δx₁),
    CoDual(x₂, Δx₂),
)
```

In Mooncake, `rrule!!` is called with `CoDual`-wrapped arguments, including the function itself, to carry both primal values and tangents. `Δf` is shown here for completeness, but is not essential to this example. Assume the incoming tangents use different concrete array representations (schematically shown as SparseArray and DenseArray).

```julia
Δx₁ :: SparseArray  
Δx₂ :: DenseArray
```

Inside the rule, these can be normalised explicitly at the boundary:

```julia
# Tf is the primal type of f, and TΔf is its associated tangent type.
function rrule!!(::CoDual(Tf, TΔf), x₁::CoDual, x₂::CoDual)
    # arrayify defines the canonical representation expected by the rule.
    Δx₁′ = arrayify(tangent(x₁))
    Δx₂′ = arrayify(tangent(x₂))

    y = f(primal(x₁), primal(x₂))

    function pullback!!(Δy)
        return NoTangent(),
               Δx₁′ * Δy,
               Δx₂′ * Δy
    end

    return y, pullback!!
end
```

After normalisation, the remainder of the rule can assume a single, well-defined tangent representation, avoiding case splits on mixed tangent types.

### Connection to Julia's Promotion System

The tangent normalisation utilities (`arrayify`, `matrixify`, `numberify`) play a conceptual role similar to Julia's numeric promotion system:

```julia
Base.promote_rule(::Type{Type1}, ::Type{Type2}) = CommonType
```

Just as `promote_rule` reconciles heterogeneous numeric types into a common representation, these utilities reconcile heterogeneous tangent types into canonical forms. This pattern is particularly valuable for BLAS/LAPACK rules where performance-critical code must work with many array wrapper types (views, transposes, diagonals, etc.) while maintaining type stability and avoiding allocations.