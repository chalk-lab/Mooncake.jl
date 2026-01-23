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

For many chain rules, Mooncake performs an explicit canonicalisation step inside `rrule!!` that collapses heterogeneous array and tangent types into a small set of canonical representations. By canonicalising at the rule boundary, a single implementation can handle many argument- and tangent-type combinations without duplicating logic or relying on complex dispatch, allowing the remainder of the rule to assume a single, well-defined tangent representation.

### Example: `rrule!!` with `CoDual` types

In Mooncake, `rrule!!` methods receive `CoDual`-wrapped arguments, including the function itself. Each `CoDual` carries both a primal value and an associated tangent (or fdata). Consider a `kron` rule:

```julia
function Mooncake.rrule!!(
    ::CoDual{typeof(kron)},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    # Canonicalise inputs: although this method constrains `x1`/`x2` to `AbstractVecOrMat`,
    # they may still be realised by many concrete array types (e.g. vectors, matrices, views,
    # `Diagonal`, `Symmetric`, `PDMat`, and other wrappers). Canonicalising at the rule boundary
    # avoids a proliferation of specialised methods and lets the pullback operate on a single,
    # predictable dense matrix tangent representation.
    # `matrixify` returns a tuple (primal, tangent_matrix).
    px1, dx1 = matrixify(x1)
    px2, dx2 = matrixify(x2)

    # Run the primal computation
    y = kron(px1, px2)
    dy = zero(y)

    # Work with canonicalised tangent arrays.
    function kron_pb!!(::NoRData)
        # Run the pullback computation
        # Code omitted here for brevity
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), kron_pb!!
end
```

The key insight is that `matrixify` is one of several canonicalisation utilities (alongside `arrayify` and `numberify`) used to reconcile heterogeneous tangent representations into simple, uniform forms. In this case, tangents associated with vectors, matrices, views, `Diagonal`, `Symmetric`, `PDMat`, and other array wrappers are converted into a standard dense matrix representation that the rule can consume directly. Without this step, the rule would require multiple specialised methods or intricate dispatch logic to account for every admissible tangent representation.

Although this pattern is especially visible in BLAS- and LAPACK-backed rules—where performance-critical kernels must accommodate many array wrappers—it is not specific to linear algebra. Canonicalisation is a general rule-design technique: it isolates type heterogeneity at the boundary of the rule, simplifies the core logic, and improves maintainability across any domain where primitives admit many equivalent tangent representations (e.g. broadcasting, structured arrays, or custom numeric types).

### Connection to Julia's Promotion System

The tangent canonicalisation utilities (`arrayify`, `matrixify`, `numberify`) play a conceptual role similar to Julia's numeric promotion system:

```julia
Base.promote_rule(::Type{Type1}, ::Type{Type2}) = CommonType
```

Just as `promote_rule` reconciles heterogeneous numeric types into a common representation, these utilities reconcile heterogeneous tangent types into canonical forms. This pattern is particularly valuable for BLAS/LAPACK rules where performance-critical code must work with many array wrapper types (views, transposes, diagonals, etc.) while maintaining type stability.
