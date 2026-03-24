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

For some differentiation rules, Mooncake performs an explicit canonicalisation step inside `frule!!`/`rrule!!` that collapses heterogeneous array and tangent types into a small set of canonical representations. By canonicalising at the rule boundary, a single implementation can support many combinations of argument and tangent types without duplicating logic or relying on complex dispatch. This allows the remainder of the rule to assume a single, well-defined tangent representation.

Recall that `rrule!!` methods in Mooncake receive `CoDual`-wrapped arguments, including the function itself. Each `CoDual` carries both a primal value and an associated tangent (or `FData`). Consider a `kron` rule:

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

The key insight is that `matrixify` is one of several canonicalisation utilities (alongside `arrayify`) used to reconcile heterogeneous tangent representations into simple, uniform forms. In this case, tangents associated with vectors, matrices, views, `Diagonal`, `Symmetric`, `PDMat`, and other array wrappers are converted into a standard dense matrix representation that the rule can consume directly. Without this step, the rule would require multiple specialised methods or intricate dispatch logic to account for every admissible tangent representation.

Although this pattern is especially visible in BLAS- and LAPACK-backed rules—where performance-critical kernels must accommodate many array wrappers—it is not specific to linear algebra. Canonicalisation is a general rule-design technique: it isolates type heterogeneity at the boundary of the rule, simplifies the core logic, and improves maintainability across any domain where primitives admit many equivalent tangent representations (e.g. broadcasting, structured arrays, or custom numeric types).

## Customising Friendly Gradients

When `friendly_tangents=true` is passed to `value_and_gradient!!` or `prepare_gradient_cache`, Mooncake converts its internal tangent representation into user-facing values. The conversion by type is:

- **Immutable structs, mutable structs (with standard `MutableTangent`), and closures with differentiable fields**: `NamedTuple` of per-field gradients, keyed by field name.
- **`Tuple`**: `Tuple` of per-element gradients.
- **`AbstractArray` with non-`IEEEFloat` (or complex) eltype**: array of per-element gradients.
- **`AbstractArray` with `IEEEFloat` (or complex) eltype**: plain array tangent, unchanged.
- **Callables with no captured differentiable state**: `NoTangent()`, unchanged.
- **`AbstractDict`**: a dict of the same type as the primal, with the same keys and gradient values.
- **Everything else** (primitive types, zero-field types, mutable structs with custom tangent types): raw Mooncake tangent, unchanged unless customised as described below.

For example, with `friendly_tangents=false` (default), an immutable struct `Foo` with fields `a::Float64` and `b::Vector{Float64}` returns a `Mooncake.Tangent` wrapping `(a = da, b = db)`, and a mutable struct `Bar` with the same fields returns a `Mooncake.MutableTangent` wrapping `(a = da, b = db)`. With `friendly_tangents=true` both unwrap to the plain `NamedTuple` `(a = da, b = db)` where `da::Float64` and `db::Vector{Float64}`.

An override is needed when the default output is unreadable or unintuitive — for example, types that store only a compressed representation, such as `LinearAlgebra.Symmetric`, which stores only one triangle of the full matrix but logically represents both.

Two hooks control this conversion:

```@docs; canonical=false
Mooncake.FriendlyTangentCache
Mooncake.friendly_tangent_cache
Mooncake.tangent_to_friendly!!
Mooncake.tangent_to_friendly_internal!!
```

### Example: full-matrix gradient for a structured matrix type

Suppose `MyMatrix{T}` stores data compactly but represents a full matrix.
To expose a plain `Matrix{T}` gradient to the user:

```julia
# Step 1: tell Mooncake to use a pre-allocated Matrix{T} buffer.
Mooncake.friendly_tangent_cache(x::MyMatrix{T}) where {T} =
    Mooncake.FriendlyTangentCache{Val{:as_customised}}(Matrix{T}(undef, size(x)...))

# Step 2: implement the conversion from internal tangent to the buffer.
function Mooncake.tangent_to_friendly_internal!!(
    ::MyMatrix{T}, dest::Matrix{T}, tangent
) where {T}
    # `val` unwraps the stored field tangent; adjust the field name to match MyMatrix's layout.
    copyto!(dest, Mooncake.val(tangent.fields.data))
    return dest
end
```

Any struct that _contains_ a `MyMatrix` field will automatically expose that field's
gradient as a `Matrix{T}` — no additional overrides required, because the default
struct recursion builds a `NamedTuple` of per-field friendly gradients.

The existing overloads for `LinearAlgebra.Symmetric`, `LinearAlgebra.Hermitian`, and
`LinearAlgebra.SymTridiagonal` in `src/rules/linear_algebra.jl` follow exactly this
pattern and serve as reference implementations.
