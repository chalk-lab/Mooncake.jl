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

## Adding Methods To `frule!!` And `build_primitive_frule`

Forward mode has the same shape, and a new primitive should generally get both: write the reverse
rule for gradients and the forward one for directional derivatives, Jacobians and the forward half
of higher-order AD. A signature declared primitive in only one mode is transformed in the other.
That fallback works only if its implementation uses operations supported by that mode;
otherwise, provide a rule for both modes.

`@is_primitive` takes the mode, so declare the forward direction explicitly — `@is_primitive
MinimalCtx ForwardMode Tuple{typeof(f),P}`. Then implement a method of one of:
```@docs; canonical=false
Mooncake.frule!!
Mooncake.build_primitive_frule
```

A `frule!!` takes and returns [`Mooncake.Lifted`](@ref) slots. Its result's inner
representation has the canonical type `dual_type(Val(N), typeof(result))`.
For a zero derivative, return `zero_lifted(Val(N), result)`;
`zero_dual(Val(N), result)` constructs only the inner representation.
The rule propagates `N` directional derivatives at once, so write it for general `N`
rather than assuming a single lane.

For scalar primitives there is usually less to write than this suggests: teaching `NDual` the local
derivative once gives the `frule!!` for free. See
[Scalar And Low-Dimensional Rules Via `NDual`](@ref).

## Canonicalising Tangent Types

Canonicalising array tangents at the rule boundary lets a rule use array operations
across different tangent representations. The resulting arrays can retain wrappers
and their storage constraints.

Recall that `rrule!!` methods in Mooncake receive `CoDual`-wrapped arguments, including the function itself. Each `CoDual` carries both a primal value and an associated tangent (or `FData`). Consider a `kron` rule:

```julia
function Mooncake.rrule!!(
    ::CoDual{typeof(kron)},
    x1::CoDual{<:AbstractVecOrMat{<:T}},
    x2::CoDual{<:AbstractVecOrMat{<:T}},
) where {T<:Base.IEEEFloat}
    # Matrix-shaped tangents retain wrapper storage constraints, such as a stored diagonal.
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

`arrayify` expresses the tangent as an array with the primal's wrapper.
`matrixify` additionally reshapes vector operands into column matrices.
The result need not be dense or support arbitrary writes, so the rule must respect
the wrapper's storage constraints. For example, a `Diagonal` tangent remains a `Diagonal`.

Canonicalisation converts structured `Tangent`/`FData` representations into arrays at the
rule boundary, so the rule body can use array operations.

## Customising Friendly Gradients

When `friendly_tangents=true` is passed to `value_and_gradient!!` or `prepare_gradient_cache`, Mooncake converts its internal tangent representation into user-facing values. The conversion by type is:

- **Immutable structs, mutable structs (with standard `MutableTangent`), and closures with differentiable fields**: `NamedTuple` of per-field gradients, keyed by field name.
- **`Tuple`**: `Tuple` of per-element gradients.
- **`AbstractArray` with non-`IEEEFloat` (or complex) eltype**: array of per-element gradients.
- **`AbstractArray` with `IEEEFloat` (or complex) eltype**: plain array tangent, unchanged.
- **Callables with no captured differentiable state**: `NoTangent()`, unchanged.
- **`AbstractDict`**: a dict of the same type as the primal, with the same keys and gradient values.
- **`LinearAlgebra.Adjoint` / `Transpose`** over a differentiable parent eltype: the same wrapper around the parent's gradient — an `AbstractMatrix` of `size(x)`, not a dense `Matrix`.
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
    Mooncake.FriendlyTangentCache{Mooncake.AsCustomised}(Matrix{T}(undef, size(x)...))

# Step 2: implement the conversion from internal tangent to the buffer.
# Argument order: (dest, primal, tangent) — dest is first, used for dispatch on its type.
function Mooncake.tangent_to_friendly_internal!!(
    dest::Matrix{T}, ::MyMatrix{T}, tangent
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
