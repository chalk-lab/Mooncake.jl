# Tutorial

This tutorial walks through computing gradients, Jacobians, Hessian-vector
products, and Hessians with Mooncake.jl's native API.

```@example tutorial
import Mooncake
```

!!! info "Using Mooncake via DifferentiationInterface.jl"
    Mooncake.jl is also fully supported through
    [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
    via `ADTypes.AutoMooncake(; config=Mooncake.Config(...))`. Use that if you
    want a uniform API across multiple AD backends. The rest of this tutorial
    uses Mooncake's native API directly, which gives you the lowest-overhead
    path and the most direct access to Mooncake-specific options.

## Single argument

Suppose you want to differentiate the function

```@example tutorial
f(x) = sum(abs2, x)
x = float.(1:3)
```

The simplest entry point is [`Mooncake.value_and_gradient!!`](@ref). Before evaluating it,
prepare a cache once on a typical input — this is where Mooncake compiles the
differentiation rule:

```@example tutorial
typical_x = rand(3)
cache = Mooncake.prepare_gradient_cache(f, typical_x)
```

The contents of `typical_x` do not matter; only its type and shape. Subsequent calls on
inputs with matching shape are fast:

```@example tutorial
val, grad = Mooncake.value_and_gradient!!(cache, f, x)
(val, grad)
```

The returned `grad` has one entry per argument, preceded by the entry for `f` itself: here
`(df, dx)`, where `df` is the gradient with respect to any differentiable fields of `f`
(`NoTangent()` since `f` is not a callable struct), and `dx` is the gradient with respect
to `x`.

The cache owns the gradient buffers, so `grad` aliases storage inside `cache`. If you need
to keep it across calls, take a `copy` or `deepcopy` first; otherwise the next call to
`value_and_gradient!!` will overwrite it.

### Friendly tangents

By default, Mooncake represents tangents using internal types such as `Mooncake.Tangent`
for `struct`s (see [Mooncake.jl's Rule System](@ref)). To return tangents in the same shape
as the primal — for example a `Symmetric` tangent for a `Symmetric` matrix, or a
`NamedTuple` mirroring a custom struct — set `friendly_tangents=true` in the
[`Mooncake.Config`](@ref):

```@example tutorial
config = Mooncake.Config(; friendly_tangents=true)
cache = Mooncake.prepare_gradient_cache(f, typical_x; config)
val, grad = Mooncake.value_and_gradient!!(cache, f, x)
(val, grad)
```

The performance impact of `friendly_tangents=true` should be negligible. If it is
noticeable, something is likely wrong — please open an issue.

## Multiple arguments

Functions of several arguments work directly: pass each argument to
`prepare_gradient_cache` and `value_and_gradient!!`, and the returned gradient tuple
contains one entry per argument (preceded by the entry for `f` itself).

```@example tutorial
g(x, a, b) = a * f(x) + b
typical_a, typical_b = 1.0, 1.0
a, b = 42.0, 3.14

cache = Mooncake.prepare_gradient_cache(g, typical_x, typical_a, typical_b)
val, grad = Mooncake.value_and_gradient!!(cache, g, x, a, b)
(val, grad)
```

You can also pack the arguments into a tuple and differentiate with respect to the tuple:

```@example tutorial
g_tup(xab) = xab[2] * f(xab[1]) + xab[3]
cache = Mooncake.prepare_gradient_cache(g_tup, (typical_x, typical_a, typical_b))
val, grad = Mooncake.value_and_gradient!!(cache, g_tup, (x, a, b))
(val, grad)
```

For finer per-argument control over tangent zeroing in performance-sensitive code, see the
`args_to_zero` keyword discussed in [Interface](@ref).

## Beyond gradients

Mooncake exposes prepare/run pairs for several derivative kinds. Each cache stores a
compiled rule plus any buffers needed to make repeated calls allocation-light.

### Forward mode and Jacobians

[`Mooncake.prepare_derivative_cache`](@ref) prepares a forward-mode cache. The same cache
backs both [`Mooncake.value_and_jacobian!!`](@ref) (vector outputs) and `value_and_gradient!!`
(scalar outputs); for the latter, Mooncake seeds standard-basis directions internally and
evaluates them in chunks. See [Interface](@ref) for the chunked forward-mode controls.

For a vector-valued function of a dense vector input, `value_and_jacobian!!` returns the
primal output together with a dense Jacobian whose columns correspond to input
coordinates:

```@example tutorial
h(x) = cos.(x) .* sin.(reverse(x))
cache = Mooncake.prepare_derivative_cache(h, x)
Mooncake.value_and_jacobian!!(cache, h, x)
```

### Pullbacks

For outputs that are not scalars or vectors — for example a matrix or a custom struct —
use [`Mooncake.prepare_pullback_cache`](@ref) and `Mooncake.value_and_pullback!!`,
supplying a cotangent `ȳ` that matches the shape of `f(x...)`.

### Hessian-vector products

For a scalar-valued function with vector inputs,
[`Mooncake.prepare_hvp_cache`](@ref) sets up forward-over-reverse AD:

```@example tutorial
q(x) = sum(x .* x)
cache = Mooncake.prepare_hvp_cache(q, x)
v = [1.0, 0.0, 0.0]
Mooncake.value_and_hvp!!(cache, q, v, x)
```

The returned tuple is `(value, gradient, Hv)`.

### Hessians

To materialise the full Hessian, use [`Mooncake.prepare_hessian_cache`](@ref) and
[`Mooncake.value_gradient_and_hessian!!`](@ref):

```@example tutorial
cache = Mooncake.prepare_hessian_cache(q, x)
Mooncake.value_gradient_and_hessian!!(cache, q, x)
```

## Terminology comparison with DifferentiationInterface.jl

Mooncake.jl discusses Frechet derivatives and their adjoints, as described in detail in
[Algorithmic Differentiation](@ref). This differs from the conventions used by
[DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) and
some other AD packages.

**General cases:**

- **Frechet derivative**: In forward mode, Mooncake computes the Frechet derivative
  `D f[x]`, which maps tangent vectors to tangent vectors. This corresponds to what
  DifferentiationInterface refers to as a "pushforward", and is implemented in
  `Mooncake.value_and_derivative!!`.

- **Adjoint of derivative and pullback**: In reverse mode, Mooncake computes the adjoint
  `D f[x]*` of the Frechet derivative, which maps cotangent vectors backwards through the
  computation. This corresponds to what DifferentiationInterface calls a "pullback" and is
  implemented in `Mooncake.value_and_pullback!!`.

**Special cases (scalar input/output):**

- **Derivative**: When the input is scalar, the Frechet derivative `f'(x) = D f[x](v)`
  with `v = 1` gives the ordinary derivative. This corresponds to DI's `derivative`,
  while Mooncake handles it as a special case of `Mooncake.value_and_derivative!!`.

- **Gradient**: When the output is scalar, the adjoint of the derivative applied to `1`
  gives the gradient `∇f`. This corresponds to DI's `gradient` and is implemented in
  `Mooncake.value_and_gradient!!`.

!!! info
    For a detailed mathematical treatment of these concepts, see
    [Algorithmic Differentiation](@ref), particularly the sections on [Derivatives](@ref).
