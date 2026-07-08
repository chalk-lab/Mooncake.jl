# Tutorial

This tutorial introduces Mooncake's API one step at a time, starting with gradients
and building up to Jacobians, Hessian-vector products, and full Hessians. Each derivative
follows the same two-step pattern: prepare a cache once, then call a fast run function as
many times as you like.

```@example tutorial
import Mooncake
```

## Computing gradients

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

Functions of several arguments work the same way: pass each argument to both
`prepare_gradient_cache` and `value_and_gradient!!`, and `grad` gains one entry per
argument, again led by the entry for `f`.

```@example tutorial
g(x, a, b) = a * f(x) + b
typical_a, typical_b = 1.0, 1.0
a, b = 42.0, 3.14

cache = Mooncake.prepare_gradient_cache(g, typical_x, typical_a, typical_b)
val, grad = Mooncake.value_and_gradient!!(cache, g, x, a, b)
(val, grad)
```

!!! note "Varying input sizes"
    A prepared cache is tied to each input's *type and size*, so a call with a
    differently sized input errors. If your sizes vary, build a reusable rule with
    `Mooncake.build_rrule` instead — see [Reusing a cache, and varying input sizes](@ref).

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

## Beyond gradients

The same prepare-once, call-many-times pattern extends to the other derivatives below.

### Forward mode

[`Mooncake.prepare_derivative_cache`](@ref) prepares a forward-mode cache. For a
scalar-valued function it also backs `value_and_gradient!!`, computing the gradient in
forward mode. See [Interface](@ref) for the chunked forward-mode controls.

### Jacobians

[`Mooncake.value_and_jacobian!!`](@ref) computes the full Jacobian of a vector-valued
function of a single dense vector input. It is not tied to a single mode: the cache it uses
can come from either forward mode ([`Mooncake.prepare_derivative_cache`](@ref)) or reverse
mode ([`Mooncake.prepare_pullback_cache`](@ref)). Either way it returns the primal output
together with a dense Jacobian whose columns correspond to input coordinates:

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

## Terminology

Mooncake.jl is built around Fréchet derivatives and their adjoints, described in detail in
[Algorithmic Differentiation](@ref).

**General cases:**

- **Fréchet derivative**: In forward mode, Mooncake computes the Fréchet derivative
  `D f[x]`, which maps tangent vectors to tangent vectors. It is implemented in
  `Mooncake.value_and_derivative!!`.

- **Adjoint of the derivative (pullback)**: In reverse mode, Mooncake computes the adjoint
  `D f[x]*` of the Fréchet derivative, which maps cotangent vectors backwards through the
  computation. It is implemented in `Mooncake.value_and_pullback!!`.

!!! note "Relationship to pushforward/pullback terminology"
    Other AD frameworks often name these operations differently. Mooncake's Fréchet
    derivative `D f[x]` (forward mode, `value_and_derivative!!`) is the operation elsewhere
    called a **pushforward** (e.g. `value_and_pushforward`). Its adjoint `D f[x]*` (reverse
    mode, `value_and_pullback!!`) is the **pullback** — here the name coincides with the
    `value_and_pullback` found in other packages.

**Special cases (scalar input/output):**

- **Derivative**: When the input is scalar, the Fréchet derivative `f'(x) = D f[x](v)`
  with `v = 1` gives the ordinary derivative, handled as a special case of
  `Mooncake.value_and_derivative!!`.

- **Gradient**: When the output is scalar, the adjoint of the derivative applied to `1`
  gives the gradient `∇f`, implemented in `Mooncake.value_and_gradient!!`.

!!! info
    For a detailed mathematical treatment of these concepts, see
    [Algorithmic Differentiation](@ref), particularly the sections on [Derivatives](@ref).
