# Tutorial

There are two ways to compute gradients with Mooncake.jl:

- through the standardized [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) API
- through the native Mooncake.jl API

We recommend the former to start with, especially if you want to experiment with other automatic differentiation packages.

```@example tuto
import DifferentiationInterface as DI
import Mooncake
```

## DifferentiationInterface.jl API

DifferentiationInterface.jl (or DI for short) provides a common entry point for every automatic differentiation package in Julia.
To specify that you want to use Mooncake.jl, just create the right "backend" object (with an optional [`Mooncake.Config`](@ref)):

```@example tuto
backend = DI.AutoMooncake(; config=nothing)
```

This object is actually defined by a third package called [ADTypes.jl](https://github.com/SciML/ADTypes.jl), but re-exported by DI.

### Single argument

Suppose you want to differentiate the following function

```@example tuto
f(x) = sum(abs2, x)
```

on the following input

```@example tuto
x = float.(1:3)
```

The naive way is to simply call [`DI.gradient`](@extref DifferentiationInterface.gradient):

```@example tuto
DI.gradient(f, backend, x)  # slow, do not do this
```

This returns the correct gradient, but it is very slow because it includes the time taken by Mooncake.jl to compute a differentiation rule for `f` (see [Mooncake.jl's Rule System](@ref)).
If you anticipate you will need more than one gradient, it is better to call [`DI.prepare_gradient`](@extref DifferentiationInterface.prepare_gradient) on a typical (e.g. random) input first:

```@example tuto
typical_x = rand(3)
prep = DI.prepare_gradient(f, backend, typical_x)
```

The typical input should have the same size and type as the actual inputs we will provide later on.
As for the contents of the preparation result, they do not matter.
What matters is that it captures everything you need for `DI.gradient` to be fast:

```@example tuto
DI.gradient(f, prep, backend, x)  # fast
```

For optimal speed, you can provide storage space for the gradient and call [`DI.gradient!`](@extref DifferentiationInterface.gradient!) instead:

```@example tuto
grad = similar(x)
DI.gradient!(f, grad, prep, backend, x)  # very fast
```

If you also need the value of the function, check out [`DI.value_and_gradient`](@extref DifferentiationInterface.value_and_gradient) or [`DI.value_and_gradient!`](@extref DifferentiationInterface.value_and_gradient!):

```@example tuto
DI.value_and_gradient(f, prep, backend, x)
```

### Multiple arguments

What should you do if your function takes more than one input argument?
Well, DI can still handle it, _assuming that you only want the derivative with respect to one of them_ (the first one, by convention).
For instance, consider the function

```@example tuto
g(x, a, b) = a * f(x) + b
```

You can easily compute the gradient with respect to `x`, while keeping `a` and `b` fixed.
To do that, just wrap these two arguments inside [`DI.Constant`](@extref DifferentiationInterface.Constant), like so:

```@example tuto
typical_a, typical_b = 1.0, 1.0
prep = DI.prepare_gradient(g, backend, typical_x, DI.Constant(typical_a), DI.Constant(typical_b))

a, b = 42.0, 3.14
DI.value_and_gradient(g, prep, backend, x, DI.Constant(a), DI.Constant(b))
```

Note that this works even when you change the value of `a` or `b` (those are not baked into the preparation result).

If one of your additional arguments behaves like a scratch space in memory (instead of a meaningful constant), you can use [`DI.Cache`](@extref DifferentiationInterface.Cache) instead.

Now what if you care about the derivatives with respect to every argument?
You can always go back to the single-argument case by putting everything inside a tuple:

```@example tuto
g_tup(xab) = xab[2] * f(xab[1]) + xab[3]
prep = DI.prepare_gradient(g_tup, backend, (typical_x, typical_a, typical_b))
DI.value_and_gradient(g_tup, prep, backend, (x, a, b))
```

You can also use the native API of Mooncake.jl, discussed below.

### Beyond gradients

Going through DI allows you to compute other kinds of derivatives, like (reverse-mode) Jacobian matrices.
The syntax is very similar:

```@example tuto
h(x) = cos.(x) .* sin.(reverse(x))
prep = DI.prepare_jacobian(h, backend, x)
DI.jacobian(h, prep, backend, x)
```

## Mooncake.jl API

### Terminology Comparison with DifferentiationInterface

Mooncake.jl uses terminology rooted in mathematical analysis, specifically Frechet derivatives and their adjoints, as described in detail in [Algorithmic Differentiation](@ref). This differs somewhat from the conventions used by [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) and other AD packages.

The following table shows the correspondence between terminology:

| **Mode** | **General Case** | **Special Case** | **Mooncake Function** |
|----------|------------------|------------------|----------------------|
| **Forward mode** | **Frechet derivative** | **derivative** | [`Mooncake.value_and_derivative!!`](@ref) |
| | *(DI: pushforward)* | *(DI: derivative)* | |
| **Reverse mode** | **adjoint of derivative** | **gradient** | [`Mooncake.value_and_gradient!!`](@ref) |
| | *(DI: pullback)* | *(DI: gradient)* | [`Mooncake.value_and_pullback!!`](@ref) |

**Key points:**

- **Frechet derivative**: In forward mode, Mooncake computes the Frechet derivative `D f[x]`, which maps tangent vectors to tangent vectors. The derivative `D f[x](v)` is the special case when the input is scalar.

- **Adjoint of derivative**: In reverse mode, Mooncake computes the adjoint `D f[x]*` of the Frechet derivative. This maps cotangent vectors (gradients) backwards through the computation.

- **Gradient**: When the output is scalar, the adjoint of the derivative applied to `1` gives the gradient `∇f`. This is implemented in [`Mooncake.value_and_gradient!!`](@ref).

- **Pullback**: The more general case where the adjoint is applied to an arbitrary cotangent vector `ȳ` is available through [`Mooncake.value_and_pullback!!`](@ref).

!!! info
    For a detailed mathematical treatment of these concepts, see [Algorithmic Differentiation](@ref), particularly the sections on [Derivatives](@ref) and [Directional Derivatives and Gradients](@ref).

### Native API Functions

!!! warning
    Work in progress.
