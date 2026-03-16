# 0.5.23

## CUDA extension

Differentiation support for standard Julia/CUDA operations, focusing on

**Linear algebra** — BLAS matrix–vector products, `dot`, `norm`, and reductions (`sum`, `prod`, `cumsum`, `cumprod`, `mapreduce`) are supported, including complex inputs. Vector indexing is also supported for CUDA arrays. Scalar indexing is not supported by design.

```julia
# matrix multiply
f = (A, B) -> sum(A * B)
A, B = CUDA.randn(Float32, 4, 4), CUDA.randn(Float32, 4, 4)
cache = prepare_gradient_cache(f, A, B)
_, (_, ∂A, ∂B) = value_and_gradient!!(cache, f, A, B)

# matrix-vector multiply
f = (A, x) -> sum(A * x)
A, x = CUDA.randn(Float32, 4, 4), CUDA.randn(Float32, 4)
cache = prepare_gradient_cache(f, A, x)
_, (_, ∂A, ∂x) = value_and_gradient!!(cache, f, A, x)

# norm², dot, mean — same pattern
f = x -> norm(x)^2
f = (x, y) -> dot(x, y)
f = x -> mapreduce(abs2, +, x) / length(x)

# complex inputs work too
f = A -> real(sum(A * adjoint(A)))
```

**Broadcasting** — CUDA.jl compiles a specialised GPU kernel for each broadcast expression 
at runtime via `cufunction`. From Mooncake's perspective, this kernel appears as 
a `foreigncall`—opaque LLVM or PTX code that cannot be traced. To differentiate
through it, Mooncake exploits CUDA.jl's support for user-defined GPU-compatible
types: `NDual` dual numbers are registered as valid GPU element types, so the same
`cufunction` machinery re-compiles the kernel for dual-number inputs. Derivatives
are carried alongside primal values in a single GPU pass — no separate AD kernel
is required, and any broadcastable function is automatically differentiable. This
is the same strategy as Zygote's `broadcast_forward`:

```julia
f = x -> sum(sin.(x) .* cos.(x))
x = CUDA.randn(Float32, 8)
cache = prepare_gradient_cache(f, x)
_, (_, ∂x) = value_and_gradient!!(cache, f, x)  # ∂x::CuArray{Float32}
```

**Mutation and reshape** — rules for `fill!`, `unsafe_copyto!`, `unsafe_convert`, `materialize!`, 
`reshape`, `CuPtr` arithmetic, and CPU↔GPU transfers:

```julia
f = x -> sum(reshape(x, 4, 2))     # reshape on GPU
f = x -> sum(sin.(cu(x)))           # CPU → GPU (gradient flows back to CPU)
f = x -> sum(Array(x).^2)           # GPU → CPU
```

CI integration tests added for Flux and Lux models (CPU + GPU). Flux/Lux-specific rules 
are outside Mooncake's scope — models run via the general CUDA extension rules.

**Known limitation — Flux/Lux GPU performance:** without explicit reverse-mode rules
for neural network operators, Mooncake falls back to the NDual forward-mode broadcast
described above, which is correct but scales as O(params) in memory and kernel
launches. Large models are prohibitively slow on GPU until explicit `rrule!!`s are
added for key operations (e.g. cuDNN `BatchNorm`, …). CPU differentiation is unaffected 
by this performance limitation.

# 0.5.0

## Breaking Changes
- The tangent type of a `Complex{P<:IEEEFloat}` is now `Complex{P}` instead of `Tangent{@NamedTuple{re::P, im::P}}`.
- The `prepare_pullback_cache`, `prepare_gradient_cache` and `prepare_derivative_cache` interface functions now accept a `Mooncake.Config` directly.

# 0.4.147

## Public Interface
- Mooncake offers forward mode AD.
- Two new functions added to the public interface: `prepare_derivative_cache` and `value_and_derivative!!`.
- One new type added to the public interface: `Dual`.

## Internals
- `get_interpreter` was previously a zero-arg function. Is now a unary function, called with a "mode" argument: `get_interpreter(ForwardMode)`, `get_interpreter(ReverseMode)`.
- `@zero_derivative` should now be preferred to `@zero_adjoint`. `@zero_adjoint` will be removed in 0.5.
- `@from_chainrules` should now be preferred to `@from_rrule`. `@from_rrule` will be removed in 0.5.
