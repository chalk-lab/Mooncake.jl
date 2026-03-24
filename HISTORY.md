# 0.5.26

The `friendly_tangents=true` path previously converted every internal tangent back to a value of the primal type via `tangent_to_primal!!`. This relied on `_copy_output` to pre-allocate a buffer and `tangent_to_primal_internal!!` to fill it on every call. Both steps are problematic:

- `_copy_output` is best-effort and not guaranteed correct for all types — [#1084](https://github.com/chalk-lab/Mooncake.jl/issues/1084) is a recent silent failure  
- The primal round-trip was wrong for types with shared storage (e.g. `Symmetric`, where one stored entry represents two logical positions), silently returning an incorrect gradient [#937](https://github.com/chalk-lab/Mooncake.jl/issues/937)

## Default behaviour change

| Before                      | After                         |
|----------------------------|-------------------------------|
| default: value of primal   | default: raw Mooncake tangent |
| primal round-trip: always  | primal round-trip: explicit opt-in |
| custom gradient: not possible | custom gradient: explicit opt-in |

The raw-tangent default is safer: it never silently drops or corrupts information, avoids the allocation overhead, and the only types that go through reconstruction are explicit opt-ins (`AbstractDict`, `Symmetric`, `Hermitian`, `SymTridiagonal`) with their own tests.

# 0.5.24

Add `stop_gradient(x)` to block gradient propagation (TensorFlow/JAX analogue).
```julia
julia> using Mooncake

julia> f(x) = x[1] * Mooncake.stop_gradient(x)[2]
f (generic function with 1 method)

julia> cache = Mooncake.prepare_gradient_cache(f, [3.0, 4.0]);

julia> _, (_, g) = Mooncake.value_and_gradient!!(cache, f, [3.0, 4.0]);

julia> g  # g[2] == 0: gradient through x[2] inside stop_gradient is blocked
2-element Vector{Float64}:
 4.0
 0.0
```

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
