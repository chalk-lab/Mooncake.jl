# 0.6.0

Breaking release: the forward-mode AD representation was rewritten.

- Removed the public `Mooncake.Dual{P,T}` type. Forward-mode values are now carried by the
  `Mooncake.Lifted{P,N,V}` slot (now `@public`), whose forward value `V === dual_type(Val(N), P)`
  is built from the parallel-arrays representation (`NDual` for scalars, `NDualArray` for arrays,
  etc.) rather than a single interleaved tangent. `N` is the chunk width.
- `value_and_derivative!!` now takes `(f, df)` / `(x, dx)` tuples (or `Lifted` slots) and returns a
  plain `(value, derivative)` tuple, instead of consuming and returning `Dual`s. Hand-written
  `frule!!`s now dispatch on `Lifted` rather than `Dual`.
- Forward mode is now batched ("chunked"): a width-`N` rule propagates `N` directional derivatives
  per pass (`chunk_size`), powering forward-mode gradients (`value_and_gradient!!` over a forward
  cache) and Jacobians (`value_and_jacobian!!`). Forward-over-reverse HVPs (`value_and_hvp!!`) and
  Hessians (`value_gradient_and_hessian!!`) currently run the outer forward pass at width 1;
  chunking them is a planned follow-up.
- Forward-mode seed factories are width-parameterized: `zero_dual(Val(N), x)` / `uninit_dual` /
  `randn_dual` (and the `zero_lifted` / `uninit_lifted` / `randn_lifted` slot wrappers).

# 0.5.32

- Fix forward-over-reverse Hessian-vector products on closures that capture a `Ref` wrapped in a `NoTangent`-typed aggregate, which previously threw `UndefRefError`. `prepare_hvp_cache` now eagerly compiles the inner `rrule!!` together with its forward-mode dual callables and routes the outer forward pass through a new `DerivedFoRRule`, so the inner `IdDict` constructor is no longer inlined past Mooncake's rule ([#1193](https://github.com/chalk-lab/Mooncake.jl/pull/1193), [#1202](https://github.com/chalk-lab/Mooncake.jl/pull/1202)).
- Fix the gradient of `copysign(x, y)` with respect to `x`: the derivative is `sign(x) * sign(y)`, and the missing `sign(x)` factor previously gave the wrong gradient sign when `x < 0` ([#1196](https://github.com/chalk-lab/Mooncake.jl/pull/1196)).
- Handle mutation of non-`const` globals (`setglobal!`) in forward mode on Julia 1.12+ ([#1194](https://github.com/chalk-lab/Mooncake.jl/pull/1194)).
- Version-bound the `Core._call_latest(CoreLogging.handle_message, ...)` rule (and its keyword-argument variant) to Julia below 1.12 ([#1200](https://github.com/chalk-lab/Mooncake.jl/pull/1200)).

# 0.5.31

- Guard `codual_type` / `fcodual_type` against unbound `TypeVar`s ([#1191](https://github.com/chalk-lab/Mooncake.jl/pull/1191), [#1192](https://github.com/chalk-lab/Mooncake.jl/pull/1192)).

# 0.5.30

- Bump the `LogExpFunctions` compat bound to 1 ([#1186](https://github.com/chalk-lab/Mooncake.jl/pull/1186)).

# 0.5.29

- Add a `max_fd_step` keyword to `TestUtils.test_rule` that caps the finite-difference step sizes, keeping perturbations of domain-restricted functions (`log`, `sqrt`, `cholesky`) inside their domains ([#1173](https://github.com/chalk-lab/Mooncake.jl/pull/1173)).
- Pre-allocate and reuse the Hessian, gradient, and basis-direction buffers in the Hessian cache so that repeated `value_gradient_and_hessian!!` calls avoid allocation ([#1178](https://github.com/chalk-lab/Mooncake.jl/pull/1178)).
- Add consistency checks for rule reuse ([#1172](https://github.com/chalk-lab/Mooncake.jl/pull/1172)).
- Mark `CUDACore.cudaError_enum` as having no tangent ([#1175](https://github.com/chalk-lab/Mooncake.jl/pull/1175)).

# 0.5.28

- Throw a clear `UnhandledLanguageFeatureException` for `try` / `catch` blocks in reverse-mode AD instead of an opaque IR-verification failure ([#1161](https://github.com/chalk-lab/Mooncake.jl/pull/1161)).
- Import the ChainRules `svd` rule via `@from_rrule` ([#1163](https://github.com/chalk-lab/Mooncake.jl/pull/1163), closes [#670](https://github.com/chalk-lab/Mooncake.jl/issues/670)).
- Guard the `friendly_tangent_cache` array branch against `NoTangent` element types (e.g. `SparseMatrixCSC{Int}`) ([#1150](https://github.com/chalk-lab/Mooncake.jl/pull/1150)).

# 0.5.27

- Add a cached `value_and_jacobian!!` interface for both forward-mode (chunked) and reverse-mode (row-by-row) caches ([#1153](https://github.com/chalk-lab/Mooncake.jl/pull/1153)).
- Fix `tangent_type` for `Union{NoRData, RData{...}}` ([#1133](https://github.com/chalk-lab/Mooncake.jl/pull/1133)).
- Add foreigncall zero-derivative rules for `jl_get_world_counter` and `jl_matching_methods`, supporting forward-over-reverse over those calls ([#1143](https://github.com/chalk-lab/Mooncake.jl/pull/1143)).
- Handle `Ptr` in `zero_tangent` by delegating to `uninit_tangent` ([#1139](https://github.com/chalk-lab/Mooncake.jl/pull/1139)).
- Update the CUDA extension for CUDA + cuDNN 6 ([#1148](https://github.com/chalk-lab/Mooncake.jl/pull/1148)).

# 0.5.26

- Add `Config(empty_cache=true)` to free internal caches before rebuilding rules.

```julia
config = Mooncake.Config(empty_cache=true)
cache = Mooncake.prepare_gradient_cache(sin, 1.0; config)
```

# 0.5.25

- Add `nfwd`: a new N-wide forward-mode implementation built around `NDual`, with `Nfwd` / `NfwdMooncake` internals and broad tests for scalar, array, and rule-building paths.
- Expand Mooncake's forward-mode interface and caching around `nfwd`, including prepared derivative/gradient cache improvements and lower-allocation hot paths for repeated calls.
- Route a broader scalar-math set through nfwd-backed direct primitive `frule!!` / `rrule!!` wrappers, reducing dependence on imported ChainRules rules for these cases.
- Move the ChainRules-backed matrix `exp` rule into `MooncakeChainRulesExt`, making `ChainRules` a weak dependency rather than a core dependency.
- Add precompile workloads, including complex scalar reverse/forward-mode paths for `ComplexF64` and `ComplexF32`.
- Improve docs for `nfwd`, including usage examples, interface notes, and clarification of nfwd/public-interface overheads.

The `friendly_tangents=true` path previously converted every internal tangent to a value of the primal type via `tangent_to_primal!!`. This relied on `_copy_output` to pre-allocate a buffer and `tangent_to_primal_internal!!` to fill it on every call. Both steps proved problematic:

- `_copy_output` is best-effort and not guaranteed correct for all types — [#1084](https://github.com/chalk-lab/Mooncake.jl/issues/1084) shows a recent silent failure
- The primal round-trip was wrong for types with shared storage (e.g. `Symmetric`, where one stored entry represents two logical positions), silently returning an incorrect gradient — [#937](https://github.com/chalk-lab/Mooncake.jl/issues/937)

## Default behaviour change

| Before                         | After                           |
|--------------------------------|---------------------------------|
| default: value of primal type  | default: raw Mooncake tangent   |
| primal round-trip: always      | primal round-trip: explicit opt-in |
| custom gradient: not possible  | custom gradient: explicit opt-in |

The raw-tangent default (`friendly_tangents=false`) is safer: it never silently drops or corrupts information and avoids unnecessary allocation. Under the default, arrays of `IEEEFloat` (or complex) elements have plain array tangents; callables with no captured differentiable state return `NoTangent`; and structs or closures with differentiable fields return a `Mooncake.Tangent` (immutable) or `Mooncake.MutableTangent` (mutable) wrapping a named tuple of their field tangents.

With `friendly_tangents=true`, structs (both immutable and mutable with the standard `MutableTangent` tangent type) and closures additionally unwrap to plain `NamedTuple`s. Mutable structs with custom tangent types return raw tangent unchanged. Types whose raw tangent reflects internal implementation layout rather than user-visible structure — `AbstractDict` (hash-table internals), `Symmetric`, `Hermitian`, `SymTridiagonal` — require explicit gradient reconstruction and are opt-in, each with their own tests.

# 0.5.24

Add `stop_gradient(x)` to block gradient propagation, analogous to `tf.stop_gradient` in TensorFlow and `jax.lax.stop_gradient` in JAX.
```julia
julia> using Mooncake

julia> f(x) = x[1] * Mooncake.stop_gradient(x)[2]
f (generic function with 1 method)

julia> cache = Mooncake.prepare_gradient_cache(f, [3.0, 4.0]);

julia> _, (_, g) = Mooncake.value_and_gradient!!(cache, f, [3.0, 4.0]);

julia> g  # g[2] == 0.0: gradient through x[2] inside stop_gradient is blocked
2-element Vector{Float64}:
 4.0
 0.0
```

# 0.5.23

## CUDA extension

Differentiation support for standard Julia/CUDA operations, focusing on:

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

**Broadcasting** — CUDA.jl compiles a specialised GPU kernel for each broadcast expression at runtime via `cufunction`. From Mooncake's perspective, this kernel appears as a `foreigncall` — opaque LLVM or PTX code that cannot be traced. To differentiate through it, Mooncake exploits CUDA.jl's support for user-defined GPU-compatible types: `NDual` dual numbers are registered as valid GPU element types, so the same `cufunction` machinery re-compiles the kernel for dual-number inputs. Derivatives are carried alongside primal values in a single GPU pass — no separate AD kernel is required, and any broadcastable function is automatically differentiable. This is the same strategy as Zygote's `broadcast_forward`:

```julia
f = x -> sum(sin.(x) .* cos.(x))
x = CUDA.randn(Float32, 8)
cache = prepare_gradient_cache(f, x)
_, (_, ∂x) = value_and_gradient!!(cache, f, x)  # ∂x::CuArray{Float32}
```

**Mutation and reshape** — rules for `fill!`, `unsafe_copyto!`, `unsafe_convert`, `materialize!`, `reshape`, `CuPtr` arithmetic, and CPU↔GPU transfers:

```julia
f = x -> sum(reshape(x, 4, 2))     # reshape on GPU
f = x -> sum(sin.(cu(x)))           # CPU → GPU (gradient flows back to CPU)
f = x -> sum(Array(x).^2)           # GPU → CPU
```

CI integration tests added for Flux and Lux models (CPU + GPU). Flux/Lux-specific rules are outside Mooncake's scope — models run via the general CUDA extension rules.

**Known limitation — Flux/Lux GPU performance:** without explicit reverse-mode rules for neural network operators, Mooncake falls back to the NDual forward-mode broadcast described above, which is correct but scales as O(params) in memory and kernel launches. Large models are prohibitively slow on GPU until explicit `rrule!!`s are added for key operations (e.g. cuDNN `BatchNorm`). CPU differentiation is unaffected by this performance limitation.

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
- `@zero_derivative` should now be preferred to `@zero_adjoint`. `@zero_adjoint` was removed in 0.5.
- `@from_chainrules` should now be preferred to `@from_rrule`. `@from_rrule` was removed in 0.5.
