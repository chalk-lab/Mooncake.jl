# 0.6.0

- Replace `Dual{P,T}` with public `Lifted{P,N,V}` in custom `frule!!` methods; use `V = dual_type(Val(N), P)` and read lane derivatives with `tangent(x, k)`; `N` is chunk width ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Replace `value_and_derivative!!` inputs `Dual(f, df), Dual(x, dx)` with `(f, df), (x, dx)`; results become `(value, derivative)`. `Lifted` inputs return `Lifted` ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Replace multi-input HVP/Hessian calls with a single input; concatenate arguments into one vector. Hessians require a vector; HVPs still accept scalars ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Replace `Config(enable_nfwd=...)` with `Config(...)`; the removed keyword raises `MethodError`. `NfwdMooncake` is removed; all forward differentiation uses `frule!!` ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Replace implicit `set_tangent_field!(t, :s, 1f0)` conversion into a `Float64` field with `set_tangent_field!(t, :s, Float64(1))`; mismatched types now raise `ArgumentError` ([#1338](https://github.com/chalk-lab/Mooncake.jl/pull/1338)).
- Replace `zero_tangent(ptr)` with explicitly allocated tangent storage when dereferencing derivatives; it now raises `ArgumentError`. `zero_codual(ptr)` remains an undereferenceable placeholder ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Custom reverse pointer rules must replace `Ptr{Cvoid}` tangents with `VoidPtrTangent`, retaining both the tangent address and erased element type ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- For aliased arguments, replace forward-gradient caches with reverse caches; `value_and_gradient!!` now raises `ArgumentError` for unsupported shared storage ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Replace independent forward seeds for aliased inputs with shared tangent storage; conflicting tuple or `Lifted` inputs to `value_and_derivative!!` now raise `ArgumentError` ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Rebuild prepared gradient/pullback caches when checked input alias relationships change; previously accepted mismatches now raise `PreparedCacheError` ([#1338](https://github.com/chalk-lab/Mooncake.jl/pull/1338)).
- Replace arguments identical to directly read mutable globals/constants with copies, or pass those values solely as arguments; both modes now raise `ArgumentError` ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Use chunk width one or reverse mode for raw pointers into nested arrays; wider forward chunks now refuse unsupported tangent layouts ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Report unsupported `ScopedValue` reads/writes with `UnhandledLanguageFeatureException`; avoid the access or define an enclosing rule. Compile-time constants are unaffected ([#1348](https://github.com/chalk-lab/Mooncake.jl/pull/1348)).
- Reject rectangular `LAPACK.getrf!` inputs with `DimensionMismatch` before mutation in both modes ([#1348](https://github.com/chalk-lab/Mooncake.jl/pull/1348)).
- Reject overlapping BLAS/LAPACK inputs and outputs with `ArgumentError` before mutation; copy overlapping inputs. Supported exact self-copies remain valid ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336), [#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Fix lowercase BLAS flags and `LAPACK.lacpy!` triangle selection; differentiated calls retain the underlying routine’s flag validation ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Fix BLAS derivatives over strided views; unsupported memory walks now raise `ArgumentError` instead of differentiating the wrong elements ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Return the zero subgradient for `BLAS.nrm2` at the zero vector, replacing NaN derivatives ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Preserve untouched outputs and zero coefficient derivatives for empty `BLAS.gemv!` calls ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Fix BLAS primal values and coefficient derivatives at zero coefficients, including `gemm!` with NaN output storage and `trmm!`/`trsm!` at zero alpha ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Prevent zero weights from contaminating BLAS coefficient gradients with NaN/Inf; preserve finite vector coefficient gradients at extreme magnitudes ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Keep BLAS/LAPACK reverse contributions exactly zero for whole-zero output cotangents, including beside NaN/Inf operands ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Keep wholly inactive lanes zero in level-2/3 BLAS forward rules beside NaN/Inf operands, including zero-alpha derivative products ([#1348](https://github.com/chalk-lab/Mooncake.jl/pull/1348)).
- Preserve coefficient directions through `LinearAlgebra.MulAddMul` shortcuts when alpha equals one or beta equals zero ([#1348](https://github.com/chalk-lab/Mooncake.jl/pull/1348)).
- Preserve cotangents through exact `LAPACK.lacpy!` and pointer `unsafe_copyto!` self-copies; correctly restore partially overlapping pointer copies ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Fix shared cotangent accumulation in `axpby!`; preserve real `axpy!` self-aliasing and reject unsupported overlapping walks, including complex self-aliasing ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Reject missing, placeholder, or incompatible pointer tangent storage before derivative loads/stores, including unsafe retyping through `Ptr{Cvoid}` ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336), [#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Reject `pointer_from_objref` derivative accesses with incompatible layouts; forward mode refuses numeric `Ref` object pointers instead of risking incorrect values ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Initialise `Core.memorynew` reverse tangents to zero rather than allocator-dependent contents ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Fix derivatives through structured matrix wrappers, including unit-triangular views/reshapes in `kron` and `logsumexp`; ignore implicit unit diagonals ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Accumulate reverse gradients across repeated arguments, views, reshapes, and, on Julia 1.11+, shared `Memory`/`MemoryRef` storage ([#1338](https://github.com/chalk-lab/Mooncake.jl/pull/1338)).
- Preserve shared storage during tangent seeding and arithmetic, preventing double-counted dot products/increments and duplicated `SimpleVector` tangents ([#1338](https://github.com/chalk-lab/Mooncake.jl/pull/1338)).
- Respect subsequently defined `tangent_type` methods when building friendly-gradient caches ([#1338](https://github.com/chalk-lab/Mooncake.jl/pull/1338)).
- Avoid reading undefined primal slots during primal-to-tangent conversion, including `Memory{Symbol}` ([#1338](https://github.com/chalk-lab/Mooncake.jl/pull/1338)).
- Reject changed storage sharing in structured forward-gradient caches with `PreparedCacheError` before refreshing inputs ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Restore cached forward tuple-call arguments after success or failure, including rebound fields and resized arrays; bare-rule and `Lifted` calls retain mutations ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Leave RNGs advanced after cached forward calls, including nested RNGs and failures; random draws execute afresh for each gradient/Jacobian chunk ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Fix singular real symmetric/Hermitian determinant derivatives; rank-deficient matrices now receive the adjugate derivative instead of unconditional zero ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Support forward-over-reverse differentiation of finite, nonsingular real symmetric/Hermitian `det`, `logdet`, and `logabsdet` calls ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Fix extreme-magnitude derivatives for division, two-argument `atan`, `asinh`, and `acosh`, avoiding spurious overflow/underflow ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Fix `ldexp`, `significand`, and `frexp` derivatives at extreme scales by preserving representable scaled directions ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Preserve primal NaN/signed-zero selection in `min`/`max`, correct crossed-bound `clamp` derivatives, and fix `copysign` derivatives with zero sign arguments ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Correct `rem` derivatives for negative quotients and preserve `iszero` branch behaviour when forward directions are nonzero ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Extend zero-direction guards at scalar singularities and retain second derivatives through zero cotangents and removable power singularities ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Match primal `tan` rounding and mixed-precision power/division promotion in forward mode ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Preserve non-finite real/imaginary components when adding, scaling, and dotting complex CUDA tangents ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Avoid LLVM crashes in BFloat16 tangent dot products and `Float64` conversions on Julia 1.11 x86_64 ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336), [#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Prevent Mooncake inference results from corrupting native package-image caches on Julia 1.10/1.11 ([#1336](https://github.com/chalk-lab/Mooncake.jl/pull/1336)).
- Avoid Julia 1.10 OpaqueClosure code-generation crashes after nested-rule invalidation ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340), [julia#61368](https://github.com/JuliaLang/julia/issues/61368)).
- Extend chunked forward differentiation through transformed Julia code, including `value_and_gradient!!` and vector-input `value_and_jacobian!!` via `Config(chunk_size=N)` ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Batch `value_gradient_and_hessian!!` columns with `chunk_size`, capped at input dimension; standalone `value_and_hvp!!` and one-dimensional Hessians use width one ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Add width-aware factories such as `zero_dual(Val(N), x)`, plus `uninit_dual`/`randn_dual` and `zero_lifted`/`uninit_lifted`/`randn_lifted` slot constructors; existing width-one spellings remain available ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Support Unicode character predicates such as `isuppercase` and `isletter` inside differentiated code ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Preserve forward derivatives through differentiable `SimpleVector` elements ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).
- Forward representations use `NDual` scalars and `NDualArray` numeric arrays; array primals share user storage while derivative lanes occupy separate storage ([#1340](https://github.com/chalk-lab/Mooncake.jl/pull/1340)).

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
