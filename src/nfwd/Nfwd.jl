"""
    Nfwd

Low-level N-wide dual arithmetic used by Mooncake's `nfwd` machinery.

## High-level examples

Construct an `NDual` directly and inspect its primal value and partials:

```julia
julia> d = NDual{Float64,2}(1.5, (1.0, 0.0))
NDual{Float64, 2}(1.5, (1.0, 0.0))

julia> ndual_value(d)
1.5

julia> ndual_partial(d, 1)
1.0
```

Propagate multiple directions through ordinary scalar code in one pass:

```julia
julia> x = NDual{Float64,2}(2.0, (1.0, 0.0));  # seed dx

julia> y = NDual{Float64,2}(3.0, (0.0, 1.0));  # seed dy

julia> z = x * y + sin(x);

julia> ndual_value(z)
6.909297426825682

julia> ndual_partials(z)  # (dz/dx, dz/dy)
(2.5838531634528574, 2.0)
```

Use `NDual` arrays with reductions such as `sum(abs2, xs)`:

```julia
julia> xs = [
           NDual{Float64,2}(1.0, (1.0, 0.0)),
           NDual{Float64,2}(2.0, (0.0, 1.0)),
       ];

julia> y = sum(abs2, xs);

julia> ndual_value(y)
5.0

julia> ndual_partials(y)  # (d/dx₁, d/dx₂)
(2.0, 4.0)
```

`Nfwd.jl` provides the N-wide dual arithmetic and signature helpers; the primitive
reverse-mode core in `src/rules/rules_via_nfwd.jl` packages that machinery into the
primitive `rrule!!`s defined there.
"""
module Nfwd

using Base: IEEEFloat
using LinearAlgebra

export NDual,
    NDualArray,
    NDualEltype,
    NDualUnsupportedError,
    ndual_value,
    ndual_partial,
    ndual_partials,
    UnsupportedError,
    UnsupportedInputError,
    UnsupportedOutputError,
    _NFWD_PREFERRED_CHUNK_SIZE,
    _nfwd_check_callable_sig,
    _nfwd_check_chunk_size,
    _nfwd_check_primal,
    _nfwd_default_chunk_size,
    _nfwd_fold_slots,
    _nfwd_infer_scalar_output,
    _nfwd_input_dof,
    _nfwd_input_error,
    _nfwd_is_supported_primal,
    _nfwd_is_supported_scalar,
    _nfwd_output_error,
    _nfwd_resolve_chunk_size,
    _nfwd_sig_default_chunk_size,
    _nfwd_sig_dof,
    _nfwd_type_dof,
    _nfwd_unfold_slots,
    _nfwd_validate

#
# ── Role of `ntuple` ──────────────────────────────────────────────────────────────
# `ntuple(f, Val(N))` is the workhorse for constructing and transforming NDual
# partials.  Its role differs by context:
#
#   On the CPU (rule setup, before kernel launch):
#     ntuple(f, Val(N)) also unrolls at compile time — Julia's Base implementation
#     is @generated and emits N independent expressions, which LLVM then sees as a
#     fixed-size tuple and may vectorise (e.g. a single <N x double> select for the
#     standard-basis seed).  So seed construction:
#       NDual{T,N}(x, ntuple(i -> i == k ? one(T) : zero(T), Val(N)))
#     is branchless on CPU too.  Performance is not critical here because this runs
#     once per input slot (host code), not once per array element.
#
#   Inside GPU kernels (arithmetic rules):
#     ntuple(f, Val(N)) with a statically-known N unrolls to N independent PTX
#     instructions at compile time — no loop, no heap allocation, no runtime
#     dispatch.  LLVM/NVVM sees a fixed-size tuple and vectorises each partial
#     slot independently, keeping everything in registers.  This is the key reason
#     N is a *type parameter* and not a runtime integer: the unrolling requires N
#     to be a compile-time constant.
#
# !! GPU KERNEL ARITHMETIC — PREFER BRANCHLESS OPERATIONS !!
# NDual arithmetic executes inside GPU kernels. Prefer `ifelse(cond, a, b)` over
# `cond ? a : b` or `if/else` blocks: `ifelse` evaluates both branches
# unconditionally and reliably lowers to a single PTX `selp` instruction.
# `?:` may also be optimised to `selp` by LLVM for simple scalar expressions,
# but this is not guaranteed — for data-dependent conditions (values that differ
# across threads) an unoptimised branch causes warp divergence.

"""
    NDual{T<:IEEEFloat, N} <: AbstractFloat

An N-wide dual number: carries one primal `value::T` and `N` partial derivatives
`partials::NTuple{N,T}`.  It is a plain `isbits` type — lives in GPU registers and
compiles to PTX without heap allocation.

## Analogy to ForwardDiff chunk mode

ForwardDiff's chunk mode computes N directional derivatives simultaneously by using
`ForwardDiff.Dual{Tag,T,N}` — a dual number with N partial slots.  `NDual{T,N}` is
the same idea, stripped of the tag parameter and defined entirely within Mooncake:

| Type                         | Tangent width | Tag parameter | Use case                        |
|------------------------------|---------------|---------------|---------------------------------|
| `NDual{T,1}`                 | 1             | n/a           | Standard width-1 `frule!!`      |
| `ForwardDiff.Dual{Tag,T,N}`  | N             | yes           | ForwardDiff chunk mode          |
| `NDual{T,N}`                 | N             | no            | GPU kernel widening (this type) |

`NDual` is a drop-in replacement for `ForwardDiff.Dual` in GPU broadcast kernels.
Removing the tag simplifies the type signature and eliminates the ForwardDiff
dependency from GPU AD.  The arithmetic rules are identical: each operation applies
the chain rule to all N slots at once.

## NDual vs the canonical Lifted V: scalar leaves and flattening

The canonical forward representation `Lifted{P, N, V}` wraps any differentiable value `P` —
it threads through Mooncake's tangent system and handles arbitrary structs transparently
(its `V` recurses structurally).

`NDual{T,N}` only wraps **scalar IEEEFloat (or Complex{IEEEFloat}) leaves**.
For a complex input type (e.g. a struct with several float fields), you must
**flatten** it to its scalar leaves before wrapping:

```
struct S; a::Float64; b::ComplexF64; end   # dof = 3 slots

S(a, b) → flatten → [a, re(b), im(b)]
             ↓ wrap each leaf as NDual{Float64,3}(x, eₖ)
             ↓ kernel runs
             ↓ extract partials
             ↓ unflatten → Tangent{S}(∂a, Complex(∂re_b, ∂im_b))
```

GPU kernels cannot receive a Dict or arbitrary struct; flattening to scalars
must happen on the CPU before launch, and gradient reassembly happens on the
CPU after.  The broadcast rule in `MooncakeCUDAExt.jl` implements this for the
specific node types that appear in a `Broadcasted` tree
(`_gpu_bcast_leaves` / `_gpu_fill_args_rdata`).

## Complex support

For complex inputs the kernel uses `Complex{NDual{T,N}}` where each component
(`re`, `im`) carries its own N partials.  Julia's generic `Complex` arithmetic
(`+`, `*`, `sin`, etc.) composes with `NDual` naturally because `NDual <: AbstractFloat`.

## Usage in GPU kernels

```julia
# Wrap input scalar at slot k (1-indexed) out of N total slots
d = NDual{T,N}(x, ntuple(j -> T(j == k), Val(N)))

# After kernel: extract primal and k-th partial
v  = ndual_value(d)
dk = ndual_partial(d, k)
```

To extend to a new scalar type S (non-IEEEFloat): define `_broadcast_elem_dof_type(::Type{S})`
and handle the wrapping / gradient extraction in `_leaf_effective_tangent`,
`materialize_pb!!`, and `_gpu_fill_args_rdata` in `MooncakeCUDAExt.jl`.

## Chunk-mode AD via NfwdMode{N}

### Background: Mooncake forward mode is width-1

Mooncake's forward mode computes one JVP per pass. `DerivedFRule` is called **once**
with all arguments seeded simultaneously:

```julia
value_and_derivative!!(cache, (f, df), (x, dx), (y, dy))
# computes:  ḟ = ∂f/∂f·df + ∂f/∂x·dx + ∂f/∂y·dy  — one direction
```

To recover the full Jacobian of `f : ℝⁿ → ℝᵐ`, the caller must invoke the rule **n
times**, once per basis vector `eₖ`.  There is no built-in chunk loop.  This is why
reverse mode is preferred for many-input scalar-output functions, and why NDual's GPU
trick — packing N directions into one kernel launch — is only worthwhile at GPU kernel
boundaries where each pass would otherwise incur a full launch overhead.

### Why standard width-1 forward mode cannot carry NDual tangents

The width-1 canonical V ties `V = dual_type(Val(1), P)`; for `P = Float64` that is
`NDual{Float64,1}` (one partial slot).  Widening to `NDual{Float64,N}` everywhere would
require `dual_type(Val(1), Float64) = NDual{Float64,N}` globally, infecting every `frule!!`
in the call graph and breaking width coherence throughout.

### NfwdMode{N}: NDual as the tangent type

The clean solution is a new AD context that overrides `tangent_type` for scalar leaves:

```julia
struct NfwdMode{N} end

# NDual is the tangent type — value field=0 by convention, partials carry N directions
tangent_type(::NfwdMode{N}, ::Type{T}) where {N, T<:IEEEFloat}          = NDual{T,N}
tangent_type(::NfwdMode{N}, ::Type{Complex{T}}) where {N, T<:IEEEFloat} = Complex{NDual{T,N}}

zero_ntangent(::Val{N}, ::Type{T}) where {N,T<:IEEEFloat} =
    NDual{T,N}(zero(T), ntuple(_ -> zero(T), Val(N)))
seed_ntangent(::Val{N}, ::Type{T}, k::Int) where {N,T<:IEEEFloat} =
    NDual{T,N}(zero(T), ntuple(i -> i == k ? one(T) : zero(T), Val(N)))
```

**The transform change is surgical**: `generate_dual_ir` calls `dual_type(P)` at 7
sites to assign IR argument types.  Threading the mode through those calls is the only
required modification — all statement rewriting (PhiNode, ReturnNode, GotoIfNot, …) is
tangent-type-agnostic.  `is_primitive` dispatch is unchanged (it operates on primal
signatures, not tangent types).

### Scalar `frule!!`s and CPU compatibility

Rules written generically in the tangent require no changes:

```julia
# Existing frule!! for sin — tangent(x)::NDual{T,N} in NfwdMode
frule!!(::Lifted{typeof(sin),N}, x::Lifted{T,N}) where {T<:IEEEFloat,N} =
    Lifted{T,N}(sin(primal(x)), cos(primal(x)) * tangent(x))
#                               ^^^^^^^^^^^^^^^^ T (scalar) * NDual{T,N} → NDual{T,N}  ✓
```

`cos(x) * NDual` scales the partials — already defined on NDual.  All chain rules
composed of scalar multiplication and addition propagate the N directions automatically.

**Two categories of CPU scalar rules:**

1. **`@from_chainrules` rules** (`sin`, `cos`, `exp`, …) — routed through `frule_wrapper`
   → `CRC.frule(tangents, primal...)`.  The ChainRules rule body does arithmetic like
   `cos(x) * ẋ` where `ẋ::NDual`.  These work transparently with NDual because NDual
   defines all the required scalar operations.

2. **Hand-coded rules using `nan_tangent_guard`** (`log`, `sqrt`, `cbrt`, …) —
   `nan_tangent_guard` is explicitly constrained to `IEEEFloat | Complex{<:IEEEFloat}`.
   Passing an NDual tangent would produce a `MethodError`, so these functions need
   dedicated NDual methods that preserve the same zero-mask behavior instead of calling
   `nan_tangent_guard` directly.

In practice `NfwdMode{N}` is designed for **GPU kernel boundaries** where each
width-1 pass costs a full kernel launch.  For CPU scalar ops the overhead is negligible
and there is no motivation to use chunk mode.

### `frule!!` template in NfwdMode

This pattern applies at any opaque boundary — most commonly a GPU kernel, but equally
valid for any CPU operation that needs an explicit N-wide rule (e.g. to override a
hand-coded rule that uses `nan_tangent_guard`, or to differentiate through an external
library call).  The only difference between GPU and CPU versions is the array type
(`CuArray` vs `Array`) and the absence of a kernel launch on CPU.

In NfwdMode the tangent of a `CuArray{T}` arg is `CuArray{NDual{T,N}}`, so the
NDual kernel input is built by a trivial merge — no `flatten_to_ndual` needed:

```julia
function frule!!(
    ::Lifted{typeof(my_kernel!)},
    _out::Lifted{<:CuArray{T}, N},
    _x  ::Lifted{<:CuArray{T}, N},
) where {T<:IEEEFloat, N}
    out, ∂out = primal(_out), tangent(_out)   # ∂out updated in-place
    x,   ∂x  = primal(_x),   tangent(_x)

    # Merge primal values with tangent directions into NDual kernel input.
    # ∂x[i].value == 0 (convention); ∂x[i].partials holds the N seed directions.
    x_nd   = map((v, t) -> NDual{T,N}(v, t.partials), x, ∂x)
    out_nd = similar(out, NDual{T,N})
    my_kernel!(out_nd, x_nd)   # one launch — all N directions at once

    out  .= ndual_value.(out_nd)
    ∂out .= map(d -> NDual{T,N}(zero(T), d.partials), out_nd)
    return _out
end
```

### Full Jacobian in one call

```julia
function full_jacobian(f!, out::CuArray{T}, x::CuArray{T}) where {T}
    N  = length(x)
    ∂x  = CuArray([seed_ntangent(Val(N), T, i) for i in 1:N])
    ∂out = fill!(similar(out, NDual{T,N}), zero_ntangent(Val(N), T))

    # A hypothetical NfwdMode{N}-aware forward rule run once with all N directions:
    rule = nfwd_mode_rule(NfwdMode{N}(), typeof(f!), CuArray{T}, CuArray{T})
    rule(lift(f!, NoTangent()), lift(out, ∂out), lift(x, ∂x))

    # ∂out[i].partials == (∂out[i]/∂x[1], …, ∂out[i]/∂x[N]) — full m×N Jacobian
    J = [ndual_partial.(∂out, k) for k in 1:N]
    return ndual_value.(out), J
end
```

Versus N separate width-1 passes, NfwdMode{N} needs **one** pass.  NDual is the
natural tangent type because its arithmetic is already register-friendly and no
conversion is needed at the kernel boundary.

### Open challenges

- Non-float leaves (`Int`, `Bool`, …) carry zero partial and must bypass NDual wrapping.
- Mixed-precision structs (`Float32` + `Float64` fields) require a promoted `T` or
  separate NDual blocks per precision group.
- `NfwdMode{N}` requires N to be chosen before compilation; adaptive chunk sizing
  (as in ForwardDiff) would need dynamic dispatch or recompilation.
"""
struct NDual{T<:IEEEFloat,N} <: AbstractFloat
    value::T
    partials::NTuple{N,T}
end

# ── Constructors ─────────────────────────────────────────────────────────────────

# Promote a plain scalar to a NDual with zero partials (acts as a constant).
NDual{T,N}(x::Real) where {T<:IEEEFloat,N} = NDual{T,N}(T(x), ntuple(_ -> zero(T), Val(N)))
# NDual{T,N}(::Rational{S}) is ambiguous between the NDual{T,N}(::Real) method above and
# Base's `(::Type{T})(x::Rational{S}) where {S, T<:AbstractFloat}` (rational.jl).
# Resolve by making the Rational case concrete.
function NDual{T,N}(x::Rational{S}) where {T<:IEEEFloat,N,S}
    NDual{T,N}(T(x), ntuple(_ -> zero(T), Val(N)))
end
# Identity / same-precision constructor: NDual{T,N}(d::NDual{T,N}) must not call T(d).
NDual{T,N}(d::NDual{T,N}) where {T<:IEEEFloat,N} = d
# Cross-precision constructor: NDual{T,N}(d::NDual{S,N}) where S ≠ T.
function NDual{T,N}(d::NDual{S,N}) where {T<:IEEEFloat,N,S<:IEEEFloat}
    NDual{T,N}(T(d.value), ntuple(i -> T(d.partials[i]), Val(N)))
end

# ── Accessors ────────────────────────────────────────────────────────────────────

@inline ndual_value(d::NDual) = d.value
@inline ndual_partial(d::NDual, k::Int) = d.partials[k]
@inline ndual_partials(d::NDual) = d.partials

# Internal NDual decode helpers shared by nfwd and the CUDA extension.
@inline _nfwd_dual_value(d::NDual) = ndual_value(d)
@inline _nfwd_dual_value(z::Complex{<:NDual}) = complex(
    ndual_value(real(z)), ndual_value(imag(z))
)
@inline _nfwd_dual_value(x) = x

@inline _nfwd_dual_partial(d::NDual, k::Int) = ndual_partial(d, k)
@inline _nfwd_dual_partial(z::Complex{<:NDual}, k::Int) = complex(
    ndual_partial(real(z), k), ndual_partial(imag(z), k)
)
@inline _nfwd_dual_partial(x, ::Int) = false

@inline _nfwd_dual_primal_type(::Type{<:NDual{T}}) where {T} = T
@inline _nfwd_dual_primal_type(::Type{Complex{NDual{T,N}}}) where {T,N} = Complex{T}
@inline _nfwd_dual_primal_type(::Type{T}) where {T} = T

@inline _nfwd_dual_has_partials(::Type{<:NDual}) = true
@inline _nfwd_dual_has_partials(::Type{<:Complex{<:NDual}}) = true
@inline _nfwd_dual_has_partials(::Type) = false

# Scalar analog of `_nfwd_lift`: assemble one value's canonical forward V from its
# primal and per-lane primal-typed partials — `NDual` for real, `Complex{NDual}`
# (interleaving real/imag) for complex. Shared by the element-read frules.
@inline _scalar_ndual(y::T, parts::NTuple{N,T}) where {T<:IEEEFloat,N} = NDual{T,N}(
    y, parts
)
@inline function _scalar_ndual(
    y::Complex{R}, parts::NTuple{N,Complex{R}}
) where {R<:IEEEFloat,N}
    return Complex(
        NDual{R,N}(real(y), ntuple(k -> real(parts[k]), Val(N))),
        NDual{R,N}(imag(y), ntuple(k -> imag(parts[k]), Val(N))),
    )
end

# ── NTuple arithmetic helpers ─────────────────────────────────────────────────────
# All fully unrolled at compile time via Val(N) — safe for GPU registers.

@inline _pt_scale(p::NTuple{N,T}, s::T) where {N,T} = ntuple(i -> s * p[i], Val(N))
# N=1 specializations avoid closure heap-allocation on the scalar (chunk_size=1) hot path.
@inline _pt_scale(p::NTuple{1,T}, s::T) where {T} = (s * p[1],)
# `_nfwd_zero_mask` plays the same role as `nan_tangent_guard` for scalar NDual algebra:
# when the local seed / upstream factor `a` is zero, replace `b` by zero(b) before the
# multiply so `0 * Inf` and `0 * NaN` collapse to zero instead of poisoning the tangent.
# nfwd uses this in forward mode through `_pt_guarded_scale`, which masks zero NDual lanes
# in singular formulas such as `log`, `sqrt`, `cbrt`, and `hypot`, and in reverse mode
# through `_nfwd_real_dot`, which masks zero upstream cotangents before contracting them
# against nfwd output tangents. This is the same strong-zero idea used in other AD systems,
# including ForwardDiff, to keep inactive directions from turning into NaNs.
@inline _nfwd_zero_mask(a, b) = ifelse(iszero(a), zero(b), b)
@inline function _pt_guarded_scale(p::NTuple{N,T}, s::T) where {N,T}
    return ntuple(i -> begin
        pi = p[i]
        pi * _nfwd_zero_mask(pi, s)
    end, Val(N))
end
@inline _pt_add(p::NTuple{N}, q::NTuple{N}) where {N} = ntuple(i -> p[i] + q[i], Val(N))
@inline _pt_sub(p::NTuple{N}, q::NTuple{N}) where {N} = ntuple(i -> p[i] - q[i], Val(N))
@inline _pt_neg(p::NTuple{N}) where {N} = ntuple(i -> -p[i], Val(N))
@inline _pt_zero(::Val{N}, ::Type{T}) where {N,T} = ntuple(_ -> zero(T), Val(N))
@inline _pt_add(p::NTuple{1,T}, q::NTuple{1,T}) where {T} = (p[1] + q[1],)
@inline _pt_sub(p::NTuple{1,T}, q::NTuple{1,T}) where {T} = (p[1] - q[1],)
@inline _pt_neg(p::NTuple{1,T}) where {T} = (-p[1],)
@inline _pt_zero(::Val{1}, ::Type{T}) where {T} = (zero(T),)

# These helpers define the scalar edge-case behavior used by nfwd for non-smooth
# primitives: `^` keeps the removable-singularity cases at x == 0, while `mod` and
# `mod2pi` return NaN coefficients at their discontinuity points.
@inline _nfwd_pow_grad_x(x, p, y) = p * y / x
@inline function _nfwd_pow_grad_x(x::P, p::P, y) where {P<:IEEEFloat}
    return ifelse(
        !iszero(x) || p < zero(P),
        p * y / x,
        ifelse(isone(p), one(y), ifelse(iszero(p) || p > one(P), zero(y), oftype(y, Inf))),
    )
end

@inline _nfwd_pow_grad_p(x, p, y) = y * log(complex(x))
@inline function _nfwd_pow_grad_p(x::P, p::P, y) where {P<:IEEEFloat}
    return ifelse(
        !iszero(x), y * real(log(complex(x))), ifelse(p > zero(P), zero(y), oftype(y, NaN))
    )
end

@inline function _nfwd_mod_grad_coeffs(x::P, y::P) where {P<:IEEEFloat}
    u = x / y
    nan = oftype(u, NaN)
    isint = isinteger(u)
    return ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))
end

@inline _nfwd_mod2pi_grad(x::P) where {P<:IEEEFloat} = ifelse(
    isinteger(x / P(2π)), P(NaN), one(P)
)

# ── AbstractFloat traits (needed for promote_rule with Complex etc.) ──────────────

Base.float(a::NDual) = a
Base.AbstractFloat(a::NDual) = a
Base.floatmin(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(floatmin(T))
Base.floatmax(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(floatmax(T))
Base.typemin(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(typemin(T))
Base.typemax(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(typemax(T))
# Instance dispatch: floatmin(x::NDual) and floatmax(x::NDual) forward to the type method.
Base.floatmin(x::NDual{T,N}) where {T,N} = NDual{T,N}(floatmin(T))
Base.floatmax(x::NDual{T,N}) where {T,N} = NDual{T,N}(floatmax(T))
Base.precision(::Type{NDual{T,N}}) where {T<:AbstractFloat,N} = precision(T)
Base.precision(::NDual{T,N}) where {T<:AbstractFloat,N} = precision(T)
# nextfloat / prevfloat are treated as identity maps for differentiation, so preserve the
# partials while advancing or retreating the scalar value by one representable step.
Base.nextfloat(a::NDual{T,N}) where {T,N} = NDual{T,N}(nextfloat(a.value), a.partials)
Base.prevfloat(a::NDual{T,N}) where {T,N} = NDual{T,N}(prevfloat(a.value), a.partials)
# exponent / significand: scalar operations; return scalar value (integer / NDual).
Base.exponent(a::NDual) = exponent(a.value)

# ── Zero / One ────────────────────────────────────────────────────────────────────

Base.zero(::NDual{T,N}) where {T,N} = NDual{T,N}(zero(T), _pt_zero(Val(N), T))
Base.one(::NDual{T,N}) where {T,N} = NDual{T,N}(one(T), _pt_zero(Val(N), T))
Base.zero(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(zero(T), _pt_zero(Val(N), T))
Base.one(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(one(T), _pt_zero(Val(N), T))
# Default oneunit(T) = T(one(T)) would call NDual{T,N}(::NDual) → Float64(::NDual) → error.
# Override to use the scalar constructor directly.
Base.oneunit(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(oneunit(T))
Base.oneunit(::NDual{T,N}) where {T,N} = NDual{T,N}(oneunit(T))

# ── Promotion / Conversion ────────────────────────────────────────────────────────

@inline function Base.convert(::Type{NDual{T,N}}, x::Real) where {T,N}
    return NDual{T,N}(T(x), _pt_zero(Val(N), T))
end
Base.convert(::Type{NDual{T,N}}, d::NDual{T,N}) where {T,N} = d

@inline function Base.promote_rule(::Type{NDual{T,N}}, ::Type{S}) where {T,N,S<:Real}
    return NDual{promote_type(T, S),N}
end
Base.promote_rule(::Type{NDual{T,N}}, ::Type{NDual{T,N}}) where {T,N} = NDual{T,N}
# Cross-precision: NDual{Float32,N} op NDual{Float64,N} → NDual{Float64,N}
@inline function Base.promote_rule(::Type{NDual{T1,N}}, ::Type{NDual{T2,N}}) where {T1,T2,N}
    return NDual{promote_type(T1, T2),N}
end
@inline function Base.convert(::Type{NDual{T,N}}, d::NDual{S,N}) where {T,N,S<:IEEEFloat}
    return NDual{T,N}(T(d.value), ntuple(i -> T(d.partials[i]), Val(N)))
end
@inline function NDual{T,N}(x::Real, r::RoundingMode) where {T<:IEEEFloat,N}
    return NDual{T,N}(T(x, r), _pt_zero(Val(N), T))
end

@noinline function _throw_ndual_lane_mismatch(op::Symbol, n1::Int, n2::Int)
    throw(
        DimensionMismatch(
            "NDual lane count mismatch in `$op`: left operand has $n1 lanes, right operand has $n2 lanes.",
        ),
    )
end

@inline function _promote_matching_nduals(
    op::Symbol, a::NDual{T1,N1}, b::NDual{T2,N2}
) where {T1,T2,N1,N2}
    N1 == N2 || _throw_ndual_lane_mismatch(op, N1, N2)
    return promote(a, b)
end

# ── Arithmetic ────────────────────────────────────────────────────────────────────

@inline function Base.:+(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(a.value + b.value, _pt_add(a.partials, b.partials))
end
@inline function Base.:-(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(a.value - b.value, _pt_sub(a.partials, b.partials))
end
@inline Base.:-(a::NDual{T,N}) where {T,N} = NDual{T,N}(-a.value, _pt_neg(a.partials))

# Real ± NDual: skip promotion — partials are unchanged for add, negated for sub.
# Without these, `c + x` promotes c to NDual(c, zeros) then adds zero partials,
# generating `fadd 0.0, p[k]` per slot that LLVM cannot fold (signed-zero corner case).
# Output type is promote_type(T, R) so that Float64 + NDual{Float32} → NDual{Float64}.
@inline function Base.:+(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(c) + S(x.value), ntuple(i -> S(x.partials[i]), Val(N)))
end
@inline function Base.:+(x::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(x.value) + S(c), ntuple(i -> S(x.partials[i]), Val(N)))
end
@inline function Base.:-(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(c) - S(x.value), ntuple(i -> -S(x.partials[i]), Val(N)))
end
@inline function Base.:-(x::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(S(x.value) - S(c), ntuple(i -> S(x.partials[i]), Val(N)))
end

# Disambiguate cross-precision NDual ± NDual: the Real ± NDual methods above match
# NDual{T2,N} as the "Real" argument when T1≠T2, creating ambiguity.  Explicit
# cross-precision methods resolve this by promoting to a common precision first.
@inline Base.:+(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = +(promote(a, b)...)
@inline Base.:-(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = -(promote(a, b)...)
@inline Base.:*(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = *(promote(a, b)...)
@inline Base.:/(a::NDual{T1,N}, b::NDual{T2,N}) where {T1,T2,N} = /(promote(a, b)...)
@inline Base.:+(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = +(_promote_matching_nduals(
    :+, a, b
)...)
@inline Base.:-(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = -(_promote_matching_nduals(
    :-, a, b
)...)
@inline Base.:*(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = *(_promote_matching_nduals(
    :*, a, b
)...)
@inline Base.:/(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = /(_promote_matching_nduals(
    :/, a, b
)...)

# Product rule: d(a*b) = a*db + b*da
@inline function Base.:*(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        a.value * b.value,
        _pt_add(_pt_scale(a.partials, b.value), _pt_scale(b.partials, a.value)),
    )
end

# Mixed Real*NDual: skip promotion and product rule — just scale the partials.
# Without these, `c * x` where c::Real promotes c to NDual(c, zeros) and runs
# the full product rule, generating a useless `x.value * 0.0` fmul per partial
# slot that IEEE semantics prevent LLVM from folding (-0 * NaN ≠ 0).
# Output type is promote_type(T, R) so that Float64 * NDual{Float32} → NDual{Float64}.
@inline function Base.:*(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    s = S(c)
    return NDual{S,N}(s * S(x.value), ntuple(i -> s * S(x.partials[i]), Val(N)))
end
@inline function Base.:*(x::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    s = S(c)
    return NDual{S,N}(S(x.value) * s, ntuple(i -> S(x.partials[i]) * s, Val(N)))
end

# Bool ± NDual and Bool * NDual: Base defines concrete overloads for (Bool, AbstractFloat)
# in bool.jl (+(::Bool, ::T), +(::T, ::Bool), *(::Bool, ::T), *(::T, ::Bool) where T<:AbstractFloat).
# Since NDual <: AbstractFloat these are now ambiguous with our (Real, NDual) methods.
# Resolve with concrete Bool overloads:
#   + : Bool acts as its numeric value (false=0, true=1) — same as T(b) + x.
#   * : preserves Base's "strong zero" contract (false*NaN == 0.0 via ifelse).
@inline Base.:+(b::Bool, x::NDual{T,N}) where {T,N} = NDual{T,N}(T(b) + x.value, x.partials)
@inline Base.:+(x::NDual{T,N}, b::Bool) where {T,N} = b + x
@inline Base.:*(b::Bool, x::NDual{T,N}) where {T,N} = ifelse(b, x, copysign(zero(x), x))
@inline Base.:*(x::NDual{T,N}, b::Bool) where {T,N} = b * x

# Quotient rule: d(a/b) = (da - (a/b)*db) / b
@inline function Base.:/(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    v = a.value / b.value
    return NDual{T,N}(
        v, _pt_scale(_pt_sub(a.partials, _pt_scale(b.partials, v)), inv(b.value))
    )
end

# NDual / Real: multiply by reciprocal — avoids promoting c to NDual.
@inline function Base.:/(x::NDual{T,N}, c::Real) where {T,N}
    s = inv(T(c))
    return NDual{T,N}(x.value * s, _pt_scale(x.partials, s))
end

# Real / NDual: d(c/b) = -(c/b²) db.  Without this, c::Real is promoted to
# NDual(c, zeros) and the quotient rule runs with a zero-partial numerator,
# producing a fneg(partial) chain that cancels with inv's -r² scaling but forces
# LLVM to emit a `fadd x, 0.0` canonicalization per partial slot (IEEE -0 rule).
# Defining this explicitly computes the scale as -(c*vi²) — a single scalar fneg —
# which pairs with the fneg already in the partial to give fmul(neg,neg)=pos,
# eliminating the fsub/fadd artifact.  Mirrors ForwardDiff's /(::Real,::Dual).
@inline function Base.:/(c::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    vi = inv(S(x.value))
    return NDual{S,N}(S(c) * vi, _pt_scale(x.partials, -(S(c) * vi * vi)))
end

# Direct inv: d(1/x)/dx = -1/x² = -(1/x)².  Avoids the quotient-rule path that
# promoting one(T)/a would trigger, eliminating a useless `0*x.value` fmul per slot.
@inline function Base.inv(a::NDual{T,N}) where {T,N}
    vi = inv(a.value)
    return NDual{T,N}(vi, _pt_scale(a.partials, -(vi * vi)))
end

# FMA (Fused Multiply-Add) based muladd: a single CPU instruction computes a*b+c
# in one step instead of separate fmul+fadd.  The default `a*b+c` would compute
# the product rule in two passes, emitting separate fmul+fadd per partial slot.
# Using nested muladd fuses both into two FMA instructions per slot:
#   value:   muladd(va, vb, vc)
#   partial: muladd(va, pb[i], muladd(vb, pa[i], pc[i]))
# This halves the instruction count for the matmul inner loop and triangular-
# solve back-substitution, which are the dominant cost in sum_matmat / sum_linsolve.
@inline function Base.muladd(a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        muladd(a.value, b.value, c.value),
        ntuple(
            i -> muladd(
                a.value, b.partials[i], muladd(b.value, a.partials[i], c.partials[i])
            ),
            Val(N),
        ),
    )
end
# Base.fma guarantees a single hardware FMA instruction (no intermediate rounding),
# whereas muladd may or may not fuse depending on platform/compiler flags.
@inline function Base.fma(a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        fma(a.value, b.value, c.value),
        ntuple(
            i -> fma(a.value, b.partials[i], fma(b.value, a.partials[i], c.partials[i])),
            Val(N),
        ),
    )
end

# Real*NDual+NDual and NDual*Real+NDual: the mixed cases arise in triangular solves
# where the factor matrix is Float64 and the rhs is NDual.  One FMA per partial slot.
@inline function Base.muladd(a::R, b::NDual{T,N}, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        muladd(S(a), S(b.value), S(c.value)),
        ntuple(i -> muladd(S(a), S(b.partials[i]), S(c.partials[i])), Val(N)),
    )
end
@inline function Base.fma(a::R, b::NDual{T,N}, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        fma(S(a), S(b.value), S(c.value)),
        ntuple(i -> fma(S(a), S(b.partials[i]), S(c.partials[i])), Val(N)),
    )
end
@inline function Base.muladd(a::NDual{T,N}, b::R, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        muladd(S(a.value), S(b), S(c.value)),
        ntuple(i -> muladd(S(a.partials[i]), S(b), S(c.partials[i])), Val(N)),
    )
end
@inline function Base.fma(a::NDual{T,N}, b::R, c::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        fma(S(a.value), S(b), S(c.value)),
        ntuple(i -> fma(S(a.partials[i]), S(b), S(c.partials[i])), Val(N)),
    )
end

# NDual*NDual+Real: product rule with a scalar addend.  Without this, c::Real is promoted
# to NDual(c, zeros) and the inner muladd becomes muladd(bv, ap, 0.0) per partial slot,
# emitting a wasted `fadd 0.0` (IEEE -0 semantics prevent LLVM from folding it).
# Specialising drops the zero addend: partial_i = muladd(av, bp[i], bv * ap[i]).
@inline function Base.muladd(a::NDual{T,N}, b::NDual{T,N}, c::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    return NDual{S,N}(
        muladd(S(a.value), S(b.value), S(c)),
        ntuple(
            i -> muladd(S(a.value), S(b.partials[i]), S(b.value) * S(a.partials[i])), Val(N)
        ),
    )
end

# ── Integer and real power ────────────────────────────────────────────────────────

# Literal-integer power: n is a compile-time Val{n}, so scalar sub-expressions use
# Base.literal_pow (e.g. x^2 → x*x, x^3 → x*x*x) rather than a runtime dispatch.
# This is the fast path for source-code literals like t^2 or t^3.
@inline function Base.literal_pow(::typeof(^), a::NDual{T,N}, ::Val{n}) where {T,N,n}
    v = Base.literal_pow(^, a.value, Val(n))
    dv = ifelse(iszero(n), zero(T), T(n) * Base.literal_pow(^, a.value, Val(n - 1)))
    # Guarded scale: at a singularity (e.g. n<0, a.value==0 makes dv=±Inf) inactive
    # lanes (zero partial) must stay zero rather than become 0*Inf=NaN, matching the
    # real-exponent `^` path.
    return NDual{T,N}(v, _pt_guarded_scale(a.partials, dv))
end
# Base defines literal_pow(^, ::AbstractFloat, ::Val{-1}) = inv(x) as a concrete
# specialisation.  Since NDual <: AbstractFloat, this creates an ambiguity with the
# general Val{n} method above (NDual wins on arg 2, Base wins on the concrete Val).
# Resolve with a concrete override that delegates to our inv rule.
@inline Base.literal_pow(::typeof(^), a::NDual{T,N}, ::Val{-1}) where {T,N} = inv(a)

# d(x^n) = n * x^(n-1) * dx  (ifelse keeps this branchless; see file header)
@inline function Base.:^(a::NDual{T,N}, n::Integer) where {T,N}
    v = a.value^n
    dv = ifelse(iszero(n), zero(T), T(n) * a.value^(n - 1))
    # Guarded scale: keeps inactive (zero-partial) lanes at zero when dv is ±Inf at a
    # singularity (n<0, a.value==0), matching the real-exponent `^` path.
    return NDual{T,N}(v, _pt_guarded_scale(a.partials, dv))
end

@inline Base.:^(a::NDual{T,N}, b::Rational) where {T,N} = a ^ T(b)

@inline function Base.:^(a::NDual{T,N}, b::Real) where {T,N}
    bT = T(b)
    v = a.value^bT
    return NDual{T,N}(v, _pt_guarded_scale(a.partials, _nfwd_pow_grad_x(a.value, bT, v)))
end

@inline function Base.:^(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    v = a.value^b.value
    coeff_a = _nfwd_pow_grad_x(a.value, b.value, float(v))
    coeff_b = _nfwd_pow_grad_p(a.value, b.value, float(v))
    return NDual{T,N}(
        v,
        _pt_add(
            _pt_guarded_scale(a.partials, coeff_a), _pt_guarded_scale(b.partials, coeff_b)
        ),
    )
end
@inline Base.:^(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = ^(_promote_matching_nduals(
    :^, a, b
)...)

# d(b^a)/da = b^a * log(b)  (b a plain Real, a the NDual)
@inline function Base.:^(b::Real, a::NDual{T,N}) where {T,N}
    v = T(b)^a.value
    return NDual{T,N}(v, _pt_scale(a.partials, v * T(log(b))))
end
@inline Base.:^(::Irrational{:ℯ}, a::NDual{T,N}) where {T,N} = exp(a)

@inline function Base.FastMath.pow_fast(a::NDual{T,N}, n::Integer) where {T,N}
    v = Base.FastMath.pow_fast(a.value, n)
    return NDual{T,N}(v, _pt_guarded_scale(a.partials, _nfwd_pow_grad_x(a.value, T(n), v)))
end
@inline function Base.FastMath.pow_fast(a::NDual{T,N}, ::Val{p}) where {T,N,p}
    v = Base.FastMath.pow_fast(a.value, Val(p))
    return NDual{T,N}(v, _pt_guarded_scale(a.partials, _nfwd_pow_grad_x(a.value, T(p), v)))
end

# ── Math functions ─────────────────────────────────────────────────────────────────
# Each follows: f(NDual(v,p)) = NDual(f(v), f'(v)*p)

# Trig
# Use sincos / sincosd to share the cordic/libm computation between sin and cos.
@inline function Base.sin(a::NDual{T,N}) where {T,N}
    s, c = sincos(a.value)
    return NDual{T,N}(s, _pt_scale(a.partials, c))
end
@inline function Base.cos(a::NDual{T,N}) where {T,N}
    s, c = sincos(a.value)
    return NDual{T,N}(c, _pt_scale(a.partials, -s))
end
@inline function Base.tan(a::NDual{T,N}) where {T,N}
    s, c = sincos(a.value)
    return NDual{T,N}(s / c, _pt_scale(a.partials, inv(c)^2))
end
@inline function Base.asin(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(asin(a.value), _pt_scale(a.partials, inv(sqrt(one(T) - a.value^2))))
end
@inline function Base.acos(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(acos(a.value), _pt_scale(a.partials, -inv(sqrt(one(T) - a.value^2))))
end
@inline function Base.atan(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(atan(a.value), _pt_scale(a.partials, inv(one(T) + a.value^2)))
end
@inline function Base.atan(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    r2 = a.value^2 + b.value^2
    return NDual{T,N}(
        atan(a.value, b.value),
        _pt_scale(
            _pt_sub(_pt_scale(a.partials, b.value), _pt_scale(b.partials, a.value)), inv(r2)
        ),
    )
end
@inline Base.atan(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = atan(
    _promote_matching_nduals(:atan, a, b)...
)

# NDual*Real atan: d/dy[atan(y,x)] = x/(y²+x²).  Without this, x::Real is promoted to
# NDual(x, zeros), and _pt_scale(x.partials, y.value) generates a fmul(partial, 0.0) per
# slot (zero-partial scale), followed by a wasted subtraction of that zero from the result.
@inline function Base.atan(y::NDual{T,N}, x::R) where {R<:Real,T,N}
    S = promote_type(T, R)
    r2 = S(y.value)^2 + S(x)^2
    return NDual{S,N}(atan(S(y.value), S(x)), _pt_scale(y.partials, S(x) / r2))
end

# Real*NDual atan: d/dx[atan(y,x)] = -y/(y²+x²).  Without this, y::Real is promoted to
# NDual(y, zeros), and _pt_scale(y.partials, x.value) = 0 per slot, then fsub(0, partial)
# hits the same IEEE -0 canonicalization that the old Real/NDual division had.
@inline function Base.atan(y::R, x::NDual{T,N}) where {R<:Real,T,N}
    S = promote_type(T, R)
    r2 = S(y)^2 + S(x.value)^2
    return NDual{S,N}(atan(S(y), S(x.value)), _pt_scale(x.partials, -S(y) / r2))
end

# Hyperbolic
@inline function Base.sinh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sinh(a.value), _pt_scale(a.partials, cosh(a.value)))
end
@inline function Base.cosh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(cosh(a.value), _pt_scale(a.partials, sinh(a.value)))
end
@inline function Base.tanh(a::NDual{T,N}) where {T,N}
    tv = tanh(a.value)
    return NDual{T,N}(tv, _pt_scale(a.partials, one(T) - tv^2))
end
@inline function Base.asinh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(asinh(a.value), _pt_scale(a.partials, inv(sqrt(a.value^2 + one(T)))))
end
@inline function Base.acosh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(acosh(a.value), _pt_scale(a.partials, inv(sqrt(a.value^2 - one(T)))))
end
@inline function Base.atanh(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(atanh(a.value), _pt_scale(a.partials, inv(one(T) - a.value^2)))
end

# Reciprocal hyperbolic: sech, csch, coth and their inverses.
@inline function Base.sech(a::NDual{T,N}) where {T,N}
    sv = sech(a.value)
    return NDual{T,N}(sv, _pt_scale(a.partials, -tanh(a.value) * sv))
end
@inline function Base.csch(a::NDual{T,N}) where {T,N}
    cv = csch(a.value)
    return NDual{T,N}(cv, _pt_scale(a.partials, -coth(a.value) * cv))
end
@inline function Base.coth(a::NDual{T,N}) where {T,N}
    sv = csch(a.value)
    return NDual{T,N}(coth(a.value), _pt_scale(a.partials, -(sv^2)))
end
@inline function Base.asech(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        asech(a.value), _pt_scale(a.partials, -inv(a.value * sqrt(one(T) - a.value^2)))
    )
end
@inline function Base.acsch(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        acsch(a.value), _pt_scale(a.partials, -inv(abs(a.value) * sqrt(one(T) + a.value^2)))
    )
end
@inline function Base.acoth(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(acoth(a.value), _pt_scale(a.partials, inv(one(T) - a.value^2)))
end

# Exp / Log
@inline function Base.exp(a::NDual{T,N}) where {T,N}
    return (ev=exp(a.value); NDual{T,N}(ev, _pt_scale(a.partials, ev)))
end
@inline function Base.exp2(a::NDual{T,N}) where {T,N}
    return (ev=exp2(a.value); NDual{T,N}(ev, _pt_scale(a.partials, ev * T(log(2)))))
end
@inline function Base.exp10(a::NDual{T,N}) where {T,N}
    return (ev=exp10(a.value); NDual{T,N}(ev, _pt_scale(a.partials, ev * T(log(10)))))
end
@inline function Base.log(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(log(a.value), _pt_guarded_scale(a.partials, inv(a.value)))
end
@inline function Base.log2(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        log2(a.value), _pt_guarded_scale(a.partials, inv(a.value * T(log(2))))
    )
end
@inline function Base.log10(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        log10(a.value), _pt_guarded_scale(a.partials, inv(a.value * T(log(10))))
    )
end
@inline function Base.log1p(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(log1p(a.value), _pt_guarded_scale(a.partials, inv(one(T) + a.value)))
end
@inline function Base.expm1(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(expm1(a.value), _pt_scale(a.partials, exp(a.value)))
end

# Two-argument log: log(b, x) = log(x)/log(b); d/dx = inv(x * log(b)),
# d/db = -log(x) / (b * log(b)^2) = -log(b, x) / (b * log(b)).
@inline function Base.log(b::Real, a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        log(b, a.value), _pt_guarded_scale(a.partials, inv(a.value * T(log(b))))
    )
end
@inline function Base.log(b::NDual{T,N}, a::NDual{T,N}) where {T,N}
    log_b = log(b.value)
    y = log(b.value, a.value)
    return NDual{T,N}(
        y,
        _pt_add(
            _pt_guarded_scale(b.partials, -y / (b.value * log_b)),
            _pt_guarded_scale(a.partials, inv(a.value * log_b)),
        ),
    )
end
@inline Base.log(b::NDual{T1,N1}, a::NDual{T2,N2}) where {T1,T2,N1,N2} = log(
    _promote_matching_nduals(:log, b, a)...
)
@inline Base.log(::Irrational{:ℯ}, a::NDual{T,N}) where {T,N} = log(a)

# ldexp(a, n) = a * 2^n — linear; derivative = 2^n.
@inline function Base.ldexp(a::NDual{T,N}, n::Integer) where {T,N}
    return NDual{T,N}(ldexp(a.value, n), _pt_scale(a.partials, T(exp2(n))))
end

# Roots
@inline function Base.sqrt(a::NDual{T,N}) where {T,N}
    return (sv=sqrt(a.value); NDual{T,N}(sv, _pt_guarded_scale(a.partials, inv(2 * sv))))
end
@inline function Base.cbrt(a::NDual{T,N}) where {T,N}
    return (cv=cbrt(a.value); NDual{T,N}(cv, _pt_guarded_scale(a.partials, inv(3 * cv^2))))
end

# Absolute value and sign
@inline function Base.abs(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(abs(a.value), _pt_scale(a.partials, sign(a.value)))
end
@inline function Base.abs2(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(abs2(a.value), _pt_scale(a.partials, 2 * a.value))
end
Base.sign(a::NDual{T,N}) where {T,N} = NDual{T,N}(sign(a.value), _pt_zero(Val(N), T))

# sincos — fused sin+cos; returns (sin(a), cos(a)) as a tuple of 
@inline function Base.sincos(a::NDual{T,N}) where {T,N}
    sv, cv = sincos(a.value)
    return NDual{T,N}(sv, _pt_scale(a.partials, cv)),
    NDual{T,N}(cv, _pt_scale(a.partials, -sv))
end

# sinpi / cospi — sin(π·x) and cos(π·x); derivative gains a π factor.
@inline function Base.sinpi(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sinpi(a.value), _pt_scale(a.partials, T(π) * cospi(a.value)))
end
@inline function Base.cospi(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(cospi(a.value), _pt_scale(a.partials, -T(π) * sinpi(a.value)))
end

# tanpi(x) = tan(π·x); derivative = π·sec²(π·x) = π·(1 + tan²(π·x)).
@inline function Base.tanpi(a::NDual{T,N}) where {T<:IEEEFloat,N}
    v = tanpi(a.value)
    return NDual{T,N}(v, _pt_scale(a.partials, T(π) * (one(T) + v^2)))
end

# sincospi — fused sin(π·x)+cos(π·x); each derivative gains a π factor.
@inline function Base.sincospi(a::NDual{T,N}) where {T<:IEEEFloat,N}
    sv, cv = sincospi(a.value)
    return NDual{T,N}(sv, _pt_scale(a.partials, T(π) * cv)),
    NDual{T,N}(cv, _pt_scale(a.partials, -T(π) * sv))
end

# Reciprocal trigonometric: sec, csc, cot and their inverses.
@inline function Base.sec(a::NDual{T,N}) where {T,N}
    sv = sec(a.value)
    return NDual{T,N}(sv, _pt_scale(a.partials, sv * tan(a.value)))
end
@inline function Base.csc(a::NDual{T,N}) where {T,N}
    cv = csc(a.value)
    return NDual{T,N}(cv, _pt_scale(a.partials, -cv * cot(a.value)))
end
@inline function Base.cot(a::NDual{T,N}) where {T,N}
    cv = cot(a.value)
    return NDual{T,N}(cv, _pt_scale(a.partials, -(one(T) + cv^2)))
end
@inline function Base.asec(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        asec(a.value), _pt_scale(a.partials, inv(abs(a.value) * sqrt(a.value^2 - one(T))))
    )
end
@inline function Base.acsc(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        acsc(a.value), _pt_scale(a.partials, -inv(abs(a.value) * sqrt(a.value^2 - one(T))))
    )
end
@inline function Base.acot(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(acot(a.value), _pt_scale(a.partials, -inv(one(T) + a.value^2)))
end

# Degree-based trigonometric functions — argument in degrees, derivative gains π/180.
@inline function Base.sind(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sind(a.value), _pt_scale(a.partials, T(deg2rad(cosd(a.value)))))
end
@inline function Base.cosd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(cosd(a.value), _pt_scale(a.partials, T(-deg2rad(sind(a.value)))))
end
@inline function Base.tand(a::NDual{T,N}) where {T,N}
    tv = tand(a.value)
    return NDual{T,N}(tv, _pt_scale(a.partials, T(deg2rad(one(T) + tv^2))))
end
@inline function Base.secd(a::NDual{T,N}) where {T,N}
    sv = secd(a.value)
    return NDual{T,N}(sv, _pt_scale(a.partials, T(deg2rad(sv * tand(a.value)))))
end
@inline function Base.cscd(a::NDual{T,N}) where {T,N}
    cv = cscd(a.value)
    return NDual{T,N}(cv, _pt_scale(a.partials, T(-deg2rad(cv * cotd(a.value)))))
end
@inline function Base.cotd(a::NDual{T,N}) where {T,N}
    cv = cotd(a.value)
    return NDual{T,N}(cv, _pt_scale(a.partials, T(-deg2rad(one(T) + cv^2))))
end
@inline function Base.asind(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        asind(a.value), _pt_scale(a.partials, inv(T(deg2rad(sqrt(one(T) - a.value^2)))))
    )
end
@inline function Base.acosd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        acosd(a.value), _pt_scale(a.partials, -inv(T(deg2rad(sqrt(one(T) - a.value^2)))))
    )
end
@inline function Base.atand(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        atand(a.value), _pt_scale(a.partials, inv(T(deg2rad(one(T) + a.value^2))))
    )
end
@inline function Base.asecd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        asecd(a.value),
        _pt_scale(a.partials, inv(T(deg2rad(abs(a.value) * sqrt(a.value^2 - one(T)))))),
    )
end
@inline function Base.acscd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        acscd(a.value),
        _pt_scale(a.partials, -inv(T(deg2rad(abs(a.value) * sqrt(a.value^2 - one(T)))))),
    )
end
@inline function Base.acotd(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        acotd(a.value), _pt_scale(a.partials, -inv(T(deg2rad(one(T) + a.value^2))))
    )
end

# Angle unit conversions — linear transforms; derivative is the constant scale factor.
@inline function Base.deg2rad(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(deg2rad(a.value), _pt_scale(a.partials, T(deg2rad(one(T)))))
end
@inline function Base.rad2deg(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(rad2deg(a.value), _pt_scale(a.partials, T(rad2deg(one(T)))))
end

# sinc(x) = sin(πx)/(πx) for x≠0, 1 at x=0; derivative = cosc(x).
@inline function Base.sinc(a::NDual{T,N}) where {T,N}
    return NDual{T,N}(sinc(a.value), _pt_scale(a.partials, T(cosc(a.value))))
end

# hypot — d/da hypot(a,b) = a / hypot(a,b), d/db = b / hypot(a,b).
@inline function Base.hypot(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    h = hypot(a.value, b.value)
    coeff_a = _nfwd_zero_mask(a.value, a.value / h)
    coeff_b = _nfwd_zero_mask(b.value, b.value / h)
    return NDual{T,N}(
        h, _pt_add(_pt_scale(a.partials, coeff_a), _pt_scale(b.partials, coeff_b))
    )
end
@inline Base.hypot(a::NDual{T1,N1}, b::NDual{T2,N2}) where {T1,T2,N1,N2} = hypot(
    _promote_matching_nduals(:hypot, a, b)...
)
@inline Base.hypot(a::NDual{T,N}) where {T,N} = abs(a)
@inline function Base.hypot(
    a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}, xs::Vararg{NDual{T,N},M}
) where {T,N,M}
    return hypot(hypot(a, b), c, xs...)
end

@inline function _ndual_pick_max(a, b)
    v = max(a, b)
    a_matches = isequal(v, a)
    b_matches = isequal(v, b)
    return ifelse(
        a_matches & !b_matches, true, ifelse(b_matches & !a_matches, false, false)
    )
end

@inline function _ndual_pick_min(a, b)
    v = min(a, b)
    a_matches = isequal(v, a)
    b_matches = isequal(v, b)
    return ifelse(a_matches & !b_matches, true, ifelse(b_matches & !a_matches, false, true))
end

# min / max — preserve Base's scalar result on NaN and signed-zero ties, then select the
# corresponding tangent. When both operands are exactly the same scalar value, keep the
# existing ordinary-tie convention (second arg for max, first arg for min).
@inline function Base.max(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(_ndual_pick_max(a.value, b.value), a, b)
end
@inline function Base.min(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(_ndual_pick_min(a.value, b.value), a, b)
end

# clamp — subgradient: zero tangent at the clamped endpoints.
# Nested ifelse keeps all branches branchless (no warp divergence on GPU).
@inline function Base.clamp(a::NDual{T,N}, lo::NDual{T,N}, hi::NDual{T,N}) where {T,N}
    return ifelse(a.value <= lo.value, lo, ifelse(a.value >= hi.value, hi, a))
end
@inline function Base.clamp(a::NDual{T,N}, lo::Real, hi::Real) where {T,N}
    return ifelse(
        a.value <= T(lo), NDual{T,N}(T(lo)), ifelse(a.value >= T(hi), NDual{T,N}(T(hi)), a)
    )
end

# flipsign / copysign — sign of result determined by primal; tangent follows.
@inline function Base.flipsign(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(signbit(b.value), -a, a)
end
@inline function Base.copysign(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return ifelse(signbit(a.value) == signbit(b.value), a, -a)
end

# ── Real / imag / conj — for Complex{NDual} to compose generically ────────────────
# A NDual is always the "real part" of itself; conj is the identity for reals.

@inline Base.real(a::NDual) = a
Base.imag(a::NDual{T,N}) where {T,N} = zero(NDual{T,N})
@inline Base.conj(a::NDual) = a
Base.reim(a::NDual{T,N}) where {T,N} = (a, zero(NDual{T,N}))
Base.isreal(::NDual) = true

# ── LinearAlgebra.dot specialisation ──────────────────────────────────────────────
# The generic AbstractArray fallback calls dot(x[i], y[i]) per element via an
# out-of-line function (sret convention for large structs), which prevents LLVM from
# fusing the inner loop.  For NDual{T,8} each element costs 2×72-byte memcpys plus
# an external call.  This specialisation keeps the loop body inlinable so LLVM can
# vectorise the partials accumulation.
@inline function LinearAlgebra.dot(
    x::StridedVector{NDual{T,N}}, y::StridedVector{NDual{T,N}}
) where {T,N}
    lx = length(x)
    lx == length(y) || throw(
        DimensionMismatch(
            lazy"first array has length $(lx) which does not match the length of the second, $(length(y)).",
        ),
    )
    lx == 0 && return NDual{T,N}(zero(T))
    @inbounds s = x[1] * y[1]
    @inbounds for i in 2:lx
        s = muladd(x[i], y[i], s)
    end
    return s
end

# ── LinearAlgebra.ldiv for LU{T} with NDual rhs ──────────────────────────────────
# The generic ldiv(F::Factorization, B) (non-mutating) converts the factorization to
# LU{NDual} before calling ldiv!, allocating a full Matrix{NDual} for no reason.
# The generic ldiv!(A::LU, B::AbstractVecOrMat) already handles mixed element types
# (Float64 factors, NDual rhs) correctly via _apply_ipiv_rows! + triangular solves.
# Override ldiv to bypass the conversion and call ldiv! directly.
@inline function LinearAlgebra.ldiv(
    F::LinearAlgebra.LU{T,<:AbstractMatrix{T}}, b::AbstractVector{<:NDual{T}}
) where {T}
    bb = copy(b)
    LinearAlgebra.ldiv!(F, bb)
    return bb
end

# ── Comparisons (on value only — for control flow in kernels) ──────────────────────

Base.:<(a::NDual, b::NDual) = a.value < b.value
Base.:>(a::NDual, b::NDual) = a.value > b.value
Base.:<=(a::NDual, b::NDual) = a.value <= b.value
Base.:>=(a::NDual, b::NDual) = a.value >= b.value
Base.:(==)(a::NDual, b::NDual) = a.value == b.value
Base.isless(a::NDual, b::NDual) = isless(a.value, b.value)

# NDual vs plain Real: compare value directly, avoiding zero-partial NDual construction
# via promotion.  The NDual×NDual methods above are more specific and still win when
# both sides are NDual.
Base.:<(a::NDual, b::Real) = a.value < b
Base.:>(a::NDual, b::Real) = a.value > b
Base.:<=(a::NDual, b::Real) = a.value <= b
Base.:>=(a::NDual, b::Real) = a.value >= b
Base.:<(a::Real, b::NDual) = a < b.value
Base.:>(a::Real, b::NDual) = a > b.value
Base.:<=(a::Real, b::NDual) = a <= b.value
Base.:>=(a::Real, b::NDual) = a >= b.value
Base.:<(a::NDual, b::Rational) = a.value < b
Base.:<(a::Rational, b::NDual) = a < b.value
Base.:<=(a::NDual, b::Rational) = a.value <= b
Base.:<=(a::Rational, b::NDual) = a <= b.value
Base.:<=(a::NDual, b::AbstractIrrational) = a.value <= b
Base.:<=(a::AbstractIrrational, b::NDual) = a <= b.value
Base.isnan(a::NDual) = isnan(a.value)
Base.isinf(a::NDual) = isinf(a.value)
Base.isfinite(a::NDual) = isfinite(a.value)
Base.signbit(a::NDual) = signbit(a.value)

# ── Utility ───────────────────────────────────────────────────────────────────────
Base.eps(d::NDual) = eps(d.value)
Base.eps(::Type{NDual{T,N}}) where {T,N} = eps(T)
# Checks both the primal value and all partial slots.  In GPU kernels this evaluates
# N partial values before short-circuiting; prefer `iszero(d.value)` inside hot kernel
# loops where the partial check is unnecessary and could cause warp divergence.
# This method is intended for host-side utility and correctness checks (e.g. hash, tests).
@inline function Base.iszero(d::NDual{T,N}) where {T,N}
    return iszero(d.value) && all(iszero, d.partials)
end
Base.hash(d::NDual, hsh::UInt) = hash(d.value, hsh)

# ── ifelse ────────────────────────────────────────────────────────────────────────
# Standard subgradient convention: branch on primal, propagate selected tangent.

Base.ifelse(c::Bool, a::NDual{T,N}, b::NDual{T,N}) where {T,N} = c ? a : b
Base.complex(re::NDual{T,N}, im::NDual{T,N}) where {T,N} = Complex{NDual{T,N}}(re, im)

# ── Complex{NDual} math — explicit GPU-safe implementations ───────────────────────
# Julia's generic Complex math (sin, cos, exp, log, sqrt) calls float(T::Type) and
# has isnan-guard branches that do not compile cleanly to PTX for custom T.
# Explicit implementations use only NDual scalar ops and compile without issues.

@inline function Base.abs(z::Complex{NDual{T,N}}) where {T,N}
    return hypot(real(z), imag(z))
end
@inline function Base.abs2(z::Complex{NDual{T,N}}) where {T,N}
    return real(z)^2 + imag(z)^2
end
@inline function Base.conj(z::Complex{NDual{T,N}}) where {T,N}
    return Complex(real(z), -imag(z))
end

# sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
@inline function Base.sin(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    sa, ca = sincos(a)
    return Complex(sa * cosh(b), ca * sinh(b))
end

# cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
@inline function Base.cos(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    sa, ca = sincos(a)
    return Complex(ca * cosh(b), -(sa * sinh(b)))
end

# exp(a + bi) = exp(a)·(cos(b) + i·sin(b))
@inline function Base.exp(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    er = exp(a)
    sb, cb = sincos(b)
    return Complex(er * cb, er * sb)
end

# log(a + bi) = log(|z|) + i·atan(b, a)
@inline function Base.log(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    return Complex(log(hypot(a, b)), atan(b, a))
end

# sqrt(a + bi) = sqrt((|z|+a)/2) + i·sign(b)·sqrt((|z|-a)/2)
# Construct the NDual arguments to sqrt directly with _pt_scale to avoid an
# unnecessary NDual*NDual product-rule evaluation (the factor 0.5 has zero partials).
@inline function Base.sqrt(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    r = hypot(a, b)
    half = T(0.5)
    re = sqrt(
        NDual{T,N}(
            (r.value + a.value) * half, _pt_scale(_pt_add(r.partials, a.partials), half)
        ),
    )
    im =
        copysign(one(NDual{T,N}), b) * sqrt(
            NDual{T,N}(
                (r.value - a.value) * half, _pt_scale(_pt_sub(r.partials, a.partials), half)
            ),
        )
    return Complex(re, im)
end

# tan(z) = sin(z)/cos(z)
@inline function Base.tan(z::Complex{NDual{T,N}}) where {T,N}
    return sin(z) / cos(z)
end

# ── Unsupported-operation error ───────────────────────────────────────────────────
# Operations that would silently destroy partial information (integer/rounding ops,
# integer division, modulo) throw a clear error instead of falling through to a
# confusing MethodError or, worse, silently dropping gradients.
#
# If you hit this, the function you are differentiating calls one of these
# non-differentiable operations on a floating-point argument.
# Options:
#   • Replace the operation with a differentiable approximation.
#   • Mark that argument as non-differentiable so NDual wrapping is skipped.
#   • Open an issue if you believe the operation should have a subgradient rule.

struct NDualUnsupportedError <: Exception
    op::Symbol
end
@inline function Base.showerror(io::IO, e::NDualUnsupportedError)
    return _nfwd_print_boxed_error(
        io,
        [
            "NDual does not support `$(e.op)`.",
            "This operation cannot propagate partial derivatives.",
            "Use a differentiable alternative, or open an issue if a subgradient rule makes sense.",
        ],
    )
end

# Keep the integer-conversion entrypoints explicit as well. These are the user-facing
# typed rounding paths (`floor(Int, x)`, `round(Int, x)`, etc.) and should fail with the
# same NDual-specific error instead of falling through to AbstractFloat methods.
for _op in (:floor, :ceil, :round, :trunc)
    @eval Base.$_op(::Type{I}, ::NDual{T,N}) where {I<:Union{Signed,Unsigned},T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
end

# Rounding ops have zero partial derivatives (piecewise constant). Define specific methods
# so that functions like `modf` (which calls `trunc`) work through NDual on the CPU.
for _op in (:floor, :ceil, :trunc)
    @eval function Base.$_op(x::NDual{T,N}) where {T<:IEEEFloat,N}
        return NDual{T,N}(Base.$_op(ndual_value(x)), ntuple(_ -> zero(T), Val(N)))
    end
end
@inline Base.round(x::NDual{T,N}) where {T<:IEEEFloat,N} = NDual{T,N}(
    round(ndual_value(x)), ntuple(_ -> zero(T), Val(N))
)
for _r in (
    RoundNearest,
    RoundNearestTiesAway,
    RoundNearestTiesUp,
    RoundToZero,
    RoundFromZero,
    RoundUp,
    RoundDown,
)
    @eval @inline function Base.round(x::NDual{T,N}, ::typeof($_r)) where {T<:IEEEFloat,N}
        return NDual{T,N}(round(ndual_value(x), $_r), ntuple(_ -> zero(T), Val(N)))
    end
    @eval @inline function Base.round(
        ::Type{I}, x::NDual{T,N}, ::typeof($_r)
    ) where {I<:Union{Signed,Unsigned},T<:IEEEFloat,N}
        throw(NDualUnsupportedError(:round))
    end
end

for _op in (:div, :fld, :cld, :gcd, :lcm)
    @eval Base.$_op(x::NDual{T,N}) where {T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
    @eval Base.$_op(x::NDual{T,N}, y::Real) where {T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
    @eval Base.$_op(x::Real, y::NDual{T,N}) where {T<:IEEEFloat,N} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
    @eval Base.$_op(x::NDual{T,N}, y::NDual{S,M}) where {T<:IEEEFloat,S<:IEEEFloat,N,M} = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
end

# `rem(x, y) = x - trunc(x/y)*y` (rounds toward zero), so its subgradient is ∂x=1,
# ∂y=-trunc(x/y) (a.e.) — `trunc`, not `floor` (they differ for negative x/y; `mod` uses `floor`).
# Defining the two-NDual method here resolves the ambiguity with Base's `rem(x::T, y::T) where
# T<:Real` and enables functions like `modf` that call `rem(x, T(1))` internally. Unlike `mod`,
# `rem` keeps the finite one-sided subgradient at integer ratios rather than NaN: `modf` differentiates
# only through the first argument (`y == 1` is constant), so the one-sided value is what it needs.
@inline function Base.rem(x::NDual{T,N}, y::NDual{T,N}) where {T<:IEEEFloat,N}
    pv, yv = ndual_value(x), ndual_value(y)
    c = trunc(pv / yv)
    return NDual{T,N}(
        rem(pv, yv), ntuple(k -> ndual_partial(x, k) - c * ndual_partial(y, k), Val(N))
    )
end

@inline function Base.mod(x::NDual{T,N}, y::NDual{T,N}) where {T<:IEEEFloat,N}
    coeff_x, coeff_y = _nfwd_mod_grad_coeffs(x.value, y.value)
    return NDual{T,N}(
        mod(x.value, y.value),
        _pt_add(_pt_scale(x.partials, coeff_x), _pt_scale(y.partials, coeff_y)),
    )
end
@inline Base.mod(x::NDual{T1,N1}, y::NDual{T2,N2}) where {T1<:IEEEFloat,T2<:IEEEFloat,N1,N2} = mod(
    _promote_matching_nduals(:mod, x, y)...
)
@inline Base.mod(x::NDual{T,N}) where {T<:IEEEFloat,N} = throw(NDualUnsupportedError(:mod))
@inline Base.mod(x::NDual{T,N}, y::Real) where {T<:IEEEFloat,N} = throw(
    NDualUnsupportedError(:mod)
)
@inline Base.mod(x::Real, y::NDual{T,N}) where {T<:IEEEFloat,N} = throw(
    NDualUnsupportedError(:mod)
)

@inline function Base.mod2pi(x::NDual{T,N}) where {T<:IEEEFloat,N}
    coeff = _nfwd_mod2pi_grad(x.value)
    return NDual{T,N}(mod2pi(x.value), _pt_scale(x.partials, coeff))
end

# ── Future: tiled GPU kernels with NDual ──────────────────────────────────────────
#
# The current broadcast AD uses one NDual{T,N} per thread: every thread computes
# the primal and all N partials in registers in a single kernel pass.  This is
# already efficient for small N and element-wise functions.  For larger N or
# functions with cross-element data reuse (reductions, softmax, layer norm),
# *tiled* kernels offer further gains:
#
# ── Conceptual note: tiling applied to the NDual itself ─────────────────────
# An NDual{T,N} is a tile in the partial-derivative dimension.  Just as spatial
# tiling partitions an M-element array into ceil(M/K) tiles of width K — each
# processed in one pass with data reuse in shared memory — slot-tiling partitions
# the N-wide NDual into ceil(N/K) tiles of width K, each processed in one kernel
# launch:
#
#   Jf = [∂f/∂x₁  ∂f/∂x₂  …  ∂f/∂xₙ]          (1×N Jacobian row per element)
#
#   tile b covers columns  [(b-1)K+1, min(bK, N)]
#   each thread carries   NDual{T,K}  with those K slots live, rest zero
#
# The primal f(x) is recomputed in each of the ceil(N/K) launches (cost), but
# register usage per thread drops from O(N) to O(K), restoring warp occupancy.
# This is the GPU spatial analogue of ForwardDiff's CPU chunk mode, where N is
# the Jacobian width and K is the chunk size.
#
# ── N vs D ───────────────────────────────────────────────────────────────────
# With D differentiable input parameters, the total slot count is
#   N = Σᵢ dof(inputᵢ),   dof = 1 (real),  dof = 2 (complex)
# so N ≥ D in general.  For all-real inputs dof = 1 for every input and N = D
# exactly — this is a consequence of the slot definition, not a separate choice.
# The tiling logic is uniform over N regardless; the real/complex distinction
# only affects how N is computed from D (via _broadcast_elem_dof_type).
#
# ── Slot-tiled execution (reduce register pressure for large N) ───────────────
#    Background: with D differentiable inputs, the total slot count is
#    N = Σᵢ dof(inputᵢ) where dof = 1 (real) or 2 (complex).  Currently every
#    thread carries ONE NDual{T,N} whose N partials cover ALL D inputs at once.
#
#    Slot-tiling partitions those N slots across ceil(N/K) kernel launches:
#      batch b → slots (b-1)K+1 .. bK:  only these inputs wrapped as NDual{T,K},
#                                        all others passed as plain T.
#    Each thread carries NDual{T,K} instead of NDual{T,N}, using (K+1)·(sizeof T/4)
#    registers instead of (N+1)·(sizeof T/4).  Partial results from each batch are
#    assembled into the full gradient vector after all ceil(N/K) launches complete.
#
#    Cost: ceil(N/K) re-evaluations of f on the same input data.
#    Useful when N > ~8 and register pressure is reducing warp occupancy.
#    Ref: CUDA occupancy calculator —
#    https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy
#
# ── Memory complexity: forward (NDual) vs reverse mode ───────────────────────
# Let M = number of output elements in the broadcast (length of y in y .= f.(args...)).
# Each of the M output elements is computed by one GPU thread carrying NDual{T,N},
# so the output dual array has M·(N+1) scalars.  For N total slots (one pass, K=N):
#
#   Forward (NDual):   O(M·N·sizeof T)   — write N gradient arrays of length M;
#                                          no tape, sequential coalesced access.
#   Reverse mode:      O(M·N·sizeof T)   — same gradient storage,
#                      + O(M·depth)       — forward tape for backward pass
#                                          (random-access reads, cache-unfriendly).
#
# Both are O(M·N) in gradient storage, but reverse mode carries an additional
# tape term proportional to the computation graph depth.  For shallow element-wise
# broadcasts (depth ~ constant) this is negligible; for deep networks it dominates.
# NDual avoids the tape entirely at the cost of recomputing the primal ceil(N/K)
# times when tiling is used:
#
#   Tiled forward:     O(M·K·sizeof T)   peak memory per launch (K < N),
#                      ceil(N/K) passes  over input data.

# ── Cholesky factorization for matrices of NDuals ─────────────────────────────────
#
# Forward-mode Cholesky derivative.  For A = L·Lᵀ (lower Cholesky) with symmetric
# perturbation Ȧ, the corresponding perturbation of L is
#
#   L̇ = L · Φ(L⁻¹ · Ȧ · L⁻ᵀ)
#
# where Φ zeroes the strict upper triangle and halves the diagonal:
#
#   Φ(S)ᵢⱼ = Sᵢⱼ    for i > j   (strict lower triangle)
#   Φ(S)ᵢᵢ = Sᵢᵢ/2  for i = j   (diagonal halved)
#   Φ(S)ᵢⱼ = 0      for i < j   (upper triangle)
#
# Equivalently: Φ(S) = LowerTriangular(S) - Diagonal(diag(S)/2).
#
# The rule computes the primal Cholesky once in Float64, then applies the derivative
# formula N times (once per partial slot) using triangular solves, avoiding the need
# to run the full Cholesky algorithm on NDual-typed elements.

function _cholesky_ndual_fwd(
    L::AbstractMatrix{T}, Ȧ::AbstractMatrix{T}
) where {T<:IEEEFloat}
    Lt = LinearAlgebra.LowerTriangular(L)
    S = Lt \ (Ȧ / Lt')
    Φ = LinearAlgebra.LowerTriangular(S) - LinearAlgebra.Diagonal(diag(S) / 2)
    return Matrix(Lt * Φ)
end

function LinearAlgebra.cholesky(
    A::AbstractMatrix{NDual{T,N}},
    (::LinearAlgebra.NoPivot)=LinearAlgebra.NoPivot();
    check::Bool=true,
) where {T<:IEEEFloat,N}
    A₀ = map(ndual_value, A)
    F₀ = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(A₀); check)
    L₀ = Matrix(F₀.L)                    # dense lower-triangular Matrix{T}
    L̇s = ntuple(k -> _cholesky_ndual_fwd(L₀, map(x -> ndual_partial(x, k), A)), Val(N))
    n = size(L₀, 1)
    L_nd = Matrix{NDual{T,N}}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        L_nd[i, j] = NDual{T,N}(L₀[i, j], ntuple(k -> L̇s[k][i, j], Val(N)))
    end
    return LinearAlgebra.Cholesky(L_nd, 'L', 0)
end

# Hermitian and Symmetric wrappers: materialise the symmetric view, then defer to
# the Matrix{NDual} method above.
#
# Critically, we call `copytri!` to mirror the active triangle before passing the
# matrix to `cholesky(::Matrix{NDual})`.  Without this, the `_cholesky_ndual_fwd`
# helper would see raw (unmirrored) NDual values for the inactive triangle, producing
# wrong partial derivatives for those positions.  For real NDual, Hermitian and
# Symmetric are equivalent (conj is identity), so a single `copytri!` suffices.
for _WrapType in (:Hermitian, :Symmetric)
    @eval function LinearAlgebra.cholesky(
        A::LinearAlgebra.$_WrapType{NDual{T,N},<:StridedMatrix{NDual{T,N}}},
        (::LinearAlgebra.NoPivot)=LinearAlgebra.NoPivot();
        check::Bool=true,
    ) where {T<:IEEEFloat,N}
        data = LinearAlgebra.copytri!(copy(A.data), A.uplo)
        return LinearAlgebra.cholesky(data, LinearAlgebra.NoPivot(); check)
    end
end

# ── Symmetric / Hermitian matrix operations for NDual ────────────────────────────────
#
# LinearAlgebra's BLAS-backed `mul!` specialisations don't accept NDual elements.
# Materialise the lazy symmetric/hermitian wrapper to a plain Matrix{NDual} before
# dispatching, so the generic (non-BLAS) matrix multiply is used.
for _WrapType in (:Symmetric, :Hermitian)
    @eval begin
        # AbstractVecOrMat covers vectors; the AbstractMatrix overloads below resolve
        # the ambiguity with LinearAlgebra's *(AbstractMatrix, AbstractMatrix) when B
        # is a plain Matrix (LinearAlgebra wins on B, we win on A — ambiguous without
        # a more specific method that wins on both).
        function Base.:*(
            A::LinearAlgebra.$_WrapType{NDual{T,N},<:AbstractMatrix{NDual{T,N}}},
            B::Union{StridedVector,StridedMatrix},
        ) where {T<:IEEEFloat,N}
            return Matrix(A) * B
        end

        function Base.:*(
            A::Union{StridedVector,StridedMatrix},
            B::LinearAlgebra.$_WrapType{NDual{T,N},<:AbstractMatrix{NDual{T,N}}},
        ) where {T<:IEEEFloat,N}
            return A * Matrix(B)
        end
    end
end

# logdet(Cholesky{NDual}): 2·∑ᵢ log(Lᵢᵢ).
# The generic LinearAlgebra path reaches this formula via `sum(log, diag(C.L))`, but
# `diag(LowerTriangular{NDual})` may trigger a BLAS-adjacent specialisation.  Spelling
# it out explicitly avoids that ambiguity.
function LinearAlgebra.logdet(
    C::LinearAlgebra.Cholesky{NDual{T,N},Matrix{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    L = C.L
    n = size(L, 1)
    s = log(L[1, 1])
    for i in 2:n
        s = s + log(L[i, i])
    end
    return 2 * s
end

# ── Array reductions for NDual ─────────────────────────────────────────────────
# Base's generic mapreduce_impl is @noinline, which prevents the compiler from
# fusing the inner reduction loop with surrounding code.  For NDual{T,N} compound
# types this is particularly costly: the noinline barrier defeats register-level
# accumulation of all N partial slots simultaneously.  These inlineable overrides
# replace the barrier with a simple sequential left-fold that LLVM can optimise.
@inline function _ndual_mapreduce_impl(
    f::F, op::O, A::AbstractArray{<:NDual{T,N}}, ifirst::Integer, ilast::Integer
) where {F,O,T,N}
    ifirst > ilast && return Base.mapreduce_empty(f, op, eltype(A))
    @inbounds acc = f(A[ifirst])
    @inbounds for i in (ifirst + 1):ilast
        acc = op(acc, f(A[i]))
    end
    return acc
end

@inline function Base.mapreduce_impl(
    f::F, op::O, A::AbstractArray{<:NDual{T,N}}, ifirst::Integer, ilast::Integer
) where {F,O,T,N}
    return _ndual_mapreduce_impl(f, op, A, ifirst, ilast)
end

@inline function Base.mapreduce_impl(
    f::F,
    op::Union{typeof(max),typeof(min)},
    A::AbstractArray{<:NDual{T,N}},
    ifirst::Int,
    ilast::Int,
) where {F,T,N}
    return _ndual_mapreduce_impl(f, op, A, ifirst, ilast)
end

# 6-arg form (blksize is unused; pairwise recursion is never beneficial for NDual).
@inline function Base.mapreduce_impl(
    f::F, op::O, A::AbstractArray{<:NDual{T,N}}, ifirst::Integer, ilast::Integer, ::Int
) where {F,O,T,N}
    return Base.mapreduce_impl(f, op, A, ifirst, ilast)
end

# Infer at rule-build time whether `sig` has a scalar IEEEFloat output, allowing a
# rule to skip a redundant primal type-check call for known-scalar functions.
#
# Uses `Base.return_types`, which is a best-effort hint: it may return `[Any]` for
# type-unstable functions or under some world-age conditions. In those cases this
# function safely returns `false`, falling through to the runtime primal check. There
# is no correctness risk from a missed inference and only a missed optimisation.
function _nfwd_infer_scalar_output(sig::Type{<:Tuple})
    F = sig.parameters[1]
    Base.issingletontype(F) || return false
    argtypes = Tuple{(sig.parameters[i] for i in 2:length(sig.parameters))...}
    rt = Base.return_types(F.instance, argtypes)
    return !isempty(rt) && rt[1] <: IEEEFloat
end

@inline function _nfwd_check_chunk_size(chunk_size::Integer)
    chunk_size > 0 && return Int(chunk_size)
    throw(ArgumentError("`chunk_size` must be a positive integer, got $chunk_size."))
end

# Shared preamble for frule/rrule builders: validate chunk_size, callable sig, and
# debug_mode.
@inline function _nfwd_validate(sig, chunk_size::Integer; debug_mode=false)
    chunk_size = _nfwd_check_chunk_size(chunk_size)
    _nfwd_check_callable_sig(sig)
    debug_mode && throw(ArgumentError("nfwd does not currently support `debug_mode=true`."))
    return chunk_size
end

# Conservative SIMD-friendly default: 8 lanes covers one AVX-512 register (8×Float64)
# and two AVX2 registers. Chunk sizes beyond 8 add register pressure without
# proportional throughput gains on most hardware.
const _NFWD_PREFERRED_CHUNK_SIZE = 8

@inline function _nfwd_default_chunk_size(x::Tuple)
    return max(1, min(sum(_nfwd_input_dof, x), _NFWD_PREFERRED_CHUNK_SIZE))
end

# Type-level DOF: returns the number of differentiable scalar components for a
# concrete type, or `nothing` when the size cannot be determined from the type
# alone (e.g. heap-allocated Array whose length is a runtime value).
@inline _nfwd_type_dof(::Type{<:IEEEFloat}) = 1
@inline _nfwd_type_dof(::Type{<:Complex{<:IEEEFloat}}) = 2
@inline _nfwd_type_dof(T::Type{<:Tuple}) = sum(_nfwd_type_dof, T.parameters; init=0)
@inline _nfwd_type_dof(::Type{<:AbstractArray}) = nothing
@inline _nfwd_type_dof(::Type) = 0

@inline function _nfwd_sig_dof(::Type{sig}) where {sig<:Tuple}
    params = sig.parameters
    total = 0
    for i in 2:length(params)
        d = _nfwd_type_dof(params[i])
        d === nothing && return nothing
        total += d
    end
    return total
end

@inline function _nfwd_sig_default_chunk_size(::Type{sig}) where {sig<:Tuple}
    dof = _nfwd_sig_dof(sig)
    preferred = _NFWD_PREFERRED_CHUNK_SIZE
    return dof === nothing ? preferred : max(1, min(dof, preferred))
end

@inline function _nfwd_resolve_chunk_size(chunk_size, x::Tuple)
    return if isnothing(chunk_size)
        _nfwd_default_chunk_size(x)
    else
        _nfwd_check_chunk_size(chunk_size)
    end
end

@inline _nfwd_is_supported_scalar(::Type{<:IEEEFloat}) = true
@inline _nfwd_is_supported_scalar(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nfwd_is_supported_scalar(::Type) = false

@inline _nfwd_tuple_primal_supported(::Tuple{}) = true
@inline function _nfwd_tuple_primal_supported(x::Tuple)
    return _nfwd_is_supported_primal(first(x)) && _nfwd_tuple_primal_supported(Base.tail(x))
end

@inline _nfwd_is_supported_primal(::IEEEFloat) = true
@inline _nfwd_is_supported_primal(::Complex{<:IEEEFloat}) = true
@inline function _nfwd_is_supported_primal(x::Array{ET}) where {ET}
    _nfwd_is_supported_scalar(ET)
end
@inline _nfwd_is_supported_primal(x::Tuple) = _nfwd_tuple_primal_supported(x)
@inline _nfwd_is_supported_primal(::Any) = false

abstract type UnsupportedError <: Exception end

struct UnsupportedInputError <: UnsupportedError
    msg::String
end

struct UnsupportedOutputError <: UnsupportedError
    msg::String
end

@inline function _nfwd_boxed_message_width(io::IO, prefix::AbstractString)
    cols = get(io, :displaysize, displaysize(io))[2]
    return max(20, cols - textwidth(prefix))
end

function _nfwd_wrap_boxed_line(line, width::Int)
    text = string(line)
    isempty(text) && return (text,)
    width < 1 && return (text,)
    textwidth(text) <= width && return (text,)

    wrapped = String[]
    remaining = text
    while textwidth(remaining) > width
        split_idx = nothing
        for idx in eachindex(remaining)
            textwidth(SubString(remaining, 1, idx)) > width && break
            remaining[idx] == ' ' && (split_idx = idx)
        end
        if isnothing(split_idx)
            split_idx = firstindex(remaining)
            for idx in eachindex(remaining)
                textwidth(SubString(remaining, firstindex(remaining), idx)) > width && break
                split_idx = idx
            end
        end
        push!(wrapped, rstrip(SubString(remaining, firstindex(remaining), split_idx)))
        remaining = lstrip(SubString(remaining, nextind(remaining, split_idx)))
        isempty(remaining) && break
    end
    isempty(remaining) || push!(wrapped, remaining)
    return Tuple(wrapped)
end

function _nfwd_print_boxed_error(io::IO, lines)
    first_item = iterate(lines)
    isnothing(first_item) && return nothing
    line, state = first_item
    rest_prefix = "  │ "
    first_width = _nfwd_boxed_message_width(io, "")
    rest_width = _nfwd_boxed_message_width(io, rest_prefix)
    first_wrapped = _nfwd_wrap_boxed_line(line, first_width)
    println(io, first(first_wrapped))
    for wrapped_line in Base.tail(first_wrapped)
        println(io, rest_prefix, wrapped_line)
    end
    while true
        item = iterate(lines, state)
        isnothing(item) && break
        line, state = item
        for wrapped_line in _nfwd_wrap_boxed_line(line, rest_width)
            println(io, rest_prefix, wrapped_line)
        end
    end
    print(io, "  └")
end

@inline function Base.showerror(
    io::IO, err::Union{UnsupportedInputError,UnsupportedOutputError}
)
    return _nfwd_print_boxed_error(io, split(err.msg, '\n'))
end

@inline _nfwd_supported_input_summary() = "IEEEFloat scalars, Complex{<:IEEEFloat} scalars, and dense Arrays with those element types"

@inline _nfwd_supported_output_summary() = "IEEEFloat scalars, Complex{<:IEEEFloat} scalars, dense Arrays with those element types, and tuples thereof"

@inline _nfwd_shape_summary(::IEEEFloat) = "scalar"
@inline _nfwd_shape_summary(::Complex{<:IEEEFloat}) = "scalar"
@inline _nfwd_shape_summary(x::AbstractArray) = "size $(size(x))"
@inline _nfwd_shape_summary(x::Tuple) = "tuple length $(length(x))"
@inline _nfwd_shape_summary(::Any) = "not size-bearing"

@inline _nfwd_value_summary(x) = "$(typeof(x)) ($(_nfwd_shape_summary(x)))"

@inline function _nfwd_inputs_summary(xs::Tuple)
    isempty(xs) && return "  (none)"
    return join(ntuple(i -> "  $i. $(_nfwd_value_summary(xs[i]))", Val(length(xs))), '\n')
end

@inline _nfwd_input_error(x) = throw(
    UnsupportedInputError(
        "nfwd input unsupported.\n" *
        "Supported nfwd inputs: $(_nfwd_supported_input_summary()).\n" *
        "Input:\n" *
        "  $(_nfwd_value_summary(x))",
    ),
)

@inline function _nfwd_output_error(y)
    throw(
        UnsupportedOutputError(
            "nfwd output unsupported.\n" *
            "Supported nfwd inputs: $(_nfwd_supported_input_summary()).\n" *
            "Supported nfwd outputs: $(_nfwd_supported_output_summary()).\n" *
            "Output:\n" *
            "  $(_nfwd_value_summary(y))",
        ),
    )
end

@inline function _nfwd_output_error(xs::Tuple, y)
    throw(
        UnsupportedOutputError(
            "nfwd output unsupported.\n" *
            "Supported nfwd inputs: $(_nfwd_supported_input_summary()).\n" *
            "Supported nfwd outputs: $(_nfwd_supported_output_summary()).\n" *
            "Inputs:\n" *
            "$(_nfwd_inputs_summary(xs))\n" *
            "Output:\n" *
            "  $(_nfwd_value_summary(y))",
        ),
    )
end

@inline function _nfwd_check_primal(x)
    _nfwd_is_supported_primal(x) || _nfwd_input_error(x)
    return x
end

@inline function _nfwd_check_callable_sig(sig::Type{<:Tuple})
    F = sig.parameters[1]
    Base.issingletontype(F) || throw(
        ArgumentError(
            "nfwd only supports stateless callables for rule construction. Got $F. " *
            "Stateless callables are required because nfwd re-evaluates the function " *
            "multiple times with different tangent seeds; a mutable callable would " *
            "produce incorrect gradients on the second and subsequent evaluations.",
        ),
    )
    f = F.instance
    argsig = Tuple{(sig.parameters[i] for i in 2:length(sig.parameters))...}
    hasmethod(f, argsig) && return sig
    throw(ArgumentError("nfwd rule construction expected a callable signature, got $sig."))
end

#
# ── Canonical slot traversal ──────────────────────────────────────────────────────
#
# Every supported nfwd primal decomposes into a fixed number of scalar "slots" in
# a canonical order:
#   • IEEEFloat              → 1 slot  (the value itself)
#   • Complex{<:IEEEFloat}   → 2 slots (real, imag)
#   • AbstractArray{<:above} → one slot per scalar component, in eachindex order
#   • Tuple of the above     → concatenation, left to right
#
# `_nfwd_fold_slots` and `_nfwd_unfold_slots` define this order exactly once.
# All DOF counting, basis seeding, and gradient scatter must use these helpers so
# that the slot order is guaranteed to agree everywhere.

"""
    _nfwd_fold_slots(f, init, x, state) -> (acc, state)

Left-fold over the scalar slots of `x` in canonical order.  Each slot corresponds
to one differentiable scalar degree of freedom.  Real IEEE-float values contribute
one floating-point slot.  Complex IEEE-float values contribute two scalar slots,
visited as real then imaginary.  Tuples are visited left to right, and arrays are
visited in `eachindex` order.

Each slot visit calls `(acc, state) = f(acc, x_leaf, slot_index_within_leaf, state)`
and returns the updated accumulator and state.  The slot cursor should be threaded
through `state` by the caller.
"""
@inline function _nfwd_fold_slots(f::F, init, x::IEEEFloat, state) where {F}
    return f(init, x, 1, state)
end

@inline function _nfwd_fold_slots(f::F, init, x::Complex{<:IEEEFloat}, state) where {F}
    acc, state = f(init, x, 1, state)  # real part
    return f(acc, x, 2, state)          # imag part
end

@inline function _nfwd_fold_slots(
    f::F, init, x::AbstractArray{T}, state
) where {F,T<:IEEEFloat}
    acc = init
    @inbounds for i in eachindex(x)
        acc, state = f(acc, x, i, state)
    end
    return acc, state
end

@inline function _nfwd_fold_slots(
    f::F, init, x::AbstractArray{Complex{T}}, state
) where {F,T<:IEEEFloat}
    acc = init
    @inbounds for i in eachindex(x)
        acc, state = f(acc, x, 2i - 1, state)  # real part of element i
        acc, state = f(acc, x, 2i, state)       # imag part of element i
    end
    return acc, state
end

@inline _nfwd_fold_slots(f::F, init, x::Tuple{}, state) where {F} = (init, state)
@inline function _nfwd_fold_slots(f::F, init, x::Tuple, state) where {F}
    acc, state = _nfwd_fold_slots(f, init, first(x), state)
    return _nfwd_fold_slots(f, acc, Base.tail(x), state)
end

"""
    _nfwd_unfold_slots(f, x, state) -> (rebuilt, state)

Map-like structural rebuild over the primitive leaves of `x`.  Each slot
corresponds to one differentiable scalar degree of freedom (same semantics as
`_nfwd_fold_slots`).  Each leaf visit calls `(result, state) = f(x_leaf, state)` and
returns the rebuilt value for that leaf position.  For tuples, the unfold recurses
left to right and collects per-leaf results into a new tuple.

The returned value at each leaf position may have a different type from the input
leaf (e.g. seeding produces NTuples from scalar inputs).  The slot cursor should
be threaded through `state` by the caller.

`_nfwd_fold_slots` and `_nfwd_unfold_slots` agree on traversal order: tuples left to right,
arrays in `eachindex` order.  Within each leaf, the number of slots consumed equals
`_nfwd_input_dof(leaf)`.
"""
@inline function _nfwd_unfold_slots(
    f::F,
    x::Union{
        IEEEFloat,
        Complex{<:IEEEFloat},
        AbstractArray{<:IEEEFloat},
        AbstractArray{<:Complex{<:IEEEFloat}},
    },
    state,
) where {F}
    return f(x, state)
end

@inline _nfwd_unfold_slots(f::F, x::Tuple{}, state) where {F} = ((), state)
@inline function _nfwd_unfold_slots(f::F, x::Tuple, state) where {F}
    head, state = _nfwd_unfold_slots(f, first(x), state)
    tail, state = _nfwd_unfold_slots(f, Base.tail(x), state)
    return (head, tail...), state
end

# ── DOF counting ─────────────────────────────────────────────────────────────────

@inline _nfwd_input_dof(x::IEEEFloat) = 1
@inline _nfwd_input_dof(x::Complex{<:IEEEFloat}) = 2
@inline _nfwd_input_dof(x::AbstractArray{<:IEEEFloat}) = length(x)
@inline _nfwd_input_dof(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 2 * length(x)
@inline _nfwd_input_dof(x::Tuple) = sum(_nfwd_input_dof, x; init=0)

# ──────────────────────────────────────────────────────────────────────────
# `NDualArray{Element, N, D, A, Wrapped}` — parallel-arrays canonical V for arrays.
#
# `NDualArray` and a plain `Array` of `NDual`s carry the same information, but keeping the
# primal and each lane's partials as separate same-typed arrays (rather than interleaving them
# per element) is more friendly: `primal` is a genuine `A` that can be passed straight to a
# BLAS/LAPACK `ccall`, and each `partials[k]` is too, so a width-`N` rule reduces to `N+1`
# native array calls instead of needing to de-interleave a `Array{NDual}` first.
#
# `primal::A` aliases user storage; `partials::NTuple{N, A}` holds slot-local
# lane tangents (separately allocated, same shape). `Wrapped` is determined by `(Element, N)`
# — `NDual{T, N}` for real `Element=T<:IEEEFloat` and `Complex{NDual{T, N}}` for
# `Element=Complex{T<:IEEEFloat}`. Subtype `AbstractArray{Wrapped, D}` so
# element-wise code through the array interface continues to dispatch; element
# access is lazy (constructs an `NDual` on the fly from the parallel-arrays storage).
#
# Mooncake-namespace method extensions (`primal`/`tangent`/`unpack_ndual`/
# `unlift`) for `NDualArray` live in `src/tangents/lifted.jl`.
# ──────────────────────────────────────────────────────────────────────────

const NDualEltype = Union{IEEEFloat,Complex{<:IEEEFloat}}

struct NDualArray{Element<:NDualEltype,N,D,A<:AbstractArray{Element,D},Wrapped} <:
       AbstractArray{Wrapped,D}
    primal::A
    partials::NTuple{N,A}
end

# 4-parameter outer constructors fill in `Wrapped` from `Element`.
@inline function NDualArray{Element,N,D,A}(
    p::A, ts::NTuple{N,A}
) where {Element<:IEEEFloat,N,D,A<:AbstractArray{Element,D}}
    return NDualArray{Element,N,D,A,NDual{Element,N}}(p, ts)
end
@inline function NDualArray{Element,N,D,A}(
    p::A, ts::NTuple{N,A}
) where {T<:IEEEFloat,Element<:Complex{T},N,D,A<:AbstractArray{Element,D}}
    return NDualArray{Element,N,D,A,Complex{NDual{T,N}}}(p, ts)
end

# Zero-init seed: allocate fresh slot-local partials matching the primal.
@inline function NDualArray{Element,N,D,A}(
    p::A
) where {Element<:IEEEFloat,N,D,A<:AbstractArray{Element,D}}
    return NDualArray{Element,N,D,A}(p, ntuple(_ -> zero(p), Val(N)))
end
@inline function NDualArray{Element,N,D,A}(
    p::A
) where {T<:IEEEFloat,Element<:Complex{T},N,D,A<:AbstractArray{Element,D}}
    return NDualArray{Element,N,D,A}(p, ntuple(_ -> zero(p), Val(N)))
end

# AbstractArray interface.
Base.size(a::NDualArray) = size(a.primal)
function Base.IndexStyle(::Type{<:NDualArray{<:Any,<:Any,<:Any,A}}) where {A}
    return IndexStyle(A)
end

@inline function Base.getindex(
    a::NDualArray{Element,N}, i::Vararg{Int}
) where {Element<:IEEEFloat,N}
    return NDual{Element,N}(a.primal[i...], ntuple(k -> a.partials[k][i...], Val(N)))
end
@inline function Base.setindex!(
    a::NDualArray{Element,N}, x::NDual{Element,N}, i::Vararg{Int}
) where {Element<:IEEEFloat,N}
    a.primal[i...] = x.value
    ntuple(k -> (a.partials[k][i...]=x.partials[k]; nothing), Val(N))
    return a
end
# Complex eltype: the V element is `Complex{NDual{T,N}}` (real/imag each an `NDual`). The primal
# and per-lane partial arrays hold `Complex{T}`, so split/recombine the real and imaginary parts.
@inline function Base.getindex(
    a::NDualArray{Element,N}, i::Vararg{Int}
) where {T<:IEEEFloat,Element<:Complex{T},N}
    return _scalar_ndual(a.primal[i...], ntuple(k -> a.partials[k][i...], Val(N)))
end
@inline function Base.setindex!(
    a::NDualArray{Element,N}, x::Complex{NDual{T,N}}, i::Vararg{Int}
) where {T<:IEEEFloat,Element<:Complex{T},N}
    a.primal[i...] = Complex(x.re.value, x.im.value)
    ntuple(
        k -> (a.partials[k][i...]=Complex(x.re.partials[k], x.im.partials[k]); nothing),
        Val(N),
    )
    return a
end

# ──────────────────────────────────────────────────────────────────────────
# `NDualRef{P, N}` — canonical V for `Base.RefValue{P<:NDualEltype}` (real or complex
# scalar), the scalar analogue of `NDualArray`. Carries the same information as a `Ref` of an
# interleaved `NDual`, but the `N` per-lane scalar partials live in their own parallel `Ref`,
# not interleaved with the value, so a raw pointer taken via `pointer_from_objref` lands them
# at a parallel address (correct forward raw-pointer access). Being a *distinct* type (not a bare `RefValue`) stops the generic struct
# recursion from re-lifting it — so the seed factories, `_unlift_seed`, `_new_`,
# `lgetfield`/`lsetfield!`, and raw-pointer frules each carry an explicit branch (as for
# `NDualArray`). The slot's primal `Ref` lives in the enclosing `Lifted`, not here.
# ──────────────────────────────────────────────────────────────────────────
export NDualRef
struct NDualRef{P<:NDualEltype,N}
    partials::Base.RefValue{NTuple{N,P}}
    # Explicit inner constructor: suppresses the auto-generated implicit `NDualRef(partials)`, whose
    # `P` is unbound at `N=0` (`NTuple{0,P}` mentions no `P`) and would trip Aqua's unbound-args check.
    # All call sites use the explicit `NDualRef{P,N}(...)` form, which binds both params.
    function NDualRef{P,N}(partials::Base.RefValue{NTuple{N,P}}) where {P<:NDualEltype,N}
        return new{P,N}(partials)
    end
end
# Zero-init seed: fresh slot-local partials.
@inline function NDualRef{P,N}() where {P<:NDualEltype,N}
    return NDualRef{P,N}(Base.RefValue{NTuple{N,P}}(ntuple(_ -> zero(P), Val(N))))
end

# ──────────────────────────────────────────────────────────────────────────
# `NDualMemoryRef{Element, N, M}` — parallel-arrays wrapper for `MemoryRef`
# (Julia 1.11+). `MemoryRef` is the low-level reference-to-memory-slot
# primitive and is *not* `<: AbstractArray`, so `NDualArray` does not
# cover it. Like `NDualArray`, this carries the same information as a
# `MemoryRef` of interleaved `NDual`s but keeps the primal and each lane's
# partials as separate native `MemoryRef`s — friendlier because each is a
# genuine `MemoryRef{Element}` usable directly in a `ccall`. `partials[k]` is a
# framework-allocated `MemoryRef` at the same offset as `primal`, into a fresh
# `Memory{Element}` of the same length.
# ──────────────────────────────────────────────────────────────────────────

@static if VERSION >= v"1.11-rc4"
    export NDualMemoryRef

    struct NDualMemoryRef{Element<:NDualEltype,N,M<:Memory{Element}}
        primal::MemoryRef{Element}
        partials::NTuple{N,MemoryRef{Element}}
    end

    # Zero-init seed: allocate fresh slot-local partials at the same offset
    # as `primal`. Element types in `NDualEltype` are bits types, so undef
    # iteration is not needed.
    @inline function NDualMemoryRef{Element,N,M}(
        p::MemoryRef{Element}
    ) where {Element<:NDualEltype,N,M<:Memory{Element}}
        offset = Core.memoryrefoffset(p)
        len = length(p.mem)
        alloc_partial() = (
            mem=Memory{Element}(undef, len);
            fill!(mem, zero(Element));
            Core.memoryref(mem, offset)
        )
        return NDualMemoryRef{Element,N,M}(p, ntuple(_ -> alloc_partial(), Val(N)))
    end
end

end
