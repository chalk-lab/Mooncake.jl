# ── NDual: N-wide dual number for GPU kernel forward-mode AD ──────────────────────
# This type lives entirely within the CUDA extension and is not exported from
# Mooncake core.  All GPU broadcast forward-mode rules use it internally.
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
    NDual{T<:IEEEFloat, N} <: Real

An N-wide dual number: carries one primal `value::T` and `N` partial derivatives
`partials::NTuple{N,T}`.  It is a plain `isbits` type — lives in GPU registers and
compiles to PTX without heap allocation.

## Analogy to ForwardDiff chunk mode

ForwardDiff's chunk mode computes N directional derivatives simultaneously by using
`ForwardDiff.Dual{Tag,T,N}` — a dual number with N partial slots.  `NDual{T,N}` is
the same idea, stripped of the tag parameter and defined entirely within Mooncake:

| Type                         | Tangent width | Tag parameter | Use case                        |
|------------------------------|---------------|---------------|---------------------------------|
| `Dual{P,T}`                  | 1             | n/a           | Standard `frule!!` dispatch     |
| `ForwardDiff.Dual{Tag,T,N}`  | N             | yes           | ForwardDiff chunk mode          |
| `NDual{T,N}`                 | N             | no            | GPU kernel widening (this type) |

`NDual` is a drop-in replacement for `ForwardDiff.Dual` in GPU broadcast kernels.
Removing the tag simplifies the type signature and eliminates the ForwardDiff
dependency from GPU AD.  The arithmetic rules are identical: each operation applies
the chain rule to all N slots at once.

## NDual vs Dual: scalar leaves and flattening

`Dual{P,T}` wraps any differentiable value `P` — it threads through Mooncake's
tangent system and handles arbitrary structs transparently.

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
(`+`, `*`, `sin`, etc.) composes with `NDual` naturally because `NDual <: Real`.

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

## Chunk-mode AD via NForwardMode{N}

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

### Why standard `frule!!` cannot carry NDual tangents

`Dual{P,T}` enforces `T = tangent_type(P)`.  For `P = Float64`, `tangent_type` returns
`Float64` (width-1).  Stuffing `NDual{Float64,N}` into the tangent slot would require
`tangent_type(Float64) = NDual{Float64,N}` globally, infecting every `frule!!` in the
call graph and breaking type coherence throughout.

### NForwardMode{N}: NDual as the tangent type

The clean solution is a new AD context that overrides `tangent_type` for scalar leaves:

```julia
struct NForwardMode{N} end

# NDual is the tangent type — value field=0 by convention, partials carry N directions
tangent_type(::NForwardMode{N}, ::Type{T}) where {N, T<:IEEEFloat}          = NDual{T,N}
tangent_type(::NForwardMode{N}, ::Type{Complex{T}}) where {N, T<:IEEEFloat} = Complex{NDual{T,N}}

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
# Existing frule!! for sin — tangent(x)::NDual{T,N} in NForwardMode
frule!!(::Dual{typeof(sin)}, x::Dual{T}) where {T<:IEEEFloat} =
    Dual(sin(primal(x)), cos(primal(x)) * tangent(x))
#                        ^^^^^^^^^^^^^^^^ T (scalar) * NDual{T,N} → NDual{T,N}  ✓
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
   Passing an NDual tangent would produce a `MethodError`.  An `NDual` overload of
   `nan_tangent_guard` would be needed to use these functions inside `NForwardMode{N}`.

In practice `NForwardMode{N}` is designed for **GPU kernel boundaries** where each
width-1 pass costs a full kernel launch.  For CPU scalar ops the overhead is negligible
and there is no motivation to use chunk mode.

### `frule!!` template in NForwardMode

This pattern applies at any opaque boundary — most commonly a GPU kernel, but equally
valid for any CPU operation that needs an explicit N-wide rule (e.g. to override a
hand-coded rule that uses `nan_tangent_guard`, or to differentiate through an external
library call).  The only difference between GPU and CPU versions is the array type
(`CuArray` vs `Array`) and the absence of a kernel launch on CPU.

In NForwardMode the tangent of a `CuArray{T}` arg is `CuArray{NDual{T,N}}`, so the
NDual kernel input is built by a trivial merge — no `flatten_to_ndual` needed:

```julia
function frule!!(
    ::Dual{typeof(my_kernel!)},
    _out::Dual{<:CuArray{T}, <:CuArray{NDual{T,N}}},
    _x  ::Dual{<:CuArray{T}, <:CuArray{NDual{T,N}}},
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

    rule = build_frule(NForwardMode{N}(), typeof(f!), CuArray{T}, CuArray{T})
    rule(Dual(f!, NoTangent()), Dual(out, ∂out), Dual(x, ∂x))

    # ∂out[i].partials == (∂out[i]/∂x[1], …, ∂out[i]/∂x[N]) — full m×N Jacobian
    J = [ndual_partial.(∂out, k) for k in 1:N]
    return ndual_value.(out), J
end
```

Versus N separate width-1 passes, NForwardMode{N} needs **one** pass.  NDual is the
natural tangent type because its arithmetic is already register-friendly and no
conversion is needed at the kernel boundary.

### Open challenges

- Non-float leaves (`Int`, `Bool`, …) carry zero partial and must bypass NDual wrapping.
- Mixed-precision structs (`Float32` + `Float64` fields) require a promoted `T` or
  separate NDual blocks per precision group.
- `NForwardMode{N}` requires N to be chosen before compilation; adaptive chunk sizing
  (as in ForwardDiff) would need dynamic dispatch or recompilation.
"""
struct NDual{T<:IEEEFloat,N} <: Real
    value::T
    partials::NTuple{N,T}
end

# ── Constructors ─────────────────────────────────────────────────────────────────

# Promote a plain scalar to a NDual with zero partials (acts as a constant).
NDual{T,N}(x::Real) where {T<:IEEEFloat,N} = NDual{T,N}(T(x), ntuple(_ -> zero(T), Val(N)))

# ── Accessors ────────────────────────────────────────────────────────────────────

@inline ndual_value(d::NDual) = d.value
@inline ndual_partial(d::NDual, k::Int) = d.partials[k]
@inline ndual_partials(d::NDual) = d.partials

# ── NTuple arithmetic helpers ─────────────────────────────────────────────────────
# All fully unrolled at compile time via Val(N) — safe for GPU registers.

@inline _pt_scale(p::NTuple{N,T}, s::T) where {N,T} = ntuple(i -> s * p[i], Val(N))
@inline _pt_add(p::NTuple{N,T}, q::NTuple{N,T}) where {N,T} = ntuple(
    i -> p[i] + q[i], Val(N)
)
@inline _pt_sub(p::NTuple{N,T}, q::NTuple{N,T}) where {N,T} = ntuple(
    i -> p[i] - q[i], Val(N)
)
@inline _pt_neg(p::NTuple{N,T}) where {N,T} = ntuple(i -> -p[i], Val(N))
@inline _pt_zero(::Val{N}, ::Type{T}) where {N,T} = ntuple(_ -> zero(T), Val(N))

# ── AbstractFloat traits (needed for promote_rule with Complex etc.) ──────────────

Base.float(a::NDual) = a
Base.AbstractFloat(a::NDual) = a
Base.floatmin(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(floatmin(T))
Base.floatmax(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(floatmax(T))
Base.typemin(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(typemin(T))
Base.typemax(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(typemax(T))

# ── Zero / One ────────────────────────────────────────────────────────────────────

Base.zero(::NDual{T,N}) where {T,N} = NDual{T,N}(zero(T), _pt_zero(Val(N), T))
Base.one(::NDual{T,N}) where {T,N} = NDual{T,N}(one(T), _pt_zero(Val(N), T))
Base.zero(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(zero(T), _pt_zero(Val(N), T))
Base.one(::Type{NDual{T,N}}) where {T,N} = NDual{T,N}(one(T), _pt_zero(Val(N), T))

# ── Promotion / Conversion ────────────────────────────────────────────────────────

function Base.convert(::Type{NDual{T,N}}, x::Real) where {T,N}
    NDual{T,N}(T(x), _pt_zero(Val(N), T))
end
Base.convert(::Type{NDual{T,N}}, d::NDual{T,N}) where {T,N} = d

function Base.promote_rule(::Type{NDual{T,N}}, ::Type{S}) where {T,N,S<:Real}
    NDual{promote_type(T, S),N}
end
Base.promote_rule(::Type{NDual{T,N}}, ::Type{NDual{T,N}}) where {T,N} = NDual{T,N}
# Cross-precision: NDual{Float32,N} op NDual{Float64,N} → NDual{Float64,N}
function Base.promote_rule(::Type{NDual{T1,N}}, ::Type{NDual{T2,N}}) where {T1,T2,N}
    NDual{promote_type(T1, T2),N}
end
function Base.convert(::Type{NDual{T,N}}, d::NDual{S,N}) where {T,N,S<:IEEEFloat}
    NDual{T,N}(T(d.value), ntuple(i -> T(d.partials[i]), Val(N)))
end

# ── Arithmetic ────────────────────────────────────────────────────────────────────

function Base.:+(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    NDual{T,N}(a.value + b.value, _pt_add(a.partials, b.partials))
end
function Base.:-(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    NDual{T,N}(a.value - b.value, _pt_sub(a.partials, b.partials))
end
Base.:-(a::NDual{T,N}) where {T,N} = NDual{T,N}(-a.value, _pt_neg(a.partials))

# Product rule: d(a*b) = a*db + b*da
function Base.:*(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    return NDual{T,N}(
        a.value * b.value,
        _pt_add(_pt_scale(a.partials, b.value), _pt_scale(b.partials, a.value)),
    )
end

# Quotient rule: d(a/b) = (da - (a/b)*db) / b
function Base.:/(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    v = a.value / b.value
    return NDual{T,N}(
        v, _pt_scale(_pt_sub(a.partials, _pt_scale(b.partials, v)), inv(b.value))
    )
end

Base.inv(a::NDual{T,N}) where {T,N} = one(NDual{T,N}) / a

Base.muladd(a::NDual{T,N}, b::NDual{T,N}, c::NDual{T,N}) where {T,N} = a * b + c

# ── Integer and real power ────────────────────────────────────────────────────────

# d(x^n) = n * x^(n-1) * dx  (ifelse keeps this branchless; see file header)
function Base.:^(a::NDual{T,N}, n::Integer) where {T,N}
    v = a.value^n
    dv = ifelse(iszero(n), zero(T), T(n) * a.value^(n - 1))
    return NDual{T,N}(v, _pt_scale(a.partials, dv))
end

function Base.:^(a::NDual{T,N}, b::Real) where {T,N}
    v = a.value^T(b)
    dv = ifelse(iszero(T(b)), zero(T), T(b) * a.value^(T(b) - one(T)))
    return NDual{T,N}(v, _pt_scale(a.partials, dv))
end

function Base.:^(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    # d(a^b) = a^b * (b/a * da + log(a) * db)
    v = a.value^b.value
    coeff_a = b.value / a.value
    coeff_b = log(a.value)
    return NDual{T,N}(
        v,
        _pt_scale(
            _pt_add(_pt_scale(a.partials, coeff_a), _pt_scale(b.partials, coeff_b)), v
        ),
    )
end

# d(b^a)/da = b^a * log(b)  (b a plain Real, a the NDual)
function Base.:^(b::Real, a::NDual{T,N}) where {T,N}
    v = T(b)^a.value
    NDual{T,N}(v, _pt_scale(a.partials, v * T(log(b))))
end

# ── Math functions ─────────────────────────────────────────────────────────────────
# Each follows: f(Dual(v,p)) = Dual(f(v), f'(v)*p)

# Trig
function Base.sin(a::NDual{T,N}) where {T,N}
    NDual{T,N}(sin(a.value), _pt_scale(a.partials, cos(a.value)))
end
function Base.cos(a::NDual{T,N}) where {T,N}
    NDual{T,N}(cos(a.value), _pt_scale(a.partials, -sin(a.value)))
end
function Base.tan(a::NDual{T,N}) where {T,N}
    NDual{T,N}(tan(a.value), _pt_scale(a.partials, inv(cos(a.value))^2))
end
function Base.asin(a::NDual{T,N}) where {T,N}
    NDual{T,N}(asin(a.value), _pt_scale(a.partials, inv(sqrt(one(T) - a.value^2))))
end
function Base.acos(a::NDual{T,N}) where {T,N}
    NDual{T,N}(acos(a.value), _pt_scale(a.partials, -inv(sqrt(one(T) - a.value^2))))
end
function Base.atan(a::NDual{T,N}) where {T,N}
    NDual{T,N}(atan(a.value), _pt_scale(a.partials, inv(one(T) + a.value^2)))
end
function Base.atan(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    r2 = a.value^2 + b.value^2
    return NDual{T,N}(
        atan(a.value, b.value),
        _pt_scale(
            _pt_sub(_pt_scale(a.partials, b.value), _pt_scale(b.partials, a.value)), inv(r2)
        ),
    )
end

# Hyperbolic
function Base.sinh(a::NDual{T,N}) where {T,N}
    NDual{T,N}(sinh(a.value), _pt_scale(a.partials, cosh(a.value)))
end
function Base.cosh(a::NDual{T,N}) where {T,N}
    NDual{T,N}(cosh(a.value), _pt_scale(a.partials, sinh(a.value)))
end
function Base.tanh(a::NDual{T,N}) where {T,N}
    NDual{T,N}(tanh(a.value), _pt_scale(a.partials, one(T) - tanh(a.value)^2))
end
function Base.asinh(a::NDual{T,N}) where {T,N}
    NDual{T,N}(asinh(a.value), _pt_scale(a.partials, inv(sqrt(a.value^2 + one(T)))))
end
function Base.acosh(a::NDual{T,N}) where {T,N}
    NDual{T,N}(acosh(a.value), _pt_scale(a.partials, inv(sqrt(a.value^2 - one(T)))))
end
function Base.atanh(a::NDual{T,N}) where {T,N}
    NDual{T,N}(atanh(a.value), _pt_scale(a.partials, inv(one(T) - a.value^2)))
end

# Reciprocal hyperbolic: sech, csch, coth and their inverses.
function Base.sech(a::NDual{T,N}) where {T,N}
    sv = sech(a.value)
    NDual{T,N}(sv, _pt_scale(a.partials, -tanh(a.value) * sv))
end
function Base.csch(a::NDual{T,N}) where {T,N}
    cv = csch(a.value)
    NDual{T,N}(cv, _pt_scale(a.partials, -coth(a.value) * cv))
end
function Base.coth(a::NDual{T,N}) where {T,N}
    sv = csch(a.value)
    NDual{T,N}(coth(a.value), _pt_scale(a.partials, -(sv^2)))
end
function Base.asech(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        asech(a.value), _pt_scale(a.partials, -inv(a.value * sqrt(one(T) - a.value^2)))
    )
end
function Base.acsch(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        acsch(a.value), _pt_scale(a.partials, -inv(abs(a.value) * sqrt(one(T) + a.value^2)))
    )
end
function Base.acoth(a::NDual{T,N}) where {T,N}
    NDual{T,N}(acoth(a.value), _pt_scale(a.partials, inv(one(T) - a.value^2)))
end

# Exp / Log
function Base.exp(a::NDual{T,N}) where {T,N}
    (ev=exp(a.value); NDual{T,N}(ev, _pt_scale(a.partials, ev)))
end
function Base.exp2(a::NDual{T,N}) where {T,N}
    (ev=exp2(a.value); NDual{T,N}(ev, _pt_scale(a.partials, ev * T(log(2)))))
end
function Base.exp10(a::NDual{T,N}) where {T,N}
    (ev=exp10(a.value); NDual{T,N}(ev, _pt_scale(a.partials, ev * T(log(10)))))
end
function Base.log(a::NDual{T,N}) where {T,N}
    NDual{T,N}(log(a.value), _pt_scale(a.partials, inv(a.value)))
end
function Base.log2(a::NDual{T,N}) where {T,N}
    NDual{T,N}(log2(a.value), _pt_scale(a.partials, inv(a.value * T(log(2)))))
end
function Base.log10(a::NDual{T,N}) where {T,N}
    NDual{T,N}(log10(a.value), _pt_scale(a.partials, inv(a.value * T(log(10)))))
end
function Base.log1p(a::NDual{T,N}) where {T,N}
    NDual{T,N}(log1p(a.value), _pt_scale(a.partials, inv(one(T) + a.value)))
end
function Base.expm1(a::NDual{T,N}) where {T,N}
    NDual{T,N}(expm1(a.value), _pt_scale(a.partials, exp(a.value)))
end

# Two-argument log: log(b, x) = log(x)/log(b); d/dx = inv(x * log(b)).
function Base.log(b::Real, a::NDual{T,N}) where {T,N}
    NDual{T,N}(log(b, a.value), _pt_scale(a.partials, inv(a.value * T(log(b)))))
end

# ldexp(a, n) = a * 2^n — linear; derivative = 2^n.
function Base.ldexp(a::NDual{T,N}, n::Integer) where {T,N}
    NDual{T,N}(ldexp(a.value, n), _pt_scale(a.partials, T(exp2(n))))
end

# Roots
function Base.sqrt(a::NDual{T,N}) where {T,N}
    (sv=sqrt(a.value); NDual{T,N}(sv, _pt_scale(a.partials, inv(2 * sv))))
end
function Base.cbrt(a::NDual{T,N}) where {T,N}
    (cv=cbrt(a.value); NDual{T,N}(cv, _pt_scale(a.partials, inv(3 * cv^2))))
end

# Absolute value and sign
function Base.abs(a::NDual{T,N}) where {T,N}
    NDual{T,N}(abs(a.value), _pt_scale(a.partials, sign(a.value)))
end
function Base.abs2(a::NDual{T,N}) where {T,N}
    NDual{T,N}(abs2(a.value), _pt_scale(a.partials, 2 * a.value))
end
Base.sign(a::NDual{T,N}) where {T,N} = NDual{T,N}(sign(a.value), _pt_zero(Val(N), T))

# sincos — fused sin+cos; returns (sin(a), cos(a)) as a tuple of NDuals.
function Base.sincos(a::NDual{T,N}) where {T,N}
    sv, cv = sincos(a.value)
    NDual{T,N}(sv, _pt_scale(a.partials, cv)), NDual{T,N}(cv, _pt_scale(a.partials, -sv))
end

# sinpi / cospi — sin(π·x) and cos(π·x); derivative gains a π factor.
function Base.sinpi(a::NDual{T,N}) where {T,N}
    NDual{T,N}(sinpi(a.value), _pt_scale(a.partials, T(π) * cospi(a.value)))
end
function Base.cospi(a::NDual{T,N}) where {T,N}
    NDual{T,N}(cospi(a.value), _pt_scale(a.partials, -T(π) * sinpi(a.value)))
end

# Reciprocal trigonometric: sec, csc, cot and their inverses.
function Base.sec(a::NDual{T,N}) where {T,N}
    sv = sec(a.value)
    NDual{T,N}(sv, _pt_scale(a.partials, sv * tan(a.value)))
end
function Base.csc(a::NDual{T,N}) where {T,N}
    cv = csc(a.value)
    NDual{T,N}(cv, _pt_scale(a.partials, -cv * cot(a.value)))
end
function Base.cot(a::NDual{T,N}) where {T,N}
    cv = cot(a.value)
    NDual{T,N}(cv, _pt_scale(a.partials, -(one(T) + cv^2)))
end
function Base.asec(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        asec(a.value), _pt_scale(a.partials, inv(abs(a.value) * sqrt(a.value^2 - one(T))))
    )
end
function Base.acsc(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        acsc(a.value), _pt_scale(a.partials, -inv(abs(a.value) * sqrt(a.value^2 - one(T))))
    )
end
function Base.acot(a::NDual{T,N}) where {T,N}
    NDual{T,N}(acot(a.value), _pt_scale(a.partials, -inv(one(T) + a.value^2)))
end

# Degree-based trigonometric functions — argument in degrees, derivative gains π/180.
function Base.sind(a::NDual{T,N}) where {T,N}
    NDual{T,N}(sind(a.value), _pt_scale(a.partials, T(deg2rad(cosd(a.value)))))
end
function Base.cosd(a::NDual{T,N}) where {T,N}
    NDual{T,N}(cosd(a.value), _pt_scale(a.partials, T(-deg2rad(sind(a.value)))))
end
function Base.tand(a::NDual{T,N}) where {T,N}
    tv = tand(a.value)
    NDual{T,N}(tv, _pt_scale(a.partials, T(deg2rad(one(T) + tv^2))))
end
function Base.secd(a::NDual{T,N}) where {T,N}
    sv = secd(a.value)
    NDual{T,N}(sv, _pt_scale(a.partials, T(deg2rad(sv * tand(a.value)))))
end
function Base.cscd(a::NDual{T,N}) where {T,N}
    cv = cscd(a.value)
    NDual{T,N}(cv, _pt_scale(a.partials, T(-deg2rad(cv * cotd(a.value)))))
end
function Base.cotd(a::NDual{T,N}) where {T,N}
    cv = cotd(a.value)
    NDual{T,N}(cv, _pt_scale(a.partials, T(-deg2rad(one(T) + cv^2))))
end
function Base.asind(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        asind(a.value), _pt_scale(a.partials, inv(T(deg2rad(sqrt(one(T) - a.value^2)))))
    )
end
function Base.acosd(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        acosd(a.value), _pt_scale(a.partials, -inv(T(deg2rad(sqrt(one(T) - a.value^2)))))
    )
end
function Base.atand(a::NDual{T,N}) where {T,N}
    NDual{T,N}(atand(a.value), _pt_scale(a.partials, inv(T(deg2rad(one(T) + a.value^2)))))
end
function Base.asecd(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        asecd(a.value),
        _pt_scale(a.partials, inv(T(deg2rad(abs(a.value) * sqrt(a.value^2 - one(T)))))),
    )
end
function Base.acscd(a::NDual{T,N}) where {T,N}
    NDual{T,N}(
        acscd(a.value),
        _pt_scale(a.partials, -inv(T(deg2rad(abs(a.value) * sqrt(a.value^2 - one(T)))))),
    )
end
function Base.acotd(a::NDual{T,N}) where {T,N}
    NDual{T,N}(acotd(a.value), _pt_scale(a.partials, -inv(T(deg2rad(one(T) + a.value^2)))))
end

# Angle unit conversions — linear transforms; derivative is the constant scale factor.
function Base.deg2rad(a::NDual{T,N}) where {T,N}
    NDual{T,N}(deg2rad(a.value), _pt_scale(a.partials, T(deg2rad(one(T)))))
end
function Base.rad2deg(a::NDual{T,N}) where {T,N}
    NDual{T,N}(rad2deg(a.value), _pt_scale(a.partials, T(rad2deg(one(T)))))
end

# sinc(x) = sin(πx)/(πx) for x≠0, 1 at x=0; derivative = cosc(x).
function Base.sinc(a::NDual{T,N}) where {T,N}
    NDual{T,N}(sinc(a.value), _pt_scale(a.partials, T(cosc(a.value))))
end

# hypot — d/da hypot(a,b) = a / hypot(a,b), d/db = b / hypot(a,b).
function Base.hypot(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    h = hypot(a.value, b.value)
    NDual{T,N}(
        h, _pt_add(_pt_scale(a.partials, a.value / h), _pt_scale(b.partials, b.value / h))
    )
end

# min / max — subgradient: select the tangent of the winning branch.
function Base.max(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    a.value >= b.value ? a : b
end
function Base.min(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    a.value <= b.value ? a : b
end

# clamp — subgradient: zero tangent at the clamped endpoints.
function Base.clamp(a::NDual{T,N}, lo::NDual{T,N}, hi::NDual{T,N}) where {T,N}
    a.value <= lo.value ? lo : (a.value >= hi.value ? hi : a)
end
function Base.clamp(a::NDual{T,N}, lo::Real, hi::Real) where {T,N}
    a.value <= T(lo) ? NDual{T,N}(T(lo)) : (a.value >= T(hi) ? NDual{T,N}(T(hi)) : a)
end

# flipsign / copysign — sign of result determined by primal; tangent follows.
function Base.flipsign(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    signbit(b.value) ? -a : a
end
function Base.copysign(a::NDual{T,N}, b::NDual{T,N}) where {T,N}
    signbit(a.value) == signbit(b.value) ? a : -a
end

# ── Real / imag / conj — for Complex{NDual} to compose generically ────────────────
# A NDual is always the "real part" of itself; conj is the identity for reals.

Base.real(a::NDual) = a
Base.imag(a::NDual{T,N}) where {T,N} = zero(NDual{T,N})
Base.conj(a::NDual) = a
Base.reim(a::NDual{T,N}) where {T,N} = (a, zero(NDual{T,N}))
Base.isreal(::NDual) = true

# ── Comparisons (on value only — for control flow in kernels) ──────────────────────

Base.:<(a::NDual, b::NDual) = a.value < b.value
Base.:>(a::NDual, b::NDual) = a.value > b.value
Base.:<=(a::NDual, b::NDual) = a.value <= b.value
Base.:>=(a::NDual, b::NDual) = a.value >= b.value
Base.:(==)(a::NDual, b::NDual) = a.value == b.value
Base.isless(a::NDual, b::NDual) = isless(a.value, b.value)
Base.isnan(a::NDual) = isnan(a.value)
Base.isinf(a::NDual) = isinf(a.value)
Base.isfinite(a::NDual) = isfinite(a.value)
Base.signbit(a::NDual) = signbit(a.value)

# ── Utility ───────────────────────────────────────────────────────────────────────
Base.eps(d::NDual) = eps(d.value)
Base.eps(::Type{NDual{T,N}}) where {T,N} = eps(T)
function Base.iszero(d::NDual{T,N}) where {T,N}
    iszero(d.value) && all(iszero, d.partials)
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

function Base.abs(z::Complex{NDual{T,N}}) where {T,N}
    hypot(real(z), imag(z))
end
function Base.abs2(z::Complex{NDual{T,N}}) where {T,N}
    real(z)^2 + imag(z)^2
end
function Base.conj(z::Complex{NDual{T,N}}) where {T,N}
    Complex(real(z), -imag(z))
end

# sin(a + bi) = sin(a)cosh(b) + i·cos(a)sinh(b)
function Base.sin(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    sa, ca = sincos(a)
    Complex(sa * cosh(b), ca * sinh(b))
end

# cos(a + bi) = cos(a)cosh(b) - i·sin(a)sinh(b)
function Base.cos(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    sa, ca = sincos(a)
    Complex(ca * cosh(b), -(sa * sinh(b)))
end

# exp(a + bi) = exp(a)·(cos(b) + i·sin(b))
function Base.exp(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    er = exp(a)
    sb, cb = sincos(b)
    Complex(er * cb, er * sb)
end

# log(a + bi) = log(|z|) + i·atan(b, a)
function Base.log(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    Complex(log(hypot(a, b)), atan(b, a))
end

# sqrt(a + bi) = sqrt((|z|+a)/2) + i·sign(b)·sqrt((|z|-a)/2)
function Base.sqrt(z::Complex{NDual{T,N}}) where {T,N}
    a, b = real(z), imag(z)
    r = hypot(a, b)
    re = sqrt((r + a) * NDual{T,N}(T(0.5)))
    im = copysign(one(NDual{T,N}), b) * sqrt((r - a) * NDual{T,N}(T(0.5)))
    Complex(re, im)
end

# tan(z) = sin(z)/cos(z)
function Base.tan(z::Complex{NDual{T,N}}) where {T,N}
    sin(z) / cos(z)
end

# ── Unsupported-operation error ───────────────────────────────────────────────────
# Operations that would silently destroy partial information (integer/rounding ops,
# integer division, modulo) throw a clear error instead of falling through to a
# confusing MethodError or, worse, silently dropping gradients.
#
# If you hit this from a GPU broadcast kernel, the function you are differentiating
# calls one of these non-differentiable operations on a floating-point argument.
# Options:
#   • Replace the operation with a differentiable approximation.
#   • Mark that argument as non-differentiable so NDual wrapping is skipped.
#   • Open an issue if you believe the operation should have a subgradient rule.

struct NDualUnsupportedError <: Exception
    op::Symbol
end
function Base.showerror(io::IO, e::NDualUnsupportedError)
    print(
        io,
        "NDual does not support `$(e.op)`. ",
        "This operation cannot propagate partial derivatives through a GPU broadcast kernel. ",
        "Use a differentiable alternative, or open an issue if a subgradient rule makes sense.",
    )
end

for _op in (:floor, :ceil, :round, :trunc, :div, :fld, :cld, :mod, :rem, :gcd, :lcm)
    @eval Base.$_op(::NDual, args...) = throw(NDualUnsupportedError($(QuoteNode(_op))))
    @eval Base.$_op(::Type, ::NDual, args...) = throw(
        NDualUnsupportedError($(QuoteNode(_op)))
    )
end

# ── Future: tiled GPU kernels with NDual ──────────────────────────────────────────
#
# The current broadcast AD uses one NDual{T,N} per thread: every thread computes
# the primal and all N partials in registers in a single kernel pass.  This is
# already efficient for small N and element-wise functions.  For larger N or
# functions with cross-element data reuse (reductions, softmax, layer norm),
# *tiled* kernels offer further gains:
#
# ── Conceptual note: tiling applied to the Dual itself ──────────────────────
# An NDual{T,N} is a tile in the partial-derivative dimension.  Just as spatial
# tiling partitions an M-element array into ceil(M/K) tiles of width K — each
# processed in one pass with data reuse in shared memory — slot-tiling partitions
# the N-wide Dual into ceil(N/K) tiles of width K, each processed in one kernel
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
