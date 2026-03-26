using ..NDuals
# ── nforward: NDual-backed forward-mode engine ────────────────────────────────────
# `nforward` evaluates code by lifting inputs into `NDual`s and running the primal
# function directly on those lifted values. It does not reuse Mooncake's ordinary forward
# interpreter, even when `chunk_size == 1`.
#
# ── File layout ────────────────────────────────────────────────────────────────────
# This file is organized as:
# - core types
# - public rule/cache entrypoints
# - shared validation and layout helpers
# - reverse accumulation and execution
# - forward evaluation pipeline
# - cache spec checks
# - cached scalar/array fast paths
#
# ── High-level interfaces ──────────────────────────────────────────────────────────
#   nforward_build_frule(f, x...; chunk_size)
#     returns `NForwardRule`
#     consumed via `rule(f::Dual, x::Dual...)`
#     obeys the standard `frule!!` interface
#     also accepts `sig::Type{<:Tuple}` for signature-based construction
#
#   nforward_build_rrule(f, x...; chunk_size)
#     returns `NForwardRRule`
#     consumed via `rule(f::CoDual, x::CoDual...)`
#     obeys the standard `rrule!!` interface
#     also accepts `sig::Type{<:Tuple}` for signature-based construction
#
#   nforward_prepare_cache(f, x...; chunk_size)
#     returns `NForwardCache`
#     consumed via `value_and_derivative!!(cache, f::Dual, x::Dual...)`
#     consumed via `value_and_gradient!!(cache, f, x...)`
#
# ── Constraints ────────────────────────────────────────────────────────────────────
# - `chunk_size` is global across the whole call
# - supported primals are IEEE float scalars, complex IEEE float scalars, dense arrays
#   with those element types, and tuples thereof
# - rule construction requires stateless callables (singleton callable types)
# - `friendly_tangents=true`, `debug_mode=true`, and differentiation with respect to `f`
#   are intentionally unsupported here
#
# ── Primitive Reverse-Mode Example ────────────────────────────────────────────────
# `nforward` can also be used to define reverse-mode rules, and is especially useful
# when dual-number forward differentiation is more compiler-friendly than IR-transform-
# based AD, for example when differentiating through CUDA kernels. It also often has
# significantly lower compilation latency:
#   f(x) = sum(abs2, x)
#   sig = Tuple{typeof(f),Vector{Float64}}
#   Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode sig
#   Mooncake.build_primitive_rrule(::Type{sig}) =
#       Mooncake.nforward_build_rrule(sig; chunk_size=4)
#
#
# Core types
#
# High-level rule/cache objects first. The implementation details they rely on are defined
# in later sections.

"""
    NForwardRule

Callable forward-mode rule used by `nforward`.

`NForwardRule` is built from a statically-known call signature. `buf` holds a reusable
typed scratch buffer for in-place array lifting when a chunk-layout tangent is available.
"""
struct NForwardRule{sig,N,Tbuf<:Base.RefValue}
    buf::Tbuf
end

# Backward-compatible zero-arg constructor used by primitive rules in rule_via_nforward_patches.jl.
function NForwardRule{sig,N}() where {sig,N}
    NForwardRule{sig,N,Base.RefValue{Any}}(Ref{Any}(nothing))
end

"""
    NForwardRRule

Callable reverse-mode rule used by `nforward`.

`NForwardRRule` is built from a statically-known call signature. Both direct
[`nforward_build_rrule`](@ref) calls and primitive reverse-mode registration route through
that signature-based construction path. `buf` holds reusable typed scratch buffers for
cached scalar-output fast paths when that is available. `grad_buf` holds a separate
pre-allocated gradient buffer for the single-array-input scalar-output fast paths, allowing
the rrule to stay allocation-free at steady state without copying the computed gradient.

The `scalar_out` type parameter is `true` when inference confirms at rule-build time that
`f` returns an `IEEEFloat` scalar for the given input types. This allows the single-array
rrule specialisation to skip the redundant primal type-check call, which otherwise costs
one full function evaluation per gradient call.
"""
struct NForwardRRule{sig,N,Tbuf<:Base.RefValue,scalar_out,Tgbuf<:Base.RefValue}
    buf::Tbuf
    grad_buf::Tgbuf
end

"""
    NForwardCache

Cache returned by [`nforward_prepare_cache`](@ref). It owns the forward/reverse rules,
runtime input signatures, and reusable zero tangents used by cached `nforward` calls.
"""
struct NForwardCache{RF,RR,S<:Tuple,TT<:Tuple}
    frule::RF
    rrule::RR
    sigs::S
    tangents::TT
end

"""
    NForwardPullback

Concrete pullback object for `nforward` reverse rules. It stores the primal callable,
primals, input tangents, and output fdata needed to rerun chunked NDual passes during the
reverse sweep.

!!! note
    The scalar specialization `NForwardPullback{F,N,Tuple{T},Tuple{NoFData},Y}` with
    `T<:Number` must remain an `isbits` type for that path to stay allocation-free. The
    generic path (array or multi-input primals) is not isbits and allocates as usual.
    Do not add heap-allocated fields without auditing both paths.
"""
struct NForwardPullback{F,N,P,T,Y}
    f::F
    primals::P
    tangents::T
    y_fdata::Y
end

"""
    NForwardArrayScalarPullback

Lightweight pullback returned by the optimised single-array-input / scalar-output rrule fast
path.  The full gradient (∂f/∂x_i for all i) is computed eagerly during the rrule call and
stored in `grad` (a separate copy, not aliased with `tangent(x_codual)`).  `fdata` is a
reference to `tangent(x_codual)`.  The pullback accumulates `ȳ * grad` into `fdata`,
satisfying Mooncake's standard increment semantics for mutable array tangents.
"""
struct NForwardArrayScalarPullback{G<:AbstractArray}
    grad::G   # precomputed ∂f/∂x; does NOT alias fdata
    fdata::G  # tangent(x_cd); the accumulation target
end

function (pb::NForwardArrayScalarPullback)(y_rdata)
    if isone(y_rdata)
        pb.fdata .+= pb.grad
    else
        pb.fdata .+= y_rdata .* pb.grad
    end
    return (NoRData(), NoRData())
end

#
# Public construction and cached execution
#
# These are the main `nforward` entry points. A reviewer should be able to read this section
# first, then dive into the lower-level pipelines only as needed.

"""
    nforward_build_frule(f, x...; chunk_size=nothing)
    nforward_build_frule(sig::Type{<:Tuple}; chunk_size=nothing)

Build a forward-mode rule through `nforward`.

This path is independent from Mooncake's ordinary forwards-mode interpreter and obeys the
standard `frule!!` interface. It evaluates the primal function directly on NDual-lifted
scalar / dense-array inputs. Rule construction is signature-based, so `nforward` only
supports stateless callables here.

If `chunk_size` is omitted, nforward automatically selects `min(DOF, hardware_preferred_width)`
from the signature, where `hardware_preferred_width` is 8 (one AVX-512 / two AVX2 Float64
registers). For scalar-only signatures the DOF is known exactly at type level; for
signatures containing arrays the preferred width is used directly.

!!! warning "Not thread-safe"
    The returned `NForwardRule` holds a mutable workspace buffer that is updated in-place
    on every call. Do not share a single rule across threads; build one rule per thread.

!!! note "debug_mode"
    The `debug_mode` keyword is accepted for API compatibility with `nforward_prepare_cache`
    but always throws when `true`; nforward-specific debug checks are not yet implemented.
    Mooncake's outer debug wrapper still validates CoDual inputs/outputs when the rule is
    invoked inside a debug-mode rrule.
"""
function nforward_build_frule(
    sig::Type{<:Tuple}; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    resolved = isnothing(chunk_size) ? _nforward_sig_default_chunk_size(sig) : chunk_size
    resolved = _nforward_validate(sig, resolved; debug_mode)
    buf = _nforward_frule_buf_ref(sig, Val(resolved))
    return NForwardRule{sig,resolved,typeof(buf)}(buf)
end

function nforward_build_frule(
    f, x...; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    return nforward_build_frule(
        typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages
    )
end

function (rule::NForwardRule{sig,N})(f::Dual, x::Vararg{Dual,M}) where {sig,N,M}
    _nforward_verify_sig(rule, (f, x...))
    _nforward_check_function_tangent(tangent(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    y, dy = _nforward_eval(primal(f), primals, tangents, Val(N))
    return Dual(y, dy)
end

# Optimised single-array-input frule: reuses a pre-allocated lifted buffer when the tangent
# is in chunk layout (ndims(dx) == ndims(x) + 1, i.e. N > 1).  Falls through to the
# generic allocating path when the tangent is in plain layout (N == 1).
function (rule::NForwardRule{sig,N})(
    f::Dual, x::Dual{Array{T,Nd},Array{T,Nd1}}
) where {sig,N,T<:IEEEFloat,Nd,Nd1}
    _nforward_verify_sig(rule, (f, x))
    _nforward_check_function_tangent(tangent(f))
    px = _nforward_check_primal(primal(x))
    dx = tangent(x)
    if Nd1 == Nd + 1  # chunk layout — use in-place lift with pre-allocated buffer
        lifted = _nforward_frule_lifted!(rule.buf, px, dx, Val(N))
        y, dy = _nforward_extract(primal(f)(lifted), Val(N))
    else  # plain layout (N == 1) — fall back to the allocating path
        y, dy = _nforward_eval(primal(f), (px,), (dx,), Val(N))
    end
    return Dual(y, dy)
end

"""
    nforward_build_rrule(f, x...; chunk_size=nothing)
    nforward_build_rrule(sig::Type{<:Tuple}; chunk_size=nothing)

Build a reverse-mode rule through `nforward`.

The reverse rule is derived from chunked NDual forward passes and obeys the standard
`rrule!!` interface. Rule construction is signature-based, so `nforward` only supports
stateless callables here.

If `chunk_size` is omitted, nforward automatically selects `min(DOF, hardware_preferred_width)`
from the signature, where `hardware_preferred_width` is 8 (one AVX-512 / two AVX2 Float64
registers). For scalar-only signatures the DOF is known exactly at type level; for
signatures containing arrays the preferred width is used directly.

!!! warning "Not thread-safe"
    The returned `NForwardRRule` holds mutable workspace buffers (`buf`, `grad_buf`) that
    are updated in-place on every call. Do not share a single rule across threads; build
    one rule per thread, or use [`nforward_prepare_cache`](@ref) and create one cache per
    thread.

!!! note "debug_mode"
    The `debug_mode` keyword is accepted for API compatibility with `nforward_prepare_cache`
    but always throws when `true`; nforward-specific debug checks are not yet implemented.
    Mooncake's outer debug wrapper still validates CoDual inputs/outputs when the rule is
    invoked inside a debug-mode rrule.
"""
function nforward_build_rrule(
    sig::Type{<:Tuple}; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    resolved = isnothing(chunk_size) ? _nforward_sig_default_chunk_size(sig) : chunk_size
    resolved = _nforward_validate(sig, resolved; debug_mode)
    buf = _nforward_buf_ref(sig, Val(resolved))
    grad_buf = _nforward_grad_buf_ref(sig)
    scalar_out = _nforward_infer_scalar_output(sig)
    return NForwardRRule{sig,resolved,typeof(buf),scalar_out,typeof(grad_buf)}(
        buf, grad_buf
    )
end

function nforward_build_rrule(
    f, x...; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
)
    return nforward_build_rrule(
        typeof((f, x...)); chunk_size, debug_mode, silence_debug_messages
    )
end

"""
    (rule::NForwardRRule)(f::CoDual, x::Vararg{CoDual})

Evaluate an `nforward` reverse rule and return both the output `CoDual` and pullback.
`f` must be a stateless callable: `tangent(f)` must be `NoFData`, otherwise an
`ArgumentError` is thrown. Differentiating with respect to `f` is not supported.
"""
function (rule::NForwardRRule{sig,N})(f::CoDual, x::Vararg{CoDual,M}) where {sig,N,M}
    _nforward_verify_sig(rule, (f, x...))
    tangent(f) isa NoFData || throw(
        ArgumentError("nforward does not support differentiating with respect to `f`.")
    )
    return _nforward_rrule_call(primal(f), x, Val(N))
end

# Optimised single-real-array-input rrule, scalar-output fast path.
#
# When `scalar_out=true` (inferred at rule-build time), the output is known to be an
# IEEEFloat scalar so we skip the redundant primal type-check call.  For small DOF / single-
# chunk problems that extra call would cost one full function evaluation — e.g. for
# `large_single_block` (DOF=2, 400 scalar ops) it was ~23% of total rrule time.
#
# The pullback only needs to scale the pre-computed gradient by the output cotangent —
# zero per-call allocations at steady state.
function (rule::NForwardRRule{sig,N,Tbuf,true})(
    f::CoDual, x::CoDual{A}
) where {sig,N,Tbuf,T<:IEEEFloat,Nd,A<:Array{T,Nd}}
    _nforward_verify_sig(rule, (f, x))
    tangent(f) isa NoFData || throw(
        ArgumentError("nforward does not support differentiating with respect to `f`.")
    )
    f_runtime = primal(f)
    _nforward_check_primal(primal(x))
    # Output type is known scalar — skip primal call and go straight to gradient sweep.
    # Gradient is written to the pre-allocated grad_buf (not into tangent(x)), so the
    # pullback can accumulate into the existing fdata without a copy.
    grad_arr = _nforward_lazy_grad_buf!(rule.grad_buf, primal(x))
    y, _ = _nforward_array_scalar_value_and_gradient(
        f_runtime, x, rule.buf, grad_arr, Val(N)
    )
    y_cd = CoDual(y, fdata(zero_tangent(y)))
    return y_cd, NForwardArrayScalarPullback(grad_arr, tangent(x))
end

# Fallback: output type not known to be scalar at build time.  Run a primal call to
# dispatch between the scalar fast path and the generic chunked path.
function (rule::NForwardRRule{sig,N,Tbuf,false})(
    f::CoDual, x::CoDual{A}
) where {sig,N,Tbuf,T<:IEEEFloat,Nd,A<:Array{T,Nd}}
    _nforward_verify_sig(rule, (f, x))
    tangent(f) isa NoFData || throw(
        ArgumentError("nforward does not support differentiating with respect to `f`.")
    )
    f_runtime = primal(f)
    x_primal = _nforward_check_primal(primal(x))
    y_primal = f_runtime(x_primal)
    if y_primal isa IEEEFloat
        grad_arr = _nforward_lazy_grad_buf!(rule.grad_buf, x_primal)
        y, _ = _nforward_array_scalar_value_and_gradient(
            f_runtime, x, rule.buf, grad_arr, Val(N)
        )
        y_cd = CoDual(y, fdata(zero_tangent(y)))
        return y_cd, NForwardArrayScalarPullback(grad_arr, tangent(x))
    else
        _nforward_is_supported_primal(typeof(y_primal)) || _nforward_output_error(y_primal)
        y_cd = CoDual(y_primal, fdata(zero_tangent(y_primal)))
        return y_cd,
        _nforward_pullback(f_runtime, (x_primal,), (tangent(x),), tangent(y_cd), Val(N))
    end
end

"""
    nforward_prepare_cache(f, x...; chunk_size=nothing, config=Mooncake.Config())

Prepare a cache for [`value_and_derivative!!`](@ref) and [`value_and_gradient!!`](@ref)
when used with an [`NForwardCache`](@ref).

If `chunk_size` is omitted, nforward uses `min(total_dof, 8)` with a floor of 1
(same default as [`nforward_build_rrule`](@ref)).

!!! warning "Not thread-safe"
    Each `NForwardCache` owns mutable workspace buffers (including pre-allocated tangent
    arrays) that are mutated in-place during every call to `value_and_gradient!!`. Do not
    share a single cache across threads; create one cache per thread instead.
"""
@unstable @inline function nforward_prepare_cache(
    f, x::Vararg{Any,N}; chunk_size=nothing, config=Config()
) where {N}
    _nforward_check_config(config)
    map(_nforward_check_primal, x)
    chunk_size = _nforward_resolve_chunk_size(chunk_size, x)
    sigs = map(_nforward_sig, (f, x...))
    frule = nforward_build_frule(
        f,
        x...;
        chunk_size,
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    rrule = nforward_build_rrule(
        f,
        x...;
        chunk_size,
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    tangents = map(zero_tangent, (f, x...))
    return NForwardCache(frule, rrule, sigs, tangents)
end

"""
    value_and_derivative!!(cache::NForwardCache, f::Dual, x::Dual...)

Run the cached `nforward` forward rule after checking that the runtime inputs match the cache
signature.
"""
function value_and_derivative!!(cache::NForwardCache, f::Dual, x::Vararg{Dual,N}) where {N}
    _nforward_check_sigs(cache.sigs, (primal(f), map(primal, x)...))
    return cache.frule(f, x...)
end

"""
    value_and_gradient!!(cache::NForwardCache, f, x...; args_to_zero=...)

Run the cached `nforward` reverse rule after validating cache specs and zeroing the cached
input tangents.
"""
function value_and_gradient!!(
    cache::NForwardCache,
    f::F,
    x::Vararg{Any,N};
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    _nforward_check_args_to_zero(args_to_zero, N + 1)
    _nforward_check_sigs(cache.sigs, (f, x...))
    tangents = tuple_map(set_to_zero_maybe!!, cache.tangents, args_to_zero)
    return __value_and_gradient!!(cache.rrule, tuple_map(CoDual, (f, x...), tangents)...)
end

# Optimization note:
# This scalar specialization bypasses the general pullback-based reverse path for cached
# `value_and_gradient!!` calls. Evaluating one NDual-lifted primal directly is enough to recover
# the scalar primal and derivative, which removes the remaining steady-state allocations for
# singleton scalar inputs.
#
# Complex scalars (CoDual{<:Complex}) have no matching specialization here and fall through
# to the generic `__value_and_gradient!!` in src/interface.jl, which runs the full pullback.
# This is correct and tested; it is simply not on the allocation-free fast path.
"""
    __value_and_gradient!!(rule::NForwardRRule, f::CoDual, x::CoDual)

Dispatch the scalar cached fast path for `nforward` reverse mode.
"""
function __value_and_gradient!!(
    rule::NForwardRRule{sig,N}, f::CoDual, x::CoDual{T}
) where {sig,N,T<:IEEEFloat}
    _nforward_verify_sig(rule, (f, x))
    return _nforward_scalar_value_and_gradient(primal(f), f, x, Val(N))
end

"""
    __value_and_gradient!!(rule::NForwardRRule, f::CoDual, x::CoDual{<:Array})

Scalar-output dense-array fast path for `nforward` reverse rules. Dispatches to a
typed-workspace path for `Vector{T}` inputs (via the buf type parameter) and a generic
workspace path for higher-dimensional arrays.
"""
function __value_and_gradient!!(
    rule::NForwardRRule{sig,chunk_size}, f::CoDual, x::CoDual{A}
) where {sig,chunk_size,T<:IEEEFloat,N,A<:Array{T,N}}
    _nforward_verify_sig(rule, (f, x))
    _nforward_check_function_tangent(tangent(f))
    y, x_grad = _nforward_array_scalar_value_and_gradient(
        primal(f), x, rule.buf, Val(chunk_size)
    )
    return y, (_nforward_function_gradient(f), x_grad)
end

const NFORWARD_DEBUG_MODE_WARNING =
    "nforward-backed reverse-mode rules ignore `debug_mode=true`; " *
    "Mooncake's outer debug wrapper still checks CoDual inputs/outputs, but the " *
    "inner nforward rule executes without nforward-specific debug checks."

"""
    _copy(rule::NForwardRule)

Copy an `NForwardRule` while resetting cached workspace state.
"""
function _copy(x::NForwardRule{sig,N,Tbuf}) where {sig,N,Tbuf}
    return NForwardRule{sig,N,Tbuf}(Tbuf(nothing))
end

"""
    _copy(rule::NForwardRRule)

Copy an `NForwardRRule` while resetting cached workspace state.
"""
function _copy(
    x::NForwardRRule{sig,N,Tbuf,scalar_out,Tgbuf}
) where {sig,N,Tbuf,scalar_out,Tgbuf}
    return NForwardRRule{sig,N,Tbuf,scalar_out,Tgbuf}(Tbuf(nothing), Tgbuf(nothing))
end

# NForwardRRule bakes sig into its type parameters and validates internally via
# _nforward_verify_sig on every call; no redundant check is needed here.
__verify_sig(::NForwardRRule, ::Tuple) = nothing

"""
    verify_fwds_inputs(rule::NForwardRRule, x)

Emit the outer debug-mode warning before delegating to generic forward-input checks.
"""
@noinline function verify_fwds_inputs(rule::NForwardRRule, @nospecialize(x::Tuple))
    @warn NFORWARD_DEBUG_MODE_WARNING
    return invoke(verify_fwds_inputs, Tuple{Any,Tuple}, rule, x)
end

#
# Validation and layout helpers
#
# Shared validation, sizing, and shape utilities used across the forward, reverse, and cached
# execution paths.

# Infer at rule-build time whether `sig` has a scalar IEEEFloat output.
# Used to set the `scalar_out` type parameter on `NForwardRRule`, allowing the hot-path
# rrule to skip the redundant primal type-check call for known-scalar functions.
#
# Uses `Base.return_types`, which is a best-effort hint: it may return `[Any]` for
# type-unstable functions or under some world-age conditions.  In those cases this
# function safely returns `false`, and the rrule falls through to the runtime primal
# check (`scalar_out=false` path).  There is no correctness risk from a missed inference
# — only a missed optimisation.
function _nforward_infer_scalar_output(sig::Type{<:Tuple})
    F = sig.parameters[1]
    Base.issingletontype(F) || return false
    argtypes = Tuple{(sig.parameters[i] for i in 2:length(sig.parameters))...}
    rt = Base.return_types(F.instance, argtypes)
    return !isempty(rt) && rt[1] <: IEEEFloat
end

@inline function _nforward_check_chunk_size(chunk_size::Integer)
    chunk_size > 0 && return Int(chunk_size)
    throw(ArgumentError("`chunk_size` must be a positive integer, got $chunk_size."))
end

# Shared preamble for frule/rrule builders: validate chunk_size, callable sig, and debug_mode.
@inline function _nforward_validate(sig, chunk_size::Integer; debug_mode=false)
    chunk_size = _nforward_check_chunk_size(chunk_size)
    _nforward_check_callable_sig(sig)
    debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    return chunk_size
end

# Conservative SIMD-friendly default: 8 lanes covers one AVX-512 register (8×Float64)
# and two AVX2 registers.  Chunk sizes beyond 8 add register pressure without
# proportional throughput gains on most hardware.
const _NFORWARD_PREFERRED_CHUNK_SIZE = 8

@inline function _nforward_default_chunk_size(x::Tuple)
    return max(1, min(sum(_nforward_input_dof, x), _NFORWARD_PREFERRED_CHUNK_SIZE))
end

# Type-level DOF: returns the number of differentiable scalar components for a
# concrete type, or `nothing` when the size cannot be determined from the type
# alone (e.g. heap-allocated Array whose length is a runtime value).
@inline _nforward_type_dof(::Type{<:IEEEFloat}) = 1
@inline _nforward_type_dof(::Type{<:Complex{<:IEEEFloat}}) = 2
@inline _nforward_type_dof(T::Type{<:Tuple}) = sum(_nforward_type_dof, T.parameters; init=0)
# Arrays: size is a runtime value, so DOF is unknown at type level.
@inline _nforward_type_dof(::Type{<:AbstractArray}) = nothing
@inline _nforward_type_dof(::Type) = 0  # non-differentiable (e.g. function type)

# Sum type-level DOF across all argument types in a signature, skipping the
# function type at position 1.  Returns `nothing` if any input has unknown DOF
# (i.e. contains an array whose size is not encoded in its type).
@inline function _nforward_sig_dof(::Type{sig}) where {sig<:Tuple}
    params = sig.parameters
    total = 0
    for i in 2:length(params)
        d = _nforward_type_dof(params[i])
        d === nothing && return nothing
        total += d
    end
    return total
end

# Default chunk size from a signature alone: min(DOF, preferred) when DOF is
# statically known; preferred width when any input has an array type.
@inline function _nforward_sig_default_chunk_size(::Type{sig}) where {sig<:Tuple}
    dof = _nforward_sig_dof(sig)
    preferred = _NFORWARD_PREFERRED_CHUNK_SIZE
    return dof === nothing ? preferred : max(1, min(dof, preferred))
end

"""
    _nforward_resolve_chunk_size(chunk_size, x)

Return the chunk size used for an `nforward` call, defaulting to
`min(total_dof(x), _NFORWARD_PREFERRED_CHUNK_SIZE)` when `chunk_size === nothing`.
"""
@inline function _nforward_resolve_chunk_size(chunk_size, x::Tuple)
    return if isnothing(chunk_size)
        _nforward_default_chunk_size(x)
    else
        _nforward_check_chunk_size(chunk_size)
    end
end

@inline _nforward_is_supported_scalar(::Type{<:IEEEFloat}) = true
@inline _nforward_is_supported_scalar(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nforward_is_supported_scalar(::Type) = false

@inline _nforward_is_supported_primal(::Type{<:IEEEFloat}) = true
@inline _nforward_is_supported_primal(::Type{<:Complex{<:IEEEFloat}}) = true
@inline function _nforward_is_supported_primal(::Type{<:Array{ET}}) where {ET}
    _nforward_is_supported_scalar(ET)
end
@inline function _nforward_is_supported_primal(T::Type{<:Tuple})
    return all(_nforward_is_supported_primal, T.parameters)
end
@inline _nforward_is_supported_primal(::Type) = false

@inline _nforward_input_error(x) = throw(
    ArgumentError(
        "nforward currently supports only IEEEFloat / Complex IEEEFloat scalars and " *
        "dense Array inputs with those element types. Got $(typeof(x)).",
    ),
)

@inline function _nforward_output_error(y)
    throw(
        ArgumentError(
            "nforward supports IEEEFloat / Complex IEEEFloat scalars, dense Arrays with " *
            "those element types, and tuples thereof as outputs. Got $(typeof(y)).",
        ),
    )
end

@inline function _nforward_check_primal(x)
    _nforward_is_supported_primal(typeof(x)) || _nforward_input_error(x)
    return x
end

@inline function _nforward_check_function_tangent(df)
    df isa NoTangent && return nothing
    throw(ArgumentError("nforward does not support differentiating with respect to `f`."))
end

@inline function _nforward_check_callable_sig(sig::Type{<:Tuple})
    F = sig.parameters[1]
    Base.issingletontype(F) || throw(
        ArgumentError(
            "nforward only supports stateless callables for rule construction. Got $F. " *
            "Stateless callables are required because nforward re-evaluates the function " *
            "multiple times with different tangent seeds; a mutable callable would " *
            "produce incorrect gradients on the second and subsequent evaluations.",
        ),
    )
    f = F.instance
    argsig = Tuple{(sig.parameters[i] for i in 2:length(sig.parameters))...}
    hasmethod(f, argsig) && return sig
    throw(
        ArgumentError("nforward rule construction expected a callable signature, got $sig.")
    )
end

@inline _nforward_rule_sig(::NForwardRule{sig}) where {sig} = sig
@inline _nforward_rule_sig(::NForwardRRule{sig}) where {sig} = sig

@inline function _nforward_verify_sig(rule::Union{NForwardRule,NForwardRRule}, fx::Tuple)
    sig = _nforward_rule_sig(rule)
    Tfx = Tuple{map(_typeof ∘ primal, fx)...}
    # Use <: (subtype) rather than == so that a rule built for an abstract signature
    # (e.g. Tuple{typeof(f), AbstractVector{Float64}}) also accepts concrete subtypes
    # at call time. This mirrors the convention used elsewhere in Mooncake's dispatch.
    Tfx <: sig && return nothing
    throw(ArgumentError("Arguments with sig $Tfx do not subtype rule signature, $sig"))
end

@inline function _nforward_check_config(config)
    config.friendly_tangents && throw(
        ArgumentError("nforward does not currently support `friendly_tangents=true`.")
    )
    config.debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    return nothing
end

@inline function _nforward_check_args_to_zero(args_to_zero::NTuple, expected::Int)
    length(args_to_zero) == expected ||
        throw(ArgumentError("`args_to_zero` must have length $expected for this call."))
    all(args_to_zero) && return nothing
    throw(
        ArgumentError(
            "nforward does not currently support selective `args_to_zero` (partial " *
            "gradient computation). To differentiate only w.r.t. a subset of inputs, " *
            "call the NForwardRRule directly and manually zero the tangents for the " *
            "inputs you do not want to differentiate through.",
        ),
    )
end

@inline _nforward_input_dof(x::IEEEFloat) = 1
@inline _nforward_input_dof(x::Complex{<:IEEEFloat}) = 2
@inline _nforward_input_dof(x::AbstractArray{<:IEEEFloat}) = length(x)
@inline _nforward_input_dof(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 2 * length(x)
# Tuple inputs: sum DOFs of each element (tuples are supported as top-level inputs).
@inline _nforward_input_dof(x::Tuple) = sum(_nforward_input_dof, x; init=0)

#
# Reverse accumulation utilities
#
# These helpers seed input directions, contract output tangents with cotangents, and scatter
# each chunk's contributions into gradient storage.

function _nforward_seed_tangent(x::IEEEFloat, chunk_size::Int, start_slot::Int, offset::Int)
    # offset+1 is this scalar's global slot; lane is its 1-indexed position in the chunk.
    global_slot = offset + 1
    lane = global_slot - start_slot + 1
    if chunk_size == 1
        return lane == 1 ? one(x) : zero(x)
    end
    return ntuple(k -> typeof(x)(k == lane), Val(chunk_size))
end

function _nforward_seed_tangent(
    x::Complex{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    if chunk_size == 1
        if offset + 1 == start_slot
            return complex(one(T), zero(T))
        elseif offset + 2 == start_slot
            return complex(zero(T), one(T))
        else
            return zero(x)
        end
    end
    return ntuple(k -> begin
        slot = start_slot + k - 1
        if offset + 1 == slot
            complex(one(T), zero(T))
        elseif offset + 2 == slot
            complex(zero(T), one(T))
        else
            zero(x)
        end
    end, Val(chunk_size))
end

function _nforward_seed_tangent(
    x::AbstractArray{T}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    if chunk_size == 1
        dx = zero_tangent(x)
        global_slot = start_slot
        if offset < global_slot <= offset + length(x)
            dx[global_slot - offset] = one(T)
        end
        return dx
    end
    dx = zeros(T, size(x)..., chunk_size)
    cart_inds = CartesianIndices(x)
    for lane in 1:chunk_size
        global_slot = start_slot + lane - 1
        if offset < global_slot <= offset + length(x)
            idx = Tuple(cart_inds[global_slot - offset])
            dx[idx..., lane] = one(T)
        end
    end
    return dx
end

function _nforward_seed_tangent(
    x::AbstractArray{Complex{T}}, chunk_size::Int, start_slot::Int, offset::Int
) where {T<:IEEEFloat}
    # Each complex element contributes 2 DOFs in consecutive global slots:
    #   odd  local_slot → seed the real part  (complex(1, 0))
    #   even local_slot → seed the imaginary part (complex(0, 1))
    # So element index = cld(local_slot, 2) and part = isodd(local_slot).
    if chunk_size == 1
        dx = zero_tangent(x)
        global_slot = start_slot
        if offset < global_slot <= offset + 2 * length(x)
            local_slot = global_slot - offset
            elem = cld(local_slot, 2)
            dx[elem] = if isodd(local_slot)
                complex(one(T), zero(T))
            else
                complex(zero(T), one(T))
            end
        end
        return dx
    end
    dx = zeros(Complex{T}, size(x)..., chunk_size)
    cart_inds = CartesianIndices(x)
    for lane in 1:chunk_size
        global_slot = start_slot + lane - 1
        if offset < global_slot <= offset + 2 * length(x)
            local_slot = global_slot - offset
            elem = cld(local_slot, 2)
            idx = Tuple(cart_inds[elem])
            dx[idx..., lane] =
                isodd(local_slot) ? complex(one(T), zero(T)) : complex(zero(T), one(T))
        end
    end
    return dx
end

@inline function _nforward_add_slot!(
    g::Base.RefValue{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    local_slot == 1 && (g[] += v)
    return nothing
end

@inline function _nforward_add_slot!(
    g::Base.RefValue{Complex{T}}, local_slot::Int, v
) where {T<:IEEEFloat}
    if local_slot == 1
        g[] += complex(v, zero(T))
    elseif local_slot == 2
        g[] += complex(zero(T), v)
    end
    return nothing
end

@inline function _nforward_add_slot!(
    g::AbstractArray{T}, local_slot::Int, v
) where {T<:IEEEFloat}
    g[local_slot] += v
    return nothing
end

@inline function _nforward_add_slot!(
    g::AbstractArray{Complex{T}}, local_slot::Int, v
) where {T<:IEEEFloat}
    elem = cld(local_slot, 2)
    g[elem] += isodd(local_slot) ? complex(v, zero(T)) : complex(zero(T), v)
    return nothing
end

function _nforward_scatter_chunk!(grads::Tuple, inputs::Tuple, dy::Tuple, start_slot::Int)
    global_slot = start_slot
    for lane_val in dy
        offset = 0
        for (i, x) in enumerate(inputs)
            dof = _nforward_input_dof(x)
            if offset < global_slot <= offset + dof
                local_slot = global_slot - offset
                _nforward_add_slot!(grads[i], local_slot, lane_val)
                break
            end
            offset += dof
        end
        global_slot += 1
    end
    return nothing
end

function _nforward_gradient_refs(primals::Tuple, tangents::Tuple)
    return map(primals, tangents) do x, dx
        if x isa Number
            g = zero_tangent(x, dx)
            Ref(g)
        else
            # Use a fresh zeros array (not the fdata) for VJP accumulation.  The generic
            # pullback adds this into the fdata at the end so that existing fdata content
            # (e.g. contributions from other uses of the same array) is preserved.
            zero_tangent(x)
        end
    end
end

_nforward_unwrap_gradient(g::Base.RefValue) = g[]
_nforward_unwrap_gradient(g) = g

# `slot` is the 1-based DOF index within the scalar/complex input: 1 for the real
# component (or the sole IEEEFloat slot), 2 for the imaginary component of a complex.
# Called from `_nforward_scalar_gradient_rdata` with the loop's global_slot, which
# equals the local slot because that path is specialised to a single input at offset 0.
@inline function _nforward_accumulate_scalar_gradient(
    g::T, slot::Int, v
) where {T<:IEEEFloat}
    slot == 1 ? g + v : g
end

@inline function _nforward_accumulate_scalar_gradient(
    g::Complex{T}, slot::Int, v
) where {T<:IEEEFloat}
    if slot == 1
        return g + complex(v, zero(T))
    elseif slot == 2
        return g + complex(zero(T), v)
    end
    return g
end

@inline function _nforward_real_dot(a::T, b::T) where {T<:IEEEFloat}
    return a * b
end

@inline function _nforward_real_dot(a::Complex{T}, b::Complex{T}) where {T<:IEEEFloat}
    return real(conj(a) * b)
end

# Scalar (real or complex): chunk_size=1 → plain scalar zero; chunk_size=N → N-tuple of zeros.
@inline _nforward_zero_output_tangent(y::Union{IEEEFloat,Complex{<:IEEEFloat}}, ::Val{1}) = zero(
    y
)
@inline _nforward_zero_output_tangent(y::Union{IEEEFloat,Complex{<:IEEEFloat}}, ::Val{N}) where {N} = ntuple(
    _ -> zero(y), Val(N)
)

# Array (real or complex elements): chunk_size=1 → same-shape zero array; chunk_size=N → N extra lanes.
@inline _nforward_zero_output_tangent(y::AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}, ::Val{1}) = zero_tangent(
    y
)
@inline function _nforward_zero_output_tangent(
    y::AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}, ::Val{N}
) where {N}
    return zeros(eltype(y), size(y)..., N)
end

# Tuple outputs: recurse element-wise.
@inline function _nforward_zero_output_tangent(y::Tuple, ::Val{N}) where {N}
    return map(yi -> _nforward_zero_output_tangent(yi, Val(N)), y)
end

# chunk_size=1: tangent is a plain scalar — return it regardless of which lane is requested.
@inline _nforward_scalar_lane(dy, ::Val{1}, _) = dy
# chunk_size=N: tangent is an NTuple — index with a static Val{K} or a runtime Int.
@inline _nforward_scalar_lane(dy::NTuple{N}, ::Val{N}, ::Val{K}) where {N,K} = dy[K]
@inline _nforward_scalar_lane(dy::NTuple{N}, ::Val{N}, lane::Int) where {N} = dy[lane]

function _nforward_contract_output(ȳ::T, dy::T) where {T<:IEEEFloat}
    return (_nforward_real_dot(ȳ, dy),)
end

function _nforward_contract_output(ȳ::Complex{T}, dy::Complex{T}) where {T<:IEEEFloat}
    return (_nforward_real_dot(ȳ, dy),)
end

function _nforward_contract_output(ȳ::T, dy::NTuple{N,T}) where {T<:IEEEFloat,N}
    return ntuple(k -> _nforward_real_dot(ȳ, dy[k]), Val(N))
end

function _nforward_contract_output(
    ȳ::Complex{T}, dy::NTuple{N,Complex{T}}
) where {T<:IEEEFloat,N}
    return ntuple(k -> _nforward_real_dot(ȳ, dy[k]), Val(N))
end

# Single-chunk array case (ȳ and dy have the same shape — real or complex elements).
function _nforward_contract_output(
    ȳ::A, dy::A
) where {A<:AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}}
    acc = zero(real(eltype(ȳ)))
    @inbounds for I in CartesianIndices(ȳ)
        acc += _nforward_real_dot(ȳ[I], dy[I])
    end
    return (acc,)
end

# Multi-chunk array case (dy has one extra trailing dimension of size N — real or complex).
# Both arrays must share the same element type T.  Mixed-precision cases (e.g.
# ȳ::Vector{Float32} with dy::Matrix{Float64}) fall through to the generic error overload
# below.  In practice nforward keeps element types consistent across primal/tangent, so
# this situation only arises from incorrect external use.
function _nforward_contract_output(
    ȳ::A, dy::B
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},A<:AbstractArray{T},B<:AbstractArray{T}}
    ndims(dy) == ndims(ȳ) + 1 || _nforward_output_error(dy)
    size(dy)[1:(end - 1)] == size(ȳ) || _nforward_output_error(dy)
    N = size(dy, ndims(dy))
    return ntuple(Val(N)) do k
        acc = zero(real(T))
        @inbounds for I in CartesianIndices(ȳ)
            idx = Tuple(I)
            acc += _nforward_real_dot(ȳ[I], dy[idx..., k])
        end
        acc
    end
end

# Tuple outputs: contract each element independently and sum lane contributions.
function _nforward_contract_output(ȳ::Tuple, dy::Tuple)
    length(ȳ) == length(dy) || _nforward_output_error(dy)
    contributions = map(_nforward_contract_output, ȳ, dy)
    return foldl((a, b) -> map(+, a, b), contributions)
end

function _nforward_contract_output(ȳ, dy)
    _nforward_output_error(dy)
end

#
# Reverse execution
#
# `NForwardPullback` is a concrete callable struct rather than a closure so direct
# `nforward_build_rrule(...)(...)` calls can stay allocation-free on the scalar path.
# The pullback still carries the cached primals / tangents / output fdata needed to rerun
# chunked NDual passes during the reverse sweep.

"""
    _nforward_rrule_call(f, x, chunk_size_or_val)

Run the shared reverse-mode `nforward` path: evaluate the primal on the runtime primals,
wrap the result in the `CoDual` shape expected by `rrule!!`, and build the pullback that
reruns chunked NDual passes during the reverse sweep.
"""
@inline function _nforward_rrule_call(f, x::Tuple{Vararg{CoDual,M}}, ::Val{N}) where {M,N}
    primals = map(primal, x)
    tangents = map(tangent, x)
    y_primal = f(primals...)
    _nforward_is_supported_primal(typeof(y_primal)) || _nforward_output_error(y_primal)
    y = CoDual(y_primal, fdata(zero_tangent(y_primal)))
    return y, _nforward_pullback(f, primals, tangents, tangent(y), Val(N))
end

@inline function _nforward_rrule_call(f, x::Tuple, chunk_size::Integer)
    return _nforward_rrule_call(f, x, Val(_nforward_check_chunk_size(chunk_size)))
end

"""
    _nforward_pullback(rule, primals, tangents, y_fdata)

Package the state needed for a later reverse sweep into an `NForwardPullback`.
"""
function _nforward_pullback(f, primals::Tuple, tangents::Tuple, y_fdata, ::Val{N}) where {N}
    return NForwardPullback{typeof(f),N,typeof(primals),typeof(tangents),typeof(y_fdata)}(
        f, primals, tangents, y_fdata
    )
end

"""
    _nforward_scalar_gradient_rdata(pb, y_rdata)

Compute scalar-input reverse data for the specialized scalar pullback path.
"""
function _nforward_scalar_gradient_rdata(
    pb::NForwardPullback{F,N,Tuple{T},Tuple{NoFData},Y}, y_rdata
) where {F,N,T<:Number,Y}
    ȳ = tangent(pb.y_fdata, y_rdata)
    x = pb.primals[1]
    g = zero_tangent(x, pb.tangents[1])
    total_dof = _nforward_input_dof(x)
    for start_slot in 1:N:total_dof
        tangents = (_nforward_seed_tangent(x, N, start_slot, 0),)
        _, dy = _nforward_eval(pb.f, pb.primals, tangents, Val(N))
        lane_vals = _nforward_contract_output(ȳ, dy)
        global_slot = start_slot
        for lane_val in lane_vals
            g = _nforward_accumulate_scalar_gradient(g, global_slot, lane_val)
            global_slot += 1
        end
    end
    return rdata(g)
end

"""
    (pb::NForwardPullback)(y_rdata)

Scalar-input pullback specialization returning reverse data without the generic scatter path.
"""
function (pb::NForwardPullback{F,N,Tuple{T},Tuple{NoFData},Y})(
    y_rdata
) where {F,N,T<:Number,Y}
    return (rdata(zero_tangent(pb.f)), _nforward_scalar_gradient_rdata(pb, y_rdata))
end

"""
    (pb::NForwardPullback)(y_rdata)

Generic `nforward` pullback that reruns chunked NDual passes and scatters VJP contributions
into the cached gradient containers.
"""
function (pb::NForwardPullback{F,N})(y_rdata) where {F,N}
    ȳ = tangent(pb.y_fdata, y_rdata)
    grads = _nforward_gradient_refs(pb.primals, pb.tangents)
    total_dof = sum(_nforward_input_dof, pb.primals)
    for start_slot in 1:N:total_dof
        offset = 0
        tangents = map(pb.primals) do x
            t = _nforward_seed_tangent(x, N, start_slot, offset)
            offset += _nforward_input_dof(x)
            t
        end
        _, dy = _nforward_eval(pb.f, pb.primals, tangents, Val(N))
        lane_vals = _nforward_contract_output(ȳ, dy)
        _nforward_scatter_chunk!(grads, pb.primals, lane_vals, start_slot)
    end
    # For array inputs the gradient lives in grads[i] (a fresh zeros array).  Accumulate it
    # into the fdata (pb.tangents[i]) so that existing fdata contributions are preserved.
    foreach(pb.tangents, grads) do fdata, grad
        fdata isa AbstractArray && (fdata .+= _nforward_unwrap_gradient(grad))
    end
    return tuple(
        rdata(zero_tangent(pb.f)), map(g -> rdata(_nforward_unwrap_gradient(g)), grads)...
    )
end

#
# Forward evaluation pipeline
#
# `_nforward_eval` is the high-level lifted evaluation step used by both the forward rule and
# the reverse pullback. The lift/extract helpers below are the data-conversion pieces it uses.

"""
    _nforward_eval(f, primals, tangents, ::Val{N})

Evaluate `f` on NDual-lifted primals and extract both primal output and chunked tangent data.
"""
function _nforward_eval(f, primals::Tuple, tangents::Tuple, ::Val{N}) where {N}
    lifted = map(
        (x, dx) -> _nforward_lift(_nforward_check_primal(x), dx, Val(N)), primals, tangents
    )
    return _nforward_extract(f(lifted...), Val(N))
end

"""
    _nforward_eval(f, primals::Tuple{<:Number}, tangents, ::Val{N})

Scalar-input specialization of `_nforward_eval` that avoids tuple-based lifting overhead.
"""
function _nforward_eval(
    f, primals::Tuple{T}, tangents::Tuple{D}, ::Val{N}
) where {T<:Number,D,N}
    lifted = _nforward_lift(_nforward_check_primal(primals[1]), tangents[1], Val(N))
    return _nforward_extract(f(lifted), Val(N))
end

#
# Forward lift/extract helpers
#
# These utilities translate between Mooncake tangent layouts and NDual-based lifted values.

@inline function _nforward_scalar_partials(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    if N == 1 && dx isa Real
        return (T(dx),)
    elseif dx isa Tuple && length(dx) == N
        return ntuple(i -> T(dx[i]), Val(N))
    elseif dx isa AbstractVector && length(dx) == N
        return ntuple(i -> T(dx[i]), Val(N))
    end
    throw(
        ArgumentError(
            "Expected scalar tangent for $(T) to be a Real when chunk_size == 1, or " *
            "a length-$N tuple/vector of reals. Got $(typeof(dx)): $dx.",
        ),
    )
end

@inline function _nforward_complex_partials(
    x::Complex{T}, dx, ::Val{N}
) where {T<:IEEEFloat,N}
    if N == 1 && dx isa Complex
        return (T(real(dx)),), (T(imag(dx)),)
    elseif dx isa Tuple && length(dx) == N
        return ntuple(i -> T(real(dx[i])), Val(N)), ntuple(i -> T(imag(dx[i])), Val(N))
    elseif dx isa AbstractVector && length(dx) == N
        return ntuple(i -> T(real(dx[i])), Val(N)), ntuple(i -> T(imag(dx[i])), Val(N))
    end
    throw(
        ArgumentError(
            "Expected complex scalar tangent for $(typeof(x)) to be a Complex when " *
            "chunk_size == 1, or a length-$N tuple/vector of complex values. " *
            "Got $(typeof(dx)): $dx.",
        ),
    )
end

@inline function _nforward_array_tangent_dims(x::AbstractArray, ::Val{N}) where {N}
    return (size(x)..., N)
end

@inline function _nforward_check_array_tangent(
    x::AbstractArray, dx::AbstractArray, ::Val{N}
) where {N}
    if N == 1 && size(dx) == size(x)
        return :plain
    elseif size(dx) == _nforward_array_tangent_dims(x, Val(N))
        return :chunked
    end
    throw(
        ArgumentError(
            "Expected array tangent for input of size $(size(x)) to have size $(size(x)) " *
            "when chunk_size == 1, or size $(_nforward_array_tangent_dims(x, Val(N))) " *
            "otherwise. Got size $(size(dx)).",
        ),
    )
end

function _nforward_lift(x::T, dx, ::Val{N}) where {T<:IEEEFloat,N}
    return NDual{T,N}(x, _nforward_scalar_partials(x, dx, Val(N)))
end

function _nforward_lift(x::Complex{T}, dx, ::Val{N}) where {T<:IEEEFloat,N}
    re, im = _nforward_complex_partials(x, dx, Val(N))
    return Complex(NDual{T,N}(real(x), re), NDual{T,N}(imag(x), im))
end

function _nforward_lift(x::A, dx::AbstractArray, ::Val{N}) where {ET,A<:AbstractArray{ET},N}
    _nforward_is_supported_scalar(ET) || _nforward_input_error(x)
    tangent_layout = _nforward_check_array_tangent(x, dx, Val(N))
    out = similar(x, ET <: IEEEFloat ? NDual{ET,N} : Complex{NDual{ET.parameters[1],N}})
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        if tangent_layout === :plain
            out[I] = _nforward_lift(x[I], dx[I], Val(N))
        else
            if ET <: IEEEFloat
                out[I] = NDual{ET,N}(x[I], ntuple(k -> ET(dx[idx..., k]), Val(N)))
            else
                T = ET.parameters[1]
                out[I] = Complex(
                    NDual{T,N}(real(x[I]), ntuple(k -> T(real(dx[idx..., k])), Val(N))),
                    NDual{T,N}(imag(x[I]), ntuple(k -> T(imag(dx[idx..., k])), Val(N))),
                )
            end
        end
    end
    return out
end

@inline function _nforward_extract_scalar(d::NDual{T,N}, ::Val{N}) where {T,N}
    return if N == 1
        NDuals.ndual_value(d), NDuals.ndual_partial(d, 1)
    else
        NDuals.ndual_value(d), ntuple(k -> NDuals.ndual_partial(d, k), Val(N))
    end
end

@inline function _nforward_extract_scalar(z::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    primal = complex(NDuals.ndual_value(real(z)), NDuals.ndual_value(imag(z)))
    tangent = if N == 1
        complex(NDuals.ndual_partial(real(z), 1), NDuals.ndual_partial(imag(z), 1))
    else
        ntuple(
            k -> complex(
                NDuals.ndual_partial(real(z), k), NDuals.ndual_partial(imag(z), k)
            ),
            Val(N),
        )
    end
    return primal, tangent
end

function _nforward_extract(y::NDual{T,N}, ::Val{N}) where {T,N}
    return _nforward_extract_scalar(y, Val(N))
end

function _nforward_extract(y::Complex{NDual{T,N}}, ::Val{N}) where {T,N}
    return _nforward_extract_scalar(y, Val(N))
end

function _nforward_extract(y::AbstractArray{<:NDual{T,N}}, ::Val{N}) where {T,N}
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = NDuals.ndual_value(y[I])
        idx = Tuple(I)
        if N == 1
            tangent[I] = NDuals.ndual_partial(y[I], 1)
        else
            for k in 1:N
                tangent[idx..., k] = NDuals.ndual_partial(y[I], k)
            end
        end
    end
    return primal, tangent
end

function _nforward_extract(
    y::AbstractArray{<:Complex{NDual{Treal,N}}}, ::Val{N}
) where {Treal,N}
    T = Complex{Treal}
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = complex(NDuals.ndual_value(real(y[I])), NDuals.ndual_value(imag(y[I])))
        idx = Tuple(I)
        if N == 1
            tangent[I] = complex(
                NDuals.ndual_partial(real(y[I]), 1), NDuals.ndual_partial(imag(y[I]), 1)
            )
        else
            for k in 1:N
                tangent[idx..., k] = complex(
                    NDuals.ndual_partial(real(y[I]), k), NDuals.ndual_partial(imag(y[I]), k)
                )
            end
        end
    end
    return primal, tangent
end

# Tuple outputs: recurse into each element; primal and tangent are both tuples.
function _nforward_extract(y::Tuple, ::Val{N}) where {N}
    pairs = map(yi -> _nforward_extract(yi, Val(N)), y)
    return map(first, pairs), map(last, pairs)
end

# Non-NDual outputs: the primal carries no tangent information; synthesize a zero tangent.
# Unsupported types fall through to _nforward_output_error via the is_supported_primal guard.
function _nforward_extract(y, ::Val{N}) where {N}
    _nforward_is_supported_primal(typeof(y)) || _nforward_output_error(y)
    return y, _nforward_zero_output_tangent(y, Val(N))
end

#
# Cache specs
#
# Cached entrypoints use these shape/type descriptors before dispatching into the shared
# forward or reverse pipelines.

@inline function _nforward_sig(x)
    if x isa Function
        return (kind=:function, type=typeof(x), size=())
    elseif x isa AbstractArray
        return (kind=:array, type=typeof(x), size=size(x))
    else
        return (kind=:scalar, type=typeof(x), size=())
    end
end

function _nforward_check_sigs(sigs::Tuple, fx::Tuple)
    length(sigs) == length(fx) || throw(ArgumentError("nforward cache arity mismatch."))
    for (sig, x) in zip(sigs, fx)
        typeof(x) == sig.type || throw(
            ArgumentError(
                "nforward cache type mismatch: expected $(sig.type), got $(typeof(x))."
            ),
        )
        x isa AbstractArray || continue
        size(x) == sig.size || throw(
            ArgumentError(
                "nforward cache size mismatch: expected $(sig.size), got $(size(x))."
            ),
        )
    end
    return nothing
end

@inline _nforward_function_gradient(f::CoDual) = tangent(fdata(tangent(f)), NoRData())

@inline function _nforward_scalar_value_and_gradient(
    f_runtime, f::CoDual, x::CoDual{T}, ::Val{N}
) where {T<:IEEEFloat,N}
    _nforward_check_function_tangent(tangent(f))
    x_primal = _nforward_check_primal(primal(x))
    seed = _nforward_seed_tangent(x_primal, N, 1, 0)
    y, dy = _nforward_extract(f_runtime(_nforward_lift(x_primal, seed, Val(N))), Val(N))
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    x_grad = tangent(fdata(tangent(x)), _nforward_scalar_lane(dy, Val(N), Val(1)))
    return y, (_nforward_function_gradient(f), x_grad)
end

@inline function _nforward_scalar_value_and_gradient(
    f_runtime, f::CoDual, x::CoDual{T}, chunk_size::Integer
) where {T<:IEEEFloat}
    return _nforward_scalar_value_and_gradient(
        f_runtime, f, x, Val(_nforward_check_chunk_size(chunk_size))
    )
end

#
# Cached array scalar fast path
#
# A single helper covers both Vector{T} and higher-dimensional Array{T,N} inputs.
# The lifted array is fetched (and lazily allocated) via _nforward_rrule_lifted!, which
# dispatches on the buf ref type: typed-ref bufs (Vector{T} inputs) stay fully inferred;
# untyped Ref{Any} bufs (N-D arrays) fall back to a runtime type check.
#
# The inner loop uses an incremental seeding strategy:
#   1. The lifted array is initialised once with zero partials — O(n).
#   2. Per chunk: only the C active elements are set to unit-partial form — O(C).
#   3. After f is evaluated, those C elements are reset to zero — O(C).
#
# This replaces the previous O(n×C) approach (fill! the full seed + lift! every chunk)
# with O(n) + O(C)×chunks, matching how ForwardDiff manages its GradientConfig.

# Primary overload: gradient is written to an explicitly provided buffer `grad`.
# `grad` must have the same shape and element type as `primal(x)`.
# The buffer is zeroed at the start of each call.  Does NOT touch `tangent(x)`.
function _nforward_array_scalar_value_and_gradient(
    f_runtime, x::CoDual{A}, buf::Base.RefValue, grad::A, ::Val{C}
) where {C,T<:IEEEFloat,N,A<:Array{T,N}}
    x_primal = _nforward_check_primal(primal(x))
    fill!(grad, zero(T))
    lifted = _nforward_rrule_lifted!(buf, x_primal, Val(C))

    # Initialise all elements to (x[i], 0̄) once; chunks then update only C slots each.
    _nforward_init_lifted!(lifted, x_primal, Val(C))
    cart = CartesianIndices(x_primal)

    y = zero(T)
    for start_slot in 1:C:length(x_primal)
        _nforward_seed_lifted_chunk!(lifted, x_primal, cart, start_slot, Val(C))
        y, lane_vals = _nforward_scalar_lanes(f_runtime(lifted), T, Val(C))
        _nforward_unseed_lifted_chunk!(lifted, x_primal, cart, start_slot, Val(C))
        global_slot = start_slot
        @inbounds for lane_val in lane_vals
            global_slot > length(x_primal) && break
            grad[global_slot] += lane_val
            global_slot += 1
        end
    end

    return y, grad
end

# Backward-compatible overload: accumulates into tangent(x) directly (the buffer is
# zeroed inside the primary overload before use).
# Used by __value_and_gradient!! where the caller expects the returned gradient to be
# the fdata array (i.e. tangent(x) == x_grad after the call).
function _nforward_array_scalar_value_and_gradient(
    f_runtime, x::CoDual{A}, buf::Base.RefValue, ::Val{C}
) where {C,T<:IEEEFloat,N,A<:Array{T,N}}
    return _nforward_array_scalar_value_and_gradient(f_runtime, x, buf, tangent(x), Val(C))
end

#
# Workspace helpers (_nforward_buf_ref / _nforward_rrule_lifted! /
#                    _nforward_grad_buf_ref / _nforward_lazy_grad_buf!)
#
# The rrule buf stores only the lifted Array{NDual{T,C}} — no seed array needed.
# The grad_buf stores a pre-allocated gradient array matching the input shape.
# Two ref types are used for each, matching the frule buf pattern:
#   - Typed-ref (Vector{T} input): fully inferred, no runtime isa check.
#   - Generic Ref{Any} (N-D array input): workspace type recovered at runtime.

_nforward_buf_ref(sig, ::Val) = Ref{Any}(nothing)

function _nforward_buf_ref(::Type{Tuple{F,Vector{T}}}, ::Val{C}) where {F,T<:IEEEFloat,C}
    return Ref{Union{Nothing,Vector{NDual{T,C}}}}(nothing)
end

_nforward_grad_buf_ref(sig) = Ref{Any}(nothing)

function _nforward_grad_buf_ref(::Type{Tuple{F,Vector{T}}}) where {F,T<:IEEEFloat}
    return Ref{Union{Nothing,Vector{T}}}(nothing)
end

# Typed-ref path: fully inferred.
function _nforward_rrule_lifted!(
    buf::Base.RefValue{Union{Nothing,Vector{NDual{T,C}}}}, x::Vector{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    ws = buf[]
    if ws === nothing || length(ws::Vector{NDual{T,C}}) != length(x)
        buf[] = Vector{NDual{T,C}}(undef, length(x))
    end
    return buf[]::Vector{NDual{T,C}}
end

# Generic path: buf is Ref{Any}; workspace type recovered at runtime.
function _nforward_rrule_lifted!(
    buf::Base.RefValue, x::Array{T,N}, ::Val{C}
) where {T<:IEEEFloat,N,C}
    Tlift = NDual{T,C}
    ws = buf[]
    if !(ws isa Array{Tlift,N} && size(ws) == size(x))
        buf[] = Array{Tlift,N}(undef, size(x))
    end
    return buf[]::Array{Tlift,N}
end

# Typed-ref path: fully inferred.
function _nforward_lazy_grad_buf!(
    grad_buf::Base.RefValue{Union{Nothing,Vector{T}}}, x_primal::Vector{T}
) where {T<:IEEEFloat}
    ws = grad_buf[]
    if ws === nothing || length(ws::Vector{T}) != length(x_primal)
        grad_buf[] = Vector{T}(undef, length(x_primal))
    end
    return grad_buf[]::Vector{T}
end

# Generic path: grad_buf is Ref{Any}; gradient array type recovered at runtime.
function _nforward_lazy_grad_buf!(
    grad_buf::Base.RefValue, x_primal::Array{T,N}
) where {T<:IEEEFloat,N}
    ws = grad_buf[]
    if !(ws isa Array{T,N} && size(ws) == size(x_primal))
        grad_buf[] = Array{T,N}(undef, size(x_primal))
    end
    return grad_buf[]::Array{T,N}
end

# Initialise every element of `lifted` to NDual(x[i], 0̄) — O(n), called once per
# value_and_gradient!! invocation.  Chunks then update only C elements each.
function _nforward_init_lifted!(
    lifted::Array{NDual{T,C},N}, x::Array{T,N}, ::Val{C}
) where {T<:IEEEFloat,C,N}
    z = ntuple(_ -> zero(T), Val(C))
    @inbounds for I in CartesianIndices(x)
        lifted[I] = NDual{T,C}(x[I], z)
    end
    return lifted
end

# Set C elements starting at `start_slot` to unit-partial form — O(C).
function _nforward_seed_lifted_chunk!(
    lifted::Array{NDual{T,C},N},
    x::Array{T,N},
    cart::CartesianIndices,
    start_slot::Int,
    ::Val{C},
) where {T<:IEEEFloat,C,N}
    @inbounds for lane in 1:C
        gs = start_slot + lane - 1
        gs > length(x) && break
        I = cart[gs]
        lifted[I] = NDual{T,C}(x[I], ntuple(k -> T(k == lane), Val(C)))
    end
    return lifted
end

# Reset those same C elements back to zero-partial form — O(C).
function _nforward_unseed_lifted_chunk!(
    lifted::Array{NDual{T,C},N},
    x::Array{T,N},
    cart::CartesianIndices,
    start_slot::Int,
    ::Val{C},
) where {T<:IEEEFloat,C,N}
    z = ntuple(_ -> zero(T), Val(C))
    @inbounds for lane in 1:C
        gs = start_slot + lane - 1
        gs > length(x) && break
        I = cart[gs]
        lifted[I] = NDual{T,C}(x[I], z)
    end
    return lifted
end

function _nforward_lift!(
    out::Array{NDual{T,C}}, x::Array{T}, dx::Array{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        out[I] = NDual{T,C}(x[I], ntuple(k -> dx[idx..., k], Val(C)))
    end
    return out
end

#
# Frule lifted-array buffer helpers (_nforward_frule_buf_ref / _nforward_frule_lifted!)
#
# The frule receives the seed tangent directly from the caller (as the tangent part of the
# input Dual), so no seed buffer is needed — only a pre-allocated Array{NDual{T,C}} of the
# same shape as the primal input.  Two buf types are used:
#   - Typed-ref (Vector{T} input): fully inferred, no runtime isa check.
#   - Generic Ref{Any} (higher-dimensional inputs): workspace type recovered at runtime.

_nforward_frule_buf_ref(sig, ::Val) = Ref{Any}(nothing)

function _nforward_frule_buf_ref(
    ::Type{Tuple{F,Vector{T}}}, ::Val{C}
) where {F,T<:IEEEFloat,C}
    return Ref{Union{Nothing,Vector{NDual{T,C}}}}(nothing)
end

# Typed-ref path for Vector{T} inputs.
function _nforward_frule_lifted!(
    buf::Base.RefValue{Union{Nothing,Vector{NDual{T,C}}}},
    x::Vector{T},
    dx::Array{T},
    ::Val{C},
) where {T<:IEEEFloat,C}
    ws = buf[]
    if ws === nothing || length(ws::Vector{NDual{T,C}}) != length(x)
        buf[] = Vector{NDual{T,C}}(undef, length(x))
    end
    return _nforward_lift!(buf[]::Vector{NDual{T,C}}, x, dx, Val(C))
end

# Generic path for Array{T,Nd} inputs (Ref{Any} buf).
function _nforward_frule_lifted!(
    buf::Base.RefValue, x::Array{T,Nd}, dx::Array{T}, ::Val{C}
) where {T<:IEEEFloat,Nd,C}
    Tlift = NDual{T,C}
    ws = buf[]
    if !(ws isa Array{Tlift,Nd} && size(ws) == size(x))
        buf[] = Array{Tlift,Nd}(undef, size(x))
    end
    return _nforward_lift!(buf[]::Array{Tlift,Nd}, x, dx, Val(C))
end

"""
    _nforward_scalar_lanes(y_raw, ::Type{T}, ::Val{C})

Decode a scalar-output `nforward` chunk evaluation into its primal value plus the `C`
directional derivatives carried by that chunk. Constant outputs are treated as zero-tangent
outputs, so this helper works for both NDual-carrying and plain scalar results.
"""
@inline function _nforward_scalar_lanes(
    y_raw::NDual{T,C}, ::Type{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    return NDuals.ndual_value(y_raw), ntuple(k -> NDuals.ndual_partial(y_raw, k), Val(C))
end

@inline function _nforward_scalar_lanes(y_raw, ::Type{T}, ::Val{C}) where {T<:IEEEFloat,C}
    y, dy = _nforward_extract(y_raw, Val(C))
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    return y, ntuple(k -> _nforward_scalar_lane(dy, Val(C), Val(k)), Val(C))
end
