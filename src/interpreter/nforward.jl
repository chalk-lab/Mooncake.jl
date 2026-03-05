#
# NForward design note
#
# - `nforward` is a separate NDual-backed forward engine. It does not reuse Mooncake's
#   ordinary forward interpreter internally, even when `chunk_size == 1`.
# - `nforward_build_frule` produces a callable following the `frule!!` calling convention:
#   `rule(f::Dual, x::Dual...)`.
# - `nforward_build_rrule` produces a callable following the `rrule!!` calling convention:
#   `rule(f::CoDual, x::CoDual...)`.
# - `nforward_prepare_cache` wraps those rules in `NForwardCache`, plus pre-allocated zero
#   tangents for the function and inputs which are reused across
#   `value_and_gradient!!(cache, ...)` calls.
# - `NForwardCache` plugs into the existing Mooncake interface entrypoints:
#     `value_and_derivative!!(cache::NForwardCache, f::Dual, x::Dual...)`
#     `value_and_gradient!!(cache::NForwardCache, f, x...)` for scalar outputs only
# - The reverse rule is implemented by repeatedly applying chunked NDual forward passes and
#   contracting each chunk of output tangents against the output cotangent `ȳ` to
#   accumulate VJP contributions.
# - `chunk_size` is global across the whole call. For multi-argument functions, nforward
#   flattens all differentiable input degrees of freedom across every input argument into
#   one slot sequence, and each NDual lane corresponds to one slot in that global ordering.
# - If `chunk_size` is omitted in `nforward_prepare_cache`, nforward picks
#   `min(total_dof, 8)` with a floor of 1.
# - Current supported primals are IEEE float scalars, complex IEEE float scalars, and
#   dense `Array`s with those element types. Other `AbstractArray` subtypes are rejected
#   for now. `friendly_tangents=true` and `debug_mode=true` are intentionally unsupported
#   for now. Differentiation with respect to `f` is also unsupported: `nforward` captures
#   `f` at build time and only NDual-lifts the data inputs.
#
include(joinpath(@__DIR__, "..", "..", "ext", "MooncakeCUDAExt", "ndual.jl"))

export nforward_prepare_cache

struct NForwardRule{F,N}
    f::F
end

struct NForwardRRule{F,N}
    f::F
end

__verify_sig(::NForwardRRule, ::Tuple) = nothing

struct NForwardCache{RF,RR,S<:Tuple,TT<:Tuple}
    frule::RF
    rrule::RR
    specs::S
    tangents::TT
end

@inline function _nforward_check_chunk_size(chunk_size::Integer)
    chunk_size > 0 && return Int(chunk_size)
    throw(ArgumentError("`chunk_size` must be a positive integer, got $chunk_size."))
end

@inline function _nforward_default_chunk_size(x::Tuple)
    return max(1, min(sum(_nforward_input_dof, x), 8))
end

@inline _nforward_is_supported_scalar(::Type{<:IEEEFloat}) = true
@inline _nforward_is_supported_scalar(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _nforward_is_supported_scalar(::Type) = false

@inline _nforward_is_supported_primal(::Type{<:IEEEFloat}) = true
@inline _nforward_is_supported_primal(::Type{<:Complex{<:IEEEFloat}}) = true
@inline function _nforward_is_supported_primal(::Type{<:Array{ET}}) where {ET}
    _nforward_is_supported_scalar(ET)
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
            "nforward currently supports only scalar or AbstractArray outputs carrying " *
            "NDual information. Got $(typeof(y)).",
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

@inline function _nforward_check_function_instance(f_cached, f_runtime)
    f_runtime === f_cached && return nothing
    throw(ArgumentError("nforward cache/rule was built for a different function instance."))
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
            "nforward does not currently support custom `args_to_zero`; pass the default " *
            "all-true tuple instead.",
        ),
    )
end

@inline _nforward_input_dof(x::IEEEFloat) = 1
@inline _nforward_input_dof(x::Complex{<:IEEEFloat}) = 2
@inline _nforward_input_dof(x::AbstractArray{<:IEEEFloat}) = length(x)
@inline _nforward_input_dof(x::AbstractArray{<:Complex{<:IEEEFloat}}) = 2 * length(x)

function _nforward_seed_tangent(x::IEEEFloat, chunk_size::Int, start_slot::Int, offset::Int)
    lane = (offset + 1) - start_slot + 1
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
    if chunk_size == 1
        dx = zero_tangent(x)
        global_slot = start_slot
        if offset < global_slot <= offset + 2 * length(x)
            elem = cld(global_slot - offset, 2)
            dx[elem] = if isodd(global_slot - offset)
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

function _nforward_scatter_chunk!(grads::Tuple, inputs::Tuple, dy, start_slot::Int)
    lanes = dy isa Tuple ? dy : (dy,)
    global_slot = start_slot
    for lane_val in lanes
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
        g = zero_tangent(x, dx)
        x isa Number ? Ref(g) : set_to_zero!!(g)
    end
end

_nforward_unwrap_gradient(g::Base.RefValue) = g[]
_nforward_unwrap_gradient(g) = g

@inline function _nforward_real_dot(a::T, b::T) where {T<:IEEEFloat}
    return a * b
end

@inline function _nforward_real_dot(a::Complex{T}, b::Complex{T}) where {T<:IEEEFloat}
    return real(conj(a) * b)
end

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

function _nforward_contract_output(ȳ::A, dy::A) where {T<:IEEEFloat,A<:AbstractArray{T}}
    acc = zero(T)
    @inbounds for I in CartesianIndices(ȳ)
        acc += _nforward_real_dot(ȳ[I], dy[I])
    end
    return (acc,)
end

function _nforward_contract_output(
    ȳ::A, dy::A
) where {T<:Complex{<:IEEEFloat},A<:AbstractArray{T}}
    acc = zero(real(eltype(ȳ)))
    @inbounds for I in CartesianIndices(ȳ)
        acc += _nforward_real_dot(ȳ[I], dy[I])
    end
    return (acc,)
end

function _nforward_contract_output(
    ȳ::A, dy::B
) where {T<:IEEEFloat,A<:AbstractArray{T},B<:AbstractArray{T}}
    ndims(dy) == ndims(ȳ) + 1 || _nforward_output_error(dy)
    size(dy)[1:(end - 1)] == size(ȳ) || _nforward_output_error(dy)
    N = size(dy, ndims(dy))
    return ntuple(Val(N)) do k
        acc = zero(T)
        @inbounds for I in CartesianIndices(ȳ)
            idx = Tuple(I)
            acc += _nforward_real_dot(ȳ[I], dy[idx..., k])
        end
        acc
    end
end

function _nforward_contract_output(
    ȳ::A, dy::B
) where {T<:Complex{<:IEEEFloat},A<:AbstractArray{T},B<:AbstractArray{T}}
    ndims(dy) == ndims(ȳ) + 1 || _nforward_output_error(dy)
    size(dy)[1:(end - 1)] == size(ȳ) || _nforward_output_error(dy)
    N = size(dy, ndims(dy))
    Treal = real(eltype(ȳ))
    return ntuple(Val(N)) do k
        acc = zero(Treal)
        @inbounds for I in CartesianIndices(ȳ)
            idx = Tuple(I)
            acc += _nforward_real_dot(ȳ[I], dy[idx..., k])
        end
        acc
    end
end

function _nforward_contract_output(ȳ, dy)
    _nforward_output_error(dy)
end

function _nforward_pullback(
    rule::NForwardRRule{F,N}, primals::Tuple, tangents::Tuple, y_fdata
) where {F,N}
    function pb!!(y_rdata)
        ȳ = tangent(y_fdata, y_rdata)
        grads = _nforward_gradient_refs(primals, tangents)
        total_dof = sum(_nforward_input_dof, primals)
        for start_slot in 1:N:total_dof
            offset = 0
            tangents = map(primals) do x
                t = _nforward_seed_tangent(x, N, start_slot, offset)
                offset += _nforward_input_dof(x)
                t
            end
            _, dy = _nforward_eval(rule.f, primals, tangents, Val(N))
            lane_vals = _nforward_contract_output(ȳ, dy)
            _nforward_scatter_chunk!(grads, primals, lane_vals, start_slot)
        end
        return tuple(
            rdata(zero_tangent(rule.f)),
            map(g -> rdata(_nforward_unwrap_gradient(g)), grads)...,
        )
    end
    return pb!!
end

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
            "a length-$N tuple/vector of reals. Got $(typeof(dx)).",
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
            "chunk_size == 1, or a length-$N tuple/vector of complex values. Got $(typeof(dx)).",
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

@inline function _nforward_extract_scalar(d::NDual{T,N}) where {T,N}
    return if N == 1
        ndual_value(d), ndual_partial(d, 1)
    else
        ndual_value(d), ntuple(k -> ndual_partial(d, k), Val(N))
    end
end

@inline function _nforward_extract_scalar(z::Complex{NDual{T,N}}) where {T,N}
    primal = complex(ndual_value(real(z)), ndual_value(imag(z)))
    tangent = if N == 1
        complex(ndual_partial(real(z), 1), ndual_partial(imag(z), 1))
    else
        ntuple(k -> complex(ndual_partial(real(z), k), ndual_partial(imag(z), k)), Val(N))
    end
    return primal, tangent
end

function _nforward_extract(y::NDual)
    return _nforward_extract_scalar(y)
end

function _nforward_extract(y::Complex{<:NDual})
    return _nforward_extract_scalar(y)
end

function _nforward_extract(y::AbstractArray{<:NDual})
    T = eltype(y).parameters[1]
    N = eltype(y).parameters[2]
    primal = similar(y, T)
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = ndual_value(y[I])
        idx = Tuple(I)
        if N == 1
            tangent[I] = ndual_partial(y[I], 1)
        else
            for k in 1:N
                tangent[idx..., k] = ndual_partial(y[I], k)
            end
        end
    end
    return primal, tangent
end

function _nforward_extract(y::AbstractArray{<:Complex{<:NDual}})
    dual_eltype = eltype(real(first(y)))
    Treal = typeof(ndual_value(real(first(y))))
    T = Complex{Treal}
    primal = similar(y, T)
    N = dual_eltype.parameters[2]
    tangent = N == 1 ? similar(y, T) : similar(y, T, size(y)..., N)
    @inbounds for I in CartesianIndices(y)
        primal[I] = complex(ndual_value(real(y[I])), ndual_value(imag(y[I])))
        idx = Tuple(I)
        if N == 1
            tangent[I] = complex(ndual_partial(real(y[I]), 1), ndual_partial(imag(y[I]), 1))
        else
            for k in 1:N
                tangent[idx..., k] = complex(
                    ndual_partial(real(y[I]), k), ndual_partial(imag(y[I]), k)
                )
            end
        end
    end
    return primal, tangent
end

function _nforward_extract(y)
    _nforward_output_error(y)
end

@inline function _nforward_spec(x)
    if x isa Function
        return (kind=:function, type=typeof(x), size=())
    elseif x isa AbstractArray
        return (kind=:array, type=typeof(x), size=size(x))
    else
        return (kind=:scalar, type=typeof(x), size=())
    end
end

function _nforward_check_specs(specs::Tuple, fx::Tuple)
    length(specs) == length(fx) || throw(ArgumentError("nforward cache arity mismatch."))
    for (spec, x) in zip(specs, fx)
        typeof(x) == spec.type || throw(
            ArgumentError(
                "nforward cache type mismatch: expected $(spec.type), got $(typeof(x))."
            ),
        )
        x isa AbstractArray || continue
        size(x) == spec.size || throw(
            ArgumentError(
                "nforward cache size mismatch: expected $(spec.size), got $(size(x))."
            ),
        )
    end
    return nothing
end

function _nforward_eval(f, primals::Tuple, tangents::Tuple, ::Val{N}) where {N}
    lifted = map(
        (x, dx) -> _nforward_lift(_nforward_check_primal(x), dx, Val(N)), primals, tangents
    )
    return _nforward_extract(f(lifted...))
end

function _nforward_eval_rule(rule::NForwardRule{F,N}, inputs) where {F,N}
    primals = map(primal, inputs)
    tangents = map(tangent, inputs)
    y, dy = _nforward_eval(rule.f, primals, tangents, Val(N))
    return Dual(y, dy)
end

function _nforward_rrule_output(y_primal)
    _nforward_is_supported_primal(typeof(y_primal)) || _nforward_output_error(y_primal)
    return CoDual(y_primal, fdata(zero_tangent(y_primal)))
end

"""
    nforward_build_frule(args...; chunk_size)

Build a forwards-mode rule through the experimental nforward API family.

This path is independent from Mooncake's ordinary forwards-mode interpreter. It evaluates the
primal function directly on NDual-lifted scalar / array inputs.
"""
function nforward_build_frule(
    f, x...; chunk_size::Integer, debug_mode=false, silence_debug_messages=true
)
    chunk_size = _nforward_check_chunk_size(chunk_size)
    debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    silence_debug_messages
    return NForwardRule{typeof(f),chunk_size}(f)
end

function (rule::NForwardRule{F,N})(f::Dual, x::Vararg{Dual,M}) where {F,N,M}
    _nforward_check_function_tangent(tangent(f))
    _nforward_check_function_instance(rule.f, primal(f))
    return _nforward_eval_rule(rule, x)
end

"""
    nforward_build_rrule(args...; chunk_size)

Build a reverse-mode rule through the experimental nforward API family.

The reverse rule is derived from chunked NDual forward passes and obeys the standard
`rrule!!` interface.
"""
function nforward_build_rrule(
    f, x...; chunk_size::Integer, debug_mode=false, silence_debug_messages=true
)
    chunk_size = _nforward_check_chunk_size(chunk_size)
    debug_mode &&
        throw(ArgumentError("nforward does not currently support `debug_mode=true`."))
    silence_debug_messages
    return NForwardRRule{typeof(f),chunk_size}(f)
end

function (rule::NForwardRRule{F,N})(f::CoDual, x::Vararg{CoDual,M}) where {F,N,M}
    tangent(f) isa NoFData || throw(
        ArgumentError("nforward does not support differentiating with respect to `f`.")
    )
    _nforward_check_function_instance(rule.f, primal(f))
    primals = map(primal, x)
    tangents = map(tangent, x)
    y_primal = rule.f(primals...)
    y = _nforward_rrule_output(y_primal)
    return y, _nforward_pullback(rule, primals, tangents, tangent(y))
end

"""
    nforward_prepare_cache(f, x...; chunk_size=nothing, config=Mooncake.Config())

Prepare a cache for [`value_and_derivative!!`](@ref) and [`value_and_gradient!!`](@ref)
when used with an [`NForwardCache`](@ref).

If `chunk_size` is omitted, nforward uses `min(total_dof, 8)` with a floor of 1.
"""
@unstable @inline function nforward_prepare_cache(
    f, x::Vararg{Any,N}; chunk_size=nothing, config=Config()
) where {N}
    _nforward_check_config(config)
    map(_nforward_check_primal, x)
    chunk_size = if isnothing(chunk_size)
        _nforward_default_chunk_size(x)
    else
        _nforward_check_chunk_size(chunk_size)
    end
    specs = map(_nforward_spec, (f, x...))
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
    return NForwardCache(frule, rrule, specs, tangents)
end

function value_and_derivative!!(cache::NForwardCache, f::Dual, x::Vararg{Dual,N}) where {N}
    _nforward_check_specs(cache.specs, (primal(f), map(primal, x)...))
    return cache.frule(f, x...)
end

function value_and_gradient!!(
    cache::NForwardCache,
    f::F,
    x::Vararg{Any,N};
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    _nforward_check_args_to_zero(args_to_zero, N + 1)
    _nforward_check_specs(cache.specs, (f, x...))
    _nforward_check_function_instance(cache.rrule.f, f)
    tangents = tuple_map(set_to_zero_maybe!!, cache.tangents, args_to_zero)
    return __value_and_gradient!!(cache.rrule, tuple_map(CoDual, (f, x...), tangents)...)
end
