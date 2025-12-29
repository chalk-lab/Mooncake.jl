@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(prepare_gradient_cache),Vararg}
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    _kwargs::Dual{<:NamedTuple},
    ::Dual{typeof(prepare_gradient_cache)},
    args::Vararg{Dual},
)
    cache = prepare_gradient_cache(map(primal, args)...; primal(_kwargs)...)
    return zero_dual(cache)
end

@is_primitive(
    MinimalCtx,
    Tuple{typeof(Core.kwcall),NamedTuple,typeof(prepare_derivative_cache),Vararg}
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    _kwargs::Dual{<:NamedTuple},
    ::Dual{typeof(prepare_derivative_cache)},
    args::Vararg{Dual},
)
    cache = prepare_derivative_cache(map(primal, args)...; primal(_kwargs)...)
    return zero_dual(cache)
end

@is_primitive(
    MinimalCtx, Tuple{typeof(Core.kwcall),NamedTuple,typeof(prepare_pullback_cache),Vararg}
)
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    _kwargs::Dual{<:NamedTuple},
    ::Dual{typeof(prepare_pullback_cache)},
    args::Vararg{Dual},
)
    cache = prepare_pullback_cache(map(primal, args)...; primal(_kwargs)...)
    return zero_dual(cache)
end

@is_primitive(MinimalCtx, Tuple{typeof(zero_tangent),Any})
function frule!!(::Dual{typeof(zero_tangent)}, x::Dual)
    return zero_dual(zero_tangent(primal(x)))
end
