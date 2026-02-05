function _compute_val_grad(x::Float64)
    f(x) = x^4
    cache = prepare_gradient_cache(f, x)
    val, grad = value_and_gradient!!(cache, f, x)
    return val, grad[2]
end

function _second_der(x::Float64)
    frule = build_frule(_compute_val_grad, x)
    return frule(
        zero_dual(_compute_val_grad),
        Dual(x, 1.0),
    )
end