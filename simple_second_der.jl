function _compute_val_grad(x::Float64)
    f(x) = x^4
    rule = build_rrule(f, x)
    val, pb!! = rule(zero_fcodual(f), zero_fcodual(x))
    der = pb!!(1.0)
    return val, der
end

function _second_der(x::Float64)
    frule = build_frule(_compute_val_grad, x)
    return frule(
        zero_dual(_compute_val_grad),
        Dual(x, 1.0),
    )
end