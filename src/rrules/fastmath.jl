@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp_fast),IEEEFloat}
function frule!!(::Dual{typeof(Base.FastMath.exp_fast)}, x::Dual{P}) where {P<:IEEEFloat}
    y = Base.FastMath.exp_fast(primal(x))
    return Dual(y, y * tangent(x))
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    yp = Base.FastMath.exp_fast(primal(x))
    exp_fast_pb!!(dy::P) = NoRData(), dy * yp
    return CoDual(yp, NoFData()), exp_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp2_fast),IEEEFloat}
function frule!!(::Dual{typeof(Base.FastMath.exp2_fast)}, x::Dual{P}) where {P<:IEEEFloat}
    y = Base.FastMath.exp2_fast(primal(x))
    return Dual(y, y * tangent(x) * P(log(2)))
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp2_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    yp = Base.FastMath.exp2_fast(primal(x))
    exp2_fast_pb!!(dy::P) = NoRData(), dy * yp * P(log(2))
    return CoDual(yp, NoFData()), exp2_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.exp10_fast),IEEEFloat}
function frule!!(::Dual{typeof(Base.FastMath.exp10_fast)}, x::Dual{P}) where {P<:IEEEFloat}
    y = Base.FastMath.exp10_fast(primal(x))
    return Dual(y, y * tangent(x) * P(log(10)))
end
function rrule!!(
    ::CoDual{typeof(Base.FastMath.exp10_fast)}, x::CoDual{P}
) where {P<:IEEEFloat}
    yp = Base.FastMath.exp10_fast(primal(x))
    exp2_fast_pb!!(dy::P) = NoRData(), dy * yp * P(log(10))
    return CoDual(yp, NoFData()), exp2_fast_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.FastMath.sincos),IEEEFloat}
function frule!!(::Dual{typeof(Base.FastMath.sincos)}, x::Dual{P}) where {P<:IEEEFloat}
    y = Base.FastMath.sincos(primal(x))
    dx = tangent(x)
    return Dual(y, (y[2] * dx, -y[1] * dx))
end
function rrule!!(::CoDual{typeof(Base.FastMath.sincos)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = Base.FastMath.sincos(primal(x))
    sincos_fast_adj!!(dy::Tuple{P,P}) = NoRData(), dy[1] * y[2] - dy[2] * y[1]
    return CoDual(y, NoFData()), sincos_fast_adj!!
end

@is_primitive MinimalCtx Tuple{typeof(Base.log),Union{IEEEFloat,Int}}
@zero_derivative MinimalCtx Tuple{typeof(log),Int}

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:fastmath})
    test_cases = reduce(
        vcat,
        map([Float64, Float32]) do P
            return Any[
                (false, :stability_and_allocs, nothing, cosh, P(0.3)),
                (false, :stability_and_allocs, nothing, sinh, P(0.3)),
                (false, :stability_and_allocs, nothing, Base.FastMath.exp10_fast, P(0.5)),
                (false, :stability_and_allocs, nothing, Base.FastMath.exp2_fast, P(0.5)),
                (false, :stability_and_allocs, nothing, Base.FastMath.exp_fast, P(5.0)),
                (false, :stability_and_allocs, nothing, Base.FastMath.sincos, P(3.0)),
            ]
        end,
    )
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:fastmath})
    test_cases = reduce(
        vcat,
        map([Float64, Float32]) do P
            C = P === Float64 ? ComplexF64 : ComplexF32
            return Any[
                (false, :allocs, nothing, Base.FastMath.abs2_fast, P(-5.0)),
                (false, :allocs, nothing, Base.FastMath.abs_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.acos_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.acosh_fast, P(1.2)),
                (false, :allocs, nothing, Base.FastMath.add_fast, P(1.0), P(2.0)),
                (false, :allocs, nothing, Base.FastMath.angle_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.asin_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.asinh_fast, P(1.3)),
                (false, :allocs, nothing, Base.FastMath.atan_fast, P(5.4)),
                (false, :allocs, nothing, Base.FastMath.atanh_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.cbrt_fast, P(0.4)),
                (false, :allocs, nothing, Base.FastMath.cis_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.cmp_fast, P(0.5), P(0.4)),
                (false, :allocs, nothing, Base.FastMath.conj_fast, P(0.4)),
                (false, :allocs, nothing, Base.FastMath.conj_fast, C(0.5, 0.4)),
                (false, :allocs, nothing, Base.FastMath.cos_fast, P(0.4)),
                (false, :allocs, nothing, Base.FastMath.cosh_fast, P(0.3)),
                (false, :allocs, nothing, Base.FastMath.div_fast, P(5.0), P(1.1)),
                (false, :allocs, nothing, Base.FastMath.eq_fast, P(5.5), P(5.5)),
                (false, :allocs, nothing, Base.FastMath.eq_fast, P(5.5), P(5.4)),
                (false, :allocs, nothing, Base.FastMath.expm1_fast, P(5.4)),
                (false, :allocs, nothing, Base.FastMath.ge_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.ge_fast, P(4.0), P(5.0)),
                (false, :allocs, nothing, Base.FastMath.gt_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.gt_fast, P(4.0), P(5.0)),
                (false, :allocs, nothing, Base.FastMath.hypot_fast, P(5.1), P(3.2)),
                (false, :allocs, nothing, Base.FastMath.inv_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.isfinite_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.isinf_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.isnan_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.issubnormal_fast, P(0.3)),
                (false, :allocs, nothing, Base.FastMath.le_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.log10_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.log1p_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.log2_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.log_fast, P(0.5)),
                (false, :allocs, nothing, Base.FastMath.lt_fast, P(0.5), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.lt_fast, P(5.0), P(0.4)),
                (false, :allocs, nothing, Base.FastMath.max_fast, P(5.0), P(4.0)),
                (
                    false,
                    :none,
                    nothing,
                    Base.FastMath.maximum!_fast,
                    sin,
                    P.([0.0, 0.0]),
                    P.([5.0 4.0; 3.0 2.0]),
                ),
                (false, :allocs, nothing, Base.FastMath.maximum_fast, P.([5.0, 4.0, 3.0])),
                (false, :allocs, nothing, Base.FastMath.min_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.min_fast, P(4.0), P(5.0)),
                (
                    false,
                    :none,
                    nothing,
                    Base.FastMath.minimum!_fast,
                    sin,
                    P.([0.0, 0.0]),
                    P.([5.0 4.0; 3.0 2.0]),
                ),
                (false, :allocs, nothing, Base.FastMath.minimum_fast, P.([5.0, 3.0, 4.0])),
                (false, :allocs, nothing, Base.FastMath.minmax_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.mul_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.ne_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.pow_fast, P(5.0), P(2.0)),
                # (:allocs, Base.FastMath.pow_fast, P(5.0), 2), # errors -- NEEDS RULE!
                # (:allocs, Base.FastMath.rem_fast, P(5.0), P(2.0)), # error -- NEEDS RULE! 
                (false, :allocs, nothing, Base.FastMath.sign_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.sign_fast, P(-5.0)),
                (false, :allocs, nothing, Base.FastMath.sin_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.cos_fast, P(4.0)),
                (false, :allocs, nothing, Base.FastMath.sincos_fast, P(4.0)),
                (false, :allocs, nothing, Base.FastMath.sinh_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.sqrt_fast, P(5.0)),
                (false, :allocs, nothing, Base.FastMath.sub_fast, P(5.0), P(4.0)),
                (false, :allocs, nothing, Base.FastMath.tan_fast, P(4.0)),
                (false, :allocs, nothing, Base.FastMath.tanh_fast, P(0.5)),
            ]
        end,
    )
    memory = Any[]
    return test_cases, memory
end
