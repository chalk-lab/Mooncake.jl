function hand_written_rule_test_cases(rng_ctor, ::Val{:fastmath})
    # The nfwd-backed scalar fastmath rules live in `low_level_maths.jl`; this
    # test set only keeps the remaining fastmath-specific cases local.
    test_cases = reduce(
        vcat,
        map([Float64, Float32]) do P
            return TestCase[
                TestCase(Base.FastMath.angle_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.atan_fast, P(5.4); perf_flag=:allocs),
                TestCase(Base.FastMath.atan_fast, P(5.4), P(3.2); perf_flag=:allocs),
                TestCase(Base.FastMath.pow_fast, P(5.0), Int32(2); perf_flag=:allocs),
                TestCase(Base.FastMath.exp10_fast, P(0.5); perf_flag=:stability_and_allocs),
                TestCase(Base.FastMath.exp2_fast, P(0.5); perf_flag=:stability_and_allocs),
                TestCase(Base.FastMath.exp_fast, P(5.0); perf_flag=:stability_and_allocs),
            ]
        end,
    )
    memory = Any[]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:fastmath})
    test_cases = reduce(
        vcat,
        map([Float64, Float32]) do P
            C = P === Float64 ? ComplexF64 : ComplexF32
            return TestCase[
                TestCase(Base.FastMath.abs2_fast, P(-5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.abs_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.acos_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.acosh_fast, P(1.2); perf_flag=:allocs),
                TestCase(Base.FastMath.add_fast, P(1.0), P(2.0); perf_flag=:allocs),
                TestCase(Base.FastMath.asin_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.asinh_fast, P(1.3); perf_flag=:allocs),
                TestCase(Base.FastMath.atanh_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.cbrt_fast, P(0.4); perf_flag=:allocs),
                TestCase(Base.FastMath.cis_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.cmp_fast, P(0.5), P(0.4); perf_flag=:allocs),
                TestCase(Base.FastMath.conj_fast, P(0.4); perf_flag=:allocs),
                TestCase(Base.FastMath.conj_fast, C(0.5, 0.4); perf_flag=:allocs),
                TestCase(Base.FastMath.cos_fast, P(0.4); perf_flag=:allocs),
                TestCase(Base.FastMath.cosh_fast, P(0.3); perf_flag=:allocs),
                TestCase(Base.FastMath.div_fast, P(5.0), P(1.1); perf_flag=:allocs),
                TestCase(Base.FastMath.eq_fast, P(5.5), P(5.5); perf_flag=:allocs),
                TestCase(Base.FastMath.eq_fast, P(5.5), P(5.4); perf_flag=:allocs),
                TestCase(Base.FastMath.expm1_fast, P(5.4); perf_flag=:allocs),
                TestCase(Base.FastMath.ge_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.ge_fast, P(4.0), P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.gt_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.gt_fast, P(4.0), P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.hypot_fast, P(5.1), P(3.2); perf_flag=:allocs),
                TestCase(Base.FastMath.inv_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.isfinite_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.isinf_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.isnan_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.issubnormal_fast, P(0.3); perf_flag=:allocs),
                TestCase(Base.FastMath.le_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.log10_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.log1p_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.log2_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.log_fast, P(0.5); perf_flag=:allocs),
                TestCase(Base.FastMath.lt_fast, P(0.5), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.lt_fast, P(5.0), P(0.4); perf_flag=:allocs),
                TestCase(Base.FastMath.max_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(
                    Base.FastMath.maximum!_fast, sin, P.([0.0, 0.0]), P.([5.0 4.0; 3.0 2.0])
                ),
                TestCase(
                    Base.FastMath.maximum_fast, P.([5.0, 4.0, 3.0]); perf_flag=:allocs
                ),
                TestCase(Base.FastMath.min_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.min_fast, P(4.0), P(5.0); perf_flag=:allocs),
                TestCase(
                    Base.FastMath.minimum!_fast, sin, P.([0.0, 0.0]), P.([5.0 4.0; 3.0 2.0])
                ),
                TestCase(
                    Base.FastMath.minimum_fast, P.([5.0, 3.0, 4.0]); perf_flag=:allocs
                ),
                TestCase(Base.FastMath.minmax_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.mul_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.ne_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.pow_fast, P(5.0), P(2.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sign_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sign_fast, P(-5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sin_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.cos_fast, P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sincos_fast, P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sinh_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sqrt_fast, P(5.0); perf_flag=:allocs),
                TestCase(Base.FastMath.sub_fast, P(5.0), P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.tan_fast, P(4.0); perf_flag=:allocs),
                TestCase(Base.FastMath.tanh_fast, P(0.5); perf_flag=:allocs),
            ]
        end,
    )
    memory = Any[]
    return test_cases, memory
end
