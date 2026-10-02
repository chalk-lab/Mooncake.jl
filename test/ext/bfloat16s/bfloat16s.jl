include(joinpath(@__DIR__, "..", "pin_develop_or_skip.jl"))
pin_develop_or_skip(@__DIR__, "BFloat16s")

using AllocCheck, BFloat16s, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils:
    TestCase, test_rule, test_tangent_interface, test_tangent_splitting

# Core.BFloat16 requires Julia >= 1.11.
# BFloat16s.BFloat16 === Core.BFloat16 is not guaranteed on all platforms.
if VERSION < v"1.11-" || BFloat16s.BFloat16 !== Core.BFloat16
    @info "Skipping Core.BFloat16 tests: BFloat16s.BFloat16 !== Core.BFloat16 on this platform/Julia version."
    exit(0)
end

const sr = StableRNG
const P = Core.BFloat16

# NNlib's `sigmoid` shape: `exp(-abs(x))` under an `ifelse`, smooth at zero even though
# `abs` is not, because the kink cancels between the two branches.
function sigmoid_shaped(x)
    (t=exp(-abs(x)); ifelse(x >= zero(x), inv(one(x) + t), t / (one(x) + t)))
end

@testset "bfloat16s" begin
    @testset "tangent interface" begin
        rng = sr(123)
        test_tangent_interface(rng, P(1.5))
        # Mapped seeding and scaling must keep BFloat16 conversions scalar on LLVM 16.
        test_tangent_interface(rng, (P(1.5), P(2.0)))
        test_tangent_interface(rng, P[1.5, 2.0])
        test_tangent_splitting(rng, P(1.5))
    end

    cases = [
        TestCase(Float32, P(0.5)),
        TestCase(Float64, P(0.5)),
        TestCase(P, 0.5f0),
        TestCase(P, 0.5),
        TestCase(sqrt, P(0.5)),
        TestCase(cbrt, P(0.4)),
        TestCase(exp, P(0.2)),
        # In the fine-spacing range: at P(1.12) the reverse rule is correct
        # (grad == exp2(x)·log 2) but BF16's coarse spacing cannot resolve the
        # finite-difference check.
        TestCase(exp2, P(0.15)),
        TestCase(exp10, P(0.249)),
        TestCase(expm1, P(-0.3)),
        TestCase(log, P(0.1)),
        TestCase(log2, P(0.15)),
        TestCase(log10, P(0.1)),
        TestCase(log1p, P(0.95)),
        TestCase(sin, P(1.1)),
        TestCase(cos, P(0.2)),
        TestCase(tan, P(0.5)),
        TestCase(asin, P(0.77)),
        TestCase(acos, P(0.2)),
        TestCase(atan, P(0.77)),
        TestCase(sinh, P(-0.56)),
        TestCase(cosh, P(0.4)),
        TestCase(tanh, P(0.25)),
        TestCase(asinh, P(1.45)),
        TestCase(acosh, P(1.56)),
        TestCase(atanh, P(-0.44)),
        TestCase(hypot, P(0.4), P(0.3)),
        TestCase(^, P(0.4), P(0.3)),
        TestCase(max, P(0.2), P(0.15)),
        TestCase(max, P(0.22), P(0.18)),
        TestCase(min, P(1.5), P(0.5)),
        TestCase(min, P(0.22), P(0.18)),
        TestCase(abs, P(0.5)),
        TestCase(abs, P(-0.5)),
        TestCase(Base.eps, P(0.2)),
        TestCase(nextfloat, P(0.25)),
        TestCase(prevfloat, P(1.0)),
    ]

    # Tolerances reflect BFloat16's ~3-digit precision. Test values are in [0.125, 0.25)
    # so the ε=1e-2 FD perturbation (≈0.00129) exceeds the BF16 half-spacing (0.000488)
    # and is captured, but snaps to one grid step (0.000977), giving ~24% relative error
    # → rtol=0.4. Two functions (acos, exp10) also suffer output-side absorption at some
    # inputs, yielding |LHS-RHS|≈0.16 even when ẏ_fd≠0 → atol=0.2.
    # Precision tolerances are suite fallbacks.
    rule_options = (; atol=0.2, rtol=0.4)
    for (tc, name) in zip(cases, Mooncake.TestUtils._test_case_names(cases))
        # Chunked forward checks must not change the reverse finite-difference directions.
        for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
            test_rule(sr(123), tc; mode, fallbacks=rule_options, name)
        end
        if tc.f === hypot
            # Scalar arithmetic barriers must not box tuple indexing in chunked rules.
            args = map(x -> Mooncake.zero_lifted(Val(8), x), (tc.f, tc.args...))
            Mooncake.frule!!(args...)
            @test Mooncake.TestUtils.count_allocs(Mooncake.frule!!, args...) == 0
        end
    end

    # Right at zero for BFloat16 but not the IEEEFloat types: this repo's `abs` rules branch
    # on `x >= zero(P)`, never reaching the `abs_float` intrinsic's `sign(x)`. Pinned by
    # calling the rule, since `test_rule` passes when any one step agrees and a collapsed 0
    # matches six of the seven: below ε = 1e-2, `sigmoid_shaped(±ε)` rounds back to `0.5`.
    # `abs` is absent above: its central difference returns 0 at the kink, the rule 1.
    @testset "sigmoid_shaped at zero" begin
        rule = Mooncake.build_rrule(sigmoid_shaped, P(0))
        _, pb = rule(Mooncake.zero_fcodual(sigmoid_shaped), Mooncake.zero_fcodual(P(0)))
        @test Float64(pb(one(P))[2]) ≈ 0.25 rtol = 1e-2
        # `test_rule`'s value, interface and caching checks survive that blind spot.
        for x in (P(0), P(0.5)), mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
            test_rule(sr(123), sigmoid_shaped, x; is_primitive=false, mode, rule_options...)
        end
    end

    # When x == 0 and 0 < y < 1: z = 0^y = 0, log(0) = -Inf, so z * log(x) = 0 * (-Inf) = NaN
    # unless guarded. These tests verify the nan_tangent_guard fixes prevent NaN in both rules.
    @testset "^ NaN guard: x=0" begin
        x, y = P(0), P(0.5)
        dx, dy = one(P), one(P)

        # frule: x-tangent diverges to Inf (correct: d/dx(0^0.5) = +Inf);
        # y-tangent must be exactly 0 (guarded from 0*(-Inf)=NaN).
        fwd_result = Mooncake.frule!!(
            Mooncake.lift(^, Mooncake.NoTangent()),
            Mooncake.lift(x, dx),
            Mooncake.lift(y, dy),
        )
        @test Mooncake.primal(fwd_result) === x^y
        @test last(Mooncake.unlift(fwd_result)) === P(Inf)  # y-term guarded to 0; x-term = _y * 0^(-0.5) * dx = Inf

        fwd_result_zero_dx = Mooncake.frule!!(
            Mooncake.lift(^, Mooncake.NoTangent()),
            Mooncake.lift(x, zero(P)),  # dx = 0: x-term guarded to 0
            Mooncake.lift(y, dy),
        )
        @test last(Mooncake.unlift(fwd_result_zero_dx)) === zero(P)  # y-term guarded: z*log(0)*dy = 0

        # rrule: x-rdata is Inf (dz * _y * 0^(-0.5) = Inf, upstream gradient into diverging slope);
        # y-rdata must be exactly 0 (inner guard on z blocks 0*(-Inf)=NaN).
        _, pb = Mooncake.rrule!!(
            Mooncake.zero_fcodual(^), Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(y)
        )
        _, dx_r, dy_r = pb(one(P))
        @test dx_r === P(Inf)   # dz * _y * 0^(-0.5) = Inf
        @test dy_r === zero(P)  # inner guard: z=0 blocks z*log(0)*dz = NaN
    end
end
