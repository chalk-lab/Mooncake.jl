using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, BFloat16s, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule, test_tangent_interface, test_tangent_splitting

# Core.BFloat16 requires Julia >= 1.11.
# BFloat16s.BFloat16 === Core.BFloat16 is not guaranteed on all platforms.
if VERSION < v"1.11-" || BFloat16s.BFloat16 !== Core.BFloat16
    @info "Skipping Core.BFloat16 tests: BFloat16s.BFloat16 !== Core.BFloat16 on this platform/Julia version."
    exit(0)
end

const sr = StableRNG
const P = Core.BFloat16

# The shape NNlib gives `sigmoid`: `exp(-abs(x))` selected by an `ifelse`. Smooth at zero
# even though `abs` is not, because the kink cancels between the two branches.
function sigmoid_shaped(x)
    (t=exp(-abs(x)); ifelse(x >= zero(x), inv(one(x) + t), t / (one(x) + t)))
end

@testset "bfloat16s" begin
    @testset "tangent interface" begin
        rng = sr(123)
        test_tangent_interface(rng, P(1.5))
        test_tangent_splitting(rng, P(1.5))
    end

    cases = [
        (Float32, P(0.5)),
        (Float64, P(0.5)),
        (P, 0.5f0),
        (P, 0.5),
        (sqrt, P(0.5)),
        (cbrt, P(0.4)),
        (exp, P(0.2)),
        (exp2, P(1.12)),
        (exp10, P(0.249)),
        (expm1, P(-0.3)),
        (log, P(0.1)),
        (log2, P(0.15)),
        (log10, P(0.1)),
        (log1p, P(0.95)),
        (sin, P(1.1)),
        (cos, P(0.2)),
        (tan, P(0.5)),
        (asin, P(0.77)),
        (acos, P(0.2)),
        (atan, P(0.77)),
        (sinh, P(-0.56)),
        (cosh, P(0.4)),
        (tanh, P(0.25)),
        (asinh, P(1.45)),
        (acosh, P(1.56)),
        (atanh, P(-0.44)),
        (hypot, P(0.4), P(0.3)),
        (^, P(0.4), P(0.3)),
        (max, P(0.2), P(0.15)),
        (max, P(0.22), P(0.18)),
        (min, P(1.5), P(0.5)),
        (min, P(0.22), P(0.18)),
        (abs, P(0.5)),
        (abs, P(-0.5)),
        (Base.eps, P(0.2)),
        (nextfloat, P(0.25)),
        (prevfloat, P(1.0)),
    ]

    # Tolerances reflect BFloat16's ~3-digit precision. Test values are in [0.125, 0.25)
    # so the ε=1e-2 FD perturbation (≈0.00129) exceeds the BF16 half-spacing (0.000488)
    # and is captured, but snaps to one grid step (0.000977), giving ~24% relative error
    # → rtol=0.4. Two functions (acos, exp10) also suffer output-side absorption at some
    # inputs, yielding |LHS-RHS|≈0.16 even when ẏ_fd≠0 → atol=0.2.
    @testset "$(f) $(map(typeof, xs))" for (f, xs...) in cases
        test_rule(sr(123), f, xs...; is_primitive=true, atol=0.2, rtol=0.4)
    end

    # `sigmoid_shaped` at exactly zero comes out right for BFloat16, where it does not for
    # the IEEEFloat types, and the reason is this repo's own choice rather than BFloat16s'.
    # `ext/MooncakeBFloat16sExt.jl` makes `abs` a primitive whose rules branch on
    # `x >= zero(P)`, taking the positive side at zero and so reporting a derivative of 1;
    # AD therefore never traces `abs` down to the `abs_float` intrinsic, whose rule
    # multiplies by `sign(x)` and collapses at zero because `sign(0) == 0`. The chain rule
    # then recovers 1/4. So these two cases are what pin that rule's behaviour at zero: with
    # `abs` traced through BFloat16s' own `BFloat16(abs(Float32(x)))`, the derivative here
    # is 0 instead, exactly as for the IEEEFloat types — see #1257 for those.
    #
    # `abs` itself cannot be checked at zero this way, which is why it is absent from the
    # cases above: at a kink a central difference returns the midpoint of the one-sided
    # derivatives, 0, against the rule's 1. This composite is smooth there — the kink
    # cancels between the `ifelse` branches — so finite differences do agree with it.
    # Composite, so not a primitive.
    @testset "sigmoid_shaped at zero" for x in (P(0), P(0.5))
        test_rule(sr(123), sigmoid_shaped, x; is_primitive=false, atol=0.2, rtol=0.4)
    end

    # When x == 0 and 0 < y < 1: z = 0^y = 0, log(0) = -Inf, so z * log(x) = 0 * (-Inf) = NaN
    # unless guarded. These tests verify the nan_tangent_guard fixes prevent NaN in both rules.
    @testset "^ NaN guard: x=0" begin
        x, y = P(0), P(0.5)
        dx, dy = one(P), one(P)

        # frule: x-tangent diverges to Inf (correct: d/dx(0^0.5) = +Inf);
        # y-tangent must be exactly 0 (guarded from 0*(-Inf)=NaN).
        fwd_result = Mooncake.frule!!(
            Mooncake.Dual(^, Mooncake.NoTangent()),
            Mooncake.Dual(x, dx),
            Mooncake.Dual(y, dy),
        )
        @test Mooncake.primal(fwd_result) === x^y
        @test Mooncake.tangent(fwd_result) === P(Inf)  # y-term guarded to 0; x-term = _y * 0^(-0.5) * dx = Inf

        fwd_result_zero_dx = Mooncake.frule!!(
            Mooncake.Dual(^, Mooncake.NoTangent()),
            Mooncake.Dual(x, zero(P)),  # dx = 0: x-term guarded to 0
            Mooncake.Dual(y, dy),
        )
        @test Mooncake.tangent(fwd_result_zero_dx) === zero(P)  # y-term guarded: z*log(0)*dy = 0

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
