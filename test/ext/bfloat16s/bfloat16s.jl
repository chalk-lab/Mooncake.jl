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

    # When x == 0 and 0 < y < 1: z = 0^y = 0, log(0) = -Inf, so z * log(x) = 0 * (-Inf) = NaN
    # unless guarded. These tests verify the nan_tangent_guard fixes prevent NaN in both rules.
    @testset "^ NaN guard: x=0" begin
        x, y = P(0), P(0.5)
        dx, dy = one(P), one(P)
        _lift(v::P, dv::P) = Mooncake.Lifted{P,1}(
            Mooncake.Dual(v, Mooncake.NTangent((dv,)))
        )

        # frule: x-tangent diverges to Inf (correct: d/dx(0^0.5) = +Inf);
        # y-tangent must be exactly 0 (guarded from 0*(-Inf)=NaN).
        fwd_result = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(1), ^), _lift(x, dx), _lift(y, dy)
        )
        @test Mooncake.primal(fwd_result) === x^y
        @test Mooncake.tangent(fwd_result, 1) === P(Inf)  # y-term guarded; x-term = _y * 0^(-0.5) * dx = Inf

        fwd_result_zero_dx = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(1), ^), _lift(x, zero(P)), _lift(y, dy)
        )
        @test Mooncake.tangent(fwd_result_zero_dx, 1) === zero(P)  # y-term guarded: z*log(0)*dy = 0

        # rrule: x-rdata is Inf (dz * _y * 0^(-0.5) = Inf, upstream gradient into diverging slope);
        # y-rdata must be exactly 0 (inner guard on z blocks 0*(-Inf)=NaN).
        _, pb = Mooncake.rrule!!(
            Mooncake.zero_fcodual(^), Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(y)
        )
        _, dx_r, dy_r = pb(one(P))
        @test dx_r === P(Inf)   # dz * _y * 0^(-0.5) = Inf
        @test dy_r === zero(P)  # inner guard: z=0 blocks z*log(0)*dz = NaN
    end

    @testset "Lifted dispatch (unary + binary + width-N)" begin
        # Direct `Lifted` invocations of BFloat16 rules. Bare-Dual entry
        # points were removed when the central adapter family was deleted;
        # this testset pins the surviving width-1 + width-N Lifted contract.
        _lift(v::P, dv::P) = Mooncake.Lifted{P,1}(
            Mooncake.Dual(v, Mooncake.NTangent((dv,)))
        )

        # Unary: sqrt at width-1.
        r1 = Mooncake.frule!!(Mooncake.zero_lifted(Val(1), sqrt), _lift(P(4), P(1)))
        @test r1 isa Mooncake.Lifted{P,1}
        @test Mooncake.primal(r1) === P(2)
        @test Mooncake.tangent(r1, 1) === P(2)^-1 * P(0.5)  # d/dx sqrt(4) = 0.25

        # Unary: sqrt at width-2.
        x2 = Mooncake.Lifted{P,2}(P(4), Mooncake.NTangent((P(1), P(0.5))))
        f2 = Mooncake.Lifted{typeof(sqrt),2}(sqrt, Mooncake.NoTangent())
        r2 = Mooncake.frule!!(f2, x2)
        @test r2 isa Mooncake.Lifted{P,2}
        inner = Mooncake._unlift(r2)
        @test Mooncake.primal(inner) === P(2)
        @test Mooncake.tangent(inner).lanes[1] ≈ P(0.25)
        @test Mooncake.tangent(inner).lanes[2] ≈ P(0.125)

        # Binary: max — `x` wins (x >= y) so tangent should come from x.
        rmax = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(1), max), _lift(P(5), P(1)), _lift(P(3), P(0))
        )
        @test rmax isa Mooncake.Lifted{P,1}
        @test Mooncake.primal(rmax) === P(5)
        @test Mooncake.tangent(rmax, 1) === P(1)

        # Conversion: Float32(BFloat16).
        rconv = Mooncake.frule!!(Mooncake.zero_lifted(Val(1), Float32), _lift(P(1.5), P(1)))
        @test rconv isa Mooncake.Lifted{Float32,1}
        @test Mooncake.primal(rconv) === 1.5f0
        @test Mooncake.tangent(rconv, 1) === 1.0f0
    end
end
