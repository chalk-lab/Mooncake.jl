using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, BFloat16s, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule, test_tangent_interface, test_tangent_splitting

# Core.BFloat16 is only available in Julia >= 1.11; no-op on 1.10.
@static if VERSION < v"1.11-"
    @info "Skipping Core.BFloat16 tests: on this platform, " *
        "`BFloat16` has no arithmetic support."
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
end
