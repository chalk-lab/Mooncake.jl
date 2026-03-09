using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, BFloat16s, JET, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule, test_tangent_interface, test_tangent_splitting

const P = Core.BFloat16
const sr = StableRNG

if Core.BFloat16 !== BFloat16s.BFloat16
    @info "Skipping Core.BFloat16 tests: on this platform BFloat16s.BFloat16 is a " *
        "separate type and Core.BFloat16 has no arithmetic support (LLVM < 19)."
    # Tests run on x86_64 (LLVM >= 15) where BFloat16s.BFloat16 === Core.BFloat16.
    exit(0)
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
        (exp, P(1.1)),
        (exp2, P(1.12)),
        (exp10, P(0.45)),
        (expm1, P(-0.3)),
        (log, P(0.1)),
        (log2, P(0.15)),
        (log10, P(0.1)),
        (log1p, P(0.95)),
        (sin, P(1.1)),
        (cos, P(1.1)),
        (tan, P(0.5)),
        (sinpi, P(1.5)),
        (cospi, P(-0.5)),
        (asin, P(0.77)),
        (acos, P(0.53)),
        (atan, P(0.77)),
        (sinh, P(-0.56)),
        (cosh, P(0.4)),
        (tanh, P(0.25)),
        (asinh, P(1.45)),
        (acosh, P(1.56)),
        (atanh, P(-0.44)),
        (hypot, P(0.4), P(0.3)),
        (^, P(0.4), P(0.3)),
        (atan, P(1.5), P(0.23)),
        (max, P(1.5), P(0.5)),
        (max, P(0.45), P(1.1)),
        (min, P(1.5), P(0.5)),
        (min, P(0.45), P(1.1)),
        (abs, P(0.5)),
        (abs, P(-0.5)),
        (Base.eps, P(1.0)),
        (nextfloat, P(0.25)),
        (prevfloat, P(1.0)),
    ]

    @testset "$(f) $(map(typeof, xs))" for (f, xs...) in cases
        test_rule(sr(123), f, xs...; is_primitive=true, atol=0.5, rtol=0.5)
    end
end
