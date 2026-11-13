using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra, Mooncake, SparseArrays, StableRNGs, Test
using Mooncake.TestUtils: test_rule

@testset "sparsearrays" begin
    rng = StableRNG(123)
    @testset "mul!(C, A::SparseMatrixCSC, B, α, β) $T $(ndims_B == 1 ? "vector" : "matrix") α=$(typeof(α)) β=$(typeof(β))" for T in
                                                                                                                               (
            Float64, Float32, ComplexF64, ComplexF32
        ),
        ndims_B in (1, 2),
        (α, β) in (
            (true, false),
            (true, true),
            (T(0.7), T(0.3)),
            (real(T)(0.7), real(T)(-0.4)),
            (T(1.3), false),
        )

        A = sprand(rng, T, 7, 5, 0.4)
        B = ndims_B == 1 ? randn(rng, T, 5) : randn(rng, T, 5, 3)
        C = ndims_B == 1 ? randn(rng, T, 7) : randn(rng, T, 7, 3)
        test_rule(rng, mul!, C, A, B, α, β; is_primitive=true, perf_flag=:none)
    end

    # Finite differences are not reliable in half precision, so compare with Float64 instead.
    @testset "Float16 agrees with Float64 (α=$α β=$β)" for (α, β) in (
        (true, false), (true, true), (0.7, 0.3), (1.3, false)
    )
        A = sprand(rng, 7, 5, 0.4)
        B = randn(rng, 5, 3)
        C = randn(rng, 7, 3)
        function loss(A, B, C, α, β)
            C = copy(C)
            mul!(C, A, B, α, β)
            return sum(abs2, C)
        end
        to16(x) = x isa Bool ? x : Float16(x)
        args64 = (A, B, C, α, β)
        args16 = (Float16.(A), Float16.(B), Float16.(C), to16(α), to16(β))
        cache64 = Mooncake.prepare_gradient_cache(loss, args64...)
        cache16 = Mooncake.prepare_gradient_cache(loss, args16...)
        _, (_, dA64, dB64, dC64) = Mooncake.value_and_gradient!!(cache64, loss, args64...)
        _, (_, dA16, dB16, dC16) = Mooncake.value_and_gradient!!(cache16, loss, args16...)
        @test dA16.fields.nzval ≈ dA64.fields.nzval rtol = 1e-2
        @test dB16 ≈ dB64 rtol = 1e-2
        @test dC16 ≈ dC64 rtol = 1e-2 atol = 1e-2
    end

    @testset "three-argument mul! and * forward to the rule" begin
        A = sprand(rng, ComplexF64, 6, 6, 0.5)
        x = randn(rng, ComplexF64, 6)
        y = randn(rng, ComplexF64, 6)
        test_rule(rng, mul!, y, A, x; is_primitive=false, perf_flag=:none)
        test_rule(rng, *, A, x; is_primitive=false, perf_flag=:none)
    end

    @testset "gradient through a sparse linear map" begin
        A = sprand(rng, ComplexF64, 30, 30, 0.2)
        x = randn(rng, ComplexF64, 30)
        f(x) = real(dot(x, A * x))
        _, (_, g) = Mooncake.value_and_gradient!!(
            Mooncake.prepare_gradient_cache(f, x), f, x
        )
        @test g ≈ (A + A') * x
    end
end
