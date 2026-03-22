using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra, Mooncake, SparseArrays, StableRNGs, Test
using Mooncake: NoTangent, zero_tangent, build_rrule

@testset "SparseArrays CHOLMOD support" begin
    # Test 1: tangent_type returns NoTangent for Factor
    F = cholesky(sparse(1.0I(3)))
    @test Mooncake.tangent_type(typeof(F)) == NoTangent

    # Test 2: zero_tangent works on a Factor
    @test zero_tangent(F) == NoTangent()

    # Test 3: Struct containing Factor doesn't crash AD (reproduces #698)
    smat = sparse(1.0I(3))
    tup = (; smat, schol = cholesky(smat))
    f = x -> x' * tup.smat * x
    rule = build_rrule(f, [1.0, 2.0, 3.0])
    # Should not throw

    # Test 4: Gradient is correct when Factor is present but unused
    out, grad = Mooncake.value_and_gradient!!(rule, f, [1.0, 2.0, 3.0])
    @test grad[2] ≈ [2.0, 4.0, 6.0]
end
