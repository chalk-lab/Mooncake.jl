using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using DiffTests, LinearAlgebra, Mooncake, Random, StableRNGs, Test

sr(n::Int) = StableRNG(n)

const TEST_CASES = vcat(
    tuple.(
        fill(false, length(DiffTests.NUMBER_TO_NUMBER_FUNCS)),
        DiffTests.NUMBER_TO_NUMBER_FUNCS,
        rand(sr(1), length(DiffTests.NUMBER_TO_NUMBER_FUNCS)) .+ 1e-1,
    ),
    tuple.(
        fill(false, length(DiffTests.NUMBER_TO_ARRAY_FUNCS)),
        DiffTests.NUMBER_TO_ARRAY_FUNCS,
        [rand(sr(2)) + 1e-1 for _ in DiffTests.NUMBER_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS)),
        DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS,
        [rand(sr(3), 5) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
        [rand(sr(4)) + 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.VECTOR_TO_NUMBER_FUNCS)),
        DiffTests.VECTOR_TO_NUMBER_FUNCS,
        [rand(sr(5), 5) .+ 1e-1 for _ in DiffTests.VECTOR_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.MATRIX_TO_NUMBER_FUNCS)),
        DiffTests.MATRIX_TO_NUMBER_FUNCS,
        [rand(sr(6), 5, 5) .+ 1e-1 for _ in DiffTests.MATRIX_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS)),
        DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS,
        [rand(sr(7), 5, 5) .+ 1e-1 + I for _ in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS],
        [rand(sr(8), 5, 5) .+ 1e-1 + I for _ in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS)),
        DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS,
        [rand(sr(9), 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
        [rand(sr(10), 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
        [rand(sr(11), 5, 5) .+ 1e-1 for _ in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS)),
        DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS,
        [rand(sr(12), 26) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
        [rand(sr(13), 26) .+ 1e-1 for _ in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.VECTOR_TO_VECTOR_FUNCS)),
        DiffTests.VECTOR_TO_VECTOR_FUNCS,
        [rand(sr(14), 26) .+ 1e-1 for _ in DiffTests.VECTOR_TO_VECTOR_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.ARRAY_TO_ARRAY_FUNCS)),
        DiffTests.ARRAY_TO_ARRAY_FUNCS,
        [rand(sr(15), 26) .+ 1e-1 for _ in DiffTests.ARRAY_TO_ARRAY_FUNCS],
    ),
    tuple.(
        fill(false, length(DiffTests.MATRIX_TO_MATRIX_FUNCS)),
        DiffTests.MATRIX_TO_MATRIX_FUNCS,
        [rand(sr(16), 5, 5) .+ 1e-1 for _ in DiffTests.MATRIX_TO_MATRIX_FUNCS],
    ),
)

@testset "diff_tests" begin
    @testset "$f, $(typeof(x))" for (n, (interface_only, f, x...)) in enumerate(
        vcat(TEST_CASES[1:66], TEST_CASES[68:89], TEST_CASES[91:end])
    )
        @info "$n: $(typeof((f, x...)))"
        Mooncake.TestUtils.test_rule(StableRNG(123456), f, x...; is_primitive=false)
    end
end
