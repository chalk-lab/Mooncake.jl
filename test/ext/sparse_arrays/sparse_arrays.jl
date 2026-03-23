using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using LinearAlgebra, Mooncake, SelectedInversion, SparseArrays, StableRNGs, Test
using Mooncake: NoTangent, zero_tangent, build_rrule
using Mooncake.TestUtils: test_rule

@testset "SparseArrays CHOLMOD support" begin
    # Struct containing Factor doesn't crash AD (reproduces #698)
    smat = sparse(1.0I(3))
    tup = (; smat, schol=cholesky(smat))
    f = x -> x' * tup.smat * x
    rule = build_rrule(f, [1.0, 2.0, 3.0])
    out, grad = Mooncake.value_and_gradient!!(rule, f, [1.0, 2.0, 3.0])
    @test grad[2] ≈ [2.0, 4.0, 6.0]
end

# Helper: build a symmetric SPD sparse matrix from lower-triangle parameters.
# A = [p[1] p[2] 0; p[2] p[3] p[4]; 0 p[4] p[5]]
const _colptr = [1, 3, 6, 8]
const _rowval = [1, 2, 1, 2, 3, 2, 3]
function _make_spd(p)
    nzval = [p[1], p[2], p[2], p[3], p[4], p[4], p[5]]
    return SparseMatrixCSC(3, 3, _colptr, _rowval, nzval)
end
const _p0 = [4.0, 1.0, 3.0, 1.0, 5.0]

@testset "logdet(cholesky(A)) gradient — sparse" begin
    f = p -> logdet(cholesky(_make_spd(p)))
    test_rule(StableRNG(123), f, _p0; is_primitive=false, unsafe_perturb=true)
end

@testset "cholesky(A) \\ b gradient — sparse" begin
    b = [1.0, 2.0, 3.0]
    g = p -> sum(cholesky(_make_spd(p)) \ b)
    test_rule(StableRNG(123), g, _p0; is_primitive=false, unsafe_perturb=true)
end

@testset "cholesky(A) \\ B gradient — matrix RHS" begin
    B = [1.0 0.5; 2.0 1.0; 3.0 1.5]
    g = p -> sum(cholesky(_make_spd(p)) \ B)
    test_rule(StableRNG(123), g, _p0; is_primitive=false, unsafe_perturb=true)
end

function _make_sym_lower(p)
    nzval = [p[1], p[2], p[3], p[4], p[5]]
    S = SparseMatrixCSC(3, 3, [1, 3, 5, 6], [1, 2, 2, 3, 3], nzval)
    return Symmetric(S, :L)
end

@testset "logdet(cholesky(Symmetric(A,:L))) gradient" begin
    f = p -> logdet(cholesky(_make_sym_lower(p)))
    test_rule(StableRNG(123), f, _p0; is_primitive=false, unsafe_perturb=true)
end

# Upper triangle: col 1 has row [1], col 2 has rows [1,2], col 3 has rows [2,3]
function _make_sym_upper(p)
    S = SparseMatrixCSC(3, 3, [1, 2, 4, 6], [1, 1, 2, 2, 3], [p[1], p[2], p[3], p[4], p[5]])
    return Symmetric(S, :U)
end

@testset "logdet(cholesky(Symmetric(A,:U))) gradient" begin
    f = p -> logdet(cholesky(_make_sym_upper(p)))
    test_rule(StableRNG(123), f, _p0; is_primitive=false, unsafe_perturb=true)
end
