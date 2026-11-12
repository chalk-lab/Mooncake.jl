@testset "blas (ComplexF64)" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:blas_ComplexF64))
end

@testset "gemv! pullback allocations" for P in (ComplexF32, ComplexF64),
    trans in ('N', 'T', 'C'),
    alpha in (0, 2)

    x = ones(P, 256)
    out, pb = Mooncake.rrule!!(
        map(
            Mooncake.zero_fcodual,
            (BLAS.gemv!, trans, P(alpha), ones(P, 256, 256), x, one(P), zero(x)),
        )...,
    )
    fill!(Mooncake.tangent(out), one(P))
    pb(Mooncake.NoRData())
    # The transpose pullback copies dy for BLAS's missing conjugate-only operation.
    limit = trans == 'T' ? TestUtils.count_allocs(x -> conj.(x), x) : 0
    @test TestUtils.count_allocs(pb, Mooncake.NoRData()) <= limit
end
