@testset "lapack" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:lapack))

    # Transitional: pin the shared cotangent until the Lifted registry shares seeds.
    @static if VERSION > v"1.11-"
        @testset "lacpy! shared cotangent" for P in
                                               (Float32, Float64, ComplexF32, ComplexF64),
            uplo in ('U', 'L', 'A')

            A = P[2 1; 1 3]
            dA = fill(P(2), size(A))
            before = copy(dA)
            d = CoDual(A, dA)
            _, pb = Mooncake.rrule!!(
                Mooncake.zero_fcodual(LAPACK.lacpy!), d, d, Mooncake.zero_fcodual(uplo)
            )
            @test dA == before
            pb(Mooncake.NoRData())
            @test dA == before
        end
    end
end
