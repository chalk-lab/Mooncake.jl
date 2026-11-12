@testset "lapack" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:lapack))

    # Test cases cannot observe shared fdata before the pullback.
    @static if VERSION > v"1.11-"
        @testset "lacpy! shared cotangent" for P in
                                               (Float32, Float64, ComplexF32, ComplexF64),
            uplo in ('U', 'L', 'A'),
            shape in ((2, 2), (3, 2), (1, 2))

            A = if shape == (2, 2)
                P[2 1; 1 3]
            else
                view(ones(P, 5, 4), 2:(shape[1] + 1), StepRangeLen(2, 0, shape[2]))
            end
            d = Mooncake.zero_fcodual(A)
            dA = Mooncake.arrayify(d)[2]
            fill!(dA, P(2))
            before = copy(dA)
            _, pb = Mooncake.rrule!!(
                Mooncake.zero_fcodual(LAPACK.lacpy!), d, d, Mooncake.zero_fcodual(uplo)
            )
            @test dA == before
            pb(Mooncake.NoRData())
            @test dA == before
            for N in (1, 8)
                slot = Mooncake.randn_lifted(Val(N), StableRNG(71), A)
                before = deepcopy(Mooncake.arrayify(slot)[2])
                Mooncake.frule!!(
                    Mooncake.zero_lifted(Val(N), LAPACK.lacpy!),
                    slot,
                    slot,
                    Mooncake.zero_lifted(Val(N), uplo),
                )
                @test Mooncake.arrayify(slot)[2] == before
            end
        end
    end
end
