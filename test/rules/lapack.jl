@testset "lapack" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:lapack))

    @testset "zero output cotangents with nonfinite operands" for P in (Float32, Float64),
        bad in (NaN, Inf)

        A, N, B = P[2 1; 0 3], fill(P(bad), 2, 2), ones(P, 2, 2)
        cases = Any[
            ((Core.kwcall, (; check=false), LAPACK.getrf!, N), 4),
            ((LAPACK.getri!, N, [1, 2]), 2),
            ((LAPACK.potrf!, 'U', N), 3),
        ]
        for lhs in (false, true)
            append!(
                cases,
                [
                    ((LAPACK.trtrs!, 'U', 'N', 'N', lhs ? N : A, lhs ? B : N), 6),
                    ((LAPACK.getrs!, 'N', lhs ? N : A, [1, 2], lhs ? B : N), 5),
                    ((LAPACK.potrs!, 'U', lhs ? N : A, lhs ? B : N), 4),
                ],
            )
        end
        for (args, oi) in cases
            ds = map(Mooncake.zero_fcodual, deepcopy(args))
            _, pb = Mooncake.rrule!!(ds...)
            fill!(Mooncake.tangent(ds[oi]), 0)
            pb(NoRData())
            @testset "$(first(args))" begin
                @test all(
                    i ->
                        !(args[i] isa AbstractArray{P}) ||
                        all(iszero, Mooncake.tangent(ds[i])),
                    eachindex(args),
                )
            end
        end
    end
    @testset "zero seeds for symmetric determinant rules" for P in (Float32, Float64),
        bad in (NaN, Inf),
        f in (logdet, det, logabsdet)

        S = Symmetric(isnan(bad) ? P[0 bad; bad 1] : P[1 bad; bad bad])
        ds = map(Mooncake.zero_fcodual, (f, S))
        _, pb = Mooncake.rrule!!(ds...)
        pb(f === logabsdet ? (zero(P), zero(P)) : zero(P))
        @test all(iszero, Mooncake.arrayify(ds[2])[2])
    end
    @testset "zero complex LU seed" for P in (ComplexF32, ComplexF64), bad in (NaN, Inf)
        ds = map(
            Mooncake.zero_fcodual,
            (Core.kwcall, (; check=false), LAPACK.getrf!, fill(P(bad), 2, 2)),
        )
        _, pb = Mooncake.rrule!!(ds...)
        pb(NoRData())
        @test all(iszero, Mooncake.tangent(ds[4]))
    end

    # Registry correctness checks seed aliased arguments independently.
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
