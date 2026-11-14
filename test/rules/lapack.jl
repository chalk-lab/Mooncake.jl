@testset "lapack" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:lapack))
end

@testset "zero LAPACK cotangents" for P in (Float32, Float64)
    N, B = fill(P(NaN), 2, 2), ones(P, 2, 2)
    cases = Any[
        (Core.kwcall, (; check=false), LAPACK.getrf!, N),
        (LAPACK.getri!, N, [1, 2]),
        (LAPACK.potrf!, 'U', N),
        (LAPACK.trtrs!, 'U', 'N', 'N', N, B),
        (LAPACK.getrs!, 'N', N, [1, 2], B),
        (LAPACK.potrs!, 'U', N, B),
    ]
    @testset "$(first(args))" for args in cases
        ds = map(Mooncake.zero_fcodual, deepcopy(args))
        _, pb = rrule!!(ds...)
        pb(NoRData())
        @test all(
            i -> !(args[i] isa AbstractArray{P}) || all(iszero, Mooncake.tangent(ds[i])),
            eachindex(args),
        )
    end
    for f in (logdet, det, logabsdet)
        S = Symmetric(P[0 NaN; NaN 1])
        ds = map(Mooncake.zero_fcodual, (f, S))
        _, pb = rrule!!(ds...)
        pb(f === logabsdet ? (zero(P), zero(P)) : zero(P))
        @test all(iszero, Mooncake.arrayify(ds[2])[2])
    end
end
