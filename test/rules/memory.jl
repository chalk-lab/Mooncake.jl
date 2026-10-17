using Mooncake: generate_data_test_cases

function generate_mem()
    return rrule!!(zero_fcodual(Memory{Float64}), zero_fcodual(undef), zero_fcodual(10))
end

@testset "memory" begin
    @testset "$(typeof(p))" for p in generate_data_test_cases(StableRNG, Val(:memory))
        TestUtils.test_data(sr(123), p)
    end
    TestUtils.run_rule_test_cases(StableRNG, Val(:memory))

    # Check that the rule for `Memory{P}` only produces two allocations.
    generate_mem()
    @test TestUtils.count_allocs(generate_mem) <= 2

    @static if VERSION >= v"1.12-"
        y, _ = rrule!!(
            zero_fcodual(Core.memorynew), zero_fcodual(Memory{Float64}), zero_fcodual(1000)
        )
        @test all(iszero, tangent(y))
    end

    # Check that zero_tangent and randn_tangent yield consistent results.
    @testset "$f" for f in [zero_tangent, Base.Fix1(randn_tangent, Xoshiro(123))]
        arr = randn(2)
        p = [arr, arr.ref.mem]
        @test pointer(p[1].ref.mem) === pointer(p[2])
        t = f(p)
        @test pointer(t[1].ref.mem) === pointer(t[2])
    end

    @testset "to_cr_tangent" begin
        @test Mooncake.to_cr_tangent(Memory{Float64}([1.0, 2.0])) == [1.0, 2.0]
        @test Mooncake.to_cr_tangent(zero_tangent(Memory{Int}(undef, 2))) ==
            [ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()]
        t = Memory{Vector{Float64}}(undef, 2)
        t[2] = [3.0]
        @test Mooncake.to_cr_tangent(t) == [ChainRulesCore.ZeroTangent(), [3.0]]
    end
end
