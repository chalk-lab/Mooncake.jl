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

    # Check that zero_tangent and randn_tangent yield consistent results.
    @testset "$f" for f in [zero_tangent, Base.Fix1(randn_tangent, Xoshiro(123))]
        arr = randn(2)
        p = [arr, arr.ref.mem]
        @test pointer(p[1].ref.mem) === pointer(p[2])
        t = f(p)
        @test pointer(t[1].ref.mem) === pointer(t[2])
    end

    @testset "zero_derivative container dispatch" begin
        f1 = Dual(identity, NoTangent())
        x1 = Memory{Dual{Float64,Float64}}(undef, 2)
        x1[1] = Dual(1.0, 1.0)
        x1[2] = Dual(2.0, 2.0)
        y1 = Mooncake.zero_derivative(f1, x1)
        @test primal(y1) == x1
        @test tangent(y1) == tangent(zero_dual(x1))

        f2 = Dual((x, y) -> x, NoTangent())
        x2 = Memory{Dual{Float64,Float64}}(undef, 1)
        x2[1] = Dual(1.0, 1.0)
        y2 = Memory{Dual{Float64,Float64}}(undef, 2)
        y2[1] = Dual(2.0, 2.0)
        y2[2] = Dual(3.0, 3.0)
        out2 = Mooncake.zero_derivative(f2, x2, y2)
        @test primal(out2) == x2
        @test tangent(out2) == tangent(zero_dual(x2))
    end
end
