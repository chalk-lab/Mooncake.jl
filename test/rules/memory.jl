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
        # On 1.11 a capacity-hinted reshape gets a `Memory` sliced from its owner's.
        a = sizehint!(randn(2), 100)
        t = f((a, reshape(a, 2, 1).ref.mem))
        @test pointer(t[1]) === pointer(t[2])
        @static if VERSION < v"1.12-"
            m = reshape(push!(Union{String,Float64}[1.5, "a"], 2.5), 1, 3)
            msg = "Julia 1.11 cannot share derivative storage"
            TestUtils._test_throws(() -> f(m), ArgumentError, msg)
        end
    end
end
