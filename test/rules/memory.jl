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

    # Tangents can't be loaded or stored through a pointer into memory without a matching
    # tangent: with no tangent storage (bytes) or a different layout (`Tuple{Int,Float64}`).
    @testset "pointer access without a matching tangent" begin
        function store_load(x)
            b = zeros(UInt8, 8)
            p = Ptr{Float64}(pointer(b))
            return GC.@preserve b (unsafe_store!(p, x); unsafe_load(p))
        end
        bytes_load(x) = x * reinterpret(Float64, UInt8[0, 0, 0, 0, 0, 0, 0, 64])[1]
        function mixed_load(x)
            v = [(1, x)]
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v)) + 8)
        end
        function bytes_copy(x)
            b = zeros(UInt8, 8)
            w = [x]
            GC.@preserve b w unsafe_copyto!(pointer(w), Ptr{Float64}(pointer(b)), 1)
            return w[1]
        end
        @testset "$f" for f in [store_load, bytes_load, mixed_load, bytes_copy]
            TestUtils.test_rule_throws(StableRNG(123), f, 1.0; err=ArgumentError)
        end
    end
end
