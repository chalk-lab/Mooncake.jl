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

    # Pointer access that would lose or misplace tangents is an error.
    @testset "unsupported pointer access" begin
        # Storing a differentiable value into memory with no tangent storage.
        function store_load(x)
            b = zeros(UInt8, 8)
            GC.@preserve b begin
                p = Ptr{Float64}(pointer(b))
                unsafe_store!(p, x)
                return unsafe_load(p)
            end
        end
        # Tangent memory laid out differently from the primal memory, with a different
        # element size and with the same element size.
        function mixed_layout(x)
            v = [(1, x), (2, 3x)]
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v, 1)) + 8)
        end
        function offset_layout(x)
            v = [(Int32(1), Float32(x), 2.0)]
            return GC.@preserve v Float64(unsafe_load(Ptr{Float32}(pointer(v)) + 4))
        end
        function mixed_copy(x)
            v = [(1, x), (2, 3x)]
            w = similar(v)
            GC.@preserve v w unsafe_copyto!(pointer(w), pointer(v), 2)
            return w[2][2]
        end
        function mixed_wrap(x)
            v = [(1, x), (2, 3x)]
            return GC.@preserve v unsafe_wrap(Array, pointer(v), 2)[2][2]
        end
        # Copying or wrapping through a pointer into memory with no tangent storage.
        function bytes_copy(x)
            b = zeros(UInt8, 8)
            w = [x]
            GC.@preserve b w unsafe_copyto!(pointer(w), Ptr{Float64}(pointer(b)), 1)
            return w[1]
        end
        function bytes_wrap(x)
            b = zeros(UInt8, 8)
            return GC.@preserve b x * unsafe_wrap(Array, Ptr{Float64}(pointer(b)), 1)[1]
        end
        @testset "$f" for f in [
            store_load,
            mixed_layout,
            offset_layout,
            mixed_copy,
            mixed_wrap,
            bytes_copy,
            bytes_wrap,
        ]
            @test_throws ArgumentError begin
                cache = Mooncake.prepare_gradient_cache(f, 1.0)
                Mooncake.value_and_gradient!!(cache, f, 1.0)
            end
            @test_throws ArgumentError begin
                cache = Mooncake.prepare_derivative_cache(f, 1.0)
                Mooncake.value_and_derivative!!(cache, (f, NoTangent()), (1.0, 1.0))
            end
        end
    end
end
