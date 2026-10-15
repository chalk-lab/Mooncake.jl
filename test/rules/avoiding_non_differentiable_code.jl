@testset "avoiding_non_differentiable_code" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:avoiding_non_differentiable_code))

    # Tangents can't be loaded or stored through a pointer into memory with no tangent
    # storage, e.g. that of a `Vector{UInt8}`, or whose tangent is laid out differently, e.g.
    # that of a `Vector{Tuple{Int,Float64}}`.
    @testset "pointer access without a matching tangent" begin
        function store_load(x)
            b = zeros(UInt8, 8)
            p = Ptr{Float64}(pointer(b))
            return GC.@preserve b (unsafe_store!(p, x); unsafe_load(p))
        end
        bytes_load(x) = x * reinterpret(Float64, UInt8[0, 0, 0, 0, 0, 0, 0, 64])[1]
        function bytes_copy(x)
            b = zeros(UInt8, 8)
            w = [x]
            GC.@preserve b w unsafe_copyto!(pointer(w), Ptr{Float64}(pointer(b)), 1)
            return w[1]
        end
        function mixed_load(x)
            v = [(1, x), (2, 3x)]
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v, 2)) + 8)
        end
        function offset_load(x)
            v = [(Int32(1), Float32(x), 2.0)]
            return GC.@preserve v Float64(unsafe_load(Ptr{Float32}(pointer(v)) + 4))
        end
        fs = Any[store_load, bytes_copy, mixed_load, offset_load]
        # Before Julia 1.11, `reinterpret` does not load through a pointer, so this works.
        VERSION >= v"1.11.0-rc4" && push!(fs, bytes_load)
        @testset "$f" for f in fs
            TestUtils.test_rule_throws(StableRNG(123), f, 1.0; err=ArgumentError)
        end
    end
end
