@testset "pointer" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:pointer))

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
        function int_overwrite(x)
            v = [x]
            GC.@preserve v unsafe_store!(Ptr{Int}(pointer(v)), reinterpret(Int, 2.0))
            return v[1]
        end
        function union_load(x)
            v = Union{Int,Float64}[x]
            v[1] = reinterpret(Int, 2.0)
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v)))
        end
        function missing_load(x)
            v = Union{Missing,Float64}[x]
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v)))
        end
        function bytes_dot(x)
            b = zeros(UInt8, 8)
            v = [x]
            return GC.@preserve b v BLAS.dot(1, Ptr{Float64}(pointer(b)), 1, pointer(v), 1)
        end
        fs = [
            store_load,
            bytes_load,
            bytes_copy,
            bytes_dot,
            mixed_load,
            offset_load,
            int_overwrite,
            union_load,
            missing_load,
        ]
        @testset "$f" for f in fs
            TestUtils.test_rule_throws(StableRNG(123), f, 1.0; err=ArgumentError)
        end
        function objptr_null_diff(x)
            v = [("a", [x])]
            q = GC.@preserve v unsafe_load(
                Ptr{Ptr{Cvoid}}(pointer(v)) + fieldoffset(eltype(v), 2)
            )
            return (unsafe_pointer_to_objref(q)::Vector{Float64})[1]
        end
        TestUtils.test_rule_throws(
            StableRNG(123),
            objptr_null_diff,
            1.0;
            err=ArgumentError,
            msg="Mooncake cannot differentiate pointer operations without matching tangent storage, " *
                "e.g. a `Ptr{Float64}` into a `Vector{UInt8}` or a `Vector{Tuple{Int,Float64}}`, " *
                "or casting a pointer into differentiable memory to `Ptr{Int}`.",
        )
    end
end
