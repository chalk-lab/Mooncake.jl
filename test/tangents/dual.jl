@testset "Dual" begin
    @test Dual(5.0, 4.0) isa Dual{Float64,Float64}
    @test Dual(Float64, NoTangent()) isa Dual{Type{Float64},NoTangent}
    @test zero_dual(5.0) == Dual(5.0, 0.0)

    # Bare-arg zero_dual must zero NDual partials elementwise; macro-generated
    # frules for `Array{T,D}(undef, n)` rely on this for canonical bare shape.
    let nd = Mooncake.NDual{Float64,2}(1.5, (3.0, 4.0)),
        nd_zero = Mooncake.NDual{Float64,2}(1.5, (0.0, 0.0))

        @test zero_dual(nd) === nd_zero
        @test zero_dual(Complex(nd, nd)) === Complex(nd_zero, nd_zero)
        @test zero_dual([nd, nd]) == [nd_zero, nd_zero]
        @test zero_dual([Complex(nd, nd)]) == [Complex(nd_zero, nd_zero)]
    end
    @static if VERSION >= v"1.11-rc4"
        nd = Mooncake.NDual{Float64,2}(1.5, (3.0, 4.0))
        nd_zero = Mooncake.NDual{Float64,2}(1.5, (0.0, 0.0))
        m = fill!(Memory{Mooncake.NDual{Float64,2}}(undef, 2), nd)
        mz = zero_dual(m)
        @test mz isa Memory{Mooncake.NDual{Float64,2}} && all(==(nd_zero), mz)
        mrz = zero_dual(Core.memoryrefnew(m))
        @test mrz isa MemoryRef{Mooncake.NDual{Float64,2}}
        @test Core.memoryrefget(mrz, :not_atomic, false) === nd_zero
    end

    @test Mooncake._uninit_dual(Val(2), Array{Float64,1}) ===
        Dual(Array{Mooncake.NDual{Float64,2},1}, NoTangent())
    @test Mooncake._uninit_dual(Val(2), Array{ComplexF64,1}) ===
        Dual(Array{Complex{Mooncake.NDual{Float64,2}},1}, NoTangent())

    @testset "$P" for (P, D) in Any[
        (Float64, Dual{Float64,Float64}),
        (Int, Dual{Int,NoTangent}),
        (Real, Dual),
        (Any, Dual),
        (Type{UnitRange{Int}}, Dual{Type{UnitRange{Int}},NoTangent}),
        (Type{Tuple{T}} where {T}, Dual),
        (Union{Float64,Int}, Union{Dual{Float64,Float64},Dual{Int,NoTangent}}),
        (UnionAll, Dual),
        (DataType, Dual),
        (Union{}, Union{}),

        # Tuples:
        (Tuple{Float64}, Dual{Tuple{Float64},Tuple{Float64}}),
        (Tuple{Float64,Float32}, Dual{Tuple{Float64,Float32},Tuple{Float64,Float32}}),
        (
            Tuple{Int,Float64,Float32},
            Dual{Tuple{Int,Float64,Float32},Tuple{NoTangent,Float64,Float32}},
        ),

        # Small-Union Tuples
        (
            Tuple{Union{Float32,Float64}},
            Union{Dual{Tuple{Float32},Tuple{Float32}},Dual{Tuple{Float64},Tuple{Float64}}},
        ),
        (
            Tuple{Nothing,Union{Int,Float64}},
            Union{
                Dual{Tuple{Nothing,Int},NoTangent},
                Dual{Tuple{Nothing,Float64},Tuple{NoTangent,Float64}},
            },
        ),

        # General Abstract Tuples
        (Tuple{Any}, Dual),

        # Abstract Vararg / NTuple UnionAll tuples (bounded and unbounded)
        (NTuple{N,Int} where {N}, Dual),
        (Tuple{Vararg{Float64,N}} where {N}, Dual),
        (Tuple{Vararg{Float64}}, Dual),
    ]
        @test TestUtils.check_allocs(dual_type, P) == D
    end
end
