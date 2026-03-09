@testset "Dual" begin
    @test Dual(5.0, 4.0) isa Dual{Float64,Float64}
    @test Dual(Float64, NoTangent()) isa Dual{Type{Float64},NoTangent}
    @test zero_dual(5.0) == Dual(5.0, 0.0)

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

        # Abstract Vararg / NTuple UnionAll tuples
        (NTuple{N,Int} where {N}, Dual),
        (Tuple{Vararg{Float64}} where {N}, Dual),
    ]
        @test TestUtils.check_allocs(dual_type, P) == D
    end
end