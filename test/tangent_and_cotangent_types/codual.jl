@testset "codual" begin
    @test CoDual(5.0, 4.0) isa CoDual{Float64,Float64}
    @test CoDual(Float64, NoTangent()) isa CoDual{Type{Float64},NoTangent}
    @test zero_codual(5.0) == CoDual(5.0, 0.0)

    @testset "$P" for (P, D, F) in Any[
        (Float64, CoDual{Float64,Float64}, CoDual{Float64,NoFData}),
        (Int, CoDual{Int,NoTangent}, CoDual{Int,NoFData}),
        (Real, CoDual, CoDual),
        (Any, CoDual, CoDual),
        (
            Type{UnitRange{Int}},
            CoDual{Type{UnitRange{Int}},NoTangent},
            CoDual{Type{UnitRange{Int}},NoFData},
        ),
        (Type{Tuple{T}} where {T}, CoDual, CoDual),
        (
            Union{Float64,Int},
            Union{CoDual{Float64,Float64},CoDual{Int,NoTangent}},
            Union{CoDual{Float64,NoFData},CoDual{Int,NoFData}},
        ),
        (UnionAll, CoDual, CoDual),

        # Tuples:
        # Concrete tuples:
        (
            Tuple{Float64},
            CoDual{Tuple{Float64},Tuple{Float64}},
            CoDual{Tuple{Float64},NoFData},
        ),
        (
            Tuple{Float64,Float32},
            CoDual{Tuple{Float64,Float32},Tuple{Float64,Float32}},
            CoDual{Tuple{Float64,Float32},NoFData},
        ),
        (
            Tuple{Int,Float64,Float32},
            CoDual{Tuple{Int,Float64,Float32},Tuple{NoTangent,Float64,Float32}},
            CoDual{Tuple{Int,Float64,Float32},NoFData},
        ),

        # Small-Union Tuples
        (
            Tuple{Union{Float32,Float64}},
            Union{
                CoDual{Tuple{Float32},Tuple{Float32}},CoDual{Tuple{Float64},Tuple{Float64}}
            },
            Union{CoDual{Tuple{Float32},NoFData},CoDual{Tuple{Float64},NoFData}},
        ),
        (
            Tuple{Nothing,Union{Int,Float64}},
            Union{
                CoDual{Tuple{Nothing,Int},NoTangent},
                CoDual{Tuple{Nothing,Float64},Tuple{NoTangent,Float64}},
            },
            Union{
                CoDual{Tuple{Nothing,Int},NoFData},CoDual{Tuple{Nothing,Float64},NoFData}
            },
        ),

        # General Abstract Tuples
        (Tuple{Any}, CoDual, CoDual),
    ]
        @test TestUtils.check_allocs(codual_type, P) == D
        @test TestUtils.check_allocs(Mooncake.fcodual_type, P) == F
    end

    @testset "NoPullback" begin
        @test Base.issingletontype(typeof(NoPullback(zero_fcodual(5.0))))
        @test NoPullback(zero_codual(5.0))(4.0) == (0.0,)
    end
end
