using Mooncake: Dual, NoTangent, zero_dual, dual_type
using Mooncake: TestUtils
using Random: Xoshiro

@testset "Dual" begin
    rng = Xoshiro(123)
    @test Dual(5.0, 4.0) isa Dual{Float64,Mooncake.NTangent{Tuple{Float64}}}
    @test Dual(Float64, NoTangent()) isa Dual{Type{Float64},NoTangent}
    @test zero_dual(5.0) == Dual{Float64,Mooncake.NTangent{Tuple{Float64}}}(5.0, 0.0)

    @testset "$P" for (P, D) in Any[
        (Float64, Dual{Float64,Mooncake.NTangent{Tuple{Float64}}}),
        (Int, Dual{Int,NoTangent}),
        (Real, Dual),
        (Any, Dual),
        (Type{UnitRange{Int}}, Dual{Type{UnitRange{Int}},NoTangent}),
        (Type{Tuple{T}} where {T}, Dual),
        (
            Union{Float64,Int},
            Union{Dual{Float64,Mooncake.NTangent{Tuple{Float64}}},Dual{Int,NoTangent}},
        ),
        (UnionAll, Dual),
        (DataType, Dual),
        (Union{}, Union{}),

        # Tuples:
        (Tuple{Float64}, Dual{Tuple{Float64},Mooncake.NTangent{Tuple{Tuple{Float64}}}}),
        (
            Tuple{Float64,Float32},
            Dual{Tuple{Float64,Float32},Mooncake.NTangent{Tuple{Tuple{Float64,Float32}}}},
        ),
        (
            Tuple{Int,Float64,Float32},
            Dual{
                Tuple{Int,Float64,Float32},
                Mooncake.NTangent{Tuple{Tuple{NoTangent,Float64,Float32}}},
            },
        ),

        # Small-Union Tuples
        (
            Tuple{Union{Float32,Float64}},
            Union{
                Dual{Tuple{Float32},Mooncake.NTangent{Tuple{Tuple{Float32}}}},
                Dual{Tuple{Float64},Mooncake.NTangent{Tuple{Tuple{Float64}}}},
            },
        ),
        (
            Tuple{Nothing,Union{Int,Float64}},
            Union{
                Dual{Tuple{Nothing,Int},NoTangent},
                Dual{
                    Tuple{Nothing,Float64},
                    Mooncake.NTangent{Tuple{Tuple{NoTangent,Float64}}},
                },
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

    @test TestUtils.check_allocs(Mooncake.tangent_type, Val(2), Float64) ==
        Mooncake.NTangent{Tuple{Float64,Float64}}
    @test TestUtils.check_allocs(Mooncake.tangent_type, Val(2), Tuple{Float64}) ==
        Mooncake.NTangent{Tuple{Tuple{Float64},Tuple{Float64}}}
    @test TestUtils.check_allocs(Mooncake.dual_type, Val(2), Float64) ==
        Mooncake.Nfwd.NDual{Float64,2}
    @test TestUtils.check_allocs(Mooncake.dual_type, Val(2), ComplexF64) ==
        Complex{Mooncake.Nfwd.NDual{Float64,2}}
    @test TestUtils.check_allocs(Mooncake.dual_type, Val(2), Tuple{Float64}) ==
        Dual{Tuple{Float64},Mooncake.NTangent{Tuple{Tuple{Float64},Tuple{Float64}}}}
    @test TestUtils.check_allocs(Mooncake.dual_type, Val(3), Type{Float64}) ==
        Dual{Type{Float64},NoTangent}

    @testset "test_dual" begin
        for (p, D, width) in (
            (2.0, Dual{Float64,Mooncake.NTangent{Tuple{Float64}}}, 1),
            (2.0, Mooncake.Nfwd.NDual{Float64,2}, 2),
            (1.0 + 2.0im, Complex{Mooncake.Nfwd.NDual{Float64,2}}, 2),
        )
            TestUtils.test_dual(rng, p, D; width)
        end
    end

    @test_throws ArgumentError Mooncake._canonical_forward_tangent(
        (1.0, 2.0), (Mooncake.NTangent((1.0,)), Mooncake.NTangent((2.0, 3.0)))
    )

    err = try
        Dual(5.0, (1.0, 2.0))
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("NTangent(...)", sprint(showerror, err))

    err = try
        Dual([1.0, 2.0], [1.0 0.0; 0.0 1.0])
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("NTangent(...)", sprint(showerror, err))
end
