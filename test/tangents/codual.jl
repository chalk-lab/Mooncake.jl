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
        (DataType, CoDual, CoDual),
        (Union{}, Union{}, Union{}),

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

        # Abstract Vararg / NTuple UnionAll tuples (bounded and unbounded)
        (NTuple{N,Int} where {N}, CoDual, CoDual),
        (Tuple{Vararg{Float64,N}} where {N}, CoDual, CoDual),
        (Tuple{Vararg{Float64}}, CoDual, CoDual),
    ]
        @test TestUtils.check_allocs(codual_type, P) == D
        @test TestUtils.check_allocs(Mooncake.fcodual_type, P) == F
    end

    @testset "(f)codual_type with phantom TypeVar (#1191)" begin
        # `UnionAll(A, AbstractArray{T, A})` normalises to a DataType whose body
        # still references the free `T`; the generic `(f)codual_type(::Type{P})`
        # binder can't bind `P` for such shapes and throws `UndefVarError`. The
        # Tuple variant blocks a future fix that only special-cases AbstractArray.
        phantom = UnionAll(TypeVar(:A), AbstractArray{TypeVar(:T),TypeVar(:A)})
        phantom_tuple = UnionAll(TypeVar(:A), Tuple{TypeVar(:T),TypeVar(:A)})
        @test codual_type(phantom) === CoDual
        @test Mooncake.fcodual_type(phantom) === CoDual
        @test codual_type(phantom_tuple) === CoDual
        @test Mooncake.fcodual_type(phantom_tuple) === CoDual
    end

    @testset "NoPullback" begin
        @test Base.issingletontype(typeof(NoPullback(zero_fcodual(5.0))))
        @test NoPullback(zero_codual(5.0))(4.0) == (0.0,)
    end

    @testset "zero_codual and zero_fcodual for Ptr" begin
        # zero_tangent(::Ptr) throws, so zero_codual/zero_fcodual must not call it.
        # They fall back to uninit_codual/uninit_fcodual (bitcast convention).
        p = Ptr{Float64}()
        @test Mooncake.zero_codual(p) == Mooncake.uninit_codual(p)
        @test Mooncake.zero_fcodual(p) == Mooncake.uninit_fcodual(p)
    end
end
