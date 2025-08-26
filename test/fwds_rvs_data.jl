module FwdsRvsDataTestResources
struct Foo{A} end
struct Bar{A,B,C}
    a::A
    b::B
    c::C
end
struct SV{S<:Tuple,T,L}
    data::NTuple{L,T}
end
end

@testset "fwds_rvs_data" begin
    @testset "fdata_type / rdata_type($P)" for (P, F, R) in Any[
        (Union{}, Union{}, Union{}),
        (
            Tuple{Any,Vector{Float64}},
            Tuple{Any,Vector{Float64}},
            Union{NoRData,Tuple{Any,NoRData}},
        ),
        (Tuple{Any,Float64}, Union{NoFData,Tuple{Any,NoFData}}, Tuple{Any,Float64}),
    ]
        @test fdata_type(tangent_type(P)) == F
        @test rdata_type(tangent_type(P)) == R
    end
    @testset "$(typeof(p))" for (_, p, _...) in Mooncake.tangent_test_cases()
        TestUtils.test_tangent_splitting(Xoshiro(123456), p)
    end
    @testset "Test for unions involving `Nothing`" begin
        # https://github.com/chalk-lab/Mooncake.jl/issues/597 for the reason.
        TestUtils.test_tangent_splitting(
            Xoshiro(123456), TestResources.make_P_union_nothing(); test_opt_flag=false
        )
        # https://github.com/chalk-lab/Mooncake.jl/issues/598
        TestUtils.test_tangent_splitting(
            Xoshiro(123456), TestResources.make_P_union_array(); test_opt_flag=false
        )

        # https://github.com/chalk-lab/Mooncake.jl/issues/631
        TestUtils.test_tangent_splitting(
            Xoshiro(123456), TestResources.P_adam_like_union; test_opt_flag=false
        )
    end

    @testset "zero_rdata_from_type checks" begin
        @test can_produce_zero_rdata_from_type(Vector) == true
        check_allocs(can_produce_zero_rdata_from_type, Vector)
        @test zero_rdata_from_type(Vector) == NoRData()
        @test !can_produce_zero_rdata_from_type(FwdsRvsDataTestResources.Foo)
        @test can_produce_zero_rdata_from_type(Tuple{Float64,Type{Float64}})
        @test ==(
            zero_rdata_from_type(FwdsRvsDataTestResources.Foo),
            CannotProduceZeroRDataFromType(),
        )
        @test !can_produce_zero_rdata_from_type(Tuple)
        @test zero_rdata_from_type(Tuple) == CannotProduceZeroRDataFromType()
        @test !can_produce_zero_rdata_from_type(Union{Tuple{Float64},Tuple{Int}})
        @test ==(
            zero_rdata_from_type(Union{Tuple{Float64},Tuple{Int}}),
            CannotProduceZeroRDataFromType(),
        )
        @test !can_produce_zero_rdata_from_type(Tuple{T,T} where {T<:Integer})
        @test can_produce_zero_rdata_from_type(Type{Float64})
        @test can_produce_zero_rdata_from_type(Union{Tuple{Int},Tuple{Int,Int}})
        @test zero_rdata_from_type(Union{Tuple{Int},Tuple{Int,Int}}) == NoRData()
        @test zero_rdata_from_type(Union{Float64,Int}) == CannotProduceZeroRDataFromType()

        # Edge case: Types with unbound type parameters.
        P = (Type{T} where {T}).body
        @test Mooncake.can_produce_zero_rdata_from_type(P)
        @test Mooncake.zero_rdata_from_type(P) === NoRData()

        # Check for ambiguity.
        @test Mooncake.can_produce_zero_rdata_from_type(Union{})
        @test Mooncake.zero_rdata_from_type(Union{}) === NoRData()

        # Performance.
        @testset "$P" for P in Any[
            Vector{Vector{Vector{Vector{Float64}}}},
            Vector{Vector{Vector{NTuple{11,Float64}}}},
            FwdsRvsDataTestResources.Bar{Float64,Float64,Float64},
            FwdsRvsDataTestResources.Bar{
                FwdsRvsDataTestResources.SV{Tuple{1},Float64,1},Float64,Float64
            },
        ]
            @test TestUtils.is_foldable(can_produce_zero_rdata_from_type, Tuple{Type{P}})
        end
    end
    @testset "lazy construction checks" begin
        # Check that lazy construction is in fact lazy for some cases where performance
        # really matters -- floats, things with no rdata, etc.
        @testset "$p" for (P, p, fully_lazy) in Any[
            (Int, 5, true),
            (Int32, Int32(5), true),
            (Float64, 5.0, true),
            (Float32, 5.0f0, true),
            (Float16, Float16(5.0), true),
            (StructFoo, StructFoo(5.0), false),
            (StructFoo, StructFoo(5.0, randn(4)), false),
            (Type{Bool}, Bool, true),
            (
                Type{Mooncake.TestResources.StableFoo},
                Mooncake.TestResources.StableFoo,
                true,
            ),
            (Tuple{Float64,Float64}, (5.0, 4.0), true),
            (Tuple{Float64,Vararg{Float64}}, (5.0, 4.0, 3.0), false),
            (Type{Type{Tuple{T}} where {T}}, Type{Tuple{T}} where {T}, true),
        ]
            L = Mooncake.lazy_zero_rdata_type(P)
            @test fully_lazy == Base.issingletontype(typeof(lazy_zero_rdata(L, p)))
            if isconcretetype(P)
                @inferred Mooncake.instantiate(lazy_zero_rdata(L, p))
            end
            @test typeof(lazy_zero_rdata(L, p)) == Mooncake.lazy_zero_rdata_type(P)
            @test lazy_zero_rdata(p) isa LazyZeroRData{_typeof(p)}
        end
        @test isa(
            lazy_zero_rdata(Mooncake.TestResources.StableFoo),
            LazyZeroRData{Type{Mooncake.TestResources.StableFoo}},
        )
    end
    @testset "misc fdata / rdata type checking" begin
        @test(
            ==(
                Mooncake.rdata_type(tangent_type(Tuple{Union{Float32,Float64}})),
                Tuple{Union{Float32,Float64}},
            )
        )
        @test(==(Mooncake.rdata_type(tangent_type(Tuple{Union{Int32,Int}})), NoRData))
        @test(
            ==(
                Mooncake.rdata_type(
                    tangent_type(Tuple{Union{Vector{Float32},Vector{Float64}}})
                ),
                NoRData,
            )
        )
    end

    # Tests that the static type of an fdata / rdata is correct happen in
    # test_tangent_splitting, so here we only need to test the specific quirks for a given type.
    @testset "fdata and rdata verification" begin
        @testset "Array" begin
            @test_throws InvalidFDataException verify_fdata_value(randn(10), randn(11))
            @test_throws InvalidFDataException verify_fdata_value([randn(10)], [randn(11)])
            @test_throws InvalidFDataException verify_fdata_value(Any[1], [NoFData()])
        end
        @testset "Tuple" begin
            @test_throws InvalidFDataException verify_fdata_value((), ())
            @test_throws InvalidFDataException verify_fdata_value((5,), (NoFData(),))
            @test_throws InvalidRDataException verify_rdata_value((), ())
            @test_throws InvalidRDataException verify_rdata_value((5,), (NoRData(),))
        end
        @testset "Ptr" begin
            @test verify_fdata_value(Ptr{Float64}(), Ptr{Float64}()) === nothing
            @test verify_rdata_value(Ptr{Float64}(), NoRData()) === nothing
        end
    end

    @testset "Error messages for types passed to accessor functions" begin
        # Test that passing types to fdata throws helpful error message
        @test_throws "Float64 is a type. Perhaps you meant fdata_type(Float64) or fdata(instance_of_tangent)?" fdata(
            Float64
        )
        @test_throws "Int64 is a type. Perhaps you meant fdata_type(Int64) or fdata(instance_of_tangent)?" fdata(
            Int64
        )

        # Test that passing types to rdata throws helpful error message  
        @test_throws "Float64 is a type. Perhaps you meant rdata_type(Float64) or rdata(instance_of_tangent)?" rdata(
            Float64
        )
        @test_throws "Int64 is a type. Perhaps you meant rdata_type(Int64) or rdata(instance_of_tangent)?" rdata(
            Int64
        )

        # Test with more complex types - using exact string matching since the message is deterministic
        @test_throws "Vector{Float64} is a type. Perhaps you meant fdata_type(Vector{Float64}) or fdata(instance_of_tangent)?" fdata(
            Vector{Float64}
        )
        @test_throws "Vector{Float64} is a type. Perhaps you meant rdata_type(Vector{Float64}) or rdata(instance_of_tangent)?" rdata(
            Vector{Float64}
        )

        # Verify that existing functionality with instances still works
        x = 5.0
        t = zero_tangent(x)
        @test fdata(t) isa NoFData
        @test rdata(t) isa Float64

        arr = [1.0, 2.0]
        t_arr = zero_tangent(arr)
        @test fdata(t_arr) isa Vector{Float64}
        @test rdata(t_arr) isa NoRData
    end
end
