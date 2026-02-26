@testset "new" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:new))
    include("build_fdata_world_age_regression.jl")

    @testset "build_output_tangent runtime tangent_type (#1008)" begin
        # Test that build_output_tangent computes tangent_type at runtime
        struct MyStruct
            x::Float64
        end

        # Custom tangent type with NamedTuple constructor for build_output_tangent
        struct MyTangent
            val::Float64
        end
        MyTangent(nt::NamedTuple{(:x,),Tuple{Float64}}) = MyTangent(nt.x)

        function Mooncake.tangent_type(::Type{MyStruct})
            return MyTangent
        end

        result = Mooncake.build_output_tangent(MyStruct, (1.5,), (0.5,))
        @test result isa MyTangent
        @test result.val == 0.5
    end
end
