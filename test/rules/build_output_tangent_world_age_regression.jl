
# Regression test for build_output_tangent world age issue (#893, #1008)
# Tests that @generated functions can see custom tangent_type definitions.
# Mirrors the build_fdata world age regression test from PR #606.

@testset "build_output_tangent world age regression test (#893, #1008)" begin
    @testset "Custom tangent type (#1008)" begin
        # Test that build_output_tangent computes tangent_type at runtime
        struct BOTTestStruct
            x::Float64
        end

        struct BOTTestTangent
            val::Float64
        end
        BOTTestTangent(nt::NamedTuple{(:x,),Tuple{Float64}}) = BOTTestTangent(nt.x)

        function Mooncake.tangent_type(::Type{BOTTestStruct})
            return BOTTestTangent
        end

        result = Mooncake.build_output_tangent(BOTTestStruct, (1.5,), (0.5,))
        @test result isa BOTTestTangent
        @test result.val == 0.5
    end

    @testset "Wrapper with recursive type - world age issue (#893)" begin
        # Define a recursive type (mirrors the Ark.jl _GraphNode/_VecMap pattern)
        mutable struct TestRecursiveB{TB}
            x::TB
            b::Union{TestRecursiveB{TB},Nothing}

            TestRecursiveB(x::TB) where {TB} = new{TB}(x, nothing)
            TestRecursiveB(x::TB, child::TestRecursiveB{TB}) where {TB} = new{TB}(x, child)
        end

        # Define custom tangent type for TestRecursiveB
        mutable struct TangentForTestRecursiveB{Tx}
            x::Tx
            b::Union{TangentForTestRecursiveB{Tx},Mooncake.NoTangent}

            function TangentForTestRecursiveB{Tx}(x_tangent::Tx) where {Tx}
                return new{Tx}(x_tangent, Mooncake.NoTangent())
            end

            function TangentForTestRecursiveB{Tx}(
                x_tangent::Tx,
                b_tangent::Union{TangentForTestRecursiveB{Tx},Mooncake.NoTangent},
            ) where {Tx}
                return new{Tx}(x_tangent, b_tangent)
            end

            function TangentForTestRecursiveB{Tx}(
                nt::@NamedTuple{
                    x::Tx, b::Union{Mooncake.NoTangent,TangentForTestRecursiveB{Tx}}
                },
            ) where {Tx}
                return new{Tx}(nt.x, nt.b)
            end
        end

        # Register the custom tangent type
        function Mooncake.tangent_type(::Type{TestRecursiveB{T}}) where {T}
            Tx = Mooncake.tangent_type(T)
            return if Tx == Mooncake.NoTangent
                Mooncake.NoTangent
            else
                TangentForTestRecursiveB{Tx}
            end
        end

        # Wrapper type that would trigger the world age issue in build_output_tangent
        struct TestWrapperB{TW}
            x::TW
        end

        # Verify tangent_type works
        T_wrapper = Mooncake.tangent_type(TestWrapperB{TestRecursiveB{Float32}})
        @test T_wrapper ==
            Mooncake.Tangent{@NamedTuple{x::TangentForTestRecursiveB{Float32}}}

        # Test build_output_tangent with recursive types.
        # This would throw StackOverflowError before the fix because the @generated
        # function couldn't see our custom tangent_type definition.
        b = TestRecursiveB(1.0f0)
        wrapper = TestWrapperB(b)
        b_tangent = TangentForTestRecursiveB{Float32}(0.0f0)

        result = Mooncake.build_output_tangent(
            TestWrapperB{TestRecursiveB{Float32}}, (b,), (b_tangent,)
        )
        @test result isa T_wrapper
    end
end
