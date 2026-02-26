# Regression test for build_output_tangent world age issue (#893, #1008)
# Tests that @generated functions can see custom tangent_type definitions for recursive types.
# Mirrors the build_fdata world age regression test (PR #606).

@testset "build_output_tangent world age regression test (#893, #1008)" begin
    # Define a recursive type (mirrors the Ark.jl _GraphNode/_VecMap pattern from #893)
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

        # This constructor is required by Mooncake's internal machinery
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
        return Tx == Mooncake.NoTangent ? Mooncake.NoTangent : TangentForTestRecursiveB{Tx}
    end

    # Define a wrapper type that would trigger the world age issue
    struct TestWrapperB{TW}
        x::TW
    end

    @testset "Wrapper with recursive type - world age issue" begin
        # Test that tangent_type works for the wrapper
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

    @testset "Complex nested scenario" begin
        # Even more complex: wrapper of wrapper of recursive type
        struct OuterWrapperB{T}
            inner::T
        end

        T_outer = Mooncake.tangent_type(OuterWrapperB{TestWrapperB{TestRecursiveB{Float64}}})
        @test T_outer isa Type

        # Create instances with actual recursion
        b = TestRecursiveB(2.0)
        b.b = TestRecursiveB(3.0)  # Make it actually recursive
        wrapper = TestWrapperB(b)
        outer = OuterWrapperB(wrapper)

        # Test build_output_tangent in nested case
        T_inner_wrapper = Mooncake.tangent_type(TestWrapperB{TestRecursiveB{Float64}})
        b_tangent = TangentForTestRecursiveB{Float64}(0.0)
        b_tangent.b = TangentForTestRecursiveB{Float64}(0.0)

        result = Mooncake.build_output_tangent(
            OuterWrapperB{TestWrapperB{TestRecursiveB{Float64}}},
            (wrapper,),
            (T_inner_wrapper((x=b_tangent,)),),
        )
        @test result isa T_outer
    end
end
