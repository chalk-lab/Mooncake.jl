using Test

# Regression test for build_fdata world age issue (#606, #608)
# Tests that @generated functions can see custom tangent_type definitions for recursive types

@testset "build_fdata world age regression test (#606)" begin
    # Define the recursive type A from the documentation
    mutable struct TestRecursiveA{TA}
        x::TA
        a::Union{TestRecursiveA{TA},Nothing}

        TestRecursiveA(x::TA) where {TA} = new{TA}(x, nothing)
        TestRecursiveA(x::TA, child::TestRecursiveA{TA}) where {TA} = new{TA}(x, child)
    end

    # Define custom tangent type for TestRecursiveA
    mutable struct TangentForTestRecursiveA{Tx}
        x::Tx
        a::Union{TangentForTestRecursiveA{Tx},Mooncake.NoTangent}

        function TangentForTestRecursiveA{Tx}(x_tangent::Tx) where {Tx}
            return new{Tx}(x_tangent, Mooncake.NoTangent())
        end

        function TangentForTestRecursiveA{Tx}(
            x_tangent::Tx, a_tangent::Union{TangentForTestRecursiveA{Tx},Mooncake.NoTangent}
        ) where {Tx}
            return new{Tx}(x_tangent, a_tangent)
        end

        # This constructor is required by Mooncake's internal machinery
        function TangentForTestRecursiveA{Tx}(
            nt::@NamedTuple{
                x::Tx, a::Union{Mooncake.NoTangent,TangentForTestRecursiveA{Tx}}
            }
        ) where {Tx}
            return new{Tx}(nt.x, nt.a)
        end
    end

    # Register the custom tangent type
    function Mooncake.tangent_type(::Type{TestRecursiveA{T}}) where {T}
        Tx = Mooncake.tangent_type(T)
        return Tx == Mooncake.NoTangent ? Mooncake.NoTangent : TangentForTestRecursiveA{Tx}
    end

    # Define a wrapper type that would trigger the world age issue
    struct TestWrapper{TW}
        x::TW
    end

    @testset "Wrapper with recursive type - world age issue" begin
        # Test that tangent_type works for the wrapper
        T_wrapper = Mooncake.tangent_type(TestWrapper{TestRecursiveA{Float32}})
        @test T_wrapper ==
            Mooncake.Tangent{@NamedTuple{x::TangentForTestRecursiveA{Float32}}}

        # Test that we can construct the tangent type directly
        tangent_instance = T_wrapper((x=TangentForTestRecursiveA{Float32}(0.0f0),))
        @test tangent_instance isa T_wrapper

        # Test build_fdata with recursive types
        # This would throw StackOverflowError before PR #606 because the @generated
        # function couldn't see our custom tangent_type definition
        a = TestRecursiveA(1.0f0)
        wrapper = TestWrapper(a)
        a_tangent = TangentForTestRecursiveA{Float32}(0.0f0)
        fdata_tuple = (a_tangent,)

        result = Mooncake.build_fdata(
            TestWrapper{TestRecursiveA{Float32}}, (a,), fdata_tuple
        )
        @test result isa Mooncake.FData{@NamedTuple{x::TangentForTestRecursiveA{Float32}}}
    end

    @testset "Complex nested scenario" begin
        # Even more complex: wrapper of wrapper of recursive type
        struct OuterWrapper{T}
            inner::T
        end

        T_outer = Mooncake.tangent_type(OuterWrapper{TestWrapper{TestRecursiveA{Float64}}})
        @test T_outer isa Type

        # Create instances
        a = TestRecursiveA(2.0)
        a.a = TestRecursiveA(3.0)  # Make it actually recursive
        wrapper = TestWrapper(a)
        outer = OuterWrapper(wrapper)

        # This tests that the fix works even in nested scenarios
        T_inner_wrapper = Mooncake.tangent_type(TestWrapper{TestRecursiveA{Float64}})
        a_tangent = TangentForTestRecursiveA{Float64}(0.0)
        a_tangent.a = TangentForTestRecursiveA{Float64}(0.0)
        inner_tangent = T_inner_wrapper((x=a_tangent,))
        outer_tangent = T_outer((inner=inner_tangent,))
        @test outer_tangent isa T_outer

        # Test build_fdata in nested case
        # We need to pass fdata, not tangent
        inner_fdata = Mooncake.fdata(inner_tangent)
        result = Mooncake.build_fdata(
            OuterWrapper{TestWrapper{TestRecursiveA{Float64}}}, (wrapper,), (inner_fdata,)
        )
        @test result isa Mooncake.FData
    end
end
