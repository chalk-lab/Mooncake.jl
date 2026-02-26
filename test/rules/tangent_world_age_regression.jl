# Regression tests for tangent world age issues.
# Tests that @generated functions can see custom tangent_type definitions for recursive types.

@testset "tangent world age regression" begin
    # Recursive type with custom tangent (mirrors patterns from #606, #893, #1008)
    mutable struct TestRecursive{T}
        x::T
        child::Union{TestRecursive{T},Nothing}
        TestRecursive(x::T) where {T} = new{T}(x, nothing)
    end

    mutable struct TestRecursiveTangent{T}
        x::T
        child::Union{TestRecursiveTangent{T},Mooncake.NoTangent}
        TestRecursiveTangent{T}(x::T) where {T} = new{T}(x, Mooncake.NoTangent())
        function TestRecursiveTangent{T}(
            nt::@NamedTuple{x::T, child::Union{Mooncake.NoTangent,TestRecursiveTangent{T}}}
        ) where {T}
            return new{T}(nt.x, nt.child)
        end
    end

    function Mooncake.tangent_type(::Type{TestRecursive{T}}) where {T}
        Tx = Mooncake.tangent_type(T)
        return Tx == Mooncake.NoTangent ? Mooncake.NoTangent : TestRecursiveTangent{Tx}
    end

    struct TestWrapper{T}
        x::T
    end

    @testset "build_fdata (#606)" begin
        T_wrapper = Mooncake.tangent_type(TestWrapper{TestRecursive{Float32}})
        @test T_wrapper == Mooncake.Tangent{@NamedTuple{x::TestRecursiveTangent{Float32}}}
        @test T_wrapper((x=TestRecursiveTangent{Float32}(0.0f0),)) isa T_wrapper

        result = Mooncake.build_fdata(
            TestWrapper{TestRecursive{Float32}},
            (TestRecursive(1.0f0),),
            (TestRecursiveTangent{Float32}(0.0f0),),
        )
        @test result isa Mooncake.FData{@NamedTuple{x::TestRecursiveTangent{Float32}}}
    end

    @testset "build_fdata nested" begin
        struct OuterWrapper{T}
            inner::T
        end

        a = TestRecursive(2.0)
        a.child = TestRecursive(3.0)
        wrapper = TestWrapper(a)

        T_outer = Mooncake.tangent_type(OuterWrapper{TestWrapper{TestRecursive{Float64}}})
        T_inner = Mooncake.tangent_type(TestWrapper{TestRecursive{Float64}})
        @test T_outer isa Type

        a_tan = TestRecursiveTangent{Float64}(0.0)
        a_tan.child = TestRecursiveTangent{Float64}(0.0)
        inner_tangent = T_inner((x=a_tan,))
        @test T_outer((inner=inner_tangent,)) isa T_outer

        result = Mooncake.build_fdata(
            OuterWrapper{TestWrapper{TestRecursive{Float64}}},
            (wrapper,),
            (Mooncake.fdata(inner_tangent),),
        )
        @test result isa Mooncake.FData
    end

    @testset "build_output_tangent (#893, #1008)" begin
        T_wrapper = Mooncake.tangent_type(TestWrapper{TestRecursive{Float32}})
        @test T_wrapper == Mooncake.Tangent{@NamedTuple{x::TestRecursiveTangent{Float32}}}

        result = Mooncake.build_output_tangent(
            TestWrapper{TestRecursive{Float32}},
            (TestRecursive(1.0f0),),
            (TestRecursiveTangent{Float32}(0.0f0),),
        )
        @test result isa T_wrapper
    end
end
