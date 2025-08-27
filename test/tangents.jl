@testset "tangents" begin
    @testset "$(tangent_type(primal_type))" for (primal_type, expected_tangent_type) in Any[

        ## Misc. Specific Types
        (Cstring, NoTangent),
        (Cwstring, NoTangent),
        (Union{}, Union{}),

        ## Tuples

        # Unions of Tuples.
        (Union{Tuple{Float64},Tuple{Float32}}, Union{Tuple{Float64},Tuple{Float32}}),
        (Union{Tuple{Float64},Tuple{Int}}, Union{Tuple{Float64},NoTangent}),
        (Union{Tuple{},Tuple{Int}}, NoTangent),
        (
            Union{Tuple{Float64},Tuple{Int},Tuple{Float64,Int}},
            Union{Tuple{Float64},NoTangent,Tuple{Float64,NoTangent}},
        ),
        (Union{Tuple{Float64},Tuple{Any}}, Union{NoTangent,Tuple{Any}}),

        # UnionAlls of Tuples.
        (Tuple{T} where {T}, Union{NoTangent,Tuple{Any}}),
        (Tuple{T,T} where {T<:Real}, Union{NoTangent,Tuple{Any,Any}}),
        (Tuple{Float64,T} where {T<:Int}, Tuple{Float64,NoTangent}),
        (Union{Tuple{T},Tuple{T,T}} where {T<:Real}, Any),

        # Edge case: (provably) empty Tuple.
        (Tuple{}, NoTangent),

        # Vararg Tuples
        (Tuple, Any),
        (Tuple{Float64,Vararg}, Any),
        (Tuple{Float64,Vararg{Int}}, Any),
        (Tuple{Vararg{Int}}, Any),
        (Tuple{Int,Vararg{Int}}, Any),

        # Simple Tuples.
        (Tuple{Int}, NoTangent),
        (Tuple{Vararg{Int,250}}, NoTangent),
        (Tuple{Int,Int}, NoTangent),
        (Tuple{DataType,Int}, NoTangent),
        (Tuple{DataType,Vararg{Int,100}}, NoTangent),
        (Tuple{DataType,Type{Float64}}, NoTangent),
        (Tuple{DataType,Vararg{Type{Float64},100}}, NoTangent),
        (Tuple{Any}, Union{NoTangent,Tuple{Any}}),
        (Tuple{Any,Any}, Union{NoTangent,Tuple{Any,Any}}),
        (Tuple{Int,Any}, Union{NoTangent,Tuple{NoTangent,Any}}),
        (Tuple{Int,Float64}, Tuple{NoTangent,Float64}),
        (Tuple{Int,Vararg{Float64,100}}, Tuple{NoTangent,Vararg{Float64,100}}),
        (Tuple{Type{Float64},Float64}, Tuple{NoTangent,Float64}),
        (Tuple{DataType,Vararg{Float32,100}}, Tuple{NoTangent,Vararg{Float32,100}}),
        (Tuple{Tuple{Type{Int}},Float64}, Tuple{NoTangent,Float64}),

        ## NamedTuple

        # Unions of NamedTuples.
        (
            Union{@NamedTuple{a::Float64},@NamedTuple{b::Float64}},
            Union{@NamedTuple{a::Float64},@NamedTuple{b::Float64}},
        ),
        (
            Union{@NamedTuple{a::Float64},@NamedTuple{}},
            Union{@NamedTuple{a::Float64},NoTangent},
        ),
        (Union{@NamedTuple{a::Float64},@NamedTuple{a::Any}}, Any),

        # UnionAlls of NamedTuples.
        (@NamedTuple{a::T} where {T}, Union{NoTangent,NamedTuple{(:a,)}}),
        (@NamedTuple{a::T, b::T} where {T<:Real}, Union{NoTangent,NamedTuple{(:a, :b)}}),
        (
            @NamedTuple{a::Float64, b::T} where {T<:Int},
            Union{NoTangent,NamedTuple{(:a, :b)}},
        ),
        (Union{@NamedTuple{a::T},@NamedTuple{b::T,c::T}} where {T<:Any}, Any),
        (Union{@NamedTuple{T,Float64},@NamedTuple{T,Float64,Int}} where {T}, Any),

        # Edge case
        (@NamedTuple{}, NoTangent),

        # Simple NamedTuples.
        (@NamedTuple{a::Int}, NoTangent),
        (@NamedTuple{a::Int, b::Int}, NoTangent),
        (@NamedTuple{a::DataType, b::Int}, NoTangent),
        (@NamedTuple{a::DataType, b::Type{Float64}}, NoTangent),
        (@NamedTuple{a::Any}, Any),
        (@NamedTuple{a::Any, b::Any}, Any),
        (@NamedTuple{a::Int, b::Any}, Any),
        (@NamedTuple{b::Int, a::Float64}, @NamedTuple{b::NoTangent, a::Float64}),
        (@NamedTuple{a::Type{Float64}, b::Float64}, @NamedTuple{a::NoTangent, b::Float64}),
        (
            @NamedTuple{a::Tuple{Type{Int}}, b::Float64},
            @NamedTuple{a::NoTangent, b::Float64}
        ),
    ]
        TestUtils.test_tangent_type(primal_type, expected_tangent_type)
    end

    @testset "$(typeof(p))" for (interface_only, p, t...) in Mooncake.tangent_test_cases()
        test_tangent(Xoshiro(123456), p, t...; interface_only)
    end

    tangent(nt::NamedTuple) = Tangent(map(PossiblyUninitTangent, nt))
    mutable_tangent(nt::NamedTuple) = MutableTangent(map(PossiblyUninitTangent, nt))

    @testset "increment_field!!" begin
        @testset "NoTangent" begin
            nt = NoTangent()
            @test @inferred(increment_field!!(nt, nt, Val(:a))) == nt
            @test increment_field!!(nt, nt, :a) == nt
            @test @inferred(increment_field!!(nt, nt, Val(1))) == nt
            @test increment_field!!(nt, nt, 1) == nt
        end
        @testset "Tuple" begin
            nt = NoTangent()
            x = (5.0, nt)
            y = 3.0
            @test @inferred(increment_field!!(x, y, Val(1))) == (8.0, nt)
            @test @inferred(increment_field!!(x, nt, Val(2))) == (5.0, nt)

            # Slow versions.
            @test increment_field!!(x, y, 1) == (8.0, nt)
            @test increment_field!!(x, nt, 2) == (5.0, nt)

            # Homogeneous type optimisation.
            @test @inferred(increment_field!!((5.0, 4.0), 3.0, 2)) == (5.0, 7.0)

            # Homogeneous type optimisations scales to large `Tuple`s.
            @inferred(increment_field!!(Tuple(zeros(1_000)), 5.0, 3))
        end
        @testset "NamedTuple" begin
            nt = NoTangent()
            x = (a=5.0, b=nt)
            @test @inferred(increment_field!!(x, 3.0, Val(:a))) == (a=8.0, b=nt)
            @test @inferred(increment_field!!(x, nt, Val(:b))) == (a=5.0, b=nt)
            @test @inferred(increment_field!!(x, 3.0, Val(1))) == (a=8.0, b=nt)
            @test @inferred(increment_field!!(x, nt, Val(2))) == (a=5.0, b=nt)

            # Slow versions.
            @test increment_field!!(x, 3.0, :a) == (a=8.0, b=nt)
            @test increment_field!!(x, nt, :b) == (a=5.0, b=nt)
            @test increment_field!!(x, 3.0, 1) == (a=8.0, b=nt)
            @test increment_field!!(x, nt, 2) == (a=5.0, b=nt)

            # Homogeneous type optimisation.
            @test @inferred(increment_field!!((a=5.0, b=4.0), 3.0, 1)) == (a=8.0, b=4.0)
            @test @inferred(increment_field!!((a=5.0, b=4.0), 3.0, :a)) == (a=8.0, b=4.0)
        end
        @testset "Tangent" begin
            nt = NoTangent()
            x = tangent((a=5.0, b=nt))
            @test @inferred(increment_field!!(x, 3.0, Val(:a))) == tangent((a=8.0, b=nt))
            @test @inferred(increment_field!!(x, nt, Val(:b))) == tangent((a=5.0, b=nt))
            @test @inferred(increment_field!!(x, 3.0, Val(1))) == tangent((a=8.0, b=nt))
            @test @inferred(increment_field!!(x, nt, Val(2))) == tangent((a=5.0, b=nt))

            # Slow versions.
            @test increment_field!!(x, 3.0, :a) == tangent((a=8.0, b=nt))
            @test increment_field!!(x, nt, :b) == tangent((a=5.0, b=nt))
            @test increment_field!!(x, 3.0, 1) == tangent((a=8.0, b=nt))
            @test increment_field!!(x, nt, 2) == tangent((a=5.0, b=nt))
        end
        @testset "MutableTangent" begin
            nt = NoTangent()
            @testset "$f" for (f, val, comp) in [
                (Val(:a), 3.0, mutable_tangent((a=8.0, b=nt))),
                (:a, 3.0, mutable_tangent((a=8.0, b=nt))),
                (Val(:b), nt, mutable_tangent((a=5.0, b=nt))),
                (:b, nt, mutable_tangent((a=5.0, b=nt))),
                (Val(1), 3.0, mutable_tangent((a=8.0, b=nt))),
                (1, 3.0, mutable_tangent((a=8.0, b=nt))),
                (Val(2), nt, mutable_tangent((a=5.0, b=nt))),
                (2, nt, mutable_tangent((a=5.0, b=nt))),
            ]
                x = mutable_tangent((a=5.0, b=nt))
                @test @inferred(increment_field!!(x, val, f)) == comp
                @test @inferred(increment_field!!(x, val, f)) === x
            end
        end
    end
    @testset "restricted inner constructor" begin
        p = TestResources.NoDefaultCtor(5.0)
        t = Mooncake.Tangent((x=5.0,))
        @test_throws Mooncake.AddToPrimalException Mooncake._add_to_primal(p, t)
        @test Mooncake._add_to_primal(p, t, true) isa typeof(p)
    end
end

# The goal of these tests is to check that we can indeed generate tangent types for anything
# that we will encounter in the Julia language. We try to achieve this by pulling in types
# from LinearAlgebra, and Random, and generating tangents for them.
# Need to improve coverage of Base / Core really.
# It was these tests which made me aware of the issues around uninitialised values.

#
# Utility functionality for getting good test coverage of the types in a given module.
#

# function items_in(m::Module)
#     if m === Core.Compiler
#         @info "not getting names in Core"
#         return []
#     end
#     @info "gathering items in $m"
#     defined_names = filter(name -> isdefined(m, name), names(m; all=true))
#     return map(name -> getfield(m, name), defined_names)
# end

# function modules_in(m::Module)
#     if m === Core
#         @info "not getting names in Core"
#         return []
#     end
#     @info "gathering items in $m"
#     defined_names = filter(name -> isdefined(m, name), names(m; all=false))
#     items = map(name -> getfield(m, name), defined_names)
#     return filter(item -> item isa Module && item !== m, items)
# end

# function types_in(m::Module)
#     return filter(item -> item isa Type, items_in(m))
# end

# function concrete_types_in(m::Module)
#     return filter(isconcretetype, types_in(m))
# end

# function struct_types_in(m::Module)
#     return filter(isstructtype, types_in(m))
# end

# function primitive_types_in(m::Module)
#     return filter(isprimitivetype, types_in(m))
# end

# function non_abstract_types_in(m::Module)
#     return filter(!isabstracttype, types_in(m))
# end

# # Primitives are required to explicitly declare a method of `zero_tangent` which applies
# # to them. They must not hit the generic fallback. This function checks that there are no
# # primitives within the specified module which don't hit a generic fallback.
# function verify_zero_for_primitives_in(m::Module)
#     return filter(t -> isprimitivetype(t), types_in(m))
# end

# # A toy type on which to test tangent stuff in a variety of situations.
# struct Foo{T, V}
#     x::T
#     y::V
#     Foo{T, V}(x, y) where {T, V} = new{T, V}(x, y)
#     Foo{T, V}(x) where {T, V} = new{T, V}(x)
#     Foo(x, y) = new{typeof(x), typeof(y)}(x, y)
# end

# function test_basic_tangents(rng::AbstractRNG)
#     @testset "typeof($x)" for x in [
#         Foo(5.0, 5.0),
#         Foo{Float64, Ref{Float64}}(5.0),
#         Foo{Any, Ref{Float64}}(5.0),
#     ]
#         test_tangent(rng, x, 1e-3)
#     end
# end

# function test_tricky_tangents(rng::AbstractRNG)
#     @testset "typeof($x)" for x in [
#         Foo(5.0, 5.0),
#         Foo{Float64, Int64}(5.0),
#         Foo{Int64, Int64}(5, 4),
#         Foo{Real, Float64}(5.0, 4.0),
#         Foo{Real, Float64}(5.0),
#         Foo{Float64, Float64}(5.0),
#         Foo{Float64, Vector{Float64}}(5.0),
#         Foo{Real, Real}(5.0),
#         Foo{Union{Float64, Int64}, Int64}(5.0, 4),
#         Foo{Union{Float64, Int64}, Int64}(5.0),
#         Foo{DataType, Float64}(Float64),
#         Foo{DataType, Int64}(Float64),
#     ]
#         test_tangent(rng, x, 1e-3)
#     end
# end

# function primitives_for_LinearAlgebra()
#     return [
#         Adjoint(randn(4, 5)),
#         Bidiagonal(randn(5, 5), :U),
#         bunchkaufman(Symmetric(randn(5, 5))),
#         cholesky(collect(Matrix{Float64}(I, 3, 3))),
#         cholesky(collect(Matrix{Float64}(I, 3, 3)), Val(:true)),
#         ColumnNorm(),
#         Diagonal(randn(4)),
#         eigen(randn(3, 3)),
#         GeneralizedEigen(randn(3), randn(3, 3)),
#         svd(randn(3, 3), randn(3, 3)),
#         schur(randn(3, 3), randn(3, 3)),
#         Hermitian(randn(3, 3)),
#         hessenberg(randn(3, 3)),
#         LAPACKException(5),
#         ldlt(SymTridiagonal(randn(3), randn(2))),
#         lq(randn(3, 3)),
#         lu(randn(3, 3)),
#         LowerTriangular(randn(3, 3)),
#         NoPivot(),
#         PosDefException(5),
#         qr(randn(2, 4)),
#         qr(randn(2, 4), ColumnNorm()),
#         RankDeficientException(5),
#         RowMaximum(),
#         svd(randn(2, 3)),
#         schur(randn(2, 2)),
#         SingularException(5),
#         SymTridiagonal(randn(3), randn(2)),
#         Symmetric(randn(3, 3)),
#         Transpose(randn(3, 3)),
#         Tridiagonal(randn(5, 5)),
#         UniformScaling(5.0),
#         UnitLowerTriangular(randn(5, 5)),
#         UnitUpperTriangular(randn(5, 5)),
#         UpperHessenberg(randn(5, 5)),
#         UpperTriangular(randn(3, 3)),
#         ZeroPivotException(5),
#     ]
# end

# function primitives_for_Random()
#     return [
#         MersenneTwister(1), Xoshiro(1), RandomDevice(), TaskLocalRNG()
#     ]
# end

# function test_many_tangents(rng)
#     @testset "$(typeof(x))" for x in vcat(
#         [
#             Int8(5), Int16(5), Int32(5), Int64(5), Int128(5),
#             UInt8(5), UInt16(5), UInt32(5), UInt64(5), UInt128(5),
#             Float16(5.0), Float32(4.0), Float64(3.0), BigFloat(10.0),
#         ],
#         [randn(T, inds...)
#             for T in [Float16, Float32, Float64]
#             for inds in ((3, ), (3, 4), (3, 4, 5))
#         ],
#         [
#             fill(1, 3), fill(1, 3, 4), fill(1, 4, 5, 6),
#         ],
#         primitives_for_LinearAlgebra(),
#         primitives_for_Random(),
#     )
#         test_tangent(rng, x, 1e-3)
#     end
# end

# # Verify that the tangent type of everything in the `m` can be determined.
# # This is a surprisingly hard condition to satisfy.
# function get_unique_tangent_types(m::Module)
#     types_in_m = filter(x -> x isa Type, items_in(m))
#     unique(map(tangent_type, types_in_m))
#     return unique(map(tangent_type ∘ typeof, items_in(m)))
# end

# # Recursively gather all of the objects in all of the modules accessible from `m`.
# # Most of these things will be types, functions, etc, rather than instances of them.
# # This is useful, because it's lots of random stuff that you don't typically think of as
# # being AD-related, and should mostly have `NoTangent` tangents.
# # Test this to ensure that that actually happens!
# function gather_objects(m::Module)
#     get_unique_tangent_types(m)
#     foreach(gather_objects, modules_in(m))
# end

# @testset "tangent_types" begin
#     test_basic_tangents(Xoshiro(123456))
#     test_many_tangents(Xoshiro(123456))
#     test_tricky_tangents(Xoshiro(123456))
# end
