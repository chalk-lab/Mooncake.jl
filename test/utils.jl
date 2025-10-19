@testset "utils" begin
    @testset "_typeof" begin
        @test _typeof(5.0) == Float64
        @test _typeof(randn(1)) == Vector{Float64}
        @test _typeof(Float64) == Type{Float64}
        @test _typeof(Vector{Int}) == Type{Vector{Int}}
        @test _typeof(Vector{T} where {T}) == Type{Vector}
        @test _typeof((5.0, Float64)) == Tuple{Float64,Type{Float64}}
        @test _typeof((a=5.0, b=Float64)) == @NamedTuple{a::Float64, b::Type{Float64}}
    end
    @testset "tuple_map" begin
        @test map(sin, (5.0, 4.0)) == Mooncake.tuple_map(sin, (5.0, 4.0))
        @test ==(
            map(*, (5, 4.0, 3), (5.0, 4, 3.0)),
            Mooncake.tuple_map(*, (5, 4.0, 3), (5.0, 4, 3.0)),
        )

        @test map(sin, (a=5.0, b=4)) == Mooncake.tuple_map(sin, (a=5.0, b=4))
        @test ==(
            map(*, (a=5, b=4.0, c=3), (a=5.0, b=4, c=3.0)),
            Mooncake.tuple_map(*, (a=5, b=4.0, c=3), (a=5.0, b=4, c=3.0)),
        )

        # Require that length of arguments are equal.
        @test_throws ArgumentError Mooncake.tuple_map(*, (5.0, 4.0), (4.0,))
        @test_throws ArgumentError Mooncake.tuple_map(*, (4.0,), (5.0, 4.0))
    end
    @testset "_findall" begin
        @test @inferred Mooncake._findall(identity, (false, true, false)) == (2,)
        @test @inferred Mooncake._findall(identity, (true, false, true)) == (3, 1)
        # regression test for https://github.com/chalk-lab/Mooncake.jl/issues/473
        @test @inferred Mooncake._findall(Base.Fix2(isa, Union), Tuple(zeros(1000))) == ()
    end
    @testset "stable_all" begin
        @test Mooncake.stable_all((false,)) == false
        @test Mooncake.stable_all((true,)) == true
        @test Mooncake.stable_all((false, true)) == false
        @test Mooncake.stable_all((false, false)) == false
        @test Mooncake.stable_all((true, false)) == false
        @test Mooncake.stable_all((true, true)) == true

        # regression test for https://github.com/chalk-lab/Mooncake.jl/issues/473
        @test Mooncake.stable_all(Tuple(fill(false, 1000))) == false
        @test Mooncake.stable_all(Tuple(fill(true, 1000))) == true
    end
    @testset "_map_if_assigned!" begin
        @testset "unary bits type" begin
            x = Vector{Float64}(undef, 10)
            y = randn(10)
            z = Mooncake._map_if_assigned!(sin, y, x)
            @test z === y
            @test all(map(isequal, z, map(sin, x)))
        end
        @testset "unary non bits type" begin
            x = Vector{Vector{Float64}}(undef, 2)
            x[1] = randn(5)
            y = [1.0, 1.0]
            z = Mooncake._map_if_assigned!(sum, y, x)
            @test z === y

            # The first element of `x` is assigned, so z[1] should be its sum.
            @test z[1] ≈ sum(x[1])

            # The second element of `x` is unassigned, so z[2] should be unchanged.
            @test z[2] == 1.0
        end
        @testset "binary bits type" begin
            x1 = Vector{Float64}(undef, 7)
            x2 = Vector{Float64}(undef, 7)
            y = Vector{Float64}(undef, 7)
            z = Mooncake._map_if_assigned!(*, y, x1, x2)
            @test z === y
            @test all(map(isequal, z, map(*, x1, x2)))
        end
        @testset "binary non bits type" begin
            x1 = Vector{Vector{Float64}}(undef, 2)
            x1[1] = randn(3)
            x2 = [randn(3), randn(2)]
            y = [1.0, 1.0]
            z = Mooncake._map_if_assigned!(dot, y, x1, x2)
            @test z === y

            # The first element of x1 is assigned, so should have the inner product in z[1].
            @test z[1] ≈ dot(x1[1], x2[1])

            # The second element of x2 is not assigned, so z[2] should be unchanged.
            @test z[2] == 1
        end
    end
    @testset "_map" begin
        x = randn(10)
        y = randn(10)
        @test Mooncake._map(*, x, y) == map(*, x, y)
        @assert length(map(*, x, randn(11))) == 10
        @test_throws AssertionError Mooncake._map(*, x, randn(11))
    end
    @testset "is_always_initialised" begin
        @test Mooncake.is_always_initialised(TestResources.StructFoo, 1)
        @test !Mooncake.is_always_initialised(TestResources.StructFoo, 2)
    end
    @testset "is_always_fully_initialised" begin
        @test Mooncake.is_always_fully_initialised(TestResources.Foo)
        @test !Mooncake.is_always_fully_initialised(TestResources.StructFoo)
    end
    @testset "opaque_closure and misty_closure" begin

        # Get the IRCode for `sin` applied to a `Float64`.
        ir = Base.code_ircode_by_type(Tuple{typeof(sin),Float64})[1][1]

        # Check that regular OpaqueClosure construction works as expected.
        oc = Mooncake.opaque_closure(Float64, ir)
        @test oc isa Core.OpaqueClosure{Tuple{Float64},Float64}
        @test oc(5.0) == sin(5.0)

        # Construct the same OpaqueClosure, but tell it what the type is.
        oc2 = Mooncake.opaque_closure(Any, ir)
        @test oc2 isa Core.OpaqueClosure{Tuple{Float64},Any}
        @test oc2(5.0) == sin(5.0)

        # Check that we can get a MistyClosure also.
        mc = Mooncake.misty_closure(Float64, ir)
        @test mc(5.0) == sin(5.0)
    end
end
