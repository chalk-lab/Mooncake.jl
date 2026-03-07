using Mooncake
using Test

f(x::Float64) = sin(x)

function Mooncake.frule!!(::Mooncake.Dual{typeof(f)}, x_dual::Mooncake.Dual{Float64})
    # wrong rule on purpose to check that it is used
    x = Mooncake.primal(x_dual)
    dx = Mooncake.tangent(x_dual)
    y = x^3
    dy = 3x^2 * dx
    return Mooncake.Dual(y, dy)
end

struct Multiplier
    a::Float64
end
(m::Multiplier)(x) = m.a * x

@testset "Working case" begin
    x = 5.0
    Mooncake.@is_forward Tuple{typeof(f),Float64}
    cache = prepare_gradient_cache(f, zero(x))
    val, grad = value_and_gradient!!(cache, f, x)
    @test val == x^3
    @test grad[1] == Mooncake.NoTangent()
    @test grad[2] == 3x^2
end

@testset "Failing cases" begin
    @testset "Wrong expression (not a Tuple{...})" begin
        @test_throws LoadError @eval Mooncake.@is_forward f(x::Float64)
        @test_throws "AssertionError" @eval Mooncake.@is_forward(:(f(x::Float64)))
    end
    @testset "Wrong number of arguments" begin
        @test_throws LoadError @eval Mooncake.@is_forward Tuple{typeof(f),Float64,Float64}
        @test_throws "ArgumentError: `Mooncake.@is_forward` does not yet support functions with 2 arguments." @eval Mooncake.@is_forward(
            Tuple{typeof(f),Float64,Float64}
        )
    end
    @testset "Closures" begin
        Mooncake.@is_forward Tuple{Multiplier,Float64}
        @test_throws ArgumentError prepare_gradient_cache(Multiplier(2.0), 5.0)
        @test_throws "`Mooncake.@is_forward` does not support functions which close over data." prepare_gradient_cache(
            Multiplier(2.0), 5.0
        )
    end
    @testset "No hardcoded `frule!!`" begin
        g(x) = 2 * f(x)
        Mooncake.@is_forward(Tuple{typeof(g),Float64})
        @test_throws MethodError prepare_gradient_cache(g, 5.0)
    end
end
