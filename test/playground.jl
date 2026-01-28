using Mooncake
using Test

kwargs = (;)
struct SimplePair
    x1::Float64
    x2::Float64
end

@testset verbose = true "Plop" begin
    @testset "forwards mode ($kwargs)" for kwargs in [
        (;),
        (; debug_mode=true),
        (; debug_mode=false),
        (; debug_mode=true, silence_debug_messages=true),
    ]
        f = (x, y) -> x * y + cos(x)
        g = (sp::SimplePair) -> SimplePair(f(sp.x1, sp.x2), 2.0)

        x, y = 5.0, 4.0
        dx, dy = 3.0, 2.0
        fx = (f, x, y)
        dfx = (Mooncake.zero_tangent(f), dx, dy)
        z = f(x, y)
        dz = dx * y + x * dy + dx * (-sin(x))

        fx_sp = (g, SimplePair(x, y))
        dfx_sp = (Mooncake.zero_tangent(g), SimplePair(dx, dy))
        z_sp = g(SimplePair(x, y))

        @testset "Simple types" begin
            cache = Mooncake.prepare_derivative_cache(
                fx...; config=Mooncake.Config(; kwargs...)
            )

            # legacy Dual interface
            z_and_dz_dual = Mooncake.value_and_derivative!!(
                cache, map(Mooncake.Dual, fx, dfx)...
            )
            @test z_and_dz_dual isa Mooncake.Dual
            @test Mooncake.primal(z_and_dz_dual) == z
            @test Mooncake.tangent(z_and_dz_dual) == dz

            # new tuple interface
            z_and_dz_tup = Mooncake.value_and_derivative!!(cache, zip(fx, dfx)...)
            @test z_and_dz_tup isa Tuple{Float64,Float64}
            @test first(z_and_dz_tup) == z
            @test last(z_and_dz_tup) == dz
        end

        @testset "Structured types" begin
            cache_sp_friendly = Mooncake.prepare_derivative_cache(
                fx_sp...; config=Mooncake.Config(; friendly_tangents=true, kwargs...)
            )
            # friendly input doesn't error
            z_and_dz_sp = Mooncake.value_and_derivative!!(
                cache_sp_friendly, zip(fx_sp, dfx_sp)...
            )
            # output is friendly
            @test z_and_dz_sp isa Tuple{SimplePair,SimplePair}
            @test first(z_and_dz_sp) == SimplePair(z, 2.0)
            @test last(z_and_dz_sp) == SimplePair(dz, 0.0)

            cache_sp_unfriendly = Mooncake.prepare_derivative_cache(
                fx_sp...; config=Mooncake.Config(; friendly_tangents=false, kwargs...)
            )
            if get(kwargs, :debug_mode, false)
                @test_throws ErrorException Mooncake.value_and_derivative!!(
                    cache_sp_unfriendly, zip(fx_sp, dfx_sp)...
                )
            else
                @test_throws TypeError Mooncake.value_and_derivative!!(
                    cache_sp_unfriendly, zip(fx_sp, dfx_sp)...
                )
            end
        end
    end
end
