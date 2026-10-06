module WorldCarryFixtures
using Mooncake
@noinline leaf(x) = x + 1
@noinline middle(x) = leaf(x) * 2
top(x) = middle(x) + 1
end

# `top(x)` and its derivative at `x = 0.5` through a freshly requested forward rule.
function world_carry_top(x)
    interp = Mooncake.get_interpreter(Mooncake.ForwardMode)
    rule = Mooncake.build_frule(interp, Tuple{typeof(WorldCarryFixtures.top),Float64})
    result = rule(Mooncake.zero_dual(WorldCarryFixtures.top), Mooncake.Dual(x, 1.0))
    return Mooncake.primal(result), Mooncake.tangent(result)
end

@static if VERSION >= v"1.12-"
    @testset "rules across world moves" begin
        before = Mooncake.get_interpreter(Mooncake.ForwardMode)
        @test world_carry_top(0.5) == (4.0, 2.0)
        derived = Mooncake.get_interpreter(Mooncake.ForwardMode)

        @testset "an unrelated method keeps the cache" begin
            Core.eval(WorldCarryFixtures, :(unrelated(x) = x))
            after = Mooncake.get_interpreter(Mooncake.ForwardMode)
            @test after.world > derived.world
            @test after.oc_cache === derived.oc_cache
            @test after.cache_world == derived.cache_world
            @test !isempty(after.oc_cache)
            # A Lazy/Dynamic rule predicted before the move rebuilds into the same cache.
            @test Mooncake.get_interpreter(Mooncake.ForwardMode, derived.world).oc_cache ===
                derived.oc_cache
            @test world_carry_top(0.5) == (4.0, 2.0)
        end

        @testset "redefining a callee derives a new rule" begin
            Core.eval(WorldCarryFixtures, :(@noinline leaf(x) = x + 2))
            @test world_carry_top(0.5) == (6.0, 2.0)
        end

        @testset "a more specific method of a callee derives a new rule" begin
            Core.eval(WorldCarryFixtures, :(@noinline leaf(x::Float64) = x + 3))
            @test world_carry_top(0.5) == (8.0, 2.0)
        end

        @testset "a new primitive declaration derives a new rule" begin
            Core.eval(
                WorldCarryFixtures,
                :(Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(leaf),Float64}),
            )
            @test world_carry_top(0.5) == (8.0, 0.0)
        end
    end
end
