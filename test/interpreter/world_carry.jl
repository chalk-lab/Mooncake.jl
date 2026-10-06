module WorldCarryFixtures
using Mooncake
@noinline leaf(x) = x + 1
@noinline middle(x) = leaf(x) * 2
top(x) = middle(x) + 1
const TARGETS = Function[leaf]
dynamic(x) = TARGETS[1](x) * 3
@noinline overlaid(x) = 3x
overlaid_top(x) = overlaid(x) * 2
@noinline stale_leaf(x) = 5x
stale_top(x) = stale_leaf(x) * 2
plain(x) = 2x
end

# The primal and derivative at `x` of `f`, from a rule requested now in each mode.
function world_carry_forward(f, x)
    interp = Mooncake.get_interpreter(Mooncake.ForwardMode)
    rule = Mooncake.build_frule(interp, Tuple{typeof(f),Float64})
    result = rule(Mooncake.zero_dual(f), Mooncake.Dual(x, 1.0))
    return Mooncake.primal(result), Mooncake.tangent(result)
end

function world_carry_reverse(f, x)
    rule = Mooncake.build_rrule(f, x)
    value, gradient = Mooncake.value_and_gradient!!(rule, f, x)
    return value, gradient[2]
end

# Every rule requested now must agree with the functions as currently defined; the derivative of
# `top` and `dynamic` is zero once `leaf` is a zero-derivative primitive.
function world_carry_agrees(x; zero_leaf=false)
    fixtures = WorldCarryFixtures
    agrees = true
    for f in (fixtures.top, fixtures.dynamic, fixtures.leaf)
        expected = Base.invokelatest(f, x)
        slope = (Base.invokelatest(f, x + 1e-3) - Base.invokelatest(f, x - 1e-3)) / 2e-3
        if zero_leaf
            slope = 0.0
        end
        for (primal_value, derivative) in (world_carry_forward(f, x), world_carry_reverse(f, x))
            agrees &= primal_value ≈ expected
            agrees &= derivative ≈ slope
        end
    end
    return agrees
end

@static if VERSION >= v"1.12-"
    @testset "rules across world moves" begin
        @test world_carry_agrees(0.5)
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
            @test world_carry_agrees(0.5)
        end

        @testset "redefining a callee derives a new rule" begin
            Core.eval(WorldCarryFixtures, :(@noinline leaf(x) = x + 2))
            @test world_carry_agrees(0.5)
        end

        @testset "a more specific method of a callee derives a new rule" begin
            Core.eval(WorldCarryFixtures, :(@noinline leaf(x::Float64) = x + 3))
            @test world_carry_agrees(0.5)
        end

        @testset "redefining the method a rule is requested for derives a new rule" begin
            Core.eval(WorldCarryFixtures, :(@noinline middle(x) = leaf(x) * 4))
            @test world_carry_agrees(0.5)
        end

        @testset "a new primitive declaration derives a new rule" begin
            Core.eval(
                WorldCarryFixtures,
                :(Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(leaf),Float64}),
            )
            @test world_carry_agrees(0.5; zero_leaf=true)
        end

        @testset "a new overlay method of a callee derives a new rule" begin
            @test world_carry_forward(WorldCarryFixtures.overlaid_top, 2.0) == (12.0, 6.0)
            Core.eval(WorldCarryFixtures, :(Mooncake.@mooncake_overlay overlaid(x::Float64) = 10x))
            @test world_carry_forward(WorldCarryFixtures.overlaid_top, 2.0) == (40.0, 20.0)
        end

        @testset "a rule derived by an interpreter of an older world is not served later" begin
            stale_interp = Mooncake.get_interpreter(Mooncake.ForwardMode)
            Core.eval(WorldCarryFixtures, :(@noinline stale_leaf(x) = 7x))
            signature = Tuple{typeof(WorldCarryFixtures.stale_top),Float64}
            # A Lazy/Dynamic rule rebuilds at its old world this way.
            Mooncake.build_frule(stale_interp, signature; skip_world_age_check=true)
            @test world_carry_forward(WorldCarryFixtures.stale_top, 2.0) == (28.0, 14.0)
        end

        @testset "a rule for a non-dispatch signature lives in its own world only" begin
            signature = Tuple{typeof(WorldCarryFixtures.plain),Union{Float64,Float32}}
            interp = Mooncake.get_interpreter(Mooncake.ForwardMode)
            Mooncake.build_frule(interp, signature)
            key = Mooncake.rule_cache_key(interp, signature, false, :forward)
            @test Mooncake.cached_rule(interp, key) !== nothing
            Core.eval(WorldCarryFixtures, :(unrelated_again(x) = x))
            @test Mooncake.cached_rule(Mooncake.get_interpreter(Mooncake.ForwardMode), key) === nothing
        end
    end
end
