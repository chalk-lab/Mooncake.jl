using DispatchDoctor:
    @stable, @unstable, allow_unstable, TypeInstabilityError, _Utils.JULIA_OK
using Test
using StableRNGs: StableRNG

@stable stable_square(x) = x^2
@stable type_unstable_square(x) = x > 0 ? x^2 : 0
@stable default_union_limit = 2 type_unstable_square2(x) = x > 0 ? x^2 : 0

@testset "dispatch_doctor" begin
    @testset "stable function" begin
        f = stable_square
        x = 2.0

        result = f(x)
        @test result ≈ 4.0

        result = allow_unstable(() -> type_unstable_square(x))
        @test result ≈ 4.0

        result_neg = allow_unstable(() -> type_unstable_square(-1.0))
        @test result_neg ≈ 0.0

        # Skip this test on unsupported Julia versions to avoid misleading test failures.
        JULIA_OK && @test_throws TypeInstabilityError type_unstable_square(2.0)

        # No allow_unstable needed
        result = type_unstable_square2(2.0)
        @test result ≈ 4.0
    end

    @testset "derivatives are correct" begin
        rng = StableRNG(123)
        @testset "test_rule - $f with input $x" for x in [5.0, -5.0],
            f in [stable_square, type_unstable_square, type_unstable_square2]

            allow_unstable() do
                TestUtils.test_rule(rng, f, x; is_primitive=false)
            end
        end
    end
end
