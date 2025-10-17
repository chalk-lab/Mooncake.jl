@testset "debug_mode" begin

    # Unless we explicitly check that the arguments are of the type as expected by the rule,
    # this will segfault.
    @testset "argument checking" begin
        f = x -> 5x
        rule = build_rrule(f, 5.0; debug_mode=true)
        @test_throws ErrorException rule(zero_fcodual(f), CoDual(0.0f0, 1.0f0))
    end

    # Forwards-pass tests.
    x = (CoDual(sin, NoTangent()), CoDual(5.0, NoFData()))
    @test_throws(ErrorException, Mooncake.DebugRRule(rrule!!)(x...))
    x = (CoDual(sin, NoFData()), CoDual(5.0, NoFData()))
    @test_throws(
        ErrorException, Mooncake.DebugRRule((x...,) -> (CoDual(1.0, 0.0), nothing))(x...)
    )

    # Basic type checking.
    x = (CoDual(size, NoFData()), CoDual(randn(10), randn(Float16, 11)))
    @test_throws ErrorException Mooncake.DebugRRule(rrule!!)(x...)

    # Element type checking. Abstractly typed-elements prevent determining incorrectness
    # just by looking at the array.
    x = (
        CoDual(size, NoFData()),
        CoDual(Any[rand() for _ in 1:10], Any[rand(Float16) for _ in 1:10]),
    )
    @test_throws ErrorException Mooncake.DebugRRule(rrule!!)(x...)

    # Test that bad rdata is caught as a pre-condition.
    y, pb!! = Mooncake.DebugRRule(rrule!!)(zero_fcodual(sin), zero_fcodual(5.0))
    @test_throws(InvalidRDataException, pb!!(5))

    # Test that bad rdata is caught as a post-condition.
    rule_with_bad_pb(x::CoDual{Float64}) = x, dy -> (5,) # returns the wrong type
    y, pb!! = Mooncake.DebugRRule(rule_with_bad_pb)(zero_fcodual(5.0))
    @test_throws InvalidRDataException pb!!(1.0)

    # Test that bad rdata is caught as a post-condition.
    rule_with_bad_pb_length(x::CoDual{Float64}) = x, dy -> (5, 5.0) # returns the wrong type
    y, pb!! = Mooncake.DebugRRule(rule_with_bad_pb_length)(zero_fcodual(5.0))
    @test_throws ErrorException pb!!(1.0)

    # Forward mode debug tests
    @testset "forward debug mode" begin

        @testset "valid inputs pass" begin
            # Single argument - use Float64, not π which has NoTangent
            rule = Mooncake.build_frule(zero_dual(sin), 0.0; debug_mode=true)
            @test rule(zero_dual(sin), Mooncake.Dual(3.14, 1.0)) isa Mooncake.Dual

            # Multiple arguments
            f_mul(x, y) = x * y
            rule = Mooncake.build_frule(zero_dual(f_mul), 2.0, 3.0; debug_mode=true)
            @test rule(zero_dual(f_mul), Mooncake.Dual(2.0, 1.0), Mooncake.Dual(3.0, 0.5)) isa Mooncake.Dual

            # Arrays
            h(x) = sum(x)
            rule = Mooncake.build_frule(zero_dual(h), randn(5); debug_mode=true)
            @test rule(zero_dual(h), Mooncake.Dual(randn(5), randn(5))) isa Mooncake.Dual

            # NoTangent (non-differentiable)
            rule = Mooncake.build_frule(zero_dual(identity), 5; debug_mode=true)
            @test rule(zero_dual(identity), Mooncake.Dual(5, NoTangent())) isa Mooncake.Dual
        end

        @testset "size mismatch detected" begin
            rule = Mooncake.build_frule(zero_dual(size), randn(10); debug_mode=true)
            @test_throws ErrorException rule(zero_dual(size), Mooncake.Dual(randn(11), randn(10)))
        end

        @testset "element type mismatch detected" begin
            rule = Mooncake.build_frule(zero_dual(identity), Any[1.0]; debug_mode=true)
            @test_throws ErrorException rule(zero_dual(identity), Mooncake.Dual(Any[1.0], Any[Float16(1.0)]))
        end

        @testset "scalar type mismatch detected" begin
            rule = Mooncake.build_frule(zero_dual(identity), 1.0; debug_mode=true)
            @test_throws ErrorException rule(zero_dual(identity), Mooncake.Dual(1.0, Float32(1.0)))
        end

        @testset "container type mismatch detected" begin
            rule = Mooncake.build_frule(zero_dual(identity), (1.0, 2.0); debug_mode=true)
            @test_throws ErrorException rule(zero_dual(identity), Mooncake.Dual((1.0, 2.0), [1.0, 2.0]))
        end

        @testset "error messages include type info" begin
            rule = Mooncake.build_frule(zero_dual(identity), [1.0]; debug_mode=true)

            try
                rule(zero_dual(identity), Mooncake.Dual([1.0], [Float32(1.0)]))
                @test false  # Should not reach
            catch e
                msg = sprint(showerror, e)
                @test occursin("input types", msg)
                @test occursin("Float", msg)  # Type info present
            end
        end

        @testset "integration with test_rule" begin
            # Test basic case - test_rule expects primal functions, not Duals
            Mooncake.TestUtils.test_rule(
                sr(123456), sin, 1.0;
                mode=ForwardMode,
                debug_mode=true,
                perf_flag=:none,
            )

            # Test with array
            Mooncake.TestUtils.test_rule(
                sr(123456), sum, randn(5);
                mode=ForwardMode,
                debug_mode=true,
                perf_flag=:none,
            )
        end
    end
end
