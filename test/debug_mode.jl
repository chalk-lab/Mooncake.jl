@testset "debug_mode" begin
    @testset "reverse debug mode" begin
        # Unless we explicitly check that the arguments are of the type as expected by the rule,
        # this will segfault.
        @testset "argument checking" begin
            f = x -> 5x
            rule = build_rrule(f, 5.0; debug_mode=true)
            @test_throws ErrorException rule(zero_fcodual(f), CoDual(0.0f0, 1.0f0))
            @test_throws ErrorException rule(zero_fcodual(f), CoDual(5.0, 1.0))
        end

        # Forwards-pass tests.
        x = (CoDual(sin, NoTangent()), CoDual(5.0, NoFData()))
        @test_throws(ErrorException, Mooncake.DebugRRule(rrule!!)(x...))
        x = (CoDual(sin, NoFData()), CoDual(5.0, NoFData()))
        @test_throws(
            ErrorException,
            Mooncake.DebugRRule((x...,) -> (CoDual(1.0, 0.0), nothing))(x...)
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
    end

    @testset "forward debug mode" begin
        @testset "valid inputs pass" begin
            # Single argument - use Float64, not π which has NoTangent
            rule = Mooncake.build_frule(zero_dual(sin), 0.0; debug_mode=true)
            @test rule(Mooncake.lift(sin, NoTangent()), Mooncake.lift(3.14, 1.0)) isa Lifted

            # Multiple arguments
            f_mul(x, y) = x * y
            rule = Mooncake.build_frule(zero_dual(f_mul), 2.0, 3.0; debug_mode=true)
            @test rule(
                Mooncake.lift(f_mul, NoTangent()),
                Mooncake.lift(2.0, 1.0),
                Mooncake.lift(3.0, 0.5),
            ) isa Lifted

            # Arrays
            h(x) = sum(x)
            rule = Mooncake.build_frule(zero_dual(h), randn(5); debug_mode=true)
            @test rule(Mooncake.lift(h, NoTangent()), Mooncake.lift(randn(5), randn(5))) isa
                Lifted

            # NoTangent (non-differentiable)
            rule = Mooncake.build_frule(zero_dual(identity), 5; debug_mode=true)
            @test rule(
                Mooncake.lift(identity, NoTangent()), Mooncake.lift(5, NoTangent())
            ) isa Lifted
        end

        # A concrete-primal slot whose V is not the canonical dual_type must be caught;
        # Ptr primals are exempt (bitcast chains legitimately re-type per-lane pointers).
        @testset "V-coherence check" begin
            bad = Lifted{Float64,1,Mooncake.NoDual}(1.0, Mooncake.NoDual())
            @test_throws ErrorException Mooncake.verify_canonical_dual_type(bad)
            @test Mooncake.verify_canonical_dual_type(Mooncake.lift(3.14, 1.0)) === nothing
            xv = [1.0]
            GC.@preserve xv begin
                ptr_slot = Lifted{Ptr{Float64},1}(
                    pointer(xv), (Ptr{Tuple{Float64}}(UInt(pointer(xv))),)
                )
                @test Mooncake.verify_canonical_dual_type(ptr_slot) === nothing
            end
        end

        @testset "integration with test_rule" begin
            # Test basic case - test_rule expects primal functions, not Duals
            Mooncake.TestUtils.test_rule(
                sr(123456), sin, 1.0; mode=ForwardMode, debug_mode=true, perf_flag=:none
            )

            # Test with array
            Mooncake.TestUtils.test_rule(
                sr(123456),
                sum,
                randn(5);
                mode=ForwardMode,
                debug_mode=true,
                perf_flag=:none,
            )
        end
    end
end
