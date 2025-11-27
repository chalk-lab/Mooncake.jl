@testset "reverse debug mode" begin
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
end
