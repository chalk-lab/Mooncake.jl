@testset "nightly profile" begin
    @test isnothing(Base.get_extension(Mooncake, :MooncakeJETExt))
    @test isnothing(Base.get_extension(Mooncake, :MooncakeAllocCheckExt))

    x = [2.0]
    skipped = @testset "unavailable static analysis" begin
        TestUtils.test_opt(sin, (Float64,))
        TestUtils.report_opt(Tuple{typeof(sin),Float64})
        result = TestUtils.check_allocs(x -> (x[1] += 3; x), x)
        @test result === x
        @test x == [5.0]
        @test_throws ErrorException TestUtils.check_allocs(() -> error("propagated"))
    end
    @test Test.get_test_counts(skipped).broken == 4

    # The profile must not replace runtime allocation measurements with zero.
    @test TestUtils.count_allocs(copy, x) > 0
end
