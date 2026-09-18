@testset "nightly profile" begin
    @test isnothing(Base.get_extension(Mooncake, :MooncakeJETExt))
    @test isnothing(Base.get_extension(Mooncake, :MooncakeAllocCheckExt))

    # Open type values need the Julia 1.14 wrapper fallback.
    open_type = AbstractArray{TypeVar(:T),1}
    d = Dual(open_type, NoTangent())
    @test primal(d) === open_type
    @test d isa dual_type(Type{open_type})
    for (data, wrapper_type) in
        ((NoTangent(), codual_type), (NoFData(), Mooncake.fcodual_type))
        d = CoDual(open_type, data)
        @test primal(d) === open_type
        @test d isa wrapper_type(Type{open_type})
    end

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
