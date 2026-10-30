@testset "prerelease profile" begin
    @test !isnothing(Base.get_extension(Mooncake, :MooncakeJETExt))
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

    # Inference widens a constant type to `Core.TypeEgal`; it must get `Type{P}`'s wrappers.
    for wrapper_type in (dual_type, codual_type, Mooncake.fcodual_type)
        @test wrapper_type(Core.Typeof(Float64)) == wrapper_type(Type{Float64})
    end

    x = [2.0]
    skipped = @testset "unavailable static analysis" begin
        result = TestUtils.check_allocs(x -> (x[1] += 3; x), x)
        @test result === x
        @test x == [5.0]
        @test_throws ErrorException TestUtils.check_allocs(() -> error("propagated"))
    end
    @test Test.get_test_counts(skipped).broken == 2

    # The profile must not replace runtime allocation measurements with zero.
    @test TestUtils.count_allocs(copy, x) > 0
end
