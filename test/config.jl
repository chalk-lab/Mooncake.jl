@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
    @test isnothing(Mooncake.Config().chunk_size)
    @test !Mooncake.Config().empty_cache
    @test !hasproperty(Mooncake.Config(), :enable_nfwd)
    for enable_nfwd in (true, false, nothing)
        @test_throws MethodError Mooncake.Config(; enable_nfwd)
    end
end
