@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
    @test isnothing(Mooncake.Config().chunk_size)
    @test_throws "`enable_nfwd` was removed" Mooncake.Config(; enable_nfwd=false)
    @test !Mooncake.Config().empty_cache
end
