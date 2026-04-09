@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
    @test isnothing(Mooncake.Config().chunk_size)
    @test Mooncake.Config().enable_nfwd
    @test !Mooncake.Config().empty_cache
    @test !Mooncake.Config().run_gc
end
