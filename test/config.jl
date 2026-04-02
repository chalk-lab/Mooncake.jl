@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
    @test isnothing(Mooncake.Config().chunk_size)
    @test Mooncake.Config().enable_nfwd
end
