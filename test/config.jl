@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
end
