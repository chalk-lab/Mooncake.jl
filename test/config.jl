@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
    @test isnothing(Mooncake.Config().chunk_size)
    @test !Mooncake.Config().empty_cache
    @test Mooncake.Config().second_order_mode === :forward_over_reverse
    @test Mooncake.Config(second_order_mode=:reverse_over_forward).second_order_mode ===
        :reverse_over_forward
    @test_throws ArgumentError Mooncake.Config(second_order_mode=:bad)
    @test_deprecated Mooncake.Config(enable_nfwd=false)
    @test_deprecated Mooncake.Config(false, false, false, nothing, false, false)
end
