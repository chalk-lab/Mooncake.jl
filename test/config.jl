@testset "config" begin
    @test !Mooncake.Config().debug_mode
    @test !Mooncake.Config().silence_debug_messages
    @test isnothing(Mooncake.Config().chunk_size)
    @test !Mooncake.Config().empty_cache
    # `enable_nfwd` is deprecated: accepted but ignored, and no longer a field.
    @test !hasproperty(Mooncake.Config(), :enable_nfwd)
    @test Mooncake.Config(; enable_nfwd=false) isa Mooncake.Config
end
