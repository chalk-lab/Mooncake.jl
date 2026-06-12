@testset "new" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:new))
    include("tangent_world_age_regression.jl")

    # Forward `_new_` on a type whose canonical V is a dedicated container (not a struct-lift
    # wrapper) must fail with the clear coherence error pointing at the supported primitive,
    # not a baffling MethodError from the backing construction. `MemoryRef`'s canonical V is
    # `NDualMemoryRef`; its supported forward construction path is `memoryrefnew`.
    @static if VERSION >= v"1.11"
        @testset "_new_ struct-lift coherence guard (MemoryRef)" begin
            mem = fill!(Memory{Float64}(undef, 3), 1.0)
            ref = memoryref(mem)
            @test_throws "memoryrefnew" Mooncake.frule!!(
                Mooncake.zero_lifted(Val(1), Mooncake._new_),
                Mooncake.zero_lifted(Val(1), MemoryRef{Float64}),
                Mooncake.zero_lifted(Val(1), ref.ptr_or_offset),
                Mooncake.zero_lifted(Val(1), mem),
            )
        end
    end
end
