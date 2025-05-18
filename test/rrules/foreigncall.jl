@testset "foreigncall" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:foreigncall))

    @testset "foreigncalls that should never be hit: $name" for name in [
        :jl_alloc_array_1d,
        :jl_alloc_array_2d,
        :jl_alloc_array_3d,
        :jl_new_array,
        :jl_array_copy,
        :jl_type_intersection,
        :memset,
        :jl_get_tls_world_age,
        :memmove,
        :jl_object_id,
        :jl_array_sizehint,
        :jl_array_grow_beg,
        :jl_array_grow_end,
        :jl_array_grow_at,
        :jl_array_del_beg,
        :jl_array_del_end,
        :jl_array_del_at,
        :jl_value_ptr,
        :jl_threadid,
        :memhash_seed,
        :memhash32_seed,
        :jl_get_field_offset,
    ]
        @test_throws(
            ErrorException,
            Mooncake.frule!!(zero_dual(Mooncake._foreigncall_), zero_dual(Val(name))),
        )
        @test_throws(
            ErrorException,
            Mooncake.rrule!!(zero_codual(Mooncake._foreigncall_), zero_codual(Val(name))),
        )
    end
end
