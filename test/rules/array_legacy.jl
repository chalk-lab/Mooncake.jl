using Mooncake: frule!!

@testset "array_legacy" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:array_legacy))
end

@testset "arrayset correctness with Ptr-containing arrays" begin
    # Regression test for https://github.com/chalk-lab/Mooncake.jl/issues/999.
    # isbits_arrayset_rrule previously called zero_tangent(primal(v)) which throws for
    # Ptr types. The fix uses zero_tangent(primal(v), tangent(v)) (two-arg form).
    # FD-based correctness testing is not possible for Ptr primals, so we verify
    # that AD runs without error and returns the correct value and gradient.
    function f_arrayset_ptr(x)
        c_1 = Ref(x)
        c_2 = Ref(x * 2.0)
        GC.@preserve c_1 c_2 begin
            arr = fill(Base.unsafe_convert(Ptr{Float64}, c_1), 2)
            Base.arrayset(true, arr, Base.unsafe_convert(Ptr{Float64}, c_2), 1)
            unsafe_load(arr[1])
        end
    end
    cache = prepare_gradient_cache(f_arrayset_ptr, 3.0)
    val, grad = value_and_gradient!!(cache, f_arrayset_ptr, 3.0)
    @test val ≈ 6.0
    @test grad[2] ≈ 2.0
end

# test_rule_throws cannot seed shared tangents or assert the post-throw state.
@testset "shared tangent resize refusal" begin
    for mode in (frule!!, rrule!!),
        borrowed in (false, true),
        (f, args, capacity, refuses) in (
            (Base._growbeg!, (1,), 2, true),
            (Base._growend!, (1,), 2, true),
            (Base._growat!, (2, 1), 2, true),
            (Base._deletebeg!, (1,), 2, true),
            (Base._deleteend!, (1,), 2, true),
            (Base._deleteat!, (2, 1), 2, true),
            (sizehint!, (100,), 2, true),
            (sizehint!, (2,), 2, false),
            (sizehint!, (2,), 1000, true),
        )

        a, da = [1.0, 2.0], zeros(2)
        sizehint!(a, borrowed ? capacity : 2)
        sizehint!(da, capacity)
        alias = reshape(reshape(da, 2, 1), 2)
        dual = mode === frule!! ? Dual : CoDual
        zero = mode === frule!! ? zero_dual : zero_fcodual
        d = dual(a, borrowed ? alias : da)
        if refuses
            @test_throws r"ArgumentError:.*array.*copy.*before resizing" mode(
                zero(f), d, map(zero, args)...
            )
        else
            mode(zero(f), d, map(zero, args)...)
            @test pointer(da) == pointer(alias)
        end
        @test a == [1.0, 2.0]
        @test da == zeros(2)
    end
end
