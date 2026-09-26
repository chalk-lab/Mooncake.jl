using Mooncake: frule!!, zero_fcodual

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

@testset "shared tangent resize refusal" begin
    # Use tuple elements for plain-Array forward storage; numeric vectors resize
    # their NDualArray partial block separately.
    for (f, args) in (
            (Base._growbeg!, (1,)),
            (Base._growend!, (1,)),
            (Base._growat!, (2, 1)),
            (Base._deletebeg!, (1,)),
            (Base._deleteend!, (1,)),
            (Base._deleteat!, (2, 1)),
            (sizehint!, (100,)),
        ),
        mode in (frule!!, rrule!!),
        borrowed in (false, true)

        a = mode === frule!! ? [(1.0,), (2.0,)] : [1.0, 2.0]
        da = mode === frule!! ? zero_dual(Val(1), a) : zeros(2)
        alias = reshape(reshape(da, 2, 1), 2)
        da = borrowed ? alias : da
        dual = mode === frule!! ? Lifted{typeof(a),1} : CoDual
        zero = mode === frule!! ? Base.Fix1(Mooncake.zero_lifted, Val(1)) : zero_fcodual
        a_before, da_before = copy(a), copy(da)
        @test_throws r"ArgumentError:.*array.*copy.*before resizing" mode(
            zero(f), dual(a, da), map(zero, args)...
        )
        @test a == a_before
        @test da == da_before
    end
    for mode in (frule!!, rrule!!), capacity in (2, 1000), borrowed in (false, true)
        a = mode === frule!! ? [(1.0,), (2.0,)] : [1.0, 2.0]
        da = mode === frule!! ? zero_dual(Val(1), a) : zeros(2)
        sizehint!(a, borrowed ? capacity : 2)
        sizehint!(da, capacity)
        alias = reshape(reshape(da, 2, 1), 2)
        dual = mode === frule!! ? Lifted{typeof(a),1} : CoDual
        zero = mode === frule!! ? Base.Fix1(Mooncake.zero_lifted, Val(1)) : zero_fcodual
        d = dual(a, borrowed ? alias : da)
        if capacity == 2
            mode(zero(sizehint!), d, zero(2))
            @test pointer(da) == pointer(alias)
        else
            @test_throws ArgumentError mode(zero(sizehint!), d, zero(2))
        end
    end
end
