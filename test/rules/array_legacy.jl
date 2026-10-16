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

@testset "numeric refusal preserves storage" begin
    leaves(v::Mooncake.Nfwd.NDualArray) = [
        getfield(v, :primal), getfield(getfield(v, :partials_block), :parent)
    ]
    leaves(v::Tuple) = reduce(vcat, map(leaves, v); init=Any[])
    leaves(v) = Any[]
    cases, memory = Mooncake.derived_rule_test_cases(StableRNG, Val(:array_legacy))
    GC.@preserve memory for (io, perf, opts, f, args...) in cases
        opts isa NamedTuple && haskey(opts, :throws) || continue
        get(opts, :mode, nothing) === ForwardMode || continue
        c = IdDict{Any,Any}()
        slots = map(
            x -> TestUtils._throws_slot(StableRNG(123), Val(opts.chunk_size), x, c), args
        )
        arrays = reduce(vcat, map(x -> leaves(tangent(x)), slots); init=Any[])
        before, pointers = map(copy, arrays), map(pointer, arrays)
        TestUtils.test_rule(
            StableRNG(123),
            f,
            slots...;
            interface_only=io,
            perf_flag=perf,
            is_primitive=false,
            opts...,
        )
        @test arrays == before
        @test map(pointer, arrays) == pointers
    end
end

@testset "shared filter predicate effects" begin
    for N in (1, 8), keep in (true, false)
        calls = Ref(0)
        f = (x, calls, keep) -> begin
            a = vec(reshape(x, 1, 2))
            filter!(a) do v
                calls[] += 1
                keep || v < 3
            end
            sum(a)
        end
        z = Base.Fix1(Mooncake.zero_lifted, Val(N))
        x = z([2.0, 3.0])
        rule = build_frule(f, primal(x), calls, keep; chunk_size=N)
        if keep
            @test primal(rule(z(f), x, z(calls), z(keep))) == 5
        else
            @test_throws ArgumentError rule(z(f), x, z(calls), z(keep))
        end
        @test calls[] == 2
        @test primal(x) == [2.0, 3.0]
    end
end

@testset "sole pre-reshaped input" begin
    for N in (1, 8)
        p = vec(reshape([2.0, 3.0], 1, 2))
        z = Base.Fix1(Mooncake.zero_lifted, Val(N))
        x = z(p)
        fill!(getfield(tangent(x), :partials_block), 1.0)
        f = x -> (push!(x, 0.0); sum(x))
        y = build_frule(f, p; chunk_size=N)(z(f), x)
        @test primal(y) == 5.0
        @test all(tangent(y, k) == 2.0 for k in 1:N)
    end
end
