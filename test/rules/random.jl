@testset "random" begin
    TestUtils.run_rule_test_cases(StableRNG, Val(:random))
end

@testset "Random state advancement" begin
    rng = Xoshiro(123)
    for f in (rand!, rand, randn!, randn)
        rng2 = deepcopy(rng)
        args = f in (rand!, randn!) ? (rng, [0.0]) : (rng,)
        rule = Mooncake.build_rrule(f, args...)
        @test rng == rng2
        if f in (rand!, randn!)
            Mooncake.value_and_pullback!!(rule, [1.0], f, args...)
        else
            Mooncake.value_and_gradient!!(rule, f, args...)
        end
        @test rng != rng2
    end
end

# test_rule's reverse correctness check requires restoration of every input, including
# the RNG. Random fills deliberately advance it, so check their state contract here,
# using the registered interface/performance cases above.
@testset "Random fill aliases, restoration, and subsequent draw" begin
    cases, _ = Mooncake.hand_written_rule_test_cases(StableRNG, Val(:random))
    for (_, _, _, f, args...) in cases
        f in (rand!, randn!, randexp!) || continue
        rng, x, sampler... = args
        if !isconcretetype(eltype(x))
            for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
                @test !Mooncake.is_primitive(
                    Mooncake.DefaultCtx,
                    mode,
                    Mooncake._typeof((f, args...)),
                    Base.get_world_counter(),
                )
            end
            continue
        end
        initial = eltype(x) <: Integer ? 1 : 2
        initial_dx = eltype(x) <: Integer ? Mooncake.NoTangent() : 3
        replayed_next = rand(Random.seed!(deepcopy(rng), 123))
        expected_rng = Random.seed!(deepcopy(rng), 123)
        expected_x = f(expected_rng, deepcopy(x), sampler...)
        expected_next = rand(expected_rng)
        @testset "$mode $(typeof((f, args...)))" for mode in (
            Mooncake.ForwardMode, Mooncake.ReverseMode
        )
            r = Random.seed!(deepcopy(rng), 123)
            fill!(x isa SubArray ? parent(x) : x, initial)
            tx = Mooncake.zero_tangent(x)
            dx = x isa SubArray ? view(tx.fields.parent, x.indices...) : tx
            fill!(x isa SubArray ? tx.fields.parent : dx, initial_dx)
            if mode === Mooncake.ForwardMode
                rule = Mooncake.build_frule(f, r, x, sampler...)
                out = rule(
                    Mooncake.zero_dual(f),
                    Mooncake.zero_dual(r),
                    Mooncake.Dual(x, tx),
                    map(Mooncake.zero_dual, sampler)...,
                )
            else
                rule = Mooncake.build_rrule(f, r, x, sampler...)
                out, pb = rule(
                    Mooncake.zero_fcodual(f),
                    Mooncake.zero_fcodual(r),
                    Mooncake.CoDual(x, Mooncake.fdata(tx)),
                    map(Mooncake.zero_fcodual, sampler)...,
                )
            end
            @test Mooncake.primal(out) === x
            @test Mooncake.tangent(out) ===
                (mode === Mooncake.ForwardMode ? tx : Mooncake.fdata(tx))
            @test x == expected_x
            @test all(d -> d isa Mooncake.NoTangent || iszero(d), dx)
            if x isa SubArray
                @test parent(x) == parent(expected_x)
                @test tx.fields.parent[1] == (initial_dx)
                @test all(==(initial_dx), tx.fields.parent[(length(x) + 2):end])
            end
            if mode === Mooncake.ReverseMode
                eltype(x) <: Integer || fill!(dx, 7)
                pb(Mooncake.NoRData())
                @test all(==(initial), x)
                @test all(==(initial_dx), dx)
            end
            next = rand(r)
            @test next == expected_next
            @test next != replayed_next
        end
    end
end

# Direct kernels also protect callers which do not enter through rand!.
# SIMD must leave its unconsumed tail and both sentinels untouched.
@testset "Xoshiro bulk storage and RNG state" begin
    cases, memory = Mooncake.hand_written_rule_test_cases(StableRNG, Val(:random))
    GC.@preserve memory for (_, _, _, f, args...) in cases
        f in
        (Random.XoshiroSimd.xoshiro_bulk_simd, Random.XoshiroSimd.xoshiro_bulk_nosimd) ||
            continue
        rng, dst, len, T, rest... = args
        x = unsafe_wrap(
            Vector{T}, Ptr{T}(Mooncake.primal(dst) - sizeof(T)), div(len, sizeof(T)) + 2
        )
        dx = unsafe_wrap(
            Vector{T}, Ptr{T}(Mooncake.tangent(dst) - sizeof(T)), div(len, sizeof(T)) + 2
        )
        for mode in (Mooncake.ForwardMode, Mooncake.ReverseMode)
            fill!(x, 2)
            fill!(dx, 3)
            ref = copy(x)
            expected_rng = Xoshiro(123)
            result = GC.@preserve ref f(
                expected_rng, Ptr{UInt8}(pointer(ref, 2)), len, T, rest...
            )
            written = div(result === nothing ? len : result, sizeof(T))
            expected_next = rand(expected_rng)
            r = Xoshiro(123)
            if mode === Mooncake.ForwardMode
                out = Mooncake.frule!!(
                    Mooncake.zero_dual(f),
                    Mooncake.zero_dual(r),
                    Mooncake.Dual(Mooncake.primal(dst), Mooncake.tangent(dst)),
                    map(Mooncake.zero_dual, (len, T, rest...))...,
                )
            else
                out, pb = Mooncake.rrule!!(
                    Mooncake.zero_fcodual(f),
                    Mooncake.zero_fcodual(r),
                    dst,
                    map(Mooncake.zero_fcodual, (len, T, rest...))...,
                )
            end
            @test Mooncake.primal(out) === result
            @test x == ref
            @test dx[1] == 3
            @test all(==(3), dx[(written + 2):end])
            @test all(iszero, dx[2:(written + 1)])
            if mode === Mooncake.ReverseMode
                fill!(view(dx, 2:(written + 1)), 7)
                pb(Mooncake.NoRData())
                @test all(==(2), x)
                @test all(==(3), dx)
            end
            @test rand(r) == expected_next
        end
    end
end

@testset "Unsupported Xoshiro bulk writes fail before mutation" begin
    x, dx = fill(2.0, 2), fill(3.0, 2)
    GC.@preserve x dx for f in (
            Random.XoshiroSimd.xoshiro_bulk_simd, Random.XoshiroSimd.xoshiro_bulk_nosimd
        ),
        (len, T, convert) in (
            (8, Int64, Random.XoshiroSimd._id),
            (8, UInt8, Random.XoshiroSimd._id),
            (8, Bool, Random.XoshiroSimd._id),
            (8, Float64, Random.XoshiroSimd._id),
            (1, Float64, Random.XoshiroSimd._bits2float),
        )

        f === Random.XoshiroSimd.xoshiro_bulk_simd && len == 1 && continue
        width = f === Random.XoshiroSimd.xoshiro_bulk_simd ? (Val(8),) : ()
        for (rule, dual, zero) in (
            (Mooncake.frule!!, Mooncake.Dual, Mooncake.zero_dual),
            (Mooncake.rrule!!, Mooncake.CoDual, Mooncake.zero_fcodual),
        )
            rng = Xoshiro(123)
            dst = dual(Ptr{UInt8}(pointer(x)), Ptr{Mooncake.NoTangent}(pointer(dx)))
            @test_throws ArgumentError rule(
                zero(f), zero(rng), dst, map(zero, (len, T, width..., convert))...
            )
            @test x == [2.0, 2.0]
            @test dx == [3.0, 3.0]
            @test rng == Xoshiro(123)
        end
    end
end
