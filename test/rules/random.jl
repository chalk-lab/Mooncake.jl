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

# Construct the pointer from the real backing array so AD supplies its own tangent.
function random_bulk_fill!(rng, x, f, T, width, convert)
    GC.@preserve x f(rng, Ptr{UInt8}(pointer(x)), sizeof(x), T, width..., convert)
    return 1.0
end

@testset "Xoshiro bulk calls refuse unknown tangent layouts" begin
    for f in (Random.XoshiroSimd.xoshiro_bulk_simd, Random.XoshiroSimd.xoshiro_bulk_nosimd),
        P in (Float16, Float32, Float64, UInt64),
        rng in (Xoshiro(123), TaskLocalRNG())

        width = f === Random.XoshiroSimd.xoshiro_bulk_simd ? (Val(8),) : ()
        # UInt64 backing with Float64 format is valid primal storage, but its tangent
        # has zero-sized NoTangent elements. Never reinterpret it as floating storage.
        T = P === UInt64 ? Float64 : P
        P === Float16 && VERSION < v"1.12" && continue
        x = fill(P(2), 1000)
        Random.seed!(rng, 123)
        TestUtils.test_rule_throws(
            StableRNG(123),
            random_bulk_fill!,
            rng,
            x,
            f,
            T,
            width,
            Random.XoshiroSimd._bits2float;
            err=ArgumentError,
            msg="use rand! on a typed Array, Memory, or contiguous view",
        )
        @test all(==(2), x)
        @test rand(rng) == rand(Random.seed!(deepcopy(rng), 123))
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
