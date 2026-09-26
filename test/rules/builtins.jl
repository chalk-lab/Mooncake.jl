foo_throws(e) = throw(e)

@testset "builtins" begin
    @test_throws(
        ErrorException,
        Mooncake.rrule!!(CoDual(IntrinsicsWrappers.add_ptr, NoTangent()), 5.0, 4.0),
    )
    @test_throws(
        ErrorException,
        Mooncake.rrule!!(CoDual(IntrinsicsWrappers.sub_ptr, NoTangent()), 5.0, 4.0),
    )

    @testset "_apply_iterate_equivalent with $(typeof(args))" for args in Any[
        (*, 5.0, 4.0),
        (*, (5.0, 4.0)),
        (*, [1.0, 2.0]),
        (*, 1.0, [2.0]),
        (*, [1.0, 2.0], ()),
    ]
        @test ==(
            Core._apply_iterate(Base.iterate, args...),
            Mooncake._apply_iterate_equivalent(Base.iterate, args...),
        )
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:builtins))

    # Unhandled built-in throws an intelligible error.
    @test_throws(
        Mooncake.MissingRuleForBuiltinException,
        invoke(Mooncake.rrule!!, Tuple{CoDual{<:Core.Builtin}}, zero_fcodual(getfield)),
    )

    # Check that Base.showerror runs.
    @test ==(
        showerror(IOBuffer(; write=true), Mooncake.MissingRuleForBuiltinException("hmm")),
        nothing,
    )

    # Unhandled intrinsic throws an intelligible error.
    @test_throws(
        Mooncake.IntrinsicsWrappers.MissingIntrinsicWrapperException,
        invoke(Mooncake.IntrinsicsWrappers.translate, Tuple{Any}, Val(:foo)),
    )

    @testset "bitcast for Ptr->Ptr" begin
        # Narrowing to a non-differentiable element: it asks nothing of the tangent buffer (an
        # `Int64` read out of `Float64` bytes has no derivative), so the pair is re-typed together.
        res, pb = rrule!!(
            zero_fcodual(bitcast),
            zero_fcodual(Ptr{Int64}),
            CoDual(Ptr{Float64}(5), Ptr{Float64}(5)),
        )
        @test pb isa Mooncake.NoPullback
        @test res == CoDual(Ptr{Int64}(5), Ptr{Mooncake.NoTangent}(5))

        # Widening would write eight-byte cotangents across four-byte tangent slots.
        @test_throws ArgumentError rrule!!(
            zero_fcodual(bitcast),
            zero_fcodual(Ptr{Float64}),
            CoDual(Ptr{Float32}(5), Ptr{Float32}(5)),
        )
    end

    @testset "throw_inexacterror propagation" begin
        # Generic function that triggers throw_inexacterror via an inexact integer conversion.
        f_inexact(x) = Int8(round(Int, x))
        @test_throws InexactError prepare_gradient_cache(f_inexact, 200.0)
    end

    @static if isdefined(Core, :throw_methoderror)
        @testset "throw_methoderror propagation" begin
            # Generic function that triggers throw_methoderror (no matching method).
            f_nomatch(x) = x + "not a number"
            @test_throws MethodError prepare_gradient_cache(f_nomatch, 1.0)
        end
    end
end

@testset "pointer-to-pointer pointerset & atomic_pointerset correctness tests" begin
    function f_pointerset(x)
        c_1 = Ref(x)
        c_2 = Ref(x * 2.0)
        p = Ref(Base.unsafe_convert(Ptr{Float64}, c_1))
        GC.@preserve c_1 c_2 p begin
            Core.Intrinsics.pointerset(
                Base.unsafe_convert(Ptr{Ptr{Float64}}, p),
                Base.unsafe_convert(Ptr{Float64}, c_2),
                1,
                1,
            )
            unsafe_load(p[])
        end
    end

    function f_atomic_pointerset(x)
        c_1 = Ref(x)
        c_2 = Ref(x * 2.0)
        p = Ref(Base.unsafe_convert(Ptr{Float64}, c_1))
        GC.@preserve c_1 c_2 p begin
            Core.Intrinsics.atomic_pointerset(
                Base.unsafe_convert(Ptr{Ptr{Float64}}, p),
                Base.unsafe_convert(Ptr{Float64}, c_2),
                :monotonic,
            )
            unsafe_load(p[])
        end
    end

    for f in (f_pointerset, f_atomic_pointerset)
        cache = prepare_gradient_cache(f, 3.0)
        val, grad = value_and_gradient!!(cache, f, 3.0)
        @test val ≈ 6.0
        @test grad[2] ≈ 2.0
    end
end

@testset "unsafe_wrap forward rule on a non-differentiable pointer" begin
    # Non-differentiable pointers still need the wrapped array's canonical V.
    # Vector works on 1.10 too; keep buf alive while p is used.
    buf = UInt8[1, 2, 3, 4]
    p = pointer(buf)
    for N in (1, 2)
        out = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(N), unsafe_wrap),
            Mooncake.zero_lifted(Val(N), Array),
            Mooncake.zero_lifted(Val(N), p),
            Mooncake.zero_lifted(Val(N), (4,)),
        )
        @test typeof(Mooncake.tangent(out)) === Mooncake.dual_type(Val(N), Vector{UInt8})
        @test Mooncake.primal(out) == UInt8[1, 2, 3, 4]
    end
end

@testset "unsafe_wrap pointer shadow aliasing" begin
    # The registry checks the wrap call, but cannot mutate its explicit shadow buffer afterwards.
    z(x) = Mooncake.zero_lifted(Val(1), x)
    a, b = [3.0], [5.0]
    da, db = [1.0], [7.0]
    p, dp = fill(pointer(a), 2), fill(pointer(da), 2)
    GC.@preserve a b da db p dp begin
        ps = Mooncake.lift(pointer(p), pointer(dp))
        y = Mooncake.frule!!(z(unsafe_wrap), z(Array), ps, z((1, 2)))
        bs = Mooncake.lift(pointer(b), pointer(db))
        Mooncake.frule!!(z(IntrinsicsWrappers.pointerset), ps, bs, z(2), z(1))
        q = Mooncake.Lifted{Ptr{Float64},1}(Mooncake.primal(y)[2], Mooncake.tangent(y)[2])
        out = Mooncake.frule!!(z(IntrinsicsWrappers.pointerref), q, z(1), z(1))
        @test Mooncake.tangent(out, 1) == 7.0
        Mooncake.tangent(y)[2] = (pointer(da),)
        @test dp[2] == pointer(da)
    end
end

@testset "NaN handling in builtins rules" begin
    test_cases = mapreduce(vcat, [Float16, Float32, Float64]) do T
        [(Base.sqrt_llvm, T(0)), (Base.sqrt_llvm_fast, T(0))]
    end

    # The registry's chunk invariant requires finite partials. Exercise mixed inactive
    # and nonfinite active lanes directly, preserving that invariant for registered cases.
    for P in (Float16, Float32, Float64),
        f in (IntrinsicsWrappers.sqrt_llvm, IntrinsicsWrappers.sqrt_llvm_fast),
        x in (P(-1), P(0))

        partials = ntuple(k -> isodd(k) ? P(1) : P(0), 8)
        out = Mooncake.frule!!(
            Mooncake.zero_lifted(Val(8), f),
            Lifted{P,8}(x, Mooncake.NDual{P,8}(x, partials)),
        )
        y = x < 0 ? P(NaN) : P(0)
        active = x < 0 ? P(NaN) : P(Inf)
        @test isequal(primal(out), y)
        @test isequal(tangent(out).value, y)
        @test isequal(tangent(out).partials, ntuple(k -> isodd(k) ? active : P(0), 8))
    end

    # Test cases for avoiding `NaN` poisoning. 
    #  See https://github.com/chalk-lab/Mooncake.jl/issues/807 
    function builtins_nantester(f, args)
        a = f(args)
        b = args
        return b
    end

    for (f, args) in test_cases
        cache = prepare_gradient_cache(builtins_nantester, f, args)
        _, grad = value_and_gradient!!(cache, builtins_nantester, f, args)
        @test all(map(isone, grad[3:end]...))
    end
end

@testset "div_float pullback keeps `d/db` in range" begin
    # Squaring b overflows/underflows; dividing twice keeps d/db representable.
    # FD cannot resolve a derivative of -1e-200 against a value of 1.0.
    for (a, b) in ((1e200, 1e200), (1e-200, 1e-200), (2.0, 4.0))
        for f in (/, Base.FastMath.div_fast)
            g = Mooncake.value_and_gradient!!(
                Mooncake.prepare_gradient_cache(f, a, b), f, a, b
            )
            @test g[2][2] == 1 / b
            @test g[2][3] ≈ -(a / b) / b
            @test isfinite(g[2][3])
        end
    end
end
