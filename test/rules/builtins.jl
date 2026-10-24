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

    @testset "is_homogeneous_and_immutable" begin
        x = Tuple(randn(1000))
        @test @inferred Mooncake.is_homogeneous_and_immutable(x)
        @test TestUtils.count_allocs(Mooncake.is_homogeneous_and_immutable, x) == 0
    end

    TestUtils.run_rule_test_cases(StableRNG, Val(:builtins))

    # `b^2` under/overflows Float32 while `a / b` and its derivatives are finite.
    @testset "div_float derivatives at extreme magnitudes: $f, $b" for f in (
            IntrinsicsWrappers.div_float, IntrinsicsWrappers.div_float_fast
        ),
        b in (1.0f-30, 1.0f30)

        a = 2b
        dual = Mooncake.frule!!(Dual(f, NoTangent()), Dual(a, 1.0f0), Dual(b, 1.0f0))
        @test tangent(dual) ≈ (1 - 2) / b
        _, pullback = Mooncake.rrule!!(zero_fcodual(f), zero_fcodual(a), zero_fcodual(b))
        _, da, db = pullback(1.0f0)
        @test da ≈ 1 / b
        @test db ≈ -2 / b
    end

    @testset "div_float with scaled seeds: $f" for f in (
        IntrinsicsWrappers.div_float, IntrinsicsWrappers.div_float_fast
    )
        dual = Mooncake.frule!!(
            Dual(f, NoTangent()), Dual(-1.0f0, 3.0f38), Dual(2.0f0, 3.0f38)
        )
        @test tangent(dual) ≈ 2.25f38
        _, pullback = Mooncake.rrule!!(
            zero_fcodual(f), zero_fcodual(1.0f30), zero_fcodual(1.0f10)
        )
        @test pullback(1.0f-36)[3] ≈ -1.0f-26
    end

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

    @testset "Disable bitcast to differentiable type, or bitcast from Int/UInt to Ptr" begin
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Float64), zero_fcodual(5))
        )
        @test_throws(
            ArgumentError,
            rrule!!(zero_fcodual(bitcast), zero_fcodual(Ptr{Float64}), zero_fcodual(5))
        )
    end

    # Tangents can't be loaded or stored through a pointer into memory with no tangent
    # storage, e.g. that of a `Vector{UInt8}`, or whose tangent is laid out differently, e.g.
    # that of a `Vector{Tuple{Int,Float64}}`.
    @testset "pointer access without a matching tangent" begin
        function store_load(x)
            b = zeros(UInt8, 8)
            p = Ptr{Float64}(pointer(b))
            return GC.@preserve b (unsafe_store!(p, x); unsafe_load(p))
        end
        bytes_load(x) = x * reinterpret(Float64, UInt8[0, 0, 0, 0, 0, 0, 0, 64])[1]
        function bytes_copy(x)
            b = zeros(UInt8, 8)
            w = [x]
            GC.@preserve b w unsafe_copyto!(pointer(w), Ptr{Float64}(pointer(b)), 1)
            return w[1]
        end
        function mixed_load(x)
            v = [(1, x), (2, 3x)]
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v, 2)) + 8)
        end
        function offset_load(x)
            v = [(Int32(1), Float32(x), 2.0)]
            return GC.@preserve v Float64(unsafe_load(Ptr{Float32}(pointer(v)) + 4))
        end
        function int_overwrite(x)
            v = [x]
            GC.@preserve v unsafe_store!(Ptr{Int}(pointer(v)), reinterpret(Int, 2.0))
            return v[1]
        end
        function union_load(x)
            v = Union{Int,Float64}[x]
            v[1] = reinterpret(Int, 2.0)
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v)))
        end
        function missing_load(x)
            v = Union{Missing,Float64}[x]
            return GC.@preserve v unsafe_load(Ptr{Float64}(pointer(v)))
        end
        function bytes_dot(x)
            b = zeros(UInt8, 8)
            v = [x]
            return GC.@preserve b v BLAS.dot(1, Ptr{Float64}(pointer(b)), 1, pointer(v), 1)
        end
        fs = [
            store_load,
            bytes_load,
            bytes_copy,
            bytes_dot,
            mixed_load,
            offset_load,
            int_overwrite,
            union_load,
            missing_load,
        ]
        @testset "$f" for f in fs
            TestUtils.test_rule_throws(StableRNG(123), f, 1.0; err=ArgumentError)
        end
        function objptr_null_diff(x)
            v = [("a", [x])]
            q = GC.@preserve v unsafe_load(
                Ptr{Ptr{Cvoid}}(pointer(v)) + fieldoffset(eltype(v), 2)
            )
            return (unsafe_pointer_to_objref(q)::Vector{Float64})[1]
        end
        TestUtils.test_rule_throws(
            StableRNG(123),
            objptr_null_diff,
            1.0;
            err=ArgumentError,
            msg="Mooncake cannot differentiate pointer operations without matching tangent storage, " *
                "e.g. a `Ptr{Float64}` into a `Vector{UInt8}` or a `Vector{Tuple{Int,Float64}}`, " *
                "or casting a pointer into differentiable memory to `Ptr{Int}`.",
        )
    end

    @testset "bitcast for Ptr->Ptr" begin
        res, pb = rrule!!(
            zero_fcodual(bitcast),
            zero_fcodual(Ptr{Float64}),
            CoDual(Ptr{Float32}(5), Ptr{Float32}(5)),
        )
        @test pb isa Mooncake.NoPullback
        @test res == CoDual(Ptr{Float64}(5), Ptr{Float64}(5))
    end

    @testset "throw" begin
        # Throw primitive continues to throw the exception it is meant to.
        @test_throws(
            ArgumentError,
            Mooncake.rrule!!(zero_fcodual(throw), zero_fcodual(ArgumentError("hello")))
        )
        @test_throws(
            AssertionError,
            Mooncake.rrule!!(zero_fcodual(throw), zero_fcodual(AssertionError("hello")))
        )

        # Derived rule throws the correct exception.
        rule_arg = Mooncake.build_rrule(Tuple{typeof(foo_throws),ArgumentError})
        @test_throws(
            ArgumentError,
            rule_arg(zero_fcodual(foo_throws), zero_fcodual(ArgumentError("hello")))
        )
        rule_assert = Mooncake.build_rrule(Tuple{typeof(foo_throws),AssertionError})
        @test_throws(
            AssertionError,
            rule_assert(zero_fcodual(foo_throws), zero_fcodual(AssertionError("hmmm")))
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

    cache_p = prepare_gradient_cache(f_pointerset, 3.0)
    val_p, grad_p = value_and_gradient!!(cache_p, f_pointerset, 3.0)
    @test val_p ≈ 6.0
    @test grad_p[2] ≈ 2.0

    cache_a = prepare_gradient_cache(f_atomic_pointerset, 3.0)
    val_a, grad_a = value_and_gradient!!(cache_a, f_atomic_pointerset, 3.0)
    @test val_a ≈ 6.0
    @test grad_a[2] ≈ 2.0
end

@testset "NaN handling in builtins rrules" begin
    test_cases = mapreduce(vcat, [Float16, Float32, Float64]) do T
        [(Base.sqrt_llvm, T(0)), (Base.sqrt_llvm_fast, T(0))]
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
