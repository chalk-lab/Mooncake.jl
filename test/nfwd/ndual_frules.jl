# Direct unit tests for the NDual frule!!s. These exercise dispatch on
# `NDual{T,1}` arguments rather than going through `prepare_derivative_cache(...; chunk_size=1)`,
# so a missing or broken NDual overload (e.g. for one of the
# rules_via_nfwd primitives) surfaces as a focused failure.
#
# Approach: for each primitive `f` known to have both a Dual{P} frule and an
# NDual{T,1} frule, evaluate both at the same `(x, ẋ)` and check that the
# value/tangent of the Dual result matches the .value/.partials[1] of the
# NDual result.
#
# Functions are sourced from `src/rules/rules_via_nfwd.jl` and the manual NDual
# frule!!s in `src/rules/builtins.jl`.

using Mooncake: NDual, Dual, NoTangent, frule!!, zero_dual, primal, tangent
using LinearAlgebra: hypot

@testset "NDual frule!! direct sweep" begin
    function check_unary(f, x; check_finite=true)
        ẋ = one(typeof(x))
        d_dual = frule!!(zero_dual(f), Dual(x, ẋ))
        d_ndual = frule!!(zero_dual(f), NDual{typeof(x),1}(x, (ẋ,)))
        @test d_ndual isa NDual{typeof(x),1}
        @test d_ndual.value === primal(d_dual)
        if check_finite && isfinite(tangent(d_dual)) && isfinite(d_ndual.partials[1])
            @test d_ndual.partials[1] ≈ tangent(d_dual)
        else
            # NaN/Inf: just check both sides agree.
            @test isequal(d_ndual.partials[1], tangent(d_dual))
        end
    end

    function check_binary(f, x, y; ẋ=one(typeof(x)), ẏ=zero(typeof(y)))
        d_dual = frule!!(zero_dual(f), Dual(x, ẋ), Dual(y, ẏ))
        d_ndual = frule!!(
            zero_dual(f), NDual{typeof(x),1}(x, (ẋ,)), NDual{typeof(y),1}(y, (ẏ,))
        )
        @test d_ndual isa NDual
        @test d_ndual.value === primal(d_dual)
        if isfinite(tangent(d_dual)) && isfinite(d_ndual.partials[1])
            @test d_ndual.partials[1] ≈ tangent(d_dual)
        else
            @test isequal(d_ndual.partials[1], tangent(d_dual))
        end
    end

    @testset "unary: $f" for f in (
        # Smooth functions defined on R+ where 1.5 is well inside the domain.
        exp,
        exp2,
        exp10,
        expm1,
        log,
        log10,
        log2,
        log1p,
        sqrt,
        cbrt,
        sin,
        cos,
        cospi,
        sind,
        cosd,
        sinpi,
        sinh,
        cosh,
        tanh,
        sech,
        asinh,
        sinc,
        deg2rad,
        rad2deg,
        nextfloat,
        prevfloat,
        tanpi,
    )
        check_unary(f, 0.7)
    end
    @testset "unary at 1.0: $f" for f in (sec, csc, cot, secd, cscd, cotd, csch, coth, atan)
        check_unary(f, 1.0)
    end
    @testset "unary on (-1, 1): $f" for f in (asin, acos, asech, atanh, asind, acosd)
        check_unary(f, 0.5; check_finite=false)
    end
    @testset "unary on (1, ∞): $f" for f in (asec, acsc, acoth, asecd, acscd, acosh)
        check_unary(f, 1.5; check_finite=false)
    end
    @testset "tan-family at 0.5" for f in (tan, tand, acot, atand, acotd)
        check_unary(f, 0.5; check_finite=false)
    end
    @testset "mod2pi / acsch / asinh: $f" for f in (mod2pi, acsch, asinh)
        check_unary(f, 1.5)
    end

    @testset "binary: $f" for f in (atan, log, ^, mod, max, min)
        check_binary(f, 1.5, 2.5)
    end

    @testset "fpext / fptrunc convert" begin
        # fpext from Float32 to Float64
        x32 = 1.5f0
        d = frule!!(
            zero_dual(Mooncake.IntrinsicsWrappers.fpext),
            Dual(Float64, NoTangent()),
            NDual{Float32,1}(x32, (1.0f0,)),
        )
        @test d isa NDual{Float64,1}
        @test d.value === Float64(x32)
        @test d.partials[1] === 1.0
        # fptrunc from Float64 to Float32
        x64 = 1.5
        d2 = frule!!(
            zero_dual(Mooncake.IntrinsicsWrappers.fptrunc),
            Dual(Float32, NoTangent()),
            NDual{Float64,1}(x64, (1.0,)),
        )
        @test d2 isa NDual{Float32,1}
        @test d2.value === Float32(x64)
        @test d2.partials[1] === 1.0f0
    end

    @testset "tuple-output: $f" for f in (sincosd, sincospi, modf)
        x = 0.7
        d_ndual = frule!!(zero_dual(f), NDual{Float64,1}(x, (1.0,)))
        @test d_ndual isa Tuple
        @test length(d_ndual) == 2
    end

    @testset "vararg hypot $n-arg" for n in 1:3
        xs = ntuple(i -> NDual{Float64,1}(Float64(i + 1), (Float64(i),)), n)
        d = frule!!(zero_dual(hypot), xs...)
        @test d isa NDual{Float64,1}
        @test d.value ≈ hypot(Float64.(2:(n + 1))...)
    end

    @testset "ternary clamp" begin
        d = frule!!(
            zero_dual(clamp),
            NDual{Float64,1}(1.5, (1.0,)),
            NDual{Float64,1}(0.0, (0.0,)),
            NDual{Float64,1}(2.0, (0.0,)),
        )
        @test d isa NDual{Float64,1}
        @test d.value === 1.5
        @test d.partials[1] === 1.0
    end

    @testset "intrinsic float scalar: $f" for f in (
        Mooncake.IntrinsicsWrappers.abs_float,
        Mooncake.IntrinsicsWrappers.neg_float,
        Mooncake.IntrinsicsWrappers.neg_float_fast,
        Mooncake.IntrinsicsWrappers.sqrt_llvm,
        Mooncake.IntrinsicsWrappers.sqrt_llvm_fast,
    )
        x = 1.5
        d_dual = frule!!(zero_dual(f), Dual(x, 1.0))
        d_ndual = frule!!(zero_dual(f), NDual{Float64,1}(x, (1.0,)))
        @test d_ndual.value ≈ primal(d_dual)
        @test d_ndual.partials[1] ≈ tangent(d_dual)
    end

    @testset "binary intrinsic: $f" for f in (
        Mooncake.IntrinsicsWrappers.add_float,
        Mooncake.IntrinsicsWrappers.sub_float,
        Mooncake.IntrinsicsWrappers.mul_float,
        Mooncake.IntrinsicsWrappers.div_float,
        Mooncake.IntrinsicsWrappers.copysign_float,
    )
        x, y = 1.5, 2.5
        ẋ, ẏ = 1.0, 0.0
        d_dual = frule!!(zero_dual(f), Dual(x, ẋ), Dual(y, ẏ))
        d_ndual = frule!!(
            zero_dual(f), NDual{Float64,1}(x, (ẋ,)), NDual{Float64,1}(y, (ẏ,))
        )
        @test d_ndual.value ≈ primal(d_dual)
        @test d_ndual.partials[1] ≈ tangent(d_dual)
    end

    @testset "binary mixed NDual×Dual: $f" for f in (
        Mooncake.IntrinsicsWrappers.add_float, Mooncake.IntrinsicsWrappers.mul_float
    )
        x, y = 1.5, 2.5
        d = frule!!(zero_dual(f), NDual{Float64,1}(x, (1.0,)), Dual(y, 0.0))
        @test d isa NDual{Float64,1}
    end

    @testset "ternary intrinsic: $f" for f in (
        Mooncake.IntrinsicsWrappers.fma_float, Mooncake.IntrinsicsWrappers.muladd_float
    )
        x, y, z = 1.5, 2.5, 0.5
        d = frule!!(
            zero_dual(f),
            NDual{Float64,1}(x, (1.0,)),
            NDual{Float64,1}(y, (0.0,)),
            NDual{Float64,1}(z, (0.0,)),
        )
        @test d isa NDual{Float64,1}
        @test d.value ≈ f(x, y, z)
    end
end
