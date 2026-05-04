# Regression tests for the NDual container dispatch helpers in
# `src/nfwd/NfwdMooncake.jl`. These guard against the silent-fallthrough failure
# mode where a missing container overload causes FCache to drop to width-1.

@testset "NfwdMooncake" begin
    @testset "NDual dispatch helpers" begin
        nd = Mooncake.NDual{Float64,3}(1.0, (1.0, 0.0, 0.0))
        cnd = Complex(nd, nd)
        and = [nd, nd]
        acnd = [cnd, cnd]
        containers = Any[
            ("NDual", nd, 3),
            ("Complex{NDual}", cnd, 3),
            ("Array{NDual}", and, 3),
            ("Array{Complex{NDual}}", acnd, 3),
        ]
        @static if VERSION >= v"1.11-rc4"
            mnd = Memory{Mooncake.NDual{Float64,3}}(undef, 2)
            fill!(mnd, nd)
            mcnd = Memory{Complex{Mooncake.NDual{Float64,3}}}(undef, 2)
            fill!(mcnd, cnd)
            push!(containers, ("Memory{NDual}", mnd, 3))
            push!(containers, ("Memory{Complex{NDual}}", mcnd, 3))
            push!(containers, ("MemoryRef{NDual}", Core.memoryrefnew(mnd), 3))
            push!(containers, ("MemoryRef{Complex{NDual}}", Core.memoryrefnew(mcnd), 3))
        end
        @testset "$name" for (name, c, W) in containers
            @test Mooncake._has_ndual(c)
            @test Mooncake._ndual_width(c) === Val(W)
        end

        # `_dual_or_ndual(val, NTangent(...))` must produce an NDual-bearing value for
        # each scalar/Complex/Memory primal that can appear as a struct field. Memory
        # lanes are themselves Memory{T}, mirroring the per-direction fdata layout.
        nt3 = Mooncake.NTangent((1.0, 0.0, 0.0))
        @test Mooncake._dual_or_ndual(1.0, nt3) isa Mooncake.NDual{Float64,3}
        complex_nt = Mooncake.NTangent((1.0 + 0im, 0.0 + 1im, 0.0 + 0im))
        @test Mooncake._dual_or_ndual(1.0 + 2im, complex_nt) isa
            Complex{<:Mooncake.NDual{Float64,3}}
        @static if VERSION >= v"1.11-rc4"
            mfloat = Memory{Float64}(undef, 2)
            fill!(mfloat, 1.0)
            mlane = Memory{Float64}(undef, 2)
            fill!(mlane, 0.5)
            mnt3 = Mooncake.NTangent((mlane, mlane, mlane))
            @test Mooncake._dual_or_ndual(mfloat, mnt3) isa
                Memory{<:Mooncake.NDual{Float64,3}}
        end
    end

    # Width-aware seed constructors (`zero_dual`, `uninit_dual`, `randn_dual`)
    # should return values whose static type matches `dual_type(Val(N), P)` for
    # every primal type covered by the NDual-aware `dual_type` table. Without
    # this, external callers cannot exercise the user-facing Dual-wrapped NDual
    # frule entry points at width N.
    @testset "Width-aware dual seed constructors" begin
        rng = StableRNG(42)
        # Type-equality: each constructor's output type must match dual_type.
        for (val, label) in Any[
            (1.0, "Float64"),
            (1.0f0, "Float32"),
            (1.0 + 2im, "ComplexF64"),
            ([1.0, 2.0], "Vector{Float64}"),
            ([1.0 2.0; 3.0 4.0], "Matrix{Float64}"),
            ([1.0 + 0im, 2.0 + 0im], "Vector{ComplexF64}"),
        ]
            P = typeof(val)
            @testset "$label" begin
                for N_val in (0, 1, 2, 4)
                    expected = Mooncake.dual_type(Val(N_val), P)
                    @test typeof(Mooncake.zero_dual(Val(N_val), val)) == expected
                    @test typeof(Mooncake.uninit_dual(Val(N_val), val)) == expected
                    @test typeof(Mooncake.randn_dual(Val(N_val), rng, val)) == expected
                end
            end
        end
        @static if VERSION >= v"1.11-rc4"
            m = Memory{Float64}(undef, 3)
            fill!(m, 1.0)
            mr = memoryref(m)
            cm = Memory{ComplexF64}(undef, 3)
            fill!(cm, 1.0 + 0im)
            for (val, label) in (
                (m, "Memory{Float64}"),
                (mr, "MemoryRef{Float64}"),
                (cm, "Memory{ComplexF64}"),
            )
                @testset "$label" begin
                    P = typeof(val)
                    for N_val in (0, 2)
                        expected = Mooncake.dual_type(Val(N_val), P)
                        @test typeof(Mooncake.zero_dual(Val(N_val), val)) == expected
                        @test typeof(Mooncake.uninit_dual(Val(N_val), val)) == expected
                        @test typeof(Mooncake.randn_dual(Val(N_val), rng, val)) == expected
                    end
                end
            end
        end

        # zero_dual / uninit_dual produce zero partials by construction.
        nd_zero = Mooncake.zero_dual(Val(3), 5.0)
        @test nd_zero.value == 5.0
        @test nd_zero.partials == (0.0, 0.0, 0.0)
        nd_uninit = Mooncake.uninit_dual(Val(3), 5.0)
        @test nd_uninit.partials == (0.0, 0.0, 0.0)

        # randn_dual partials should actually use the RNG (sanity check that the
        # generator threads through correctly; not a stochastic distribution test).
        nd_randn = Mooncake.randn_dual(Val(3), StableRNG(0), 1.0)
        @test all(!iszero, nd_randn.partials)

        # Val(0) is the primal passthrough — round-trips identically.
        @test Mooncake.zero_dual(Val(0), 1.0) === 1.0
        @test Mooncake.uninit_dual(Val(0), 1.0) === 1.0
        @test Mooncake.randn_dual(Val(0), rng, 1.0) === 1.0

        # The Dual-wrapped NDual frule entry point in `memory.jl` is reachable
        # from external callers via this width-aware constructor.
        @static if VERSION >= v"1.11-rc4"
            md = Mooncake.zero_dual(Val(2), Memory{Float64}(undef, 3))
            @test md isa Memory{<:Mooncake.NDual{Float64,2}}
        end
    end

    # Direct unit tests for the NDual frule!!s. Existing tests reach them only
    # indirectly via chunked `prepare_derivative_cache(...; chunk_size=N)`, so
    # a missing or broken NDual overload (e.g. for one of the ~50 functions in
    # the `rules_via_nfwd.jl` unary loop) often surfaces as a confusing failure
    # several layers up. Each case here calls the NDual `frule!!` and compares
    # against the known-good `Dual{P}` reference at the same `(x, ẋ)` pair.
    @testset "NDual frule!! direct sweep" begin
        function check_unary(f, x; check_finite=true)
            ẋ = one(typeof(x))
            d_dual = Mooncake.frule!!(Mooncake.zero_dual(f), Mooncake.Dual(x, ẋ))
            d_ndual = Mooncake.frule!!(
                Mooncake.zero_dual(f), Mooncake.NDual{typeof(x),1}(x, (ẋ,))
            )
            @test d_ndual isa Mooncake.NDual{typeof(x),1}
            @test d_ndual.value === Mooncake.primal(d_dual)
            d_tangent = Mooncake.tangent(d_dual)
            if check_finite && isfinite(d_tangent) && isfinite(d_ndual.partials[1])
                @test d_ndual.partials[1] ≈ d_tangent
            else
                # NaN/Inf: just check both sides agree.
                @test isequal(d_ndual.partials[1], d_tangent)
            end
        end

        function check_binary(f, x, y; ẋ=one(typeof(x)), ẏ=zero(typeof(y)))
            d_dual = Mooncake.frule!!(
                Mooncake.zero_dual(f), Mooncake.Dual(x, ẋ), Mooncake.Dual(y, ẏ)
            )
            d_ndual = Mooncake.frule!!(
                Mooncake.zero_dual(f),
                Mooncake.NDual{typeof(x),1}(x, (ẋ,)),
                Mooncake.NDual{typeof(y),1}(y, (ẏ,)),
            )
            @test d_ndual isa Mooncake.NDual
            @test d_ndual.value === Mooncake.primal(d_dual)
            d_tangent = Mooncake.tangent(d_dual)
            if isfinite(d_tangent) && isfinite(d_ndual.partials[1])
                @test d_ndual.partials[1] ≈ d_tangent
            else
                @test isequal(d_ndual.partials[1], d_tangent)
            end
        end

        @testset "unary: $f" for f in (
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
        @testset "unary at 1.0: $f" for f in
                                        (sec, csc, cot, secd, cscd, cotd, csch, coth, atan)
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
            x32 = 1.5f0
            d = Mooncake.frule!!(
                Mooncake.zero_dual(Mooncake.IntrinsicsWrappers.fpext),
                Mooncake.Dual(Float64, NoTangent()),
                Mooncake.NDual{Float32,1}(x32, (1.0f0,)),
            )
            @test d isa Mooncake.NDual{Float64,1}
            @test d.value === Float64(x32)
            @test d.partials[1] === 1.0
            x64 = 1.5
            d2 = Mooncake.frule!!(
                Mooncake.zero_dual(Mooncake.IntrinsicsWrappers.fptrunc),
                Mooncake.Dual(Float32, NoTangent()),
                Mooncake.NDual{Float64,1}(x64, (1.0,)),
            )
            @test d2 isa Mooncake.NDual{Float32,1}
            @test d2.value === Float32(x64)
            @test d2.partials[1] === 1.0f0
        end

        @testset "tuple-output: $f" for f in (sincosd, sincospi, modf)
            x = 0.7
            d_ndual = Mooncake.frule!!(
                Mooncake.zero_dual(f), Mooncake.NDual{Float64,1}(x, (1.0,))
            )
            @test d_ndual isa Tuple
            @test length(d_ndual) == 2
        end

        @testset "vararg hypot $n-arg" for n in 1:3
            xs = ntuple(i -> Mooncake.NDual{Float64,1}(Float64(i + 1), (Float64(i),)), n)
            d = Mooncake.frule!!(Mooncake.zero_dual(hypot), xs...)
            @test d isa Mooncake.NDual{Float64,1}
            @test d.value ≈ hypot(Float64.(2:(n + 1))...)
        end

        @testset "ternary clamp" begin
            d = Mooncake.frule!!(
                Mooncake.zero_dual(clamp),
                Mooncake.NDual{Float64,1}(1.5, (1.0,)),
                Mooncake.NDual{Float64,1}(0.0, (0.0,)),
                Mooncake.NDual{Float64,1}(2.0, (0.0,)),
            )
            @test d isa Mooncake.NDual{Float64,1}
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
            d_dual = Mooncake.frule!!(Mooncake.zero_dual(f), Mooncake.Dual(x, 1.0))
            d_ndual = Mooncake.frule!!(
                Mooncake.zero_dual(f), Mooncake.NDual{Float64,1}(x, (1.0,))
            )
            @test d_ndual.value ≈ Mooncake.primal(d_dual)
            @test d_ndual.partials[1] ≈ Mooncake.tangent(d_dual)
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
            d_dual = Mooncake.frule!!(
                Mooncake.zero_dual(f), Mooncake.Dual(x, ẋ), Mooncake.Dual(y, ẏ)
            )
            d_ndual = Mooncake.frule!!(
                Mooncake.zero_dual(f),
                Mooncake.NDual{Float64,1}(x, (ẋ,)),
                Mooncake.NDual{Float64,1}(y, (ẏ,)),
            )
            @test d_ndual.value ≈ Mooncake.primal(d_dual)
            @test d_ndual.partials[1] ≈ Mooncake.tangent(d_dual)
        end

        @testset "binary mixed NDual×Dual: $f" for f in (
            Mooncake.IntrinsicsWrappers.add_float, Mooncake.IntrinsicsWrappers.mul_float
        )
            x, y = 1.5, 2.5
            d = Mooncake.frule!!(
                Mooncake.zero_dual(f),
                Mooncake.NDual{Float64,1}(x, (1.0,)),
                Mooncake.Dual(y, 0.0),
            )
            @test d isa Mooncake.NDual{Float64,1}
        end

        @testset "ternary intrinsic: $f" for f in (
            Mooncake.IntrinsicsWrappers.fma_float, Mooncake.IntrinsicsWrappers.muladd_float
        )
            x, y, z = 1.5, 2.5, 0.5
            d = Mooncake.frule!!(
                Mooncake.zero_dual(f),
                Mooncake.NDual{Float64,1}(x, (1.0,)),
                Mooncake.NDual{Float64,1}(y, (0.0,)),
                Mooncake.NDual{Float64,1}(z, (0.0,)),
            )
            @test d isa Mooncake.NDual{Float64,1}
            @test d.value ≈ f(x, y, z)
        end
    end
end
