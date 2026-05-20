# Regression tests for the NDual container dispatch helpers in
# `src/nfwd/NfwdMooncake.jl`. These guard against the silent-fallthrough failure
# mode where a missing container overload causes FCache to drop to width-1.
#
# Scope:
#   1. NDual container dispatch helpers (`_has_ndual`, `_ndual_width`,
#      `_dual_or_ndual`) over each supported container shape.
#   2. Width-aware seed constructors (`zero_dual`, `uninit_dual`, `randn_dual`)
#      and their static return-type contract against `dual_type(Val(N), P)`.
#   3. NDual frule!! direct sweep — `test_rule` lifts via no-`Val` `dual_type(P)`
#      (Dual-only), so the per-rule NDual overloads can only be reached through
#      `prepare_derivative_cache(...; chunk_size=N)` end-to-end tests or direct
#      `frule!!` calls. The sweep here is the direct-dispatch counterpart.
#
# Out of scope (covered elsewhere): end-to-end chunked AD → `test/interface.jl`;
# Memory/Array NDual rule integration → `test/rules/memory.jl`; primal-mode
# interpreter sweep → `test/interpreter/primal_mode.jl`.

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

    # `test_rule` lifts inputs via no-`Val` `dual_type(P)` → `Dual{P,T}`, never
    # `NDual{T,N}`, so it cannot reach NDual frule!!s — those are dispatched
    # only through chunked `prepare_derivative_cache(...; chunk_size=N)`.
    # `ndual_test_rule` is the direct-dispatch counterpart: lift each
    # `(value, tangent)` to NDual{T,1} (or Dual for non-NDual args like
    # `Type{Pext}` in fpext), call `frule!!`, and compare against the parallel
    # Dual{P} reference. Tuple-output frules (sincosd / sincospi / modf) need
    # special unwrap because the NDual side returns `Tuple{NDual...}` while the
    # Dual side returns `Dual{Tuple,Tuple}`.
    @testset "NDual frule!! direct sweep" begin
        @inline _to_ndual_arg(v::T, t) where {T<:Base.IEEEFloat} = Mooncake.NDual{T,1}(
            v, (t,)
        )
        @inline function _to_ndual_arg(v::Complex{T}, t) where {T<:Base.IEEEFloat}
            return Complex(
                Mooncake.NDual{T,1}(real(v), (real(t),)),
                Mooncake.NDual{T,1}(imag(v), (imag(t),)),
            )
        end
        @inline _to_ndual_arg(v, t) = Mooncake.Dual(v, t)

        function _match_scalar(n::Mooncake.NDual, p, t; check_finite)
            @test n.value === p
            np = n.partials[1]
            if check_finite && isfinite(t) && isfinite(np)
                @test np ≈ t
            else
                @test isequal(np, t)
            end
        end

        function ndual_test_rule(f, vt_pairs::Tuple...; check_finite=true)
            fdual = Mooncake.zero_dual(f)
            d_dual = Mooncake.frule!!(fdual, map(p -> Mooncake.Dual(p...), vt_pairs)...)
            d_ndual = Mooncake.frule!!(fdual, map(p -> _to_ndual_arg(p...), vt_pairs)...)
            if d_ndual isa Tuple
                ps = Mooncake.primal(d_dual)
                ts = Mooncake.tangent(d_dual)
                @test length(d_ndual) == length(ps) == length(ts)
                for (n, p, t) in zip(d_ndual, ps, ts)
                    _match_scalar(n, p, t; check_finite)
                end
            else
                _match_scalar(
                    d_ndual, Mooncake.primal(d_dual), Mooncake.tangent(d_dual); check_finite
                )
            end
        end

        # Test cases: `(f, ((value, tangent), ...), check_finite)`.
        # `check_finite=false` for inputs that hit a derivative singularity
        # (asin/acos at ±1, asec/acsc at ±1, tan-family near π/2 etc.).
        intr = Mooncake.IntrinsicsWrappers
        cases = Any[]
        # Smooth unary primitives evaluated at 0.7 (well inside R+).
        for f in (
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
            push!(cases, (f, ((0.7, 1.0),), true))
        end
        for f in (sec, csc, cot, secd, cscd, cotd, csch, coth, atan)
            push!(cases, (f, ((1.0, 1.0),), true))
        end
        for f in (asin, acos, asech, atanh, asind, acosd)
            push!(cases, (f, ((0.5, 1.0),), false))
        end
        for f in (asec, acsc, acoth, asecd, acscd, acosh)
            push!(cases, (f, ((1.5, 1.0),), false))
        end
        for f in (tan, tand, acot, atand, acotd)
            push!(cases, (f, ((0.5, 1.0),), false))
        end
        for f in (mod2pi, acsch, asinh)
            push!(cases, (f, ((1.5, 1.0),), true))
        end
        for f in (atan, log, ^, mod, max, min)
            push!(cases, (f, ((1.5, 1.0), (2.5, 0.0)), true))
        end
        for f in (
            intr.abs_float,
            intr.neg_float,
            intr.neg_float_fast,
            intr.sqrt_llvm,
            intr.sqrt_llvm_fast,
        )
            push!(cases, (f, ((1.5, 1.0),), true))
        end
        for f in (
            intr.add_float,
            intr.sub_float,
            intr.mul_float,
            intr.div_float,
            intr.copysign_float,
        )
            push!(cases, (f, ((1.5, 1.0), (2.5, 0.0)), true))
        end
        for f in (intr.fma_float, intr.muladd_float)
            push!(cases, (f, ((1.5, 1.0), (2.5, 0.0), (0.5, 0.0)), true))
        end
        for f in (sincosd, sincospi, modf)
            push!(cases, (f, ((0.7, 1.0),), true))
        end
        # `Type{Pext}` first arg is NoTangent (non-NDual) — `_to_ndual_arg`
        # falls back to `Dual` for that arg, so both paths share it.
        push!(cases, (intr.fpext, ((Float64, NoTangent()), (1.5f0, 1.0f0)), true))
        push!(cases, (intr.fptrunc, ((Float32, NoTangent()), (1.5, 1.0)), true))
        for n in 1:3
            push!(cases, (hypot, ntuple(i -> (Float64(i + 1), Float64(i)), n), true))
        end
        push!(cases, (clamp, ((1.5, 1.0), (0.0, 0.0), (2.0, 0.0)), true))

        @testset "$f $(map(first, vt_pairs))" for (f, vt_pairs, check_finite) in cases
            ndual_test_rule(f, vt_pairs...; check_finite)
        end

        # Mixed NDual × Dual{<:IEEEFloat} — exercises the "scalar Dual on the
        # right" branch in the `$op_sym` loop in `rules_via_nfwd.jl`. No Dual
        # reference: result type alone confirms dispatch landed on the
        # NDual-aware overload rather than falling through.
        @testset "binary mixed $f" for f in (intr.add_float, intr.mul_float)
            d = Mooncake.frule!!(
                Mooncake.zero_dual(f),
                Mooncake.NDual{Float64,1}(1.5, (1.0,)),
                Mooncake.Dual(2.5, 0.0),
            )
            @test d isa Mooncake.NDual{Float64,1}
        end
    end
end
