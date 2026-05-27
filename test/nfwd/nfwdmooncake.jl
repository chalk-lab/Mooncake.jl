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
end
