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
end
