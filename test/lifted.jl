@testset "lifted" begin
    @testset "Lifted struct + accessors" begin
        inner = Mooncake.NDual{Float64,2}(3.0, (1.0, -1.0))
        slot = Mooncake.Lifted{Float64,2}(3.0, inner)
        @test typeof(slot) === Mooncake.Lifted{Float64,2,Mooncake.NDual{Float64,2}}
        @test Mooncake.primal(slot) === 3.0
        @test Mooncake.tangent(slot) === inner
        @test Mooncake.extract(slot) === (3.0, inner)
    end

    @testset "equality + copy" begin
        inner = Mooncake.NDual{Float64,1}(2.0, (0.5,))
        a = Mooncake.Lifted{Float64,1}(2.0, inner)
        b = Mooncake.Lifted{Float64,1}(2.0, inner)
        @test a == b
        @test copy(a) == a
    end

    @testset "dual_type / lifted_type (IEEEFloat scalars)" begin
        @test Mooncake.dual_type(Val(1), Float64) === Mooncake.NDual{Float64,1}
        @test Mooncake.dual_type(Val(3), Float32) === Mooncake.NDual{Float32,3}
        @test Mooncake.lifted_type(Val(1), Float64) ===
            Mooncake.Lifted{Float64,1,Mooncake.NDual{Float64,1}}
        @test Mooncake.lifted_type(Val(4), Float64) ===
            Mooncake.Lifted{Float64,4,Mooncake.NDual{Float64,4}}
    end

    @testset "seed factories (IEEEFloat scalars)" begin
        # Layer-2 bare inner V.
        v = Mooncake.zero_dual(Val(2), 7.0)
        @test typeof(v) === Mooncake.NDual{Float64,2}
        @test v.value == 7.0
        @test v.partials == (0.0, 0.0)

        u = Mooncake.uninit_dual(Val(3), 1.0f0)
        @test typeof(u) === Mooncake.NDual{Float32,3}
        @test u.value === 1.0f0

        rng = Random.MersenneTwister(42)
        r = Mooncake.randn_dual(Val(2), rng, 0.0)
        @test typeof(r) === Mooncake.NDual{Float64,2}
        @test r.value == 0.0
        # Partials are random; just check they're not both zero (with prob ~1).
        @test any(!iszero, r.partials)

        # Layer-3 wrapped Lifted slots.
        z = Mooncake.zero_lifted(Val(2), 7.0)
        @test typeof(z) === Mooncake.Lifted{Float64,2,Mooncake.NDual{Float64,2}}
        @test Mooncake.primal(z) === 7.0
        @test Mooncake.tangent(z) === v

        ul = Mooncake.uninit_lifted(Val(3), 1.0f0)
        @test typeof(ul) === Mooncake.Lifted{Float32,3,Mooncake.NDual{Float32,3}}

        rng2 = Random.MersenneTwister(42)
        rl = Mooncake.randn_lifted(Val(2), rng2, 0.0)
        @test typeof(rl) === Mooncake.Lifted{Float64,2,Mooncake.NDual{Float64,2}}
        @test Mooncake.tangent(rl) == r  # Same rng + width.
    end

    @testset "type-stability" begin
        # The canonical width-N path is type-stable for IEEEFloat primals.
        @test @inferred(Mooncake.zero_dual(Val(2), 1.0)) isa Mooncake.NDual{Float64,2}
        @test @inferred(Mooncake.zero_lifted(Val(2), 1.0)) isa
            Mooncake.Lifted{Float64,2,Mooncake.NDual{Float64,2}}
        @test @inferred(Mooncake.dual_type(Val(2), Float64)) === Mooncake.NDual{Float64,2}
        @test @inferred(Mooncake.lifted_type(Val(2), Float64)) ===
            Mooncake.Lifted{Float64,2,Mooncake.NDual{Float64,2}}
    end
end
