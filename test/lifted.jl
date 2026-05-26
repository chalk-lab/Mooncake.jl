struct LiftedTest_Point
    x::Float64
    y::Float64
end
mutable struct LiftedTest_RefF
    v::Float64
end

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

    @testset "dual_type / lifted_type (Array{T<:IEEEFloat, D})" begin
        @test Mooncake.dual_type(Val(2), Vector{Float64}) ===
            Vector{Mooncake.NDual{Float64,2}}
        @test Mooncake.dual_type(Val(1), Matrix{Float32}) ===
            Matrix{Mooncake.NDual{Float32,1}}
        @test Mooncake.lifted_type(Val(2), Vector{Float64}) ===
            Mooncake.Lifted{Vector{Float64},2,Vector{Mooncake.NDual{Float64,2}}}
        @test Mooncake.lifted_type(Val(3), Array{Float64,3}) ===
            Mooncake.Lifted{Array{Float64,3},3,Array{Mooncake.NDual{Float64,3},3}}
    end

    @testset "seed factories (Array{T<:IEEEFloat, D})" begin
        x = [1.0, 2.0, 3.0]

        v = Mooncake.zero_dual(Val(2), x)
        @test typeof(v) === Vector{Mooncake.NDual{Float64,2}}
        @test [d.value for d in v] == x
        @test all(d -> d.partials == (0.0, 0.0), v)

        u = Mooncake.uninit_dual(Val(3), x)
        @test typeof(u) === Vector{Mooncake.NDual{Float64,3}}
        @test [d.value for d in u] == x

        rng = Random.MersenneTwister(0)
        r = Mooncake.randn_dual(Val(2), rng, x)
        @test typeof(r) === Vector{Mooncake.NDual{Float64,2}}
        @test [d.value for d in r] == x
        @test any(d -> any(!iszero, d.partials), r)

        # Layer-3 wrapped slot.
        z = Mooncake.zero_lifted(Val(2), x)
        @test typeof(z) ===
            Mooncake.Lifted{Vector{Float64},2,Vector{Mooncake.NDual{Float64,2}}}
        @test Mooncake.primal(z) === x  # primal aliases user storage.
        @test Mooncake.tangent(z) == v
        @test Mooncake.tangent(z) !== x  # tangent is slot-local, not aliased.

        # Matrix shape (D = 2).
        M = [1.0 2.0; 3.0 4.0]
        zM = Mooncake.zero_lifted(Val(2), M)
        @test typeof(zM) ===
            Mooncake.Lifted{Matrix{Float64},2,Matrix{Mooncake.NDual{Float64,2}}}
        @test size(Mooncake.tangent(zM)) == size(M)
        @test all(d -> d.partials == (0.0, 0.0), Mooncake.tangent(zM))
    end

    @testset "dual_type / lifted_type (Complex{<:IEEEFloat})" begin
        @test Mooncake.dual_type(Val(2), Complex{Float64}) ===
            Complex{Mooncake.NDual{Float64,2}}
        @test Mooncake.dual_type(Val(1), Complex{Float32}) ===
            Complex{Mooncake.NDual{Float32,1}}
        @test Mooncake.lifted_type(Val(2), Complex{Float64}) ===
            Mooncake.Lifted{Complex{Float64},2,Complex{Mooncake.NDual{Float64,2}}}
    end

    @testset "frule!! one-to-one parallels (complex.jl)" begin
        N = 2
        P = Float64
        # Build a Complex{NDual{P,N}} inner V manually for the input slot.
        re_inner = Mooncake.NDual{P,N}(1.5, (1.0, 0.0))
        im_inner = Mooncake.NDual{P,N}(-0.5, (0.0, 1.0))
        z_primal = Complex{P}(1.5, -0.5)
        z_inner = Complex{Mooncake.NDual{P,N}}(re_inner, im_inner)
        z = Mooncake.Lifted{Complex{P},N}(z_primal, z_inner)

        # lgetfield :re
        f_slot = Mooncake.Lifted{typeof(Mooncake.lgetfield),N}(
            Mooncake.lgetfield, Mooncake.NoTangent()
        )
        val_re = Mooncake.Lifted{Val{:re},N}(Val(:re), Mooncake.NoTangent())
        r = Mooncake.frule!!(f_slot, z, val_re)
        @test typeof(r) === Mooncake.Lifted{P,N,Mooncake.NDual{P,N}}
        @test Mooncake.primal(r) == 1.5
        @test Mooncake.tangent(r) === re_inner

        # lgetfield :im
        val_im = Mooncake.Lifted{Val{:im},N}(Val(:im), Mooncake.NoTangent())
        r2 = Mooncake.frule!!(f_slot, z, val_im)
        @test typeof(r2) === Mooncake.Lifted{P,N,Mooncake.NDual{P,N}}
        @test Mooncake.primal(r2) == -0.5
        @test Mooncake.tangent(r2) === im_inner

        # _new_(Complex{P}, re, im)
        new_slot = Mooncake.Lifted{typeof(Mooncake._new_),N}(
            Mooncake._new_, Mooncake.NoTangent()
        )
        type_slot = Mooncake.Lifted{Type{Complex{P}},N}(Complex{P}, Mooncake.NoTangent())
        re_slot = Mooncake.Lifted{P,N}(1.5, re_inner)
        im_slot = Mooncake.Lifted{P,N}(-0.5, im_inner)
        r3 = Mooncake.frule!!(new_slot, type_slot, re_slot, im_slot)
        @test typeof(r3) === Mooncake.Lifted{Complex{P},N,Complex{Mooncake.NDual{P,N}}}
        @test Mooncake.primal(r3) == Complex{P}(1.5, -0.5)
        @test Mooncake.tangent(r3) === z_inner
    end

    @testset "frule!! one-to-one parallels (performance_patches.jl)" begin
        N = 2
        P = Float64
        x = [1.0, 2.0, 3.0]

        # Build the slot with a non-trivial tangent so partial computations are tested.
        x_inner = [
            Mooncake.NDual{P,N}(1.0, (1.0, 0.5)),
            Mooncake.NDual{P,N}(2.0, (0.0, 1.0)),
            Mooncake.NDual{P,N}(3.0, (-0.5, 0.0)),
        ]
        x_slot = Mooncake.Lifted{Vector{P},N}(x, x_inner)

        # sum(::Array)
        sum_slot = Mooncake.Lifted{typeof(sum),N}(sum, Mooncake.NoTangent())
        r = Mooncake.frule!!(sum_slot, x_slot)
        @test typeof(r) === Mooncake.Lifted{P,N,Mooncake.NDual{P,N}}
        @test Mooncake.primal(r) == sum(x)
        # Tangent's partials should be element-wise sums of x_inner partials.
        rt = Mooncake.tangent(r)
        @test rt.value == sum(x)
        @test rt.partials == (1.0 + 0.0 + -0.5, 0.5 + 1.0 + 0.0)

        # sum(abs2, ::Array)
        abs2_slot = Mooncake.Lifted{typeof(abs2),N}(abs2, Mooncake.NoTangent())
        r2 = Mooncake.frule!!(sum_slot, abs2_slot, x_slot)
        @test typeof(r2) === Mooncake.Lifted{P,N,Mooncake.NDual{P,N}}
        @test Mooncake.primal(r2) == sum(abs2, x)
        # Forward derivative of sum(abs2, x) along direction d is 2*dot(x, d).
        # For lane 1, d=[1.0, 0.0, -0.5]; for lane 2, d=[0.5, 1.0, 0.0].
        rt2 = Mooncake.tangent(r2)
        @test rt2.value == sum(abs2, x)
        @test rt2.partials[1] ≈ 2 * (1.0 * 1.0 + 2.0 * 0.0 + 3.0 * -0.5)
        @test rt2.partials[2] ≈ 2 * (1.0 * 0.5 + 2.0 * 1.0 + 3.0 * 0.0)
    end

    @testset "dual_type / lifted_type (Tuple)" begin
        @test Mooncake.dual_type(Val(2), Tuple{}) === Tuple{}
        @test Mooncake.dual_type(Val(2), Tuple{Float64}) ===
            Tuple{Mooncake.NDual{Float64,2}}
        @test Mooncake.dual_type(Val(2), Tuple{Float64,Float32}) ===
            Tuple{Mooncake.NDual{Float64,2},Mooncake.NDual{Float32,2}}
        # Nested: a tuple of (scalar, array).
        @test Mooncake.dual_type(Val(2), Tuple{Float64,Vector{Float64}}) ===
            Tuple{Mooncake.NDual{Float64,2},Vector{Mooncake.NDual{Float64,2}}}
        @test Mooncake.lifted_type(Val(2), Tuple{Float64,Float64}) === Mooncake.Lifted{
            Tuple{Float64,Float64},
            2,
            Tuple{Mooncake.NDual{Float64,2},Mooncake.NDual{Float64,2}},
        }
    end

    @testset "seed factories (Tuple)" begin
        x = (1.0, 2.0f0, [3.0, 4.0])

        v = Mooncake.zero_dual(Val(2), x)
        @test typeof(v) === Tuple{
            Mooncake.NDual{Float64,2},
            Mooncake.NDual{Float32,2},
            Vector{Mooncake.NDual{Float64,2}},
        }
        @test v[1].value === 1.0 && v[1].partials == (0.0, 0.0)
        @test v[2].value === 2.0f0 && v[2].partials == (0.0f0, 0.0f0)
        @test [d.value for d in v[3]] == [3.0, 4.0]

        z = Mooncake.zero_lifted(Val(2), x)
        @test typeof(z) === Mooncake.Lifted{typeof(x),2,typeof(v)}
        @test Mooncake.primal(z) === x  # outer tuple aliases user storage.
        @test Mooncake.tangent(z) == v
    end

    @testset "dual_type / lifted_type (NamedTuple)" begin
        NT_ab = NamedTuple{(:a, :b),Tuple{Float64,Float32}}
        @test Mooncake.dual_type(Val(2), NT_ab) === NamedTuple{
            (:a, :b),Tuple{Mooncake.NDual{Float64,2},Mooncake.NDual{Float32,2}}
        }
        NT_xy = NamedTuple{(:x, :y),Tuple{Float64,Vector{Float64}}}
        @test Mooncake.dual_type(Val(2), NT_xy) === NamedTuple{
            (:x, :y),Tuple{Mooncake.NDual{Float64,2},Vector{Mooncake.NDual{Float64,2}}}
        }
        @test Mooncake.lifted_type(Val(2), NT_ab) === Mooncake.Lifted{
            NT_ab,
            2,
            NamedTuple{(:a, :b),Tuple{Mooncake.NDual{Float64,2},Mooncake.NDual{Float32,2}}},
        }
    end

    @testset "seed factories (NamedTuple)" begin
        x = (; a=1.0, b=2.0f0, c=[3.0, 4.0])

        v = Mooncake.zero_dual(Val(2), x)
        @test v isa NamedTuple{(:a, :b, :c)}
        @test v.a.value === 1.0 && v.a.partials == (0.0, 0.0)
        @test v.b.value === 2.0f0
        @test [d.value for d in v.c] == [3.0, 4.0]

        z = Mooncake.zero_lifted(Val(2), x)
        @test typeof(z) === Mooncake.Lifted{typeof(x),2,typeof(v)}
        @test Mooncake.primal(z) === x
        @test Mooncake.tangent(z) == v
    end

    @testset "seed factories (Complex{<:IEEEFloat})" begin
        z = 1.5 + (-0.5)im

        v = Mooncake.zero_dual(Val(2), z)
        @test typeof(v) === Complex{Mooncake.NDual{Float64,2}}
        @test real(v).value === 1.5 && real(v).partials == (0.0, 0.0)
        @test imag(v).value === -0.5 && imag(v).partials == (0.0, 0.0)

        zl = Mooncake.zero_lifted(Val(2), z)
        @test typeof(zl) ===
            Mooncake.Lifted{Complex{Float64},2,Complex{Mooncake.NDual{Float64,2}}}
        @test Mooncake.primal(zl) === z
        @test Mooncake.tangent(zl) === v
    end

    @testset "seed factories (concrete struct lift)" begin
        p = LiftedTest_Point(1.0, 2.0)
        v = Mooncake.zero_dual(Val(2), p)
        @test typeof(v) === Mooncake.dual_type(Val(2), LiftedTest_Point)
        @test v isa Mooncake.ImmutableDual
        @test v.value.x.value === 1.0 && v.value.x.partials == (0.0, 0.0)
        @test v.value.y.value === 2.0 && v.value.y.partials == (0.0, 0.0)

        zl = Mooncake.zero_lifted(Val(2), p)
        @test typeof(zl) === Mooncake.lifted_type(Val(2), LiftedTest_Point)
        @test Mooncake.primal(zl) === p
        @test Mooncake.tangent(zl) === v

        # Mutable struct lift.
        r = LiftedTest_RefF(3.0)
        vm = Mooncake.zero_dual(Val(2), r)
        @test typeof(vm) === Mooncake.dual_type(Val(2), LiftedTest_RefF)
        @test vm isa Mooncake.MutableDual
        @test vm.value.v.value === 3.0 && vm.value.v.partials == (0.0, 0.0)
    end

    @testset "dual_type / lifted_type (concrete struct lift)" begin
        # Immutable struct → ImmutableDual{NamedTuple{...}}.
        P_imm = LiftedTest_Point
        @test Mooncake.dual_type(Val(2), P_imm) === Mooncake.ImmutableDual{
            NamedTuple{(:x, :y),Tuple{Mooncake.NDual{Float64,2},Mooncake.NDual{Float64,2}}}
        }
        @test Mooncake.lifted_type(Val(2), P_imm) ===
            Mooncake.Lifted{P_imm,2,Mooncake.dual_type(Val(2), P_imm)}

        # Mutable struct → MutableDual{NamedTuple{...}}.
        P_mut = LiftedTest_RefF
        @test Mooncake.dual_type(Val(3), P_mut) ===
            Mooncake.MutableDual{NamedTuple{(:v,),Tuple{Mooncake.NDual{Float64,3}}}}
        @test Mooncake.lifted_type(Val(3), P_mut) ===
            Mooncake.Lifted{P_mut,3,Mooncake.dual_type(Val(3), P_mut)}
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
