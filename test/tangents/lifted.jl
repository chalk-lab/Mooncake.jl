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
            Mooncake.NDualArray{Float64,2,1,Vector{Float64},Mooncake.NDual{Float64,2}}
        @test Mooncake.dual_type(Val(1), Matrix{Float32}) ===
            Mooncake.NDualArray{Float32,1,2,Matrix{Float32},Mooncake.NDual{Float32,1}}
        @test Mooncake.lifted_type(Val(2), Vector{Float64}) === Mooncake.Lifted{
            Vector{Float64},
            2,
            Mooncake.NDualArray{Float64,2,1,Vector{Float64},Mooncake.NDual{Float64,2}},
        }
    end

    @testset "seed factories (Array{T<:IEEEFloat, D})" begin
        x = [1.0, 2.0, 3.0]

        v = Mooncake.zero_dual(Val(2), x)
        @test typeof(v) ===
            Mooncake.NDualArray{Float64,2,1,Vector{Float64},Mooncake.NDual{Float64,2}}
        @test Mooncake.primal(v) === x  # primal aliases user storage.
        @test all(iszero, v.partials[1]) && all(iszero, v.partials[2])

        # Lazy element access reconstructs an NDual on the fly.
        d = v[1]
        @test typeof(d) === Mooncake.NDual{Float64,2}
        @test d.value === 1.0
        @test d.partials === (0.0, 0.0)

        rng = Random.MersenneTwister(0)
        r = Mooncake.randn_dual(Val(2), rng, x)
        @test typeof(r) === typeof(v)
        @test Mooncake.primal(r) === x
        @test any(!iszero, r.partials[1])

        # Layer-3 wrapped slot.
        z = Mooncake.zero_lifted(Val(2), x)
        @test typeof(z) === Mooncake.lifted_type(Val(2), Vector{Float64})
        @test Mooncake.primal(z) === x  # primal aliases user storage.
        @test Mooncake.tangent(z) === v || Mooncake.tangent(z) isa Mooncake.NDualArray

        # Matrix shape (D = 2).
        M = [1.0 2.0; 3.0 4.0]
        zM = Mooncake.zero_lifted(Val(2), M)
        @test typeof(zM) === Mooncake.lifted_type(Val(2), Matrix{Float64})
        @test size(Mooncake.tangent(zM)) == size(M)
        @test all(iszero, Mooncake.tangent(zM).partials[1])
    end

    @testset "NDualArray accessors + AbstractArray interface" begin
        N = 2
        T = Float64
        x = [1.0, 2.0, 3.0]
        a = Mooncake.NDualArray{T,N,1,Vector{T}}(
            x, (similar(x).=[0.5, -0.5, 1.5], similar(x).=[0.0, 1.0, -1.0])
        )
        @test a isa AbstractArray{Mooncake.NDual{T,N},1}
        @test size(a) == (3,)
        @test length(a) == 3
        @test Mooncake.primal(a) === x
        @test Mooncake.tangent(a) === Mooncake.NTangent(a.partials)
        @test Mooncake.unpack_ndual(a) === (a.primal, a.partials)

        # Lazy getindex.
        d = a[2]
        @test typeof(d) === Mooncake.NDual{T,N}
        @test d.value === 2.0
        @test d.partials === (-0.5, 1.0)

        # setindex! writes both channels.
        Mooncake.setindex!(a, Mooncake.NDual{T,N}(9.0, (7.0, -7.0)), 1)
        @test x[1] === 9.0
        @test a.partials[1][1] === 7.0
        @test a.partials[2][1] === -7.0
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "NDualMemoryRef (MemoryRef{T<:IEEEFloat})" begin
            T = Float64
            N = 2
            mem = Memory{T}(undef, 3)
            mem[1] = 1.0
            mem[2] = 2.0
            mem[3] = 3.0
            p = Core.memoryref(mem, 1)

            # Type-level query.
            @test Mooncake.dual_type(Val(N), MemoryRef{T}) ===
                Mooncake.NDualMemoryRef{T,N,Memory{T}}
            @test Mooncake.lifted_type(Val(N), MemoryRef{T}) ===
                Mooncake.Lifted{MemoryRef{T},N,Mooncake.NDualMemoryRef{T,N,Memory{T}}}

            # Seed factory: zero-init partials at the same offset, slot-local.
            a = Mooncake.zero_dual(Val(N), p)
            @test typeof(a) === Mooncake.NDualMemoryRef{T,N,Memory{T}}
            @test Mooncake.primal(a) === p  # aliases user storage.
            @test Mooncake.tangent(a) === Mooncake.NTangent(a.partials)
            @test Mooncake.unpack_ndual(a) === (a.primal, a.partials)
            @test all(iszero, a.partials[1].mem)
            @test all(iszero, a.partials[2].mem)

            # Element access via _memoryrefget_ndual.
            d = Mooncake._memoryrefget_ndual(a, :not_atomic, false)
            @test typeof(d) === Mooncake.NDual{T,N}
            @test d.value === 1.0
            @test d.partials === (0.0, 0.0)

            # Layer-3 wrapped slot.
            z = Mooncake.zero_lifted(Val(N), p)
            @test typeof(z) === Mooncake.lifted_type(Val(N), MemoryRef{T})
            @test Mooncake.primal(z) === p
        end
    end

    @testset "dual_type / lifted_type (Array{Complex{R}, D})" begin
        @test Mooncake.dual_type(Val(2), Vector{Complex{Float64}}) === Mooncake.NDualArray{
            Complex{Float64},2,1,Vector{Complex{Float64}},Complex{Mooncake.NDual{Float64,2}}
        }
        @test Mooncake.lifted_type(Val(2), Vector{Complex{Float64}}) === Mooncake.Lifted{
            Vector{Complex{Float64}},
            2,
            Mooncake.NDualArray{
                Complex{Float64},
                2,
                1,
                Vector{Complex{Float64}},
                Complex{Mooncake.NDual{Float64,2}},
            },
        }

        # Seed factory.
        x = Complex{Float64}[1.0 + 2.0im, 3.0 - 1.0im]
        v = Mooncake.zero_dual(Val(2), x)
        @test typeof(v) === Mooncake.dual_type(Val(2), Vector{Complex{Float64}})
        @test Mooncake.primal(v) === x
        @test all(iszero, v.partials[1]) && all(iszero, v.partials[2])

        z = Mooncake.zero_lifted(Val(2), x)
        @test typeof(z) === Mooncake.lifted_type(Val(2), Vector{Complex{Float64}})
        @test Mooncake.primal(z) === x
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
        # Nested: a tuple of (scalar, array). Array V is the SoA NDualArray.
        @test Mooncake.dual_type(Val(2), Tuple{Float64,Vector{Float64}}) === Tuple{
            Mooncake.NDual{Float64,2},
            Mooncake.NDualArray{Float64,2,1,Vector{Float64},Mooncake.NDual{Float64,2}},
        }
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
            Mooncake.NDualArray{Float64,2,1,Vector{Float64},Mooncake.NDual{Float64,2}},
        }
        @test v[1].value === 1.0 && v[1].partials == (0.0, 0.0)
        @test v[2].value === 2.0f0 && v[2].partials == (0.0f0, 0.0f0)
        @test Mooncake.primal(v[3]) === x[3]

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
            (:x, :y),
            Tuple{
                Mooncake.NDual{Float64,2},
                Mooncake.NDualArray{Float64,2,1,Vector{Float64},Mooncake.NDual{Float64,2}},
            },
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
        @test v.c isa Mooncake.NDualArray
        @test Mooncake.primal(v.c) === x.c

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

    @testset "MutableDualTangentView (NDual field)" begin
        N = 2
        r = LiftedTest_RefF(3.0)
        slot = Mooncake.zero_lifted(Val(N), r)

        # Per-lane view via `tangent(::Lifted, lane)`.
        view = Mooncake.tangent(slot, 1)
        @test view isa Mooncake.MutableDualTangentView
        @test getfield(view, :parent) === slot.value
        @test getfield(view, :primal) === r
        @test getfield(view, :lane) === 1

        # Read: getproperty returns the lane-1 partial of the `v` field.
        @test view.v === 0.0

        # Write: setproperty! routes back to parent.value via setfield!.
        view.v = 5.0
        @test view.v === 5.0
        @test slot.value.value.v.partials === (5.0, 0.0)
        # Other lane unchanged.
        view2 = Mooncake.tangent(slot, 2)
        @test view2.v === 0.0
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
