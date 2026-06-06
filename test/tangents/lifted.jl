struct LiftedTest_Point
    x::Float64
    y::Float64
end
mutable struct LiftedTest_RefF
    v::Float64
end
mutable struct LiftedTest_Cycle
    next::Any
    w::Float64
end
mutable struct LiftedTest_Aliased
    a::Vector{Float64}
    b::Vector{Float64}
end
mutable struct LiftedTest_MaybeInit
    x::Float64
    y::Float64
    LiftedTest_MaybeInit(x::Float64) = new(x)
end

@testset "lifted" begin
    @testset "cyclic MutableDual tangent arithmetic" begin
        # A self-referential mutable struct lifts to a cyclic `MutableDual` V; the
        # tangent-arithmetic helpers must terminate via their aliasing caches.
        n = LiftedTest_Cycle(nothing, 2.0)
        n.next = n
        v = Mooncake.tangent(Mooncake.lift(n, Mooncake.zero_tangent(n)))
        @test v.value.next === v                      # cyclic V built, no overflow
        @test Mooncake._dot(v, v) == 0.0
        s = Mooncake._scale(2.0, v)
        @test s.value.next === s                      # scale preserves the cycle
        p = Mooncake._add_to_primal(n, v)
        @test p isa LiftedTest_Cycle && p.next === p  # add_to_primal preserves the cycle
    end

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

    @testset "dual_type base-case coherence" begin
        # Bottom type: must mirror tangent_type(Union{}) === Union{} and
        # lifted_type(Val(N), Union{}) === Union{} (previously MethodError'd).
        @test Mooncake.dual_type(Val(1), Union{}) === Union{}
        @test Mooncake.dual_type(Val(3), Union{}) === Union{}
        # SimpleVector cache-free seed factories must match dual_type === Vector{Any}
        # (previously the fieldcount-0 fallback returned NTuple{N, Vector{Any}}).
        sv = Core.svec(1.0, 2.0)
        @test Mooncake.dual_type(Val(2), Core.SimpleVector) === Vector{Any}
        @test Mooncake.zero_dual(Val(2), sv) isa Vector{Any}
        @test Mooncake.uninit_dual(Val(2), sv) isa Vector{Any}
        @test Mooncake.randn_dual(Val(2), StableRNG(1), sv) isa Vector{Any}
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
        @test Mooncake.tangent(a) === a.partials
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
            @test Mooncake.tangent(a) === a.partials
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

        @testset "cache-threaded float Memory/MemoryRef lift is SoA" begin
            # Under forward-over-reverse, a reverse rule's float `dx::MemoryRef`/`Memory`
            # field is lifted via the 3-arg cache form (`_lift_backing`). It must reach
            # the SoA overload, not the generic AoS path (which yields `MemoryRef{NDual}`).
            T = Float64
            mem = Memory{T}(undef, 3)
            ẋmem = Memory{T}(undef, 3)
            p = Core.memoryref(mem, 1)
            ẋ = Core.memoryref(ẋmem, 1)
            for c in (nothing, IdDict())
                @test typeof(Mooncake.tangent(Mooncake.lift(p, ẋ, c))) ===
                    Mooncake.dual_type(Val(1), MemoryRef{T})
                @test typeof(Mooncake.tangent(Mooncake.lift(mem, ẋmem, c))) ===
                    Mooncake.dual_type(Val(1), Memory{T})
            end
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

        # Canonical V for `Vector{Float64}` is `NDualArray` (SoA). Two lane
        # partial vectors carry the directional derivatives.
        x_inner = Mooncake.NDualArray{P,N,1,Vector{P}}(
            x, ([1.0, 0.0, -0.5], [0.5, 1.0, 0.0])
        )
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

        # LinearAlgebra._kron!: per-lane Kronecker product into dout.
        # x1 (1×2), x2 (2×1) → out (2×2). Pure NDualArray slot construction.
        x1m = [1.0 2.0]
        x2m = reshape([3.0, 4.0], 2, 1)
        outm = Matrix{P}(undef, 2, 2)
        # Seed: lane-1 perturbs x1, lane-2 perturbs x2.
        x1_slot = Mooncake.Lifted{Matrix{P},N}(
            x1m, Mooncake.NDualArray{P,N,2,Matrix{P}}(x1m, (copy(x1m), zeros(P, 1, 2)))
        )
        x2_slot = Mooncake.Lifted{Matrix{P},N}(
            x2m, Mooncake.NDualArray{P,N,2,Matrix{P}}(x2m, (zeros(P, 2, 1), copy(x2m)))
        )
        out_slot = Mooncake.Lifted{Matrix{P},N}(
            outm,
            Mooncake.NDualArray{P,N,2,Matrix{P}}(outm, (zeros(P, 2, 2), zeros(P, 2, 2))),
        )
        kron_slot = Mooncake.Lifted{typeof(LinearAlgebra._kron!),N}(
            LinearAlgebra._kron!, Mooncake.NoTangent()
        )
        r_kron = Mooncake.frule!!(kron_slot, out_slot, x1_slot, x2_slot)
        # Primal: kron(x1m, x2m) = [[3 6]; [4 8]].
        @test outm == [3.0 6.0; 4.0 8.0]
        # Lane 1 perturbs x1 by x1 itself: d(kron(x1+ε*x1, x2)) = kron(x1, x2)*ε → same.
        @test r_kron.value.partials[1] == [3.0 6.0; 4.0 8.0]
        # Lane 2 perturbs x2 by x2 itself: kron(x1, x2+ε*x2) gives kron(x1, x2)*ε.
        @test r_kron.value.partials[2] == [3.0 6.0; 4.0 8.0]
    end

    @testset "dual_type / lifted_type (Tuple)" begin
        # Empty tuple is non-differentiable: its dual_type collapses to NoDual (coherent with
        # the all-non-diff vararg-group V), not an empty `Tuple{}`.
        @test Mooncake.dual_type(Val(2), Tuple{}) === Mooncake.NoDual
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

    @testset "frule!! one-to-one parallels (builtins.jl abs/add)" begin
        # Use the local wrappers defined by Mooncake's @intrinsic macro —
        # rule dispatch is on those, not on `Core.Intrinsics.*`.
        abs_float = Mooncake.IntrinsicsWrappers.abs_float
        add_float = Mooncake.IntrinsicsWrappers.add_float
        add_float_fast = Mooncake.IntrinsicsWrappers.add_float_fast
        copysign_float = Mooncake.IntrinsicsWrappers.copysign_float
        div_float = Mooncake.IntrinsicsWrappers.div_float
        div_float_fast = Mooncake.IntrinsicsWrappers.div_float_fast
        mul_float = Mooncake.IntrinsicsWrappers.mul_float
        mul_float_fast = Mooncake.IntrinsicsWrappers.mul_float_fast
        neg_float = Mooncake.IntrinsicsWrappers.neg_float
        neg_float_fast = Mooncake.IntrinsicsWrappers.neg_float_fast
        sub_float = Mooncake.IntrinsicsWrappers.sub_float
        sub_float_fast = Mooncake.IntrinsicsWrappers.sub_float_fast
        fma_float = Mooncake.IntrinsicsWrappers.fma_float
        muladd_float = Mooncake.IntrinsicsWrappers.muladd_float
        fpext = Mooncake.IntrinsicsWrappers.fpext
        fptrunc = Mooncake.IntrinsicsWrappers.fptrunc
        N = 2
        T = Float64

        # `test_rule` exercises each rule end-to-end through the framework's
        # interpreter against finite-difference references. It currently
        # tests the bare-`Dual` path (the interpreter wraps args as `Dual`);
        # post-Final-task interpreter cutover it will automatically cover the
        # `Lifted` parallels below.
        if isdefined(@__MODULE__, :test_rule)
            test_rule(MersenneTwister(0), abs_float, -3.0; perf_flag=:none)
            test_rule(MersenneTwister(0), add_float, 1.0, 2.0; perf_flag=:none)
            test_rule(MersenneTwister(0), add_float_fast, 1.0, 2.0; perf_flag=:none)
            test_rule(MersenneTwister(0), copysign_float, 2.0, -3.0; perf_flag=:none)
            test_rule(MersenneTwister(0), div_float, 6.0, 2.0; perf_flag=:none)
            test_rule(MersenneTwister(0), div_float_fast, 6.0, 2.0; perf_flag=:none)
            test_rule(MersenneTwister(0), mul_float, 1.5, 2.5; perf_flag=:none)
            test_rule(MersenneTwister(0), mul_float_fast, 1.5, 2.5; perf_flag=:none)
            test_rule(MersenneTwister(0), neg_float, 4.0; perf_flag=:none)
            test_rule(MersenneTwister(0), neg_float_fast, 4.0; perf_flag=:none)
            test_rule(MersenneTwister(0), sub_float, 5.0, 2.0; perf_flag=:none)
            test_rule(MersenneTwister(0), sub_float_fast, 5.0, 2.0; perf_flag=:none)
            test_rule(MersenneTwister(0), fma_float, 1.5, 2.0, 0.5; perf_flag=:none)
            test_rule(MersenneTwister(0), muladd_float, 1.5, 2.0, 0.5; perf_flag=:none)
            test_rule(MersenneTwister(0), fpext, Float64, 1.5f0; perf_flag=:none)
            test_rule(MersenneTwister(0), fptrunc, Float32, 1.5; perf_flag=:none)
        end

        # abs_float: y = abs(x); dy = sign(x) * dx.
        x_inner = Mooncake.NDual{T,N}(-3.0, (1.0, -1.0))
        x_slot = Mooncake.Lifted{T,N}(-3.0, x_inner)
        f_slot = Mooncake.Lifted{typeof(abs_float),N}(abs_float, Mooncake.NoTangent())
        r = Mooncake.frule!!(f_slot, x_slot)
        @test typeof(r) === Mooncake.Lifted{T,N,Mooncake.NDual{T,N}}
        @test Mooncake.primal(r) === 3.0
        @test Mooncake.tangent(r).partials === (-1.0, 1.0)  # sign(-3)*(1,-1)

        # add_float: c = a + b; dc = da + db.
        a_inner = Mooncake.NDual{T,N}(1.0, (1.0, 0.0))
        b_inner = Mooncake.NDual{T,N}(2.0, (0.0, 1.0))
        a_slot = Mooncake.Lifted{T,N}(1.0, a_inner)
        b_slot = Mooncake.Lifted{T,N}(2.0, b_inner)
        addf = Mooncake.Lifted{typeof(add_float),N}(add_float, Mooncake.NoTangent())
        r_add = Mooncake.frule!!(addf, a_slot, b_slot)
        @test Mooncake.primal(r_add) === 3.0
        @test Mooncake.tangent(r_add).partials === (1.0, 1.0)

        # add_float_fast: same body shape.
        addf_fast = Mooncake.Lifted{typeof(add_float_fast),N}(
            add_float_fast, Mooncake.NoTangent()
        )
        r_fast = Mooncake.frule!!(addf_fast, a_slot, b_slot)
        @test Mooncake.primal(r_fast) === 3.0
        @test Mooncake.tangent(r_fast).partials === (1.0, 1.0)

        # copysign_float: z = copysign(x, y); dz = sign(y) * dx.
        x2 = Mooncake.Lifted{T,N}(2.0, Mooncake.NDual{T,N}(2.0, (1.0, 0.0)))
        yneg = Mooncake.Lifted{T,N}(-3.0, Mooncake.NDual{T,N}(-3.0, (0.0, 1.0)))
        cf = Mooncake.Lifted{typeof(copysign_float),N}(copysign_float, Mooncake.NoTangent())
        r_cs = Mooncake.frule!!(cf, x2, yneg)
        @test Mooncake.primal(r_cs) === -2.0
        @test Mooncake.tangent(r_cs).partials == (-1.0, 0.0)  # sign(-3)*(1,0); -0.0 ok

        # div_float: c = a/b; dc = (da*b - a*db)/b^2.
        a2 = Mooncake.Lifted{T,N}(6.0, Mooncake.NDual{T,N}(6.0, (1.0, 0.0)))
        b2 = Mooncake.Lifted{T,N}(2.0, Mooncake.NDual{T,N}(2.0, (0.0, 1.0)))
        df = Mooncake.Lifted{typeof(div_float),N}(div_float, Mooncake.NoTangent())
        r_div = Mooncake.frule!!(df, a2, b2)
        @test Mooncake.primal(r_div) === 3.0  # 6/2
        # ∂(a/b)/∂a = 1/b = 0.5; ∂(a/b)/∂b = -a/b² = -1.5.
        @test Mooncake.tangent(r_div).partials === (0.5, -1.5)

        # div_float_fast: same shape.
        dff = Mooncake.Lifted{typeof(div_float_fast),N}(
            div_float_fast, Mooncake.NoTangent()
        )
        r_div_fast = Mooncake.frule!!(dff, a2, b2)
        @test Mooncake.primal(r_div_fast) === 3.0
        @test Mooncake.tangent(r_div_fast).partials === (0.5, -1.5)

        # mul_float: product rule (a*db + b*da).
        a3 = Mooncake.Lifted{T,N}(1.5, Mooncake.NDual{T,N}(1.5, (1.0, 0.0)))
        b3 = Mooncake.Lifted{T,N}(2.5, Mooncake.NDual{T,N}(2.5, (0.0, 1.0)))
        mf = Mooncake.Lifted{typeof(mul_float),N}(mul_float, Mooncake.NoTangent())
        r_mul = Mooncake.frule!!(mf, a3, b3)
        @test Mooncake.primal(r_mul) === 3.75  # 1.5 * 2.5
        # ∂(a*b)/∂a = b = 2.5; ∂(a*b)/∂b = a = 1.5.
        @test Mooncake.tangent(r_mul).partials === (2.5, 1.5)

        # mul_float_fast: same.
        mff = Mooncake.Lifted{typeof(mul_float_fast),N}(
            mul_float_fast, Mooncake.NoTangent()
        )
        r_mul_fast = Mooncake.frule!!(mff, a3, b3)
        @test Mooncake.primal(r_mul_fast) === 3.75
        @test Mooncake.tangent(r_mul_fast).partials === (2.5, 1.5)

        # neg_float: dy = -dx.
        x4 = Mooncake.Lifted{T,N}(4.0, Mooncake.NDual{T,N}(4.0, (1.0, -2.0)))
        nf = Mooncake.Lifted{typeof(neg_float),N}(neg_float, Mooncake.NoTangent())
        r_neg = Mooncake.frule!!(nf, x4)
        @test Mooncake.primal(r_neg) === -4.0
        @test Mooncake.tangent(r_neg).partials === (-1.0, 2.0)

        # neg_float_fast: same.
        nff = Mooncake.Lifted{typeof(neg_float_fast),N}(
            neg_float_fast, Mooncake.NoTangent()
        )
        r_neg_fast = Mooncake.frule!!(nff, x4)
        @test Mooncake.primal(r_neg_fast) === -4.0
        @test Mooncake.tangent(r_neg_fast).partials === (-1.0, 2.0)

        # sub_float: dc = da - db.
        a5 = Mooncake.Lifted{T,N}(5.0, Mooncake.NDual{T,N}(5.0, (1.0, 0.0)))
        b5 = Mooncake.Lifted{T,N}(2.0, Mooncake.NDual{T,N}(2.0, (0.0, 1.0)))
        sf = Mooncake.Lifted{typeof(sub_float),N}(sub_float, Mooncake.NoTangent())
        r_sub = Mooncake.frule!!(sf, a5, b5)
        @test Mooncake.primal(r_sub) === 3.0
        @test Mooncake.tangent(r_sub).partials === (1.0, -1.0)

        # sub_float_fast: same.
        sff = Mooncake.Lifted{typeof(sub_float_fast),N}(
            sub_float_fast, Mooncake.NoTangent()
        )
        r_sub_fast = Mooncake.frule!!(sff, a5, b5)
        @test Mooncake.primal(r_sub_fast) === 3.0
        @test Mooncake.tangent(r_sub_fast).partials === (1.0, -1.0)

        # fma_float / muladd_float: a = x*y + z; product rule on first two + z.
        x6 = Mooncake.Lifted{T,N}(1.5, Mooncake.NDual{T,N}(1.5, (1.0, 0.0)))
        y6 = Mooncake.Lifted{T,N}(2.0, Mooncake.NDual{T,N}(2.0, (0.0, 1.0)))
        z6 = Mooncake.Lifted{T,N}(0.5, Mooncake.NDual{T,N}(0.5, (0.0, 0.0)))
        for (op, op_slot_T) in
            ((fma_float, typeof(fma_float)), (muladd_float, typeof(muladd_float)))
            sl = Mooncake.Lifted{op_slot_T,N}(op, Mooncake.NoTangent())
            r = Mooncake.frule!!(sl, x6, y6, z6)
            @test Mooncake.primal(r) === 3.5  # 1.5*2.0 + 0.5
            # ∂(x*y+z)/∂x = y = 2.0, ∂/∂y = x = 1.5, ∂/∂z = 1.
            # Lane 1 seeds dx; lane 2 seeds dy → (2.0, 1.5).
            @test Mooncake.tangent(r).partials === (2.0, 1.5)
        end

        # fpext: cross-precision lift, NDual{Float64,N}(NDual{Float32,N}).
        xf32 = Mooncake.Lifted{Float32,N}(
            1.5f0, Mooncake.NDual{Float32,N}(1.5f0, (1.0f0, 0.0f0))
        )
        ext_sl = Mooncake.Lifted{typeof(fpext),N}(fpext, Mooncake.NoTangent())
        ty_sl = Mooncake.Lifted{Type{Float64},N}(Float64, Mooncake.NoTangent())
        r_ext = Mooncake.frule!!(ext_sl, ty_sl, xf32)
        @test typeof(r_ext) === Mooncake.Lifted{Float64,N,Mooncake.NDual{Float64,N}}
        @test Mooncake.primal(r_ext) === 1.5
        @test Mooncake.tangent(r_ext).partials === (1.0, 0.0)

        # fptrunc: cross-precision truncate, NDual{Float32,N}(NDual{Float64,N}).
        xf64 = Mooncake.Lifted{Float64,N}(1.5, Mooncake.NDual{Float64,N}(1.5, (1.0, 0.0)))
        tr_sl = Mooncake.Lifted{typeof(fptrunc),N}(fptrunc, Mooncake.NoTangent())
        tyf32 = Mooncake.Lifted{Type{Float32},N}(Float32, Mooncake.NoTangent())
        r_tr = Mooncake.frule!!(tr_sl, tyf32, xf64)
        @test typeof(r_tr) === Mooncake.Lifted{Float32,N,Mooncake.NDual{Float32,N}}
        @test Mooncake.primal(r_tr) === 1.5f0
        @test Mooncake.tangent(r_tr).partials === (1.0f0, 0.0f0)
    end

    @testset "frule!! one-to-one parallels (rules_via_nfwd.jl)" begin
        if isdefined(@__MODULE__, :test_rule)
            rng = MersenneTwister(0)
            # Representative coverage of each pattern in the file.
            for f in (exp, log, sin, cos, sqrt, cbrt)
                test_rule(rng, f, 1.5; perf_flag=:none)
            end
            # tanpi at a non-singular point (1.5 = singularity).
            test_rule(rng, tanpi, 0.1; perf_flag=:none)
            test_rule(rng, atan, 1.0, 2.0; perf_flag=:none)
            test_rule(rng, ^, 2.0, 3.0; perf_flag=:none)
            test_rule(rng, max, 1.5, 0.5; perf_flag=:none)
            test_rule(rng, Base.FastMath.pow_fast, 2.0, 3; perf_flag=:none)
            test_rule(rng, clamp, 0.5, 0.0, 1.0; perf_flag=:none)
            test_rule(rng, sincos, 1.0; perf_flag=:none)
            test_rule(rng, sincosd, 30.0; perf_flag=:none)
            test_rule(rng, sincospi, 0.25; perf_flag=:none)
            test_rule(rng, modf, 1.7; perf_flag=:none)
            test_rule(rng, hypot, 3.0, 4.0; perf_flag=:none)
            test_rule(rng, hypot, 1.0, 2.0, 2.0; perf_flag=:none)
        end

        # Direct Lifted-arg invocation check for one unary representative.
        N = 2
        T = Float64
        x_inner = Mooncake.NDual{T,N}(1.0, (1.0, 0.0))
        x_slot = Mooncake.Lifted{T,N}(1.0, x_inner)
        sin_slot = Mooncake.Lifted{typeof(sin),N}(sin, Mooncake.NoTangent())
        r = Mooncake.frule!!(sin_slot, x_slot)
        @test typeof(r) === Mooncake.Lifted{T,N,Mooncake.NDual{T,N}}
        @test Mooncake.primal(r) === sin(1.0)
        @test Mooncake.tangent(r).partials[1] ≈ cos(1.0)
        @test Mooncake.tangent(r).partials[2] == 0.0
    end

    @testset "frule!! one-to-one parallels (tasks.jl)" begin
        N = 2
        task = Task(() -> nothing)

        # Type-level: dual_type / lifted_type for Task.
        @test Mooncake.dual_type(Val(N), Task) === Mooncake.TaskTangent
        @test Mooncake.lifted_type(Val(N), Task) ===
            Mooncake.Lifted{Task,N,Mooncake.TaskTangent}

        # Lifted-arg lgetfield: a Task field is non-differentiable, so its forward V is `NoDual`
        # (the forward-mode non-diff sentinel — reverse mode would use `NoTangent`).
        task_slot = Mooncake.Lifted{Task,N}(task, Mooncake.TaskTangent())
        f_slot = Mooncake.Lifted{typeof(Mooncake.lgetfield),N}(
            Mooncake.lgetfield, Mooncake.NoTangent()
        )
        v_slot = Mooncake.Lifted{Val{:rngState1},N}(Val(:rngState1), Mooncake.NoTangent())
        r = Mooncake.frule!!(f_slot, task_slot, v_slot)
        @test Mooncake.primal(r) === getfield(task, :rngState1)
        @test Mooncake.tangent(r) === Mooncake.NoDual()

        # Lifted-arg getfield.
        gf_slot = Mooncake.Lifted{typeof(getfield),N}(getfield, Mooncake.NoTangent())
        sym_slot = Mooncake.Lifted{Symbol,N}(:rngState1, Mooncake.NoTangent())
        r2 = Mooncake.frule!!(gf_slot, task_slot, sym_slot)
        @test Mooncake.primal(r2) === getfield(task, :rngState1)
        @test Mooncake.tangent(r2) === Mooncake.NoDual()
    end

    @testset "frule!! one-to-one parallels (new.jl _new_)" begin
        N = 2
        T = Float64

        # Concrete struct branch: ImmutableDual{NamedTuple{...}} V.
        type_slot_imm = Mooncake.Lifted{Type{LiftedTest_Point},N}(
            LiftedTest_Point, Mooncake.NoTangent()
        )
        x_slot = Mooncake.Lifted{T,N}(1.5, Mooncake.NDual{T,N}(1.5, (1.0, 0.0)))
        y_slot = Mooncake.Lifted{T,N}(2.5, Mooncake.NDual{T,N}(2.5, (0.0, 1.0)))
        new_slot = Mooncake.Lifted{typeof(Mooncake._new_),N}(
            Mooncake._new_, Mooncake.NoTangent()
        )
        r_imm = Mooncake.frule!!(new_slot, type_slot_imm, x_slot, y_slot)
        @test typeof(r_imm) === Mooncake.Lifted{
            LiftedTest_Point,N,Mooncake.dual_type(Val(N), LiftedTest_Point)
        }
        @test Mooncake.primal(r_imm) === LiftedTest_Point(1.5, 2.5)
        @test Mooncake.tangent(r_imm) isa Mooncake.ImmutableDual

        # Mutable struct branch: MutableDual{NamedTuple{...}} V.
        type_slot_mut = Mooncake.Lifted{Type{LiftedTest_RefF},N}(
            LiftedTest_RefF, Mooncake.NoTangent()
        )
        v_slot = Mooncake.Lifted{T,N}(7.0, Mooncake.NDual{T,N}(7.0, (1.0, -1.0)))
        r_mut = Mooncake.frule!!(new_slot, type_slot_mut, v_slot)
        @test typeof(r_mut) ===
            Mooncake.Lifted{LiftedTest_RefF,N,Mooncake.dual_type(Val(N), LiftedTest_RefF)}
        @test Mooncake.primal(r_mut).v === 7.0
        @test Mooncake.tangent(r_mut) isa Mooncake.MutableDual
    end

    @testset "frule!! one-to-one parallels (iddict.jl)" begin
        N = 2
        T = Float64

        # Type-level: dual_type / lifted_type for IdDict{K, V}.
        @test Mooncake.dual_type(Val(N), IdDict{Symbol,T}) ===
            IdDict{Symbol,Mooncake.NDual{T,N}}
        @test Mooncake.lifted_type(Val(N), IdDict{Symbol,T}) ===
            Mooncake.Lifted{IdDict{Symbol,T},N,IdDict{Symbol,Mooncake.NDual{T,N}}}

        # Constructor: Type{IdDict{K,V}}() → Lifted{IdDict{K,V},N}.
        ctor_slot = Mooncake.Lifted{Type{IdDict{Symbol,T}},N}(
            IdDict{Symbol,T}, Mooncake.NoTangent()
        )
        r_ctor = Mooncake.frule!!(ctor_slot)
        @test typeof(r_ctor) === Mooncake.lifted_type(Val(N), IdDict{Symbol,T})
        @test isempty(Mooncake.primal(r_ctor))
        @test isempty(Mooncake.tangent(r_ctor))

        # setindex! + getindex round trip.
        d_primal = IdDict{Symbol,T}()
        d_tan = IdDict{Symbol,Mooncake.NDual{T,N}}()
        d_slot = Mooncake.Lifted{IdDict{Symbol,T},N}(d_primal, d_tan)
        val_slot = Mooncake.Lifted{T,N}(3.0, Mooncake.NDual{T,N}(3.0, (1.0, -1.0)))
        key_slot = Mooncake.Lifted{Symbol,N}(:a, Mooncake.NoTangent())
        si_slot = Mooncake.Lifted{typeof(setindex!),N}(setindex!, Mooncake.NoTangent())
        Mooncake.frule!!(si_slot, d_slot, val_slot, key_slot)
        @test d_primal[:a] === 3.0
        @test d_tan[:a].partials === (1.0, -1.0)

        gi_slot = Mooncake.Lifted{typeof(getindex),N}(getindex, Mooncake.NoTangent())
        r_gi = Mooncake.frule!!(gi_slot, d_slot, key_slot)
        @test Mooncake.primal(r_gi) === 3.0
        @test Mooncake.tangent(r_gi).partials === (1.0, -1.0)
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "frule!! one-to-one parallels (memory.jl)" begin
            N = 2
            T = Float64

            # dual_type / lifted_type for Memory{T<:IEEEFloat}.
            @test Mooncake.dual_type(Val(N), Memory{T}) ===
                Mooncake.NDualArray{T,N,1,Memory{T},Mooncake.NDual{T,N}}
            @test Mooncake.lifted_type(Val(N), Memory{T}) === Mooncake.Lifted{
                Memory{T},N,Mooncake.NDualArray{T,N,1,Memory{T},Mooncake.NDual{T,N}}
            }

            # Memory{P}(undef, n) constructor.
            ctor_slot = Mooncake.Lifted{Type{Memory{T}},N}(Memory{T}, Mooncake.NoTangent())
            undef_slot = Mooncake.Lifted{UndefInitializer,N}(undef, Mooncake.NoTangent())
            n_slot = Mooncake.Lifted{Int,N}(3, Mooncake.NoTangent())
            r_mem = Mooncake.frule!!(ctor_slot, undef_slot, n_slot)
            @test typeof(r_mem) === Mooncake.lifted_type(Val(N), Memory{T})
            @test length(Mooncake.primal(r_mem)) == 3
            @test all(iszero, Mooncake.tangent(r_mem).partials[1])
            @test all(iszero, Mooncake.tangent(r_mem).partials[2])

            # memoryrefnew(::Memory) → MemoryRef slot.
            mn_slot = Mooncake.Lifted{typeof(Core.memoryrefnew),N}(
                Core.memoryrefnew, Mooncake.NoTangent()
            )
            r_ref = Mooncake.frule!!(mn_slot, r_mem)
            @test typeof(r_ref) === Mooncake.lifted_type(Val(N), MemoryRef{T})
            @test Mooncake.primal(r_ref) === Core.memoryref(Mooncake.primal(r_mem))

            # lmemoryrefget — read NDual from a position-1 MemoryRef.
            lg_slot = Mooncake.Lifted{typeof(Mooncake.lmemoryrefget),N}(
                Mooncake.lmemoryrefget, Mooncake.NoTangent()
            )
            ord_slot = Mooncake.Lifted{Val{:not_atomic},N}(
                Val(:not_atomic), Mooncake.NoTangent()
            )
            bc_slot = Mooncake.Lifted{Val{false},N}(Val(false), Mooncake.NoTangent())
            r_get = Mooncake.frule!!(lg_slot, r_ref, ord_slot, bc_slot)
            @test typeof(r_get) === Mooncake.Lifted{T,N,Mooncake.NDual{T,N}}
            @test Mooncake.primal(r_get) === Mooncake.primal(r_mem)[1]
        end
    end

    @testset "frule!! one-to-one parallels (threads.jl)" begin
        # `_foreigncall_` has a strict signature `(Val{name}, Val{RT}, Tuple,
        # Val{nreq}, Val{calling_convention}, args…)`; constructing a
        # representative call to exercise the body is fragile. Verify
        # registration via `methods` lookup instead.
        ms = methods(Mooncake.frule!!)
        sigs = [sprint(show, m) for m in ms]
        has_lifted_threading_rule = any(
            s -> occursin("jl_in_threaded_region", s) && occursin("Lifted", s), sigs
        )
        @test has_lifted_threading_rule
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

    @testset "basis_lifted!!" begin
        # `basis_lifted!!(zero_lifted(...), slots)` sets lane k hot at the
        # slots[k]-th scalar dof, mutating mutable V in place and rebuilding
        # immutable V. dofs are counted in `dof`/`zero_tangent` order.
        bl(x, slots) = Mooncake.basis_lifted!!(
            Mooncake.zero_lifted(Val(length(slots)), x), slots
        )

        @test Mooncake.tangent(bl(3.0, (1,)), 1) == 1.0
        @test Mooncake.tangent(bl([5.0, 6.0, 7.0], (2,)), 1) == [0.0, 1.0, 0.0]
        @test Mooncake.tangent(bl(1.0 + 2.0im, (2,)), 1) == 0.0 + 1.0im  # imag dof
        let t = Mooncake.tangent(bl(([1.0, 2.0], 9.0), (3,)), 1)
            @test t[1] == [0.0, 0.0] && t[2] == 1.0  # the scalar is dof 3
        end

        # width-2: two basis directions in one seed.
        let b = bl([5.0, 6.0, 7.0], (1, 3))
            @test Mooncake.tangent(b, 1) == [1.0, 0.0, 0.0]
            @test Mooncake.tangent(b, 2) == [0.0, 0.0, 1.0]
        end

        # Aliased fields: `dof` dedups the shared array, so the seed visits it
        # once and both fields share the (single, mutated) V.
        shared = [10.0, 20.0]
        let nt = bl(LiftedTest_Aliased(shared, shared), (1,)).value.value
            @test nt.a === nt.b
            @test nt.a.partials[1] == [1.0, 0.0]
        end

        # Self-cyclic mutable struct: terminates; `.next` V is the node's own V.
        c = LiftedTest_Cycle(nothing, 5.0)
        c.next = c
        let b = bl(c, (1,))
            @test b.value.value.next === b.value
            @test b.value.value.w.partials[1] == 1.0  # `w` is the only dof
        end

        # Uninit field stays uninit/zero; the defined field gets the basis.
        let nt = bl(LiftedTest_MaybeInit(3.0), (1,)).value.value
            @test nt.x.partials[1] == 1.0
        end
    end
end
