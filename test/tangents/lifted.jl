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
mutable struct LiftedTest_AliasedNested
    a::Vector{Vector{Float64}}
    b::Vector{Vector{Float64}}
end
mutable struct LiftedTest_ParentField  # field name collides with the view's internals
    parent::Float64
end
mutable struct LiftedTest_MaybeInit
    x::Float64
    y::Float64
    LiftedTest_MaybeInit(x::Float64) = new(x)
end
abstract type LiftedTest_AbsScalar end
struct LiftedTest_ConcScalar <: LiftedTest_AbsScalar
    μ::Float64
    σ::Float64
end

using Mooncake:
    NDual,
    NDualArray,
    NoDual,
    ImmutableDual,
    MutableDual,
    MutableDualTangentView,
    lifted_type,
    zero_lifted,
    uninit_lifted,
    randn_lifted,
    uninit_dual,
    randn_dual,
    basis_lifted!!,
    lift,
    unlift,
    extract,
    unpack_ndual,
    frule!!,
    TaskTangent,
    _new_

# Shorthands for the table-driven type tests below (top level: parametric aliases are
# not allowed in local scope).
const ND = NDual
NDA{T,N,D,A} = NDualArray{T,N,D,A,NDual{T,N}}

@testset "lifted" begin
    # Slot/inner-dual shorthands. `sl` wraps once at the top level with the sharp
    # `Base._stable_typeof` P (`Type{T}` for type-valued primals, matching rule dispatch).
    sl(N, p, v=NoTangent()) = Lifted{Base._stable_typeof(p),N}(p, v)
    nd(v::T, parts::Vararg{T,N}) where {T,N} = NDual{T,N}(v, parts)

    @testset "cyclic MutableDual tangent arithmetic" begin
        # A self-referential mutable struct lifts to a cyclic `MutableDual` V; the
        # tangent-arithmetic helpers must terminate via their aliasing caches.
        n = LiftedTest_Cycle(nothing, 2.0)
        n.next = n
        v = tangent(lift(n, zero_tangent(n)))
        @test v.value.next === v                      # cyclic V built, no overflow
        @test Mooncake._dot(v, v) == 0.0
        s = Mooncake._scale(2.0, v)
        @test s.value.next === s                      # scale preserves the cycle
        p = Mooncake._add_to_primal(n, v)
        @test p isa LiftedTest_Cycle && p.next === p  # add_to_primal preserves the cycle
    end

    @testset "cyclic plain-array tangent arithmetic" begin
        # A self-referential plain `Array` lifts to a cyclic element-wise V (the seed registers its
        # shell before filling); the tangent-arithmetic helpers must terminate via their aliasing
        # caches, as for the cyclic mutable struct above. (Seed/per-lane/lift/unlift across widths
        # are covered by the `test_lifted` drive below.)
        x = Any[]
        push!(x, x)
        v = tangent(zero_lifted(Val(1), x))
        @test v[1] === v                              # cyclic V built, no overflow
        @test Mooncake._dot(v, v) == 0.0
        s = Mooncake._scale(2.0, v)
        @test s[1] === s                              # scale preserves the cycle
        p = Mooncake._add_to_primal(x, v)
        @test p[1] === p                              # add_to_primal preserves the cycle
    end

    @testset "Lifted slot basics" begin
        inner = nd(3.0, 1.0, -1.0)
        slot = sl(2, 3.0, inner)
        @test typeof(slot) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(slot) === 3.0
        @test tangent(slot) === inner
        @test extract(slot) === (3.0, inner)
        @test slot == sl(2, 3.0, inner)
        @test copy(slot) == slot
    end

    # `dual_type(Val(N), P) === V` and `lifted_type(Val(N), P) === Lifted{P,N,V}` per shape.
    @testset "dual/lifted_type $P (N=$N)" for (N, P, V) in Any[
        # IEEEFloat scalars.
        (1, Float64, ND{Float64,1}),
        (3, Float32, ND{Float32,3}),
        (4, Float64, ND{Float64,4}),
        # Dense arrays: parallel-arrays NDualArray.
        (2, Vector{Float64}, NDA{Float64,2,1,Vector{Float64}}),
        (1, Matrix{Float32}, NDA{Float32,1,2,Matrix{Float32}}),
        # Complex scalars and arrays.
        (2, Complex{Float64}, Complex{ND{Float64,2}}),
        (1, Complex{Float32}, Complex{ND{Float32,1}}),
        (
            2,
            Vector{ComplexF64},
            NDualArray{ComplexF64,2,1,Vector{ComplexF64},Complex{ND{Float64,2}}},
        ),
        # Tuples / NamedTuples: element-wise recursion.
        (2, Tuple{Float64}, Tuple{ND{Float64,2}}),
        (2, Tuple{Float64,Float32}, Tuple{ND{Float64,2},ND{Float32,2}}),
        (
            2,
            Tuple{Float64,Vector{Float64}},
            Tuple{ND{Float64,2},NDA{Float64,2,1,Vector{Float64}}},
        ),
        (
            2,
            NamedTuple{(:a, :b),Tuple{Float64,Float32}},
            NamedTuple{(:a, :b),Tuple{ND{Float64,2},ND{Float32,2}}},
        ),
        (
            2,
            NamedTuple{(:x, :y),Tuple{Float64,Vector{Float64}}},
            NamedTuple{(:x, :y),Tuple{ND{Float64,2},NDA{Float64,2,1,Vector{Float64}}}},
        ),
        # Struct lifts: Immutable/MutableDual over the per-field NamedTuple.
        (
            2,
            LiftedTest_Point,
            ImmutableDual{NamedTuple{(:x, :y),Tuple{ND{Float64,2},ND{Float64,2}}}},
        ),
        (3, LiftedTest_RefF, MutableDual{NamedTuple{(:v,),Tuple{ND{Float64,3}}}}),
        # Custom canonical Vs.
        (2, Task, TaskTangent),
        (2, IdDict{Symbol,Float64}, IdDict{Symbol,ND{Float64,2}}),
    ]
        @test dual_type(Val(N), P) === V
        @test lifted_type(Val(N), P) === Lifted{P,N,V}
    end

    @testset "dual_type base-case coherence" begin
        # Bottom type mirrors tangent_type(Union{}) === Union{}; empty tuple is
        # non-differentiable so its V collapses to NoDual, not Tuple{}.
        @test dual_type(Val(1), Union{}) === Union{}
        @test dual_type(Val(3), Union{}) === Union{}
        @test dual_type(Val(2), Tuple{}) === NoDual
        # SimpleVector: cache-free seed factories must match dual_type === Vector{Any}.
        sv = Core.svec(1.0, 2.0)
        @test dual_type(Val(2), Core.SimpleVector) === Vector{Any}
        @test zero_dual(Val(2), sv) isa Vector{Any}
        @test uninit_dual(Val(2), sv) isa Vector{Any}
        @test randn_dual(Val(2), StableRNG(1), sv) isa Vector{Any}
    end

    @testset "seed factories" begin
        # Scalar: zero/uninit/randn duals + the Lifted wrappers; randn replays under the same rng.
        v = zero_dual(Val(2), 7.0)
        @test v === nd(7.0, 0.0, 0.0)
        @test uninit_dual(Val(3), 1.0f0).value === 1.0f0
        r = randn_dual(Val(2), Random.MersenneTwister(42), 0.0)
        @test r.value == 0.0 && any(!iszero, r.partials)
        z = zero_lifted(Val(2), 7.0)
        @test typeof(z) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(z) === 7.0 && tangent(z) === v
        @test typeof(uninit_lifted(Val(3), 1.0f0)) === Lifted{Float32,3,NDual{Float32,3}}
        @test tangent(randn_lifted(Val(2), Random.MersenneTwister(42), 0.0)) == r

        # Array: primal aliases user storage; partials slot-local and zeroed; lazy getindex.
        x = [1.0, 2.0, 3.0]
        va = zero_dual(Val(2), x)
        @test typeof(va) === dual_type(Val(2), Vector{Float64})
        @test primal(va) === x
        @test all(iszero, va.partials[1]) && all(iszero, va.partials[2])
        @test va[1] === nd(1.0, 0.0, 0.0)
        @test any(!iszero, randn_dual(Val(2), Random.MersenneTwister(0), x).partials[1])
        za = zero_lifted(Val(2), x)
        @test typeof(za) === lifted_type(Val(2), Vector{Float64})
        @test primal(za) === x
        M = [1.0 2.0; 3.0 4.0]
        @test size(tangent(zero_lifted(Val(2), M))) == size(M)

        # Complex scalar and complex array.
        vz = zero_dual(Val(2), 1.5 - 0.5im)
        @test vz === Complex(nd(1.5, 0.0, 0.0), nd(-0.5, 0.0, 0.0))
        @test tangent(zero_lifted(Val(2), 1.5 - 0.5im)) === vz
        xc = ComplexF64[1.0 + 2.0im, 3.0 - 1.0im]
        vc = zero_dual(Val(2), xc)
        @test typeof(vc) === dual_type(Val(2), Vector{ComplexF64})
        @test primal(vc) === xc
        @test all(iszero, vc.partials[1]) && all(iszero, vc.partials[2])
        @test typeof(zero_lifted(Val(2), xc)) === lifted_type(Val(2), Vector{ComplexF64})

        # Tuple / NamedTuple: outer container is per-element; array elements alias.
        t = (1.0, 2.0f0, [3.0, 4.0])
        vt = zero_dual(Val(2), t)
        @test typeof(vt) === dual_type(Val(2), typeof(t))
        @test vt[1] === nd(1.0, 0.0, 0.0) && vt[2] === nd(2.0f0, 0.0f0, 0.0f0)
        @test primal(vt[3]) === t[3]
        zt = zero_lifted(Val(2), t)
        @test typeof(zt) === Lifted{typeof(t),2,typeof(vt)}
        @test primal(zt) === t && tangent(zt) == vt
        nt = (; a=1.0, c=[3.0, 4.0])
        vnt = zero_dual(Val(2), nt)
        @test vnt.a === nd(1.0, 0.0, 0.0) && primal(vnt.c) === nt.c
        @test primal(zero_lifted(Val(2), nt)) === nt

        # Struct lifts: per-field duals inside Immutable/MutableDual.
        p = LiftedTest_Point(1.0, 2.0)
        vp = zero_dual(Val(2), p)
        @test typeof(vp) === dual_type(Val(2), LiftedTest_Point)
        @test vp.value.x === nd(1.0, 0.0, 0.0) && vp.value.y === nd(2.0, 0.0, 0.0)
        zp = zero_lifted(Val(2), p)
        @test typeof(zp) === lifted_type(Val(2), LiftedTest_Point)
        @test primal(zp) === p && tangent(zp) === vp
        vm = zero_dual(Val(2), LiftedTest_RefF(3.0))
        @test typeof(vm) === dual_type(Val(2), LiftedTest_RefF)
        @test vm.value.v === nd(3.0, 0.0, 0.0)
    end

    @testset "NDualArray accessors + AbstractArray interface" begin
        x = [1.0, 2.0, 3.0]
        a = NDualArray{Float64,2,1,Vector{Float64}}(
            x, (similar(x).=[0.5, -0.5, 1.5], similar(x).=[0.0, 1.0, -1.0])
        )
        @test a isa AbstractArray{NDual{Float64,2},1}
        @test size(a) == (3,) && length(a) == 3
        @test primal(a) === x && tangent(a) === a.partials
        @test unpack_ndual(a) === (a.primal, a.partials)
        @test a[2] === nd(2.0, -0.5, 1.0)  # lazy getindex
        a[1] = nd(9.0, 7.0, -7.0)          # setindex! writes both channels
        @test x[1] === 9.0 && a.partials[1][1] === 7.0 && a.partials[2][1] === -7.0
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "NDualMemoryRef (MemoryRef{T<:IEEEFloat})" begin
            mem = Memory{Float64}(undef, 3) .= [1.0, 2.0, 3.0]
            p = Core.memoryref(mem, 1)
            @test dual_type(Val(2), MemoryRef{Float64}) ===
                Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            @test lifted_type(Val(2), MemoryRef{Float64}) === Lifted{
                MemoryRef{Float64},2,Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            }
            @test dual_type(Val(2), Memory{Float64}) ===
                NDualArray{Float64,2,1,Memory{Float64},NDual{Float64,2}}

            # Seed factory: zero-init partials at the same offset, slot-local.
            a = zero_dual(Val(2), p)
            @test typeof(a) === Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            @test primal(a) === p && tangent(a) === a.partials
            @test unpack_ndual(a) === (a.primal, a.partials)
            @test all(iszero, a.partials[1].mem) && all(iszero, a.partials[2].mem)
            @test Mooncake._memoryrefget_ndual(a, :not_atomic, false) === nd(1.0, 0.0, 0.0)
            @test primal(zero_lifted(Val(2), p)) === p
        end

        @testset "cache-threaded float Memory/MemoryRef lift is parallel-arrays" begin
            # Under forward-over-reverse, a reverse rule's float `dx::MemoryRef`/`Memory`
            # field is lifted via the 3-arg cache form (`_lift_backing`). It must reach
            # the parallel-arrays overload, not the generic element-wise path.
            mem, ẋmem = Memory{Float64}(undef, 3), Memory{Float64}(undef, 3)
            p, ẋ = Core.memoryref(mem, 1), Core.memoryref(ẋmem, 1)
            for c in (nothing, IdDict())
                @test typeof(tangent(lift(p, ẋ, c))) ===
                    dual_type(Val(1), MemoryRef{Float64})
                @test typeof(tangent(lift(mem, ẋmem, c))) ===
                    dual_type(Val(1), Memory{Float64})
            end
        end
    end

    @testset "frule!! one-to-one parallels (complex.jl)" begin
        re_inner, im_inner = nd(1.5, 1.0, 0.0), nd(-0.5, 0.0, 1.0)
        z_inner = Complex(re_inner, im_inner)
        z = sl(2, Complex(1.5, -0.5), z_inner)

        r = frule!!(sl(2, lgetfield), z, sl(2, Val(:re)))
        @test typeof(r) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(r) == 1.5 && tangent(r) === re_inner
        r2 = frule!!(sl(2, lgetfield), z, sl(2, Val(:im)))
        @test primal(r2) == -0.5 && tangent(r2) === im_inner

        r3 = frule!!(
            sl(2, _new_), sl(2, ComplexF64), sl(2, 1.5, re_inner), sl(2, -0.5, im_inner)
        )
        @test typeof(r3) === Lifted{ComplexF64,2,Complex{NDual{Float64,2}}}
        @test primal(r3) == Complex(1.5, -0.5) && tangent(r3) === z_inner
    end

    @testset "frule!! one-to-one parallels (performance_patches.jl)" begin
        x = [1.0, 2.0, 3.0]
        x_slot = sl(
            2,
            x,
            NDualArray{Float64,2,1,Vector{Float64}}(x, ([1.0, 0.0, -0.5], [0.5, 1.0, 0.0])),
        )

        r = frule!!(sl(2, sum), x_slot)
        @test typeof(r) === Lifted{Float64,2,NDual{Float64,2}}
        @test tangent(r) === nd(sum(x), 0.5, 1.5)

        # Forward derivative of sum(abs2, x) along d is 2*dot(x, d).
        r2 = frule!!(sl(2, sum), sl(2, abs2), x_slot)
        @test primal(r2) == sum(abs2, x)
        @test tangent(r2).value == sum(abs2, x)
        @test tangent(r2).partials[1] ≈ 2 * (1.0 * 1.0 + 3.0 * -0.5)
        @test tangent(r2).partials[2] ≈ 2 * (1.0 * 0.5 + 2.0 * 1.0)

        # LinearAlgebra._kron!: per-lane Kronecker product into dout. Lane 1 perturbs x1
        # by x1 itself, lane 2 perturbs x2 by x2: both give kron(x1, x2) as the partial.
        x1m, x2m, outm = [1.0 2.0], reshape([3.0, 4.0], 2, 1), Matrix{Float64}(undef, 2, 2)
        arr_slot(m, p1, p2) = sl(2, m, NDualArray{Float64,2,2,Matrix{Float64}}(m, (p1, p2)))
        r_kron = frule!!(
            sl(2, LinearAlgebra._kron!),
            arr_slot(outm, zeros(2, 2), zeros(2, 2)),
            arr_slot(x1m, copy(x1m), zeros(1, 2)),
            arr_slot(x2m, zeros(2, 1), copy(x2m)),
        )
        @test outm == [3.0 6.0; 4.0 8.0]
        @test r_kron.value.partials[1] == [3.0 6.0; 4.0 8.0]
        @test r_kron.value.partials[2] == [3.0 6.0; 4.0 8.0]
    end

    @testset "MutableDualTangentView (NDual field)" begin
        r = LiftedTest_RefF(3.0)
        slot = zero_lifted(Val(2), r)
        view = tangent(slot, 1)  # per-lane view
        @test view isa MutableDualTangentView
        @test getfield(view, :_parent) === slot.value
        @test getfield(view, :_primal) === r
        @test getfield(view, :_lane) === 1
        @test view.v === 0.0          # read: lane-1 partial of field `v`
        view.v = 5.0                  # write: routes back to parent.value via setfield!
        @test view.v === 5.0
        @test slot.value.value.v.partials === (5.0, 0.0)
        @test tangent(slot, 2).v === 0.0  # other lane unchanged

        # A user field literally named `parent` must resolve to its lane tangent, not the
        # view's internal `_parent` (why the internals are underscore-prefixed).
        pview = tangent(zero_lifted(Val(2), LiftedTest_ParentField(3.0)), 1)
        @test pview.parent === 0.0
        pview.parent = 7.0
        @test pview.parent === 7.0
    end

    @testset "element-wise Vector with abstract eltype (concrete struct elements)" begin
        # Regression: each element's lane tangent must be extracted via the CONCRETE
        # `typeof(pe)` — the abstract static `eltype(P)` has no fields, so the struct-lift
        # previously threw "type ... has no field μ" (the distributions-1.10 failure).
        v = LiftedTest_AbsScalar[
            LiftedTest_ConcScalar(1.0, 2.0), LiftedTest_ConcScalar(3.0, 4.0)
        ]
        slot = zero_lifted(Val(2), v)
        @test slot isa Lifted{Vector{LiftedTest_AbsScalar},2}
        for lane in 1:2
            t = tangent(slot, lane)
            @test t isa AbstractVector && length(t) == 2
            @test t[1] isa Tangent
            nt = getfield(t[1], :fields)
            @test nt.μ === 0.0 && nt.σ === 0.0
        end
        # Width-1 boundary unpack (the `unlift` path used by `test_frule_correctness`).
        _, ts = unlift(zero_lifted(Val(1), v))
        @test ts isa AbstractVector && ts[1] isa Tangent
    end

    @testset "frule!! one-to-one parallels (builtins.jl intrinsics)" begin
        IW = IntrinsicsWrappers

        # End-to-end through the interpreter (widths 1,2,3 + FD via test_rule).
        if isdefined(@__MODULE__, :test_rule)
            for args in Any[
                (IW.abs_float, -3.0),
                (IW.add_float, 1.0, 2.0),
                (IW.add_float_fast, 1.0, 2.0),
                (IW.copysign_float, 2.0, -3.0),
                (IW.div_float, 6.0, 2.0),
                (IW.div_float_fast, 6.0, 2.0),
                (IW.mul_float, 1.5, 2.5),
                (IW.mul_float_fast, 1.5, 2.5),
                (IW.neg_float, 4.0),
                (IW.neg_float_fast, 4.0),
                (IW.sub_float, 5.0, 2.0),
                (IW.sub_float_fast, 5.0, 2.0),
                (IW.fma_float, 1.5, 2.0, 0.5),
                (IW.muladd_float, 1.5, 2.0, 0.5),
                (IW.fpext, Float64, 1.5f0),
                (IW.fptrunc, Float32, 1.5),
            ]
                test_rule(MersenneTwister(0), args...; perf_flag=:none)
            end
        end

        # Direct Lifted-arg checks of the per-lane partials (lane k seeds input k; each
        # row also asserts the inner-value invariant `.value === primal`). Args are
        # `(value, lane1_seed, lane2_seed)` triples.
        @testset "$op" for (op, args, ep, et) in Any[
            (IW.abs_float, ((-3.0, 1.0, -1.0),), 3.0, (-1.0, 1.0)),       # dy = sign(x) dx
            (IW.add_float, ((1.0, 1.0, 0.0), (2.0, 0.0, 1.0)), 3.0, (1.0, 1.0)),
            (IW.add_float_fast, ((1.0, 1.0, 0.0), (2.0, 0.0, 1.0)), 3.0, (1.0, 1.0)),
            # copysign: dz = sign(x) sign(y) dx, ∂/∂y = 0. The x<0 row is the regression
            # for the bug that dropped the sign(x) factor (only visible for x < 0).
            (IW.copysign_float, ((2.0, 1.0, 0.0), (-3.0, 0.0, 1.0)), -2.0, (-1.0, 0.0)),
            (IW.copysign_float, ((-2.0, 1.0, 0.0), (-3.0, 0.0, 1.0)), -2.0, (1.0, 0.0)),
            # div: ∂(a/b)/∂a = 1/b, ∂/∂b = -a/b².
            (IW.div_float, ((6.0, 1.0, 0.0), (2.0, 0.0, 1.0)), 3.0, (0.5, -1.5)),
            (IW.div_float_fast, ((6.0, 1.0, 0.0), (2.0, 0.0, 1.0)), 3.0, (0.5, -1.5)),
            # mul: product rule.
            (IW.mul_float, ((1.5, 1.0, 0.0), (2.5, 0.0, 1.0)), 3.75, (2.5, 1.5)),
            (IW.mul_float_fast, ((1.5, 1.0, 0.0), (2.5, 0.0, 1.0)), 3.75, (2.5, 1.5)),
            (IW.neg_float, ((4.0, 1.0, -2.0),), -4.0, (-1.0, 2.0)),
            (IW.neg_float_fast, ((4.0, 1.0, -2.0),), -4.0, (-1.0, 2.0)),
            (IW.sub_float, ((5.0, 1.0, 0.0), (2.0, 0.0, 1.0)), 3.0, (1.0, -1.0)),
            (IW.sub_float_fast, ((5.0, 1.0, 0.0), (2.0, 0.0, 1.0)), 3.0, (1.0, -1.0)),
            # fma/muladd: ∂(xy+z)/∂x = y, ∂/∂y = x (lane 1 seeds dx, lane 2 dy).
            (
                IW.fma_float,
                ((1.5, 1.0, 0.0), (2.0, 0.0, 1.0), (0.5, 0.0, 0.0)),
                3.5,
                (2.0, 1.5),
            ),
            (
                IW.muladd_float,
                ((1.5, 1.0, 0.0), (2.0, 0.0, 1.0), (0.5, 0.0, 0.0)),
                3.5,
                (2.0, 1.5),
            ),
        ]
            slots = map(a -> sl(2, a[1], nd(a...)), args)
            r = frule!!(sl(2, op), slots...)
            @test typeof(r) === Lifted{typeof(ep),2,NDual{typeof(ep),2}}
            @test primal(r) === ep
            @test tangent(r).value === ep  # inner-value invariant
            @test tangent(r).partials == et  # `==`: a -0.0 partial matches 0.0
        end

        # fpext / fptrunc: cross-precision lifts change the dual's scalar type.
        r_ext = frule!!(
            sl(2, IW.fpext), sl(2, Float64), sl(2, 1.5f0, nd(1.5f0, 1.0f0, 0.0f0))
        )
        @test typeof(r_ext) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(r_ext) === 1.5 && tangent(r_ext).partials === (1.0, 0.0)
        r_tr = frule!!(sl(2, IW.fptrunc), sl(2, Float32), sl(2, 1.5, nd(1.5, 1.0, 0.0)))
        @test typeof(r_tr) === Lifted{Float32,2,NDual{Float32,2}}
        @test primal(r_tr) === 1.5f0 && tangent(r_tr).partials === (1.0f0, 0.0f0)
    end

    @testset "frule!! one-to-one parallels (rules_via_nfwd.jl)" begin
        # Representative coverage of each registration pattern in the file
        # (tanpi away from its 1.5 singularity).
        if isdefined(@__MODULE__, :test_rule)
            for args in Any[
                (exp, 1.5),
                (log, 1.5),
                (sin, 1.5),
                (cos, 1.5),
                (sqrt, 1.5),
                (cbrt, 1.5),
                (tanpi, 0.1),
                (atan, 1.0, 2.0),
                (^, 2.0, 3.0),
                (max, 1.5, 0.5),
                (Base.FastMath.pow_fast, 2.0, 3),
                (clamp, 0.5, 0.0, 1.0),
                (sincos, 1.0),
                (sincosd, 30.0),
                (sincospi, 0.25),
                (modf, 1.7),
                (hypot, 3.0, 4.0),
                (hypot, 1.0, 2.0, 2.0),
            ]
                test_rule(MersenneTwister(0), args...; perf_flag=:none)
            end
        end

        # Direct Lifted-arg invocation for one unary representative.
        r = frule!!(sl(2, sin), sl(2, 1.0, nd(1.0, 1.0, 0.0)))
        @test typeof(r) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(r) === sin(1.0)
        @test tangent(r).partials[1] ≈ cos(1.0)
        @test tangent(r).partials[2] == 0.0
    end

    @testset "frule!! one-to-one parallels (tasks.jl)" begin
        task = Task(() -> nothing)
        # A Task field is non-differentiable: forward V is `NoDual` (the forward-mode
        # sentinel; reverse would use `NoTangent`).
        task_slot = sl(2, task, TaskTangent())
        r = frule!!(sl(2, lgetfield), task_slot, sl(2, Val(:rngState1)))
        @test primal(r) === getfield(task, :rngState1)
        @test tangent(r) === NoDual()
        r2 = frule!!(sl(2, getfield), task_slot, sl(2, :rngState1))
        @test primal(r2) === getfield(task, :rngState1)
        @test tangent(r2) === NoDual()
    end

    @testset "frule!! one-to-one parallels (new.jl _new_)" begin
        # Immutable struct branch → ImmutableDual V; mutable branch → MutableDual V.
        r_imm = frule!!(
            sl(2, _new_),
            sl(2, LiftedTest_Point),
            sl(2, 1.5, nd(1.5, 1.0, 0.0)),
            sl(2, 2.5, nd(2.5, 0.0, 1.0)),
        )
        @test typeof(r_imm) ===
            Lifted{LiftedTest_Point,2,dual_type(Val(2), LiftedTest_Point)}
        @test primal(r_imm) === LiftedTest_Point(1.5, 2.5)
        @test tangent(r_imm) isa ImmutableDual

        r_mut = frule!!(
            sl(2, _new_), sl(2, LiftedTest_RefF), sl(2, 7.0, nd(7.0, 1.0, -1.0))
        )
        @test typeof(r_mut) === Lifted{LiftedTest_RefF,2,dual_type(Val(2), LiftedTest_RefF)}
        @test primal(r_mut).v === 7.0
        @test tangent(r_mut) isa MutableDual
    end

    @testset "frule!! one-to-one parallels (iddict.jl)" begin
        # Constructor, then setindex! + getindex round trip.
        r_ctor = frule!!(sl(2, IdDict{Symbol,Float64}))
        @test typeof(r_ctor) === lifted_type(Val(2), IdDict{Symbol,Float64})
        @test isempty(primal(r_ctor)) && isempty(tangent(r_ctor))

        d_primal, d_tan = IdDict{Symbol,Float64}(), IdDict{Symbol,NDual{Float64,2}}()
        d_slot = sl(2, d_primal, d_tan)
        frule!!(sl(2, setindex!), d_slot, sl(2, 3.0, nd(3.0, 1.0, -1.0)), sl(2, :a))
        @test d_primal[:a] === 3.0
        @test d_tan[:a].partials === (1.0, -1.0)
        r_gi = frule!!(sl(2, getindex), d_slot, sl(2, :a))
        @test primal(r_gi) === 3.0
        @test tangent(r_gi).partials === (1.0, -1.0)
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "frule!! one-to-one parallels (memory.jl)" begin
            # Memory{P}(undef, n) constructor → zeroed parallel-arrays V.
            r_mem = frule!!(sl(2, Memory{Float64}), sl(2, undef), sl(2, 3))
            @test typeof(r_mem) === lifted_type(Val(2), Memory{Float64})
            @test length(primal(r_mem)) == 3
            @test all(iszero, tangent(r_mem).partials[1])
            @test all(iszero, tangent(r_mem).partials[2])

            # memoryrefnew(::Memory) → MemoryRef slot.
            r_ref = frule!!(sl(2, Core.memoryrefnew), r_mem)
            @test typeof(r_ref) === lifted_type(Val(2), MemoryRef{Float64})
            @test primal(r_ref) === Core.memoryref(primal(r_mem))

            # lmemoryrefget — read an NDual back out at position 1.
            r_get = frule!!(
                sl(2, Mooncake.lmemoryrefget),
                r_ref,
                sl(2, Val(:not_atomic)),
                sl(2, Val(false)),
            )
            @test typeof(r_get) === Lifted{Float64,2,NDual{Float64,2}}
            @test primal(r_get) === primal(r_mem)[1]
        end
    end

    @testset "frule!! one-to-one parallels (threads.jl)" begin
        # `_foreigncall_`'s strict signature makes a representative call fragile;
        # verify the Lifted threading rules are registered instead.
        sigs = [sprint(show, m) for m in methods(frule!!)]
        @test any(s -> occursin("jl_in_threaded_region", s) && occursin("Lifted", s), sigs)
    end

    @testset "type-stability" begin
        # The canonical width-N path is type-stable for IEEEFloat primals.
        @test @inferred(zero_dual(Val(2), 1.0)) isa NDual{Float64,2}
        @test @inferred(zero_lifted(Val(2), 1.0)) isa Lifted{Float64,2,NDual{Float64,2}}
        @test @inferred(dual_type(Val(2), Float64)) === NDual{Float64,2}
        @test @inferred(lifted_type(Val(2), Float64)) === Lifted{Float64,2,NDual{Float64,2}}
    end

    @testset "basis_lifted!!" begin
        # `basis_lifted!!(zero_lifted(...), slots)` sets lane k hot at the slots[k]-th
        # scalar dof (counted in `dof`/`zero_tangent` order), mutating mutable V in place
        # and rebuilding immutable V.
        bl(x, slots) = basis_lifted!!(zero_lifted(Val(length(slots)), x), slots)

        @test tangent(bl(3.0, (1,)), 1) == 1.0
        @test tangent(bl([5.0, 6.0, 7.0], (2,)), 1) == [0.0, 1.0, 0.0]
        @test tangent(bl(1.0 + 2.0im, (2,)), 1) == 0.0 + 1.0im  # imag dof
        let t = tangent(bl(([1.0, 2.0], 9.0), (3,)), 1)
            @test t[1] == [0.0, 0.0] && t[2] == 1.0  # the scalar is dof 3
        end

        # width-2: two basis directions in one seed.
        let b = bl([5.0, 6.0, 7.0], (1, 3))
            @test tangent(b, 1) == [1.0, 0.0, 0.0]
            @test tangent(b, 2) == [0.0, 0.0, 1.0]
        end

        # Aliased fields: `dof` dedups the shared array, so both fields share one V.
        shared = [10.0, 20.0]
        let nt = bl(LiftedTest_Aliased(shared, shared), (1,)).value.value
            @test nt.a === nt.b
            @test nt.a.partials[1] == [1.0, 0.0]
        end

        # Aliasing on the `lift` (reverse→forward) path, non-float-element array: the
        # element-wise array lift must register the shared array in the cache so both
        # fields get one V (matching reverse). Float-element fields are safe via
        # `ẋ`-aliasing; this exercises the gap.
        let h = LiftedTest_AliasedNested([[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0]])
            h.b = h.a
            nt = tangent(lift(h, randn_tangent(Xoshiro(1), h))).value
            @test nt.a === nt.b
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

    @testset "test_lifted (representation interface)" begin
        # Drive the forward representation contract over the SAME canonical primal table
        # that `test_tangent` / `test_data` use for reverse mode, so the two stay in sync.
        @testset "$(typeof(p))" for (interface_only, p, t...) in
                                    Mooncake.tangent_test_cases()
            test_lifted(Xoshiro(123456), p)
        end

        @testset "cyclic mutable struct" begin
            test_lifted(Xoshiro(1), make_circular_reference_struct())
        end

        @testset "self-referential plain array" begin
            # The whole representation battery (seed, per-lane extraction, lift/unlift) is
            # cycle-aware for plain-array cycles, not just the seed.
            x = Any[]
            push!(x, x)
            test_lifted(Xoshiro(1), x)
        end

        # Type-level widening / sentinel cases the value-drive cannot reach (abstract and
        # `Union` primals). Free-TypeVar `Tuple` phantoms are exercised in `codual.jl`.
        @testset "test_lifted_type $P (N=$N)" for P in (
                Real,
                AbstractVector{Float64},
                Union{Float64,Float32},
                Union{Float64,Int},
                Complex{Float64},
                Vector{Any},
            ),
            N in (1, 2, 3)

            test_lifted_type(P, Val(N))
        end
    end
end
