# True when `x` reaches itself. Only ancestors count, so a value merely shared between two
# fields (a DAG) is not self-referential — hence the removal on the way out.
function _self_referential(@nospecialize(x), seen=Base.IdSet{Any}())
    (isbits(x) || x isa Type || x isa Symbol) && return false
    x in seen && return true
    push!(seen, x)
    try
        if x isa AbstractArray && !isbitstype(eltype(x))
            for i in eachindex(x)
                isassigned(x, i) && _self_referential(x[i], seen) && return true
            end
        else
            for f in 1:nfields(x)
                isdefined(x, f) && _self_referential(getfield(x, f), seen) && return true
            end
        end
    finally
        delete!(seen, x)
    end
    return false
end

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
mutable struct LiftedTest_ParentField  # field names collide with the view's internals
    parent::Float64
    _parent::Float64
    _primal::Float64
    _lane::Float64
end
mutable struct LiftedTest_TupleField  # a tuple field has no per-lane view either
    t::Tuple{Float64,Float64}
end
mutable struct LiftedTest_ComplexField
    z::ComplexF64
end
mutable struct LiftedTest_AbstractField  # abstract field type -> dual field NamedTuple is abstract
    x::Real
end
# As above, but the abstract field HOLDS a structured value rather than a leaf. `x::Real` above
# only ever holds a `Float64`, whose child slot never has to answer a structural question, so a
# declared-type annotation is harmless there and hides the recursion bug. Immutable, so the lane
# tangent is a `Tangent` built through `_field_lane_tangent`; a mutable one would route `.x`
# through `_lane_tangent`, which handles only scalar `NDual` V — a separate documented gap.
struct LiftedTest_AbstractHeld
    x::Any
end
struct LiftedTest_RefContainer  # immutable struct holding a Ref (reverse tangent field = MutableTangent)
    r::Base.RefValue{Float64}
end
mutable struct LiftedTest_MaybeInit
    x::Float64
    y::Float64
    LiftedTest_MaybeInit(x::Float64) = new(x)
end
mutable struct LiftedTest_MaybeInitHeap  # non-bitstype field genuinely undefined via 1-arg ctor
    x::Float64
    y::Vector{Float64}
    LiftedTest_MaybeInitHeap(x::Float64) = new(x)
end
struct LiftedTest_TwoArrays  # two fields that may alias one mutable (non-float-eltype) array
    p::Vector{Vector{Float64}}
    q::Vector{Vector{Float64}}
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
    TaskTangent

# Shorthands for the table-driven type tests below (top level: parametric aliases are
# not allowed in local scope).
const ND = NDual
# The array V's 6th parameter `B` (= `_block_type(A)`) can't be derived inside a type alias (no
# function calls on a free typevar), so spell the array V shapes the table below needs as named
# consts.
const NDA_VecF64 = NDualArray{
    Float64,2,1,Vector{Float64},NDual{Float64,2},Mooncake.NDualBlock{Float64,2}
}
const NDA_MatF32 = NDualArray{
    Float32,1,2,Matrix{Float32},NDual{Float32,1},Mooncake.NDualBlock{Float32,3}
}
const NDAC_VecC64 = NDualArray{
    ComplexF64,
    2,
    1,
    Vector{ComplexF64},
    Complex{NDual{Float64,2}},
    Mooncake.NDualBlock{ComplexF64,2},
}

@testset "lifted" begin
    # Slot/inner-dual shorthands. `sl` wraps once at the top level with the sharp
    # `Base._stable_typeof` P (`Type{T}` for type-valued primals, matching rule dispatch).
    sl(N, p, v=NoDual()) = Lifted{Base._stable_typeof(p),N}(p, v)
    nd(v::T, parts::Vararg{T,N}) where {T,N} = NDual{T,N}(v, parts)

    @testset "cyclic MutableDual tangent arithmetic" begin
        # A self-referential mutable struct lifts to a cyclic `MutableDual` V; the
        # tangent-arithmetic helpers must terminate via their aliasing caches.
        n = LiftedTest_Cycle(nothing, 2.0)
        n.next = n
        v = tangent(lift(n, zero_tangent(n)))
        @test v.fields.next === v                    # cyclic V built, no overflow
        @test Mooncake._dot(v, v) == 0.0
        s = Mooncake._scale(2.0, v)
        @test s.fields.next === s                    # scale preserves the cycle
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

    @testset "one `Ref` reached twice lifts to one V" begin
        # Every `lift` leaf that owns partial storage registers in the threaded cache, so two
        # occurrences of one primal share a V. `Ref`'s leaf took the cache and discarded it, giving
        # two independent partials buffers over one primal: a write through one occurrence updated
        # the shared value but only its own partials, and the read through the other returned the
        # new value with a stale partial. Not expressible through `test_lifted`, which drives the
        # factories over ONE value and so cannot state a property about two slots.
        mk() = (r=Ref(1.0); (r, r))
        p = mk()
        v = Mooncake.tangent(Mooncake.lift(p, Mooncake.zero_tangent(p)))
        @test v[1] === v[2]
        @test getfield(v[1], :partials) === getfield(v[2], :partials)
        # The seed factory always agreed; the two entry points must not disagree.
        z = Mooncake.tangent(Mooncake.zero_lifted(Val(1), mk()))
        @test z[1] === z[2]
        # End to end: `g` is `3r`, so the derivative is 3.0, and reverse mode agrees.
        g(q) = (q[1][]=q[1][] * 3.0; q[2][])
        dp = Mooncake.zero_tangent(p)
        Ft = typeof(dp[1].fields)
        dp[1].fields = Ft((Mooncake.PossiblyUninitTangent{Float64}(1.0),))
        cache = Mooncake.prepare_derivative_cache(g, p)
        @test Mooncake.value_and_derivative!!(cache, (g, Mooncake.NoTangent()), (p, dp)) ==
            (3.0, 3.0)
    end

    @static if VERSION < v"1.11-"
        @testset "1.10 forward seed and lift alias two arrays over one buffer" begin
            # 1.11+ windows the backing `Memory`'s block, which 1.10 has no equivalent of, so both
            # the seed and the lift keyed by ARRAY OBJECT and gave `a` and `reshape(a)` independent
            # partials while the primal still aliased. The reverse side got storage keying earlier;
            # this is the forward half of the same problem.
            fm(x, y) = (x[1] *= 3.0; y[1])
            mkv() = collect(1.0:4.0)
            a = mkv()
            b = reshape(a, 2, 2)
            cache = Mooncake.prepare_derivative_cache(fm, a, b)
            aa = mkv()
            bb = reshape(aa, 2, 2)
            # The tangents must ALIAS exactly as the primals do. One buffer carries one
            # direction, so two independent tangent arrays over aliased primals is ill-posed and
            # refused; `reshape` shares `da`'s storage, where `copy` would not. They cannot be the
            # same OBJECT here -- one is a vector, the other a matrix -- which is why the rule is
            # shared storage rather than identity.
            da = [1.0, 0.0, 0.0, 0.0]
            db = reshape(da, 2, 2)
            # `fm` over one buffer is `3*a[1]`, so the directional derivative along e1 is 3.0.
            @test Mooncake.value_and_derivative!!(
                cache, (fm, Mooncake.NoTangent()), (aa, da), (bb, db)
            ) == (3.0, 3.0)
            # Distinct arrays must not be merged by the storage key.
            a2, b2 = mkv(), collect(5.0:8.0)
            c2 = Mooncake.prepare_derivative_cache(fm, a2, b2)
            @test Mooncake.value_and_derivative!!(
                c2,
                (fm, Mooncake.NoTangent()),
                (mkv(), [1.0, 0.0, 0.0, 0.0]),
                (collect(5.0:8.0), zeros(4)),
            ) == (5.0, 0.0)
        end
    end

    @testset "one array reached twice through a `SimpleVector` lifts to one V" begin
        # Same defect as the `Ref` leaf above, one leaf over. `SimpleVector` had only a two-argument
        # `lift`, so the three-argument call fell to the generic passthrough, which DISCARDS the
        # cache — and its elements were lifted through the two-argument form too, so no cache
        # existed anywhere below it. A type with no three-argument method at all does not show up in
        # a search for methods that ignore their cache argument.
        v = [[1.0, 2.0]]
        sv = Core.svec(v, v)
        dv = [[1.0, 0.0]]
        f(s) = (Core._svec_ref(s, 1)[1][1] *= 3.0; Core._svec_ref(s, 2)[1][1])
        cache = Mooncake.prepare_derivative_cache(
            f, sv; config=Mooncake.Config(; friendly_tangents=false)
        )
        @test Mooncake.value_and_derivative!!(
            cache, (f, Mooncake.NoTangent()), (sv, Any[dv, dv])
        ) == (3.0, 3.0)
    end

    @testset "scalar NDual _add_to_primal adds only the partials" begin
        # Regression: an inner `NDual`'s `.value` is the primal it shadows (inner-value
        # invariant), so `_add_to_primal` must add only the partials — adding `.value` too would
        # double-count the primal (a zero-partials V would return `2x` instead of the identity `x`).
        x = 3.0
        @test Mooncake._add_to_primal(x, nd(x, 0.0, 0.0)) == x        # zero partials → identity
        @test Mooncake._add_to_primal(x, nd(x, 1.0, 2.0)) == x + 3.0  # adds sum(partials) = 3
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
        (2, Vector{Float64}, NDA_VecF64),
        (1, Matrix{Float32}, NDA_MatF32),
        # Complex scalars and arrays.
        (2, Complex{Float64}, Complex{ND{Float64,2}}),
        (1, Complex{Float32}, Complex{ND{Float32,1}}),
        (2, Vector{ComplexF64}, NDAC_VecC64),
        # Tuples / NamedTuples: element-wise recursion.
        (2, Tuple{Float64}, Tuple{ND{Float64,2}}),
        (2, Tuple{Float64,Float32}, Tuple{ND{Float64,2},ND{Float32,2}}),
        (2, Tuple{Float64,Vector{Float64}}, Tuple{ND{Float64,2},NDA_VecF64}),
        (
            2,
            NamedTuple{(:a, :b),Tuple{Float64,Float32}},
            NamedTuple{(:a, :b),Tuple{ND{Float64,2},ND{Float32,2}}},
        ),
        (
            2,
            NamedTuple{(:x, :y),Tuple{Float64,Vector{Float64}}},
            NamedTuple{(:x, :y),Tuple{ND{Float64,2},NDA_VecF64}},
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

    @testset "abstract field holding a structured value" begin
        # The child slot must be annotated with the field's CONCRETE type, as the element-wise array
        # recursion does. With the declared `Any` the child is `Lifted{Any,N,...}`, and the lane and
        # unlift methods dispatch on `P`: they evaluate `fieldtype(Any, name)` and throw for a struct,
        # find no method for a NamedTuple V, and — since `Any` is not `<:Tuple` — fall to the
        # per-lane-`Ptr` method for a Tuple V, which indexes the tuple by LANE instead of returning
        # that lane's partials.
        # Assert the recursion, not the wrapper: a mutable struct's lane tangent is a
        # `MutableDualTangentView`, so the field's own tangent is what carries the evidence.
        s = zero_lifted(Val(1), LiftedTest_AbstractHeld(LiftedTest_Point(1.0, 2.0)))
        @test tangent(s, 1).fields.x isa Mooncake.Tangent
        @test unlift(s) isa Tuple
        # A lane index beyond the held tuple's length: the per-lane-`Ptr` misdispatch returned that
        # element's whole V and raised `BoundsError` past the arity, so this pins the Tuple branch.
        st = zero_lifted(Val(3), LiftedTest_AbstractHeld((1.0, 2.0)))
        @test length(tangent(st, 3).fields.x) == 2
    end

    @testset "a per-lane Tuple V unlifts to its lane, not element-wise" begin
        # `value_and_derivative!!` unlifts its output slot, so this is every differentiated
        # function returning a raw pointer. Both Vs are per-lane copies of one leaf.
        x, t = [1.0, 2.0], [3.0, 4.0]
        @test unlift(Lifted{Ptr{Float64},1}(pointer(x), (pointer(t),))) ===
            (pointer(x), pointer(t))
        w = Base.TwicePrecision{Float64}(1.0, 0.0)
        @test unlift(Lifted{typeof(w),1}(w, (w,))) === (w, w)
    end

    @testset "the unlift terminal refuses a non-leaf" begin
        # The terminal hands back the lane accessor, which is the reverse tangent only for a leaf.
        # An aggregate V with no `_unlift_seed` of its own silently took that path and failed
        # several frames later inside reverse tangent arithmetic; it now says so at the boundary.
        s = TestResources.StructFoo(6.0, [1.0, 2.0])
        @test_throws ArgumentError unlift(
            Lifted{Tuple{typeof(s)},1,Tuple{NoDual}}((s,), (NoDual(),))
        )
        # An ARRAY of `Ptr` to a non-differentiable element takes the all-`NoDual` fast path,
        # whose lane accessor is the reverse tangent only when the ELEMENT's tangent is
        # `NoTangent`. Here it is `Ptr{NoTangent}`, so the accessor's `Vector{NoTangent}` is the
        # wrong shape and the element-wise path has to rebuild it.
        pv = [Ptr{Int}(0), Ptr{Int}(0)]
        @test last(unlift(Mooncake.zero_lifted(Val(1), pv))) isa
            Mooncake.tangent_type(Vector{Ptr{Int}})
        # A `Ptr` nested in an aggregate reaches the terminal too, and rebuilds the reverse
        # placeholder rather than handing back the raw lane address.
        x = [1.0]
        @test Mooncake._unlift_seed(
            Lifted{Ptr{Nothing},1}(
                Ptr{Nothing}(UInt(pointer(x))), (Ptr{Nothing}(UInt(pointer(x))),)
            ),
            IdDict(),
        ) isa Mooncake.VoidPtrTangent
    end

    @testset "metatype kinds get an unbounded slot" begin
        # A type-valued result inferred as its KIND is wrapped at runtime as `Lifted{Type{X}}`, and
        # `Lifted` is invariant in `P`, so a bounded slot rejects it. All four kinds are
        # `isconcretetype`, so each needs the carve-out — naming only `DataType` left the other
        # three with bounded slots that no runtime value satisfies.
        for T in (DataType, UnionAll, Union, Core.TypeofBottom), N in (1, 3)
            @test lifted_type(Val(N), T) isa UnionAll
        end
        # The carve-out must not swallow ordinary concrete types, whose bounded slots are exact.
        @test lifted_type(Val(1), Float64) === Lifted{Float64,1,ND{Float64,1}}
        @test lifted_type(Val(1), Int) === Lifted{Int,1,NoDual}
        @test lifted_type(Val(1), Type{Float64}) === Lifted{Type{Float64},1,NoDual}
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
        @test all(iszero, tangent_view(va, 1)) && all(iszero, tangent_view(va, 2))
        @test va[1] === nd(1.0, 0.0, 0.0)
        @test any(
            !iszero, tangent_view(randn_dual(Val(2), Random.MersenneTwister(0), x), 1)
        )
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
        @test all(iszero, tangent_view(vc, 1)) && all(iszero, tangent_view(vc, 2))
        @test typeof(zero_lifted(Val(2), xc)) === lifted_type(Val(2), Vector{ComplexF64})

        # Complex MemoryRef (1.11+): the forward seed factories must produce the canonical
        # NDualMemoryRef{Complex} that dual_type advertises (and that reverse zero_tangent supports).
        # Previously only float MemoryRefs had factories, so a complex one fell to the generic
        # @generated factory and threw a confusing nested `memoryref` MethodError.
        @static if VERSION >= v"1.11-"
            mrc = ComplexF64[1.0 + 2.0im, 3.0 - 1.0im].ref
            DTc = dual_type(Val(2), typeof(mrc))
            @test typeof(tangent(zero_lifted(Val(2), mrc))) === DTc
            @test typeof(tangent(Mooncake.uninit_lifted(Val(2), mrc))) === DTc
            @test typeof(tangent(Mooncake.randn_lifted(Val(2), Xoshiro(1), mrc))) === DTc
        end

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
        @test vp.fields.x === nd(1.0, 0.0, 0.0) && vp.fields.y === nd(2.0, 0.0, 0.0)
        zp = zero_lifted(Val(2), p)
        @test typeof(zp) === lifted_type(Val(2), LiftedTest_Point)
        @test primal(zp) === p && tangent(zp) === vp
        vm = zero_dual(Val(2), LiftedTest_RefF(3.0))
        @test typeof(vm) === dual_type(Val(2), LiftedTest_RefF)
        @test vm.fields.v === nd(3.0, 0.0, 0.0)
    end

    @testset "NDualArray accessors + AbstractArray interface" begin
        x = [1.0, 2.0, 3.0]
        a = NDualArray{Float64,2,1,Vector{Float64}}(
            x, (similar(x).=[0.5, -0.5, 1.5], similar(x).=[0.0, 1.0, -1.0])
        )
        @test a isa AbstractArray{NDual{Float64,2},1}
        @test size(a) == (3,) && length(a) == 3
        @test primal(a) === x && tangent(a) == ([0.5, -0.5, 1.5], [0.0, 1.0, -1.0])
        @test unpack_ndual(a) == (a.primal, tangent(a))
        @test a[2] === nd(2.0, -0.5, 1.0)  # lazy getindex
        a[1] = nd(9.0, 7.0, -7.0)          # setindex! writes both channels
        @test x[1] === 9.0 &&
            tangent_view(a, 1)[1] === 7.0 &&
            tangent_view(a, 2)[1] === -7.0
    end

    # Regression: the per-lane tangent of a `Ref{<:IEEEFloat}` (NDualRef V) must be the
    # reverse-oracle shape — a `MutableTangent{@NamedTuple{x::PossiblyUninitTangent{P}}}` (a `Ref` is a
    # mutable struct) — NOT the bare lane partial. Returning the scalar diverged from `unlift`/the
    # reverse oracle and made struct-recursion field extraction (which converts each field into its
    # declared reverse tangent) throw a convert `MethodError` for a `Ref`-valued field.
    @testset "NDualRef per-lane tangent is reverse-shaped" begin
        # Bare Ref: per-lane shape must equal the width-1 unlift (reverse) shape.
        sref = zero_lifted(Val(2), Ref(3.0))
        Tt = Mooncake.tangent_type(Base.RefValue{Float64})
        @test tangent(sref, 1) isa Tt
        @test tangent(sref, 2) isa Tt
        _, t1 = unlift(zero_lifted(Val(1), Ref(3.0)))
        @test typeof(tangent(sref, 1)) === typeof(t1)
        # Immutable struct with a Ref field: field extraction must not throw and must yield the
        # declared reverse Tangent shape.
        s2 = zero_lifted(Val(1), LiftedTest_RefContainer(Ref(2.5)))
        tv = tangent(s2, 1)
        @test tv isa Tangent
        @test getfield(tv, :fields).r isa Tt
    end

    @static if VERSION >= v"1.11-"
        @testset "aliased Memory shares one V, on both the seed and lift paths" begin
            # The V packs a fresh block per call, so without the aliasing cache two fields holding
            # one `Memory` get independent blocks: a mutation through one is invisible through the
            # other and `dof` counts the shared storage twice. The float `Array` overloads honour the
            # cache; these two did not.
            mem = Memory{Float64}(undef, 1)
            mem[1] = 3.0
            d = IdDict{Any,Any}()
            @test Mooncake._zero_dual_internal(Val(1), mem, d) ===
                Mooncake._zero_dual_internal(Val(1), mem, d)
            dmem = Mooncake.zero_tangent(mem)
            c = IdDict{Any,Any}()
            @test Mooncake.lift(mem, dmem, c) === Mooncake.lift(mem, dmem, c)
            # A `MemoryRef` into it, same requirement.
            r = Core.memoryref(mem, 1)
            dr = Mooncake.zero_tangent(r)
            cr = IdDict{Any,Any}()
            @test Mooncake.lift(r, dr, cr) === Mooncake.lift(r, dr, cr)
            # `randn_dual`'s twin shares the shape, so it shares the requirement.
            dr2 = IdDict{Any,Any}()
            @test Mooncake._randn_dual_internal(Val(1), Xoshiro(1), mem, dr2) ===
                Mooncake._randn_dual_internal(Val(1), Xoshiro(1), mem, dr2)
        end

        @testset "an Array's block windows its backing Memory's" begin
            # The primal aliasing was already preserved (`a.ref.mem === mem`) while the partials
            # were not: two independent derivative stores over one buffer, so a lane written
            # through the array was invisible through the `Memory`. Value right, derivative
            # wrong. `test_rule` cannot catch this — its finite-difference oracle perturbs the
            # primal through the same aliasing-blind seed it hands the rule, so both agree.
            a = [1.0, 2.0, 3.0]
            v = tangent(zero_lifted(Val(2), (a, a.ref.mem)))
            Mooncake.Nfwd.tangent_view(v[1], 1)[2] = 5.0
            @test Mooncake.Nfwd.tangent_view(v[2], 1)[2] == 5.0
            Mooncake.Nfwd.tangent_view(v[2], 2)[3] = 9.0
            @test Mooncake.Nfwd.tangent_view(v[1], 2)[3] == 9.0

            # An array that does not start at slot 1 of its `Memory` must window at its offset.
            b = Float64[]
            foreach(i -> push!(b, i), 1:5)
            Base._growbeg!(b, 3)
            Base._deletebeg!(b, 3)
            off = Core.memoryrefoffset(b.ref)
            @test off > 1
            vb = tangent(zero_lifted(Val(2), (b, b.ref.mem)))
            Mooncake.Nfwd.tangent_view(vb[1], 1)[1] = 4.0
            @test Mooncake.Nfwd.tangent_view(vb[2], 1)[off] == 4.0
            @test Mooncake.Nfwd.tangent_view(vb[2], 1)[1] == 0.0   # the slot before the array is untouched

            # The lift path shares the requirement, and its input satisfies it: `zero_tangent`
            # over the aliased pair returns a tangent with the same `Memory` geometry.
            t = (a, a.ref.mem)
            vl = tangent(Mooncake.lift(t, Mooncake.zero_tangent(t), IdDict{Any,Any}()))
            Mooncake.Nfwd.tangent_view(vl[1], 1)[2] = 6.0
            @test Mooncake.Nfwd.tangent_view(vl[2], 1)[2] == 6.0

            # Growth needs no bookkeeping: the block's `Memory` is the primal's scaled by the
            # chunk width, so the slack on each side is `N` times the primal's and the two
            # reallocate on exactly the same calls. Checked both ways round.
            for (N, d) in ((1, 2), (2, 2), (2, 10_000))
                g = Float64[]
                foreach(i -> push!(g, i), 1:20)
                blk = getfield(
                    tangent(zero_lifted(Val(N), (g, g.ref.mem)))[1], :partials_block
                )
                pm, bm = g.ref.mem, getfield(blk, :parent).ref.mem
                Base._growend!(g, d)
                Mooncake.Nfwd._resize_block!(Base._growend!, blk, N, d)
                @test (g.ref.mem === pm) == (getfield(blk, :parent).ref.mem === bm)
            end
        end

        @testset "NDualMemoryRef (MemoryRef{T<:IEEEFloat})" begin
            mem = Memory{Float64}(undef, 3) .= [1.0, 2.0, 3.0]
            p = Core.memoryref(mem, 1)
            @test dual_type(Val(2), MemoryRef{Float64}) ===
                Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            @test lifted_type(Val(2), MemoryRef{Float64}) === Lifted{
                MemoryRef{Float64},2,Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            }
            @test dual_type(Val(2), Memory{Float64}) === NDualArray{
                Float64,2,1,Memory{Float64},NDual{Float64,2},Mooncake.NDualBlock{Float64,2}
            }

            # Seed factory: a zero block covering the whole backing memory, slot-local;
            # the referenced element's column is the ref's offset.
            a = zero_dual(Val(2), p)
            @test typeof(a) === Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            # The block is stored as its backing `MemoryRef` (`partials_ref`) + `ncols`;
            # `tangent`/`unpack_ndual` reconstruct the `(N, ncols)` block view on demand.
            @test primal(a) === p
            @test unpack_ndual(a)[1] === a.primal && unpack_ndual(a)[2] == tangent(a)
            @test size(tangent(a)) == (2, 3) && all(iszero, tangent(a))
            @test a.col == Core.memoryrefoffset(p) == 1
            @test primal(zero_lifted(Val(2), p)) === p

            # Empty backing memory: `zero_dual` must not `BoundsError` on the unguarded
            # `Core.memoryref(mem, offset)` (offset==1, out of bounds for len 0).
            empty_p = Float64[].ref
            @test typeof(zero_dual(Val(2), empty_p)) ===
                Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
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

    # complex.jl rules: `lgetfield(::Complex, Val(:re)/:im)` and `_new_(ComplexF64, re, im)` are
    # registered in `hand_written_rule_test_cases(:complex)`, driven through test_rule (both modes,
    # widths 1 and 8, :stability_and_allocs) by the complex group — no bespoke parallel needed.

    # performance_patches.jl rules (sum, sum(abs2,·), LinearAlgebra._kron!) are registered in
    # `hand_written_rule_test_cases(:performance_patches)`; their NDualArray V is per-lane
    # oracle-checkable, so `test_rule` covers value + per-lane partials across widths 1 and 8.

    @testset "MutableDualTangentView (NDual field)" begin
        r = LiftedTest_RefF(3.0)
        slot = zero_lifted(Val(2), r)
        view = tangent(slot, 1)  # per-lane view
        @test view isa MutableDualTangentView
        @test getfield(view, :_parent) === slot.rep
        @test getfield(view, :_primal) === r
        @test getfield(view, :_lane) === 1
        @test view.v === 0.0          # read: lane-1 partial of field `v`
        view.v = 5.0                  # write: routes back to parent.fields via setfield!
        @test view.v === 5.0
        @test slot.rep.fields.v.partials === (5.0, 0.0)
        @test tangent(slot, 2).v === 0.0  # other lane unchanged

        # A user field must resolve to its lane tangent whatever it is called — including the
        # underscored names the view uses for its own fields, which `getproperty` used to
        # short-circuit on, returning the view's parent on a read while a write went to the
        # field.
        pview = tangent(zero_lifted(Val(2), LiftedTest_ParentField(3.0, 4.0, 5.0, 6.0)), 1)
        @testset "$name" for name in (:parent, :_parent, :_primal, :_lane)
            @test getproperty(pview, name) === 0.0
            setproperty!(pview, name, 7.0)
            @test getproperty(pview, name) === 7.0
        end

        # Regression: a mutable struct with an ABSTRACT field type lifts to a `MutableDual`
        # whose backing NamedTuple is abstract (`@NamedTuple{x}`, x::Any). Writing a lane tangent
        # narrows the merged NamedTuple to a concrete element type, which is not `isa` the stored
        # abstract type — a bare `setfield!` throws `TypeError`. The view must `convert` back.
        aview = tangent(zero_lifted(Val(2), LiftedTest_AbstractField(1.0)), 1)
        @test aview.x === 0.0
        aview.x = 4.0
        @test aview.x === 4.0
    end

    @testset "MutableDualTangentView (array, complex and nested fields)" begin
        # An ARRAY field reads as the write-through lane view, so `view.field[i] = x` from a rule
        # body lands in the block. `tangent(::Lifted, lane)` would hand back a dense copy instead,
        # making that assignment a silent no-op.
        av = zero_lifted(Val(2), LiftedTest_Aliased([1.0, 2.0], [3.0, 4.0]))
        aview = tangent(av, 1)
        @test collect(aview.a) == [0.0, 0.0]
        aview.a[2] = 9.0
        @test collect(tangent(av, 1).a) == [0.0, 9.0]   # the write reached the block
        @test collect(tangent(av, 2).a) == [0.0, 0.0]   # and only lane 1
        aview.a = [7.0, 8.0]
        @test collect(aview.a) == [7.0, 8.0]
        @test collect(tangent(av, 2).a) == [0.0, 0.0]

        cv = zero_lifted(Val(2), LiftedTest_ComplexField(ComplexF64(1, 2)))
        cview = tangent(cv, 1)
        @test cview.z === ComplexF64(0, 0)
        cview.z = ComplexF64(4, 5)
        @test cview.z === ComplexF64(4, 5)
        @test tangent(cv, 2).z === ComplexF64(0, 0)

        # A nested mutable struct's lane tangent would have to be another view, which needs a
        # primal the field's V alone does not carry. Name the shape instead of erroring inside
        # `ntuple`/`copyto!`.
        nv = tangent(
            zero_lifted(Val(2), LiftedTest_Cycle(LiftedTest_Cycle(nothing, 2.0), 1.0)), 1
        )
        @test_throws ArgumentError nv.next
        @test_throws ArgumentError (nv.next = 1.0)

        # Same for a TUPLE field: its V is a tuple of `NDual`s, which is not one of the shapes
        # `_lane_tangent` decomposes, so both directions name the shape rather than erroring
        # inside `ntuple`/`copyto!`.
        tv = tangent(zero_lifted(Val(2), LiftedTest_TupleField((1.0, 2.0))), 1)
        @test_throws ArgumentError tv.t
        @test_throws ArgumentError (tv.t = (1.0, 0.0))
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

    # builtins.jl intrinsics (abs/add/copysign/div/mul/neg/sub/fma/muladd/fpext/fptrunc) are
    # registered in `hand_written_rule_test_cases(:builtins)`, which drives them through
    # `test_rule` (both modes, widths 1 and 8, FD) plus the per-lane oracle, so no bespoke
    # one-to-one parallel testset is needed here.

    # low_level_maths.jl scalar primitives are registry-covered under Val{:low_level_maths}
    # (exp/log/sin/.../hypot plus tanpi/pow_fast/clamp/sincos/sincosd/sincospi/modf). test_rule's
    # per-lane oracle already checks per-lane partials for these numeric-dual primitives, so the
    # explicit-seed direct `sin` check added nothing.

    # tasks.jl: `lgetfield`/`getfield` of a `Task` field is registered in
    # `hand_written_rule_test_cases(:tasks)`; `test_frule_interface` asserts the `NoDual` V via
    # `verify_lifted_type` across widths 1 and 8, so no bespoke parallel is needed. `_new_` on immutable
    # and mutable structs is likewise registered (Val{:new}: StructFoo -> ImmutableDual, MutableFoo
    # -> MutableDual, with the V shape checked by verify_lifted_type), so no new.jl parallel either.
    # The iddict.jl parallel below IS kept: the IdDict setindex!->getindex persistence (mutation
    # threaded across two rule calls on the same slot) is not expressible as a registry case.
    # (memory.jl's ctor / memoryrefnew / lmemoryrefget are all registered, so no memory parallel.)

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

        # The rebuilt V must keep the DECLARED backing NamedTuple. Re-deriving each field type from
        # the rebuilt VALUE narrows an `Any`-declared field to the seed's concrete type, so
        # `V !== dual_type(Val(N), P)` and the OpaqueClosure argument typeassert rejects the slot.
        # The value assertions here cannot see it — the partials are correct, only the type is wrong.
        # A `Float64` in the field takes the isbits path, a struct the general one.
        @test Mooncake.verify_lifted_type(bl(LiftedTest_AbstractHeld(1.0), (1,)))
        @test Mooncake.verify_lifted_type(
            bl(LiftedTest_AbstractHeld(LiftedTest_Point(1.0, 2.0)), (1,))
        )

        # width-2: two basis directions in one seed.
        let b = bl([5.0, 6.0, 7.0], (1, 3))
            @test tangent(b, 1) == [1.0, 0.0, 0.0]
            @test tangent(b, 2) == [0.0, 0.0, 1.0]
        end

        # Aliased fields: `dof` dedups the shared array, so both fields share one V.
        shared = [10.0, 20.0]
        let nt = bl(LiftedTest_Aliased(shared, shared), (1,)).rep.fields
            @test nt.a === nt.b
            @test tangent_view(nt.a, 1) == [1.0, 0.0]
        end

        # Aliasing on the `lift` (reverse→forward) path, non-float-element array: the
        # element-wise array lift must register the shared array in the cache so both
        # fields get one V (matching reverse). Float-element fields are safe via
        # `ẋ`-aliasing; this exercises the gap.
        let h = LiftedTest_AliasedNested([[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0]])
            h.b = h.a
            nt = tangent(lift(h, randn_tangent(Xoshiro(1), h))).fields
            @test nt.a === nt.b
        end

        # Self-cyclic mutable struct: terminates; `.next` V is the node's own V.
        c = LiftedTest_Cycle(nothing, 5.0)
        c.next = c
        let b = bl(c, (1,))
            @test b.rep.fields.next === b.rep
            @test b.rep.fields.w.partials[1] == 1.0  # `w` is the only dof
        end

        # Uninit field stays uninit/zero; the defined field gets the basis.
        let nt = bl(LiftedTest_MaybeInit(3.0), (1,)).rep.fields
            @test nt.x.partials[1] == 1.0
        end

        # Complex `MemoryRef` (Julia 1.11+): the complex `NDualMemoryRef` `_basis_seed!!`
        # mirrors the complex `NDualArray` (two dofs per element — real then imag). Regression
        # for the missing complex method, which previously `MethodError`d here.
        @static if VERSION >= v"1.11-"
            let m = Memory{ComplexF64}(undef, 2)
                m .= [1.0 + 0.0im, 2.0 + 0.0im]
                b = bl(Core.memoryref(m), (2,))  # slot 2 = imag part of element 1
                # `tangent(b)` is the `NDualMemoryRef` V; `tangent` of that reconstructs its block.
                @test tangent(tangent(b))[1, :] == [0.0 + 1.0im, 0.0 + 0.0im]
            end
        end

        # A differentiable-eltype `Ptr` field's V is `NTuple{N,Ptr}`, which dispatches through
        # the `::Tuple` basis-seed methods to a bare `Ptr`. That lane has no addressable tangent
        # (0 dof, like `NoDual`); previously it MethodError'd for lack of a terminal `Ptr` method.
        let b = bl((Ptr{Float64}(0), 2.0), (1,))  # only the Float64 field carries a dof
            @test tangent(b)[1] === (Ptr{Float64}(0),)  # Ptr lane left unchanged
            @test tangent(b)[2].partials[1] == 1.0      # Float64 lane seeded hot
        end
    end

    @testset "lift preserves cross-field array aliasing (D3)" begin
        # A top-level 2-arg `lift` of an aggregate (Tuple / NamedTuple / immutable struct)
        # must thread one shared cache so two fields aliasing the same mutable array get one
        # shared V, matching reverse `zero_tangent_internal`. Previously each field upgraded
        # `nothing` to its own IdDict, producing independent Vs (a silently-wrong JVP).
        # Uses a non-float element type (`Vector{Vector{Float64}}`): float arrays alias via
        # the shared reverse tangent regardless, so they can't exhibit the bug.
        a = [[1.0], [2.0]]
        let vt = tangent(lift((a, a), zero_tangent((a, a))))
            @test vt[1] === vt[2]
        end
        let vn = tangent(lift((p=a, q=a), zero_tangent((p=a, q=a))))
            @test vn.p === vn.q
        end
        let x = LiftedTest_TwoArrays(a, a)
            v = tangent(lift(x, zero_tangent(x))).fields
            @test v.p === v.q
        end
    end

    @testset "_add_to_primal with non-always-init struct fields (D4/D9)" begin
        # Forward `_add_to_primal` must handle a `PossiblyUninitTangent`-wrapped field the way
        # reverse mode does: unwrap an initialised PUT via `is_init`/`val`, map an undefined
        # field to `FieldUndefined()`, and reconstruct through `__construct_type` (honouring
        # `unsafe`). Previously it hand-rolled the field loop with `_new_` and MethodError'd on
        # the PUT (bitstype field) or hit `UndefRefError` (genuinely-undefined heap field).
        @testset "bitstype uninit field (PUT initialised)" begin
            x = LiftedTest_MaybeInit(3.0)  # `y` is bitstype ⇒ isdefined, PUT carries a value
            for N in (1, 2, 3)
                r = randn_lifted(Val(N), Xoshiro(1), x)
                xp = Mooncake._add_to_primal(primal(r), tangent(r), true)
                @test xp isa LiftedTest_MaybeInit
                @test xp.x != 3.0
            end
            z = zero_lifted(Val(1), x)
            @test Mooncake._add_to_primal(primal(z), tangent(z), true).x === 3.0
            # D9: with `unsafe=false`, reconstruction goes through the public constructor. Both
            # fields are present here, but `LiftedTest_MaybeInit` has only a 1-arg constructor, so
            # `P(x, y)` fails and a diagnostic `AddToPrimalException` is thrown — the reverse-oracle
            # contract (cf. the reverse test in tangents.jl). The pre-D9 `_new_` path bypassed the
            # constructor and would silently succeed, so this pins the `unsafe` half of the fix.
            @test_throws Mooncake.AddToPrimalException Mooncake._add_to_primal(
                primal(z), tangent(z), false
            )
        end
        @testset "heap uninit field (FieldUndefined)" begin
            x = LiftedTest_MaybeInitHeap(3.0)  # `y::Vector` genuinely undefined
            for N in (1, 2, 3)
                r = randn_lifted(Val(N), Xoshiro(2), x)
                xp = Mooncake._add_to_primal(primal(r), tangent(r), true)
                @test xp isa LiftedTest_MaybeInitHeap
                @test xp.x != 3.0
                @test !isdefined(xp, :y)  # undefined field stays undefined, matching reverse
            end
            # D9: here `y` maps to `FieldUndefined`, so `__construct_type` calls the 1-arg
            # `P(x)` — which exists — and `unsafe=false` succeeds, leaving `y` undefined.
            let z = randn_lifted(Val(1), Xoshiro(3), x)
                r = Mooncake._add_to_primal(primal(z), tangent(z), false)
                @test r isa LiftedTest_MaybeInitHeap
                @test !isdefined(r, :y)
                @test r.x != 3.0
            end
        end
    end

    @testset "test_lifted (representation interface)" begin
        # Drive the forward representation contract over the SAME canonical primal table
        # that `test_tangent` / `test_data` use for reverse mode, so the two stay in sync.
        @testset "$(typeof(p))" for (interface_only, p, t...) in
                                    Mooncake.tangent_test_cases()
            # The cache-free seed factories recurse with no visited set, so a self-referential
            # primal overflows the stack there — only the cache-threading path can seed one.
            # Detect the cycle rather than listing the cases, so a new cyclic table entry needs
            # no bookkeeping here.
            test_lifted(Xoshiro(123456), p; cache_free=(!_self_referential(p)))
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

        # Pointer primals cannot go in `tangent_test_cases()`: that table also drives reverse
        # `test_tangent`, which has no `test_tangent_type` method for a `Ptr`. Drive them here
        # so the forward contract still covers them. Both the seed factories and the
        # `lift`/`unlift` bridge translate between a lane pointer and a reverse placeholder
        # tangent, and those two coincide only for `Ptr{Float64}` — the other eltypes are
        # exactly where a translation gets skipped. `Ptr{Int}` covers the fourth shape, a
        # non-differentiable pointee, whose forward V is `NoDual` while reverse still keeps a
        # typed `Ptr{NoTangent}` placeholder.
        ptr_backing = [1.0, 2.0]
        @testset "test_lifted $(typeof(p))" for p in (
            Ptr{Nothing}(pointer(ptr_backing)),
            pointer(ptr_backing),
            Ptr{Mooncake.NoTangent}(pointer(ptr_backing)),
            Ptr{Int}(pointer(ptr_backing)),
        )
            test_lifted(Xoshiro(123456), p)
        end
    end
end
