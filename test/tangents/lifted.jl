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
# An `Any` field holding an aggregate exercises recursive lane materialisation.
struct LiftedTest_AbstractHeld
    x::Any
end
# An immutable struct holding a `Ref`, whose reverse tangent field is a `MutableTangent`.
struct LiftedTest_RefContainer
    r::Base.RefValue{Float64}
end
mutable struct LiftedTest_MaybeInit
    x::Float64
    y::Float64
    LiftedTest_MaybeInit(x::Float64) = new(x)
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

# An opaque handle, the shape CUDA's `DataRef` has: no forward derivative (`dual_type` is
# `NoDual`); its reverse tangent is the handle itself, reused as shared cotangent storage.
struct LiftedTest_Handle
    id::Int
end
Mooncake.tangent_type(::Type{LiftedTest_Handle}) = LiftedTest_Handle
Mooncake.dual_type(::Val{N}, ::Type{LiftedTest_Handle}) where {N} = Mooncake.NoDual
Mooncake.zero_tangent_internal(x::LiftedTest_Handle, ::Mooncake.MaybeCache) = x
# A custom reverse tangent with no `dual_type`: the structural lift must refuse it.
struct LiftedTest_CustomTangent
    id::Int
end
Mooncake.tangent_type(::Type{LiftedTest_CustomTangent}) = LiftedTest_CustomTangent
struct LiftedTest_FloatInt <: Integer
    x::Float64
end
mutable struct LiftedTest_Partial
    a::Float64
    b::Vector{Float64}
    LiftedTest_Partial(a) = new(a)
end

using Mooncake:
    Lifted,
    tangent_view,
    NDual,
    NDualArray,
    NoDual,
    ImmutableDual,
    MutableDual,
    lifted_type,
    zero_lifted,
    uninit_lifted,
    randn_lifted,
    uninit_dual,
    randn_dual,
    basis_lifted!!,
    lift,
    unlift,
    extract
using Mooncake.TestUtils: test_lifted, test_lifted_type

# Shorthands for the table-driven type tests below (top level: parametric aliases are
# not allowed in local scope).
const ND = NDual
# The array V's 6th parameter `B` (= `_block_type(A)`) can't be derived inside a type alias (no
# function calls on a free typevar), so spell the array V shapes the table below needs as named
# consts.
const NDA_VecF64 = NDualArray{
    Float64,2,1,Vector{Float64},NDual{Float64,2},Mooncake.PartialsBlock{Float64,2}
}
const NDA_MatF32 = NDualArray{
    Float32,1,2,Matrix{Float32},NDual{Float32,1},Mooncake.PartialsBlock{Float32,3}
}
const NDAC_VecC64 = NDualArray{
    ComplexF64,
    2,
    1,
    Vector{ComplexF64},
    Complex{NDual{Float64,2}},
    Mooncake.PartialsBlock{ComplexF64,2},
}

@testset "lifted" begin
    # Slot/inner-dual shorthands. `sl` wraps once at the top level with the sharp
    # `Base._stable_typeof` P (`Type{T}` for type-valued primals, matching rule dispatch).
    sl(N, p, v=NoDual()) = Lifted{Base._stable_typeof(p),N}(p, v)
    nd(v::T, parts::Vararg{T,N}) where {T,N} = NDual{T,N}(v, parts)

    @testset "one `Ref` reached twice lifts to one V" begin
        # Repeated Refs must share V storage on both the seed and lift paths. Bespoke:
        # test_lifted counts storages but never writes through one position and reads the
        # other.
        mk() = (r=Ref(1.0); (r, r))
        p = mk()
        v = Mooncake.tangent(Mooncake.lift(p, Mooncake.zero_tangent(p)))
        @test v[1] === v[2]
        @test getfield(v[1], :partials) === getfield(v[2], :partials)
        z = Mooncake.tangent(Mooncake.zero_lifted(Val(1), mk()))
        @test z[1] === z[2]
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "a `Memory` and a ref into it share one partials store" begin
            # Memory and MemoryRef must share partials, on both seed and lift paths.
            mk() = (m=Memory{Float64}(undef, 1); m[1]=1.0; (m, Core.memoryref(m, 1)))
            # Storage identity, which no value check can see on its own: two buffers holding equal
            # numbers give the right answer here and the wrong one after the next write.
            v = Mooncake.tangent(Mooncake.zero_lifted(Val(1), mk()))
            @test getfield(getfield(getfield(v[1], :partials_block), :parent), :ref).mem ===
                getfield(v[2], :partials_ref).mem
            # A top-level `lift` of a `Memory` holding one array twice must thread one cache
            # through its elements, so they share too.
            a = [1.0, 2.0]
            mem = Memory{Vector{Float64}}(undef, 2)
            mem[1] = a
            mem[2] = a
            lv = Mooncake.tangent(Mooncake.lift(mem, Mooncake.zero_tangent(mem)))
            @test getfield(getfield(lv[1], :partials_block), :parent) ===
                getfield(getfield(lv[2], :partials_block), :parent)
        end
    end

    @testset "Lifted slot basics" begin
        inner = nd(3.0, 1.0, -1.0)
        slot = sl(2, 3.0, inner)
        @test typeof(slot) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(slot) === 3.0
        @test tangent(slot) === inner
        @test extract(slot) === (3.0, inner)
        @test slot == sl(2, 3.0, inner)

        # An `NDualArray` names the primal it was built over, so the slot takes its primal from
        # there: pairing array `a` with a representation over `b` cannot name `a`.
        a, b = [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]
        @test primal(Lifted{Vector{Float64},1}(a, tangent(zero_lifted(Val(1), b)))) === b
        # A short seed would be read past its end, a long one silently truncated.
        @test_throws DimensionMismatch lift(ones(11), ones(10))
        @test_throws DimensionMismatch lift(ones(10), ones(11))
        @test_throws DimensionMismatch lift((ones(3),), ([1.0, 1.0],))
        @test_throws DimensionMismatch lift((ones(2),), ([1.0, 1.0, 1.0],))
        seeds = Any[
            ([ones(2), ones(2)], [ones(2)], [ones(2), ones(2), ones(2)]),
            ([:a, :b], [NoTangent()], fill(NoTangent(), 3)),
        ]
        @static if VERSION >= v"1.11-rc4"
            M(v) = copyto!(Memory{eltype(v)}(undef, length(v)), v)
            seeds = [
                seeds;
                map(c -> map(M, c), seeds);
                [(M(ones(2)), M(ones(1)), M(ones(3)))]
            ]
            push!(seeds, map(memoryref, seeds[end]), map(memoryref, seeds[3]))
            # A seed ref at another offset names another element as its `[]`.
            for m in (M(ones(3)), M([ones(1), ones(1), ones(1)]))
                @test_throws DimensionMismatch lift((memoryref(m, 2),), (memoryref(m, 1),))
            end
        end
        @testset for (x, short, long) in seeds, s in (short, long)
            @test_throws DimensionMismatch lift((x,), (s,))
        end

        # Pointer metadata must match the slot's width and share one buffer.
        p, q = Ptr{Float64}(16), Ptr{Cvoid}(16)
        @test_throws "width 2" Lifted{Ptr{Float64},2}(p, Mooncake.PointerDual((p,)))
        @test_throws "matching width" Mooncake.PointerDual{2}(
            (q, q), Mooncake.PointerDual{1}
        )
        @test_throws "one shared buffer" Mooncake.PointerDual{2}(
            (q, q + 8), Mooncake.PointerDual{2}
        )
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
        # Structural arrays: element-wise.
        (2, Vector{Int}, Vector{Mooncake.NoDual}),
        (2, Vector{Vector{Float64}}, Vector{NDA_VecF64}),
        # Custom canonical Vs.
        (2, Base.RefValue{Float64}, Mooncake.NDualRef{Float64,2}),
        (2, Ptr{Float64}, Mooncake.PointerDual{2}),
    ]
        @test dual_type(Val(N), P) === V
        @test lifted_type(Val(N), P) === Lifted{P,N,V}
    end

    @testset "abstract field holding a structured value" begin
        # Child slots take the held value's type: a declared `Any` reads tuple elements as
        # lanes. The `P<:Tuple` container branch is reached only through an abstract slot
        # the IR annotates, which no seed factory builds, so annotate one by hand.
        tp = (LiftedTest_Point(1.0, 2.0),)
        @test tangent(Lifted{Tuple{Any},1}(tp, tangent(zero_lifted(Val(1), tp))), 1)[1] isa
            Mooncake.Tangent
        # Each lane of a held tuple is the whole tuple's lane, not one element's V.
        tt = ((1.0, 2.0),)
        @test tangent(Lifted{Tuple{Any},3}(tt, tangent(zero_lifted(Val(3), tt))), 2) ==
            ((0.0, 0.0),)
    end

    @testset "the unlift terminal refuses a non-leaf" begin
        # A malformed NoDual over a differentiable aggregate must fail at the boundary.
        s = TestResources.StructFoo(6.0, [1.0, 2.0])
        @test_throws ArgumentError unlift(
            Lifted{Tuple{typeof(s)},1,Tuple{NoDual}}((s,), (NoDual(),))
        )
    end

    @testset "a canonical NoDual over a non-`NoTangent` primal is not refused" begin
        # dual_type licenses NoDual despite a real reverse placeholder, as for CUDA DataRef.
        h = LiftedTest_Handle(7)
        @test unlift(Lifted{LiftedTest_Handle,1,NoDual}(h, NoDual())) === (h, h)
    end

    @testset "metatype kinds get an unbounded slot" begin
        # All four concrete metatype kinds need unbounded slots for runtime Type{X} primals.
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
    end

    @testset "non-concrete, fieldless and non-differentiable primals" begin
        # Tuple unions and `Vararg` tuples have no single element list to recurse over.
        @test dual_type(Val(2), Union{Tuple{Float64},Tuple{Float64,Float64}}) === Any
        @test dual_type(Val(2), Union{Tuple{},Tuple{Float64}}) === Any
        @test dual_type(Val(2), Union{Tuple{T},Tuple{T,T}} where {T<:AbstractFloat}) === Any
        @test dual_type(Val(2), Tuple{Vararg{Float64}}) === Any
        # A non-concrete tuple admits its wholly non-differentiable concretisations.
        @test dual_type(Val(2), Tuple{Real}) === Union{NoDual,Tuple{Any}}
        @test dual_type(Val(2), Integer) === Any
        @test dual_type(Val(2), Union{Float64,Nothing}) === Any
        nt = @NamedTuple{a::Float64, b::Any}
        @test dual_type(Val(2), nt) === Any
        @test lifted_type(Val(2), nt) == (Lifted{nt,2,W} where {W})
        # `NoDual` exactly where reverse mode has `NoTangent`.
        for P in (
            Bool,
            Int8,
            UInt128,
            BigInt,
            Char,
            Symbol,
            Nothing,
            DataType,
            TypeVar,
            Module,
            Expr,
            Cstring,
            Cwstring,
        )
            @test tangent_type(P) === NoTangent
            @test dual_type(Val(2), P) === NoDual
        end
        @test dual_type(Val(2), LiftedTest_FloatInt) ===
            ImmutableDual{@NamedTuple{x::ND{Float64,2}}}
        @test dual_type(Val(2), LiftedTest_Partial) === MutableDual{
            @NamedTuple{a::ND{Float64,2},b::Mooncake.PossiblyUninitTangent{NDA_VecF64}}
        }
        # A fieldless differentiable type carries one tangent per lane.
        @test dual_type(Val(2), Core.SimpleVector) === NTuple{2,Vector{Any}}
        if isdefined(Core, :BFloat16)
            @test dual_type(Val(2), Core.BFloat16) === NTuple{2,Core.BFloat16}
        end
        # Only a union of non-differentiable members splits into its members' slots.
        @test lifted_type(Val(2), Union{Int,Nothing}) ===
            Union{Lifted{Int,2,NoDual},Lifted{Nothing,2,NoDual}}
        @test lifted_type(Val(2), Union{Float64,Nothing}) ==
            (Lifted{T,2,V} where {T<:Union{Float64,Nothing},V})
    end

    @testset "a Ptr slot holds PointerDual lanes of its width" begin
        p = Ptr{Float64}(16)
        @test Lifted{Ptr{Float64},2}(p, (p, p)) isa lifted_type(Val(2), Ptr{Float64})
        @test_throws ArgumentError Lifted{Ptr{Float64},2}(p, (p,))
        @test_throws ArgumentError Lifted{Ptr{Float64},2,NoDual}(p, NoDual())
    end

    @testset "a custom reverse tangent without a `dual_type` is refused" begin
        # Bespoke: `test_lifted` asserts a coherent V, and this is a refusal, deferred to
        # the call so that the type query and a rule's seed both name the missing method.
        @test_throws "no `dual_type` method" dual_type(Val(2), LiftedTest_CustomTangent)
        @test_throws "no `dual_type` method" zero_lifted(
            Val(2), LiftedTest_CustomTangent(1)
        )
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

        @static if VERSION >= v"1.11-rc4"
            for N in (1, 8), removed in 0:2
                x = sizehint!([2.0, 3.0], 20)
                foreach(_ -> popfirst!(x), 1:removed)
                for v in (
                    zero_dual(Val(N), x),
                    uninit_dual(Val(N), x),
                    randn_dual(Val(N), Xoshiro(123), x),
                    tangent(lift(x, ones(length(x)))),
                )
                    width = size(v.partials_block, 1)
                    ref = v.partials_block.parent.ref
                    @test length(ref.mem) == width * length(x.ref.mem)
                    @test Core.memoryrefoffset(ref) ==
                        width * (Core.memoryrefoffset(x.ref) - 1) + 1
                end
            end
        end

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

        # Complex MemoryRef factories must agree with the canonical NDualMemoryRef V.
        @static if VERSION >= v"1.11-rc4"
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

        # An `Integer` with a float field seeds structurally, as its `dual_type` says.
        @test typeof(zero_dual(Val(2), LiftedTest_FloatInt(1.5))) ===
            dual_type(Val(2), LiftedTest_FloatInt)

        # A world counter has no V but a `Vector{Any}` of methods has one, which zero
        # derivatives alone cannot tell from `NoDual`.
        w = Base.get_world_counter()
        ym = Base._methods_by_ftype(Tuple{typeof(sin),Float64}, -1, w)
        for N in (1, 2)
            @test tangent(zero_lifted(Val(N), w)) === NoDual()
            @test tangent(zero_lifted(Val(N), ym)) isa Vector{Any}
        end
    end

    @testset "`uninit_dual` on a float array allocates the block only" begin
        # Bespoke: `test_lifted` has no performance half. Per-lane `similar` copies packed
        # into the block would cost `N` extra allocations on every `getfield` off a
        # non-differentiable parent, for content that is garbage by definition.
        x = randn(StableRNG(1), 100)
        uninit_dual(Val(8), x)
        zero_dual(Val(8), x)
        @test (@allocated uninit_dual(Val(8), x)) <= (@allocated zero_dual(Val(8), x))
    end

    @testset "NDualArray accessors + AbstractArray interface" begin
        x = [1.0, 2.0, 3.0]
        a = NDualArray{Float64,2,1,Vector{Float64}}(
            x, (similar(x).=[0.5, -0.5, 1.5], similar(x).=[0.0, 1.0, -1.0])
        )
        @test a isa AbstractArray{NDual{Float64,2},1}
        @test size(a) == (3,) && length(a) == 3
        @test primal(a) === x && tangent(a) == ([0.5, -0.5, 1.5], [0.0, 1.0, -1.0])
        @test a[2] === nd(2.0, -0.5, 1.0)  # lazy getindex
        a[1] = nd(9.0, 7.0, -7.0)          # setindex! writes both channels
        @test x[1] === 9.0 &&
            tangent_view(a, 1)[1] === 7.0 &&
            tangent_view(a, 2)[1] === -7.0
    end

    @static if VERSION >= v"1.11-rc4"
        @testset "aliased Memory shares one V, on both the seed and lift paths" begin
            # Repeated Memory/MemoryRef primals must reuse partial storage across calls.
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
            # Partial storage must alias with the primal. test_rule's finite-difference oracle
            # uses the same seeding machinery, so an independent mutation check is needed.
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
                Base._growend!(getfield(blk, :parent), N * d)
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
                Float64,
                2,
                1,
                Memory{Float64},
                NDual{Float64,2},
                Mooncake.PartialsBlock{Float64,2},
            }

            # Seed factory: a zero block covering the whole backing memory, slot-local;
            # the referenced element's column is the ref's offset.
            a = zero_dual(Val(2), p)
            @test typeof(a) === Mooncake.NDualMemoryRef{Float64,2,Memory{Float64}}
            # The block is stored as its backing `MemoryRef` (`partials_ref`) + `ncols`;
            # `tangent` reconstructs the `(N, ncols)` block view on demand.
            @test primal(a) === p
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

    @testset "a non-differentiable element array's lane matches reverse" begin
        # A pointer has no forward partial, but its reverse tangent carries the address:
        # `Vector{Ptr{Int}}` has reverse tangent `Vector{Ptr{NoTangent}}`, not `NoTangent`s.
        # The element type must come from the primal, not the V. Bespoke: no registered
        # `test_lifted` case holds a pointer-element array, and this also pins the values.
        pv = Ptr{Int}[Ptr{Int}(0), Ptr{Int}(8)]
        @test tangent(zero_lifted(Val(2), pv), 1) == zero_tangent(pv)
        @test typeof(tangent(zero_lifted(Val(2), pv), 1)) === tangent_type(Vector{Ptr{Int}})
        # The ordinary non-differentiable elements are unchanged.
        for v in ([1, 2], [:a, :b])
            @test tangent(zero_lifted(Val(2), v), 1) == zero_tangent(v)
        end
    end

    @testset "element-wise Vector with abstract eltype (concrete struct elements)" begin
        # Each element's lane must be read through its concrete `typeof(pe)`: the abstract
        # `eltype(P)` has no fields to lift.
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

    @testset "type-stability" begin
        # The canonical width-N path is type-stable for IEEEFloat primals.
        @test @inferred(zero_dual(Val(2), 1.0)) isa NDual{Float64,2}
        @test @inferred(zero_lifted(Val(2), 1.0)) isa Lifted{Float64,2,NDual{Float64,2}}
        @test @inferred(dual_type(Val(2), Float64)) === NDual{Float64,2}
        @test @inferred(lifted_type(Val(2), Float64)) === Lifted{Float64,2,NDual{Float64,2}}
    end

    @testset "basis_lifted!!" begin
        # `basis_lifted!!(zero_lifted(...), slots)` sets lane k hot at the slots[k]-th
        # scalar dimension, in `tangent_dim`/`zero_tangent` order, mutating mutable V in
        # place and rebuilding immutable V.
        bl(x, slots) = basis_lifted!!(zero_lifted(Val(length(slots)), x), slots)

        @test tangent(bl(3.0, (1,)), 1) == 1.0
        @test tangent(bl([5.0, 6.0, 7.0], (2,)), 1) == [0.0, 1.0, 0.0]
        @test tangent(bl(1.0 + 2.0im, (2,)), 1) == 0.0 + 1.0im  # imag tangent_dim
        let t = tangent(bl(([1.0, 2.0], 9.0), (3,)), 1)
            @test t[1] == [0.0, 0.0] && t[2] == 1.0  # the scalar is tangent_dim 3
        end

        # Aliased fields: `tangent_dim` dedups the shared array, so both fields share one V.
        shared = [10.0, 20.0]
        let nt = bl(LiftedTest_Aliased(shared, shared), (1,)).rep.fields
            @test nt.a === nt.b
            @test tangent_view(nt.a, 1) == [1.0, 0.0]
        end

        # Headers over one storage share one seed, so they have no basis positions of their
        # own; one object reached twice is one header.
        let a = [1.0, 2.0], b = Any[1.0, 2.0]
            shared = Any[(a, reshape(a, 1, 2)), (b, vec(reshape(b, 1, 2)))]
            @static if VERSION >= v"1.11-rc4"
                h = sizehint!([1.0, 2.0], 8)
                # The empty `y` reaches the `Memory` first; both later headers carry.
                y = empty!([1.0, 2.0])
                ym = y.ref.mem
                push!(shared, (h, h.ref.mem), (y, ym, Base.wrap(Array, ym, 1)))
            end
            @testset "refuse $(typeof(x))" for x in shared
                TestUtils._test_throws(ArgumentError, "different headers") do
                    Mooncake.tangent_dim(x)
                end
                TestUtils._test_throws(ArgumentError, "different headers") do
                    bl(x, (1,))
                end
            end
            @test Mooncake.tangent_dim((a, a, b)) == 4
            @test tangent(bl((a, a), (2,)), 1) == ([0.0, 1.0], [0.0, 1.0])
            # Shared storage without degrees of freedom holds no basis position.
            ai, e = Any[1, 2], Float64[]
            for x in ((ai, reshape(ai, 1, 2), 3.0), (e, reshape(e, 0, 1), 3.0))
                @test Mooncake.tangent_dim(x) == 1
                @test tangent(bl(x, (1,)), 1)[3] == 1.0
            end
            @static if VERSION >= v"1.11-rc4"
                # Only the `Memory` carries a derivative, in either order or reached twice.
                v = sizehint!(Any[2, 3], 4)
                m = v.ref.mem
                m[3] = 1.0
                for (x, d) in
                    (((ym, y), 2), ((y, ym), 2), ((y, ym, ym), 2), ((m, v), 1), ((v, m), 1))
                    @test Mooncake.tangent_dim(x) == d
                end
                @test tangent(bl((y, ym), (2,)), 1)[2] == [0.0, 1.0]
                @test tangent(bl((m, v), (1,)), 1)[1][3] == 1.0
            end
        end

        # The control: the storage key must not merge unrelated stores, and a REUSED seed
        # must be cleared on every reseed. A false merge would skip the clear and leave the
        # previous chunk's hot lane behind.
        let s = zero_lifted(Val(1), ([1.0, 2.0], [3.0, 4.0]))
            @test tangent(basis_lifted!!(s, (1,)), 1) == ([1.0, 0.0], [0.0, 0.0])
            @test tangent(basis_lifted!!(s, (4,)), 1) == ([0.0, 0.0], [0.0, 1.0])
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
            @test b.rep.fields.w.partials[1] == 1.0  # `w` is the only tangent_dim
        end

        # Uninit field stays uninit/zero; the defined field gets the basis.
        let nt = bl(LiftedTest_MaybeInit(3.0), (1,)).rep.fields
            @test nt.x.partials[1] == 1.0
        end

        @static if VERSION >= v"1.11-rc4"
            # Structural MemoryRefs leave the following scalar at the first basis position.
            for T in (Int, Union{})
                p = (memoryref(Memory{T}(undef, 0)), 1.0)
                @test tangent(bl(p, (1,)), 1)[2] == 1.0
            end

            # A complex `MemoryRef` seeds two dimensions per element, real then imaginary.
            let m = Memory{ComplexF64}(undef, 2)
                m .= [1.0 + 0.0im, 2.0 + 0.0im]
                b = bl(Core.memoryref(m), (2,))  # slot 2 = imag part of element 1
                # `tangent(b)` is the `NDualMemoryRef` V; its `tangent` rebuilds the block.
                @test tangent(tangent(b))[1, :] == [0.0 + 1.0im, 0.0 + 0.0im]
            end
        end

        # Memory and MemoryRef contribute one set of dimensions. Check a later array's basis
        # positions too, so a duplicate cursor advance cannot silently drop its gradient.
        # test_lifted compares basis seeds with each other, never with known positions.
        @static if VERSION >= v"1.11-rc4"
            let m = Memory{Float64}(undef, 2), b = [3.0, 4.0]
                m .= [1.0, 2.0]
                x = (m, Core.memoryref(m), b)
                @test tangent_view(tangent(bl(x, (2,)))[1], 1) == [0.0, 1.0]
                @test tangent_view(tangent(bl(x, (3,)))[3], 1) == [1.0, 0.0]
                @test tangent_view(tangent(bl(x, (4,)))[3], 1) == [0.0, 1.0]
            end
        end

        # Pointer fields carry no basis direction.
        let b = bl((Ptr{Float64}(0), 2.0), (1,))
            @test tangent(b)[1] == Mooncake.uninit_dual(Val(1), Ptr{Float64}(0))
            @test tangent(b)[2].partials[1] == 1.0      # Float64 lane seeded hot
        end
    end

    @static if VERSION < v"1.11-rc4"
        @testset "structural headers with unequal extents" begin
            # Raw-pointer aliases need independent tangents; `test_lifted` only bounds
            # storage counts.
            for T in (Tuple{Float64}, Any)
                a = T[(1.0,), (2.0,), (3.0,), (4.0,)]
                GC.@preserve a begin
                    w = unsafe_wrap(Array, pointer(a), 2)
                    # These reshapes only mark the aliases as shared in Julia 1.10.
                    reshape(a, 2, 2)
                    reshape(w, 1, 2)
                    for x in ((w, a), (a, w))
                        for s in (
                            zero_lifted(Val(1), x),
                            uninit_lifted(Val(1), x),
                            randn_lifted(Val(1), Xoshiro(123), x),
                            lift(x, map(b -> fill((1.0,), size(b)), x)),
                        )
                            v = tangent(s)
                            @test pointer(v[1]) != pointer(v[2])
                            lane = tangent(s, 1)
                            @test pointer(lane[1]) != pointer(lane[2])
                        end
                    end
                end
            end
        end
    end

    @testset "lift preserves cross-field array aliasing" begin
        # Aggregate lift must share one cache across repeated mutable fields.
        # Non-float elements expose independent container partials hidden by float tangent aliases.
        a = [[1.0], [2.0]]
        for x in ((a, a), (p=a, q=a), LiftedTest_TwoArrays(a, a))
            v = tangent(lift(x, zero_tangent(x)))
            fields = x isa LiftedTest_TwoArrays ? v.fields : v
            @test fields[1] === fields[2]
        end
    end

    @testset "headers over one buffer share element-wise Vs" begin
        b = [[1.0], [2.0]]
        x = (b, reshape(b, 2, 1))
        for v in (tangent(zero_lifted(Val(1), x)), tangent(lift(x, zero_tangent(x))))
            e = zero_dual(Val(1), [5.0])
            v[1][2] = e
            @test v[2][2] === e
        end
    end

    @testset "lift of `NoTangent` elements follows `dual_type`" begin
        for x in
            (Union{String,Nothing}["a", nothing], Ref(Union{String,Nothing}["a", nothing]))
            @test typeof(tangent(lift(x, zero_tangent(x)))) === dual_type(Val(1), typeof(x))
        end
    end

    @testset "lift refuses seeds that differ over one storage" begin
        a, b, r, s = [1.0, 2.0, 3.0], [(1.0,), (2.0,)], Ref(1.0), LiftedTest_RefF(1.0)
        e = Any[[1.0]]
        cases = Any[
            (a, reshape(a, 3, 1)) => (ones(3), fill(10.0, 3, 1)),
            (a, a) => (ones(3), fill(10.0, 3)),
            (a, a) => (ones(3), NoTangent()),
            (b, b) => ([(1.0,), (2.0,)], [(1.0,), (3.0,)]),
            (r, r) => (zero_tangent(r), randn_tangent(StableRNG(1), r)),
            (s, s) => (zero_tangent(s), randn_tangent(StableRNG(1), s)),
            # Either order: an element seed of another kind or size never agrees.
            (e, e) => (Any[[1.0]], Any[NoTangent()]),
            (e, e) => (Any[NoTangent()], Any[[1.0]]),
            (e, e) => (Any[[1.0]], Any[[1.0, 2.0]]),
        ]
        @static if VERSION >= v"1.11-rc4"
            # On 1.11 a capacity-hinted `reshape` gives a `Memory` sliced from its owner's.
            h = sizehint!([1.0, 2.0], 100)
            m = reshape(h, 2, 1).ref.mem
            push!(cases, (h, m) => (ones(2), fill!(similar(m), 10.0)))
        end
        @testset "$(typeof(x))" for (x, t) in cases
            @test_throws "conflicting tangents" lift(x, t)
        end
        # A NaN matches itself and signed zeros match, so these seeds agree and share.
        x = (a, reshape(a, 3, 1))
        v = tangent(lift(x, ([NaN, 0.0, 1.0], [NaN; -0.0; 1.0;;])))
        tangent_view(v[1], 1)[3] = 7.0
        @test tangent_view(v[2], 1)[3] == 7.0
        v = tangent(lift((a, a), (NoTangent(), NoTangent())))
        @test v[1] === v[2]
    end

    @testset "lift shares partials across unequal backing capacities" begin
        # The registry supplies canonical seeds; this needs caller-owned capacity slack.
        for reserve_primal in (false, true)
            a, da = [3.0, 4.0], [2.0, 1.0]
            sizehint!(reserve_primal ? a : da, 100)
            ps, ts = (a, reshape(a, 2, 1)), (da, reshape(da, 2, 1))
            @static if VERSION >= v"1.11-rc4"
                # A `Memory` seed must match its primal's length, unlike the arrays' capacities.
                m = ps[2].ref.mem
                ps, ts = (ps..., m), (ts..., copyto!(fill!(similar(m), 0), 1, da, 1, 2))
            end
            for width in (1, 8),
                factory in (
                    x -> zero_lifted(Val(width), x),
                    x -> randn_lifted(Val(width), StableRNG(123), x),
                )

                vs = tangent(factory(ps))
                tangent_view(vs[1], width)[1] = 7.0
                @test all(t -> tangent_view(t, width)[1] == 7.0, vs)
            end
            v = tangent(lift(ps, ts))
            tangent_view(v[1], 1)[1] *= 2
            @test all(t -> tangent_view(t, 1)[1] == 4.0, v)
        end
        @static if VERSION >= v"1.11-rc4"
            m = Memory{Float64}([1, 2, 3, 4])
            dm = Memory{Float64}([10, 20, 30, 40])
            a = Base.wrap(Array, memoryref(m), (2,))
            da = Base.wrap(Array, memoryref(dm), (2,))
            for (ps, ts) in (((a, m), (da, dm)), ((m, a), (dm, da)))
                vs = tangent(lift(ps, ts))
                @test map(v -> collect(tangent_view(v, 1)), vs) == map(collect, ts)
            end
        end
    end

    @testset "raw pointer aliases have independent forward storage" begin
        buf = [1.0, 2.0, 3.0]
        GC.@preserve buf begin
            a = unsafe_wrap(Array, pointer(buf), 3)
            b = unsafe_wrap(Array, pointer(buf), 3)
            v = tangent(lift((a, b), (ones(3), fill(10.0, 3))))
            @test tangent_view(v[1], 1) == ones(3)
            @test tangent_view(v[2], 1) == fill(10.0, 3)
            z = tangent(zero_lifted(Val(1), (a, b)))
            tangent_view(z[1], 1)[1] = 5.0
            @test tangent_view(z[2], 1)[1] == 0.0
            z = tangent(zero_lifted(Val(1), (b, reshape(b, 3, 1))))
            tangent_view(z[1], 1)[1] = 5.0
            @test tangent_view(z[2], 1)[1] == 5.0
            b32 = unsafe_wrap(Array, Ptr{Float32}(pointer(buf)), 3)
            v = tangent(lift((a, b32), (ones(3), fill(10.0f0, 3))))
            @test tangent_view(v[2], 1) == fill(10.0f0, 3)
        end
    end

    @testset "test_lifted (representation interface)" begin
        # Drive the forward representation contract over the SAME canonical primal table
        # that `test_tangent` / `test_data` use for reverse mode, so the two stay in sync.
        @testset "$(typeof(p))" for (interface_only, p, t...) in
                                    Mooncake.tangent_test_cases()
            # The cache-free seed factories recurse with no visited set, so a
            # self-referential primal overflows the stack there — only the cache-threading
            # path can seed one. Detect the cycle rather than listing the cases, so a new
            # cyclic table entry needs no bookkeeping here.
            # `SimpleVector`'s V is defined beside its rules.
            p isa Core.SimpleVector && continue
            test_lifted(Xoshiro(123456), p; cache_free=(!_self_referential(p)))
        end

        # Storage kinds: a union or bottom element type is structural, a concrete float or
        # complex one numeric. A cycle through a tuple, an array of mutable structs, and one
        # array in two positions.
        tuple_cycle = Any[]
        push!(tuple_cycle, (tuple_cycle,))
        aliased = randn(5)
        @testset "test_lifted $(typeof(p))" for p in Any[
            Union{}[],
            (Union{}[], 1.0),
            Ref(Union{}[]),
            Ref{Union{}}(),
            Union{Float32,Float64}[2.0f0, 3.0],
            Ref{Union{Float32,Float64}}(2.0),
            Union{Float32,ComplexF64}[2.0f0, 3.0 + 1.0im],
            Ref{Union{Float32,ComplexF64}}(3.0 + 1.0im),
            1.0 + 2.0im,
            ComplexF64[1.0 + 2.0im, -3.0 + 0.5im],
            [TestResources.TypeStableMutableStruct{Float64}(5.0, 3.0)],
            tuple_cycle,
            (aliased, aliased),
        ]
            test_lifted(Xoshiro(123456), p; cache_free=(!_self_referential(p)))
        end

        # Arrays of two lengths, a `MemoryRef` and their backing `Memory`: lane reads tie
        # each numeric `Array` and `MemoryRef` to the `Memory`, as reverse derives them.
        @static if VERSION >= v"1.11-rc4"
            test_lifted(Xoshiro(123456), TestResources.make_array_and_its_buffer())
        end

        # Headers over one storage, not in `tangent_test_cases`: reverse checks there fail
        # (1.10's `deepcopy` splits reshapes; `increment!!` dispatches on `Any` elements).
        @testset "test_lifted shared storage $(eltype(a))" for a in (
            [(3.0,), (4.0,)], Any[(3.0,), (4.0,)]
        )
            p = (a, reshape(a, 1, 2), view(a, 1:1))
            # 1.10's reverse seeds give reshaped headers independent tangents, so the round
            # trip through `lift` cannot hold there.
            @static if VERSION >= v"1.11-rc4"
                test_lifted(Xoshiro(123456), p)
                test_lifted(Xoshiro(123456), (a, vec(reshape(a, 1, 2)), a))
                test_lifted(Xoshiro(123456), (vec(reshape(a, 1, 2)), a))
            end
            for width in (1, 8),
                seed in (
                    zero_lifted(Val(width), p),
                    uninit_lifted(Val(width), p),
                    randn_lifted(Val(width), Xoshiro(123), p),
                )

                @test TestUtils._count_storages(tangent(seed)) == 1
            end
        end

        @testset "test_lifted Ref fields $(typeof(p))" for p in (
            Ref(3.0), LiftedTest_RefContainer(Ref(2.5))
        )
            test_lifted(Xoshiro(123456), p; widths=(1, 2, 8))
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
                Ptr{Tuple{Int32,Int32,Float64}},
                Ptr{Ptr{Float64}},
            ),
            N in (1, 2, 3, 8)

            test_lifted_type(P, Val(N))
        end

        for N in (1, 8)
            @test isbits(tangent(zero_lifted(Val(N), Ptr{Float64}(0))))
        end

        for lanes in ((Ptr{Float64}(0),), (Ptr{Float64}(0), Ptr{Float32}(0)))
            @test_throws ArgumentError Lifted{Ptr{Float64},2}(Ptr{Float64}(0), lanes)
        end

        # Primals whose `dual_type` widens to an upper bound instead of the exact V — one
        # per `lifted_type` method that can widen (the `Tuple` overload and the generic
        # one). The slot must stay inhabited: `Lifted` is invariant in V, so an exact
        # `Lifted{P,N,Any}` annotation admits no runtime value at all.
        @testset "test_lifted widened V $(typeof(p))" for p in (
            convert(@NamedTuple{x::Any}, (x=3.0,)),
            # Same widening with a NON-differentiable value, whose V is a `NoDual`: the
            # per-lane read must decide from the value, not from the widened `Any`.
            convert(@NamedTuple{x::Any}, (x=nothing,)),
            # An AGGREGATE in the widened field, or in a struct's `Any` field: the child
            # slot must take the value's type, not the declared `Any`, or a struct throws
            # from `fieldtype(Any, ·)` and a tuple takes the per-lane accessor, which the
            # round trip catches as another value.
            convert(@NamedTuple{x::Any}, (x=TestResources.StructFoo(6.0, [1.0, 2.0]),)),
            convert(@NamedTuple{x::Any}, (x=(1.0, 2.0),)),
            LiftedTest_AbstractHeld(LiftedTest_Point(1.0, 2.0)),
            LiftedTest_AbstractHeld((1.0, 2.0)),
        )
            test_lifted(Xoshiro(123456), p)
        end

        # Keyed Dict tangents compose with mutable array values.
        @testset "test_lifted Dict" begin
            test_lifted(Xoshiro(123456), Dict(:a => randn(Xoshiro(1), 2)))
        end
    end
end
