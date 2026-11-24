struct LiftedTest_Point
    x::Float64
    y::Float64
end
mutable struct LiftedTest_RefF
    v::Float64
end

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
    extract

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

    @testset "Lifted slot basics" begin
        inner = nd(3.0, 1.0, -1.0)
        slot = sl(2, 3.0, inner)
        @test typeof(slot) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(slot) === 3.0
        @test tangent(slot) === inner
        @test extract(slot) === (3.0, inner)
        @test slot == sl(2, 3.0, inner)
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
        # The refusal is deferred to the call, so each query names the missing method.
        @test_throws "no `dual_type` method" dual_type(Val(2), LiftedTest_CustomTangent)
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
end
