# A linked-list node: the recursion is inside a union, which is how it is usually spelled.
mutable struct _RecursiveNode
    v::Float64
    next::Union{Nothing,_RecursiveNode}
end

# Wider than `_reaches_recursive_type`'s node budget, but not recursive: the walk cannot answer.
struct _WideConst{T}
    junk::T
    v::Vector{Float64}
end

@testset "codual" begin
    @test CoDual(5.0, 4.0) isa CoDual{Float64,Float64}
    @test CoDual(Float64, NoTangent()) isa CoDual{Type{Float64},NoTangent}
    @test zero_codual(5.0) == CoDual(5.0, 0.0)

    @testset "$P" for (P, D, F) in Any[
        (Float64, CoDual{Float64,Float64}, CoDual{Float64,NoFData}),
        (Int, CoDual{Int,NoTangent}, CoDual{Int,NoFData}),
        (Real, CoDual, CoDual),
        (Any, CoDual, CoDual),
        (
            Type{UnitRange{Int}},
            CoDual{Type{UnitRange{Int}},NoTangent},
            CoDual{Type{UnitRange{Int}},NoFData},
        ),
        (Type{Tuple{T}} where {T}, CoDual, CoDual),
        (
            Union{Float64,Int},
            Union{CoDual{Float64,Float64},CoDual{Int,NoTangent}},
            Union{CoDual{Float64,NoFData},CoDual{Int,NoFData}},
        ),
        (UnionAll, CoDual, CoDual),
        (DataType, CoDual, CoDual),
        (Union{}, Union{}, Union{}),

        # Tuples:
        # Concrete tuples:
        (
            Tuple{Float64},
            CoDual{Tuple{Float64},Tuple{Float64}},
            CoDual{Tuple{Float64},NoFData},
        ),
        (
            Tuple{Float64,Float32},
            CoDual{Tuple{Float64,Float32},Tuple{Float64,Float32}},
            CoDual{Tuple{Float64,Float32},NoFData},
        ),
        (
            Tuple{Int,Float64,Float32},
            CoDual{Tuple{Int,Float64,Float32},Tuple{NoTangent,Float64,Float32}},
            CoDual{Tuple{Int,Float64,Float32},NoFData},
        ),

        # Small-Union Tuples
        (
            Tuple{Union{Float32,Float64}},
            Union{
                CoDual{Tuple{Float32},Tuple{Float32}},CoDual{Tuple{Float64},Tuple{Float64}}
            },
            Union{CoDual{Tuple{Float32},NoFData},CoDual{Tuple{Float64},NoFData}},
        ),
        (
            Tuple{Nothing,Union{Int,Float64}},
            Union{
                CoDual{Tuple{Nothing,Int},NoTangent},
                CoDual{Tuple{Nothing,Float64},Tuple{NoTangent,Float64}},
            },
            Union{
                CoDual{Tuple{Nothing,Int},NoFData},CoDual{Tuple{Nothing,Float64},NoFData}
            },
        ),

        # General Abstract Tuples
        (Tuple{Any}, CoDual, CoDual),

        # Abstract Vararg / NTuple UnionAll tuples (bounded and unbounded)
        (NTuple{N,Int} where {N}, CoDual, CoDual),
        (Tuple{Vararg{Float64,N}} where {N}, CoDual, CoDual),
        (Tuple{Vararg{Float64}}, CoDual, CoDual),
    ]
        @test TestUtils.check_allocs(codual_type, P) == D
        @test TestUtils.check_allocs(Mooncake.fcodual_type, P) == F
    end

    @testset "(f)codual_type/dual_type with phantom TypeVar (#1191)" begin
        # Both shapes normalise to a DataType whose body still references the free `T`, so
        # the generic `::Type{P}` binders can't bind `P` and throw `UndefVarError`. The
        # Tuple variant blocks a future fix that only special-cases AbstractArray.
        phantom = UnionAll(TypeVar(:A), AbstractArray{TypeVar(:T),TypeVar(:A)})
        phantom_tuple = UnionAll(TypeVar(:A), Tuple{TypeVar(:T),TypeVar(:A)})
        @test codual_type(phantom) === CoDual
        @test Mooncake.fcodual_type(phantom) === CoDual
        # Forward `dual_type` is width-parameterised (`dual_type(::Val{N}, ::Type)`) on this branch;
        # the legacy one-arg `Dual` mapping is gone. A phantom-TypeVar primal widens to `Any`.
        @test dual_type(Val(1), phantom) === Any
        @test codual_type(phantom_tuple) === CoDual
        @test Mooncake.fcodual_type(phantom_tuple) === CoDual
        # A free-TypeVar `Tuple` (e.g. `Tuple{T,A}`) leaves the `P<:Tuple` static parameter
        # unbound; the forward `dual_type`/`lifted_type` bodies guard with `@isdefined(P)` (the
        # idiom the `CoDual` constructor uses) and widen to `Any` rather than referencing the
        # undefined `P` and throwing `UndefVarError`.
        @test dual_type(Val(1), phantom_tuple) === Any
        # `lifted_type` returns a (broad) `Lifted` *slot* type, like the generic `lifted_type`
        # phantom guard — not the inner-V `Any` that `dual_type` returns.
        @test Mooncake.lifted_type(Val(1), phantom_tuple) === Lifted
    end

    @testset "NoPullback" begin
        @test Base.issingletontype(typeof(NoPullback(zero_fcodual(5.0))))
        @test NoPullback(zero_codual(5.0))(4.0) == (0.0,)
    end

    @testset "zero_codual and zero_fcodual for Ptr" begin
        # zero_tangent(::Ptr) throws, so zero_codual/zero_fcodual must not call it.
        # They fall back to uninit_codual/uninit_fcodual (bitcast convention).
        p = Ptr{Float64}()
        @test Mooncake.zero_codual(p) == Mooncake.uninit_codual(p)
        @test Mooncake.zero_fcodual(p) == Mooncake.uninit_fcodual(p)
    end

    @testset "_reaches_recursive_type" begin
        # `record_const_alias!` must not ask `tangent_type` about a type it cannot answer for.
        # `Base.ImmutableDict` holds a `parent` of its own type; `IOContext` holds such a dict and
        # so is equally out of reach, which is why the question is reachability and not
        # self-reference.
        @test Mooncake._reaches_recursive_type(Base.ImmutableDict{Symbol,Any})
        @test Mooncake._reaches_recursive_type(IOContext{IOStream})
        # Constants that do own shareable derivative storage stay guarded.
        @test !Mooncake._reaches_recursive_type(Vector{Float64})
        @test !Mooncake._reaches_recursive_type(Diagonal{Float64,Vector{Float64}})
        @test !Mooncake._reaches_recursive_type(Tuple{Float64,Vector{Float64}})
        @test !Mooncake._reaches_recursive_type(String)
        # A union is how recursion is usually spelled, and `tangent_type` distributes over one,
        # so the walk must: without this the guard reported a linked-list node as answerable and
        # then overflowed asking for its tangent type.
        @test Mooncake._reaches_recursive_type(_RecursiveNode)
    end

    @testset "record_const_alias! records what it cannot ask about" begin
        # The guard exists to refuse, so an unknown resolves towards refusing. A struct wider than
        # the walk's budget is not recursive at all, but the answer is unavailable: recording it
        # yields a loud refusal where skipping yielded a gradient of [1.0, 1.0] against [2.0, 2.0].
        wide = _WideConst(ntuple(i -> Val(i), 601), [1.0, 2.0])
        consts = Any[]
        Mooncake.record_const_alias!(consts, wide)
        @test length(consts) == 1
        # A recursive type is recorded without asking `tangent_type`, which would not terminate.
        consts = Any[]
        Mooncake.record_const_alias!(consts, Base.ImmutableDict{Symbol,Any}(:a, 1))
        @test length(consts) == 1
        # Non-differentiable constants are still skipped, so `===` cannot match one spuriously.
        consts = Any[]
        Mooncake.record_const_alias!(consts, "abc")
        @test isempty(consts)
    end
end
