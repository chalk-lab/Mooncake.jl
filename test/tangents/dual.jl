@testset "Dual" begin
    NDual = Mooncake.NDual
    @test Dual(5.0, 4.0) isa Dual{Float64,Float64}
    @test Dual(Float64, NoTangent()) isa Dual{Type{Float64},NoTangent}
    @test zero_dual(5.0) == Dual(5.0, 0.0)

    # Bare-arg zero_dual must zero NDual partials elementwise; macro-generated
    # frules for `Array{T,D}(undef, n)` rely on this for canonical bare shape.
    let nd = Mooncake.NDual{Float64,2}(1.5, (3.0, 4.0)),
        nd_zero = Mooncake.NDual{Float64,2}(1.5, (0.0, 0.0))

        @test zero_dual(nd) === nd_zero
        @test zero_dual(Complex(nd, nd)) === Complex(nd_zero, nd_zero)
        @test zero_dual([nd, nd]) == [nd_zero, nd_zero]
        @test zero_dual([Complex(nd, nd)]) == [Complex(nd_zero, nd_zero)]
    end
    @static if VERSION >= v"1.11-rc4"
        let nd = Mooncake.NDual{Float64,2}(1.5, (3.0, 4.0)),
            nd_zero = Mooncake.NDual{Float64,2}(1.5, (0.0, 0.0))

            m = fill!(Memory{Mooncake.NDual{Float64,2}}(undef, 2), nd)
            mz = zero_dual(m)
            @test mz isa Memory{Mooncake.NDual{Float64,2}} && all(==(nd_zero), mz)
            mrz = zero_dual(Core.memoryrefnew(m))
            @test mrz isa MemoryRef{Mooncake.NDual{Float64,2}}
            @test Core.memoryrefget(mrz, :not_atomic, false) === nd_zero
        end
    end

    # Phase-3 boundary: `_uninit_dual` at width Val(N) returns a `Lifted` whose
    # inner V is the substituted Dual{Type{Array{NDual{...},D}}, NoTangent}.
    @test Mooncake._uninit_dual(Val(2), Array{Float64,1}) ===
        Mooncake.Lifted{Type{Array{Float64,1}},2}(
        Dual(Array{Mooncake.NDual{Float64,2},1}, NoTangent())
    )
    @test Mooncake._uninit_dual(Val(2), Array{ComplexF64,1}) ===
        Mooncake.Lifted{Type{Array{ComplexF64,1}},2}(
        Dual(Array{Complex{Mooncake.NDual{Float64,2}},1}, NoTangent())
    )

    # Audit step 5: `dual_type(P)` delegates to `dual_type(Val(1), P)`. The
    # IEEEFloat / Complex / Array specialised overloads now apply at width 1,
    # so concrete scalar/array primals return `NDual`-shaped duals; generic
    # immutable concrete `P` stays at the bare `Dual{P, tangent_type(P)}`
    # carve-out (`Int → Dual{Int, NoTangent}`); abstract / UnionAll `P`
    # stays at the bare `Dual` UnionAll.
    @testset "$P" for (P, D) in Any[
        (Float64, NDual{Float64,1}),
        (Int, Dual{Int,NoTangent}),
        (Real, Dual),
        (Any, Dual),
        (Type{UnitRange{Int}}, Dual{Type{UnitRange{Int}},NoTangent}),
        (Type{Tuple{T}} where {T}, Dual),
        (Union{Float64,Int}, Union{NDual{Float64,1},Dual{Int,NoTangent}}),
        (UnionAll, Dual),
        (DataType, Dual),
        (Union{}, Union{}),

        # Tuples lift element-wise:
        (Tuple{Float64}, Tuple{NDual{Float64,1}}),
        (Tuple{Float64,Float32}, Tuple{NDual{Float64,1},NDual{Float32,1}}),
        (
            Tuple{Int,Float64,Float32},
            Tuple{Dual{Int,NoTangent},NDual{Float64,1},NDual{Float32,1}},
        ),

        # Small-Union Tuples (split + element-wise)
        (
            Tuple{Union{Float32,Float64}},
            Union{Tuple{NDual{Float32,1}},Tuple{NDual{Float64,1}}},
        ),
        (
            Tuple{Nothing,Union{Int,Float64}},
            Union{
                Tuple{Dual{Nothing,NoTangent},Dual{Int,NoTangent}},
                Tuple{Dual{Nothing,NoTangent},NDual{Float64,1}},
            },
        ),

        # General Abstract Tuples
        (Tuple{Any}, Dual),

        # Abstract Vararg / NTuple UnionAll tuples (bounded and unbounded)
        (NTuple{N,Int} where {N}, Dual),
        (Tuple{Vararg{Float64,N}} where {N}, Dual),
        (Tuple{Vararg{Float64}}, Dual),
    ]
        # Audit step 5: concrete heterogeneous Tuples take the element-wise
        # lift path. The generator-expression splice into `Tuple{...}` is
        # `@unstable` by design (per-field dispatch), so relax the alloc-
        # check to a value-only check for concrete `Tuple` primals.
        if P isa DataType && P <: Tuple
            @test dual_type(P) == D
        else
            @test TestUtils.check_allocs(dual_type, P) == D
        end
    end

    # Audit test #2: the no-`Val` and `Val(1)` queries agree by construction.
    @testset "audit #2: dual_type(P) === dual_type(Val(1), P)" begin
        for P in (Float64, ComplexF64, Vector{Float64}, Int, Real, Tuple{Float64,Int})
            @test Mooncake.dual_type(P) === Mooncake.dual_type(Val(1), P)
        end
    end
end

@testset "Lifted" begin
    Lifted = Mooncake.Lifted
    NDual = Mooncake.NDual
    NTangent = Mooncake.NTangent
    lifted_type = Mooncake.lifted_type
    tangent_type = Mooncake.tangent_type
    dual_type = Mooncake.dual_type
    _lift = Mooncake._lift
    _unlift = Mooncake._unlift
    extract = Mooncake.extract

    @testset "tangent_type wraps once at width 1" begin
        # Audit test #1: width 1 wraps to a top-level `NTangent{Tuple{T}}` so
        # the width-aware tangent type is parallel with width N>=2 chunked.
        @test tangent_type(Val(1), Float64) === NTangent{Tuple{Float64}}
        @test tangent_type(Val(1), Vector{Float64}) === NTangent{Tuple{Vector{Float64}}}
        @test tangent_type(Val(2), Float64) === NTangent{Tuple{Float64,Float64}}
        # `NoTangent`-leaf primals stay as `NoTangent` at any width.
        @test tangent_type(Val(1), Int) === NoTangent
        # `dual_type(Val(1), P)` for generic concrete `P` deliberately keeps
        # the legacy bare-`T` parallel `Dual{P,T}` form. The OC slot type must
        # match the runtime `Dual{P,T}` produced by `zero_dual` / `Dual(p,t)`
        # call sites; flipping `dual_type` too would force a parallel
        # migration of every generic-P construction site (audit step 5,
        # remaining bulk).
        @test dual_type(Val(1), Int) === Mooncake.Dual{Int,NoTangent}
    end

    @testset "lifted_type" begin
        # Val(0) is primal passthrough.
        @test lifted_type(Val(0), Float64) === Float64
        @test lifted_type(Val(0), Vector{Float64}) === Vector{Float64}
        @test lifted_type(Val(0), Tuple{Float64,Float64}) === Tuple{Float64,Float64}

        # Val(N) wraps once at the top, with V === dual_type(Val(N), P).
        @test lifted_type(Val(2), Float64) === Lifted{Float64,2,NDual{Float64,2}}
        @test lifted_type(Val(2), ComplexF64) ===
            Lifted{ComplexF64,2,Complex{NDual{Float64,2}}}
        @test lifted_type(Val(2), Vector{Float64}) ===
            Lifted{Vector{Float64},2,Vector{NDual{Float64,2}}}

        # Tuple primal: ONE outer Lifted whose V is a bare element-wise Tuple.
        # Lifted never nests inside another Lifted's V.
        @test lifted_type(Val(2), Tuple{Float64,Float64}) ===
            Lifted{Tuple{Float64,Float64},2,Tuple{NDual{Float64,2},NDual{Float64,2}}}

        # Abstract P preserves the width parameter — a width-N abstract slot
        # must NOT accept a width-(N±k) concrete lifted value. The previous
        # behaviour of returning bare `Lifted` allowed that and was unsound.
        T2 = lifted_type(Val(2), Any)
        @test Lifted{Float64,2,NDual{Float64,2}} <: T2
        @test !(Lifted{Float64,1,NDual{Float64,1}} <: T2)

        T1real = lifted_type(Val(1), Real)
        @test Lifted{Float64,1,NDual{Float64,1}} <: T1real
        @test !(Lifted{Float64,2,NDual{Float64,2}} <: T1real)

        # Audit test #4: `AbstractFloat` should accept width-2 concrete
        # lifted values and reject width-1, mirroring the `Real` case.
        T2af = lifted_type(Val(2), AbstractFloat)
        @test Lifted{Float64,2,NDual{Float64,2}} <: T2af
        @test !(Lifted{Float64,1,NDual{Float64,1}} <: T2af)
    end

    @testset "verify_lifted_type" begin
        # Canonical: V === dual_type(Val(N), P).
        ok = Lifted{Float64,2}(3.0, (1.0, 2.0))
        @test Mooncake.verify_lifted_type(ok)

        # Noncanonical: width-1 Dual inside a width-2 slot.
        bad_width = Lifted{Float64,2,Dual{Float64,Float64}}(Dual(1.0, 0.0))
        @test !Mooncake.verify_lifted_type(bad_width)

        # Noncanonical: parallel array tangent inside a width-2 slot.
        bad_arr = Lifted{Vector{Float64},2,Dual{Vector{Float64},Vector{Float64}}}(
            Dual([1.0], [0.0])
        )
        @test !Mooncake.verify_lifted_type(bad_arr)

        # Noncanonical: concrete V inside an abstract-P slot.
        bad_abstract = Lifted{Real,1,NDual{Float64,1}}(NDual{Float64,1}(1.0, (1.0,)))
        @test !Mooncake.verify_lifted_type(bad_abstract)

        # Noncanonical: nested Lifted in V (forbidden by design).
        inner_lifted = Lifted{Float64,1}(3.0, 1.0)
        nested = Lifted{Float64,1,typeof(inner_lifted)}(inner_lifted)
        @test !Mooncake.verify_lifted_type(nested)
    end

    @testset "typeassert keeps runtime Lifted concrete (audit test #14)" begin
        # Per the revised audit (`primal-mode-branch-audit.md` Todo 1):
        # abstract `T` slot annotations should NOT produce exact-abstract
        # runtime wrappers. `typeassert(::Real)` on an `NDual{Float64,1}`-
        # backed `Lifted` returns a concrete `Lifted{Float64, 1, NDual{Float64,1}}`
        # that fits the abstract `lifted_type(Val(N), Real)` slot annotation.
        x = Lifted{Float64,1}(3.0, 1.5)
        type_assert_f = Lifted{typeof(typeassert),1}(typeassert, NoTangent())
        real_type_lifted = Lifted{Type{Real},1}(Real, NoTangent())
        y = Mooncake.frule!!(type_assert_f, x, real_type_lifted)
        # Runtime P is the concrete primal type, not the abstract `Real`.
        @test y isa Lifted{Float64,1}
        # The inner V is the canonical concrete shape, unchanged.
        @test Mooncake._unlift(y) === Mooncake._unlift(x)
        # Compatibility with the abstract slot annotation holds via `isa`.
        @test y isa Mooncake.lifted_type(Val(1), Real)
    end

    @testset "verify_dual_type accepts valid bare inner shapes" begin
        # Bare canonical-V leaves leak through helper-API boundaries; they
        # should validate as legitimate inner duals.
        @test Mooncake.verify_dual_type(NDual{Float64,2}(1.0, (2.0, 3.0)))
        @test Mooncake.verify_dual_type([NDual{Float64,1}(1.0, (2.0,))])
        @test Mooncake.verify_dual_type(
            Complex(NDual{Float64,1}(1.0, (1.0,)), NDual{Float64,1}(2.0, (0.0,)))
        )
        # Canonical Lifted always passes verify_dual_type (the strict check
        # is verify_lifted_type).
        @test Mooncake.verify_dual_type(Lifted{Float64,2}(3.0, (1.0, 2.0)))
    end

    @testset "tangent(x, dir) per-lane access" begin
        # Single-direction extraction for NDual leaves.
        x_ndual = NDual{Float64,3}(10.0, (1.0, 2.0, 3.0))
        @test Mooncake.tangent(x_ndual, 1) === 1.0
        @test Mooncake.tangent(x_ndual, 3) === 3.0
        # Whole-tangent access still returns NTangent.
        @test Mooncake.tangent(x_ndual) === NTangent((1.0, 2.0, 3.0))
    end

    @testset "generic structural dual_type mirrors recursive tangent_type" begin
        # Audit step 4. The recursive struct lift must fire for any user-
        # defined struct whose tangent_type matches the default structural
        # form — no broad `_is_lift_safe_field_type` allowlist on field types.
        # Inner has the canonical structural lift.
        @test Mooncake.dual_type(Val(2), Mooncake.TestResources.StableFoo) ===
            NamedTuple{(:x, :y),Tuple{NDual{Float64,2},Dual{Symbol,NoTangent}}}
    end

    @testset "explicit dual_type overrides win over generic structural lift" begin
        # `dual_type(Val(N), Vector{Float64})` is registered with a canonical-
        # V array shape in `nfwd/NfwdMooncake.jl`; the generic structural
        # struct-lift must not fire on Array types and clobber that.
        @test Mooncake.dual_type(Val(2), Vector{Float64}) === Vector{NDual{Float64,2}}
        # Per-wrapper Diagonal lift in nfwd: must use the wrapper-shaped V.
        @test Mooncake.dual_type(
            Val(2), LinearAlgebra.Diagonal{Float64,Vector{Float64}}
        ) === LinearAlgebra.Diagonal{NDual{Float64,2},Vector{NDual{Float64,2}}}
    end

    @testset "1-arg constructor (V inferred from inner)" begin
        inner = NDual{Float64,2}(1.0, (1.0, 0.0))
        d = Lifted{Float64,2}(inner)
        @test typeof(d) === Lifted{Float64,2,NDual{Float64,2}}
        @test _unlift(d) === inner
    end

    @testset "2-arg constructor: bare NDual (IEEEFloat)" begin
        # Pre-computed lanes: NTuple of partials hits the default NDual constructor.
        d_lanes = Lifted{Float64,2}(3.0, (1.0, 0.0))
        @test typeof(d_lanes) === Lifted{Float64,2,NDual{Float64,2}}
        @test _unlift(d_lanes) === NDual{Float64,2}(3.0, (1.0, 0.0))

        # Scalar tangent broadcasts across N lanes.
        d_scalar = Lifted{Float64,2}(3.0, 0.5)
        @test _unlift(d_scalar) === NDual{Float64,2}(3.0, (0.5, 0.5))

        # NTangent input extracts lanes.
        d_ntan = Lifted{Float64,2}(3.0, NTangent((1.0, 2.0)))
        @test _unlift(d_ntan) === NDual{Float64,2}(3.0, (1.0, 2.0))
    end

    @testset "2-arg constructor: Complex{<:NDual}" begin
        # Element-wise from Complex primal + Complex tangent (scalar broadcast within each NDual).
        d = Lifted{ComplexF64,2}(complex(1.0, 2.0), complex(3.0, 4.0))
        @test typeof(d) === Lifted{ComplexF64,2,Complex{NDual{Float64,2}}}
        re = NDual{Float64,2}(1.0, (3.0, 3.0))
        im = NDual{Float64,2}(2.0, (4.0, 4.0))
        @test _unlift(d) === Complex(re, im)
    end

    @testset "2-arg constructor: Array{<:NDual}" begin
        pri = [1.0, 2.0]
        tan = [10.0, 20.0]
        d = Lifted{Vector{Float64},2}(pri, tan)
        @test typeof(d) === Lifted{Vector{Float64},2,Vector{NDual{Float64,2}}}
        @test _unlift(d) ==
            [NDual{Float64,2}(1.0, (10.0, 10.0)), NDual{Float64,2}(2.0, (20.0, 20.0))]

        # Audit step 5 / test #7: top-level `NTangent{NTuple{N, Array}}` form,
        # matching the canonical width-N return of `tangent(::Array{NDual,D})`.
        d_nt1 = Lifted{Vector{Float64},1}([1.0], NTangent(([2.0],)))
        @test typeof(d_nt1) === Lifted{Vector{Float64},1,Vector{NDual{Float64,1}}}
        @test _unlift(d_nt1) == [NDual{Float64,1}(1.0, (2.0,))]
        # Audit step 5 / test #8: outer tangent of width-1 array slot is the
        # top-level `NTangent{Tuple{Vector{Float64}}}`, not bare Array.
        @test tangent(d_nt1) isa NTangent{Tuple{Vector{Float64}}}
        @test tangent(d_nt1).lanes[1] == [2.0]
        # Width-N NTangent form round-trips: per-lane arrays zip into NDual elements.
        d_nt2 = Lifted{Vector{Float64},2}([1.0], NTangent(([2.0], [3.0])))
        @test _unlift(d_nt2) == [NDual{Float64,2}(1.0, (2.0, 3.0))]
    end

    @testset "2-arg constructor: Tuple primal (single outer Lifted)" begin
        # Per-element scalar tangents: each broadcasts across N lanes within its own NDual.
        d_scalar = Lifted{Tuple{Float64,Float64},2}((1.0, 2.0), (10.0, 20.0))
        @test typeof(d_scalar) ===
            Lifted{Tuple{Float64,Float64},2,Tuple{NDual{Float64,2},NDual{Float64,2}}}
        @test _unlift(d_scalar) ===
            (NDual{Float64,2}(1.0, (10.0, 10.0)), NDual{Float64,2}(2.0, (20.0, 20.0)))

        # Per-element per-lane tangents (NTuple per element).
        d_lanes = Lifted{Tuple{Float64,Float64},2}((1.0, 2.0), ((10.0, 11.0), (20.0, 21.0)))
        @test _unlift(d_lanes) ===
            (NDual{Float64,2}(1.0, (10.0, 11.0)), NDual{Float64,2}(2.0, (20.0, 21.0)))
    end

    @testset "Inner-type constructors (direct)" begin
        # 1-tuple convenience for bare Dual{P, T}: lets per-lane rule bodies use
        # uniform `ntuple(closure, Val(N))` at any width, including width 1.
        @test Dual{Float64,Float64}(1.0, (2.0,)) === Dual(1.0, 2.0)

        # Chunked structured Dual{P, NTangent{NTuple{N, T}}} constructors. Tested
        # directly with synthetic types since the dual_type table routes IEEEFloat
        # through NDual; the structured-chunked path is for non-IEEEFloat primals.
        DualChunkedT = Dual{Float32,NTangent{NTuple{2,Float32}}}
        @test DualChunkedT(1.0f0, (2.0f0, 3.0f0)) === Dual(1.0f0, NTangent((2.0f0, 3.0f0)))
        @test DualChunkedT(1.0f0, 5.0f0) === Dual(1.0f0, NTangent((5.0f0, 5.0f0)))

        # NDual NTangent constructor (parametric form, used by Lifted's 2-arg ctor).
        @test NDual{Float64,2}(3.0, NTangent((1.0, 2.0))) ===
            NDual{Float64,2}(3.0, (1.0, 2.0))
    end

    @testset "Memory / MemoryRef inner constructors" begin
        @static if VERSION >= v"1.11-rc4"
            pri = Memory{Float64}(undef, 2)
            tan = Memory{Float64}(undef, 2)
            pri[1] = 1.0
            pri[2] = 2.0
            tan[1] = 10.0
            tan[2] = 20.0
            m = Memory{NDual{Float64,2}}(pri, tan)
            @test m isa Memory{NDual{Float64,2}}
            @test m[1] === NDual{Float64,2}(1.0, (10.0, 10.0))
            @test m[2] === NDual{Float64,2}(2.0, (20.0, 20.0))

            mr_pri = Core.memoryrefnew(pri)
            mr_tan = Core.memoryrefnew(tan)
            mr = MemoryRef{NDual{Float64,2}}(mr_pri, mr_tan)
            @test mr isa MemoryRef{NDual{Float64,2}}
            @test Core.memoryrefget(mr, :not_atomic, false) ===
                NDual{Float64,2}(1.0, (10.0, 10.0))
        end
    end

    @testset "Accessors and round-trip" begin
        # Scalar IEEEFloat: primal/tangent/extract delegate to the inner NDual.
        d = Lifted{Float64,2}(3.0, (1.0, 0.0))
        @test primal(d) === 3.0
        @test tangent(d) === NTangent((1.0, 0.0))
        @test extract(d) === (3.0, NTangent((1.0, 0.0)))

        # Tuple primal: primal maps element-wise; tangent returns top-level
        # `NTangent{NTuple{N, Tuple{...}}}` per audit test #9.
        dt = Lifted{Tuple{Float64,Float64},2}((1.0, 2.0), (10.0, 20.0))
        @test primal(dt) === (1.0, 2.0)
        @test tangent(dt) === NTangent(((10.0, 20.0), (10.0, 20.0)))

        # _lift / _unlift typed identity. Val(0) is primal passthrough.
        @test _lift(Val(2), Float64, _unlift(d)) === d
        @test _lift(Val(0), Float64, 3.0) === 3.0

        # Audit test #9 (struct case): `tangent(::Lifted{struct_P, N, V})`
        # returns a top-level `NTangent{NTuple{N, Tangent{...}}}`, mirroring
        # the structural-array convention. Construct via the 1-arg ctor
        # since the 2-arg path on heterogeneous struct V isn't supported
        # by NamedTuple's primal+tangent constructor.
        inner_v = (
            x=NDual{Float64,2}(1.0, (10.0, 11.0)), y=Mooncake.Dual(:sym, NoTangent())
        )
        dstruct = Lifted{Mooncake.TestResources.StableFoo,2}(inner_v)
        ts = tangent(dstruct)
        @test ts isa NTangent
        @test length(ts.lanes) == 2
        @test ts.lanes[1] isa Mooncake.Tangent
        @test ts.lanes[1].fields.x === 10.0
        @test ts.lanes[2].fields.x === 11.0
    end
end

@testset "Lifted seed factories" begin
    Lifted = Mooncake.Lifted
    NDual = Mooncake.NDual
    zero_lifted = Mooncake.zero_lifted
    uninit_lifted = Mooncake.uninit_lifted
    randn_lifted = Mooncake.randn_lifted

    # Val(0) passthrough — no wrapping. Matches `lifted_type(Val(0), P) === P`.
    @test zero_lifted(Val(0), 1.0) === 1.0
    @test uninit_lifted(Val(0), 1.0) === 1.0
    @test randn_lifted(Val(0), Random.MersenneTwister(0), 1.0) === 1.0

    # Val(N) wraps the Layer-2 result. Both factories consult the same
    # `dual_type` table, so the inner agrees by construction.
    @test zero_lifted(Val(2), 1.0) === Lifted{Float64,2}(zero_dual(Val(2), 1.0))
    @test uninit_lifted(Val(2), 1.0) ===
        Lifted{Float64,2}(Mooncake.uninit_dual(Val(2), 1.0))

    # `randn_lifted` partials are nondeterministic; check shape and primal.
    let r = randn_lifted(Val(2), Random.MersenneTwister(0), 1.0)
        @test typeof(r) === Lifted{Float64,2,NDual{Float64,2}}
        @test primal(r) === 1.0
    end

    # Container primals route through the Layer-2 zero_dual overloads. Result
    # type matches `lifted_type(Val(N), typeof(x))`.
    let v = [1.0, 2.0]
        d = zero_lifted(Val(2), v)
        @test typeof(d) === Lifted{Vector{Float64},2,Vector{NDual{Float64,2}}}
        @test primal(d) == v
    end

    # Complex primal.
    let z = complex(1.0, 2.0)
        d = zero_lifted(Val(2), z)
        @test typeof(d) === Lifted{ComplexF64,2,Complex{NDual{Float64,2}}}
        @test primal(d) === z
    end
end

@testset "Lifted-aware frule!! direct dispatch" begin
    # Phase 6: exercises the `Lifted`-typed `frule!!` overloads directly
    # (bypassing the IR-emit). The integration tests cover these via
    # `prepare_derivative_cache(...; chunk_size=N)`, but those go through the
    # IR-emit's `_is_lifted_aware` trait check and the Phase-3 wrap/unwrap
    # scaffolding for non-registered rules. Here we call the rule with `Lifted`
    # args directly — Julia dispatch picks the specific `Lifted{typeof(op), N}`
    # overload (for `tuple`, `_new_`) or the generic `frule!!(::Lifted{F, N},
    # args::Vararg{Lifted, M})` adapter (for any other rule).
    Lifted = Mooncake.Lifted
    NDual = Mooncake.NDual
    NTangent = Mooncake.NTangent
    _unlift = Mooncake._unlift
    zero_lifted = Mooncake.zero_lifted
    frule!! = Mooncake.frule!!

    @testset "tuple — specific Lifted overload, three-branch collapse" begin
        # Two NDual{Float64, 2} args representing the standard basis at width 2.
        ftuple = zero_lifted(Val(2), tuple)
        a = Lifted{Float64,2}(NDual{Float64,2}(1.0, (1.0, 0.0)))
        b = Lifted{Float64,2}(NDual{Float64,2}(2.0, (0.0, 1.0)))

        result = frule!!(ftuple, a, b)

        # Single outer Lifted{<:Tuple} whose V is the bare element-wise tuple of
        # inner duals — invariant from the design.
        @test typeof(result) ===
            Lifted{Tuple{Float64,Float64},2,Tuple{NDual{Float64,2},NDual{Float64,2}}}
        @test _unlift(result) ===
            (NDual{Float64,2}(1.0, (1.0, 0.0)), NDual{Float64,2}(2.0, (0.0, 1.0)))
        @test primal(result) === (1.0, 2.0)
        @test tangent(result) === NTangent(((1.0, 0.0), (0.0, 1.0)))
    end

    @testset "tuple — heterogeneous NDual / Dual{NoTangent} args" begin
        # Mixed-shape inputs: one differentiable Float64, one non-differentiable Int.
        # `dual_type(Val(2), Int) == Dual{Int, NoTangent}`, so the Int's V is bare Dual.
        ftuple = zero_lifted(Val(2), tuple)
        a = Lifted{Float64,2}(NDual{Float64,2}(1.5, (1.0, 0.0)))
        b = Lifted{Int,2}(Dual(7, NoTangent()))

        result = frule!!(ftuple, a, b)

        # Inner V is `Tuple{NDual{...}, Dual{Int, NoTangent}}` — element-wise
        # element types reflect each arg's V exactly.
        @test typeof(result) ===
            Lifted{Tuple{Float64,Int},2,Tuple{NDual{Float64,2},Dual{Int,NoTangent}}}
        @test primal(result) === (1.5, 7)
    end

    @testset "_new_ — specific Lifted overload, struct branch" begin
        # Build a chunked struct via `_new_`. Use the existing `OneField{Float64}`
        # test resource so we don't need to define a struct here.
        OneField = Mooncake.TestResources.OneField
        f_new = zero_lifted(Val(2), Mooncake._new_)
        # Type literal lifts to `Lifted{Type{OneField{Float64}}, 2, Dual{..., NoTangent}}`
        ptype = Lifted{Type{OneField{Float64}},2}(Dual(OneField{Float64}, NoTangent()))
        x = Lifted{Float64,2}(NDual{Float64,2}(3.0, (1.0, 0.0)))

        result = frule!!(f_new, ptype, x)

        # Result wraps a struct primal of type `OneField{Float64}` with a
        # chunked NTangent of length 2 over the per-direction field tangents.
        @test typeof(primal(result)) === OneField{Float64}
        @test primal(result).a === 3.0
        @test result isa Lifted{OneField{Float64},2}
    end

    @testset "Generic delegator — unregistered rule" begin
        # `add_float` is not registered as Lifted-aware (no specific Lifted
        # overload), so calling `frule!!` with `Lifted` args dispatches to the
        # generic `frule!!(::Lifted{F, N}, args::Vararg{Lifted, M})` adapter
        # in `primal_mode.jl`. The adapter unwraps, calls the bare frule, and
        # re-wraps via `__get_primal` of the bare result.
        addf = zero_lifted(Val(1), Mooncake.IntrinsicsWrappers.add_float)
        a = Lifted{Float64,1}(NDual{Float64,1}(1.5, (1.0,)))
        b = Lifted{Float64,1}(NDual{Float64,1}(2.5, (0.0,)))

        result = frule!!(addf, a, b)

        @test result isa Lifted{Float64,1,NDual{Float64,1}}
        @test primal(result) === 4.0
        @test tangent(result) === NTangent((1.0,))
    end
end
