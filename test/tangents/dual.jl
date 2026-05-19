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

    # `_uninit_dual` at width Val(N) returns a `Lifted` whose inner V is the
    # substituted `Dual{Type{Array{NDual{...},D}}, NoTangent}`.
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

    # The no-`Val` and `Val(1)` queries agree by construction.
    @testset "dual_type(P) === dual_type(Val(1), P)" begin
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
        # `Int` is a NoTangent-leaf case: `dual_type(Val(1), Int)` returns
        # `Dual{Int, NoTangent}` (bare-NoTangent), not the NTangent-wrapped
        # form. This is the canonical representation for non-differentiable
        # primals at every width — `NTangent{Tuple{NoTangent}}` collapses to
        # `NoTangent` per `tangent_type(Val(N), P)` when the leaf is NoTangent.
        # The generic structural case for concrete `P` with a non-NoTangent
        # tangent type uses `Dual{P, NTangent{Tuple{tangent_type(P)}}}`
        # (carve-out lifted in commit cbc5b236b).
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

    @testset "typeassert keeps runtime Lifted concrete" begin
        # Abstract `T` slot annotations should NOT produce exact-abstract
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

    @testset "_canon_return loud failure on shape mismatch" begin
        # `_canon_return` should refuse to silently
        # retag a `Lifted` whose inner V shape disagrees with
        # `dual_type(Val(N), P_target)` — that would have produced an invalid
        # parametric V (e.g. `Lifted{Matrix{Float64}, 1, Vector{NDual}}`).
        # The defensive ndims-reshape in `_wrap_dual_to_lifted` handles the
        # legitimate "rule produced a flat collection" case before retagging.
        # This test isolates the failure path.
        vec_inner = NDual{Float64,1}[
            NDual{Float64,1}(1.0, (0.0,)), NDual{Float64,1}(2.0, (0.0,))
        ]
        vec_lifted = Lifted{Vector{Float64},1,Vector{NDual{Float64,1}}}(vec_inner)
        @test_throws ArgumentError Mooncake._canon_return(
            Val(1), Matrix{Float64}, vec_lifted
        )
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
        # MemoryRef leaves alongside Memory leaves.
        @static if VERSION >= v"1.11-rc4"
            m = fill!(Memory{NDual{Float64,2}}(undef, 2), NDual{Float64,2}(1.0, (2.0, 3.0)))
            @test Mooncake.verify_dual_type(m)
            @test Mooncake.verify_dual_type(Core.memoryrefnew(m))
            cm = fill!(
                Memory{Complex{NDual{Float64,2}}}(undef, 1),
                Complex(
                    NDual{Float64,2}(1.0, (2.0, 3.0)), NDual{Float64,2}(0.0, (0.0, 0.0))
                ),
            )
            @test Mooncake.verify_dual_type(Core.memoryrefnew(cm))
        end
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

    @testset "width-1 wrapper bare-Dual durable exceptions" begin
        # Each wrapper below uses a parallel `Dual{Wrapper, tangent_type(Wrapper)}`
        # representation at width 1 because its rules dispatch through `arrayify`
        # (`Union{Tangent, FData}`-typed), which does not accept `NTangent` or
        # `NDual`-element wrapper containers. Width >= 2 stays at the chunked
        # `Dual{Wrapper, NTangent{NTuple{N, T}}}` form. This testset pins the
        # current behavior so the documented exception cannot regress silently.
        function _pin_width1_bare_dual(W::Type)
            T_W = Mooncake.tangent_type(W)
            @test Mooncake.dual_type(Val(1), W) === Mooncake.Dual{W,T_W}
            # Width 0 is the primal passthrough.
            @test Mooncake.dual_type(Val(0), W) === W
            # Width 2 must remain the canonical chunked Dual.
            T_W_N2 = Mooncake.tangent_type(Val(2), W)
            @test Mooncake.dual_type(Val(2), W) === Mooncake.Dual{W,T_W_N2}
        end
        _pin_width1_bare_dual(Base.ReshapedArray{Float64,1,Vector{Float64},Tuple{}})
        _pin_width1_bare_dual(
            Base.ReinterpretArray{Float64,1,Float64,Vector{Float64},false}
        )
        _pin_width1_bare_dual(LinearAlgebra.UpperTriangular{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.LowerTriangular{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.UnitUpperTriangular{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.UnitLowerTriangular{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.UpperHessenberg{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.Symmetric{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.Hermitian{Float64,Matrix{Float64}})
        _pin_width1_bare_dual(LinearAlgebra.Transpose{Float64,Vector{Float64}})
        # Vector-parented Adjoint: separate width-1 overload at NfwdMooncake.jl.
        _pin_width1_bare_dual(LinearAlgebra.Adjoint{Float64,Vector{Float64}})
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

    @testset "Memory / MemoryRef NDual tangent is top-level NTangent" begin
        @static if VERSION >= v"1.11-rc4"
            # `tangent(::Memory{NDual{T,N}})` returns top-level
            # `NTangent{NTuple{N, Memory{T}}}` (one Memory per lane), mirroring
            # `tangent(::Array{NDual{T,N},D}) === NTangent{NTuple{N, Array{T,D}}}`.
            m_pri = Memory{Float64}(undef, 2)
            m_tan = Memory{Float64}(undef, 2)
            m_pri[1], m_pri[2] = 1.0, 2.0
            m_tan[1], m_tan[2] = 10.0, 20.0
            m = Memory{NDual{Float64,2}}(m_pri, m_tan)
            mt = Mooncake.tangent(m)
            @test mt isa NTangent{NTuple{2,Memory{Float64}}}
            @test mt.lanes[1][1] === 10.0 && mt.lanes[1][2] === 20.0
            @test mt.lanes[2][1] === 10.0 && mt.lanes[2][2] === 20.0

            mr = Core.memoryrefnew(m)
            mrt = Mooncake.tangent(mr)
            @test mrt isa NTangent{NTuple{2,MemoryRef{Float64}}}
        end
    end

    @testset "Accessors and round-trip" begin
        # Scalar IEEEFloat: primal/tangent/extract delegate to the inner NDual.
        d = Lifted{Float64,2}(3.0, (1.0, 0.0))
        @test primal(d) === 3.0
        @test tangent(d) === NTangent((1.0, 0.0))
        @test extract(d) === (3.0, NTangent((1.0, 0.0)))

        # Tuple primal: primal maps element-wise; tangent returns top-level
        # `NTangent{NTuple{N, Tuple{...}}}`.
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

@testset "test_dual / test_lifted over tangent_test_cases" begin
    # Wires the Layer-2 / Layer-3 mirrored test utilities (`test_dual_types`,
    # `test_dual`, `test_lifted_types`, `test_lifted`) over every primal in
    # `tangent_test_cases()`, so canonicality failures (seed factories
    # returning a non-`dual_type`-matching shape, slot widening on Type
    # primals, etc.) become interface failures rather than scattered
    # regressions. Mirrors the `test_tangent` interface sweep in
    # `test/tangents/tangents.jl`.
    rng = StableRNG(123_456)
    for (_, p, _...) in Mooncake.tangent_test_cases()
        @testset "$(typeof(p))" begin
            TestUtils.test_dual_types(typeof(p))
            TestUtils.test_dual(rng, p)
            TestUtils.test_lifted_types(typeof(p))
            TestUtils.test_lifted(rng, p)
        end
    end
end

@testset "_ndual_output_to_width1 NTangent stripping" begin
    # Regression guard for the central-adapter return contract. When a
    # `Lifted{P, N, V}` has inner V `Dual{P, <:NTangent}` (the canonical
    # width-1 form for primals without an NDual specialisation — e.g.
    # `BFloat16`, generic concrete `P`), `_ndual_output_to_width1` must
    # strip the NTangent down to the lane-1 tangent so the result has the
    # documented legacy bare-T shape `Dual{P, T}`. Without this, the
    # central-adapter pattern in `ext/MooncakeBFloat16sExt.jl` and
    # `src/rules/rules_via_nfwd.jl` would surface NTangent-wrapped duals
    # to direct `frule!!` callers (test_rule, debug paths,
    # `value_and_derivative!!`).
    @testset "bare-T strip from Dual{Float64, NTangent}" begin
        # Build a width-1 Lifted with NTangent-wrapped inner V directly.
        inner = Dual(3.14, NTangent((2.0,)))
        slot = Lifted{Float64,1,typeof(inner)}(inner)
        @test slot isa Lifted{Float64,1,Dual{Float64,NTangent{Tuple{Float64}}}}
        out = Mooncake._ndual_output_to_width1(slot)
        @test out isa Dual{Float64,Float64}
        @test primal(out) === 3.14
        @test tangent(out) === 2.0
    end
    @testset "NDual inner V already bare-T" begin
        # When the inner V is NDual (IEEEFloat canonical), the existing
        # `_ndual_output_to_width1(::NDual)` path strips to bare-T Dual.
        # This is unaffected by the BFloat16s-driven fix; pin it.
        nd = Mooncake.NDual{Float64,1}(1.5, (0.25,))
        slot = Lifted{Float64,1,typeof(nd)}(nd)
        out = Mooncake._ndual_output_to_width1(slot)
        @test out isa Dual{Float64,Float64}
        @test primal(out) === 1.5
        @test tangent(out) === 0.25
    end
    @testset "end-to-end IEEEFloat central adapter dispatch" begin
        # Direct test of the central-adapter flow that `test_rule` exercises
        # for `rules_via_nfwd.jl` ops: bare-Dual `frule!!` call → Lifted
        # body → `_ndual_output_to_width1` → bare-T `Dual{P, P}`. Uses
        # `exp` (a scalar primitive in `rules_via_nfwd.jl`) at width-1.
        fdual = Mooncake.zero_dual(exp)
        d = Dual(0.7, 1.0)
        result = Mooncake.frule!!(fdual, d)
        @test result isa Dual{Float64,Float64}
        @test primal(result) ≈ exp(0.7)
        @test tangent(result) ≈ exp(0.7)  # d/dx exp(x) = exp(x)
    end
end

@testset "Lifted-aware frule!! direct dispatch" begin
    # Exercises the `Lifted`-typed `frule!!` overloads directly (bypassing the
    # IR-emit). The integration tests cover these via
    # `prepare_derivative_cache(...; chunk_size=N)`, but those go through the
    # IR-emit's `_is_lifted_aware` trait check and the wrap/unwrap
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

    @testset "chunk-size-2 with unequal lane seeds" begin
        # Chunked correctness probe. The bare `frule!!(::Dual)`
        # path computes ONE lane; if a Lifted-aware wrapper passes a bare
        # Dual{P,T} result up through `_wrap_rule_result` at `Val(2)`, the
        # broadcast inner-V ctor duplicates the scalar tangent across all
        # lanes. With unequal lane seeds, this surfaces as a duplicate value.
        #
        # The Nfwd scalar primitives (e.g. `sin`) operate directly on
        # `NDual{T, 2}` and preserve all lanes correctly. This test pins
        # that correctness so a regression that drops the chunked path is
        # caught immediately.
        x = Lifted{Float64,2}(NDual{Float64,2}(2.0, (10.0, 20.0)))
        sinf = zero_lifted(Val(2), sin)
        y = frule!!(sinf, x)

        # Expected: d(sin x)/dx evaluated at x=2 with both seed lanes,
        # producing two distinct lane values that are NOT just duplicates.
        ct = cos(2.0)
        @test primal(y) === sin(2.0)
        @test tangent(y) === NTangent((ct * 10.0, ct * 20.0))
        @test tangent(y).lanes[1] !== tangent(y).lanes[2]

        # `_wrap_rule_result` broadcasts a bare-T scalar tangent across all N
        # lanes (documented limitation): only valid when both lanes
        # are intentionally equal. With distinct lane seeds at the source
        # this is a duplicate-lane bug, but for a SCALAR tangent passed
        # explicitly (e.g. a rule that genuinely returns identical-lane
        # results), the broadcast is the intended behaviour.
        r_bare = Mooncake._wrap_rule_result(Float64, Val(2), Dual(4.0, 7.0))
        @test tangent(r_bare) === NTangent((7.0, 7.0))

        # NTangent-bearing result preserves both lanes — the canonical path.
        r_nt = Mooncake._wrap_rule_result(Float64, Val(2), Dual(4.0, NTangent((7.0, 9.0))))
        @test tangent(r_nt) === NTangent((7.0, 9.0))

        # Additional known-good chunked paths via `rules_via_nfwd` (direct
        # `NDual{T, N}` ops preserve all lanes). These pin that the unary
        # / binary IEEEFloat intrinsics and the `tuple` specific overload
        # remain width-N correct under future changes.
        @testset "rules_via_nfwd binary add_float" begin
            a = Lifted{Float64,2}(NDual{Float64,2}(2.0, (10.0, 20.0)))
            b = Lifted{Float64,2}(NDual{Float64,2}(3.0, (5.0, 15.0)))
            addf = zero_lifted(Val(2), Mooncake.IntrinsicsWrappers.add_float)
            z = frule!!(addf, a, b)
            @test primal(z) === 5.0
            @test tangent(z) === NTangent((15.0, 35.0))
        end

        @testset "rules_via_nfwd unary abs_float with negative primal" begin
            xn = Lifted{Float64,2}(NDual{Float64,2}(-3.0, (10.0, 20.0)))
            absf = zero_lifted(Val(2), Mooncake.IntrinsicsWrappers.abs_float)
            yn = frule!!(absf, xn)
            @test primal(yn) === 3.0
            @test tangent(yn) === NTangent((-10.0, -20.0))
        end

        @testset "tuple specific overload with unequal seeds" begin
            a = Lifted{Float64,2}(NDual{Float64,2}(2.0, (10.0, 20.0)))
            b = Lifted{Float64,2}(NDual{Float64,2}(3.0, (5.0, 15.0)))
            ftuple = zero_lifted(Val(2), tuple)
            result = frule!!(ftuple, a, b)
            # Element-wise tuple-of-NDual preserves each NDual's lanes.
            @test tangent(result) === NTangent(((10.0, 5.0), (20.0, 15.0)))
        end
    end

    @testset "nested struct tangent shape" begin
        # `tangent(lo, i)` and `tangent(lo)` must rebuild nested `Tangent{...}`
        # wrappers for struct fields, not leak raw `NamedTuple` from the
        # recursive lift's inner V.
        struct AuditTodo1Inner
            x::Float64
        end
        struct AuditTodo1Outer
            a::AuditTodo1Inner
            b::Tuple{AuditTodo1Inner,Float64}
        end

        o = AuditTodo1Outer(AuditTodo1Inner(1.0), (AuditTodo1Inner(2.0), 3.0))
        lo = Lifted{AuditTodo1Outer,2}(o, Mooncake.zero_tangent(o))

        # Per-lane direction tangent: nested struct fields are `Tangent{...}`.
        t1 = tangent(lo, 1)
        @test t1 isa Mooncake.Tangent
        @test t1.fields.a isa Mooncake.Tangent
        @test t1.fields.b isa Tuple
        @test t1.fields.b[1] isa Mooncake.Tangent

        # Top-level `tangent(lo)` matches `tangent_type(Val(2), AuditTodo1Outer)`.
        @test typeof(tangent(lo)) === Mooncake.tangent_type(Val(2), AuditTodo1Outer)

        # `primal(lo)` reconstructs the struct rather than leaking the lifted
        # inner V's NamedTuple.
        @test primal(lo) === o

        # NamedTuple primal containing a struct element: lane tangent rebuilds
        # the inner `Tangent{...}` and the no-`i` path matches `tangent_type`.
        nt = (a=AuditTodo1Inner(5.0), b=7.0)
        ln = Lifted{typeof(nt),2}(nt, Mooncake.zero_tangent(nt))
        @test tangent(ln, 1) isa NamedTuple
        @test tangent(ln, 1).a isa Mooncake.Tangent
        @test typeof(tangent(ln)) === Mooncake.tangent_type(Val(2), typeof(nt))

        # Tuple primal containing a struct element: same property.
        tp = (AuditTodo1Inner(11.0), 13.0)
        lt = Lifted{typeof(tp),2}(tp, Mooncake.zero_tangent(tp))
        @test tangent(lt, 1) isa Tuple
        @test tangent(lt, 1)[1] isa Mooncake.Tangent
        @test typeof(tangent(lt)) === Mooncake.tangent_type(Val(2), typeof(tp))
    end

    @testset "BLAS width-N per-lane independence" begin
        # Pins that the BLAS width-N rules (gemv!, gemm!, scal!, nrm2)
        # produce independent per-lane tangents — the pre-migration width-1
        # broadcast-via-`_wrap_rule_result` would yield identical lanes.
        using LinearAlgebra: BLAS

        # BLAS.nrm2 width-2 with distinct per-lane seeds.
        let X = [1.0, 2.0, 3.0], seeds_1 = [0.1, 0.2, 0.3], seeds_2 = [0.01, 0.02, 0.03]
            X_n2 = [NDual{Float64,2}(X[i], (seeds_1[i], seeds_2[i])) for i in 1:length(X)]
            r = Mooncake.frule!!(
                Mooncake.zero_dual(BLAS.nrm2),
                Mooncake.zero_dual(3),
                X_n2,
                Mooncake.zero_dual(1),
            )
            @test r isa NDual{Float64,2}
            # Lane 1 tangent ≠ Lane 2 tangent (distinct seeds → distinct derivatives).
            @test r.partials[1] != r.partials[2]
        end

        # BLAS.gemv! width-2 with distinct per-lane seeds.
        let A = [1.0 2.0; 3.0 4.0], x = [1.0, 2.0], y = [0.0, 0.0]
            A_n2 = reshape(
                [NDual{Float64,2}(A[i], (A[i] * 0.1, A[i] * 0.01)) for i in 1:length(A)],
                2,
                2,
            )
            x_n2 = [NDual{Float64,2}(x[i], (x[i] * 0.1, x[i] * 0.01)) for i in 1:length(x)]
            y_n2 = [NDual{Float64,2}(0.0, (0.0, 0.0)) for _ in 1:length(y)]
            α_n2 = NDual{Float64,2}(1.0, (0.0, 0.0))
            β_n2 = NDual{Float64,2}(0.0, (0.0, 0.0))
            Mooncake.frule!!(
                Mooncake.zero_dual(BLAS.gemv!),
                Mooncake.zero_dual('N'),
                α_n2,
                A_n2,
                x_n2,
                β_n2,
                y_n2,
            )
            # Per-lane tangents differ because seeds differ.
            @test y_n2[1].partials[1] != y_n2[1].partials[2]
            @test y_n2[2].partials[1] != y_n2[2].partials[2]
            # Primal is the same regardless of lanes.
            @test y_n2[1].value == 5.0   # 1*1 + 2*2 = 5
            @test y_n2[2].value == 11.0  # 3*1 + 4*2 = 11
        end

        # BLAS.scal! width-2: x ← a*x with per-lane Frechet `a*dx + da*x`.
        let X = [1.0, 2.0, 3.0]
            X_n2 = [NDual{Float64,2}(X[i], (0.1 * X[i], 0.01 * X[i])) for i in 1:length(X)]
            a_n2 = NDual{Float64,2}(2.0, (0.5, 0.05))
            Mooncake.frule!!(
                Mooncake.zero_dual(BLAS.scal!),
                Mooncake.zero_dual(3),
                a_n2,
                X_n2,
                Mooncake.zero_dual(1),
            )
            # Primal X = a*X = 2 * [1, 2, 3] = [2, 4, 6].
            @test [X_n2[i].value for i in 1:3] == [2.0, 4.0, 6.0]
            # Lane 1: a*dX_1 + da_1*X = 2*[0.1, 0.2, 0.3] + 0.5*[1, 2, 3] = [0.7, 1.4, 2.1]
            @test [X_n2[i].partials[1] for i in 1:3] ≈ [0.7, 1.4, 2.1]
            # Lane 2: 2*[0.01, 0.02, 0.03] + 0.05*[1, 2, 3] = [0.07, 0.14, 0.21]
            @test [X_n2[i].partials[2] for i in 1:3] ≈ [0.07, 0.14, 0.21]
        end
    end

    @testset "LAPACK width-N per-lane independence" begin
        using LinearAlgebra: LAPACK

        # LAPACK.getrf! width-2: A_dA is overwritten with LU factor + Frechet
        # tangent. The primal is the same for all lanes; the tangents differ
        # because seeds differ.
        let A = [4.0 3.0; 6.0 3.0]
            A_primal = copy(A)
            A_n2 = reshape(
                [NDual{Float64,2}(A[i], (A[i] * 0.1, A[i] * 0.01)) for i in 1:length(A)],
                2,
                2,
            )
            r = Mooncake.frule!!(Mooncake.zero_dual(LAPACK.getrf!), A_n2)
            # Returned tuple: (A_dA, ipiv_dual, info_dual).
            @test length(r) == 3
            # The two-lane partials differ on at least one entry because the
            # input seeds differed.
            differ_count = 0
            for i in 1:length(A_n2)
                if A_n2[i].partials[1] != A_n2[i].partials[2]
                    differ_count += 1
                end
            end
            @test differ_count > 0
        end

        # LAPACK.lacpy! width-2: copy A → B (triangular part); per-lane
        # tangent copy independent.
        let A = [1.0 2.0; 3.0 4.0], B = zeros(2, 2)
            A_n2 = reshape(
                [NDual{Float64,2}(A[i], (A[i] * 0.1, A[i] * 0.01)) for i in 1:length(A)],
                2,
                2,
            )
            B_n2 = [NDual{Float64,2}(0.0, (0.0, 0.0)) for _ in 1:length(B)]
            B_n2 = reshape(B_n2, 2, 2)
            Mooncake.frule!!(
                Mooncake.zero_dual(LAPACK.lacpy!), B_n2, A_n2, Mooncake.zero_dual('A')
            )
            # B primal == A primal after full-rectangle copy.
            @test [B_n2[i].value for i in 1:length(B_n2)] == [A[i] for i in 1:length(A)]
            # B lane-1 tangent == A lane-1 tangent (linear copy preserves).
            @test [B_n2[i].partials[1] for i in 1:length(B_n2)] ≈ [A[i] * 0.1 for i in 1:length(A)]
            @test [B_n2[i].partials[2] for i in 1:length(B_n2)] ≈ [A[i] * 0.01 for i in 1:length(A)]
            # Lane independence.
            @test [B_n2[i].partials[1] for i in 1:length(B_n2)] != [B_n2[i].partials[2] for i in 1:length(B_n2)]
        end
    end

    @testset "Lifted{P,N}(struct, NTangent) ctor" begin
        # Pre-fix: `Lifted{StructP, N}(struct_primal, NTangent_of_Tangents)`
        # at width N≥2 raised MethodError because the canonical inner V (the
        # structural NamedTuple lift) had no constructor accepting
        # `(struct, NTangent{NTuple{N, Tangent}})`. The `@generated` ctor
        # parallel to the existing `tangent::Tangent` form unrolls per-lane
        # field extraction so each field gets an N-tuple of partial values.
        #
        # Reproducer pattern: a `frule!!` that returns a Lifted with struct
        # primal at width N≥2 (e.g. `lmemoryrefget` on `MemoryRef{<:Struct}`).

        # Use LoHi: leaf-Float64 fields only (covers the canonical
        # struct-primal × width-N path without nested-array recursion).
        p = TestResources.LoHi(2.5, -1.5)
        nt = Mooncake.NTangent((
            Mooncake.Tangent((; lo=0.1, hi=0.2)), Mooncake.Tangent((; lo=0.3, hi=0.4))
        ))

        L = Mooncake.Lifted{TestResources.LoHi,2}(p, nt)
        InnerV = typeof(L).parameters[3]
        @test InnerV <: NamedTuple
        @test fieldcount(InnerV) == 2
        # Per-field NDual at width 2 with correctly-routed per-lane partials.
        @test fieldtype(InnerV, :lo) === Mooncake.Nfwd.NDual{Float64,2}
        @test fieldtype(InnerV, :hi) === Mooncake.Nfwd.NDual{Float64,2}
        # `primal(::Lifted{P,N,V<:NamedTuple})` reconstructs the original
        # struct; `tangent(...)` reconstructs the NTangent.
        @test Mooncake.primal(L) === p
        @test Mooncake.tangent(L).lanes[1].fields.lo == 0.1
        @test Mooncake.tangent(L).lanes[1].fields.hi == 0.2
        @test Mooncake.tangent(L).lanes[2].fields.lo == 0.3
        @test Mooncake.tangent(L).lanes[2].fields.hi == 0.4
        # The inner V's per-field NDuals carry the per-lane partials directly.
        @test L.value.lo.value == 2.5
        @test L.value.lo.partials == (0.1, 0.3)
        @test L.value.hi.value == -1.5
        @test L.value.hi.partials == (0.2, 0.4)
    end

    @testset "memoryrefget / lmemoryrefget width-N struct-element" begin
        # Pre-fix: `frule!!(::Lifted{typeof(memoryrefget), N}, ...)` and
        # `frule!!(::Lifted{typeof(lmemoryrefget), N}, ...)` erred at
        # width N≥2 for struct-element MemoryRef. The bare-Dual delegator
        # path called `_ntangent_unwrap_singleton` (singleton-NTangent only) on a
        # multi-lane NTangent, returning the NTangent itself which
        # memoryrefget rejected with `expected GenericMemoryRef`.
        # Fix: width-N-specific overload doing per-lane processing.
        @static if VERSION >= v"1.11-"
            m = Memory{TestResources.LoHi}(undef, 4)
            m[1] = TestResources.LoHi(10.0, 11.0)
            m[2] = TestResources.LoHi(20.0, 21.0)
            mref = Core.memoryrefnew(Core.memoryrefnew(m), 2)

            for op in (Core.memoryrefget, Mooncake.lmemoryrefget)
                # Width-1 still works (generic delegator path).
                f1 = Mooncake.zero_lifted(Val(1), op)
                x1 = Mooncake.zero_lifted(Val(1), mref)
                # lmemoryrefget takes Val args; memoryrefget takes Symbol/Bool.
                ord1 = if op === Mooncake.lmemoryrefget
                    Mooncake.zero_lifted(Val(1), Val(:not_atomic))
                else
                    Mooncake.zero_lifted(Val(1), :not_atomic)
                end
                bc1 = if op === Mooncake.lmemoryrefget
                    Mooncake.zero_lifted(Val(1), Val(false))
                else
                    Mooncake.zero_lifted(Val(1), false)
                end
                r1 = Mooncake.frule!!(f1, x1, ord1, bc1)
                @test Mooncake.primal(r1) === TestResources.LoHi(20.0, 21.0)

                # Width-2 was the broken case.
                f2 = Mooncake.zero_lifted(Val(2), op)
                x2 = Mooncake.zero_lifted(Val(2), mref)
                ord2 = if op === Mooncake.lmemoryrefget
                    Mooncake.zero_lifted(Val(2), Val(:not_atomic))
                else
                    Mooncake.zero_lifted(Val(2), :not_atomic)
                end
                bc2 = if op === Mooncake.lmemoryrefget
                    Mooncake.zero_lifted(Val(2), Val(false))
                else
                    Mooncake.zero_lifted(Val(2), false)
                end
                r2 = Mooncake.frule!!(f2, x2, ord2, bc2)
                @test Mooncake.primal(r2) === TestResources.LoHi(20.0, 21.0)
                # The result is the structural NamedTuple lift at width 2.
                @test typeof(r2).parameters[3] <: NamedTuple
            end
        end
    end

    @testset "unsafe_copyto! width-N struct-element MemoryRef" begin
        # Pre-fix: at width N≥2 the generic Lifted delegator routed through
        # the bare-Dual body which used `_ntangent_unwrap_singleton` (singleton-NTangent
        # only); the call `unsafe_copyto!(NTangent, NTangent, n)` errored.
        # Fix: width-N-specific overload doing per-lane copy.
        @static if VERSION >= v"1.11-"
            m_src = Memory{TestResources.LoHi}(undef, 4)
            m_src[1] = TestResources.LoHi(1.0, 2.0)
            m_src[2] = TestResources.LoHi(3.0, 4.0)
            m_dst = Memory{TestResources.LoHi}(undef, 4)
            m_dst[1] = TestResources.LoHi(0.0, 0.0)
            m_dst[2] = TestResources.LoHi(0.0, 0.0)
            src_ref = Core.memoryrefnew(m_src)
            dst_ref = Core.memoryrefnew(m_dst)

            for N in (1, 2)
                f = Mooncake.zero_lifted(Val(N), Base.unsafe_copyto!)
                d = Mooncake.zero_lifted(Val(N), dst_ref)
                s = Mooncake.zero_lifted(Val(N), src_ref)
                n = Mooncake.zero_lifted(Val(N), 2)
                r = Mooncake.frule!!(f, d, s, n)
                # Returns the dest Lifted; underlying primal copy executed.
                @test r === d
                @test m_dst[1] == TestResources.LoHi(1.0, 2.0)
                @test m_dst[2] == TestResources.LoHi(3.0, 4.0)
                # Reset dst for the next width.
                m_dst[1] = TestResources.LoHi(0.0, 0.0)
                m_dst[2] = TestResources.LoHi(0.0, 0.0)
            end
        end
    end

    @testset "lmemoryrefset! width-N struct-value" begin
        # Pre-fix: rule at memory.jl:881 (Lifted with `V<:NamedTuple` for the
        # value arg) called `memoryrefset!(tangent(bare_x), tangent(y), ...)`
        # but `tangent(bare_x)` is the outer NTangent wrapper, not a bare
        # MemoryRef — errored with `expected GenericMemoryRef`. Fix: set the
        # primal once and write each lane's tangent independently using
        # `tangent(value, n)` (per-lane Tangent via `_build_struct_tangent_dir`).
        @static if VERSION >= v"1.11-"
            for N in (1, 2)
                m = Memory{TestResources.LoHi}(undef, 4)
                m[1] = TestResources.LoHi(10.0, 11.0)
                mref = Core.memoryrefnew(Core.memoryrefnew(m), 1)
                new_val = TestResources.LoHi(99.0, 88.0)

                f = Mooncake.zero_lifted(Val(N), Mooncake.lmemoryrefset!)
                x = Mooncake.zero_lifted(Val(N), mref)
                v = Mooncake.zero_lifted(Val(N), new_val)
                ord = Mooncake.zero_lifted(Val(N), Val(:not_atomic))
                bc = Mooncake.zero_lifted(Val(N), Val(false))

                r = Mooncake.frule!!(f, x, v, ord, bc)
                @test r === v  # returns the original value Lifted
                # In-place primal write verified.
                @test m[1] == new_val
            end
        end
    end

    @testset "pointer_from_objref width-N" begin
        # Pre-fix: at width N≥2 the rule called
        # `pointer_from_objref(_ntangent_unwrap_singleton(tangent(inner)))`
        # where the singleton-NTangent-unwrap helper didn't match the
        # multi-lane NTangent → fell through to the no-op fallback returning
        # the NTangent unchanged → `pointer_from_objref(NTangent)` errored
        # because NTangent is immutable.
        # Fix: detect multi-lane NTangent and map pointer_from_objref over
        # each lane's tangent independently, building per-lane Ptrs.
        mutable struct _PMutable
            x::Float64
        end
        for N in (1, 2)
            m = _PMutable(1.0)
            f = Mooncake.zero_lifted(Val(N), pointer_from_objref)
            ml = Mooncake.zero_lifted(Val(N), m)
            r = Mooncake.frule!!(f, ml)
            if N == 2
                # Per-lane Ptrs differ: each points to a different mutable
                # tangent object.
                lanes = r.value.tangent.lanes
                @test lanes[1] !== lanes[2]
            end
        end
    end

    @testset "copy(::NTangent) per-lane independent" begin
        # Pre-fix: `copy(::NTangent)` had no method. Callers that
        # whole-copy a `Dual{P, NTangent}` (e.g. `Base.copy(::Memory{<:Struct})`
        # via the `:jl_genericmemory_copy` foreigncall path) errored with
        # MethodError. Fix: define `Base.copy(t::NTangent) = NTangent(map(copy,
        # t.lanes))` — per-lane copy so callers get independent lane tangents.
        t = Mooncake.NTangent(([1.0, 2.0], [3.0, 4.0]))
        c = copy(t)
        @test c isa Mooncake.NTangent
        @test c.lanes[1] !== t.lanes[1]
        @test c.lanes[2] !== t.lanes[2]
        @test c.lanes[1] == t.lanes[1]
        @test c.lanes[2] == t.lanes[2]
    end

    @testset "Core.memorynew(Memory{<:Struct}, n) width-N lane independence" begin
        # Same silent-correctness aliasing pattern as
        # `Memory{LoHi}(undef, n)`, but reached via
        # `Core.memorynew` on Julia 1.12+. The pre-fix else-branch at
        # memory.jl:1170 called `Lifted{Memory{P}, N}(x, dx)` with a
        # single `dx` Memory, broadcasting into all N NTangent lanes.
        # Fix: when canonical V is `Dual{Memory{P}, NTangent}` at N≥2,
        # allocate N independent tangent Memories via `Core.memorynew`.
        @static if VERSION >= v"1.12-"
            for N in (1, 2)
                f = Mooncake.zero_lifted(Val(N), Core.memorynew)
                ty = Mooncake.zero_lifted(Val(N), Memory{TestResources.LoHi})
                nl = Mooncake.zero_lifted(Val(N), 5)
                r = Mooncake.frule!!(f, ty, nl)
                if N == 2
                    lanes = r.value.tangent.lanes
                    @test lanes[1] !== lanes[2]
                end
            end
            for N in (1, 2)
                f = Mooncake.zero_lifted(Val(N), Core.memorynew)
                ty = Mooncake.zero_lifted(Val(N), Memory{Float64})
                nl = Mooncake.zero_lifted(Val(N), 5)
                r = Mooncake.frule!!(f, ty, nl)
                @test typeof(r).parameters[3] <: Memory{<:Mooncake.Nfwd.NDual}
            end
        end
    end

    @testset "_new_(Array{<:Struct}, ref, sz) width-N lane independence" begin
        # Pre-fix: `_new_(Vector{LoHi}, ref, sz)` at width N≥2 went through
        # the bare-Dual rule which called `_new_(Array{Tangent, M},
        # tangent(ref), size)` where `tangent(ref)` is the NTangent
        # wrapper. `_new_` stuffed the NTangent into the new Array's `.ref`
        # field via `:new`, producing a malformed Array; `_wrap_rule_result`
        # then aliased that single malformed Array into all N NTangent
        # lanes — two bugs in one: lane aliasing AND corrupted Array
        # internals.
        # Fix: detect NTangent-wrapped Array V at N≥2 and build N
        # independent per-lane tangent Arrays from per-lane MemoryRefs.
        @static if VERSION >= v"1.11-"
            for N in (1, 2)
                m = Memory{TestResources.LoHi}(undef, 3)
                m[1] = TestResources.LoHi(1.0, 2.0)
                m[2] = TestResources.LoHi(3.0, 4.0)
                m[3] = TestResources.LoHi(5.0, 6.0)
                mref = Core.memoryrefnew(m)
                f = Mooncake.zero_lifted(Val(N), Mooncake._new_)
                ty = Mooncake.zero_lifted(Val(N), Vector{TestResources.LoHi})
                refl = Mooncake.zero_lifted(Val(N), mref)
                szl = Mooncake.zero_lifted(Val(N), (3,))
                r = Mooncake.frule!!(f, ty, refl, szl)
                if N == 2
                    lanes = r.value.tangent.lanes
                    @test lanes[1] !== lanes[2]
                    @test lanes[1] isa Array
                    @test lanes[2] isa Array
                end
            end
            # Regression: IEEEFloat-element Array still uses bare
            # `Array{NDual{Float64, N}}` V at any width (no Dual wrapping).
            for N in (1, 2)
                m = Memory{Float64}(undef, 3)
                m[1] = 1.0
                m[2] = 2.0
                m[3] = 3.0
                mref = Core.memoryrefnew(m)
                f = Mooncake.zero_lifted(Val(N), Mooncake._new_)
                ty = Mooncake.zero_lifted(Val(N), Vector{Float64})
                refl = Mooncake.zero_lifted(Val(N), mref)
                szl = Mooncake.zero_lifted(Val(N), (3,))
                r = Mooncake.frule!!(f, ty, refl, szl)
                @test typeof(r).parameters[3] <: Vector{<:Mooncake.Nfwd.NDual}
            end
        end
    end

    @testset "Memory{<:Struct}(undef, n) width-N lane independence" begin
        # Pre-fix: `Memory{LoHi}(undef, n)` at width N≥2 went through
        # `_memory_init_kernel` (returns a single Dual with single Memory
        # tangent) → `_wrap_rule_result` → `Lifted{Memory{LoHi}, N}(p, t)`
        # → the Lifted ctor broadcast the single tangent Memory into N
        # NTangent lanes, producing aliased lane references (`lanes[1]
        # === lanes[2]`). Writing one lane corrupted all — a silent
        # correctness bug.
        # Fix: when the canonical inner V is `Dual{Memory{P}, NTangent}`
        # at N≥2, allocate N independent tangent Memories explicitly.
        @static if VERSION >= v"1.11-"
            for N in (1, 2)
                f = Mooncake.zero_lifted(Val(N), Memory{TestResources.LoHi})
                u = Mooncake.zero_lifted(Val(N), undef)
                nl = Mooncake.zero_lifted(Val(N), 5)
                r = Mooncake.frule!!(f, u, nl)
                if N == 2
                    lanes = r.value.tangent.lanes
                    @test lanes[1] !== lanes[2]
                end
            end
            # Regression: IEEEFloat-element Memory still uses the canonical
            # `Memory{NDual}` inner V (no Dual wrapping).
            for N in (1, 2)
                f = Mooncake.zero_lifted(Val(N), Memory{Float64})
                u = Mooncake.zero_lifted(Val(N), undef)
                nl = Mooncake.zero_lifted(Val(N), 5)
                r = Mooncake.frule!!(f, u, nl)
                @test typeof(r).parameters[3] <: Memory{<:Mooncake.Nfwd.NDual}
            end
        end
    end

    @testset "memoryrefnew width-N struct-element" begin
        # Pre-fix: `_memoryrefnew_kernel(::Dual{<:Memory})` and
        # `_memoryrefnew_kernel(::Dual{<:MemoryRef}, ::Dual{Int}[, ::Dual{Bool}])`
        # called `memoryrefnew(_ntangent_unwrap_singleton(tangent(x)), ...)` where
        # `_ntangent_unwrap_singleton` only unwraps singleton-NTangent. At width N≥2
        # the multi-lane NTangent stayed wrapped and `memoryrefnew(::NTangent)`
        # errored with `expected GenericMemory[Ref]`. Fix: add
        # `Dual{..., <:NTangent}` overloads that map memoryrefnew over each
        # lane and reassemble the NTangent.
        @static if VERSION >= v"1.11-"
            for N in (1, 2)
                m = Memory{TestResources.LoHi}(undef, 3)
                m[1] = TestResources.LoHi(1.0, 2.0)
                m[2] = TestResources.LoHi(3.0, 4.0)
                m[3] = TestResources.LoHi(5.0, 6.0)
                mref = Core.memoryrefnew(m)

                f = Mooncake.zero_lifted(Val(N), Core.memoryrefnew)
                # 1-arg
                ml = Mooncake.zero_lifted(Val(N), m)
                @test (Mooncake.frule!!(f, ml); true)
                # 2-arg with Int offset
                mrl = Mooncake.zero_lifted(Val(N), mref)
                il = Mooncake.zero_lifted(Val(N), 2)
                @test (Mooncake.frule!!(f, mrl, il); true)
                # 3-arg with boundscheck
                bl = Mooncake.zero_lifted(Val(N), false)
                @test (Mooncake.frule!!(f, mrl, il, bl); true)
            end
        end
    end

    @testset "Vector grow/shrink/sizehint width-N struct-element" begin
        # Pre-fix: `_growbeg_kernel!` / `_growend_kernel!` / `_sizehint_kernel!` /
        # `_deletebeg_kernel!` / `_deleteend_kernel!` / `_deleteat_kernel!` /
        # `_growat_kernel!` at memory.jl:1583+ all called
        # `Base._growbeg!(tangent(a), ...)` etc. where `tangent(a)` is the
        # NTangent wrapper for struct-element Vector — none of these ops have
        # methods for NTangent. Fix: add `Dual{<:Vector, <:NTangent}` overloads
        # that apply the op per-lane via `foreach(t -> op(t, ...),
        # tangent(a).lanes)`.
        @static if VERSION >= v"1.11-"
            cases = [
                (Base._growbeg!, (2,)),
                (Base._growend!, (2,)),
                (sizehint!, (10,)),
                (Base._deletebeg!, (1,)),
                (Base._deleteend!, (1,)),
                (Base._deleteat!, (1, 1)),
                (Base._growat!, (2, 1)),
            ]
            for (op, args_extra) in cases
                for N in (1, 2)
                    arr = [
                        TestResources.LoHi(1.0, 2.0),
                        TestResources.LoHi(3.0, 4.0),
                        TestResources.LoHi(5.0, 6.0),
                    ]
                    f = Mooncake.zero_lifted(Val(N), op)
                    al = Mooncake.zero_lifted(Val(N), arr)
                    extras = Tuple(Mooncake.zero_lifted(Val(N), x) for x in args_extra)
                    # Should not throw.
                    @test (Mooncake.frule!!(f, al, extras...); true)
                end
            end
        end
    end

    @testset "copy(Array{<:Struct}) width-N" begin
        # Pre-fix: `_copy_array_kernel(::Dual{<:Array})` at memory.jl:1554
        # called `copy(tangent(a))` where `tangent(a)` is the NTangent wrapper
        # for struct-element Array at any width — `copy(NTangent)` had no
        # method. Fix: add a `_copy_array_kernel(::Dual{<:Array, <:NTangent})`
        # overload that copies each lane's wrapped Array independently.
        arr = [TestResources.LoHi(1.0, 2.0), TestResources.LoHi(3.0, 4.0)]
        for N in (1, 2)
            f = Mooncake.zero_lifted(Val(N), Base.copy)
            al = Mooncake.zero_lifted(Val(N), arr)
            r = Mooncake.frule!!(f, al)
            @test Mooncake.primal(r) == arr
            @test typeof(r).parameters[2] == N
        end
    end

    @testset "memoryrefset! width-N struct-value" begin
        # Pre-fix: the Lifted delegator routed through bare-Dual dispatch
        # which constrained `value::Union{Dual, NDual, ...}` — the bare
        # NamedTuple inner V of a struct-value Lifted didn't match any
        # bare-Dual rule, so dispatch failed with MethodError at all widths.
        # Fix: Lifted-typed overload that forwards to `lmemoryrefset!` at
        # the Lifted level (whose `V<:NamedTuple` overload handles the
        # per-lane struct write).
        @static if VERSION >= v"1.11-"
            for N in (1, 2)
                m = Memory{TestResources.LoHi}(undef, 4)
                m[1] = TestResources.LoHi(10.0, 11.0)
                mref = Core.memoryrefnew(Core.memoryrefnew(m), 1)
                new_val = TestResources.LoHi(99.0, 88.0)

                f = Mooncake.zero_lifted(Val(N), Core.memoryrefset!)
                x = Mooncake.zero_lifted(Val(N), mref)
                v = Mooncake.zero_lifted(Val(N), new_val)
                ord = Mooncake.zero_lifted(Val(N), :not_atomic)
                bc = Mooncake.zero_lifted(Val(N), false)

                r = Mooncake.frule!!(f, x, v, ord, bc)
                @test r === v
                @test m[1] == new_val
            end
        end
    end
end
