# Helpers for the world-advance staleness test below (issue #1218).
# `stale_fwd_lazy` uses LazyFRule; `stale_fwd_dyn` uses DynamicFRule.
stale_fwd_inner(x) = Float32(x) * 2.0f0
@noinline stale_fwd_callee(x) = stale_fwd_inner(x)
# Two `:invoke` levels: one for the pinned rebuild, one for the rules that rebuild itself
# constructs, which predicted at the current world until they were given the pinned one.
@noinline stale_fwd_mid(x) = stale_fwd_callee(x)
stale_fwd_lazy(x) = stale_fwd_mid(x)
const STALE_FWD_FNS = Function[stale_fwd_mid]
stale_fwd_dyn(x) = (STALE_FWD_FNS[1])(x)

@noinline refined_fwd_inner(x) =
    sizeof(x) == sizeof(typeof(x)) ? x : Base.inferencebarrier(x)
@noinline refined_fwd_outer(x) = refined_fwd_inner(x)
refined_fwd(x) = refined_fwd_outer(x)

# Dynamic dispatch (`inferencebarrier` hides the callee) so the derived rule captures a
# `DynamicFRule` with a mutable `cache` Dict — used by the cache-hit `_copy` regression below.
fwd_cache_dyn(x) = Base.inferencebarrier(sin)(x)::Float64 + x

module FwdAliasGlobals
# Read while also passed as the argument, to exercise the refusal of an argument that aliases a
# differentiable global.
const alias_vector = [1.0, 2.0]
alias_read_only(x) = sum(x .* alias_vector)
alias_scalar(x) = x[1] * alias_vector[1]
comparison_vector::Vector{Float64} = ones(1024)
comparison_read(x) = x * comparison_vector[1]
end

module FwdMutableGlobals
mutable struct Params
    scale::Float64
end
struct Holder
    params::Params
end
const params = Params(3.0)
const array = [3.0]
const anyarray = Any[3.0]
const ref = Ref(3.0)
const anyref = Ref{Any}(3.0)
const dict = Dict(:scale => 3.0)
const nested = Holder(Params(3.0))
const immutable = (scale=3.0,)
const complex_array = Any[1.5 + 1.5im]
read_params(x) = x * params.scale
read_array(x) = x * array[1]
read_anyarray(x) = x * anyarray[1]
read_ref(x) = x * ref[]
read_anyref(x) = x * anyref[]
read_dict(x) = x * dict[:scale]
@noinline read_nested_inner(p) = p.params.scale
read_nested(x) = x * read_nested_inner(nested)
read_immutable(x) = x * immutable.scale
read_complex(x) = real(x * complex_array[1]) + imag(x * complex_array[1])
end

# Globals whose STORAGE is replaced or resized between calls, so the V cached at rule build
# names the old storage.
module FwdReplacedGlobals
using LinearAlgebra: dot
mutable struct Box
    array::Vector{Float64}
end

const ref = Ref([3.0])
const box = Box([3.0])
const anyarray = Any[3.0]
const dict = Dict(:a => [3.0])
const getdict = Dict(:a => 3.0)
const tupref = Ref(([3.0],))
const vector = [3.0]
const unionref = Ref{Union{Nothing,Float64}}(nothing)
mutable struct UnionBox
    array::Union{Nothing,Vector{Float64}}
end
const unionbox = UnionBox(nothing)
read_unionref(x) = unionref[] isa Float64 ? x * unionref[] : x
read_unionbox(x) = unionbox.array isa Vector ? x * unionbox.array[end] : x
read_index(x) = x * getfield(box, 1)[end]
read_sum(x) = x * sum(vector)
read_abs2(x) = x * sum(abs2, vector)
read_mapreduce(x) = x * mapreduce(abs2, +, vector)
read_bcast(x) = sum(x .* vector)
read_dot(x) = x * dot(vector, vector)
read_copy(x) = x * sum(copy(vector))
read_sort(x) = x * sum(sort(vector))
read_complex(x) = real(x * sum(complex(vector)))
function read_unsafe_copy(x)
    x * sum(unsafe_copyto!(zeros(length(vector)), 1, vector, 1, length(vector)))
end
Base.@noinline indirect_ref(r) = r[][end]
read_indirect_ref(x) = x * indirect_ref(ref)
read_ref(x) = x * ref[][end]
read_box(x) = x * box.array[end]
read_any(x) = x * sum(anyarray[1])
read_dict(x) = x * dict[:a][end]
read_get(x) = x * get(getdict, :a, 0.0)
read_tupref(x) = x * tupref[][1][end]
read_vector(x) = x * vector[end]
# A whole-aggregate consumer (`evalpoly`) reads the inner values of a replaced tuple.
const tup = Ref((3.0, 4.0))
mutable struct TupBox
    t::Tuple{Float64,Float64}
end
const tupbox = TupBox((3.0, 4.0))
const tupvec = [(3.0, 4.0)]
read_evalpoly_ref(x) = evalpoly(x, tup[])
read_evalpoly_box(x) = evalpoly(x, tupbox.t)
read_evalpoly_vec(x) = evalpoly(x, tupvec[1])
const evalpoly_readers = (read_evalpoly_ref, read_evalpoly_box, read_evalpoly_vec)
# Element-wise arrays have a plain tangent array that growth outside AD leaves short.
const vv = [[3.0]]
const anyvec = Any[3.0]
read_vv_end(x) = x * vv[end][1]
read_vv_sum(x) = x * sum(d[1] for d in vv)
read_anyvec_end(x) = x * anyvec[end]
read_anyvec_sum(x) = x * sum(anyvec)
const iddict = IdDict{Symbol,Float64}(:a => 3.0)
read_iddict_new(x) = x * get(iddict, :z, 1.0)
end

struct RecAny
    x::Any
end

module FwdReviewFollowup
mutable struct Holder
    a::Vector{Float64}
    b::Float64
end
const global_array = [1.0, 2.0]
read_global(h) = (h.a=global_array; h.a[end] * h.b)
const jacobian_array = [1.0, 2.0]
read_global_output(x) = jacobian_array
end

# GC preservation: a forward rule must keep BOTH the primal and the partials storage alive for
# the duration of a `GC.@preserve` scope. `gc_preserve_storage` reaches the backing `Memory` so
# collection can be observed without dereferencing a possibly stale pointer.
const GC_PRESERVE_OWNERS = Ref((WeakRef(nothing), WeakRef(nothing)))
gc_preserve_storage(x) = @static VERSION >= v"1.11-" ? x.ref.mem : x
# The partials of an `NDualArray` live in one block whose parent vector owns the storage.
function gc_preserve_partials(v)
    gc_preserve_storage(getfield(getfield(v, :partials_block), :parent))
end
function gc_preserve_observe(x)
    GC_PRESERVE_OWNERS[] = (
        WeakRef(gc_preserve_storage(x)), WeakRef(gc_preserve_storage(x))
    )
    return x
end
Mooncake.@is_primitive DefaultCtx ForwardMode Tuple{
    typeof(gc_preserve_observe),Vector{Float64}
}
function Mooncake.frule!!(::Lifted{typeof(gc_preserve_observe),Nw}, x::Lifted) where {Nw}
    GC_PRESERVE_OWNERS[] = (
        WeakRef(gc_preserve_storage(primal(x))), WeakRef(gc_preserve_partials(tangent(x)))
    )
    return x
end
@noinline function gc_preserve_alive(::Ptr{Float64})
    GC.gc(true)
    return map(w -> w.value !== nothing, GC_PRESERVE_OWNERS[])
end
Mooncake.@zero_derivative DefaultCtx Tuple{typeof(gc_preserve_alive),Ptr{Float64}}
function gc_preserve_probe(x)
    a = gc_preserve_observe(copy(x))
    return GC.@preserve a gc_preserve_alive(pointer(a))
end

@testset "s2s_forward_mode_ad" begin
    @testset "rule return type after normalisation (#1327)" begin
        for x in (1.0f0, 1.0), debug_mode in (false, true)
            rule = Mooncake.build_frule(refined_fwd, x; debug_mode)
            result = rule(
                Mooncake.zero_lifted(Val(1), refined_fwd), Mooncake.lift(x, one(x))
            )
            @test primal(result) === x
            @test tangent(result, 1) === one(x)
        end
    end

    @testset "GC preservation of primal and tangent storage (issue #1303)" begin
        # Observe collection without dereferencing a potentially stale pointer.
        @test gc_preserve_probe([1.0, 2.0]) == (true, true)
        rule = Mooncake.build_frule(gc_preserve_probe, [1.0, 2.0])
        for _ in 1:2
            result = rule(
                Mooncake.zero_lifted(Val(1), gc_preserve_probe),
                Mooncake.lift([1.0, 2.0], [3.0, 4.0]),
            )
            @test primal(result) == (true, true)
            GC.gc(true)
            @test all(w -> w.value === nothing, GC_PRESERVE_OWNERS[])
        end
    end

    test_cases = collect(enumerate(TestResources.generate_test_functions()))
    @testset "$n - $(_typeof((fx)))" for (n, (int_only, pf, opts, fx...)) in test_cases
        @info "$n: $(_typeof(fx))"
        TestUtils._case_skip_forward(opts) && continue
        rng = Xoshiro(123546)
        mode = ForwardMode
        skip_chunked = TestUtils._case_skip_chunked(opts)
        TestUtils.test_rule(
            rng,
            fx...;
            perf_flag=pf,
            interface_only=int_only,
            is_primitive=false,
            mode,
            skip_chunked,
            throws=TestUtils._case_throws(opts),
            primal_throws=TestUtils._case_primal_throws(opts),
        )
    end

    @testset "integration testing for invalid global ref errors" begin
        @static if VERSION > v"1.12-"
            @test_throws(
                Mooncake.UnhandledLanguageFeatureException,
                Mooncake.build_frule(Mooncake.TestResources.non_const_global_ref, 5.0)
            )
        end
    end

    @testset "argument aliasing a differentiable global is refused" begin
        G = FwdAliasGlobals.alias_vector
        @testset "$f" for f in
                          (FwdAliasGlobals.alias_read_only, FwdAliasGlobals.alias_scalar)
            cache = Mooncake.prepare_derivative_cache(f, G)
            @test_throws ArgumentError Mooncake.value_and_derivative!!(
                cache, (f, Mooncake.zero_tangent(f)), (G, [1.0, 1.0])
            )
            # The same global read with an unaliased argument is supported and unaffected.
            y = [3.0, 4.0]
            c2 = Mooncake.prepare_derivative_cache(f, y)
            @test Mooncake.value_and_derivative!!(
                c2, (f, Mooncake.zero_tangent(f)), (y, [1.0, 1.0])
            )[2] ≈ (f === FwdAliasGlobals.alias_read_only ? sum(G) : G[1])
        end
    end

    @testset "globals changed between forward calls" begin
        # The registry runs each case at its initial state; this regression needs a mutation
        # between construction and execution of the SAME rule/cache, outside the AD call.
        G = FwdMutableGlobals
        R = FwdReplacedGlobals
        cases = (
            (G.read_params, v -> G.params.scale = v),
            (G.read_array, v -> G.array[1] = v),
            (G.read_anyarray, v -> G.anyarray[1] = v),
            (G.read_ref, v -> G.ref[] = v),
            (G.read_anyref, v -> G.anyref[] = v),
            (G.read_dict, v -> G.dict[:scale] = v),
            (G.read_nested, v -> G.nested.params.scale = v),
            (G.read_immutable, _ -> nothing),
            (G.read_complex, v -> G.complex_array[1] = complex(v / 2, v / 2)),
            (R.read_ref, v -> R.ref[] = [v]),
            (R.read_indirect_ref, v -> R.ref[] = [v]),
            (R.read_ref, v -> R.ref[] = [v - 1, v]),
            (R.read_box, v -> R.box.array = [v]),
            (R.read_index, v -> R.box.array = [v]),
            (R.read_unionref, v -> R.unionref[] = v == 3.0 ? nothing : v),
            (R.read_unionbox, v -> R.unionbox.array = v == 3.0 ? nothing : [v]),
            (R.read_any, v -> R.anyarray[1] = [v]),
            # Built over a Float64 scalar; later reads see another type or an array.
            (R.read_any, v -> R.anyarray[1] = v == 3.0 ? v : Float32(v)),
            (R.read_any, v -> R.anyarray[1] = v == 3.0 ? v : [v]),
            (R.read_dict, v -> R.dict[:a] = [v]),
            (R.read_get, v -> R.getdict[:a] = v),
            (R.read_tupref, v -> R.tupref[] = ([v],)),
            (R.read_evalpoly_ref, v -> R.tup[] = (v, v)),
            (R.read_evalpoly_box, v -> R.tupbox.t = (v, v)),
            (R.read_evalpoly_vec, v -> R.tupvec[1] = (v, v)),
            # The key is absent at build time; its tangent entry is created on first read.
            (R.read_iddict_new, v -> v == 3.0 ? delete!(R.iddict, :z) : (R.iddict[:z] = v)),
        )
        for (f, change!) in cases, (prepared, width) in ((false, 1), (false, 8), (true, 1))
            change!(3.0)
            try
                rule = if prepared
                    Mooncake.prepare_derivative_cache(f, 2.0)
                else
                    Mooncake.build_frule(f, 2.0; chunk_size=width)
                end
                for scale in (7.0, 4.0)
                    change!(scale)
                    expected = f === G.read_immutable ? 3.0 : scale
                    # evalpoly(2, (c, c)) = 3c with derivative c; the others read 2c with c.
                    value = f in R.evalpoly_readers ? 3expected : 2expected
                    if width == 1
                        @test Mooncake.value_and_derivative!!(
                            rule, (f, NoTangent()), (2.0, 1.0)
                        ) == (value, expected)
                    else
                        out = rule(
                            Mooncake.zero_lifted(Val(width), f),
                            Lifted{Float64,width}(
                                2.0, Mooncake.Nfwd.NDual(2.0, ntuple(_ -> 1.0, width))
                            ),
                        )
                        @test primal(out) == value
                        @test all(k -> tangent(out, k) == expected, 1:width)
                    end
                end
            finally
                change!(3.0)
            end
        end
        # A global array resized in place keeps its identity but outgrows the partials block
        # built with the rule, so reading it is refused rather than reading past the block.
        for f in (
                R.read_vector,
                R.read_sum,
                R.read_abs2,
                R.read_mapreduce,
                R.read_bcast,
                R.read_dot,
                R.read_copy,
                R.read_sort,
                R.read_complex,
                R.read_unsafe_copy,
            ),
            (prepared, width) in ((false, 1), (false, 8), (true, 1))

            resize!(R.vector, 1)
            R.vector[1] = 3.0
            Mooncake.empty_mooncake_caches!()
            rule = if prepared
                Mooncake.prepare_derivative_cache(f, 2.0)
            else
                Mooncake.build_frule(f, 2.0; chunk_size=width)
            end
            push!(R.vector, 7.0)
            try
                @test_throws ArgumentError (
                    if prepared
                        Mooncake.value_and_derivative!!(rule, (f, NoTangent()), (2.0, 1.0))
                    else
                        rule(
                            Mooncake.zero_lifted(Val(width), f),
                            Lifted{Float64,width}(
                                2.0, Mooncake.Nfwd.NDual(2.0, ntuple(_ -> 1.0, width))
                            ),
                        )
                    end
                )
            finally
                resize!(R.vector, 1)
            end
        end
        # An element-wise array's tangent array is not resized by growth outside AD either.
        for (f, a, grown) in (
                (R.read_vv_end, R.vv, [7.0]),
                (R.read_vv_sum, R.vv, [7.0]),
                (R.read_anyvec_end, R.anyvec, 7.0),
                (R.read_anyvec_sum, R.anyvec, 7.0),
            ),
            (prepared, width) in ((false, 1), (false, 8), (true, 1))

            resize!(a, 1)
            Mooncake.empty_mooncake_caches!()
            rule = if prepared
                Mooncake.prepare_derivative_cache(f, 2.0)
            else
                Mooncake.build_frule(f, 2.0; chunk_size=width)
            end
            push!(a, grown)
            try
                @test_throws ArgumentError (
                    if prepared
                        Mooncake.value_and_derivative!!(rule, (f, NoTangent()), (2.0, 1.0))
                    else
                        rule(
                            Mooncake.zero_lifted(Val(width), f),
                            Lifted{Float64,width}(
                                2.0, Mooncake.Nfwd.NDual(2.0, ntuple(_ -> 1.0, width))
                            ),
                        )
                    end
                )
            finally
                resize!(a, 1)
            end
        end
        for f in (
                R.read_vector,
                R.read_sum,
                R.read_abs2,
                R.read_mapreduce,
                R.read_bcast,
                R.read_dot,
                R.read_copy,
                R.read_sort,
                R.read_complex,
                R.read_unsafe_copy,
            ),
            width in (1, 8)

            resize!(R.vector, 2)
            R.vector .= (3.0, 7.0)
            Mooncake.empty_mooncake_caches!()
            rule = Mooncake.build_frule(f, 2.0; chunk_size=width)
            resize!(R.vector, 1)
            out = rule(
                Mooncake.zero_lifted(Val(width), f),
                Lifted{Float64,width}(
                    2.0, Mooncake.Nfwd.NDual(2.0, ntuple(_ -> 1.0, width))
                ),
            )
            expected = f in (R.read_abs2, R.read_mapreduce, R.read_dot) ? 9.0 : 3.0
            @test primal(out) == 2expected
            @test all(k -> tangent(out, k) == expected, 1:width)
        end
        # Rebuilding by integer field index must write back, making subsequent reads free.
        R.box.array = [3.0]
        rule = Mooncake.build_frule(R.read_index, 2.0)
        fs, xs = Mooncake.zero_lifted(Val(1), R.read_index), Mooncake.lift(2.0, 1.0)
        rule(fs, xs)
        before = TestUtils.count_allocs(rule, fs, xs)
        R.box.array = [7.0]
        rule(fs, xs)
        @test TestUtils.count_allocs(rule, fs, xs) == before
        R.box.array = [3.0]
    end

    # A tuple element read refreshes its leaves and must keep the canonical V type, whose
    # element for an abstract field is `Any`: a V typed from its values fails the slot's
    # typeassert. Needs a struct with an abstract field, which the registries do not carry.
    @testset "refreshed aggregate V keeps its canonical type" begin
        f = v -> v[1][1].x * 2.0
        TestUtils.test_rule(
            StableRNG(1), f, [(RecAny(3.0),)]; is_primitive=false, mode=ForwardMode
        )
    end

    # A caller-built slot whose primal shrank after lifting: `_partials_block` hands the rule a
    # shape-matching copy, so an in-place writer must write back into the slot's block. Only a
    # `throws` registry row accepts a ready-made slot, hence a bespoke correct-value case.
    @testset "in-place BLAS writes reach a shrunk slot's block" begin
        scal = (a, x) -> (LinearAlgebra.BLAS.scal!(a, x); x[1])
        rule = Mooncake.build_frule(scal, 2.0, [3.0, 5.0])
        lx = Mooncake.lift([3.0, 5.0], [1.0, 1.0])
        resize!(primal(lx), 1)
        out = rule(Mooncake.zero_lifted(Val(1), scal), Mooncake.lift(2.0, 1.0), lx)
        @test (primal(out), tangent(out, 1)) == (6.0, 5.0)
    end

    @testset "stale forward storage is refused" begin
        G = FwdReviewFollowup
        empty!(G.global_array)
        append!(G.global_array, (1.0, 2.0))
        h = G.Holder([0.0], 3.0)
        rule = Mooncake.build_frule(G.read_global, h)
        push!(G.global_array, 5.0)
        @test_throws ArgumentError Mooncake.value_and_derivative!!(
            rule,
            (G.read_global, Mooncake.NoTangent()),
            (G.Holder([0.0], 3.0), Mooncake.MutableTangent((a=[0.0], b=1.0))),
        )
        empty!(G.global_array)
        append!(G.global_array, (1.0, 2.0))

        x = [1.0, 2.0]
        lx = Mooncake.zero_lifted(Val(8), x)
        stale_f(z) = z[3] * 2.0
        stale_rule = Mooncake.build_frule(stale_f, x; chunk_size=8)
        push!(x, 5.0)
        @test_throws ArgumentError Mooncake.value_and_derivative!!(
            stale_rule, Mooncake.zero_lifted(Val(8), stale_f), lx
        )

        a = [1.0, 2.0]
        f = z -> (push!(a, 3.0); z[3] * z[1])
        closure_rule = Mooncake.build_frule(f, a; chunk_size=8)
        @test_throws ArgumentError Mooncake.value_and_derivative!!(
            closure_rule, Mooncake.zero_lifted(Val(8), f), Mooncake.zero_lifted(Val(8), a)
        )

        empty!(G.jacobian_array)
        append!(G.jacobian_array, (1.0, 2.0))
        cache = Mooncake.prepare_derivative_cache(G.read_global_output, [1.0, 2.0])
        Mooncake.value_and_jacobian!!(cache, G.read_global_output, [1.0, 2.0])
        push!(G.jacobian_array, 3.0)
        @test_throws ArgumentError Mooncake.value_and_jacobian!!(
            cache, G.read_global_output, [1.0, 2.0]
        )
        empty!(G.jacobian_array)
        append!(G.jacobian_array, (1.0, 2.0))
    end

    @testset "capture in ReturnNode regression test" begin
        struct RegTestStruct
            x::Vector{Float64}
            RegTestStruct() = new()
        end
        f(x) = RegTestStruct()
        TestUtils.test_rule(
            StableRNG(123), f, 1.0; perf_flag=:none, is_primitive=false, mode=ForwardMode
        )
    end

    # Without the fix the lazy path throws a `convert` MethodError in _build_rule!; both
    # paths must return the build-world result (Float32), not the post-advance (Float64).
    @testset "stale rule build-world after world advance (issue #1218)" begin
        lazy = Mooncake.build_frule(stale_fwd_lazy, 1.5)
        dyn = Mooncake.build_frule(stale_fwd_dyn, 1.5)
        @eval stale_fwd_inner(x::Float64) = x * 2.0  # advance world; tightens callee's type
        lazy_out = Base.invokelatest(
            lazy, Mooncake.zero_lifted(Val(1), stale_fwd_lazy), Mooncake.lift(1.5, 1.0)
        )
        dyn_out = Base.invokelatest(
            dyn, Mooncake.zero_lifted(Val(1), stale_fwd_dyn), Mooncake.lift(1.5, 1.0)
        )
        @test Mooncake.primal(lazy_out) === 3.0f0
        @test Mooncake.primal(dyn_out) === 3.0f0
    end

    # Separate builds must not share mutable DynamicFRule caches (threads / nested AD).
    @testset "cache-hit returns an independent rule copy" begin
        interp = Mooncake.MooncakeInterpreter(ForwardMode)
        sig = Tuple{typeof(fwd_cache_dyn),Float64}
        r1 = Mooncake.build_frule(interp, sig; skip_world_age_check=true)
        r2 = Mooncake.build_frule(interp, sig; skip_world_age_check=true)  # cache HIT
        dyns1 = filter(c -> c isa Mooncake.DynamicFRule, collect(r1.fwd_oc.oc.captures))
        dyns2 = filter(c -> c isa Mooncake.DynamicFRule, collect(r2.fwd_oc.oc.captures))
        # Julia 1.10 resolves the callee without a DynamicFRule capture. On 1.11+,
        # only() also detects regressions that lose the expected capture.
        @static if VERSION >= v"1.11-"
            dyn1 = only(dyns1)
            dyn2 = only(dyns2)
            @test dyn1 !== dyn2
            @test dyn1.cache !== dyn2.cache
        end

        f = FwdAliasGlobals.comparison_read
        for W in (1, 8), _ in 1:2
            rule = Mooncake.build_frule(
                interp, Tuple{typeof(f),Float64}; chunk_size=W, skip_world_age_check=true
            )
            # Global comparison storage must not retain a second, derivative-sized array.
            @test Base.summarysize((
                rule.fwd_oc.oc.captures, FwdAliasGlobals.comparison_vector
            )) < sizeof(FwdAliasGlobals.comparison_vector) + 1024
            y = rule(
                Mooncake.zero_lifted(Val(W), f),
                Mooncake.Lifted{Float64,W}(
                    2.0, Mooncake.Nfwd.NDual(2.0, ntuple(_ -> 1.0, W))
                ),
            )
            @test primal(y) == 2.0
            @test all(i -> tangent(y, i) == 1.0, 1:W)
        end
    end
end;

@testset "a type-observing branch takes the primal's side" begin
    # Type queries must observe the primal: e.g. sizeof(NDual{Float64,1}) == 16,
    # not Float64's 8, which would change both the branch's value and derivative.
    sz(x) = sizeof(x) == 8 ? x * x : x * x * x
    ty(x) = typeof(x) === Float64 ? x * x : x * x * x
    nf(x) = nfields(x) == 0 ? x * x : x * x * x
    szt(t) = sizeof(t[1]) == 8 ? t[1] * t[1] : t[1] * t[1] * t[1]
    for (f, x, grad) in
        ((sz, 2.0, 4.0), (ty, 2.0, 4.0), (nf, 2.0, 4.0), (szt, (2.0, 5.0), (4.0, 0.0)))
        v, g = Mooncake.value_and_gradient!!(Mooncake.prepare_derivative_cache(f, x), f, x)
        @test (v, g[2]) == (4.0, grad)
    end
end
