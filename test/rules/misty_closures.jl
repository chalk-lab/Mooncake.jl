mc_foo(x) = 2x

run_misty_closure(mc::Mooncake.MistyClosure, x::Float64) = mc(x)

struct Foo
    y::Float64
end

(f::Foo)(x) = getfield(f, 1) + x

struct CapturedVector
    v::Vector{Float64}
end
(f::CapturedVector)(x) = (v=getfield(f, 1); v[1] += x; v[1] * x)

# Test cases for second derivative computation.
quadratic(x) = x^3
@noinline lazy_cubic(x) = x^3
lazy_outer(x) = lazy_cubic(x) + x
function low_level_gradient(rrule, f, x::Float64)
    _, pb!! = rrule(zero_fcodual(f), zero_fcodual(x))
    return pb!!(1.0)[2]
end

@testset "misty_closures" begin

    # Construct a sample MistyClosure.
    ir = Base.code_ircode_by_type(Tuple{typeof(mc_foo),Float64})[1][1]
    ir.argtypes[1] = Any
    mc = Mooncake.MistyClosure(ir)

    @testset "tangent interface etc" begin
        rng = StableRNG(123456)
        TestUtils.test_tangent_interface(rng, mc)
        TestUtils.test_rule_and_type_interactions(rng, mc)
    end

    TestUtils.test_rule(
        StableRNG(123),
        mc,
        5.0;
        interface_only=false,
        is_primitive=true,
        perf_flag=:none,
        unsafe_perturb=true,
        mode=ForwardMode,
    )
    TestUtils.test_rule(
        StableRNG(123),
        run_misty_closure,
        mc,
        5.0;
        interface_only=false,
        is_primitive=false,
        perf_flag=:none,
        unsafe_perturb=true,
        mode=ForwardMode,
    )

    # Construct a MistyClosure which accesses its captures. We achieve this by collecting
    # the IR associated to a callable type, and manipulating the types of various fields to
    # ensure that the MistyClosure produced using its `IRCode` is valid.
    ir = Base.code_ircode_by_type(Tuple{Foo,Float64})[1][1]
    ir.argtypes[1] = Tuple{Float64}
    mc2 = Mooncake.MistyClosure(ir, 5.0)
    @test mc2(4.0) == 9.0
    @test Mooncake._copy_to_output!!(mc2, mc2) === mc2
    @test Mooncake._copy_to_output!!(mc2.oc, mc2.oc) === mc2.oc
    cache = Mooncake.prepare_derivative_cache(run_misty_closure, mc2, 4.0)
    @test Mooncake.value_and_derivative!!(
        cache, (run_misty_closure, NoTangent()), (mc2, zero_tangent(mc2)), (4.0, 1.0)
    ) == (9.0, 1.0)
    t = zero_tangent(mc2)
    @test Mooncake.tangent_to_friendly!!(mc2, t) === t
    @test Mooncake.tangent_to_friendly!!(mc2.oc, zero_tangent(mc2.oc)).captures == (0.0,)
    for N in (1, 2)
        seed = Mooncake.basis_lifted!!(Mooncake.zero_lifted(Val(N), mc2), ntuple(_ -> 1, N))
        out = Mooncake.frule!!(seed, Mooncake.zero_lifted(Val(N), 4.0))
        @test primal(out) == 9.0
        @test all(k -> tangent(out, k) == 1.0, 1:N)
    end
    TestUtils.test_rule(
        StableRNG(123),
        mc2,
        4.0;
        interface_only=false,
        is_primitive=true,
        perf_flag=:none,
        unsafe_perturb=true,
        mode=ForwardMode,
    )

    ir = Base.code_ircode_by_type(Tuple{CapturedVector,Float64})[1][1]
    ir.argtypes[1] = Tuple{Vector{Float64}}
    v = [1.0]
    stateful = Mooncake.MistyClosure(ir, v)
    for f in (stateful, stateful.oc)
        saved = Mooncake._copy_output(f)
        v[1] = 3.0
        Mooncake._copy_to_output!!(f, saved)
        @test v == [1.0]
    end
    for N in (1, 8)
        cache = Mooncake.prepare_derivative_cache(
            run_misty_closure, stateful, 2.0; config=Mooncake.Config(chunk_size=N)
        )
        for _ in 1:3
            @test Mooncake.value_and_derivative!!(
                cache,
                (run_misty_closure, NoTangent()),
                (stateful, zero_tangent(stateful)),
                (2.0, 1.0),
            ) == (6.0, 5.0)
            @test v == [1.0]
        end
    end

    # Construct a callable which performs reverse-mode, and apply forwards-mode over it.
    rule = Mooncake.build_rrule(Tuple{typeof(quadratic),Float64})
    # `quadratic` is `x^3`, so the gradient is `3x^2 = 75` at `x = 5` and its own derivative
    # `6x = 30`. The direction is pinned through the `CoDual` channel and the references are
    # given outright: differentiating a compiled reverse rule is not something finite
    # differences can perturb through.
    TestUtils.test_rule(
        StableRNG(123),
        low_level_gradient,
        rule,
        quadratic,
        CoDual(5.0, 1.0);
        is_primitive=false,
        perf_flag=:none,
        unsafe_perturb=true,
        mode=ForwardMode,
        oracle=(value=75.0, deriv=30.0),
    )

    @testset "compiled lazy capture snapshots" begin
        rule = Mooncake.build_rrule(Tuple{typeof(lazy_outer),Float64})
        lazy = only(filter(x -> x isa Mooncake.LazyDerivedRule, rule.fwds_oc.oc.captures))
        for compiled in (false, true)
            compiled && lazy_cubic(5.0)
            cache = Mooncake.prepare_derivative_cache(
                low_level_gradient, rule, lazy_outer, 5.0
            )
            lazy_cubic(5.0)
            mi_cache = lazy.mi.cache
            for _ in 1:3
                @test Mooncake.value_and_derivative!!(
                    cache,
                    (low_level_gradient, NoTangent()),
                    (rule, zero_tangent(rule)),
                    (lazy_outer, NoTangent()),
                    (5.0, 1.0),
                ) == (76.0, 30.0)
                @test isdefined(lazy.mi, :cache) && lazy.mi.cache === mi_cache
            end
        end
    end

    # Reverse over a `MistyClosure` evaluation is not supported and must say so.
    TestUtils.test_rule(
        StableRNG(123),
        low_level_gradient,
        rule,
        quadratic,
        5.0;
        throws=ArgumentError,
        mode=ReverseMode,
        is_primitive=false,
    )

    # Test for world-age mismatch fix in _dual_mc (Julia 1.12+ only).
    # See: https://github.com/chalk-lab/Mooncake.jl/issues/916
    # When methods are defined between IR creation and MistyClosure creation,
    # oc.world exceeds ir.valid_worlds.max_world. The fix uses max_world.
    # See also: test/ext/differentiation_interface for an integration test.
    @static if VERSION > v"1.12-"
        @testset "world age mismatch fix (#916)" begin
            ir_test = Base.code_ircode_by_type(Tuple{typeof(mc_foo),Float64})[1][1]
            ir_test.argtypes[1] = Any

            # Advance world age by defining dummy methods
            for i in 1:100
                @eval $(Symbol("_mooncake_test_dummy_", i))() = $i
            end

            mc_test = Mooncake.MistyClosure(ir_test)
            @test mc_test.oc.world > mc_test.ir[].valid_worlds.max_world

            t = zero_tangent(mc_test)
            @test t isa Mooncake.MistyClosureTangent
        end
    end
end
