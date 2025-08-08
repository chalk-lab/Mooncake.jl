mc_foo(x) = 2x

run_misty_closure(mc::Mooncake.MistyClosure, x::Float64) = mc(x)

struct Foo
    y::Float64
end

(f::Foo)(x) = getfield(f, 1) + x

# Test cases for second derivative computation.
quadratic(x) = x^3
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
        # Do not run the `test_rule_and_type_interactions` test suite for
        # `MistyClosure`s as we do not implement rules for `getfield` / `_new_`.
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

    # Construct a callable which performs reverse-mode, and apply forwards-mode over it.
    rule = Mooncake.build_rrule(Tuple{typeof(quadratic),Float64})
    TestUtils.test_rule(
        StableRNG(123),
        low_level_gradient,
        rule,
        quadratic,
        5.0;
        interface_only=true,
        is_primitive=false,
        perf_flag=:none,
        unsafe_perturb=true,
        mode=ForwardMode,
    )

    # Check that the rrule for evaluating `MistyClosure`s errors appropriately.
    args = (low_level_gradient, rule, quadratic, 5.0)
    higher_rule = Mooncake.build_rrule(args...)
    @test_throws(ArgumentError, higher_rule(map(zero_fcodual, args)...))

    # Manually test that this correctly computes the second derivative.
    frule = Mooncake.build_frule(
        Mooncake.get_interpreter(Mooncake.ForwardMode),
        Tuple{typeof(low_level_gradient),typeof(rule),typeof(quadratic),Float64},
    )
    result = frule(
        zero_dual(low_level_gradient),
        zero_dual(rule),
        zero_dual(quadratic),
        Mooncake.Dual(5.0, 1.0),
    )
    @test tangent(result) == 6 * 5.0
end
