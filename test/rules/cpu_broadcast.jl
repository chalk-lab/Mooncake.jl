using Base.Broadcast: Broadcasted, DefaultArrayStyle, materialize

# The rule is declared in `DefaultCtx` only, so a `MinimalCtx` interpreter still derives the
# broadcast loop. Each case checks that both give the same value and gradient, `NaN`s in the
# same places included.
function _cpu_bcast_grads(C, f, x)
    rule = Mooncake.build_rrule(
        Mooncake.MooncakeInterpreter(C, ReverseMode), _typeof((f, x))
    )
    v, (_, g) = value_and_gradient!!(rule, f, x)
    return v, g
end
_cpu_bcast_same(a, b) = typeof(a) == typeof(b) && all(map(isequal, a, b) .| (a .≈ b))

# A broadcast style whose `materialize` some other code declares primitive.
struct _OtherBcastStyle <: Broadcast.AbstractArrayStyle{1} end
_OtherBcastStyle(::Val{1}) = _OtherBcastStyle()
Mooncake.@is_primitive(
    Mooncake.MinimalCtx, Tuple{typeof(materialize),<:Broadcasted{_OtherBcastStyle}}
)

@testset "cpu_broadcast" begin
    rng = StableRNG(123)
    x0 = [0.0, -0.0, 1e-300, 2.5, -3.0, 0.0]
    xs = Float32[0, 9.5, 20, -20, 50, -1e-30]
    x3 = randn(rng, 4, 3, 5)
    bmid = randn(rng, 1, 3, 1)
    b = randn(rng, 4)
    xf = randn(rng, Float32, 5, 4)
    w = randn(rng, 5, 4)
    cases = Any[
        (x -> sum(x .+ b), randn(rng, 4, 3)),
        (b -> sum(tanh.(x3[:, :, 1] .* 2 .+ b)), b),
        (x -> sum(abs2.(x .+ bmid)), x3),
        (b -> sum(abs2.(x3 .+ b)), bmid),
        (s -> sum(sin.(s .* xf)), 0.3),
        (x -> sum(sin.(0.3 .* x)), xf),
        (x -> sum(x .* w), xf),
        (x -> sum(x .^ 3), x0),
        (x -> sum(x .^ -1), x0),
        (x -> sum(x .* x .+ x), x0),
        (x -> sum(ifelse.(x .> 0, 1 ./ x, 0.0)), x0),
        (x -> sum(x ./ x), x0),
        (x -> sum(0.0 .* (1 ./ x)), x0),
        (x -> sum(max.(x, 0.0)), x0),
        (x -> sum(abs.(x) .* x), x0),
        (x -> sum(sqrt.(x .* x)), x0),
        (x -> sum(abs.(x) .^ 0.5), x0),
        (x -> sum(log.(abs.(x))), x0),
        (x -> sum(mod.(x, 1.0)), [0.0, 1.0, 2.0, 0.5]),
        (x -> sum(x .* Inf), [1.0, 0.0, -2.0]),
        (x -> sum(exp.(x)), [1.0, 800.0]),
        (x -> sum(tanh.(x)), xs),
        (x -> sum(x .* [1, 2, 3, 4, 5, 6]), x0),
        (x -> sum(sin.(x)), Float64[]),
    ]
    @testset "$(i): same as the derived rule" for (i, (f, x)) in enumerate(cases)
        v_new, g_new = _cpu_bcast_grads(DefaultCtx, f, x)
        v_old, g_old = _cpu_bcast_grads(Mooncake.MinimalCtx, f, x)
        @test isequal(v_new, v_old) || v_new ≈ v_old
        @test _cpu_bcast_same(g_new, g_old)
    end

    sig(bc) = Tuple{typeof(materialize),typeof(bc)}
    claimed(bc) = Mooncake.is_primitive(
        DefaultCtx, ReverseMode, sig(bc), Base.get_world_counter()
    )
    @testset "which broadcasts are claimed" begin
        @test claimed(Base.broadcasted(+, randn(3), randn(3)))
        @test claimed(Base.broadcasted(Base.literal_pow, ^, randn(3), Val(3)))
        @test !claimed(Base.broadcasted(sqrt, randn(3)))  # reaches Nfwd's strong-zero guard
        @test !claimed(Base.broadcasted(>, randn(3), 0))  # Bool result
        @test !claimed(Base.broadcasted(Float32, randn(3)))  # `Type` as function
    end
    @testset "other materialize primitives are not shadowed" begin
        bc = Broadcasted{_OtherBcastStyle}(+, (randn(3),), nothing)
        @test claimed(bc)
    end

    @testset "test_rule" begin
        for bc in Any[
            Base.broadcasted(+, randn(rng, 4, 3), randn(rng, 4)),
            Base.broadcasted(sin, Base.broadcasted(*, 0.7, randn(rng, 4, 3))),
            Base.broadcasted(*, randn(rng, Float32, 3, 2), randn(rng, Float32, 3, 2)),
        ]
            test_rule(
                rng,
                materialize,
                bc;
                is_primitive=true,
                perf_flag=:none,
                mode=ReverseMode,
                unsafe_perturb=true,
                rrule=Mooncake.CPUBroadcastRule(),
            )
        end
    end
end
