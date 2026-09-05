a_primitive(x) = sin(x)
non_primitive(x) = sin(x)

Mooncake.@is_primitive DefaultCtx Tuple{typeof(a_primitive),Float64}

# Deliberately different from cos(x), so differentiating the body cannot pass these tests.
function Mooncake.rrule!!(::CoDual{typeof(a_primitive)}, x::CoDual{Float64})
    return Mooncake.zero_fcodual(sin(primal(x))), dy -> (NoRData(), 7dy)
end
function Mooncake.frule!!(::Dual{typeof(a_primitive)}, x::Dual{Float64})
    return Dual(sin(primal(x)), 7tangent(x))
end

contains_primitive(x) = @inline a_primitive(x)
contains_non_primitive(x) = @inline non_primitive(x)
contains_primitive_behind_call(x) = @inline contains_primitive(x)
union_split_primitive_call(x::Ref{Union{Float64,Float32}}) = @inline a_primitive(x[])

invoke_primitive(x) = @inline invoke(a_primitive, Tuple{Float64}, x)
invoke_nondefault(x::Real) = sin(x)
invoke_nondefault(x::Float64) = cos(x)
invoke_nondefault_call(x) = invoke(invoke_nondefault, Tuple{Real}, x)

# Issue #955: if a primitive call's return value is inferred as `Const`,
# the compiler may fold the call away entirely. This makes the primitive 
# invisible to Mooncake,  so its custom `rrule!!` never runs.
fake_grad_955(x, y) = x
# Also exercise a cached constant-return method, not just call-site constant propagation.
fake_grad_955(::Nothing, y) = 1.0
Mooncake.@is_primitive DefaultCtx Tuple{typeof(fake_grad_955),Any,Any}

function Mooncake.rrule!!(::CoDual{typeof(fake_grad_955)}, x::CoDual, y::CoDual)
    function fake_grad_955_pullback(dy)
        return NoRData(), NoRData(), dy
    end
    return CoDual(fake_grad_955(primal(x), primal(y)), y.dx), fake_grad_955_pullback
end

function Mooncake.frule!!(::Dual{typeof(fake_grad_955)}, x::Dual, y::Dual)
    return Dual(fake_grad_955(primal(x), primal(y)), tangent(y))
end

@testset "abstract_interpretation" begin
    # Check that inlining doesn't / does happen as expected.
    @testset "MooncakeInterpreter" begin
        @testset "non-primitive continues to be inlined away" for f in (
            contains_non_primitive, x -> invoke(non_primitive, Tuple{Float64}, x)
        )

            # A non-primitive is present in the IR for contains_non_primitive. It is
            # inlined away under usual interpretation, and should also be inlined away
            # when doing AD.
            sig = Tuple{typeof(f),Float64}

            # Pre-condition: must inline away under usual compilation.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), stmt(usual_ir.stmts))
            @assert stmt(usual_ir.stmts)[invoke_line].args[2] == GlobalRef(Main, :sin)

            # Should continue to inline away under AD compilation.
            interp = Mooncake.MooncakeInterpreter(DefaultCtx, ReverseMode)
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), stmt(ad_ir.stmts))
            @test stmt(ad_ir.stmts)[invoke_line].args[2] == GlobalRef(Main, :sin)
        end
        @testset "primitive is no longer inlined away" begin

            # A primitive is present in the IR for contains_primitive. It is inlined away
            # under usual interpretation, but should not be when doing AD.
            sig = Tuple{typeof(contains_primitive),Float64}

            # Pre-condition: must inline away under usual compilation.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), stmt(usual_ir.stmts))
            @assert stmt(usual_ir.stmts)[invoke_line].args[2] == GlobalRef(Main, :sin)

            # Should not inline away under AD compilation.
            interp = Mooncake.MooncakeInterpreter(DefaultCtx, ReverseMode)
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), stmt(ad_ir.stmts))
            @test stmt(ad_ir.stmts)[invoke_line].args[2] == GlobalRef(Main, :a_primitive)
        end
        @testset "deep primitive is not inlined away" begin

            # A non-primitive is immediately visible in the IR, but this non-primitive is
            # usually inlined away to reveal a primitive. This primitive is _also_ usually
            # inlined away, but should not be when doing AD. This case is not handled if
            # various bits of information are not properly propagated in the compiler.
            sig = Tuple{typeof(contains_primitive_behind_call),Float64}

            # Pre-condition: both functions should be inlined away under usual conditions.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), stmt(usual_ir.stmts))
            @assert stmt(usual_ir.stmts)[invoke_line].args[2] == GlobalRef(Main, :sin)

            # Should not inline away under AD compilation.
            interp = Mooncake.MooncakeInterpreter(DefaultCtx, ReverseMode)
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            invoke_line = findfirst(x -> Meta.isexpr(x, :invoke), stmt(ad_ir.stmts))
            @test stmt(ad_ir.stmts)[invoke_line].args[2] == GlobalRef(Main, :a_primitive)
        end
        @testset "no inline away union splitting" begin

            # In the IR for this signature generated using the standard interpreter, calls
            # to `a_primitive` are inlined away to calls to `sin`. One call site is
            # generated for `Float32`, and another for `Float64` (you should take a look at
            # the IR to see this). However, since `a_primitive` called with a `Float64` is
            # a Mooncake primitive, this should not happen in the IR with a Mooncake
            # interpreter. This case is different from those above in that the type being
            # passed into `a_primitive` is abstract (Union{Float32,Float64}) rather than
            # concrete. In early versions of Mooncake, the implementation of `is_primitive`
            # would incorrect permit this call site to be inlined away.
            Tx = Base.RefValue{Union{Float32,Float64}}
            sig = Tuple{typeof(union_split_primitive_call),Tx}

            # Pre-condition: a invoke sites inline away to reveal calls to `sin`.
            usual_ir = Base.code_ircode_by_type(sig)[1][1]
            for line in findall(x -> Meta.isexpr(x, :invoke), stmt(usual_ir.stmts))
                @assert stmt(usual_ir.stmts)[line].args[2] == GlobalRef(Main, :sin)
            end

            # Should not inline away under AD compilation.
            interp = Mooncake.MooncakeInterpreter(DefaultCtx, ReverseMode)
            ad_ir = Base.code_ircode_by_type(sig; interp)[1][1]
            for line in findall(x -> Meta.isexpr(x, :invoke), stmt(ad_ir.stmts))
                @test stmt(ad_ir.stmts)[line].args[2] != GlobalRef(Main, :sin)
            end
        end

        # Regression test for https://github.com/chalk-lab/Mooncake.jl/issues/238
        @testset "238" begin
            f = Base.Fix1(view, [5.0, 4.0])
            fargs = (Base._mapreduce_dim, f, vcat, Float64[], [1:1, 2:2], :)
            @test Base.code_typed_by_type(typeof(fargs))[1][2] == Vector{Float64}
        end

        @testset "1300 - explicit invoke preserves primitive rules" begin
            for f in (
                contains_primitive,
                invoke_primitive,
                x -> invoke(a_primitive, Tuple{Any}, x),
                x -> invoke_primitive(x),
            )
                cache = Mooncake.prepare_gradient_cache(f, 0.3)
                @test Mooncake.value_and_gradient!!(cache, f, 0.3) ==
                    (sin(0.3), (NoTangent(), 7.0))
                rule = Mooncake.build_frule(f, 0.3)
                y = rule(zero_dual(f), Dual(0.3, 1.0))
                @test primal(y) == sin(0.3)
                @test tangent(y) == 7.0
            end

            # Before a primitive declaration, invoke must keep selecting the Real method.
            f = invoke_nondefault_call
            cache = Mooncake.prepare_gradient_cache(f, 0.3)
            rule = Mooncake.build_frule(f, 0.3)
            @test Mooncake.value_and_gradient!!(cache, f, 0.3)[2][2] ≈ cos(0.3)
            @test tangent(rule(zero_dual(f), Dual(0.3, 1.0))) ≈ cos(0.3)

            @eval Mooncake.@is_primitive DefaultCtx Tuple{typeof(invoke_nondefault),Float64}
            # A rule for ordinary Float64 dispatch cannot describe this explicit Real call.
            for build in (Mooncake.build_frule, Mooncake.build_rrule)
                @test_throws "signature-based primitive rule" build(f, 0.3)
            end
            mi = Mooncake.CC.specialize_method(
                which(invoke_nondefault, (Real,)),
                Tuple{typeof(invoke_nondefault),Float64},
                Core.svec(),
            )
            for (mode, build, lazy) in (
                (ForwardMode, Mooncake.build_frule, Mooncake.LazyFRule),
                (ReverseMode, Mooncake.build_rrule, Mooncake.LazyDerivedRule),
            )
                interp = Mooncake.MooncakeInterpreter(mode)
                @test_throws "signature-based primitive rule" build(interp, mi)
                @test_throws "signature-based primitive rule" lazy(mi, false, interp.world)
            end
            # Already prepared rules remain pinned to their original world.
            @test Mooncake.value_and_gradient!!(cache, f, 0.3)[2][2] ≈ cos(0.3)
        end

        @testset "955/1300 - primitive call not const-folded away" for f in (
            x -> fake_grad_955(1.0, 2x[1] + x[3]),
            x -> invoke(fake_grad_955, Tuple{Float64,Float64}, 1.0, 2x[1] + x[3]),
            x -> invoke(fake_grad_955, Tuple{Nothing,Float64}, nothing, 2x[1] + x[3]),
        )
            x = [1.0, 2.0, 3.0]

            cache = Mooncake.prepare_gradient_cache(f, x)
            val, grad = Mooncake.value_and_gradient!!(cache, f, x)

            @test val == 1.0
            @test grad[2] == [2.0, 0.0, 1.0]
            rule = Mooncake.build_frule(f, x)
            y = rule(zero_dual(f), Dual(x, [1.0, 1.0, 1.0]))
            @test primal(y) == 1.0
            @test tangent(y) == 3.0
        end
    end

    @testset "Config(empty_cache=true)" begin
        f = x -> sin(x[1]) + x[2]^2
        x = [1.0, 2.0]

        # Build up the cache with several functions, then clear it.
        for g in [x -> sum(x .^ 2), x -> prod(x), x -> sum(exp.(x))]
            Mooncake.prepare_gradient_cache(g, randn(10))
        end
        n_before = length(Mooncake.GLOBAL_INTERPRETERS[Mooncake.ReverseMode].oc_cache)
        @test n_before > 0

        cache = Mooncake.prepare_gradient_cache(
            f, x; config=Mooncake.Config(empty_cache=true)
        )
        @test length(Mooncake.GLOBAL_INTERPRETERS[Mooncake.ReverseMode].oc_cache) < n_before

        # AD still correct after clearing.
        val, grad = Mooncake.value_and_gradient!!(cache, f, x)
        @test val ≈ sin(x[1]) + x[2]^2
        @test grad[2] ≈ [cos(x[1]), 2x[2]]
    end
end
