a_primitive(x) = sin(x)
non_primitive(x) = sin(x)

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(a_primitive),Float64}

# Regression test: primitive with a @mooncake_overlay that changes the return type.
# Before the fix, NativeInterpreter (overlay-blind) was used for type inference of
# primitive calls, so downstream dispatch compiled against the non-overlaid type and
# produced wrong gradients.
struct _OverlayTypeA end
struct _OverlayTypeB end

_overlay_switch(::_OverlayTypeA) = _OverlayTypeA()
Mooncake.@mooncake_overlay _overlay_switch(::_OverlayTypeA) = _OverlayTypeB()

Mooncake.@is_primitive MinimalCtx ReverseMode Tuple{typeof(_overlay_switch), _OverlayTypeA}
function Mooncake.rrule!!(
    f::CoDual{typeof(_overlay_switch)}, x::CoDual{_OverlayTypeA}
)
    return zero_fcodual(_OverlayTypeB()), NoPullback(f, x)
end

_overlay_use(::_OverlayTypeA, x::Float64) = x
_overlay_use(::_OverlayTypeB, x::Float64) = 2x

Mooncake.@is_primitive MinimalCtx ReverseMode Tuple{typeof(_overlay_use), _OverlayTypeA, Float64}
function Mooncake.rrule!!(
    f::CoDual{typeof(_overlay_use)}, ::CoDual{_OverlayTypeA}, x::CoDual{Float64}
)
    pb(dy) = NoRData(), NoRData(), dy
    return zero_fcodual(Mooncake.primal(x)), pb
end

Mooncake.@is_primitive MinimalCtx ReverseMode Tuple{typeof(_overlay_use), _OverlayTypeB, Float64}
function Mooncake.rrule!!(
    f::CoDual{typeof(_overlay_use)}, ::CoDual{_OverlayTypeB}, x::CoDual{Float64}
)
    pb(dy) = NoRData(), NoRData(), 2dy
    return zero_fcodual(2.0 * Mooncake.primal(x)), pb
end

function _overlay_outer(x::Float64)
    b = _overlay_switch(_OverlayTypeA())
    return _overlay_use(b, x)
end

contains_primitive(x) = @inline a_primitive(x)
contains_non_primitive(x) = @inline non_primitive(x)
contains_primitive_behind_call(x) = @inline contains_primitive(x)
union_split_primitive_call(x::Ref{Union{Float64,Float32}}) = @inline a_primitive(x[])

# Issue #955: if a primitive call's return value is inferred as `Const`,
# the compiler may fold the call away entirely. This makes the primitive 
# invisible to Mooncake,  so its custom `rrule!!` never runs.
fake_grad_955(x, y) = x
Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(fake_grad_955),Any,Any}

function Mooncake.rrule!!(::CoDual{typeof(fake_grad_955)}, x::CoDual, y::CoDual)
    function fake_grad_955_pullback(dy)
        return NoRData(), NoRData(), dy
    end
    return CoDual(x.x, y.dx), fake_grad_955_pullback
end

@testset "abstract_interpretation" begin
    # Check that inlining doesn't / does happen as expected.
    @testset "MooncakeInterpreter" begin
        @testset "non-primitive continues to be inlined away" begin

            # A non-primitive is present in the IR for contains_non_primitive. It is
            # inlined away under usual interpretation, and should also be inlined away
            # when doing AD.
            sig = Tuple{typeof(contains_non_primitive),Float64}

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

        @testset "955 - primitive call not const-folded away" begin
            f = x -> fake_grad_955(1.0, 2x[1] + x[3])
            x = [1.0, 2.0, 3.0]

            cache = Mooncake.prepare_gradient_cache(f, x)
            val, grad = Mooncake.value_and_gradient!!(cache, f, x)

            @test val == 1.0
            @test grad[2] == [2.0, 0.0, 1.0]
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

    @testset "overlay with type-changing return is inferred correctly" begin
        # _overlay_switch has a @mooncake_overlay that returns _OverlayTypeB instead of
        # _OverlayTypeA. _overlay_outer passes the result to _overlay_use, which has
        # separate rrule!!s for each type (gradient 1x vs 2x). The correct gradient is
        # 2.0 iff Mooncake inferred the overlaid return type and dispatched to the TypeB
        # rrule!!. A wrong gradient of 1.0 indicates the overlay was ignored.
        cache = Mooncake.prepare_gradient_cache(_overlay_outer, 3.0)
        val, (_, grad) = Mooncake.value_and_gradient!!(cache, _overlay_outer, 3.0)
        @test val == 6.0
        @test grad == 2.0
    end
end
