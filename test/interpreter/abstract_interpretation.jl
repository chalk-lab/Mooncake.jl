a_primitive(x) = sin(x)
non_primitive(x) = sin(x)

function Mooncake.is_primitive(
    ::Type{DefaultCtx}, ::Type{ReverseMode}, ::Type{<:Tuple{typeof(a_primitive),Any}}
)
    return true
end
function Mooncake.is_primitive(
    ::Type{DefaultCtx}, ::Type{ReverseMode}, ::Type{<:Tuple{typeof(non_primitive),Any}}
)
    return false
end

contains_primitive(x) = @inline a_primitive(x)
contains_non_primitive(x) = @inline non_primitive(x)
contains_primitive_behind_call(x) = @inline contains_primitive(x)

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

        # Regression test for https://github.com/chalk-lab/Mooncake.jl/issues/238
        @testset "238" begin
            f = Base.Fix1(view, [5.0, 4.0])
            fargs = (Base._mapreduce_dim, f, vcat, Float64[], [1:1, 2:2], :)
            @test Base.code_typed_by_type(typeof(fargs))[1][2] == Vector{Float64}
        end
    end
end
