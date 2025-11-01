module IRUtilsGlobalRefs
__x_1 = 5.0
const __x_2 = 5.0
__x_3::Float64 = 5.0
const __x_4::Float64 = 5.0
end

@testset "ir_utils" begin
    @testset "ircode $(typeof(fargs))" for fargs in Any[(sin, 5.0), (cos, 1.0)]
        # Construct a vector of instructions from known function.
        f, args... = fargs
        src, rettype = only(code_typed(f, _typeof(args)))

        # Use Mooncake.ircode to build an `IRCode`.
        argtypes = Any[map(_typeof, fargs)...]
        ir = Mooncake.ircode(src.code, argtypes)

        # Check the validity of the `IRCode`, and that an OpaqueClosure constructed using it
        # gives the same answer as the original function.
        @test length(stmt(ir.stmts)) == length(src.code)
        oc = Mooncake.opaque_closure(rettype, ir)
        @test oc(args...) == f(args...)
    end
    @testset "infer_ir!" begin

        # Generate IR without any types. 
        ir = Mooncake.ircode(
            Any[
                Expr(:call, GlobalRef(Base, :sin), Argument(2)),
                Expr(:call, cos, SSAValue(1)),
                ReturnNode(SSAValue(2)),
            ],
            Any[Tuple{}, Float64],
        )

        # Run inference and check that the types are as expected.
        ir = Mooncake.infer_ir!(ir)
        @test ir.stmts.type[1] == Float64
        @test ir.stmts.type[2] == Float64

        # Check that the ir is runable.
        ir.argtypes[1] = Tuple
        oc = Mooncake.opaque_closure(Float64, ir)
        @test oc(5.0) == cos(sin(5.0))
    end
    @testset "lookup_ir" begin
        tt = Tuple{typeof(sin),Float64}
        @test isa(
            Mooncake.lookup_ir(CC.NativeInterpreter(), tt; optimize_until=nothing)[1],
            CC.IRCode,
        )
    end
    @testset "unhandled_feature" begin
        @test_throws(
            Mooncake.UnhandledLanguageFeatureException, Mooncake.unhandled_feature("foo")
        )
    end
    @testset "replace_uses_with!" begin
        stmt = Expr(:call, sin, SSAValue(1))
        Mooncake.replace_uses_with!(stmt, SSAValue(1), 5.0)
        @test stmt.args[end] == 5.0
    end
    @testset "characeterise_used_ssas" begin
        stmts = Any[
            Expr(:call, sin, Argument(1)),
            Expr(:call, sin, SSAValue(1)),
            Expr(:call, sin, SSAValue(1)),
            ReturnNode(SSAValue(3)),
        ]
        @test Mooncake.characterised_used_ssas(stmts) == [true, false, true, false]
    end
end
