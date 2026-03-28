#=
  Two ways to run tests (from Mooncake.jl root):

  1. Interactive — iterate on individual files; TestEnv makes deps like Aqua.jl and JET.jl available.

     One-time setup:
       mkdir -p temp/testenv
       julia --project=temp/testenv -e 'using Pkg; Pkg.add("TestEnv"); Pkg.develop(path=".")'

     Then each session:
       julia --project=temp/testenv -e 'using TestEnv; TestEnv.activate("Mooncake"); include("test/front_matter.jl")'

     Then include individual test files, e.g.: include("test/nfwd/nfwd.jl")
     or include("test/nfwd/nfwdmooncake.jl")

  2. Batch — run a named test group end-to-end via Pkg.test:
       julia --project=. -e 'import Pkg; Pkg.test(; test_args=["Nfwd"])'

     If test_args is omitted, the "basic" group runs (not the full suite).
=#
include("front_matter.jl")

@testset "Mooncake.jl" begin
    if test_group == "basic"
        Aqua.test_all(Mooncake)
        include("utils.jl")
        include(joinpath("tangents", "tangents.jl"))
        include(joinpath("tangents", "fwds_rvs_data.jl"))
        include(joinpath("tangents", "codual.jl"))
        include(joinpath("tangents", "dual.jl"))
        include("debug_mode.jl")
        include("stack.jl")
        @testset "interpreter" begin
            include(joinpath("interpreter", "contexts.jl"))
            include(joinpath("interpreter", "abstract_interpretation.jl"))
            include(joinpath("interpreter", "ir_utils.jl"))
            include(joinpath("interpreter", "bbcode.jl"))
            include(joinpath("interpreter", "ir_normalisation.jl"))
            include(joinpath("interpreter", "zero_like_rdata.jl"))
            include(joinpath("interpreter", "forward_mode.jl"))
            include(joinpath("interpreter", "reverse_mode.jl"))
        end
        include("tools_for_rules.jl")
        include("interface.jl")
        include("config.jl")
        include("developer_tools.jl")
        include("test_utils.jl")
    elseif test_group == "Nfwd"
        include(joinpath("nfwd", "nfwd.jl"))
        include(joinpath("nfwd", "nfwdmooncake.jl"))
    elseif test_group == "rules/array_legacy"
        @static if VERSION < v"1.11.0-rc4"
            include(joinpath("rules", "array_legacy.jl"))
        end
    elseif test_group == "rules/avoiding_non_differentiable_code"
        include(joinpath("rules", "avoiding_non_differentiable_code.jl"))
    elseif test_group == "rules/blas_Float64"
        include(joinpath("rules", "blas_Float64.jl"))
    elseif test_group == "rules/blas_Float32"
        include(joinpath("rules", "blas_Float32.jl"))
    elseif test_group == "rules/blas_ComplexF64"
        include(joinpath("rules", "blas_ComplexF64.jl"))
    elseif test_group == "rules/blas_ComplexF32"
        include(joinpath("rules", "blas_ComplexF32.jl"))
    elseif test_group == "rules/builtins"
        include(joinpath("rules", "builtins.jl"))
    elseif test_group == "rules/complex"
        include(joinpath("rules", "complex.jl"))
    elseif test_group == "rules/fastmath"
        include(joinpath("rules", "fastmath.jl"))
    elseif test_group == "rules/foreigncall"
        include(joinpath("rules", "foreigncall.jl"))
    elseif test_group == "rules/iddict"
        include(joinpath("rules", "iddict.jl"))
    elseif test_group == "rules/lapack"
        include(joinpath("rules", "lapack.jl"))
    elseif test_group == "rules/linear_algebra"
        include(joinpath("rules", "linear_algebra.jl"))
    elseif test_group == "rules/low_level_maths"
        include(joinpath("rules", "low_level_maths.jl"))
    elseif test_group == "rules/misc"
        include(joinpath("rules", "misc.jl"))
    elseif test_group == "rules/misty_closures"
        include(joinpath("rules", "misty_closures.jl"))
    elseif test_group == "rules/new"
        include(joinpath("rules", "new.jl"))
    elseif test_group == "rules/random"
        include(joinpath("rules", "random.jl"))
    elseif test_group == "rules/tasks"
        include(joinpath("rules", "tasks.jl"))
    elseif test_group == "rules/twice_precision"
        include(joinpath("rules", "twice_precision.jl"))
    elseif test_group == "rules/memory"
        @static if VERSION >= v"1.11.0-rc4"
            include(joinpath("rules", "memory.jl"))
        end
    elseif test_group == "rules/performance_patches"
        include(joinpath("rules", "performance_patches.jl"))
    elseif test_group == "rules/dispatch_doctor"
        include(joinpath("rules", "dispatch_doctor.jl"))
    elseif test_group == "rules/high_order_derivative_patches"
        include(joinpath("rules", "high_order_derivative_patches.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
