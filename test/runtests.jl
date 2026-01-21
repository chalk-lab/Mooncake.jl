include("front_matter.jl")

@testset "Mooncake.jl" begin
    if test_group == "basic"
        Aqua.test_all(Mooncake)
        include("utils.jl")
        include(joinpath("tangents", "tangents.jl"))
        include(joinpath("tangents", "fwds_rvs_data.jl"))
        include(joinpath("tangents", "codual.jl"))
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
    elseif test_group == "rules/array_legacy"
        include(joinpath("rules", "array_legacy.jl"))
    elseif test_group == "rules/avoiding_non_differentiable_code"
        include(joinpath("rules", "avoiding_non_differentiable_code.jl"))
    elseif test_group == "rules/blas"
        include(joinpath("rules", "blas.jl"))
    elseif test_group == "rules/blas_level_3"
        include(joinpath("rules", "blas_level_3.jl"))
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
