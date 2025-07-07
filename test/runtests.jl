include("front_matter.jl")

@testset "Mooncake.jl" begin
    if test_group == "quality"
        Aqua.test_all(Mooncake)
        @test JuliaFormatter.format(Mooncake; verbose=false, overwrite=false)
    elseif test_group == "basic"
        include("utils.jl")
        include("tangents.jl")
        include("fwds_rvs_data.jl")
        include("codual.jl")
        include("debug_mode.jl")
        include("stack.jl")
        @testset "interpreter" begin
            include(joinpath("interpreter", "contexts.jl"))
            include(joinpath("interpreter", "abstract_interpretation.jl"))
            include(joinpath("interpreter", "ir_utils.jl"))
            include(joinpath("interpreter", "bbcode.jl"))
            include(joinpath("interpreter", "ir_normalisation.jl"))
            include(joinpath("interpreter", "zero_like_rdata.jl"))
            include(joinpath("interpreter", "s2s_reverse_mode_ad.jl"))
        end
        include("tools_for_rules.jl")
        include("interface.jl")
        include("config.jl")
        include("developer_tools.jl")
        include("test_utils.jl")
    elseif test_group == "rrules/array_legacy"
        include(joinpath("rrules", "array_legacy.jl"))
    elseif test_group == "rrules/avoiding_non_differentiable_code"
        include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
    elseif test_group == "rrules/blas"
        include(joinpath("rrules", "blas.jl"))
    elseif test_group == "rrules/builtins"
        include(joinpath("rrules", "builtins.jl"))
    elseif test_group == "rrules/fastmath"
        include(joinpath("rrules", "fastmath.jl"))
    elseif test_group == "rrules/foreigncall"
        include(joinpath("rrules", "foreigncall.jl"))
    elseif test_group == "rrules/functionwrappers"
        include(joinpath("rrules", "function_wrappers.jl"))
    elseif test_group == "rrules/iddict"
        include(joinpath("rrules", "iddict.jl"))
    elseif test_group == "rrules/lapack"
        include(joinpath("rrules", "lapack.jl"))
    elseif test_group == "rrules/linear_algebra"
        include(joinpath("rrules", "linear_algebra.jl"))
    elseif test_group == "rrules/low_level_maths"
        include(joinpath("rrules", "low_level_maths.jl"))
    elseif test_group == "rrules/misc"
        include(joinpath("rrules", "misc.jl"))
    elseif test_group == "rrules/new"
        include(joinpath("rrules", "new.jl"))
    elseif test_group == "rrules/random"
        include(joinpath("rrules", "random.jl"))
    elseif test_group == "rrules/tasks"
        include(joinpath("rrules", "tasks.jl"))
    elseif test_group == "rrules/twice_precision"
        include(joinpath("rrules", "twice_precision.jl"))
    elseif test_group == "rrules/memory"
        @static if VERSION >= v"1.11.0-rc4"
            include(joinpath("rrules", "memory.jl"))
        end
    elseif test_group == "rrules/performance_patches"
        include(joinpath("rrules", "performance_patches.jl"))
    elseif test_group == "rrules/dispatch_doctor"
        include(joinpath("rrules", "dispatch_doctor.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
