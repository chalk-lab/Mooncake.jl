using
    BenchmarkTools,
    DiffRules,
    Distributions,
    FillArrays,
    FunctionWrappers,
    JET,
    LinearAlgebra,
    PDMats,
    Random,
    SpecialFunctions,
    StableRNGs,
    Taped,
    Test,
    Umlaut

using Base: unsafe_load, pointer_from_objref
using Base.Iterators: product
using Core: bitcast
using Core.Intrinsics: pointerref, pointerset
using FunctionWrappers: FunctionWrapper

using Taped:
    IntrinsicsWrappers,
    TestUtils,
    TestResources,
    CoDual,
    to_reverse_mode_ad,
    _wrap_field,
    build_coinstruction,
    const_coinstruction,
    input_primals,
    input_tangents,
    output_primal,
    output_tangent,
    pullback!,
    seed_output_tangent!,
    rrule!!,
    set_tangent!!,
    SSym,
    SInt,
    lgetfield,
    might_be_active,
    rebind,
    build_tangent

using Taped.Umlaut: __new__

using .TestUtils:
    test_rrule!!,
    test_taped_rrule!!,
    has_equal_data,
    AddressMap,
    populate_address_map!,
    populate_address_map,
    test_tangent,
    test_numerical_testing_interface

using .TestResources:
    TypeStableMutableStruct,
    StructFoo,
    MutableFoo

# The integration tests take ages to run, so we split them up. CI sets up two jobs -- the
# "basic" group runs test that, when passed, _ought_ to imply correctness of the entire
# scheme. The "extended" group runs a large battery of tests that should pick up on anything
# that has been missed in the "basic" group. As a rule, if the "basic" group passes, but the
# "extended" group fails, there are clearly new tests that need to be added to the "basic"
# group.
const test_group = get(ENV, "TEST_GROUP", "basic")

sr(n::Int) = StableRNG(n)

@testset "Taped.jl" begin
    if test_group == "basic"
        include("tracing.jl")
        include("acceleration.jl")
        include("tangents.jl")
        include("reverse_mode_ad.jl")
        include("test_utils.jl")
        @testset "rrules" begin
            @info "avoiding_non_differentiable_code"
            include(joinpath("rrules", "avoiding_non_differentiable_code.jl"))
            @info "blas"
            include(joinpath("rrules", "blas.jl"))
            @info "builtins"
            include(joinpath("rrules", "builtins.jl"))
            @info "foreigncall"
            include(joinpath("rrules", "foreigncall.jl"))
            @info "lapack"
            include(joinpath("rrules", "lapack.jl"))
            @info "low_level_maths"
            include(joinpath("rrules", "low_level_maths.jl"))
            @info "misc"
            include(joinpath("rrules", "misc.jl"))
            @info "umlaut_internals_rules"
            include(joinpath("rrules", "umlaut_internals_rules.jl"))
            @info "unrolled_function"
            include(joinpath("rrules", "unrolled_function.jl"))
        end
    elseif test_group == "integration_testing/misc"
        include(joinpath("integration_testing/", "misc.jl"))
        include(joinpath("integration_testing", "battery_tests.jl"))
    elseif test_group == "integration_testing/diff_tests"
        include(joinpath("integration_testing", "diff_tests.jl"))
    elseif test_group == "integration_testing/distributions"
        include(joinpath("integration_testing", "distributions.jl"))
    elseif test_group == "integration_testing/special_functions"
        include(joinpath("integration_testing", "special_functions.jl"))
    elseif test_group == "integration_testing/array"
        include(joinpath("integration_testing", "array.jl"))
    else
        throw(error("test_group=$(test_group) is not recognised"))
    end
end
