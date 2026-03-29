using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake: Mooncake, TestUtils, Tangent
using DispatchDoctor: allow_unstable, type_instability

# Each test_hook below intercepts a specific TestUtils function to suppress
# DispatchDoctor's type-instability checks where needed. All hooks dispatch on ::typeof(f)
# so that adding new methods to those functions automatically narrows the hook.
TestUtils.test_hook(::Any, ::typeof(TestUtils.test_opt), ::Any...) = nothing
TestUtils.test_hook(::Any, ::typeof(TestUtils.report_opt), tt) = nothing
function TestUtils.test_hook(f, ::typeof(Mooncake.hand_written_rule_test_cases), ::Any...)
    return allow_unstable(f)
end
function TestUtils.test_hook(f, ::typeof(Mooncake.derived_rule_test_cases), ::Any...)
    return allow_unstable(f)
end
function TestUtils.test_hook(f, ::Val{:allow_unstable_hvp_interface_test}, ::Any...)
    return allow_unstable(f)
end
function TestUtils.test_hook(f, ::Val{:allow_unstable_hessian_interface_test}, ::Any...)
    return allow_unstable(f)
end

# Automatically skip instability checks for types which are themselves unstable,
# or which are unreasonably hard to infer.
function allow_unstable_given_unstable_type(f::F, ::Type{T}) where {F,T}
    skip_instability_check(T) ? allow_unstable(f) : f()
end
function skip_instability_check(::Type{T}) where {T}
    type_instability(T) || (
        isstructtype(T) &&
        (fieldcount(T) > 16 || any(skip_instability_check, fieldtypes(T)))
    )
end
function skip_instability_check(::Type{<:Tangent{Tfields}}) where {Tfields}
    skip_instability_check(Tfields)
end
function skip_instability_check(::Type{NT}) where {NT<:NamedTuple}
    true
end
function skip_instability_check(::Type{NT}) where {K,V,NT<:NamedTuple{K,V}}
    skip_instability_check(V)
end

function TestUtils.test_hook(::Any, ::typeof(TestUtils.check_allocs), f, x...)
    allow_unstable_given_unstable_type(typeof(x)) do
        f(x...)
    end
end
function TestUtils.test_hook(::Any, ::typeof(TestUtils.count_allocs), f, x...)
    allow_unstable_given_unstable_type(typeof(x)) do
        f(x...)
        0
    end
end
function TestUtils.test_hook(
    f, ::typeof(TestUtils.test_tangent_interface), ::Any, p; kws...
)
    allow_unstable_given_unstable_type(f, typeof(p))
end
function TestUtils.test_hook(
    f, ::typeof(TestUtils.test_tangent_splitting), ::Any, p; kws...
)
    allow_unstable_given_unstable_type(f, typeof(p))
end
function TestUtils.test_hook(
    f, ::typeof(TestUtils.test_tangent_performance), ::Any, p; kws...
)
    allow_unstable_given_unstable_type(f, typeof(p))
end
function TestUtils.test_hook(f, ::typeof(Mooncake.compute_oc_signature), x...)
    allow_unstable(f)
end

include(joinpath(@__DIR__, "..", "..", "front_matter.jl"))

# julia#61368 workaround: loading DispatchDoctor (above) advances the world counter by
# defining many methods, which on Julia 1.10 would invalidate inner OC CodeInstances and
# trigger a segfault in emit_specsig_oc_call when building an outer frule. The @eval below
# simulates a further world-counter advance mid-test, matching the actual failure mode.
@testset "julia#61368 regression: frule after world-counter advance" begin
    _reg_dep(x::Float64) = x
    _reg_inner(x::Float64) = _reg_dep(x) * 2.0
    _reg_outer(x::Float64) = _reg_inner(x) + _reg_inner(x)
    rng = StableRNG(123)
    TestUtils.test_rule(
        rng, _reg_inner, 1.0; perf_flag=:none, is_primitive=false, mode=ForwardMode
    )
    # Redefine _reg_dep to advance the world counter and invalidate _reg_inner's
    # compiled OC CodeInstance.
    @eval _reg_dep(x::Float64) = x + 0.0
    TestUtils.test_rule(
        rng, _reg_outer, 1.0; perf_flag=:none, is_primitive=false, mode=ForwardMode
    )
end

include(joinpath(@__DIR__, "..", "..", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", joinpath("tangents", "tangents.jl")))
include(joinpath(@__DIR__, "..", "..", joinpath("tangents", "codual.jl")))
include(joinpath(@__DIR__, "..", "..", "stack.jl"))

# The interface tests include debug-mode runs that deliberately pass incorrect
# arguments, causing DispatchDoctor, Julia Base, and the compiler to raise issues
# most likely unrelated to Mooncake. In general, Mooncake should avoid depending
# on third-party compiler-based tools; JET (developed by JuliaLang) is the
# exception.
allow_unstable() do
    include(joinpath(@__DIR__, "..", "..", "interface.jl"))
end
