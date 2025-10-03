using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake: Mooncake, TestUtils, Tangent
using DispatchDoctor: allow_unstable, type_instability

TestUtils.test_hook(::Any, ::typeof(TestUtils.test_opt), ::Any...) = nothing
TestUtils.test_hook(::Any, ::typeof(TestUtils.report_opt), tt) = nothing
function TestUtils.test_hook(f, ::typeof(Mooncake.hand_written_rule_test_cases), ::Any...)
    return allow_unstable(f)
end
function TestUtils.test_hook(f, ::typeof(Mooncake.derived_rule_test_cases), ::Any...)
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

include(joinpath(@__DIR__, "..", "..", "front_matter.jl"))

include(joinpath(@__DIR__, "..", "..", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "tangents.jl"))
include(joinpath(@__DIR__, "..", "..", "codual.jl"))
include(joinpath(@__DIR__, "..", "..", "stack.jl"))
include(joinpath(@__DIR__, "..", "..", "interface.jl"))
