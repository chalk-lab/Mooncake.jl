using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Mooncake: Mooncake, TestUtils
using DispatchDoctor: allow_unstable, type_instability

TestUtils.test_hook(_, ::typeof(TestUtils.test_opt), ::Any...) = nothing
TestUtils.test_hook(_, ::typeof(TestUtils.report_opt), ::Any...) = nothing
TestUtils.test_hook(_, ::typeof(TestUtils.check_allocs), f, x...) = (f(x...); nothing)
TestUtils.test_hook(_, ::typeof(TestUtils.count_allocs), f, x...) = (f(x...); 0)
function TestUtils.test_hook(
    f, ::typeof(Mooncake.generate_hand_written_rrule!!_test_cases), ::Any...
)
    allow_unstable(f)
end
function TestUtils.test_hook(
    f, ::typeof(Mooncake.generate_derived_rrule!!_test_cases), ::Any...
)
    allow_unstable(f)
end

# Automatically skip instability checks for types which are themselves unstable.
function allow_unstable_given_unstable_type(f::F, ::Type{T}) where {F,T}
    skip_instability_check(T) ? allow_unstable(f) : f()
end
function skip_instability_check(::Type{T}) where {T}
    type_instability(T) || (isstructtype(T) && any(skip_instability_check, fieldtypes(T)))
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

function TestUtils.test_hook(f, ::typeof(TestUtils.test_tangent_interface), _, p; kws...)
    allow_unstable_given_unstable_type(f, typeof(p))
end
function TestUtils.test_hook(f, ::typeof(TestUtils.test_tangent_splitting), _, p; kws...)
    allow_unstable_given_unstable_type(f, typeof(p))
end

include(joinpath(@__DIR__, "..", "..", "front_matter.jl"))

include(joinpath(@__DIR__, "..", "..", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "tangents.jl"))
include(joinpath(@__DIR__, "..", "..", "codual.jl"))
include(joinpath(@__DIR__, "..", "..", "stack.jl"))
