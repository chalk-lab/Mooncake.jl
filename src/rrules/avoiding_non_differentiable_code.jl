# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)),Ptr,Integer}
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), tangent(x) + primal(y)), NoPullback(f, x, y)
end

@zero_adjoint MinimalCtx Tuple{typeof(randn),AbstractRNG,Vararg}
@zero_adjoint MinimalCtx Tuple{typeof(string),Vararg}
@zero_adjoint MinimalCtx Tuple{Type{Symbol},Vararg}
@zero_adjoint MinimalCtx Tuple{Type{Float64},Any,RoundingMode}
@zero_adjoint MinimalCtx Tuple{Type{Float32},Any,RoundingMode}
@zero_adjoint MinimalCtx Tuple{Type{Float16},Any,RoundingMode}
@zero_adjoint MinimalCtx Tuple{typeof(==),Type,Type}

@unstable function generate_hand_written_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code}
)
    _x = Ref(5.0)
    _dx = Ref(4.0)
    test_cases = vcat(
        Any[
        # Rules to avoid pointer type conversions.
            (
            true,
            :stability_and_allocs,
            nothing,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),
            2,
        ),],

        # Rules in order to avoid introducing determinism.
        reduce(
            vcat,
            map([Xoshiro(1), TaskLocalRNG()]) do rng
                return Any[
                    (true, :stability_and_allocs, nothing, randn, rng),
                    (true, :stability, nothing, randn, rng, 2),
                    (true, :stability, nothing, randn, rng, 3, 2),
                ]
            end,
        ),

        # Rules to make string-related functionality work properly.
        (false, :stability, nothing, string, 'H'),

        # Rules to make Symbol-related functionality work properly.
        (false, :stability_and_allocs, nothing, Symbol, "hello"),
        (false, :stability_and_allocs, nothing, Symbol, UInt8[1, 2]),
        (false, :stability_and_allocs, nothing, Float64, π, RoundDown),
        (false, :stability_and_allocs, nothing, Float64, π, RoundUp),
        (true, :stability_and_allocs, nothing, Float32, π, RoundDown),
        (true, :stability_and_allocs, nothing, Float32, π, RoundUp),
        (true, :stability_and_allocs, nothing, Float16, π, RoundDown),
        (true, :stability_and_allocs, nothing, Float16, π, RoundUp),
    )
    memory = Any[_x, _dx]
    return test_cases, memory
end

@unstable function generate_derived_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code}
)
    return Any[], Any[]
end
