# Avoid troublesome bitcast magic -- we can't handle converting from pointer to UInt,
# because we drop the gradient, because the tangent type of integers is NoTangent.
# https://github.com/JuliaLang/julia/blob/9f9e989f241fad1ae03c3920c20a93d8017a5b8f/base/pointer.jl#L282
@is_primitive MinimalCtx Tuple{typeof(Base.:(+)), Ptr, Integer}
function rrule!!(f::CoDual{typeof(Base.:(+))}, x::CoDual{<:Ptr}, y::CoDual{<:Integer})
    return CoDual(primal(x) + primal(y), tangent(x) + primal(y)), NoPullback(f, x, y)
end

@is_primitive MinimalCtx Tuple{typeof(randn), AbstractRNG, Vararg}
function rrule!!(f::CoDual{typeof(randn)}, rng::CoDual{<:AbstractRNG}, args::CoDual...)
    x = randn(primal(rng), map(primal, args)...)
    return zero_fcodual(x), NoPullback(f, rng, args...)
end

function generate_hand_written_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code}
)
    _x = Ref(5.0)
    _dx = Ref(4.0)
    test_cases = vcat(
        Any[
            # Rules to avoid pointer type conversions.
            (
                true, :stability_and_allocs, nothing,
                +,
                CoDual(
                    bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                    bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
                ),
                2,
            ),
        ],

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
    )
    memory = Any[_x, _dx]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(
    rng_ctor, ::Val{:avoiding_non_differentiable_code},
)
    return Any[], Any[]
end
