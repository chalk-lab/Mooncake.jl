# Contains a ccall, which must be avoided.
@zero_derivative MinimalCtx Tuple{Type{MersenneTwister},Any}

const KnownRNGs = Union{MersenneTwister,RandomDevice,TaskLocalRNG,Xoshiro}
@zero_derivative MinimalCtx Tuple{typeof(randn),KnownRNGs}
@zero_derivative MinimalCtx Tuple{typeof(randexp),KnownRNGs}
@zero_derivative MinimalCtx Tuple{typeof(randn),KnownRNGs,Type{<:IEEEFloat}}
@zero_derivative MinimalCtx Tuple{typeof(randexp),KnownRNGs,Type{<:IEEEFloat}}

const SpecialisedRNGs = Union{MersenneTwister,TaskLocalRNG,Xoshiro}
for f in [randn!, randexp!]
    @eval @is_primitive MinimalCtx Tuple{typeof($f),SpecialisedRNGs,Array{Float64}}
    @eval function frule!!(
        ::Dual{typeof($f)}, rng::Dual{<:SpecialisedRNGs}, x::Dual{<:Array{Float64}}
    )
        $f(primal(rng), primal(x))
        tangent(x) .= 0
        return x
    end
    @eval function rrule!!(
        ::CoDual{typeof($f)}, rng::CoDual{<:SpecialisedRNGs}, x::CoDual{<:Array{Float64}}
    )
        current_x = copy(x.x)
        current_dx = copy(x.dx)
        $f(rng.x, x.x)
        x.dx .= 0.0
        function rand_adjoint(::NoRData)
            x.x .= current_x
            x.dx .= current_dx
            return NoRData(), NoRData(), NoRData()
        end
        return x, rand_adjoint
    end
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:random})
    rngs = [MersenneTwister(123), TaskLocalRNG(), Xoshiro(123)]
    all_rngs = vcat(rngs, RandomDevice())
    test_cases = vcat(

        # Random number generator construction.
        # There are some undefined fields at construction, so we cannot run equality tests.
        (true, :none, nothing, MersenneTwister, 123),

        # Random number generation.
        map_prod([randn, randexp], all_rngs) do (f, rng)
            (true, :stability_and_allocs, nothing, f, rng)
        end...,
        map_prod([Float64, Float32], [randn, randexp], all_rngs) do (P, f, rng)
            (true, :stability_and_allocs, nothing, f, rng, P)
        end...,
        map_prod([randn!, randexp!], rngs) do (f, rng)
            (true, :stability, nothing, f, rng, randn(5))
        end...,
    )
    return test_cases, Any[]
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:random})
    test_cases = Any[

        # Random number generation.
        (false, :none, nothing, x -> x * randn(Xoshiro(123)), 3.0),
        (false, :none, nothing, x -> x * randexp(Xoshiro(123)), 3.0),
        (false, :none, nothing, x -> x * randn(Xoshiro(123), Float32), 3.0),
        (false, :none, nothing, x -> x * randexp(Xoshiro(123), Float32), 3.0),
        (false, :none, nothing, x -> x .* randn!(Xoshiro(123), x), randn(9)),
        (false, :none, nothing, x -> x .* randexp!(Xoshiro(123), x), randn(9)),
        (false, :none, nothing, x -> x .* randn(Xoshiro(123), size(x)...), randn(9)),
        (false, :none, nothing, x -> x .* randexp(Xoshiro(123), size(x)...), randn(9)),

        # RNG construction.
        (false, :none, nothing, x -> randn(MersenneTwister(x)), 123),
        (false, :none, nothing, Xoshiro, 123),
        (false, :none, nothing, x -> randn(Random.seed!(TaskLocalRNG(), x)), 123),

        # It is not possible to make the numbers produced by a `RandomDevice` be
        # deterministic, because it gets is randomness "from the device". As such, we cannot
        # test that the numbers coming out of this are consistent, just that it runs.
        (true, :none, nothing, () -> randn(RandomDevice())),
    ]
    return test_cases, Any[]
end
