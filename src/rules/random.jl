# The following functions contain ccall or llvmcall which cannot be differentiated.
@zero_derivative MinimalCtx Tuple{Type{MersenneTwister},Any}
# xoshiro_bulk_simd/nosimd(rng, dst::Ptr{UInt8}, len::Int64, eltype, [simd_width,] bits2float)
@zero_derivative MinimalCtx Tuple{
    typeof(Random.XoshiroSimd.xoshiro_bulk_simd),
    Union{TaskLocalRNG,Xoshiro},
    Ptr{UInt8},
    Int64,
    Any,
    Any,
    Any,
}
@zero_derivative MinimalCtx Tuple{
    typeof(Random.XoshiroSimd.xoshiro_bulk_nosimd),
    Union{TaskLocalRNG,Xoshiro},
    Ptr{UInt8},
    Int64,
    Any,
    Any,
}

const KnownRNGs = Union{MersenneTwister,RandomDevice,TaskLocalRNG,Xoshiro}
@zero_derivative MinimalCtx Tuple{typeof(rand),KnownRNGs}
@zero_derivative MinimalCtx Tuple{typeof(randn),KnownRNGs}
@zero_derivative MinimalCtx Tuple{typeof(randexp),KnownRNGs}
@zero_derivative MinimalCtx Tuple{typeof(rand),KnownRNGs,Type{<:IEEEFloat}}
@zero_derivative MinimalCtx Tuple{
    typeof(rand),KnownRNGs,Union{Type{UInt32},Type{UInt64},Type{UInt128}}
}
@zero_derivative MinimalCtx Tuple{typeof(randn),KnownRNGs,Type{<:IEEEFloat}}
@zero_derivative MinimalCtx Tuple{typeof(randexp),KnownRNGs,Type{<:IEEEFloat}}

# A draw is random state, not a reparameterised sample. The result aliases the destination:
# restore its primal and derivative storage on the pullback, but leave the RNG advanced.
# These rules also serve GPU arrays (`CuArray`), so destinations must stay `DenseArray`s.
# `AbstractRNG` is the narrowest type covering GPUArrays' RNG, which has no `randexp!`;
# no RNG has a complex `randexp`. The rules need the destination to be its own tangent,
# which `@is_primitive` cannot express; other `DenseArray`s (`SharedArray`) are derived.
for (f, A) in (
    (rand!, DenseArray{<:FloatOrComplex}),
    (randn!, DenseArray{<:FloatOrComplex}),
    (randexp!, Array{<:IEEEFloat}),
)
    @eval function _is_primitive(
        ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{T}
    ) where {T<:Tuple{typeof($f),AbstractRNG,$A}}
        T isa DataType || return false
        P = T.parameters[3]
        return tangent_type(P) == P
    end
    @eval function frule!!(
        ::Dual{typeof($f)}, rng::Dual{<:AbstractRNG}, x::Dual{P,P}
    ) where {P<:$A}
        $f(primal(rng), primal(x))
        fill!(tangent(x), zero(eltype(P)))
        return x
    end
    @eval function rrule!!(
        ::CoDual{typeof($f)}, rng::CoDual{<:AbstractRNG}, x::CoDual{P,P}
    ) where {P<:$A}
        px, dx = primal(x), tangent(x)
        old_x, old_dx = copy(px), copy(dx)
        $f(primal(rng), px)
        fill!(dx, zero(eltype(P)))
        rng_rdata = zero_rdata(primal(rng))
        function rand_fill_pb!!(::NoRData)
            copyto!(px, old_x)
            copyto!(dx, old_dx)
            return NoRData(), rng_rdata, NoRData()
        end
        return x, rand_fill_pb!!
    end
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:random})
    rngs = [MersenneTwister(123), TaskLocalRNG(), Xoshiro(123)]
    all_rngs = vcat(rngs, RandomDevice())
    test_cases = vcat(

        # Random number generator construction.
        # There are some undefined fields at construction, so we cannot run equality tests.
        (true, :none, nothing, MersenneTwister, 123),

        # Random number generation.
        map_prod([rand, randn, randexp], all_rngs) do (f, rng)
            (true, :stability_and_allocs, nothing, f, rng)
        end...,
        map_prod([Float64, Float32], [rand, randn, randexp], all_rngs) do (P, f, rng)
            (true, :stability_and_allocs, nothing, f, rng, P)
        end...,
        map_prod([rand!, randn!, randexp!], vcat(rngs, rng_ctor(123))) do (f, rng)
            (true, :stability, nothing, f, rng, randn(5))
        end...,
        map_prod([rand!, randn!], [Float32, ComplexF64]) do (f, P)
            (true, :stability, nothing, f, Xoshiro(123), randn(P, 5))
        end...,
    )
    return test_cases, Any[]
end

function derived_rule_test_cases(rng_ctor, ::Val{:random})
    test_cases = Any[

        # Random number generation.
        (false, :none, nothing, x -> x * rand(Xoshiro(123)), 3.0),
        (false, :none, nothing, x -> x * randn(Xoshiro(123)), 3.0),
        (false, :none, nothing, x -> x * randexp(Xoshiro(123)), 3.0),
        (false, :none, nothing, x -> x * rand(Xoshiro(123), Float32), 3.0),
        (false, :none, nothing, x -> x * randn(Xoshiro(123), Float32), 3.0),
        (false, :none, nothing, x -> x * randexp(Xoshiro(123), Float32), 3.0),
        (false, :none, nothing, x -> x .* rand!(Xoshiro(123), x), randn(9)),
        (false, :none, nothing, x -> x .* randn!(Xoshiro(123), x), randn(9)),
        (false, :none, nothing, x -> x .* randexp!(Xoshiro(123), x), randn(9)),
        (false, :none, nothing, x -> x .* rand(Xoshiro(123), size(x)...), randn(9)),
        (false, :none, nothing, x -> x .* randn(Xoshiro(123), size(x)...), randn(9)),
        (false, :none, nothing, x -> x .* randexp(Xoshiro(123), size(x)...), randn(9)),

        # RNG construction.
        (false, :none, nothing, x -> randn(MersenneTwister(x)), 123),
        (false, :none, nothing, Xoshiro, 123),
        (false, :none, nothing, x -> randn(Random.seed!(TaskLocalRNG(), x)), 123),

        # It is not possible to make the numbers produced by a `RandomDevice` be
        # deterministic, because it gets its randomness "from the device". As such, we cannot
        # test that the numbers coming out of this are consistent, just that it runs.
        (true, :none, nothing, () -> randn(RandomDevice())),
    ]
    return test_cases, Any[]
end
