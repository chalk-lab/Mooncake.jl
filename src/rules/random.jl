# The following functions contain ccall or llvmcall which cannot be differentiated.
@zero_derivative MinimalCtx Tuple{Type{MersenneTwister},Any}
# Raw bulk calls must throw: a byte pointer may target integer storage without tangents.
for (f, nargs) in
    ((Random.XoshiroSimd.xoshiro_bulk_simd, 2), (Random.XoshiroSimd.xoshiro_bulk_nosimd, 1))
    @eval @is_primitive MinimalCtx Tuple{
        typeof($f),Union{TaskLocalRNG,Xoshiro},Ptr{UInt8},Int,Any,Vararg{Any,$nargs}
    }
    for (rule, D) in ((:frule!!, :Dual), (:rrule!!, :CoDual))
        @eval function $rule(
            ::$D{typeof($f)},
            ::$D{<:Union{TaskLocalRNG,Xoshiro}},
            ::$D{Ptr{UInt8}},
            ::$D{Int},
            ::$D,
            ::Vararg{$D,$nargs},
        )
            throw(
                ArgumentError(
                    "Xoshiro bulk AD cannot determine the destination's tangent layout; " *
                    "use rand! on a typed Array, Memory, or contiguous view instead.",
                ),
            )
        end
    end
end

# 1.12's Xoshiro fill accepts contiguous views and passes them as Ptr{UInt8}; intercept.
@static if VERSION >= v"1.12"
    const XoshiroFloatView{T,N} = Base.FastContiguousSubArray{T,N,<:Union{Array,Memory}}
    @is_primitive MinimalCtx Tuple{
        typeof(rand!),
        Union{TaskLocalRNG,Xoshiro},
        XoshiroFloatView{T},
        Random.SamplerTrivial{Random.CloseOpen01{T}},
    } where {T<:IEEEFloat}
    function frule!!(
        ::Dual{typeof(rand!)},
        rng::Dual{<:Union{TaskLocalRNG,Xoshiro}},
        x::Dual{<:XoshiroFloatView{T}},
        sampler::Dual{<:Random.SamplerTrivial{Random.CloseOpen01{T}}},
    ) where {T<:IEEEFloat}
        px, dx = arrayify(x)
        rand!(primal(rng), px, primal(sampler))
        fill!(dx, zero(T))
        return x
    end
    function rrule!!(
        ::CoDual{typeof(rand!)},
        rng::CoDual{<:Union{TaskLocalRNG,Xoshiro}},
        x::CoDual{<:XoshiroFloatView{T}},
        sampler::CoDual{<:Random.SamplerTrivial{Random.CloseOpen01{T}}},
    ) where {T<:IEEEFloat}
        px, dx = arrayify(x)
        old = map(tuple, px, dx)
        rand!(primal(rng), px, primal(sampler))
        fill!(dx, zero(T))
        function rand_view_pb!!(::NoRData)
            for i in eachindex(px, dx, old)
                @inbounds px[i], dx[i] = old[i]
            end
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
        return x, rand_view_pb!!
    end
end

# Xoshiro integer fills pass destination as Ptr{UInt8}; intercept accepted types.
@static if VERSION >= v"1.12"
    const XoshiroIntegerArray{T} = Base.MutableDenseArrayType{T}
else
    const XoshiroIntegerArray{T} = Array{T}
end
@is_primitive MinimalCtx Tuple{
    typeof(rand!),Union{TaskLocalRNG,Xoshiro},XoshiroIntegerArray{T},Random.SamplerType{T}
} where {T<:Union{Base.BitInteger,Bool}}
function frule!!(
    ::Dual{typeof(rand!)},
    rng::Dual{<:Union{TaskLocalRNG,Xoshiro}},
    x::Dual{<:XoshiroIntegerArray{T}},
    sampler::Dual{Random.SamplerType{T}},
) where {T<:Union{Base.BitInteger,Bool}}
    rand!(primal(rng), primal(x), primal(sampler))
    return x
end
function rrule!!(
    ::CoDual{typeof(rand!)},
    rng::CoDual{<:Union{TaskLocalRNG,Xoshiro}},
    x::CoDual{<:XoshiroIntegerArray{T}},
    sampler::CoDual{Random.SamplerType{T}},
) where {T<:Union{Base.BitInteger,Bool}}
    old_x = copy(primal(x))
    rand!(primal(rng), primal(x), primal(sampler))
    function rand_integer_pb!!(::NoRData)
        copyto!(primal(x), old_x)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, rand_integer_pb!!
end

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

# A draw is zero-derivative; DenseArray/AbstractRNG cover CuArray fills; RNG lacks randexp!.
for (f, A) in (
        (rand!, DenseArray{<:FloatOrComplex}),
        (randn!, DenseArray{<:FloatOrComplex}),
        (randexp!, Array{<:IEEEFloat}),
    ),
    n in (f === rand! ? (0, 1) : (0,))

    # Primitive if x is own tangent, not SharedArray; concrete eltype; Unions may lack zero.
    @eval function _is_primitive(
        ::Type{MinimalCtx}, ::Type{<:Mode}, ::Type{T}
    ) where {T<:Tuple{typeof($f),AbstractRNG,$A,Vararg{Union{Type,Random.Sampler},$n}}}
        T isa DataType || return false
        P = T.parameters[3]
        return isconcretetype(eltype(P)) && tangent_type(P) == P
    end
    @eval function frule!!(
        ::Dual{typeof($f)},
        rng::Dual{<:AbstractRNG},
        x::Dual{P,P},
        sampler::Vararg{Dual{<:Union{Type,Random.Sampler}},$n},
    ) where {P<:$A}
        $f(primal(rng), primal(x), map(primal, sampler)...)
        fill!(tangent(x), zero(eltype(P)))
        return x
    end
    @eval function rrule!!(
        ::CoDual{typeof($f)},
        rng::CoDual{<:AbstractRNG},
        x::CoDual{P,P},
        sampler::Vararg{CoDual{<:Union{Type,Random.Sampler}},$n},
    ) where {P<:$A}
        px, dx = primal(x), tangent(x)
        old_x, old_dx = copy(px), copy(dx)
        $f(primal(rng), px, map(primal, sampler)...)
        fill!(dx, zero(eltype(P)))
        rng_rdata = zero_rdata(primal(rng))
        sampler_rdata = map(s -> zero_rdata(primal(s)), sampler)
        # Draws change RNG state: restore the aliased destination, never rewind the RNG.
        function rand_fill_pb!!(::NoRData)
            copyto!(px, old_x)
            copyto!(dx, old_dx)
            return NoRData(), rng_rdata, NoRData(), sampler_rdata...
        end
        return x, rand_fill_pb!!
    end
end

# Only the Julia 1.12+ sampler form is primitive; other view fills use derived rules.
function _rand_view_cases(primitive::Bool)
    parents = Any[fill(T(2), 73) for T in (Float16, Float32, Float64)]
    @static if VERSION >= v"1.11"
        append!(
            parents, [fill!(Memory{T}(undef, 73), 2) for T in (Float16, Float32, Float64)]
        )
    end
    cases = Any[]
    for x in parents, rng in (Xoshiro(123), TaskLocalRNG()), n in (7, 71)
        v = view(x, 2:(n + 1))
        for sampler in ((), (eltype(x),), (Random.Sampler(rng, eltype(x)),))
            is_primitive =
                VERSION >= v"1.12" && !isempty(sampler) && sampler[1] isa Random.Sampler
            is_primitive == primitive || continue
            push!(cases, (true, :none, nothing, rand!, rng, v, sampler...))
        end
    end
    return cases
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
        [
            (true, :stability, nothing, rand!, rng, randn(P, 5), sampler) for
            rng in vcat(rngs, rng_ctor(123)) for P in (Float32, Float64, ComplexF64) for
            sampler in (P, Random.Sampler(rng, P))
        ]...,
    )
    fills = Any[
        (MersenneTwister(123), zeros(Float16, 2, 3)),
        (rng_ctor(123), zeros(Float32, 2, 2, 2)),
        (TaskLocalRNG(), zeros(Float64, 2, 3)),
        (Xoshiro(123), zeros(ComplexF32, 2, 2, 2)),
        (rng_ctor(123), zeros(ComplexF64, 2, 3)),
    ]
    @static if VERSION >= v"1.11"
        push!(fills, (Xoshiro(123), Memory{Float16}(undef, 5)))
        push!(fills, (MersenneTwister(123), Memory{ComplexF32}(undef, 5)))
    end
    for (rng, x) in fills
        fill!(x, 2)
        P = eltype(x)
        for f in (rand!, randn!)
            push!(test_cases, (true, :stability, nothing, f, rng, x))
        end
        for sampler in (P, Random.Sampler(rng, P))
            push!(test_cases, (true, :stability, nothing, rand!, rng, x, sampler))
        end
        if x isa Array{<:IEEEFloat}
            push!(test_cases, (true, :stability, nothing, randexp!, rng, x))
        end
    end
    append!(test_cases, _rand_view_cases(true))
    integers = Any[ones(T, 73) for T in (UInt8, Int64, Bool)]
    @static if VERSION >= v"1.12"
        append!(integers, [fill!(Memory{T}(undef, 73), 1) for T in (UInt8, Int64, Bool)])
        append!(integers, [view(x, 2:72) for x in copy(integers)])
    end
    for x in integers, rng in (Xoshiro(123), TaskLocalRNG())
        push!(
            test_cases,
            (true, :stability, nothing, rand!, rng, x, Random.Sampler(rng, eltype(x))),
        )
    end
    return test_cases, Any[]
end

# Build each RNG inside the case: fills advance it, and test_rule requires inputs restored.
function _rand_fill_cases()
    rngs = Any[
        () -> Xoshiro(123),
        () -> Random.seed!(TaskLocalRNG(), 123),
        () -> MersenneTwister(123),
    ]
    containers = Any[n -> fill(2.0, n)]
    @static if VERSION >= v"1.11"
        push!(containers, n -> fill!(Memory{Float64}(undef, n), 2))
    end
    cases = Any[]
    for (k, P) in enumerate((Float32, Float64, ComplexF32, ComplexF64)),
        sub in (false, true)

        rng = rngs[mod1(k, length(rngs))]
        dest = sub ? (x -> view(x, 2:8)) : identity
        fs = P <: Real ? (rand!, randn!, randexp!) : (rand!, randn!)
        for f in fs
            push!(cases, (x -> (f(rng(), dest(x)); sum(x)), fill(P(2), 9)))
        end
        for sampler in (r -> P, r -> Random.Sampler(r, P))
            push!(
                cases,
                (x -> (r=rng(); rand!(r, dest(x), sampler(r)); sum(x)), fill(P(2), 9)),
            )
        end
    end
    for c in containers, sub in (false, true), n in (7, 71)
        dest = sub ? (x -> view(x, 2:(n + 1))) : identity
        push!(cases, (x -> (rand!(Xoshiro(123), dest(x)); sum(x)), c(n + 2)))
    end
    for T in (Int64, Bool), sub in (false, true)
        dest = sub ? (y -> view(y, 2:8)) : identity
        push!(cases, (y -> (rand!(Xoshiro(123), dest(y)); 1.0), ones(T, 9)))
    end
    return Any[(false, :none, nothing, f, x) for (f, x) in cases]
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

        # Empty Union-element fills must take the derived rules, not the primitives.
        map_prod([rand!, randn!, randexp!], [Union{Float32,Float64}]) do (f, P)
            (false, :none, nothing, f, f === rand! ? rng_ctor(123) : Xoshiro(123), P[])
        end...,
        map_prod([rand!, randn!], [Union{Float64,ComplexF64}]) do (f, P)
            (false, :none, nothing, f, f === rand! ? rng_ctor(123) : Xoshiro(123), P[])
        end...,
        _rand_view_cases(false)...,
        _rand_fill_cases()...,
    ]
    return test_cases, Any[]
end
