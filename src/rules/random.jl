# The following functions contain ccall or llvmcall which cannot be differentiated.
@zero_derivative MinimalCtx Tuple{Type{MersenneTwister},Any}
# These kernels overwrite `len` bytes through a byte pointer. Its tangent still points
# at the original tangent storage after the pointer bitcast. Float storage has the same
# layout as its tangent; integer byte pointers do not specify the tangent layout.
# Kernel rules catch `Xoshiro` fills that bypass the `DenseArray` rules (e.g. views).
for (f, nargs) in
    ((Random.XoshiroSimd.xoshiro_bulk_simd, 2), (Random.XoshiroSimd.xoshiro_bulk_nosimd, 1))
    @eval @is_primitive MinimalCtx Tuple{
        typeof($f),Union{TaskLocalRNG,Xoshiro},Ptr{UInt8},Int,Any,Vararg{Any,$nargs}
    }
    # SIMD returns the bytes actually written, leaving a tail for the non-SIMD kernel.
    written = if nargs == 2
        :(
            let block = 8 * typeof(primal(args[1])).parameters[1]
                div(primal(len), block) * block
            end
        )
    else
        :(primal(len))
    end
    @eval function frule!!(
        ::Dual{typeof($f)},
        rng::Dual{<:Union{TaskLocalRNG,Xoshiro}},
        dst::Dual{Ptr{UInt8}},
        len::Dual{Int},
        type::Dual{Type{T}},
        args::Vararg{Dual,$nargs},
    ) where {T}
        _, dx = _xoshiro_bulk_storage(dst, $written, T, primal(args[end]))
        y = $f(primal(rng), primal(dst), primal(len), T, map(primal, args)...)
        fill!(dx, zero(eltype(dx)))
        return zero_dual(y)
    end
    @eval function rrule!!(
        ::CoDual{typeof($f)},
        rng::CoDual{<:Union{TaskLocalRNG,Xoshiro}},
        dst::CoDual{Ptr{UInt8}},
        len::CoDual{Int},
        type::CoDual{Type{T}},
        args::Vararg{CoDual,$nargs},
    ) where {T}
        x, dx = _xoshiro_bulk_storage(dst, $written, T, primal(args[end]))
        old_x, old_dx = copy(x), copy(dx)
        y = $f(primal(rng), primal(dst), primal(len), T, map(primal, args)...)
        fill!(dx, zero(eltype(dx)))
        function xoshiro_bulk_pb!!(::NoRData)
            copyto!(x, old_x)
            copyto!(dx, old_dx)
            return ntuple(_ -> NoRData(), $(5 + nargs))
        end
        return zero_fcodual(y), xoshiro_bulk_pb!!
    end
end

function _xoshiro_bulk_storage(dst, n, ::Type{T}, f) where {T}
    T <: IEEEFloat && f === Random.XoshiroSimd._bits2float || throw(
        ArgumentError(
            "Xoshiro bulk AD requires a floating destination and the standard conversion. " *
            "Integer byte pointers erase the tangent layout; use rand! on a typed array instead.",
        ),
    )
    n >= 0 && n % sizeof(T) == 0 ||
        throw(ArgumentError("Xoshiro bulk writes must span whole destination elements."))
    x = unsafe_wrap(Vector{T}, Ptr{T}(primal(dst)), div(n, sizeof(T)))
    dx = unsafe_wrap(Vector{T}, Ptr{T}(tangent(dst)), length(x))
    return x, dx
end

# Integer byte pointers erase whether their shadow refers to integer or floating storage.
# Own standard integer fills while the destination type is still available; raw integer
# bulk calls must fail rather than leave a potentially floating destination tangent stale.
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

# A draw is random state, not a reparameterised sample. The result aliases the destination:
# restore its primal and derivative storage on the pullback, but leave the RNG advanced.
# These rules also serve GPU arrays (`CuArray`), so destinations must stay `DenseArray`s.
# `AbstractRNG` is the narrowest type covering GPUArrays' RNG, which has no `randexp!`;
# no RNG has a complex `randexp`. The rules need the destination to be its own tangent,
# which `@is_primitive` cannot express; other `DenseArray`s (`SharedArray`) are derived.
# Union eltypes must also be derived: they need not have a scalar `zero`.
# Only `rand!` has a trailing type/sampler argument; keep both arities in lockstep.
for (f, A) in (
        (rand!, DenseArray{<:FloatOrComplex}),
        (randn!, DenseArray{<:FloatOrComplex}),
        (randexp!, Array{<:IEEEFloat}),
    ),
    n in (f === rand! ? (0, 1) : (0,))

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
        function rand_fill_pb!!(::NoRData)
            copyto!(px, old_x)
            copyto!(dx, old_dx)
            return NoRData(), rng_rdata, NoRData(), sampler_rdata...
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
        [
            (true, :stability, nothing, rand!, rng, randn(P, 5), sampler) for
            rng in vcat(rngs, rng_ctor(123)) for P in (Float32, Float64, ComplexF64) for
            sampler in (P, Random.Sampler(rng, P))
        ]...,
        # Empty Union-element fills are valid primals, but must use derived rules.
        map_prod([rand!, randn!, randexp!], [Union{Float32,Float64}]) do (f, P)
            (
                false,
                :none,
                (is_primitive=false,),
                f,
                f === rand! ? rng_ctor(123) : Xoshiro(123),
                P[],
            )
        end...,
        map_prod([rand!, randn!], [Union{Float64,ComplexF64}]) do (f, P)
            (
                false,
                :none,
                (is_primitive=false,),
                f,
                f === rand! ? rng_ctor(123) : Xoshiro(123),
                P[],
            )
        end...,
    )
    # A small spread of eltypes, RNGs, and shapes, shared with the state checks in
    # test/rules/random.jl. Avoid the full Cartesian product of these dimensions.
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
    # Contiguous views reach the pointer-writing Xoshiro kernels on Julia 1.12+.
    parents = Any[fill(T(2), 73) for T in (Float16, Float32, Float64)]
    @static if VERSION >= v"1.11"
        append!(
            parents, [fill!(Memory{T}(undef, 73), 2) for T in (Float16, Float32, Float64)]
        )
    end
    for x in parents, rng in (Xoshiro(123), TaskLocalRNG()), n in (7, 71)
        v = view(x, 2:(n + 1))
        for sampler in ((), (eltype(x),), (Random.Sampler(rng, eltype(x)),))
            push!(
                test_cases, (true, :none, (is_primitive=false,), rand!, rng, v, sampler...)
            )
        end
    end
    # Retain the backing storage for pointer arguments used by the canonical rule tests.
    memory = Any[]
    for T in (VERSION >= v"1.12" ? (Float16, Float32, Float64) : (Float32, Float64)),
        n in (7, 641),
        f in (Random.XoshiroSimd.xoshiro_bulk_simd, Random.XoshiroSimd.xoshiro_bulk_nosimd)

        len = n * sizeof(T)
        x, dx = fill(T(2), n + 2), fill(T(3), n + 2)
        append!(memory, (x, dx))
        dst = CoDual(Ptr{UInt8}(pointer(x, 2)), Ptr{NoTangent}(pointer(dx, 2)))
        width = f === Random.XoshiroSimd.xoshiro_bulk_simd ? (Val(8),) : ()
        convert = Random.XoshiroSimd._bits2float
        push!(
            test_cases,
            (true, :stability, nothing, f, Xoshiro(123), dst, len, T, width..., convert),
        )
    end
    # Typed integer boundaries avoid interpreting NoTangent storage as floating bytes.
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
    return test_cases, memory
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
