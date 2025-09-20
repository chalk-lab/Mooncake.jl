for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f)) ||
        M == :SpecialFunctions
        # @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    (f == :rem2pi || f == :ldexp) && continue # not designed for Float64s
    (f in [:+, :*, :sin, :cos, :exp, :-, :abs2, :inv, :abs, :/, :\, :^, :hypot]) && continue # use other functionality to implement these
    if arity == 1
        dx = DiffRules.diffrule(M, f, :x)
        pb_name = Symbol("$(M).$(f)_pb!!")
        @eval begin
            @is_primitive MinimalCtx Tuple{typeof($M.$f),P} where {P<:IEEEFloat}
            function frule!!(::Dual{typeof($M.$f)}, _x::Dual{P}) where {P<:IEEEFloat}
                x = primal(_x)
                return Dual(($M.$f)(x), tangent(_x) * $dx)
            end
            function rrule!!(::CoDual{typeof($M.$f)}, _x::CoDual{P}) where {P<:IEEEFloat}
                x = primal(_x) # needed for dx expression
                $pb_name(ȳ::P) = NoRData(), ȳ * $dx
                return CoDual(($M.$f)(x), NoFData()), $pb_name
            end
        end
    elseif arity == 2
        da, db = DiffRules.diffrule(M, f, :a, :b)
        pb_name = Symbol("$(M).$(f)_pb!!")
        @eval begin
            @is_primitive MinimalCtx Tuple{typeof($M.$f),P,P} where {P<:IEEEFloat}
            function frule!!(
                ::Dual{typeof($M.$f)}, _a::Dual{P}, _b::Dual{P}
            ) where {P<:IEEEFloat}
                a = primal(_a)
                b = primal(_b)
                return Dual(($M.$f)(a, b), tangent(_a) * $da + tangent(_b) * $db)
            end
            function rrule!!(
                ::CoDual{typeof($M.$f)}, _a::CoDual{P}, _b::CoDual{P}
            ) where {P<:IEEEFloat}
                a = primal(_a)
                b = primal(_b)
                $pb_name(ȳ::P) = NoRData(), ȳ * $da, ȳ * $db
                return CoDual(($M.$f)(a, b), NoFData()), $pb_name
            end
        end
    end
end

@is_primitive MinimalCtx Mode Tuple{typeof(sin),<:IEEEFloat}
function frule!!(::Dual{typeof(sin)}, x::Dual{<:IEEEFloat})
    s, c = sincos(primal(x))
    return Dual(s, c * tangent(x))
end
function rrule!!(::CoDual{typeof(sin),NoFData}, x::CoDual{P,NoFData}) where {P<:IEEEFloat}
    s, c = sincos(primal(x))
    sin_pullback!!(dy::P) = NoRData(), dy * c
    return CoDual(s, NoFData()), sin_pullback!!
end

@is_primitive MinimalCtx Mode Tuple{typeof(cos),<:IEEEFloat}
function frule!!(::Dual{typeof(cos)}, x::Dual{<:IEEEFloat})
    s, c = sincos(primal(x))
    return Dual(c, -s * tangent(x))
end
function rrule!!(::CoDual{typeof(cos),NoFData}, x::CoDual{P,NoFData}) where {P<:IEEEFloat}
    s, c = sincos(primal(x))
    cos_pullback!!(dy::P) = NoRData(), -dy * s
    return CoDual(c, NoFData()), cos_pullback!!
end

@is_primitive MinimalCtx Mode Tuple{typeof(exp),<:IEEEFloat}
function frule!!(::Dual{typeof(exp)}, x::Dual{P}) where {P<:IEEEFloat}
    y = exp(primal(x))
    return Dual(y, y * tangent(x))
end
function rrule!!(::CoDual{typeof(exp)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = exp(primal(x))
    exp_pb!!(dy::P) = NoRData(), dy * y
    return zero_fcodual(y), exp_pb!!
end

@from_chainrules MinimalCtx Tuple{typeof(^),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(^)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    t = (ChainRules.NoTangent(), tangent(x), tangent(y))
    z, dz = ChainRules.frule(t, ^, primal(x), primal(y))
    return Dual(z, dz)
end

@is_primitive MinimalCtx Tuple{typeof(Base.eps),<:IEEEFloat}
function frule!!(::Dual{typeof(Base.eps)}, x::Dual{<:IEEEFloat})
    return Dual(eps(primal(x)), zero(primal(x)))
end
function rrule!!(::CoDual{typeof(Base.eps)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = Base.eps(primal(x))
    eps_pb!!(dy::P) = NoRData(), zero(y)
    return zero_fcodual(y), eps_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(hypot),P,P} where {P<:IEEEFloat}
function frule!!(::Dual{typeof(hypot)}, x::Dual{P}, y::Dual{P}) where {P<:IEEEFloat}
    h = hypot(primal(x), primal(y))
    dh = (primal(x) * tangent(x) + primal(y) * tangent(y)) / h
    return Dual(h, dh)
end
function rrule!!(::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P}) where {P<:IEEEFloat}
    h = hypot(primal(x), primal(y))
    hypot_pb!!(dh::P) = NoRData(), dh * (primal(x) / h), dh * (primal(y) / h)
    return CoDual(h, NoFData()), hypot_pb!!
end

@is_primitive MinimalCtx Tuple{typeof(hypot),P,P,Vararg{P}} where {P<:IEEEFloat}
function frule!!(
    ::Dual{typeof(hypot)}, x::Dual{P}, y::Dual{P}, xs::Vararg{Dual{P},N}
) where {P<:IEEEFloat,N}
    h = hypot(primal(x), primal(y), map(primal, xs)...)
    dh = sum(primal(a) * tangent(a) for a in (x, y, xs...)) / h
    return Dual(h, dh)
end
function rrule!!(
    ::CoDual{typeof(hypot)}, x::CoDual{P}, y::CoDual{P}, xs::Vararg{CoDual{P},N}
) where {P<:IEEEFloat,N}
    h = hypot(primal(x), primal(y), map(primal, xs)...)
    function hypot_pb!!(dh::P)
        grads = map(a -> dh * (primal(a) / h), (x, y, xs...))
        return NoRData(), grads...
    end
    return CoDual(h, NoFData()), hypot_pb!!
end

rand_inputs(rng, P::Type{<:IEEEFloat}, f, arity) = randn(rng, P, arity)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acosh), _) = (rand(rng) + 1 + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asech), _) = (rand(rng) * 0.9,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log), _) = (rand(rng) + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asin), _) = (rand(rng) * 0.9,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asecd), _) = (rand(rng) + 1,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log2), _) = (rand(rng) + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log10), _) = (rand(rng) + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acscd), _) = (rand(rng) + 1 + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log1p), _) = (rand(rng) + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acsc), _) = (rand(rng) + 1 + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(atanh), _) = (2 * 0.9 * rand(rng) - 0.9,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acoth), _) = (rand(rng) + 1 + 1e-3,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asind), _) = (0.9 * rand(rng),)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asec), _) = (rand(rng) + 1.001,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acosd), _) = (2 * 0.9 * rand(rng) - 0.9,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acos), _) = (2 * 0.9 * rand(rng) - 0.9,)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(sqrt), _) = (rand(rng) + 1e-3,)

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:low_level_maths})
    rng = Xoshiro(123)
    test_cases = Any[]
    foreach(DiffRules.diffrules(; filter_modules=nothing)) do (M, f, arity)
        if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f)) ||
            M == :SpecialFunctions
            return nothing  # Skip rules for methods not defined in the current scope
        end
        arity > 2 && return nothing
        (f == :rem2pi || f == :ldexp || f == :(^)) && return nothing
        (f in [:+, :*, :sin, :cos, :exp, :-, :abs2, :inv, :abs, :/, :\]) && return nothing # use other functionality to implement these
        f = @eval $M.$f
        push!(
            test_cases,
            (false, :stability, nothing, f, rand_inputs(rng, Float64, f, arity)...),
        )
        push!(
            test_cases,
            (false, :stability, nothing, f, rand_inputs(rng, Float32, f, arity)...),
        )
    end

    # test cases for additional rules written in this file.
    push!(test_cases, (false, :stability_and_allocs, nothing, sin, 1.1))
    push!(test_cases, (false, :stability_and_allocs, nothing, sin, 1.0f1))
    push!(test_cases, (false, :stability_and_allocs, nothing, cos, 1.1))
    push!(test_cases, (false, :stability_and_allocs, nothing, cos, 1.0f1))
    push!(test_cases, (false, :stability_and_allocs, nothing, exp, 1.1))
    push!(test_cases, (false, :stability_and_allocs, nothing, exp, 1.0f1))
    push!(test_cases, (false, :stability_and_allocs, nothing, ^, 4.0, 5.0))
    push!(test_cases, (false, :stability_and_allocs, nothing, ^, 4.0f0, 5.0f0))
    # push!(test_cases, (false, :stability_and_allocs, nothing, Base.eps, 4.0f0)) correctness tests fail as we compare against FDM, run manually to verify
    push!(test_cases, (false, :stability_and_allocs, nothing, Base.eps, 5.0f0))
    push!(test_cases, (false, :stability_and_allocs, nothing, hypot, 4.0, 5.0))
    push!(test_cases, (false, :stability_and_allocs, nothing, hypot, 4.0f0, 5.0f0))
    push!(test_cases, (false, :stability_and_allocs, nothing, hypot, 4.0, 5.0, 6.0))
    push!(test_cases, (false, :stability_and_allocs, nothing, hypot, 4.0f0, 5.0f0, 6.0f0))
    memory = Any[]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:low_level_maths}) = Any[], Any[]
