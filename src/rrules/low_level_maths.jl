for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f)) ||
        M == :SpecialFunctions
        # @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    (f == :rem2pi || f == :ldexp) && continue # not designed for Float64s
    (f in [:+, :*, :sin, :cos, :exp]) && continue # use other functionality to implement these
    if arity == 1
        dx = DiffRules.diffrule(M, f, :x)
        pb_name = Symbol("$(M).$(f)_pb!!")
        @eval begin
            @is_primitive MinimalCtx Tuple{typeof($M.$f), P} where {P<:IEEEFloat}
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
            @is_primitive MinimalCtx Tuple{typeof($M.$f), P, P} where {P<:IEEEFloat}
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

@is_primitive MinimalCtx Tuple{typeof(sin), <:IEEEFloat}
function rrule!!(::CoDual{typeof(sin), NoFData}, x::CoDual{P, NoFData}) where {P<:IEEEFloat}
    s, c = sincos(primal(x))
    sin_pullback!!(dy::P) = NoRData(), dy * c
    return CoDual(s, NoFData()), sin_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(cos), <:IEEEFloat}
function rrule!!(::CoDual{typeof(cos), NoFData}, x::CoDual{P, NoFData}) where {P<:IEEEFloat}
    s, c = sincos(primal(x))
    cos_pullback!!(dy::P) = NoRData(), -dy * s
    return CoDual(c, NoFData()), cos_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(exp), <:IEEEFloat}
function rrule!!(::CoDual{typeof(exp)}, x::CoDual{P}) where {P<:IEEEFloat}
    y = exp(primal(x))
    exp_pb!!(dy::P) = NoRData(), dy * y
    return zero_fcodual(y), exp_pb!!
end

rand_inputs(rng, P::Type{<:IEEEFloat}, f, arity) = randn(rng, P, arity)
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acosh), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asech), _) = (rand(rng) * 0.9, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asin), _) = (rand(rng) * 0.9, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asecd), _) = (rand(rng) + 1, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log2), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log10), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acscd), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(log1p), _) = (rand(rng) + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acsc), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(atanh), _) = (2 * 0.9 * rand(rng) - 0.9, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acoth), _) = (rand(rng) + 1 + 1e-3, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asind), _) = (0.9 * rand(rng), )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(asec), _) = (rand(rng) + 1.001, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acosd), _) = (2 * 0.9 * rand(rng) - 0.9, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(acos), _) = (2 * 0.9 * rand(rng) - 0.9, )
rand_inputs(rng, P::Type{<:IEEEFloat}, ::typeof(sqrt), _) = (rand(rng) + 1e-3, )

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:low_level_maths})
    rng = Xoshiro(123)
    test_cases = Any[]
    foreach(DiffRules.diffrules(; filter_modules=nothing)) do (M, f, arity)
        if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f)) ||
            M == :SpecialFunctions
            return  # Skip rules for methods not defined in the current scope
        end
        arity > 2 && return
        (f == :rem2pi || f == :ldexp || f == :(^)) && return
        (f == :+ || f == :*) && return # use intrinsics instead
        f = @eval $M.$f
        push!(test_cases, (false, :stability, nothing, f, rand_inputs(rng, Float64, f, arity)...))
        push!(test_cases, (true, :stability, nothing, f, rand_inputs(rng, Float32, f, arity)...))
    end
    memory = Any[]
    return test_cases, memory
end

generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:low_level_maths}) = Any[], Any[]
