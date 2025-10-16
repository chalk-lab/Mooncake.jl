#
# Performance-only rules. These should be able to be removed, and everything still works,
# just a bit slower. The effect of these is typically to remove many nodes from the tape.
# Ideally, it would be the case that acitivty analysis eliminates any run-time improvements
# that these rules provide. Possibly they would still be useful in order to avoid having to
# deduce that these bits of code are inactive though.
#

@zero_derivative DefaultCtx Tuple{typeof(in),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(iszero),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(isempty),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(isbitstype),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(sizeof),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(promote_type),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.elsize),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Core.Compiler.sizeof_nothrow),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_haspadding),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_nfields),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_pointerfree),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_alignment),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_fielddesc_type),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(LinearAlgebra.chkstride1),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Threads.nthreads),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.depwarn),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.reduced_indices),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.check_reducedims),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.throw_boundserror),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.Broadcast.eltypes),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.eltype),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(Base.padding),DataType}
@zero_derivative MinimalCtx Tuple{typeof(Base.padding),DataType,Int}
@zero_derivative MinimalCtx Tuple{Type,TypeVar,Type}

# Required to avoid an ambiguity.
@zero_derivative MinimalCtx Tuple{Type{Symbol},TypeVar,Type}

@static if VERSION >= v"1.11-"
    @zero_derivative MinimalCtx Tuple{typeof(Random.hash_seed),Vararg}
    @zero_derivative MinimalCtx Tuple{typeof(Base.dataids),Memory}
end

"""
    lgetfield(x, f::Val)

An implementation of `getfield` in which the the field `f` is specified statically via a
`Val`. This enables the implementation to be type-stable even when it is not
possible to constant-propagate `f`. Moreover, it enable the pullback to also be type-stable.

It will always be the case that
```julia
getfield(x, :f) === lgetfield(x, Val(:f))
getfield(x, 2) === lgetfield(x, Val(2))
```

This approach is identical to the one taken by `Zygote.jl` to circumvent the same problem.
`Zygote.jl` calls the function `literal_getfield`, while we call it `lgetfield`.
"""
lgetfield(x, ::Val{f}) where {f} = getfield(x, f)

@is_primitive MinimalCtx Tuple{typeof(lgetfield),Any,Val}
@inline function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{P,T}, ::Dual{Val{f}}
) where {P,T<:StandardTangentType,f}
    primal_field = getfield(primal(x), f)
    if tangent_type(P) === NoTangent
        return uninit_dual(primal_field)
    else
        Dual(primal_field, _get_tangent_field(tangent(x), f))
    end
end

_get_tangent_field(f::Union{NamedTuple,Tuple}, name) = getfield(f, name)
_get_tangent_field(f::Union{NamedTuple,Tuple}, name, inbounds) = getfield(f, name, inbounds)
_get_tangent_field(f::Union{Tangent,MutableTangent}, name) = val(getfield(f.fields, name))
function _get_tangent_field(f::Union{Tangent,MutableTangent}, name, inbounds)
    return val(getfield(f.fields, name, inbounds))
end

@inline function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{P,F}, ::CoDual{Val{f}}
) where {P,F<:StandardFDataType,f}
    pb!! = if ismutabletype(P)
        dx = tangent(x)
        function mutable_lgetfield_pb!!(dy)
            increment_field_rdata!(dx, dy, Val{f}())
            return NoRData(), NoRData(), NoRData()
        end
    else
        dx_r = lazy_zero_rdata(primal(x))
        field = Val{f}()
        function immutable_lgetfield_pb!!(dy)
            return NoRData(), increment_field!!(instantiate(dx_r), dy, field), NoRData()
        end
    end
    y = CoDual(getfield(primal(x), f), _get_fdata_field(primal(x), tangent(x), f))
    return y, pb!!
end

@unstable @inline _get_fdata_field(_, t::Union{Tuple,NamedTuple}, f) = getfield(t, f)
@unstable @inline _get_fdata_field(_, data::FData, f) = val(getfield(data.data, f))
@unstable @inline _get_fdata_field(primal, ::NoFData, f) = uninit_fdata(getfield(primal, f))
@unstable @inline _get_fdata_field(_, t::MutableTangent, f) = fdata(
    val(getfield(t.fields, f))
)

increment_field_rdata!(dx::MutableTangent, ::NoRData, ::Val) = dx
increment_field_rdata!(dx::NoFData, ::NoRData, ::Val) = dx
function increment_field_rdata!(dx::T, dy_rdata, ::Val{f}) where {T<:MutableTangent,f}
    set_tangent_field!(dx, f, increment_rdata!!(get_tangent_field(dx, f), dy_rdata))
    return dx
end

#
# lgetfield with order argument
#

# This is largely copy + pasted from the above. Attempts were made to refactor to avoid
# code duplication, but it wound up not being any cleaner than this copy + pasted version.

@is_primitive MinimalCtx Tuple{typeof(lgetfield),Any,Val,Val}
@inline function frule!!(
    ::Dual{typeof(lgetfield)},
    x::Dual{P,<:StandardTangentType},
    ::Dual{Val{f}},
    ::Dual{Val{order}},
) where {P,f,order}
    primal_field = getfield(primal(x), f, order)
    if tangent_type(P) === NoTangent
        return uninit_dual(primal_field)
    else
        return Dual(primal_field, _get_tangent_field(tangent(x), f))
    end
end
@inline function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{P,F}, ::CoDual{Val{f}}, ::CoDual{Val{order}}
) where {P,F<:StandardFDataType,f,order}
    pb!! = if ismutabletype(P)
        dx = tangent(x)
        function mutable_lgetfield_pb!!(dy)
            increment_field_rdata!(dx, dy, Val{f}())
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
    else
        dx_r = lazy_zero_rdata(primal(x))
        function immutable_lgetfield_pb!!(dy)
            tmp = increment_field!!(instantiate(dx_r), dy, Val{f}())
            return NoRData(), tmp, NoRData(), NoRData()
        end
    end
    y = CoDual(getfield(primal(x), f, order), _get_fdata_field(primal(x), tangent(x), f))
    return y, pb!!
end

@is_primitive MinimalCtx Tuple{typeof(lsetfield!),Any,Any,Any}
@inline function frule!!(
    ::Dual{typeof(lsetfield!)}, value::Dual{P,T}, name::Dual, x::Dual
) where {P,T<:StandardTangentType}
    return lsetfield_frule(value, name, x)
end
@inline function rrule!!(
    ::CoDual{typeof(lsetfield!)}, value::CoDual{P,F}, name::CoDual, x::CoDual
) where {P,F<:StandardFDataType}
    return lsetfield_rrule(value, name, x)
end

function lsetfield_frule(value::Dual{P,T}, ::Dual{Val{name}}, x::Dual) where {P,T,name}
    setfield!(primal(value), name, primal(x))
    T !== NoTangent && set_tangent_field!(tangent(value), name, tangent(x))
    return x
end

function lsetfield_rrule(
    value::CoDual{P,F}, ::CoDual{Val{name}}, x::CoDual
) where {P,F,name}
    save = isdefined(primal(value), name)
    old_x = save ? getfield(primal(value), name) : nothing
    old_dx = if F == NoFData
        NoFData()
    else
        save ? get_tangent_field(tangent(value), name) : nothing
    end
    dvalue = tangent(value)
    pb!! = if F == NoFData
        function __setfield!_pullback(dy)
            old_x !== nothing && lsetfield!(primal(value), Val(name), old_x)
            return NoRData(), NoRData(), NoRData(), dy
        end
    else
        function setfield!_pullback(dy)
            new_dx = increment!!(dy, rdata(get_tangent_field(dvalue, name)))
            old_x !== nothing && lsetfield!(primal(value), Val(name), old_x)
            old_x !== nothing && set_tangent_field!(dvalue, name, old_dx)
            return NoRData(), NoRData(), NoRData(), new_dx
        end
    end
    yf = if F == NoFData
        NoFData()
    else
        fdata(set_tangent_field!(dvalue, name, zero_tangent(primal(x), tangent(x))))
    end
    y = CoDual(lsetfield!(primal(value), Val(name), primal(x)), yf)
    return y, pb!!
end

@static if VERSION < v"1.11"
    @is_primitive MinimalCtx Tuple{typeof(copy),Dict}
    function frule!!(::Dual{typeof(copy)}, a::Dual{<:Dict})
        return Dual(copy(primal(a)), _copy_dict_tangent(tangent(a)))
    end
    function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Dict})
        dx = tangent(a)
        t = dx.fields
        new_fields = typeof(t)((
            copy(t.slots), copy(t.keys), copy(t.vals), tuple_fill(NoTangent(), Val(5))...
        ))
        dy = MutableTangent(new_fields)
        y = CoDual(copy(primal(a)), dy)
        function copy_pullback!!(::NoRData)
            increment!!(dx, dy)
            return NoRData(), NoRData()
        end
        return y, copy_pullback!!
    end
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:misc})

    # Data which needs to not be GC'd.
    _x = Ref(5.0)
    _dx = Ref(4.0)
    memory = Any[_x, _dx]

    specific_test_cases = Any[
        # Rules to avoid pointer type conversions.
        (
            true,
            :stability,
            nothing,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),
            2,
        ),

        # Lack of activity-analysis rules:
        (false, :stability_and_allocs, nothing, Base.elsize, randn(5, 4)),
        (false, :stability_and_allocs, nothing, Base.elsize, view(randn(5, 4), 1:2, 1:2)),
        (false, :stability_and_allocs, nothing, Core.Compiler.sizeof_nothrow, Float64),
        (false, :stability_and_allocs, nothing, Base.datatype_haspadding, Float64),

        # Performance-rules that would ideally be completely removed.
        (false, :stability_and_allocs, nothing, in, 5.0, randn(4)),
        (false, :stability_and_allocs, nothing, iszero, 5.0),
        (false, :stability_and_allocs, nothing, isempty, randn(5)),
        (false, :stability_and_allocs, nothing, isbitstype, Float64),
        (false, :stability_and_allocs, nothing, sizeof, Float64),
        (false, :stability_and_allocs, nothing, promote_type, Float64, Float64),
        (false, :stability_and_allocs, nothing, LinearAlgebra.chkstride1, randn(3, 3)),
        (
            false,
            :stability_and_allocs,
            nothing,
            LinearAlgebra.chkstride1,
            randn(3, 3),
            randn(2, 2),
        ),
        (false, :allocs, nothing, Threads.nthreads),
        (false, :none, nothing, Base.eltype, randn(1)),
        (false, :none, nothing, Base.padding, @NamedTuple{a::Float64}),
        (false, :none, nothing, Base.padding, @NamedTuple{a::Float64}, 1),

        # Literal replacement for setfield!.
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            MutableFoo(5.0, [1.0, 2.0]),
            Val(:a),
            4.0,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            FullyInitMutableStruct(5.0, [1.0, 2.0]),
            Val(:y),
            [1.0, 3.0, 4.0],
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            NonDifferentiableFoo(5, false),
            Val(:x),
            4,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            NonDifferentiableFoo(5, false),
            Val(:y),
            true,
        ),
    ]

    # Some specific test cases for lgetfield to test the basics.
    specific_lgetfield_test_cases = Any[

        # Tuple
        (false, :stability_and_allocs, nothing, lgetfield, (5.0, 4), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (5.0, 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (1, 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, ((), 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (randn(2),), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (randn(2), 5), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (randn(2), 5), Val(2)),

        # NamedTuple
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(:a)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(:b)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2),), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2),), Val(:y)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(:y)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(:x)),

        # structs
        (false, :stability_and_allocs, nothing, lgetfield, 1:5, Val(:start)),
        (false, :stability_and_allocs, nothing, lgetfield, 1:5, Val(:stop)),
        (true, :none, (lb=1, ub=100), lgetfield, StructFoo(5.0), Val(:a)),
        (false, :none, (lb=1, ub=100), lgetfield, StructFoo(5.0, randn(5)), Val(:a)),
        (false, :none, (lb=1, ub=100), lgetfield, StructFoo(5.0, randn(5)), Val(:b)),
        (true, :none, (lb=1, ub=100), lgetfield, StructFoo(5.0), Val(1)),
        (false, :none, (lb=1, ub=100), lgetfield, StructFoo(5.0, randn(5)), Val(1)),
        (false, :none, (lb=1, ub=100), lgetfield, StructFoo(5.0, randn(5)), Val(2)),

        # mutable structs
        (true, :none, nothing, lgetfield, MutableFoo(5.0), Val(:a)),
        (false, :none, nothing, lgetfield, MutableFoo(5.0, randn(5)), Val(:b)),
        (false, :none, nothing, lgetfield, UInt8, Val(:name)),
        (false, :none, nothing, lgetfield, UInt8, Val(:super)),
        (true, :none, nothing, lgetfield, UInt8, Val(:layout)),
        (false, :none, nothing, lgetfield, UInt8, Val(:hash)),
        (false, :none, nothing, lgetfield, UInt8, Val(:flags)),
    ]

    # Create `lgetfield` tests for each type in TestTypes in order to increase coverage.
    general_lgetfield_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        _, primal = TestTypes.instantiate((interface_only, P, args))
        names = fieldnames(P)[1:length(args)] # only query fields which get initialised
        return Any[
            (interface_only, :none, nothing, lgetfield, primal, Val(name)) for name in names
        ]
    end

    # lgetfield has both 3 and 4 argument forms. Create test cases for both scenarios.
    all_lgetfield_test_cases = Any[
        (case..., order...) for
        case in vcat(specific_lgetfield_test_cases, general_lgetfield_test_cases...) for
        order in Any[(), (Val(false),)]
    ]

    # Create `lsetfield` testsfor each type in TestTypes in order to increase coverage.
    general_lsetfield_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        ismutabletype(P) || return Any[]
        _, primal = TestTypes.instantiate((interface_only, P, args))
        names = fieldnames(P)[1:length(args)] # only query fields which get initialised
        return Any[
            (interface_only, :none, nothing, lsetfield!, primal, Val(name), args[n]) for
            (n, name) in enumerate(names)
        ]
    end

    test_cases = vcat(
        specific_test_cases, all_lgetfield_test_cases..., general_lsetfield_test_cases...
    )
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:misc})
    test_cases = Any[
        (false, :none, nothing, x -> copy(Dict("A" => x[1], "B" => x[2]))["A"], (5.0, 5.0)),
        (false, :none, nothing, copy, Dict{Any,Any}("A" => [5.0], [3.0] => 5.0)),
        (false, :none, nothing, () -> copy(Set())),
    ]
    return test_cases, Any[]
end
