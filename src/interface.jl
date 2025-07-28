"""
    __value_and_pullback!!(rule, ȳ, f::CoDual, x::CoDual...; y_cache=nothing)

*Note:* this is not part of the public Mooncake.jl interface, and may change without warning.

In-place version of `value_and_pullback!!` in which the arguments have been wrapped in
`CoDual`s. Note that any mutable data in `f` and `x` will be incremented in-place. As such,
if calling this function multiple times with different values of `x`, should be careful to
ensure that you zero-out the tangent fields of `x` each time.
"""
function __value_and_pullback!!(
    rule::R, ȳ::T, fx::Vararg{CoDual,N}; y_cache=nothing
) where {R,N,T}
    fx_fwds = tuple_map(to_fwds, fx)
    __verify_sig(rule, fx_fwds)
    out, pb!! = rule(fx_fwds...)
    @assert _typeof(tangent(out)) == fdata_type(T)
    increment!!(tangent(out), fdata(ȳ))
    v = if y_cache === nothing
        _copy_output(primal(out))
    else
        _copy_to_output!!(y_cache, primal(out))
    end
    return v, tuple_map((f, r) -> tangent(fdata(tangent(f)), r), fx, pb!!(rdata(ȳ)))
end

function __verify_sig(rule::DerivedRule{<:Any,sig}, fx::Tfx) where {sig,Tfx}
    Pfx = typeof(__unflatten_codual_varargs(_isva(rule), fx, rule.nargs))
    if sig != Pfx
        msg = "signature of arguments, $Pfx, not equal to signature required by rule, $sig."
        throw(ArgumentError(msg))
    end
end

__verify_sig(rule::DebugRRule, fx) = __verify_sig(rule.rule, fx)

# rrule!! doesn't specify specific argument types which must be used, so there's nothing to
# check here.
__verify_sig(::typeof(rrule!!), fx::Tuple) = nothing

struct ValueAndGradientReturnTypeError <: Exception
    msg::String
end

function throw_val_and_grad_ret_type_error(y)
    throw(
        ValueAndGradientReturnTypeError(
            "When calling __value_and_gradient!!, return value of primal must be a " *
            "subtype of IEEEFloat. Instead, found value of type $(typeof(y)).",
        ),
    )
end

struct ValueAndPullbackReturnTypeError <: Exception
    msg::String
end

function throw_forward_ret_type_error(y)
    throw(
        ValueAndPullbackReturnTypeError(
            "Found a value of type $(typeof(y)) in output, but output is not permitted to be or contain a pointer. This is because the amount of memory to which it refers is unknown, therefore Mooncake.jl is unable to allocate appropriate memory for its gradients.",
        ),
    )
end

function throw_circular_reference_or_alias_error(y)
    throw(
        ValueAndPullbackReturnTypeError(
            "Object with address $(objectid(y)) and type $(typeof(y)) appears more than once." *
            " Output cannot contain Circular references or aliases",
        ),
    )
end

"""
    __value_and_gradient!!(rule, f::CoDual, x::CoDual...)

*Note:* this is not part of the public Mooncake.jl interface, and may change without warning.

Equivalent to `__value_and_pullback!!(rule, 1.0, f, x...)` -- assumes `f` returns a `Float64`.

```jldoctest
# Set up the problem.
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
rule = build_rrule(f, x, y)

# Allocate tangents. These will be written to in-place. You are free to re-use these if you
# compute gradients multiple times.
tf = zero_tangent(f)
tx = zero_tangent(x)
ty = zero_tangent(y)

# Do AD.
Mooncake.__value_and_gradient!!(
    rule, Mooncake.CoDual(f, tf), Mooncake.CoDual(x, tx), Mooncake.CoDual(y, ty)
)
# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
function __value_and_gradient!!(rule::R, fx::Vararg{CoDual,N}) where {R,N}
    fx_fwds = tuple_map(to_fwds, fx)
    __verify_sig(rule, fx_fwds)
    out, pb!! = rule(fx_fwds...)
    y = primal(out)
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    return y, tuple_map((f, r) -> tangent(fdata(tangent(f)), r), fx, pb!!(one(y)))
end

"""
    value_and_pullback!!(rule, ȳ, f, x...)

Compute the value and pullback of `f(x...)`. `ȳ` must be a valid tangent for the primal
return by `f(x...)`.

`rule` should be constructed using `build_rrule`.

*Note:* There are lots of subtle ways to mis-use `value_and_pullback!!`, so we generally
recommend using `value_and_gradient!!` where possible.

*Note:* If calling `value_and_pullback!!` multiple times for various values of `x`, you
should use the same instance of `rule` each time.

*Note:* It is your responsibility to ensure that there is no aliasing in `f` and `x`.
For example,
```julia
X = randn(5, 5)
rule = build_rrule(dot, X, X)
value_and_pullback!!(rule, 1.0, dot, X, X)
```
will yield the wrong result.

*Note:* This method of `value_and_pullback!!` has to first call `zero_codual` on all of its
arguments. This may cause some additional allocations. If this is a problem in your
use-case, consider pre-allocating the `CoDual`s and calling the other method of this
function. The `CoDual`s should be primal-tangent pairs (as opposed to primal-fdata pairs).
There are lots of ways to get this wrong though, so we generally advise against doing this.
"""
function value_and_pullback!!(rule::R, ȳ, fx::Vararg{Any,N}) where {R,N}
    return __value_and_pullback!!(rule, ȳ, __create_coduals(fx)...)
end

"""
    value_and_gradient!!(rule, f, x...)

Equivalent to `value_and_pullback!!(rule, 1.0, f, x...)`, and assumes `f` returns a
`Union{Float16,Float32,Float64}`.

*Note:* There are lots of subtle ways to mis-use [`value_and_pullback!!`](@ref), so we generally
recommend using `Mooncake.value_and_gradient!!` (this function) where possible. The
docstring for [`value_and_pullback!!`](@ref) is useful for understanding this function though.

An example:
```jldoctest
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
rule = build_rrule(f, x, y)
value_and_gradient!!(rule, f, x, y)

# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
function value_and_gradient!!(rule::R, fx::Vararg{Any,N}) where {R,N}
    return __value_and_gradient!!(rule, __create_coduals(fx)...)
end

function __create_coduals(args)
    try
        return tuple_map(zero_codual, args)
    catch e
        if e isa StackOverflowError
            error(
                "Found a StackOverFlow error when trying to wrap inputs. This often " *
                "means that Mooncake.jl has encountered a self-referential type. Mooncake.jl " *
                "is not presently able to handle self-referential types, so if you are " *
                "indeed using a self-referential type somewhere, you will need to " *
                "refactor to avoid it if you wish to use Mooncake.jl.",
            )
        else
            rethrow(e)
        end
    end
end

struct Cache{Trule,Ty_cache,Ttangents<:Tuple}
    rule::Trule
    y_cache::Ty_cache
    tangents::Ttangents
end

"""
    __exclude_unsupported_output(y)
    __exclude_func_with_unsupported_output(fx)

Required for the robust design of [`value_and_pullback!!`](@ref), [`prepare_pullback_cache`](@ref).  
Ensures that `y` or returned value of `fx::Tuple{Tf, Targs...}` contains no aliasing, circular references, `Ptr`s or non differentiable datatypes. 
In the forward pass f(args...) output can only return a "Tree" like datastructure with leaf nodes as primitive types.  
Refer https://github.com/chalk-lab/Mooncake.jl/issues/517#issuecomment-2715202789 and related issue for details.  
Internally calls [`__exclude_unsupported_output_internal!`](@ref).
The design is modelled after `zero_tangent`.
"""
function __exclude_unsupported_output(y::T) where {T}
    __exclude_unsupported_output_internal!(y, Set{UInt}())
    return nothing
end

function __exclude_func_with_unsupported_output(fx)
    _fx = deepcopy(fx)
    _func, _args = _fx[1], _fx[2:end]
    _y = _func(_args...)
    __exclude_unsupported_output(_y)
end

"""
    __exclude_unsupported_output_internal(y::T, address_set::Set{UInt}) where {T}

For checking if output`y` is a valid Mutable/immutable composite or a primitive type.
Performs a recursive depth first search over the function output `y` with an `isbitstype()` check base case. The visited memory addresses are stored inside `address_set`.
If the set already contains a newly visited address, it errors out indicating an Alais or Circular reference.
Also errors out if `y` is or contains a Pointer.
It is called internally by [`__exclude_unsupported_output(y)`](@ref).
"""
function __exclude_unsupported_output_internal!(y::T, address_set::Set{UInt}) where {T}
    isbitstype(T) && return nothing
    if objectid(y) in address_set
        throw_circular_reference_or_alias_error(y)
    end

    # immutable types are copied on the stack.
    ismutable(y) && push!(address_set, objectid(y))

    # recurse over a composite type's fields.
    for y_sub in fieldnames(T)
        # isdefined() is valid for Mutable Structs, Structs.
        !isdefined(y, y_sub) && continue
        __exclude_unsupported_output_internal!(getfield(y, y_sub), address_set)
    end

    return nothing
end

const _BuiltinArrays = @static VERSION >= v"1.11" ? Union{Array,Memory} : Array

"""
    _copy_to_output!!(dst::T, src::T)

Copy the contents of `src` to `dst`, with zero or minimal new memory allocation. The type of `dst` and `src` must be the same.
Required as Base.copy!() does not work for all supported primal types. For example, `Base.copy!` does not work for `Core.svec`.
"""
_copy_to_output!!(dst::Number, src::Number) = src

# explicit copy for Core.svec
function _copy_to_output!!(dst::SimpleVector, src::SimpleVector)
    return Core.svec(map(_copy_to_output!!, dst, src)...)
end

# copy for Array, Memory
function _copy_to_output!!(dst::P, src::P) where {P<:_BuiltinArrays}
    @inbounds for i in eachindex(src)
        if isassigned(src, i)
            dst[i] = _copy_to_output!!(dst[i], src[i])
        end
    end
    return dst
end

# Tuple, NamedTuple
function _copy_to_output!!(dst::P, src::P) where {P<:Union{Tuple,NamedTuple}}
    isbitstype(P) && return src
    return map(_copy_to_output!!, dst, src)
end

# Handling structs
function _copy_to_output!!(dst::P, src::P) where {P}
    isbitstype(P) && return src
    nf = nfields(P)

    if ismutable(src)
        for src_sub in 1:nf
            if isdefined(src, src_sub)
                # using ccall as setfield! fails for const fields of a mutable struct.
                ccall(
                    :jl_set_nth_field,
                    Cvoid,
                    (Any, Csize_t, Any),
                    dst,
                    src_sub - 1,
                    _copy_to_output!!(getfield(dst, src_sub), getfield(src, src_sub)),
                )
            end
        end

        return dst
    else
        # this allocation is needed for handling undef fields in immutable structs.
        flds = Vector{Any}(undef, nf)
        for src_sub in 1:nf
            if isdefined(src, src_sub)
                flds[src_sub] = _copy_to_output!!(
                    getfield(dst, src_sub), getfield(src, src_sub)
                )
            else
                nf = src_sub - 1  # Assumes if a undefined field is found, all subsequent fields are undefined.
                break
            end
        end

        # when immutable struct object created by non initializing inner constructor. (Base.deepcopy misses this out)
        !isassigned(flds, 1) && return src
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), P, flds, nf)::P
    end
end

"""
    _copy_output(x::T)

Returns a copy of `x`, of the same type `T`. Allocates new memory for the copy.
Required as Base.copy() does not work for all supported primal types. For example, `Base.copy` does not work for `Core.svec`.
"""
_copy_output(x::SimpleVector) = Core.svec([map(_copy_output, x_sub) for x_sub in x]...)

# Array, Memory
function _copy_output(x::P) where {P<:_BuiltinArrays}
    temp = similar(x)
    Tx = eltype(P)
    @inbounds for i in eachindex(temp)
        if isassigned(x, i)
            temp[i] = _copy_output(x[i])::Tx
        end
    end
    return temp::P
end

# Tuple, NamedTuple
_copy_output(x::Union{Tuple,NamedTuple}) = map(_copy_output, x)::typeof(x)

# mutable composite types, bitstype
function _copy_output(x::P) where {P}
    isbitstype(P) && return x
    nf = nfields(P)

    if ismutable(x)
        _copy_output_mutable_cartesian(x, Val(nf))
    else
        _copy_output_immutable_cartesian(x, Val(nf))
    end
end

@generated function _copy_output_mutable_cartesian(x::P, ::Val{nf}) where {P,nf}
    quote
        temp = ccall(:jl_new_struct_uninit, Any, (Any,), P)::P
        Base.Cartesian.@nexprs(
            $nf,
            i -> if isdefined(x, i)
                ccall(
                    :jl_set_nth_field,
                    Cvoid,
                    (Any, Csize_t, Any),
                    temp,
                    i - 1,
                    _copy_output(getfield(x, i)),
                )
            end
        )
        return temp::P
    end
end

@generated function _copy_output_immutable_cartesian(x::P, ::Val{nf}) where {P,nf}
    quote
        Base.Cartesian.@nif(
            $(nf + 1),
            # Assumes if a undefined field is found, all subsequent fields are undefined.
            i -> !isdefined(x, i),
            i -> _copy_output_immutable_cartesian_upto(x, Val(i - 1)),
        )
    end
end
@generated function _copy_output_immutable_cartesian_upto(x::P, ::Val{idx}) where {P,idx}
    idx == 0 && return :(x)
    return quote
        flds = collect(Any, Base.Cartesian.@ntuple($idx, i -> _copy_output(getfield(x, i))))
        # when immutable struct object created by non initializing inner constructor. (Base.deepcopy misses this out)
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), P, flds, $idx)::P
    end
end

function __exclude_unsupported_output_internal!(
    y::T, address_set::Set{UInt}
) where {T<:_BuiltinArrays}
    if objectid(y) in address_set
        throw_circular_reference_or_alias_error(y)
    end

    # mutable types are always stored on the heap.
    push!(address_set, objectid(y))

    # recurse over iterable collections.
    for i in eachindex(y)
        # isassigned() is valid for Arrays, Memory.
        !isassigned(y, i) && continue
        __exclude_unsupported_output_internal!(y[i], address_set)
    end

    return nothing
end

function __exclude_unsupported_output_internal!(
    y::Union{Tuple,NamedTuple}, address_set::Set{UInt}
)
    map(Base.Fix2(__exclude_unsupported_output_internal!, address_set), y)
    return nothing
end

# in case f(args...) directly outputs a Ptr{T} or it contains a nested Ptr{T}.
function __exclude_unsupported_output_internal!(y::Ptr, ::Set{UInt})
    return throw_forward_ret_type_error(y)
end

"""
    prepare_pullback_cache(f, x...)

Returns a cache used with [`value_and_pullback!!`](@ref). See that function for more info.
"""
@unstable function prepare_pullback_cache(fx...; kwargs...)

    # Check that the output of `fx` is supported.
    __exclude_func_with_unsupported_output(fx)

    # Construct rule and tangents.
    rule = build_rrule(get_interpreter(), Tuple{map(_typeof, fx)...}; kwargs...)
    tangents = map(zero_tangent, fx)

    # Run the rule forwards -- this should do a decent chunk of pre-allocation.
    y, rvs!! = rule(map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents)...)

    # Run reverse-pass in order to reset stacks + state.
    rvs!!(zero_rdata(primal(y)))

    # Construct cache for output. Check that `_copy_to_output!!`ing appears to work.
    y_cache = _copy_output(primal(y))
    return Cache(rule, _copy_to_output!!(y_cache, primal(y)), tangents)
end

"""
    value_and_pullback!!(cache::Cache, ȳ, f, x...)

!!! info
    If `f(x...)` returns a scalar, you should use [`value_and_gradient!!`](@ref), not this
    function.

Computes a 2-tuple. The first element is `f(x...)`, and the second is a tuple containing the
pullback of `f` applied to `ȳ`. The first element is the component of the pullback
associated to any fields of `f`, the second w.r.t the first element of `x`, etc.

There are no restrictions on what `y = f(x...)` is permitted to return. However, `ȳ` must be
an acceptable tangent for `y`. This means that, for example, it must be true that
`tangent_type(typeof(y)) == typeof(ȳ)`.

As with all functionality in Mooncake, if `f` modifes itself or `x`, `value_and_gradient!!`
will return both to their original state as part of the process of computing the gradient.

!!! info
    `cache` must be the output of [`prepare_pullback_cache`](@ref), and (fields of) `f` and
    `x` must be of the same size and shape as those used to construct the `cache`. This is
    to ensure that the gradient can be written to the memory allocated when the `cache` was
    built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable
    components of values returned by it will be mutated if you run this function again with
    different arguments. Therefore, if you need to keep the values returned by this function
    around over multiple calls to this function with the same `cache`, you should take a
    copy (using `copy` or `deepcopy`) of them before calling again.

# Example Usage
```jldoctest
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
cache = Mooncake.prepare_pullback_cache(f, x, y)
Mooncake.value_and_pullback!!(cache, 1.0, f, x, y)

# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
function value_and_pullback!!(cache::Cache, ȳ, f::F, x::Vararg{Any,N}) where {F,N}
    tangents = tuple_map(set_to_zero!!, cache.tangents)
    coduals = tuple_map(CoDual, (f, x...), tangents)
    return __value_and_pullback!!(cache.rule, ȳ, coduals...; y_cache=cache.y_cache)
end

"""
    prepare_gradient_cache(f, x...)

Returns a cache used with [`value_and_gradient!!`](@ref). See that function for more info.
"""
@unstable function prepare_gradient_cache(fx...; kwargs...)
    rule = build_rrule(fx...; kwargs...)
    tangents = map(zero_tangent, fx)
    y, rvs!! = rule(map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents)...)
    primal(y) isa IEEEFloat || throw_val_and_grad_ret_type_error(primal(y))
    rvs!!(zero_tangent(primal(y))) # run reverse-pass to reset stacks + state
    return Cache(rule, nothing, tangents)
end

"""
    value_and_gradient!!(cache::Cache, f, x...)

Computes a 2-tuple. The first element is `f(x...)`, and the second is a tuple containing the
gradient of `f` w.r.t. each argument. The first element is the gradient w.r.t any
differentiable fields of `f`, the second w.r.t the first element of `x`, etc.

Assumes that `f` returns a `Union{Float16, Float32, Float64}`.

As with all functionality in Mooncake, if `f` modifes itself or `x`, `value_and_gradient!!`
will return both to their original state as part of the process of computing the gradient.

!!! info
    `cache` must be the output of [`prepare_gradient_cache`](@ref), and (fields of) `f` and
    `x` must be of the same size and shape as those used to construct the `cache`. This is
    to ensure that the gradient can be written to the memory allocated when the `cache` was
    built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable
    components of values returned by it will be mutated if you run this function again with
    different arguments. Therefore, if you need to keep the values returned by this function
    around over multiple calls to this function with the same `cache`, you should take a
    copy (using `copy` or `deepcopy`) of them before calling again.

# Example Usage
```jldoctest
f(x, y) = sum(x .* y)
x = [2.0, 2.0]
y = [1.0, 1.0]
cache = prepare_gradient_cache(f, x, y)
value_and_gradient!!(cache, f, x, y)

# output

(4.0, (NoTangent(), [1.0, 1.0], [2.0, 2.0]))
```
"""
function value_and_gradient!!(cache::Cache, f::F, x::Vararg{Any,N}) where {F,N}
    coduals = tuple_map(CoDual, (f, x...), tuple_map(set_to_zero!!, cache.tangents))
    return __value_and_gradient!!(cache.rule, coduals...)
end
