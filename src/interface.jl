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
    out, pb!! = __call_rule(rule, fx_fwds)
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

@static if VERSION < v"1.11-"
    # rrule!! is a plain Julia function (not an OpaqueClosure), so calling it directly is
    # safe on Julia 1.10; the inferencebarrier workaround is not needed here.
    @inline __call_rule(rule::typeof(rrule!!), args) = rule(args...)
end

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
    out, pb!! = __call_rule(rule, fx_fwds)
    y = primal(out)
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    return y, tuple_map((f, r) -> tangent(fdata(tangent(f)), r), fx, pb!!(one(y)))
end

"""
    value_and_pullback!!(rule, ȳ, f, x...; friendly_tangents=false)

Compute the value and pullback of `f(x...)`. If `friendly_tangents=false`,
`ȳ` must be a valid tangent for the primal return by `f(x...)`.
If `friendly_tangents=true`, `ȳ` must be of the same type as the primal returned by `f(x...)`.

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
# Returns NoCache when all primals are bits types (no mutable aliasing possible).
# Otherwise returns IdDict to handle aliased mutable buffers across the tuple of tangents.
_friendly_cache(fx::Tuple) = all(isbitstype ∘ typeof, fx) ? NoCache() : IdDict{Any,Any}()

# @inline forces specialisation on Vararg with function-valued arguments, avoiding severe
# perf regressions. See https://github.com/chalk-lab/Mooncake.jl/issues/1020.
@inline function value_and_pullback!!(
    rule::R, ȳ, fx::Vararg{Any,N}; friendly_tangents=false
) where {R,N}
    friendly_tangents && return _value_and_pullback_friendly!!(rule, ȳ, fx...)
    return __value_and_pullback!!(rule, ȳ, __create_coduals(fx)...)
end

@unstable function _value_and_pullback_friendly!!(
    rule::R, ȳ, fx::Vararg{Any,N}
) where {R,N}
    ȳ = primal_to_tangent!!(zero_tangent(ȳ), ȳ)
    value, pb = __value_and_pullback!!(rule, ȳ, __create_coduals(fx)...)
    dests = map(friendly_tangent_cache, (fx...,))
    c = _friendly_cache((fx...,))
    friendly_pb = tuple_map(
        (d, p, t) -> tangent_to_friendly!!(d, p, t, c), dests, (fx...,), pb
    )
    return value, friendly_pb
end

"""
    value_and_gradient!!(rule, f, x...; friendly_tangents=false)

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
@inline function value_and_gradient!!(
    rule::R, fx::Vararg{Any,N}; friendly_tangents=false
) where {R,N}
    friendly_tangents && return _value_and_gradient_friendly!!(rule, fx...)
    return __value_and_gradient!!(rule, __create_coduals(fx)...)
end

@unstable function _value_and_gradient_friendly!!(rule::R, fx::Vararg{Any,N}) where {R,N}
    value, gradient = __value_and_gradient!!(rule, __create_coduals(fx)...)
    dests = map(friendly_tangent_cache, (fx...,))
    c = _friendly_cache((fx...,))
    friendly_gradient = tuple_map(
        (d, p, t) -> tangent_to_friendly!!(d, p, t, c), dests, (fx...,), gradient
    )
    return value, friendly_gradient
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

struct Cache{Trule,Ty_cache,Ttangents<:Tuple,Tdests,Tȳ_cache,TIS<:Tuple}
    rule::Trule
    # Cache for function output; **primal** type for y.
    y_cache::Ty_cache
    # Cache for internal gradient representation; **tangent** type for (f, x...)
    tangents::Ttangents
    # Pre-allocated friendly-tangent dest tree for (f, x...), built by
    # map(friendly_tangent_cache, fx).  `nothing` when friendly_tangents=false.
    dests::Tdests
    # Cache to convert from friendly to internal representation of ȳ.
    # Tangent type for y, i.e. this is a **tangent** type for y.
    ȳ_cache::Tȳ_cache
    # Top-level type/size signature for (f, x...), used to reject cache misuse early.
    input_specs::TIS
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
    return __exclude_unsupported_output(_y)
end

"""
    __exclude_unsupported_output_internal(y::T, address_set::Set{UInt}) where {T}

For checking if output`y` is a valid Mutable/immutable composite or a primitive type.
Performs a recursive depth first search over the function output `y` with an `isbitstype()` check base case. The visited memory addresses are stored inside `address_set`.
If the set already contains a newly visited address, it errors out indicating an Alias or Circular reference.
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
For types with custom copy semantics, overload this function (see `Core.SimpleVector` for an example).
"""
_copy_to_output!!(dst::Number, src::Number) = src

# Type values (DataType, UnionAll, Union), Core.TypeName, and Modules
# cannot be deep-copied; return src as-is.
_copy_to_output!!(::Type, src::Type) = src
_copy_to_output!!(::Core.TypeName, src::Core.TypeName) = src
_copy_to_output!!(::Module, src::Module) = src

# explicit copy for Core.svec
function _copy_to_output!!(dst::SimpleVector, src::SimpleVector)
    return Core.svec(map(_copy_to_output!!, dst, src)...)
end

# copy for Array, Memory
function _copy_to_output!!(dst::P, src::P) where {P<:_BuiltinArrays}
    @inbounds for i in eachindex(src)
        if isassigned(src, i)
            dst[i] = if isassigned(dst, i)
                _copy_to_output!!(dst[i], src[i])
            else
                _copy_output(src[i])
            end
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
    # nfields(src) not nfields(P): the latter counts fields of the
    # DataType object itself.
    nf = nfields(src)

    # No Julia-visible fields (e.g. Symbol, String): nothing to update.
    # Overload _copy_to_output!! to customise.
    nf == 0 && return src

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

# fallback for invalid type combinations
function _copy_to_output!!(dst::T, src::P) where {T,P}
    throw(
        ArgumentError(
            "Mooncake.jl does not currently have a method " *
            "`_copy_to_output!!` to handle this type combination: " *
            "dst passed is of type $T, while src is a $P. " *
            "This often happens when differentiating over " *
            "non-differentiable types (e.g. integers or booleans).",
        ),
    )
end

"""
    _copy_output(x::T)

Returns a copy of `x`, of the same type `T`. Allocates new memory for the copy.
Required as Base.copy() does not work for all supported primal types. For example, `Base.copy` does not work for `Core.svec`.
For types with custom copy semantics, overload this function (see `Core.SimpleVector` for an example).
"""
# Type values (DataType, UnionAll, Union), Core.TypeName, and Modules
# cannot be deep-copied; return x as-is.
@unstable _copy_output(x::Type) = x
_copy_output(x::Core.TypeName) = x
_copy_output(x::Module) = x

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
    # nfields(x) not nfields(P): the latter counts fields of the
    # DataType object itself.
    nf = nfields(x)

    # No Julia-visible fields (e.g. Symbol, String): nothing to copy.
    # Overload _copy_output to customise.
    nf == 0 && return x

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
    prepare_pullback_cache(f, x...; config=Mooncake.Config())

Returns a cache used with [`value_and_pullback!!`](@ref). See that function for more info.

The API guarantees that tangents are initialized at zero before the first autodiff pass.
"""
@unstable function prepare_pullback_cache(fx...; config=Config())

    # Check that the output of `fx` is supported.
    __exclude_func_with_unsupported_output(fx)

    # Construct rule and tangents.
    interp = get_interpreter(ReverseMode)
    rule = build_rrule(
        interp, Tuple{map(_typeof, fx)...}; config.debug_mode, config.silence_debug_messages
    )
    tangents = map(zero_tangent, fx)

    # Run the rule forwards -- this should do a decent chunk of pre-allocation.
    y, rvs!! = __call_rule(rule, map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents))

    # Run reverse-pass in order to reset stacks + state.
    rvs!!(zero_rdata(primal(y)))

    # Construct cache for output. Check that `_copy_to_output!!`ing appears to work.
    y_cache = _copy_output(primal(y))
    y_cache = _copy_to_output!!(y_cache, primal(y))
    if config.friendly_tangents
        dests = map(friendly_tangent_cache, fx)
        return Cache(
            rule,
            y_cache,
            tangents,
            dests,
            zero_tangent(primal(y)),
            map(_prepared_cache_input_spec, fx),
        )
    else
        return Cache(
            rule, y_cache, tangents, nothing, nothing, map(_prepared_cache_input_spec, fx)
        )
    end
end

"""
    value_and_pullback!!(cache::Cache, ȳ, f, x...; args_to_zero=(true, ...))

!!! info
    If `f(x...)` returns a scalar, you should use [`value_and_gradient!!`](@ref), not this
    function.

Computes a 2-tuple. The first element is `f(x...)`, and the second is a tuple containing the
pullback of `f` applied to `ȳ`. The first element is the component of the pullback
associated to any fields of `f`, the second w.r.t the first element of `x`, etc.
If the cache was prepared with `config.friendly_tangents=true`, the pullback uses the same types as
those of `f` and `x`. Otherwise, it uses the tangent types associated to `f` and `x`.

There are no restrictions on what `y = f(x...)` is permitted to return. However, `ȳ` must be
an acceptable tangent for `y`. If the cache was prepared with `config.friendly_tangents=false`,
this means that, for example, it must be true that `tangent_type(typeof(y)) == typeof(ȳ)`.
If the cache was prepared with `config.friendly_tangents=true`, then `typeof(y) == typeof(ȳ)`.

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

The keyword argument `args_to_zero` is a tuple of boolean values specifying which cotangents should be reset to zero before differentiation.
It contains one boolean for each element of `(f, x...)`.
It is used for performance optimizations if you can guarantee that the initial cotangent allocated in `cache` (created by `zero_tangent`) never needs to be zeroed out again.

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
@inline function value_and_pullback!!(
    cache::Cache,
    ȳ,
    f::F,
    x::Vararg{Any,N};
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    _prepared_cache_check_specs(cache.input_specs, (f, x...))
    tangents = tuple_map(set_to_zero_maybe!!, cache.tangents, args_to_zero)
    coduals = tuple_map(CoDual, (f, x...), tangents)
    if !isnothing(cache.dests)
        ȳ = primal_to_tangent!!(cache.ȳ_cache, ȳ)
        value, pb = __value_and_pullback!!(
            cache.rule, ȳ, coduals...; y_cache=cache.y_cache
        )
        c = _friendly_cache((f, x...))
        friendly_pb = tuple_map(
            (d, p, t) -> tangent_to_friendly!!(d, p, t, c), cache.dests, (f, x...), pb
        )
        return value, friendly_pb
    else
        return __value_and_pullback!!(cache.rule, ȳ, coduals...; y_cache=cache.y_cache)
    end
end

"""
    prepare_gradient_cache(f, x...; config=Mooncake.Config())

Returns a cache used with [`value_and_gradient!!`](@ref). See that function for more info.

The API guarantees that tangents are initialized at zero before the first autodiff pass.
"""
@unstable function prepare_gradient_cache(fx...; config=Config())
    rule = build_rrule(fx...; config.debug_mode, config.silence_debug_messages)
    tangents = map(zero_tangent, fx)
    y, rvs!! = __call_rule(rule, map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents))
    primal(y) isa IEEEFloat || throw_val_and_grad_ret_type_error(primal(y))
    rvs!!(zero_tangent(primal(y))) # run reverse-pass to reset stacks + state
    if config.friendly_tangents
        dests = tuple(map(friendly_tangent_cache, fx)...)
        return Cache(
            rule, nothing, tangents, dests, nothing, map(_prepared_cache_input_spec, fx)
        )
    else
        return Cache(
            rule, nothing, tangents, nothing, nothing, map(_prepared_cache_input_spec, fx)
        )
    end
end

"""
    value_and_gradient!!(cache::Cache, f, x...; args_to_zero=(true, ...))

Computes a 2-tuple. The first element is `f(x...)`, and the second is a tuple containing the
gradient of `f` w.r.t. each argument. The first element is the gradient w.r.t any
differentiable fields of `f`, the second w.r.t the first element of `x`, etc.
If the cache was prepared with `config.friendly_tangents=true`, the pullback uses the same types as
those of `f` and `x`. Otherwise, it uses the tangent types associated to `f` and `x`.

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

The keyword argument `args_to_zero` is a tuple of boolean values specifying which cotangents should be reset to zero before differentiation.
It contains one boolean for each element of `(f, x...)`.
It is used for performance optimizations if you can guarantee that the initial cotangent allocated in `cache` (created by `zero_tangent`) never needs to be zeroed out again.

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
@inline function value_and_gradient!!(
    cache::Cache,
    f::F,
    x::Vararg{Any,N};
    args_to_zero::NTuple=ntuple(Returns(true), Val(N + 1)),
) where {F,N}
    friendly_tangents = !isnothing(cache.dests)
    _prepared_cache_check_specs(cache.input_specs, (f, x...))

    tangents = tuple_map(set_to_zero_maybe!!, cache.tangents, args_to_zero)
    coduals = tuple_map(CoDual, (f, x...), tangents)
    if friendly_tangents
        value, gradient = __value_and_gradient!!(cache.rule, coduals...)
        c = _friendly_cache((f, x...))
        friendly_gradient = tuple_map(
            (d, p, t) -> tangent_to_friendly!!(d, p, t, c), cache.dests, (f, x...), gradient
        )
        return value, friendly_gradient
    else
        return __value_and_gradient!!(cache.rule, coduals...)
    end
end

struct ForwardChunkFastPath{FR,PB,GR,SG,SB,SW}
    frules::FR
    pack_buffers::PB
    gradient_rrule::GR
    small_vector_gradient_frule::SG
    small_vector_gradient_buffer::SB
    small_vector_gradient_workspace::SW
end

struct ForwardCache{R,IT<:Union{Nothing,Tuple},OP,FG,GW,CF,S<:Tuple}
    rule::R
    input_tangents::IT
    output_primal::OP
    friendly_gradients::FG
    gradient_workspace::GW
    chunk_fastpath::CF
    input_specs::S
end

"""
    NTangent(lanes)

Explicit wrapper for chunked forward-mode tangents at the interface boundary.

Each element of `lanes` must itself be a valid width-1 tangent in the corresponding API
mode. Mooncake repacks chunked results in another `NTangent`, and uses an NDual-backed
single-pass fast path when the runtime values fit `nfwd`'s supported primal space. If the
fast path hits an NDual-specific runtime limitation, Mooncake falls back to ordinary width-1
forward evaluation.
"""
struct NTangent{L<:Tuple}
    lanes::L
end

Base.length(x::NTangent) = length(x.lanes)
Base.getindex(x::NTangent, i::Int) = x.lanes[i]
Base.iterate(x::NTangent, st...) = iterate(x.lanes, st...)

@inline _chunk_lane(x, ::Int) = x
@inline _chunk_lane(x::NTangent, lane::Int) = x[lane]
@inline _chunk_lane(x, ::Val) = x
@inline _chunk_lane(x::NTangent, ::Val{lane}) where {lane} = x[lane]

@inline _has_chunk_tangent(::Tuple{}) = false
@inline function _has_chunk_tangent(fx::Tuple)
    return tangent(first(fx)) isa NTangent || _has_chunk_tangent(Base.tail(fx))
end

# Bug fix note: chunked forward can return `NoTangent()` lanes for nondifferentiable outputs,
# and the generic `_copy` fallback does not support `NoTangent`.
@inline _copy_chunk_tangent_output(x::NoTangent) = x
@inline _copy_chunk_tangent_output(x) = _copy(x)

const _CHUNK_NFWD_MAX_LANES = 8

@inline function _prepared_cache_input_spec(x)
    return x isa AbstractArray ? (type=typeof(x), size=size(x)) : (type=typeof(x), size=())
end

@inline _prepared_cache_arg_label(i::Int) = i == 1 ? "`f`" : "`x$(i - 1)`"

function _throw_prepared_cache_spec_error(kind::Symbol, i::Int, expected, got)
    label = _prepared_cache_arg_label(i)
    msg = if kind === :arity
        "Cached autodiff call expected $(expected) total arguments `(f, x...)`, got $(got). " *
        "Prepared pullback, gradient, derivative, HVP, and Hessian caches must be reused " *
        "with the same top-level argument structure they were prepared with."
    elseif kind === :type
        "Cached autodiff call has a type mismatch for $label: expected $expected, got $got. " *
        "Prepared pullback, gradient, derivative, HVP, and Hessian caches must be reused " *
        "with the same top-level argument types they were prepared with."
    else
        "Cached autodiff call has a size mismatch for $label: expected $expected, got $got. " *
        "Prepared pullback, gradient, derivative, HVP, and Hessian caches must be reused " *
        "with the same top-level array sizes they were prepared with."
    end
    throw(ArgumentError(msg))
end

@inline function _prepared_cache_check_spec(i::Int, spec, x)
    typeof(x) == spec.type ||
        _throw_prepared_cache_spec_error(:type, i, spec.type, typeof(x))
    x isa AbstractArray || return nothing
    size(x) == spec.size || _throw_prepared_cache_spec_error(:size, i, spec.size, size(x))
    return nothing
end

@inline _prepared_cache_check_specs_impl(_i::Int, ::Tuple{}, ::Tuple{}) = nothing
@inline function _prepared_cache_check_specs_impl(
    i::Int, specs::Tuple{Any,Vararg{Any}}, fx::Tuple{Any,Vararg{Any}}
)
    _prepared_cache_check_spec(i, first(specs), first(fx))
    return _prepared_cache_check_specs_impl(i + 1, Base.tail(specs), Base.tail(fx))
end

@inline function _prepared_cache_check_specs(specs::Tuple, fx::Tuple)
    length(specs) == length(fx) ||
        _throw_prepared_cache_spec_error(:arity, 0, length(specs), length(fx))
    _prepared_cache_check_specs_impl(1, specs, fx)
    return nothing
end

@inline function _prepared_cache_check_specs(specs::Tuple{Any,Any}, f, x)
    _prepared_cache_check_spec(1, specs[1], f)
    _prepared_cache_check_spec(2, specs[2], x)
    return nothing
end

@inline _forward_cache_should_track(x::AbstractArray) = true
@inline _forward_cache_should_track(x::P) where {P} = Base.ismutabletype(P)

@inline function _forward_cache_mark_seen!(seen::IdDict{Any,Any}, x)
    haskey(seen, x) && return true
    seen[x] = nothing
    return false
end

@inline function _forward_cache_lookup_seed(dict::IdDict{Any,Any}, x)
    return get(dict, x, nothing)
end

@inline function _forward_cache_store_seed!(dict::IdDict{Any,Any}, x, dx)
    dict[x] = dx
    return dx
end

@noinline function _throw_forward_cache_uninit_field_error(::Type{P}, n::Int) where {P}
    throw(
        ArgumentError(
            "Forward-mode gradient seeding encountered an undefined field " *
            "`$(fieldname(P, n))` in a value of type `$P`, but that field is marked " *
            "always-initialised. This object is in a partially initialised state that " *
            "Mooncake cannot seed automatically.",
        ),
    )
end

# Bug fix note: forward-cache gradient seeding must walk the whole input tuple with an
# identity cache, otherwise aliased mutable subobjects are over-counted and cycles recurse
# forever.
@inline _forward_cache_input_dof(x) = _forward_cache_input_dof(x, IdDict{Any,Any}())
@inline _forward_cache_input_dof(::NoTangent, _seen::IdDict{Any,Any}) = 0
@inline _forward_cache_input_dof(x::IEEEFloat, _seen::IdDict{Any,Any}) = 1
@inline _forward_cache_input_dof(x::Complex{<:IEEEFloat}, _seen::IdDict{Any,Any}) = 2
@inline function _forward_cache_input_dof(
    x::AbstractArray{<:IEEEFloat}, seen::IdDict{Any,Any}
)
    _forward_cache_mark_seen!(seen, x) && return 0
    return length(x)
end
@inline function _forward_cache_input_dof(
    x::AbstractArray{Complex{<:IEEEFloat}}, seen::IdDict{Any,Any}
)
    _forward_cache_mark_seen!(seen, x) && return 0
    return 2 * length(x)
end
@inline function _forward_cache_input_dof(x::AbstractArray, seen::IdDict{Any,Any})
    tangent_type(typeof(x)) == NoTangent && return 0
    _forward_cache_mark_seen!(seen, x) && return 0
    total = 0
    for xi in x
        total += _forward_cache_input_dof(xi, seen)
    end
    return total
end
@inline function _forward_cache_input_dof(x::Tuple, seen::IdDict{Any,Any})
    total = 0
    for xi in x
        total += _forward_cache_input_dof(xi, seen)
    end
    return total
end
@inline function _forward_cache_input_dof(x::NamedTuple, seen::IdDict{Any,Any})
    total = 0
    for xi in values(x)
        total += _forward_cache_input_dof(xi, seen)
    end
    return total
end
@inline function _forward_cache_input_dof(x::P, seen::IdDict{Any,Any}) where {P}
    tangent_type(P) == NoTangent && return 0
    _forward_cache_should_track(x) && _forward_cache_mark_seen!(seen, x) && return 0
    total = 0
    inits = always_initialised(P)
    for n in 1:fieldcount(P)
        if isdefined(x, n)
            total += _forward_cache_input_dof(getfield(x, n), seen)
        elseif inits[n]
            _throw_forward_cache_uninit_field_error(P, n)
        end
    end
    return total
end

@inline _forward_cache_seed_tangent(x, slot::Int) = _forward_cache_seed_tangent(
    x, slot, Ref(0), IdDict{Any,Any}()
)
@inline _forward_cache_seed_tangent(::NoTangent, _slot::Int, _cursor, _dict) = NoTangent()
@inline function _forward_cache_seed_tangent(
    ::NoTangent, _slot::Int, _cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    return NoTangent()
end
@inline function _forward_cache_seed_tangent(
    x::IEEEFloat, slot::Int, cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    cursor[] += 1
    return cursor[] == slot ? one(x) : zero(x)
end
@inline function _forward_cache_seed_tangent(
    x::Complex{T}, slot::Int, cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
) where {T<:IEEEFloat}
    cursor[] += 1
    real_part = cursor[] == slot ? one(T) : zero(T)
    cursor[] += 1
    imag_part = cursor[] == slot ? one(T) : zero(T)
    return complex(real_part, imag_part)
end

function _forward_cache_seed_tangent(
    x::AbstractArray{T}, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {T<:IEEEFloat}
    existing = _forward_cache_lookup_seed(dict, x)
    !isnothing(existing) && return existing
    dx = _forward_cache_store_seed!(dict, x, zero_tangent(x))
    @inbounds for I in eachindex(x)
        cursor[] += 1
        dx[I] = cursor[] == slot ? one(T) : zero(T)
    end
    return dx
end

function _forward_cache_seed_tangent(
    x::AbstractArray{Complex{T}},
    slot::Int,
    cursor::Base.RefValue{Int},
    dict::IdDict{Any,Any},
) where {T<:IEEEFloat}
    existing = _forward_cache_lookup_seed(dict, x)
    !isnothing(existing) && return existing
    dx = _forward_cache_store_seed!(dict, x, zero_tangent(x))
    @inbounds for I in eachindex(x)
        cursor[] += 1
        real_part = cursor[] == slot ? one(T) : zero(T)
        cursor[] += 1
        imag_part = cursor[] == slot ? one(T) : zero(T)
        dx[I] = complex(real_part, imag_part)
    end
    return dx
end

function _forward_cache_seed_tangent(
    x::AbstractArray, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    existing = _forward_cache_lookup_seed(dict, x)
    !isnothing(existing) && return existing
    dx = _forward_cache_store_seed!(dict, x, zero_tangent(x))
    for I in eachindex(x)
        dx[I] = _forward_cache_seed_tangent(x[I], slot, cursor, dict)
    end
    return dx
end

@inline function _forward_cache_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:Tuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(
        n -> _forward_cache_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P))
    )
    return build_tangent(P, fields...)
end

@inline function _forward_cache_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:NamedTuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(
        n -> _forward_cache_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P))
    )
    return build_tangent(P, fields...)
end

function _forward_cache_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P}
    tangent_type(P) == NoTangent && return NoTangent()
    if _forward_cache_should_track(x)
        existing = _forward_cache_lookup_seed(dict, x)
        !isnothing(existing) && return existing
        tx = _forward_cache_store_seed!(dict, x, zero_tangent(x))
        if tx isa MutableTangent
            inits = always_initialised(P)
            for n in 1:fieldcount(P)
                if isdefined(x, n)
                    set_tangent_field!(
                        tx,
                        n,
                        _forward_cache_seed_tangent(getfield(x, n), slot, cursor, dict),
                    )
                elseif inits[n]
                    _throw_forward_cache_uninit_field_error(P, n)
                end
            end
        end
        return tx
    end

    inits = always_initialised(P)
    fields = ntuple(Val(fieldcount(P))) do n
        if isdefined(x, n)
            return _forward_cache_seed_tangent(getfield(x, n), slot, cursor, dict)
        elseif inits[n]
            _throw_forward_cache_uninit_field_error(P, n)
        else
            return PossiblyUninitTangent{tangent_type(fieldtype(P, n))}()
        end
    end
    return build_tangent(P, fields...)
end

@inline function _forward_cache_seed_input_tuple(input_primals::Tuple, slot::Int)
    return _forward_cache_seed_tangent(input_primals, slot)
end

function _forward_cache_seed_chunk_inputs(
    input_primals::Tuple, start_slot::Int, chunk_width::Int
)
    lane_tangents = ntuple(
        lane -> _forward_cache_seed_input_tuple(input_primals, start_slot + lane - 1),
        chunk_width,
    )
    return ntuple(
        i -> NTangent(ntuple(lane -> lane_tangents[lane][i], chunk_width)),
        Val(fieldcount(typeof(input_primals))),
    )
end

@inline function _forward_cache_accumulate_lane(grad, coeff, lane_tangent)
    lane_tangent isa NoTangent && return grad
    return increment!!(grad, _scale(coeff, lane_tangent))
end

function _forward_cache_accumulate_chunk!(
    gradients::Tuple, input_tangents::Tuple, output_tangent::NTangent
)
    for lane in 1:length(output_tangent)
        coeff = Float64(output_tangent[lane])
        gradients = tuple_map(
            (g, dx) -> _forward_cache_accumulate_lane(g, coeff, dx[lane]),
            gradients,
            input_tangents,
        )
    end
    return gradients
end

@inline function _forward_cache_output_gradient(
    cache::ForwardCache, input_primals::Tuple, native_gradients::Tuple
)
    if isnothing(cache.input_tangents)
        return native_gradients
    else
        friendly_gradients = _copy_to_output!!(cache.friendly_gradients, input_primals)
        return tangent_to_primal!!(friendly_gradients, native_gradients)
    end
end

@inline function _forward_cache_finish_value_and_gradient(
    cache::ForwardCache, input_primals::Tuple, y, native_gradients::Tuple
)
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    return y, _forward_cache_output_gradient(cache, input_primals, native_gradients)
end

function _forward_cache_gradient_workspace!(cache::ForwardCache, input_primals::Tuple)
    workspace = cache.gradient_workspace[]
    if isnothing(workspace)
        workspace = tuple_map(zero_tangent, input_primals)
        cache.gradient_workspace[] = workspace
        return workspace::Tuple
    end
    zeroed = tuple_map(set_to_zero!!, workspace)
    cache.gradient_workspace[] = zeroed
    return zeroed::Tuple
end

function _forward_cache_array_gradient_workspace!(cache::ForwardCache, x::Array)
    workspace = cache.gradient_workspace[]
    if isnothing(workspace)
        workspace = (NoTangent(), zero_tangent(x))
        cache.gradient_workspace[] = workspace
        return workspace
    end
    set_to_zero!!(workspace[2])
    return workspace
end

@inline function _forward_cache_gradient_rrule_fastpath(
    cache::ForwardCache, fastpath::ForwardChunkFastPath, f, x::Array
)
    native_gradients = _forward_cache_array_gradient_workspace!(cache, x)
    y, output = __value_and_gradient!!(
        fastpath.gradient_rrule,
        CoDual(f, native_gradients[1]),
        CoDual(x, native_gradients[2]),
    )
    return _forward_cache_finish_value_and_gradient(cache, (f, x), y, output)
end

@generated function _forward_cache_gradient_workspace_ref(::Type{T}) where {T<:Tuple}
    tangent_types = map(P -> :(tangent_type($P)), T.parameters)
    workspace_type = Expr(:curly, :Tuple, tangent_types...)
    # Bug fix note: keep the lazy gradient workspace concretely typed even before first use,
    # otherwise `Ref{Any}` makes cached forward gradients inference-opaque.
    return :(Ref{Union{Nothing,$workspace_type}}(nothing))
end

@inline _chunk_can_use_nfwd(::Type{<:IEEEFloat}) = true
@inline _chunk_can_use_nfwd(::Type{<:Complex{<:IEEEFloat}}) = true
@inline _chunk_can_use_nfwd(::Type{<:Array{ET}}) where {ET} = Nfwd._nfwd_is_supported_scalar(
    ET
)
@inline _chunk_can_use_nfwd(::Type{<:Tuple}) = false
@inline _chunk_can_use_nfwd(::Type) = false

@inline _chunk_can_use_nfwd_gradient(::Type) = false
@inline _chunk_can_use_nfwd_gradient(::Type{<:Array{<:IEEEFloat}}) = true
@inline _chunk_can_use_nfwd_small_vector_gradient(::Type) = false
@inline _chunk_can_use_nfwd_small_vector_gradient(::Type{<:Vector{<:IEEEFloat}}) = true

function _chunk_fastpath_rule_supported(rule, fx::Tuple, x_tangents::Tuple)
    fd = Dual(first(fx), NoTangent())
    x_duals = tuple_map(Dual, Base.tail(fx), x_tangents)
    try
        rule(fd, x_duals...)
    catch err
        _chunk_should_fallback_to_lane_loop(err) && return false
        rethrow(err)
    end
    return true
end

function _chunk_fastpath_supported(fx::Tuple, frules::Tuple)
    x_tangents = tuple_map(zero_tangent, Base.tail(fx))
    return _chunk_fastpath_rule_supported(frules[1], fx, x_tangents)
end

@inline function _maybe_build_chunk_fastpath(fx::Tuple, config)
    config.debug_mode && return nothing
    sig = typeof(fx)
    params = Tuple(sig.parameters)
    F = params[1]
    Base.issingletontype(F) || return nothing
    # Current NDual fast-path boundary: only scalar/complex/array top-level primals are
    # packed here. Tuple-like or structured top-level primals stay on the ordinary lane
    # loop until the chunk-aware IR frontend can repack them soundly.
    all(_chunk_can_use_nfwd, Base.tail(params)) || return nothing
    frules = ntuple(
        n -> NfwdMooncake.build_frule(
            sig; chunk_size=n, debug_mode=false, silence_debug_messages=true
        ),
        Val(_CHUNK_NFWD_MAX_LANES),
    )
    pack_buffers = tuple_map(_chunk_pack_buffer_template, Base.tail(fx))
    # Bug fix note: keep the cached nfwd gradient fast path narrow. The generic
    # `NfwdMooncake.Cache` gradient entrypoint is not a win for multi-argument scalar calls, where
    # the chunked forward frontend already gets the full gradient in one NDual pass.
    gradient_rrule = if length(params) == 2 && _chunk_can_use_nfwd_gradient(params[2])
        NfwdMooncake.build_rrule(
            sig; chunk_size=nothing, debug_mode=false, silence_debug_messages=true
        )
    else
        nothing
    end
    # Small vectors are faster through an exact-width NDual frule than through the generic
    # array gradient rrule, which defaults to width 8 and leaves a fixed overhead at n <= 8.
    small_vector_gradient_frule =
        if (
            length(params) == 2 &&
            _chunk_can_use_nfwd_small_vector_gradient(params[2]) &&
            1 <= length(last(fx)) <= _CHUNK_NFWD_MAX_LANES
        )
            frules[length(last(fx))]
        else
            nothing
        end
    if !isnothing(small_vector_gradient_frule)
        exact_width_tangents = (
            _forward_cache_small_vector_gradient_buffer(last(fx), Val(length(last(fx)))),
        )
        if !_chunk_fastpath_rule_supported(
            small_vector_gradient_frule, fx, exact_width_tangents
        )
            small_vector_gradient_frule = nothing
        end
    end
    small_vector_gradient_buffer = if !isnothing(small_vector_gradient_frule)
        _forward_cache_small_vector_gradient_buffer(last(fx), Val(length(last(fx))))
    else
        nothing
    end
    # Keep the native gradient tuple on the fast path as well, so the public vector wrapper
    # does not pay an extra `Ref` lookup before calling the exact-width helper.
    small_vector_gradient_workspace = if !isnothing(small_vector_gradient_frule)
        (NoTangent(), Vector{eltype(last(fx))}(undef, length(last(fx))))
    else
        nothing
    end
    # Bug fix note: probe the chunk_size=1 nfwd frule once at cache construction time.
    # This excludes valid Julia functions that happen not to run on NDual-lifted values from
    # the zero-allocation fast path, without paying a runtime guard cost on every call.
    _chunk_fastpath_supported(fx, frules) || return nothing
    return ForwardChunkFastPath(
        frules,
        pack_buffers,
        gradient_rrule,
        small_vector_gradient_frule,
        small_vector_gradient_buffer,
        small_vector_gradient_workspace,
    )
end

@inline function _forward_cache_small_vector_gradient_buffer(
    x::Vector{T}, ::Val{C}
) where {T<:IEEEFloat,C}
    packed = Matrix{T}(undef, length(x), C)
    fill!(packed, zero(T))
    @inbounds for i in 1:C
        packed[i, i] = one(T)
    end
    return packed
end

@inline function _forward_cache_write_small_vector_gradient!(
    grad::Vector{T}, dy::T, ::Val{1}
) where {T<:IEEEFloat}
    @inbounds grad[1] = dy
    return grad
end

@inline function _forward_cache_write_small_vector_gradient!(
    grad::Vector{T}, dy::NTuple{C,T}, ::Val{C}
) where {T<:IEEEFloat,C}
    @inbounds for i in 1:C
        grad[i] = dy[i]
    end
    return grad
end

@generated function _forward_cache_small_vector_value_and_gradient(
    rule::R, fastpath::ForwardChunkFastPath, native_gradients::Tuple, f, x::Vector{T}
) where {R,T<:IEEEFloat}
    C = NfwdMooncake.rule_chunk_size(R)
    return quote
        packed_tangent = fastpath.small_vector_gradient_buffer
        output = rule(Dual(f, NoTangent()), Dual(x, packed_tangent))
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        _forward_cache_write_small_vector_gradient!(
            native_gradients[2], tangent(output), Val($C)
        )
        return y, native_gradients
    end
end

@inline _chunk_pack_buffer_template(::IEEEFloat) = nothing
@inline _chunk_pack_buffer_template(::Complex{<:IEEEFloat}) = nothing
@inline function _chunk_pack_buffer_template(
    x::Array{T,N}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},N}
    return Ref{Union{Nothing,Array{T}}}(nothing)
end

@inline function _chunk_pack_buffer_template(x::Tuple)
    return tuple_map(_chunk_pack_buffer_template, x)
end

@inline function _chunk_nfwd_rule(fastpath::ForwardChunkFastPath, ::Val{N}) where {N}
    N <= _CHUNK_NFWD_MAX_LANES || return nothing
    return fastpath.frules[N]
end

@inline function _chunk_pack_buffer!(
    ref::Base.RefValue{Union{Nothing,Array{T}}}, x::Array{T,N}, ::Val{C}
) where {T,N,C}
    buf = ref[]
    wanted = (size(x)..., C)
    if !(buf isa Array{T} && size(buf) == wanted)
        ref[] = Array{T}(undef, wanted)
    end
    return ref[]::Array{T}
end

@inline function _chunk_pack_tangent(::IEEEFloat, dx::NTangent, _buf, ::Val{N}) where {N}
    return ntuple(k -> dx[k], Val(N))
end
@inline function _chunk_pack_tangent(::IEEEFloat, dx, _buf, ::Val{N}) where {N}
    return ntuple(_ -> dx, Val(N))
end

@inline function _chunk_pack_tangent(
    ::Complex{<:IEEEFloat}, dx::NTangent, _buf, ::Val{N}
) where {N}
    return ntuple(k -> dx[k], Val(N))
end
@inline function _chunk_pack_tangent(::Complex{<:IEEEFloat}, dx, _buf, ::Val{N}) where {N}
    return ntuple(_ -> dx, Val(N))
end

function _chunk_pack_tangent(
    x::Array{T,N}, dx::NTangent, buf_ref::Base.RefValue{Union{Nothing,Array{T}}}, ::Val{C}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},N,C}
    buf = _chunk_pack_buffer!(buf_ref, x, Val(C))
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        for lane in 1:C
            buf[idx..., lane] = dx[lane][I]
        end
    end
    return buf
end

function _chunk_pack_tangent(
    x::Array{T,N}, dx::Array{T,N}, buf_ref::Base.RefValue{Union{Nothing,Array{T}}}, ::Val{C}
) where {T<:Union{IEEEFloat,Complex{<:IEEEFloat}},N,C}
    buf = _chunk_pack_buffer!(buf_ref, x, Val(C))
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        value = dx[I]
        for lane in 1:C
            buf[idx..., lane] = value
        end
    end
    return buf
end

@inline function _chunk_rechunk_tuple_field(dx::NTangent, ::Val{i}, ::Val{N}) where {i,N}
    return NTangent(ntuple(lane -> dx[lane][i], Val(N)))
end

@inline function _chunk_pack_tangent(
    x::Tuple, dx::NTangent, bufs::Tuple, ::Val{N}
) where {N}
    return ntuple(
        i -> _chunk_pack_tangent(
            x[i], _chunk_rechunk_tuple_field(dx, Val(i), Val(N)), bufs[i], Val(N)
        ),
        Val(fieldcount(typeof(x))),
    )
end

@inline function _chunk_pack_tangent(x::Tuple, dx::Tuple, bufs::Tuple, ::Val{N}) where {N}
    return ntuple(
        i -> _chunk_pack_tangent(x[i], dx[i], bufs[i], Val(N)), Val(fieldcount(typeof(x)))
    )
end

@inline function _chunk_lane_from_nfwd(::IEEEFloat, dy::Tuple, ::Val{lane}) where {lane}
    return dy[lane]
end
@inline function _chunk_lane_from_nfwd(
    ::Complex{<:IEEEFloat}, dy::Tuple, ::Val{lane}
) where {lane}
    return dy[lane]
end
@inline function _chunk_lane_from_nfwd(y::Array, dy::Array, ::Val{lane}) where {lane}
    return selectdim(dy, ndims(dy), lane)
end
@inline function _chunk_lane_from_nfwd(y::Tuple, dy::Tuple, ::Val{lane}) where {lane}
    return tuple_map((yi, dyi) -> _chunk_lane_from_nfwd(yi, dyi, Val(lane)), y, dy)
end

@inline function _chunk_unpack_nfwd_output(y, dy, ::Val{N}) where {N}
    return y, NTangent(ntuple(lane -> _chunk_lane_from_nfwd(y, dy, Val(lane)), Val(N)))
end

@inline function _chunk_should_fallback_to_lane_loop(err)
    err isa NfwdMooncake.UnsupportedError && return true
    err isa Nfwd.NDualUnsupportedError && return true
    msg = sprint(showerror, err)
    return (err isa MethodError || err isa TypeError || err isa InexactError) &&
           occursin("NDual", msg)
end

@generated function _resolve_chunk_size(ts::T) where {T<:Tuple}
    lane_count = nothing
    for entry in T.parameters
        entry <: NTangent || continue
        current_lanes = fieldcount(entry.parameters[1])
        if isnothing(lane_count)
            lane_count = current_lanes
        elseif lane_count != current_lanes
            return quote
                throw(
                    ArgumentError(
                        "All NTangent inputs must have the same number of lanes; " *
                        "found both $(lane_count) and $(current_lanes).",
                    ),
                )
            end
        end
    end

    # Bug fix note: make chunk-size resolution purely type-driven so lane count stays
    # constant-propagated through tuple-interface forward mode.
    return isnothing(lane_count) ? :(nothing) : :(Val{$lane_count}())
end

# Chunked forward architecture:
#
#   prepare_derivative_cache / prepare_gradient_cache
#       |
#       +--> width-1 forward rule (`cache.rule`)
#       +--> batched forward backend (`chunk_frule!!`)
#              |
#              +--> generic lane loop in this file
#              +--> nfwd override in nfwd/NfwdMooncake.jl
#
#   value_and_derivative!! with single-lane tangents
#       --> width-1 forward rule (`cache.rule`)
#
#   value_and_derivative!! with `NTangent` inputs
#       --> chunk_frule!!
#
#   value_and_gradient!! fast paths
#       --> scalar `value_and_gradient!!` fast path
#       --> small-vector `value_and_gradient!!` fast path
#       --> array `value_and_gradient!!` fast path
#
#   value_and_gradient!! generic path
#       --> seed standard-basis chunk tangents
#       --> chunk_frule!!
#       --> accumulate lane contributions into gradient buffers
#
@noinline function _chunk_lane_loop_frule!!(
    cache::ForwardCache,
    input_primals::Tuple,
    input_tangents::Tuple,
    ::Val{N};
    friendly_tangents::Bool,
) where {N}
    # Canonical fallback backend for batched forward mode: evaluate one width-1 lane at a
    # time through the ordinary forward rule, then repack the outputs into `NTangent`.
    # Specialized `chunk_frule!!` methods may replace this with a true batched execution.
    function compute_lane_output(::Val{lane}) where {lane}
        lane_tangents = tuple_map(t -> _chunk_lane(t, Val(lane)), input_tangents)
        return if friendly_tangents
            native_tangents = tuple_map(
                primal_to_tangent!!, cache.input_tangents, lane_tangents
            )
            cache.rule(tuple_map(Dual, input_primals, native_tangents)...)
        else
            lane_duals = tuple_map(Dual, input_primals, lane_tangents)
            error_if_incorrect_dual_types(lane_duals...)
            cache.rule(lane_duals...)
        end
    end

    first_output = compute_lane_output(Val(1))
    y = primal(first_output)
    first_tangent = if friendly_tangents
        tangent_to_primal!!(_copy_output(cache.output_primal), tangent(first_output))
    else
        _copy_chunk_tangent_output(tangent(first_output))
    end

    # Bug fix note: keep the lane count in dispatch so chunked tuple evaluation does not
    # depend on `Val` internals, which broke ordinary interface calls during refactoring.
    rest_tangents = ntuple(
        n -> begin
            lane_output = compute_lane_output(Val(n + 1))
            return if friendly_tangents
                tangent_to_primal!!(_copy_output(cache.output_primal), tangent(lane_output))
            else
                _copy_chunk_tangent_output(tangent(lane_output))
            end
        end,
        Val(N - 1),
    )

    return y, NTangent((first_tangent, rest_tangents...))
end

@noinline function chunk_frule!!(
    cache::ForwardCache,
    input_primals::Tuple,
    input_tangents::Tuple,
    ::Val{N};
    friendly_tangents::Bool=false,
) where {N}
    N < 1 && throw(ArgumentError("NTangent inputs must contain at least one lane."))
    return _chunk_lane_loop_frule!!(
        cache, input_primals, input_tangents, Val(N); friendly_tangents
    )
end

@noinline function _value_and_derivative_chunked(
    cache::ForwardCache,
    input_primals::Tuple,
    input_tangents::Tuple,
    ::Val{N};
    friendly_tangents::Bool,
) where {N}
    return chunk_frule!!(cache, input_primals, input_tangents, Val(N); friendly_tangents)
end

@noinline function _value_and_derivative_chunked(
    cache::ForwardCache,
    input_primals::Tuple,
    input_tangents::Tuple;
    friendly_tangents::Bool,
)
    N_val = _resolve_chunk_size(input_tangents)
    isnothing(N_val) && throw(ArgumentError("No NTangent inputs were provided."))
    return _value_and_derivative_chunked(
        cache, input_primals, input_tangents, N_val; friendly_tangents
    )
end

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns a cache used with [`value_and_derivative!!`](@ref). See that function for more info.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    fx = (f, x...)
    rule = build_frule(fx...; config.debug_mode, config.silence_debug_messages)

    if config.friendly_tangents
        y = f(x...)
        input_tangents = tuple_map(zero_tangent, fx)
        output_primal = _copy_output(y)
        return ForwardCache(
            rule,
            input_tangents,
            output_primal,
            _copy_output(fx),
            _forward_cache_gradient_workspace_ref(typeof(fx)),
            _maybe_build_chunk_fastpath(fx, config),
            map(_prepared_cache_input_spec, fx),
        )
    else
        return ForwardCache(
            rule,
            nothing,
            nothing,
            nothing,
            _forward_cache_gradient_workspace_ref(typeof(fx)),
            _maybe_build_chunk_fastpath(fx, config),
            map(_prepared_cache_input_spec, fx),
        )
    end
end

#
# `value_and_gradient!!` generic `chunk_frule!!` path
#
"""
    value_and_gradient!!(cache::ForwardCache, f, x...)

Compute the value and gradient of a scalar-returning function using the generic
`chunk_frule!!` path: seed standard-basis `NTangent`s, call the batched forward
interface, then accumulate the lane contributions into gradient storage. Specialized
backends behind `chunk_frule!!` may pack/unpack those `NTangent`s into a different
representation (for example NDual lanes), but this generic path is expressed at the
`NTangent` boundary.

This overload exists so callers can prepare a forward cache once, then use it either for
directional derivatives via [`value_and_derivative!!`](@ref) or full gradients via chunked
forward mode.
"""
function _value_and_gradient_forwardcache_generic(cache::ForwardCache, input_primals::Tuple)
    native_gradients = _forward_cache_gradient_workspace!(cache, input_primals)
    total_dof = _forward_cache_input_dof(input_primals)

    if total_dof == 0
        output = cache.rule(tuple_map(Dual, input_primals, native_gradients)...)
        return _forward_cache_finish_value_and_gradient(
            cache, input_primals, primal(output), native_gradients
        )
    end

    chunk_size = min(total_dof, _CHUNK_NFWD_MAX_LANES)
    first_chunk_width = min(chunk_size, total_dof)
    first_input_tangents = _forward_cache_seed_chunk_inputs(
        input_primals, 1, first_chunk_width
    )
    # `value_and_gradient!!` is a client of the batched forward interface: it seeds
    # standard-basis chunk tangents, calls `chunk_frule!!` via `_value_and_derivative_chunked`,
    # and accumulates the resulting lane contributions into gradient storage.
    y, first_chunk_dy = _value_and_derivative_chunked(
        cache, input_primals, first_input_tangents; friendly_tangents=false
    )
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    native_gradients = _forward_cache_accumulate_chunk!(
        native_gradients, first_input_tangents, first_chunk_dy
    )

    for start_slot in (1 + chunk_size):chunk_size:total_dof
        chunk_width = min(chunk_size, total_dof - start_slot + 1)
        input_tangents = _forward_cache_seed_chunk_inputs(
            input_primals, start_slot, chunk_width
        )
        _, chunk_dy = _value_and_derivative_chunked(
            cache, input_primals, input_tangents; friendly_tangents=false
        )
        native_gradients = _forward_cache_accumulate_chunk!(
            native_gradients, input_tangents, chunk_dy
        )
    end

    return _forward_cache_finish_value_and_gradient(
        cache, input_primals, y, native_gradients
    )
end

#
# `value_and_gradient!!` fast paths
#

# Scalar `value_and_gradient!!` fast path: this is a width-1 forward evaluation, using
# `cache.rule` or the cached width-1 nfwd rule when available. The win here is
# avoiding the generic `chunk_frule!!` path's chunk seeding and lane accumulation.
function value_and_gradient!!(cache::ForwardCache, f::F, x::T) where {F,T<:IEEEFloat}
    _prepared_cache_check_specs(cache.input_specs, f, x)
    output = let fastpath = cache.chunk_fastpath
        if isnothing(fastpath)
            cache.rule(Dual(f, NoTangent()), Dual(x, one(x)))
        else
            rule = _chunk_nfwd_rule(fastpath, Val(1))
            if isnothing(rule)
                cache.rule(Dual(f, NoTangent()), Dual(x, one(x)))
            else
                # Bug fix note: the scalar 1-DOF fast path should use the cached nfwd
                # frule. Unsupported NDual cases are filtered out at cache construction time,
                # so the steady-state call path stays allocation-free and competitive.
                rule(Dual(f, NoTangent()), Dual(x, one(x)))
            end
        end
    end
    return _forward_cache_finish_value_and_gradient(
        cache, (f, x), primal(output), (NoTangent(), tangent(output))
    )
end

# Small-vector `value_and_gradient!!` fast path: this one is nfwd-specific. When the
# full gradient fits under `_CHUNK_NFWD_MAX_LANES`, use one nfwd pass whose lane
# count exactly matches the full gradient width, instead of the generic `chunk_frule!!`
# path's seed-chunk/accumulate loop.
@inline function value_and_gradient!!(
    cache::ForwardCache, f::F, x::V
) where {F,T<:IEEEFloat,V<:Vector{T}}
    _prepared_cache_check_specs(cache.input_specs, f, x)
    fastpath = cache.chunk_fastpath
    if !isnothing(fastpath) && !isnothing(fastpath.small_vector_gradient_frule)
        y, output = _forward_cache_small_vector_value_and_gradient(
            fastpath.small_vector_gradient_frule,
            fastpath,
            fastpath.small_vector_gradient_workspace,
            f,
            x,
        )
        return _forward_cache_finish_value_and_gradient(cache, (f, x), y, output)
    elseif !isnothing(fastpath) && !isnothing(fastpath.gradient_rrule)
        return _forward_cache_gradient_rrule_fastpath(cache, fastpath, f, x)
    end
    return _value_and_gradient_forwardcache_generic(cache, (f, x))
end

# Array `value_and_gradient!!` fast path: this one is also nfwd-specific, but it uses
# the cached nfwd-derived gradient `rrule` rather than the `chunk_frule!!` /
# `NTangent` interface. The win here is writing gradients directly, avoiding the
# generic chunk path's `NTangent` packing/unpacking and lane accumulation.
function value_and_gradient!!(
    cache::ForwardCache, f::F, x::A
) where {F,A<:Array{<:IEEEFloat}}
    _prepared_cache_check_specs(cache.input_specs, f, x)
    fastpath = cache.chunk_fastpath
    if !isnothing(fastpath) && !isnothing(fastpath.gradient_rrule)
        return _forward_cache_gradient_rrule_fastpath(cache, fastpath, f, x)
    end
    return _value_and_gradient_forwardcache_generic(cache, (f, x))
end

function value_and_gradient!!(cache::ForwardCache, f::F, x::Vararg{Any,N}) where {F,N}
    input_primals = (f, x...)
    _prepared_cache_check_specs(cache.input_specs, input_primals)
    return _value_and_gradient_forwardcache_generic(cache, input_primals)
end

"""
    value_and_derivative!!(cache::ForwardCache, f::Dual, x::Vararg{Dual,N})

Returns a `Dual` containing the result of applying forward-mode AD to compute the (Frechet)
derivative of `primal(f)` at the primal values in `x` in the direction of the tangent values
in `f` and `x`.
"""
function value_and_derivative!!(cache::ForwardCache, fx::Vararg{Dual,N}) where {N}
    _prepared_cache_check_specs(cache.input_specs, map(primal, fx))
    if _has_chunk_tangent(fx)
        # Bug fix note: routing chunked `Dual(...)` inputs through the tuple path hit a
        # Julia 1.10 compiler/codegen crash, so chunked inputs currently stay tuple-only.
        throw(
            ArgumentError(
                "NTangent inputs are currently supported via the tuple interface " *
                "only. Use `value_and_derivative!!(cache, (f, df), (x, dx), ...)`.",
            ),
        )
    end
    # TODO: check Dual coherence here like we do below?
    return __call_rule(cache.rule, fx)
end

"""
    value_and_derivative!!(cache::ForwardCache, (f, df), (x, dx), ...)

Returns a tuple `(y, dy)` containing the result of applying forward-mode AD to compute the (Frechet) derivative of `primal(f)` at the primal values in `x` in the direction of the tangent values contained in `df` and `dx`.

Tuples are used as inputs and outputs instead of `Dual` numbers to accommodate the case where internal Mooncake tangent types do not coincide with tangents provided by the user (in which case we translate between "friendly tangents" and internal tangents using cache storage).

!!! info
    `cache` must be the output of [`prepare_derivative_cache`](@ref), and (fields of) `f` and `x` must be of the same size and shape as those used to construct the `cache`. This is to ensure that the gradient can be written to the memory allocated when the `cache` was built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable components of values returned by it will be mutated if you run this function again with different arguments. Therefore, if you need to keep the values returned by this function around over multiple calls to this function with the same `cache`, you should take a copy (using `copy` or `deepcopy`) of them before calling again.
"""
@inline function _value_and_derivative_tuple_interface(
    cache::ForwardCache, fx::FX
) where {FX<:Tuple}
    friendly_tangents = !isnothing(cache.input_tangents)

    input_primals = tuple_map(first, fx)
    input_friendly_tangents = tuple_map(last, fx)
    _prepared_cache_check_specs(cache.input_specs, input_primals)

    if friendly_tangents && !isnothing(_resolve_chunk_size(input_friendly_tangents))
        return _value_and_derivative_chunked(
            cache, input_primals, input_friendly_tangents; friendly_tangents
        )
    end

    # translate from friendly to native
    if friendly_tangents
        input_tangents = tuple_map(
            primal_to_tangent!!, cache.input_tangents, input_friendly_tangents
        )
    else
        input_tangents = input_friendly_tangents
    end

    if !isnothing(_resolve_chunk_size(input_tangents))
        return _value_and_derivative_chunked(
            cache, input_primals, input_tangents; friendly_tangents
        )
    end

    input_duals = tuple_map(Dual, input_primals, input_tangents)

    if !friendly_tangents  # in friendly mode, conversion should ensure tangent coherence
        error_if_incorrect_dual_types(input_duals...)
    end

    output = __call_rule(cache.rule, input_duals)
    output_primal = primal(output)
    output_tangent = tangent(output)

    # translate from native back to friendly
    if friendly_tangents
        c = _friendly_cache((output_primal,))
        output_dest = friendly_tangent_cache(output_primal)
        output_friendly_tangent = tangent_to_friendly!!(
            output_dest, output_primal, output_tangent, c
        )
        return output_primal, output_friendly_tangent
    else
        return output_primal, output_tangent
    end
end

# `fwd_cache` is the derivative cache for `grad_f`. The compiled inner rrule is cached
# across `value_and_hvp!!` calls via a `LazyFoRRule` captured inside `fwd_cache`'s frule.
"""
    HVPCache

Cache type used by [`prepare_hvp_cache`](@ref) and [`prepare_hessian_cache`](@ref) for
repeated Hessian-vector product and Hessian evaluations.
"""
struct HVPCache{Tf,Tgrad_f,Tgrad_tangent,Tfwd_cache}
    f::Tf
    grad_f::Tgrad_f
    # Pre-computed zero tangent for grad_f; the function is never perturbed, only x is.
    # Safe to reuse because grad_f's closure environment is shape-stable for the lifetime
    # of the cache: grad_cache mutates stored values between calls but does not change the
    # closure/capture structure that zero_tangent depends on.
    grad_tangent::Tgrad_tangent
    fwd_cache::Tfwd_cache
end

@inline function _assert_hvp_cache_function(cache::HVPCache, f)
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
    return nothing
end

@inline function _assert_matching_tangent_shape(primal, tangent, arg_index::Int)
    if applicable(axes, primal) && applicable(axes, tangent)
        axes(primal) == axes(tangent) || throw(
            ArgumentError(
                "Tangent direction for argument $arg_index must match the primal axes; got axes $(axes(tangent)) for tangent vs $(axes(primal)) for primal",
            ),
        )
    elseif applicable(length, primal) && applicable(length, tangent)
        length(primal) == length(tangent) || throw(
            ArgumentError(
                "Tangent direction for argument $arg_index must match the primal length; got length $(length(tangent)) for tangent vs $(length(primal)) for primal",
            ),
        )
    end
    return nothing
end

"""
    prepare_hvp_cache(f, x...; config=Mooncake.Config())

Prepare a cache for computing Hessian-vector products (HVPs) of `f`. Returns an `HVPCache`
for use with [`value_and_hvp!!`](@ref).

`f` must map `x...` to a scalar. Multiple arguments are supported: see
[`value_and_hvp!!`](@ref) for the calling convention.

The cache compiles an outer forward-mode rule over an inner reverse-mode gradient. The
inner rule is compiled only once regardless of how many HVPs are subsequently evaluated.

*Note:* `cache` is tied to the types and shapes of `x...`. Evaluating at a different point
is fine, but changing the shapes requires a new cache.

```jldoctest
f(x) = sum(x .* x)
x = [1.0, 2.0]
cache = Mooncake.prepare_hvp_cache(f, x)
f_val, gradient, hvp = Mooncake.value_and_hvp!!(cache, f, [1.0, 0.0], x)
f_val ≈ 5.0 && gradient ≈ [2.0, 4.0] && hvp ≈ [2.0, 0.0]

# output

true
```
"""
@unstable @inline function prepare_hvp_cache(
    f::F, x::Vararg{Any,N}; config=Config()
) where {F,N}
    N == 0 && throw(ArgumentError("prepare_hvp_cache requires at least one x argument"))
    # Pre-build the reverse-mode gradient cache so forward-over-reverse differentiates
    # only through gradient evaluation, not through repeated rule construction.
    grad_cache = prepare_gradient_cache(f, x...; config)
    grad_f = if N == 1
        let f = f, grad_cache = grad_cache
            y -> begin
                val_and_grad = value_and_gradient!!(grad_cache, f, y)
                (val_and_grad[1], val_and_grad[2][2])
            end
        end
    else
        let f = f, grad_cache = grad_cache
            function (ys...)
                val_and_grad = value_and_gradient!!(grad_cache, f, ys...)
                # Drop the gradient w.r.t. f itself (always index 1); return only x-arg gradients.
                (val_and_grad[1], Base.tail(val_and_grad[2]))
            end
        end
    end
    fwd_cache = prepare_derivative_cache(grad_f, x...; config)
    return HVPCache(f, grad_f, zero_tangent(grad_f), fwd_cache)
end

"""
    value_and_hvp!!(cache::HVPCache, f, v, x...)

Given a cache prepared by [`prepare_hvp_cache`](@ref), compute the gradient of `f` at
`x...` and the Hessian-vector product `H v`.

**Single argument:** `v` is the tangent direction; returns `(f(x), ∇f(x), H(x)v)`. For
`f: Rⁿ → R` with `x::Vector{Float64}`, the gradient and HVP are `Vector{Float64}`.

**Multiple arguments:** `v` must be a tuple of tangent directions (one per argument);
returns `(f(x...), (∇f_x1, ∇f_x2, ...), (h1, h2, ...))` where
`hk = ∑_j (∂²f/∂xk∂xj) v[j]` is the joint Hessian-vector product for argument `xk`.

!!! warning
    `cache` must be the output of [`prepare_hvp_cache`](@ref), and `f` must be the same
    function object used to construct `cache`. All `x` arguments must have the same sizes
    and element types as used to construct the cache.

!!! warning
    `cache` owns the mutable state in the returned values. Take a copy before calling again
    if you need to retain previous results.

!!! warning
    `HVPCache` is not safe for concurrent reuse across threads. Use a separate cache per
    task/thread if calls may overlap in time.

```jldoctest
f(x) = sum(x .* x)
x = [1.0, 2.0]
cache = Mooncake.prepare_hvp_cache(f, x)
f_val, gradient, hvp = Mooncake.value_and_hvp!!(cache, f, [1.0, 0.0], x)
f_val ≈ 5.0 && gradient ≈ [2.0, 4.0] && hvp ≈ [2.0, 0.0]

# output

true
```
"""
@inline function value_and_hvp!!(cache::HVPCache, f::F, v, x1::T1) where {F,T1}
    _assert_hvp_cache_function(cache, f)
    _prepared_cache_check_specs(cache.fwd_cache.input_specs, cache.grad_f, x1)
    _assert_matching_tangent_shape(x1, v, 1)
    (f_val, grad), (_, hvp) = value_and_derivative!!(
        cache.fwd_cache, (cache.grad_f, cache.grad_tangent), (x1, v)
    )
    return f_val, grad, hvp
end

@inline function value_and_hvp!!(
    cache::HVPCache, f::F, v::Tuple, x1::T1, xrest::Vararg{Any,N}
) where {F,T1,N}
    _assert_hvp_cache_function(cache, f)
    all_x = (x1, xrest...)
    _prepared_cache_check_specs(cache.fwd_cache.input_specs, (cache.grad_f, all_x...))
    nargs = N + 1
    length(v) == nargs ||
        throw(ArgumentError("Expected one tangent direction per primal argument"))
    for i in 1:nargs
        _assert_matching_tangent_shape(all_x[i], v[i], i)
    end
    (f_val, grads), (_, hvps) = value_and_derivative!!(
        cache.fwd_cache, (cache.grad_f, cache.grad_tangent), map(tuple, all_x, v)...
    )
    return f_val, grads, hvps
end

"""
    prepare_hessian_cache(f, x...; config=Mooncake.Config())

Return a cache for computing `f(x...)`, gradients `∇f`, and the Hessian (or Hessian
blocks) of `f` via [`value_gradient_and_hessian!!`](@ref). Returns an [`HVPCache`](@ref),
which can also be used directly with [`value_and_hvp!!`](@ref).

`prepare_hessian_cache` reuses the generic HVP cache builder. It eagerly checks only
that at least one `x` argument was provided; validation that the `x...` inputs are
`AbstractVector`s of IEEE floats, all with the same element type, is deferred to
[`value_gradient_and_hessian!!`](@ref).

Hessian computation uses forward-over-reverse AD: one forward-mode pass per input
dimension over the reverse-mode gradient function.

```jldoctest
f(x) = sum(x .^ 2)
x = [1.0, 2.0, 3.0]
cache = Mooncake.prepare_hessian_cache(f, x)
Mooncake.value_gradient_and_hessian!!(cache, f, x)

# output

(14.0, [2.0, 4.0, 6.0], [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0])
```
"""
@unstable @inline function prepare_hessian_cache(
    f::F, x::Vararg{Any,N}; config=Config()
) where {F,N}
    return prepare_hvp_cache(f, x...; config)
end

function _validate_hessian_argument(x, i::Int)
    x isa AbstractVector || throw(
        ArgumentError(
            "value_gradient_and_hessian!! only supports AbstractVector inputs; argument $i has type $(typeof(x))",
        ),
    )
    T = eltype(x)
    T <: IEEEFloat || throw(
        ArgumentError(
            "value_gradient_and_hessian!! only supports AbstractVector inputs with IEEEFloat element types; argument $i has eltype $T",
        ),
    )
    return T
end

function _validate_hessian_arguments(x::Vararg{Any,N}) where {N}
    T = _validate_hessian_argument(x[1], 1)
    for i in 2:N
        Ti = _validate_hessian_argument(x[i], i)
        Ti == T || throw(
            ArgumentError(
                "value_gradient_and_hessian!! requires all arguments to share the same IEEEFloat element type; argument 1 has eltype $T but argument $i has eltype $Ti",
            ),
        )
    end
    return T
end

"""
    value_gradient_and_hessian!!(cache::HVPCache, f, x...)

Using a pre-built `cache` (from [`prepare_hessian_cache`](@ref) or
[`prepare_hvp_cache`](@ref)), compute and return `(value, gradient, hessian)` of `f`.

**Single argument:** returns `(f(x), ∇f(x), ∇²f(x))` — value, gradient vector, Hessian
matrix.

**Multiple arguments:** returns `(f(x1,...), (∇_x1 f, ∇_x2 f, ...), H_blocks)` where
`H_blocks[k][j]` is the `nk × nj` matrix `∂²f/∂xk∂xj`. The return structure differs
from the single-argument case.

Uses forward-over-reverse AD: one forward-mode pass per total input dimension.

!!! info
    `cache` must be the output of [`prepare_hessian_cache`](@ref) or
    [`prepare_hvp_cache`](@ref), and `f` must be the same function object used to
    construct `cache`. All `x` arguments must have the same sizes and element types as
    used to construct the cache. The current implementation supports only
    `AbstractVector`s of IEEE floats, with all arguments sharing the same element type.
    This restriction comes from the Hessian assembly logic, which sweeps a standard
    basis of tangent vectors and materialises dense matrix / block-matrix outputs. For
    non-vector inputs, use [`value_and_hvp!!`](@ref) to obtain second-order directional
    derivatives without forming a full Hessian.

!!! warning
    `HVPCache` is not safe for concurrent reuse across threads. Use a separate cache per
    task/thread if calls may overlap in time.

# Example
```jldoctest
f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
x = [1.2, 1.2]
cache = Mooncake.prepare_hessian_cache(f, x)
_, _, H = Mooncake.value_gradient_and_hessian!!(cache, f, x)
H

# output

2×2 Matrix{Float64}:
 1250.0  -480.0
 -480.0   200.0
```
"""
@unstable @inline function value_gradient_and_hessian!!(
    cache::HVPCache, f::F, x1::T1
) where {F,T1}
    _assert_hvp_cache_function(cache, f)
    T = _validate_hessian_argument(x1, 1)
    if length(x1) == 0
        v = similar(x1, T)
        fval, grad, _ = value_and_hvp!!(cache, f, v, x1)
        return fval, copy(grad), zeros(T, 0, 0)
    end
    n = length(x1)
    H = zeros(T, n, n)
    v = zeros(T, n)
    local value, gradient
    for i in 1:n
        v[i] = one(T)
        fval, grad, hvp = value_and_hvp!!(cache, f, v, x1)
        if i == 1
            value = fval
            gradient = copy(grad)
        end
        H[:, i] .= hvp
        v[i] = zero(T)
    end
    return value, gradient, H
end

@unstable @inline function value_gradient_and_hessian!!(
    cache::HVPCache, f::F, x1::T1, xrest::Vararg{Any,N}
) where {F,T1,N}
    _assert_hvp_cache_function(cache, f)
    all_xs = (x1, xrest...)
    T = _validate_hessian_arguments(all_xs...)
    nargs = N + 1
    ns = map(length, all_xs)
    # H_blocks[k][j] = ∂²f/∂xk∂xj, shape ns[k] × ns[j]
    H_blocks = ntuple(k -> ntuple(j -> zeros(T, ns[k], ns[j]), nargs), nargs)
    # one mutable tangent-direction buffer per argument (reused across HVP calls)
    v = map(ni -> zeros(T, ni), ns)
    # if all arguments are empty, skip the HVP loop and recover value/grads directly
    if all(==(0), ns)
        fval, gs, _ = value_and_hvp!!(cache, f, v, all_xs...)
        return fval, map(copy, gs), H_blocks
    end
    local value, grads
    first_iter = true
    for argidx in 1:nargs
        v_i = v[argidx]
        for i in 1:ns[argidx]
            v_i[i] = one(T)
            fval, gs, hvps = value_and_hvp!!(cache, f, v, all_xs...)
            if first_iter
                value = fval
                grads = map(copy, gs)
                first_iter = false
            end
            for k in 1:nargs
                H_blocks[k][argidx][:, i] .= hvps[k]
            end
            v_i[i] = zero(T)
        end
    end
    return value, grads, H_blocks
end

function value_and_derivative!!(cache::ForwardCache, fx::Vararg{Tuple{Any,Any},N}) where {N}
    return _value_and_derivative_tuple_interface(cache, fx)
end

function value_and_derivative!!(cache::ForwardCache)
    _prepared_cache_check_specs(cache.input_specs, ())
    error("unreachable")
end
