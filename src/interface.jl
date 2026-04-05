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

function Base.showerror(io::IO, err::ValueAndGradientReturnTypeError)
    _print_boxed_error(io, split("ValueAndGradientReturnTypeError: $(err.msg)", '\n'))
end

function Base.showerror(io::IO, err::ValueAndPullbackReturnTypeError)
    _print_boxed_error(io, split("ValueAndPullbackReturnTypeError: $(err.msg)", '\n'))
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
    if friendly_tangents
        ȳ_tangent = primal_to_tangent!!(zero_tangent(ȳ), ȳ)
        value, pb = __value_and_pullback!!(rule, ȳ_tangent, __create_coduals(fx)...)
        dests = map(friendly_tangent_cache, fx)
        c = _friendly_cache(fx)
        friendly_pb = tuple_map(
            (d, p, t) -> tangent_to_friendly!!(d, p, t, c), dests, fx, pb
        )
        return value, friendly_pb
    end
    return __value_and_pullback!!(rule, ȳ, __create_coduals(fx)...)
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
    if friendly_tangents
        value, gradient = __value_and_gradient!!(rule, __create_coduals(fx)...)
        dests = map(friendly_tangent_cache, fx)
        c = _friendly_cache(fx)
        friendly_gradient = tuple_map(
            (d, p, t) -> tangent_to_friendly!!(d, p, t, c), dests, fx, gradient
        )
        return value, friendly_gradient
    end
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

"""
    value_and_derivative!!(rule, f::Dual, x::Dual...)
    value_and_derivative!!(rule, (f, df), (x, dx), ...)

Run a forward rule directly, without first constructing a `ForwardCache`.

The `Dual` interface returns the rule output directly. The tuple interface returns
`(y, dy)` using the rule's native tangent representation. Specialized rule types may
add chunked `NTangent` support on top of this entrypoint.
"""
@inline function value_and_derivative!!(rule::R) where {R}
    throw(
        ArgumentError(
            "`value_and_derivative!!(rule, ...)` expects at least the function input, " *
            "either as `f::Dual` or `(f, df)`.",
        ),
    )
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Dual,N}) where {R,N}
    return __call_rule(
        rule,
        map(x -> Dual(primal(x), _canonical_forward_tangent(primal(x), tangent(x))), fx),
    )
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Tuple{Any,Any},N}) where {R,N}
    input_primals = tuple_map(first, fx)
    input_tangents = tuple_map(last, fx)
    input_duals = tuple_map(_internal_forward_dual, input_primals, input_tangents)
    output = __call_rule(rule, input_duals)
    return primal(output), tangent(output)
end

# Cache types in this file:
# - `Cache`: reusable reverse-mode cache for repeated `value_and_pullback!!` and
#   `value_and_gradient!!` calls.
# - `ForwardCache`: reusable forward-mode cache for repeated `value_and_derivative!!` and
#   `value_and_gradient!!` calls.
# - `HVPCache`: reusable forward-over-reverse cache for repeated `value_and_hvp!!` calls;
#   Hessian helpers reuse this cache rather than introducing a separate Hessian cache type.
# Internal helper cache types in this file:
# - `NfwdCache`: internal cache bundle stored inside `ForwardCache` when the prepared
#   forward cache can use the chunked backend. The type name is legacy.
# All seven parameters are load-bearing: they keep the prepared reverse cache concrete
# across the cached rule, reusable primal/tangent buffers, and cached input/output specs.
struct Cache{Trule,Ty_cache,Ttangents<:Tuple,Tdests,Tȳ_cache,TIS<:Tuple,TOS}
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
    # Top-level type/size signature for y = f(x...).
    output_spec::TOS
end

@inline _cache_input_count(cache) = length(getfield(cache, :input_specs)) - 1
@inline _cache_x_input_specs(cache::Cache) = Base.tail(getfield(cache, :input_specs))

@inline function _cache_type_size_summary(::Type{T}) where {T}
    return if T <: IEEEFloat || T <: Complex{<:IEEEFloat}
        "scalar"
    elseif T <: AbstractArray
        "size unknown"
    elseif T === Any
        "unknown"
    elseif T <: NamedTuple
        "named tuple"
    elseif T <: Tuple
        "tuple"
    elseif T <: Function
        "function"
    elseif fieldcount(T) > 0 || Base.ismutabletype(T)
        "struct"
    else
        "value"
    end
end

@inline _cache_type_summary(::Type{T}) where {T} =
    T === Any ? "unknown" : "$(T) ($(_cache_type_size_summary(T)))"

function _cache_print_io_summary(io::IO, input_specs::Tuple, output_summary)
    for (i, spec) in enumerate(input_specs)
        print(io, "\n  input_", i, ": ", _cache_spec_summary(spec))
    end
    print(io, "\n  output: ", output_summary)
end

function Base.show(io::IO, cache::Cache)
    print(
        io,
        "Mooncake.Cache(",
        "mode=:reverse, ",
        "friendly_tangents=",
        !isnothing(getfield(cache, :dests)),
        ", inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::Cache)
    print(
        io,
        "Mooncake.Cache\n",
        "  mode: reverse\n",
        "  friendly_tangents: ",
        !isnothing(getfield(cache, :dests)),
        "\n",
        "  inputs: ",
        _cache_input_count(cache),
    )
    _cache_print_io_summary(
        io, _cache_x_input_specs(cache), _cache_spec_summary(getfield(cache, :output_spec))
    )
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

!!! note
    Calls `f(x...)` once during cache preparation.
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
    y, rvs!! = __call_rule(rule, map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents))

    # Run reverse-pass in order to reset stacks + state.
    rvs!!(zero_rdata(primal(y)))

    # Construct cache for output. Check that `_copy_to_output!!`ing appears to work.
    y_cache = _copy_output(primal(y))
    y_cache = _copy_to_output!!(y_cache, primal(y))
    input_specs = map(fx) do x
        if x isa AbstractArray
            PreparedCacheInputSpec(typeof(x), size(x))
        else
            PreparedCacheInputSpec(typeof(x), ())
        end
    end
    output_primal = primal(y)
    output_spec = if output_primal isa AbstractArray
        PreparedCacheInputSpec(typeof(output_primal), size(output_primal))
    else
        PreparedCacheInputSpec(typeof(output_primal), ())
    end
    if config.friendly_tangents
        dests = map(friendly_tangent_cache, fx)
        return Cache(
            rule,
            y_cache,
            tangents,
            dests,
            zero_tangent(primal(y)),
            input_specs,
            output_spec,
        )
    else
        return Cache(rule, y_cache, tangents, nothing, nothing, input_specs, output_spec)
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
    fx = (f, x...)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), fx)
    tangents = tuple_map(set_to_zero_maybe!!, getfield(cache, :tangents), args_to_zero)
    coduals = tuple_map(CoDual, fx, tangents)
    if isnothing(cache.dests)
        return __value_and_pullback!!(cache.rule, ȳ, coduals...; y_cache=cache.y_cache)
    end
    ȳ_tangent = primal_to_tangent!!(cache.ȳ_cache, ȳ)
    value, pb = __value_and_pullback!!(
        cache.rule, ȳ_tangent, coduals...; y_cache=cache.y_cache
    )
    c = _friendly_cache(fx)
    friendly_pb = tuple_map(
        (d, p, t) -> tangent_to_friendly!!(d, p, t, c), getfield(cache, :dests), fx, pb
    )
    return value, friendly_pb
end

"""
    prepare_gradient_cache(f, x...; config=Mooncake.Config())

Returns a cache used with [`value_and_gradient!!`](@ref). See that function for more info.

The API guarantees that tangents are initialized at zero before the first autodiff pass.

!!! note
    Calls `f(x...)` once during cache preparation.
"""
@unstable function prepare_gradient_cache(fx...; config=Config())
    rule = build_rrule(fx...; config.debug_mode, config.silence_debug_messages)
    tangents = map(zero_tangent, fx)
    y, rvs!! = __call_rule(rule, map((x, dx) -> CoDual(x, fdata(dx)), fx, tangents))
    primal(y) isa IEEEFloat || throw_val_and_grad_ret_type_error(primal(y))
    rvs!!(zero_tangent(primal(y))) # run reverse-pass to reset stacks + state
    input_specs = map(fx) do x
        if x isa AbstractArray
            PreparedCacheInputSpec(typeof(x), size(x))
        else
            PreparedCacheInputSpec(typeof(x), ())
        end
    end
    output_primal = primal(y)
    output_spec = if output_primal isa AbstractArray
        PreparedCacheInputSpec(typeof(output_primal), size(output_primal))
    else
        PreparedCacheInputSpec(typeof(output_primal), ())
    end
    if config.friendly_tangents
        dests = tuple(map(friendly_tangent_cache, fx)...)
        return Cache(rule, nothing, tangents, dests, nothing, input_specs, output_spec)
    else
        return Cache(rule, nothing, tangents, nothing, nothing, input_specs, output_spec)
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
    fx = (f, x...)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), fx)
    tangents = tuple_map(set_to_zero_maybe!!, getfield(cache, :tangents), args_to_zero)
    coduals = tuple_map(CoDual, fx, tangents)
    if isnothing(cache.dests)
        return __value_and_gradient!!(cache.rule, coduals...)
    end
    value, gradient = __value_and_gradient!!(cache.rule, coduals...)
    c = _friendly_cache(fx)
    friendly_gradient = tuple_map(
        (d, p, t) -> tangent_to_friendly!!(d, p, t, c),
        getfield(cache, :dests),
        fx,
        gradient,
    )
    return value, friendly_gradient
end

# Internal chunked forward cache stored inside `ForwardCache`.
# `NfwdCache` is a legacy internal name: this bundle now stores the prepared-cache
# chunked backend in general, not just direct nfwd execution.
# It stores one forward rule per supported chunk width (1 through 8). Keeping those rules
# in separate fields lets the code pick, for example, the width-3 rule directly, instead
# of indexing into a tuple of mixed rule types.
struct NfwdCache{R1,R2,R3,R4,R5,R6,R7,R8,GR,SG,SB,SW}
    frule_1::R1
    frule_2::R2
    frule_3::R3
    frule_4::R4
    frule_5::R5
    frule_6::R6
    frule_7::R7
    frule_8::R8
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
    gradient_chunk_size::Int
    gradient_chunk_size_auto::Bool
    chunkcache::CF
    input_specs::S
end

@inline _dual_primal_type(::Type) = Any
@inline _dual_primal_type(::Type{Dual{Y,T}}) where {Y,T} = Y

@inline function _forward_cache_output_summary(cache::ForwardCache)
    output_primal = getfield(cache, :output_primal)
    return if !isnothing(output_primal)
        _cache_spec_summary(
            if output_primal isa AbstractArray
                PreparedCacheInputSpec(typeof(output_primal), size(output_primal))
            else
                PreparedCacheInputSpec(typeof(output_primal), ())
            end,
        )
    else
        dual_arg_types = Tuple{
            map(
                spec -> dual_type(typeof(spec).parameters[1]), getfield(cache, :input_specs)
            )...,
        }
        output_type = Core.Compiler.return_type(getfield(cache, :rule), dual_arg_types)
        _cache_type_summary(_dual_primal_type(output_type))
    end
end

function Base.show(io::IO, cache::ForwardCache)
    chunk_size = getfield(cache, :gradient_chunk_size)
    print(
        io,
        "Mooncake.ForwardCache(",
        "mode=:forward, ",
        "friendly_tangents=",
        !isnothing(getfield(cache, :input_tangents)),
        ", chunked=",
        !isnothing(getfield(cache, :chunkcache)),
        ", chunk_size=",
        getfield(cache, :gradient_chunk_size_auto) ? "$(chunk_size) (auto)" : chunk_size,
        ", inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::ForwardCache)
    chunk_size = getfield(cache, :gradient_chunk_size)
    print(
        io,
        "Mooncake.ForwardCache\n",
        "  mode: forward\n",
        "  friendly_tangents: ",
        !isnothing(getfield(cache, :input_tangents)),
        "\n",
        "  chunked: ",
        !isnothing(getfield(cache, :chunkcache)),
        "\n",
        "  chunk_size: ",
        getfield(cache, :gradient_chunk_size_auto) ? "$(chunk_size) (auto)" : chunk_size,
        "\n",
        "  inputs: ",
        _cache_input_count(cache),
    )
    _cache_print_io_summary(
        io, Base.tail(getfield(cache, :input_specs)), _forward_cache_output_summary(cache)
    )
end

@generated function _fcache_gradient_lazy_workspace_ref(::Type{T}) where {T<:Tuple}
    tangent_types = map(P -> :(tangent_type($P)), T.parameters)
    workspace_type = Expr(:curly, :Tuple, tangent_types...)
    # Keep the lazy gradient workspace concretely typed even before first use; otherwise
    # `Ref{Any}` makes cached forward gradients inference-opaque. Do this without
    # evaluating `zero_tangent` on the actual runtime inputs at cache-construction time.
    return :(Ref{Union{Nothing,$workspace_type}}(nothing))
end

# Cache specs are compared again when a prepared cache is reused. The input type `T` is
# encoded as a type parameter so that `_validate_prepared_cache_inputs` can read it at
# @generated specialisation time — eliminating the runtime `jl_types_equal` call that
# a `DataType`-valued field would require.
struct PreparedCacheInputSpec{T,S}
    size::S
end

PreparedCacheInputSpec(::Type{T}, s::S) where {T,S} = PreparedCacheInputSpec{T,S}(s)

@inline function _cache_spec_size_summary(spec::PreparedCacheInputSpec{T}) where {T}
    return if T <: IEEEFloat || T <: Complex{<:IEEEFloat}
        "scalar"
    elseif T <: AbstractArray
        "size $(spec.size)"
    elseif T <: NamedTuple
        "named tuple"
    elseif T <: Tuple
        "tuple"
    elseif T <: Function
        "function"
    elseif fieldcount(T) > 0 || Base.ismutabletype(T)
        "struct"
    else
        "value"
    end
end

@inline _cache_spec_summary(spec::PreparedCacheInputSpec{T}) where {T} = "$(T) ($(_cache_spec_size_summary(spec)))"

const _CHUNK_NFWD_MAX_LANES = 8

@inline _internal_forward_dual(x, dx) = Dual(x, _canonical_forward_tangent(x, dx))

@inline function _fcache_small_vector_fill_identity!(packed::NTangent)
    N = length(packed)
    @inbounds for lane in 1:N
        T = eltype(packed[lane])
        fill!(packed[lane], zero(T))
        packed[lane][lane] = one(T)
    end
    return packed
end

@inline function _fcache_small_vector_identity_seed(x::Vector{T}) where {T}
    packed = NTangent(ntuple(_ -> Vector{T}(undef, length(x)), Val(length(x))))
    _fcache_small_vector_fill_identity!(packed)
    return packed
end

@inline function _fcache_chunk_rule(fastpath::NfwdCache, ::Val{N}) where {N}
    return getfield(
        (
            fastpath.frule_1,
            fastpath.frule_2,
            fastpath.frule_3,
            fastpath.frule_4,
            fastpath.frule_5,
            fastpath.frule_6,
            fastpath.frule_7,
            fastpath.frule_8,
        ),
        N,
    )
end

struct PreparedCacheSpecError <: Exception
    msg::String
end

function Base.showerror(io::IO, err::PreparedCacheSpecError)
    _print_boxed_error(io, split("PreparedCacheSpecError:\n$(err.msg)", '\n'))
end

function _throw_prepared_cache_spec_error(kind::Symbol, i::Int, expected, got)
    label = i == 1 ? "`f`" : "`x$(i - 1)`"
    msg = if kind === :arity
        "Cached autodiff call expected $(expected) total arguments `(f, x...)`, got $(got).\n" *
        "Prepared pullback, gradient, derivative, HVP, and Hessian caches must be reused " *
        "with the same top-level argument structure they were prepared with."
    elseif kind === :type
        "Cached autodiff call has a type mismatch for $label.\n" *
        "Expected top-level type: $expected\n" *
        "Got top-level type: $got\n" *
        "Prepared pullback, gradient, derivative, HVP, and Hessian caches must be reused " *
        "with the same top-level argument types they were prepared with."
    else
        "Cached autodiff call has a size mismatch for $label.\n" *
        "Expected top-level size: $expected\n" *
        "Got top-level size: $got\n" *
        "Prepared pullback, gradient, derivative, HVP, and Hessian caches must be reused " *
        "with the same top-level array sizes they were prepared with."
    end
    throw(PreparedCacheSpecError(msg))
end

# Shared prepared-cache input validation for Cache, ForwardCache, and HVPCache entry points.
# The expected type T_i is extracted from the PreparedCacheInputSpec{T_i,S_i} type parameter
# at @generated specialisation time, so the emitted `typeof(x_i) == T_i` comparison uses a
# compile-time constant type — eliminating the runtime jl_types_equal call.
@generated function _validate_prepared_cache_inputs(specs::Tuple, fx::Tuple)
    n = length(specs.parameters)
    m = length(fx.parameters)
    n == m || return :(_throw_prepared_cache_spec_error(:arity, 0, $n, $m))
    checks = Expr(:block)
    for i in 1:n
        T_i = specs.parameters[i].parameters[1]
        push!(
            checks.args,
            quote
                let x_i = fx[$i]
                    typeof(x_i) == $T_i ||
                        _throw_prepared_cache_spec_error(:type, $i, $T_i, typeof(x_i))
                    if x_i isa AbstractArray
                        size(x_i) == specs[$i].size || _throw_prepared_cache_spec_error(
                            :size, $i, specs[$i].size, size(x_i)
                        )
                    end
                end
            end,
        )
    end
    return quote
        $checks
        return fx
    end
end

# fcache gradient bookkeeping
@noinline function _fcache_gradient_throw_uninit_field_error(::Type{P}, n::Int) where {P}
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
@inline _fcache_gradient_input_dof(x) = _fcache_gradient_input_dof(x, IdDict{Any,Any}())
# Mark mutable/shared nodes as seen before descending so aliasing contributes once and
# cycles terminate locally instead of recursing forever.
@inline _fcache_mark_seen!(seen::IdDict{Any,Any}, x) = (seen[x] = nothing)
@inline _fcache_gradient_input_dof(::NoTangent, _seen::IdDict{Any,Any}) = 0
@inline _fcache_gradient_input_dof(x::IEEEFloat, _seen::IdDict{Any,Any}) = 1
@inline _fcache_gradient_input_dof(x::Complex{<:IEEEFloat}, _seen::IdDict{Any,Any}) = 2
@inline function _fcache_gradient_input_dof(
    x::AbstractArray{<:IEEEFloat}, seen::IdDict{Any,Any}
)
    haskey(seen, x) && return 0
    _fcache_mark_seen!(seen, x)
    if x isa _BuiltinArrays
        total = 0
        for i in eachindex(x)
            isassigned(x, i) && (total += 1)
        end
        return total
    end
    return length(x)
end
@inline function _fcache_gradient_input_dof(
    x::AbstractArray{Complex{<:IEEEFloat}}, seen::IdDict{Any,Any}
)
    haskey(seen, x) && return 0
    _fcache_mark_seen!(seen, x)
    if x isa _BuiltinArrays
        total = 0
        for i in eachindex(x)
            isassigned(x, i) && (total += 2)
        end
        return total
    end
    return 2 * length(x)
end
@inline function _fcache_gradient_input_dof(x::AbstractArray, seen::IdDict{Any,Any})
    tangent_type(typeof(x)) == NoTangent && return 0
    haskey(seen, x) && return 0
    _fcache_mark_seen!(seen, x)
    total = 0
    if x isa _BuiltinArrays
        for i in eachindex(x)
            isassigned(x, i) || continue
            total += _fcache_gradient_input_dof(x[i], seen)
        end
    else
        for xi in x
            total += _fcache_gradient_input_dof(xi, seen)
        end
    end
    return total
end
@inline function _fcache_gradient_input_dof(x::Tuple, seen::IdDict{Any,Any})
    total = 0
    for xi in x
        total += _fcache_gradient_input_dof(xi, seen)
    end
    return total
end
@inline function _fcache_gradient_input_dof(x::NamedTuple, seen::IdDict{Any,Any})
    total = 0
    for xi in values(x)
        total += _fcache_gradient_input_dof(xi, seen)
    end
    return total
end
@inline function _fcache_gradient_input_dof(x::P, seen::IdDict{Any,Any}) where {P}
    tangent_type(P) == NoTangent && return 0
    if x isa AbstractArray || Base.ismutabletype(P)
        haskey(seen, x) && return 0
        _fcache_mark_seen!(seen, x)
    end
    total = 0
    inits = always_initialised(P)
    for n in 1:fieldcount(P)
        if isdefined(x, n)
            total += _fcache_gradient_input_dof(getfield(x, n), seen)
        elseif inits[n]
            _fcache_gradient_throw_uninit_field_error(P, n)
        end
    end
    return total
end

# fcache gradient seeding
@inline _fcache_gradient_seed_tangent(x, slot::Int) = _fcache_gradient_seed_tangent(
    x, slot, Ref(0), IdDict{Any,Any}()
)
@inline _fcache_gradient_seed_tangent(::NoTangent, _slot::Int, _cursor, _dict) = NoTangent()
@inline function _fcache_gradient_seed_tangent(
    ::NoTangent, _slot::Int, _cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    return NoTangent()
end
@inline function _fcache_gradient_seed_tangent(
    x::IEEEFloat, slot::Int, cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    cursor[] += 1
    return cursor[] == slot ? one(x) : zero(x)
end
@inline function _fcache_gradient_seed_tangent(
    x::Complex{T}, slot::Int, cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
) where {T<:IEEEFloat}
    cursor[] += 1
    real_part = cursor[] == slot ? one(T) : zero(T)
    cursor[] += 1
    imag_part = cursor[] == slot ? one(T) : zero(T)
    return complex(real_part, imag_part)
end

function _fcache_gradient_seed_tangent(
    x::AbstractArray{T}, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {T<:IEEEFloat}
    existing = get(dict, x, nothing)
    !isnothing(existing) && return existing
    dx = zero_tangent(x)
    dict[x] = dx
    @inbounds for I in eachindex(x)
        cursor[] += 1
        dx[I] = cursor[] == slot ? one(T) : zero(T)
    end
    return dx
end

function _fcache_gradient_seed_tangent(
    x::AbstractArray{Complex{T}},
    slot::Int,
    cursor::Base.RefValue{Int},
    dict::IdDict{Any,Any},
) where {T<:IEEEFloat}
    existing = get(dict, x, nothing)
    !isnothing(existing) && return existing
    dx = zero_tangent(x)
    dict[x] = dx
    @inbounds for I in eachindex(x)
        cursor[] += 1
        real_part = cursor[] == slot ? one(T) : zero(T)
        cursor[] += 1
        imag_part = cursor[] == slot ? one(T) : zero(T)
        dx[I] = complex(real_part, imag_part)
    end
    return dx
end

function _fcache_gradient_seed_tangent(
    x::AbstractArray, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    existing = get(dict, x, nothing)
    !isnothing(existing) && return existing
    dx = zero_tangent(x)
    dict[x] = dx
    for I in eachindex(x)
        dx[I] = _fcache_gradient_seed_tangent(x[I], slot, cursor, dict)
    end
    return dx
end

@inline function _fcache_gradient_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:Tuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(
        n -> _fcache_gradient_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P))
    )
    return build_tangent(P, fields...)
end

@inline function _fcache_gradient_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:NamedTuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(
        n -> _fcache_gradient_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P))
    )
    return build_tangent(P, fields...)
end

function _fcache_gradient_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P}
    tangent_type(P) == NoTangent && return NoTangent()
    if x isa AbstractArray || Base.ismutabletype(P)
        existing = get(dict, x, nothing)
        !isnothing(existing) && return existing
        tx = zero_tangent(x)
        dict[x] = tx
        if tx isa MutableTangent
            inits = always_initialised(P)
            for n in 1:fieldcount(P)
                if isdefined(x, n)
                    set_tangent_field!(
                        tx,
                        n,
                        _fcache_gradient_seed_tangent(getfield(x, n), slot, cursor, dict),
                    )
                elseif inits[n]
                    _fcache_gradient_throw_uninit_field_error(P, n)
                end
            end
        end
        return tx
    end

    inits = always_initialised(P)
    fields = ntuple(Val(fieldcount(P))) do n
        if isdefined(x, n)
            return _fcache_gradient_seed_tangent(getfield(x, n), slot, cursor, dict)
        elseif inits[n]
            _fcache_gradient_throw_uninit_field_error(P, n)
        else
            return PossiblyUninitTangent{tangent_type(fieldtype(P, n))}()
        end
    end
    return build_tangent(P, fields...)
end

# Shared `fcache` chunk machinery.
# Attempts to build a cache of chunked forward rules; returns `nothing` if any eligibility
# check fails.
#
# Eligibility (construction-time gates, evaluated in order):
# - `config.debug_mode` must be false.
# - `config.enable_nfwd` must be true. Despite the legacy config name, this now gates the
#   prepared-cache chunked forward backend in general.
# The prepared-cache backend stores width-specific `build_frule` rules. This keeps
# public chunked cache execution on the same chunked frontend as the direct rule API:
# derived code stays on the semantics-preserving chunked IR path, while primitive calls
# over nfwd-supported signatures may use NDual rules directly.
#
# Gradient-only shortcuts stored on the cache bundle (each narrower than
# the general chunk path):
# - `gradient_rrule`: built only for default-shape array gradients, where the
#   reverse cache remains the allocation-stable path for full gradients.
# - `frule_1` doubles as the scalar fast path for `(f, x::IEEEFloat)` calls via the
#   scalar `value_and_gradient!!` specialisation.
# - `small_vector_gradient_frule`: built only for single-argument
#   `(f, x::Vector{<:IEEEFloat})` with `1 <= length(x) <= 8` and, when a chunk_size
#   is requested, `length(x) <= chunk_size`. Uses an exact-width frule seeded with an
#   identity-matrix tangent.
@inline function _fcache_build_chunk_cache(fx::Tuple, config)
    config.debug_mode && return nothing
    getfield(config, :enable_nfwd) || return nothing
    sig = typeof(fx)
    params = Tuple(sig.parameters)
    requested_chunk_size = let requested = getfield(config, :chunk_size)
        isnothing(requested) ? 0 : Nfwd._nfwd_check_chunk_size(requested)
    end
    chunk_frules = ntuple(Val(_CHUNK_NFWD_MAX_LANES)) do n
        NfwdMooncake.build_frule(sig; chunk_size=n, silence_debug_messages=true)
    end
    # Small vectors are faster through an exact-width chunked frule than through the
    # generic chunked gradient assembly path, which otherwise seeds and accumulates lanes.
    small_vector_gradient_frule =
        if (
            length(params) == 2 &&
            params[2] <: Vector{<:IEEEFloat} &&
            1 <= length(last(fx)) <= _CHUNK_NFWD_MAX_LANES &&
            length(last(fx)) <=
            (requested_chunk_size == 0 ? _CHUNK_NFWD_MAX_LANES : requested_chunk_size)
        )
            getfield(chunk_frules, length(last(fx)))
        else
            nothing
        end
    gradient_rrule =
        if (
            requested_chunk_size == 0 &&
            first(fx) isa Function &&
            Base.issingletontype(typeof(first(fx))) &&
            length(params) == 2 &&
            params[2] <: Array{<:IEEEFloat}
        )
            # Keep this narrow: the direct nfwd reverse-rule path is appropriate for the
            # dense-array scalar-output fast path, but the broader public ForwardCache
            # contract must still preserve primal-dispatch behaviour for structured inputs.
            prepare_gradient_cache(fx...; config)
        else
            nothing
        end
    small_vector_gradient_buffer = if !isnothing(small_vector_gradient_frule)
        _fcache_small_vector_identity_seed(last(fx))
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
    return NfwdCache(
        chunk_frules...,
        gradient_rrule,
        small_vector_gradient_frule,
        small_vector_gradient_buffer,
        small_vector_gradient_workspace,
    )
end

# fcache derivative helpers
@generated function _fcache_derivative_ntangent_lane_count(ts::T) where {T<:Tuple}
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

# fcache forward architecture:
#
#   derivative machinery:
#     _fcache_derivative_chunked!! is the batched forward extension point.
#     The generic backend is _fcache_derivative_chunked_loop!!, which evaluates one
#     width-1 lane at a time through the cached frule and repacks the results as
#     `NTangent`.
#
#   gradient machinery:
#     _fcache_gradient_chunked!! seeds standard-basis `NTangent`s, calls
#     _fcache_derivative_chunked!! repeatedly, and accumulates the returned lane
#     contributions into gradient storage.
#     The scalar / small-vector / dense-array gradient fast paths bypass that generic
#     chunked gradient assembly path.
#
#   shared chunk machinery:
#     `ForwardCache.chunkcache` may hold one prebuilt chunked rule per supported width.
#     `_fcache_derivative_chunked!!` calls the matching rule directly when available,
#     and falls back to `_fcache_derivative_chunked_loop!!` only as a last resort.
#
# fcache derivative chunk execution
@noinline function _fcache_derivative_chunked_loop!!(
    cache::ForwardCache, ::Val{N}, x_dx::Vararg{Tuple,M}; friendly_tangents::Bool
) where {N,M}
    return _fcache_derivative_chunked_loop!!(cache, Val(N), Val(friendly_tangents), x_dx...)
end

@noinline function _fcache_derivative_chunked_loop!!(
    cache::ForwardCache, ::Val{N}, ::Val{friendly_tangents}, x_dx::Vararg{Tuple,M}
) where {N,friendly_tangents,M}
    # Canonical fallback backend for batched forward mode: evaluate one width-1 lane at a
    # time through the cached `frule!!` (aka ir-based forward) rule, then repack the
    # outputs into `NTangent`.
    # Specialized `_fcache_derivative_chunked!!` methods may replace this
    # with a true batched execution.
    input_primals = map(first, x_dx)
    input_tangents = map(last, x_dx)
    function compute_lane_output(lane::Int)
        lane_tangents = tuple_map(t -> _ntangent_lane(t, Val(lane)), input_tangents)
        return if friendly_tangents
            native_tangents = tuple_map(
                primal_to_tangent!!, cache.input_tangents, lane_tangents
            )
            cache.rule(tuple_map(_internal_forward_dual, input_primals, native_tangents)...)
        else
            lane_duals = tuple_map(_internal_forward_dual, input_primals, lane_tangents)
            error_if_incorrect_dual_types(lane_duals...)
            cache.rule(lane_duals...)
        end
    end
    lane_outputs = ntuple(compute_lane_output, Val(N))
    y = primal(first(lane_outputs))
    lane_tangents = ntuple(
        lane -> begin
            lane_output = lane_outputs[lane]
            return if friendly_tangents
                tangent_to_primal_internal!!(
                    _copy_output(y),
                    tangent(lane_output),
                    isbitstype(typeof(y)) ? NoCache() : IdDict{Any,Any}(),
                )
            else
                # Bug fix note: chunked forward can return `NoTangent()` lanes for
                # nondifferentiable outputs, and the generic `_copy` fallback does not
                # support `NoTangent`.
                lane_output_tangent = _unwrap_unit_ntangent(tangent(lane_output))
                if lane_output_tangent isa NoTangent
                    lane_output_tangent
                else
                    _copy(lane_output_tangent)
                end
            end
        end,
        Val(N),
    )

    return y, NTangent(lane_tangents)
end

@noinline function _fcache_derivative_chunked!!(
    cache::ForwardCache, ::Val{N}, x_dx::Vararg{Tuple,M}; friendly_tangents::Bool=false
) where {N,M}
    N < 1 && throw(ArgumentError("NTangent inputs must contain at least one lane."))
    if !friendly_tangents
        fastpath = cache.chunkcache
        if fastpath isa NfwdCache
            rule = _fcache_chunk_rule(fastpath, Val(N))
            !isnothing(rule) && return value_and_derivative!!(rule, x_dx...)
        end
    end
    return _fcache_derivative_chunked_loop!!(cache, Val(N), x_dx...; friendly_tangents)
end

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns a cache used with [`value_and_derivative!!`](@ref). See that function for more info.

!!! note
    Cache construction stays lazy and does not execute `f(x...)`, whether the prepared
    cache later runs through the IR-based `frule!!` path or an `Nfwd` fast path.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    fx = (f, x...)
    requested_chunk_size = getfield(config, :chunk_size)
    requested_chunk_size = if isnothing(requested_chunk_size)
        0
    else
        Nfwd._nfwd_check_chunk_size(requested_chunk_size)
    end
    gradient_chunk_size_auto = requested_chunk_size == 0
    chunkcache = _fcache_build_chunk_cache(fx, config)
    rule = build_frule(fx...; config.debug_mode, config.silence_debug_messages)
    input_specs = map(fx) do x
        if x isa AbstractArray
            PreparedCacheInputSpec(typeof(x), size(x))
        else
            PreparedCacheInputSpec(typeof(x), ())
        end
    end
    gradient_chunk_size = let total_dof = _fcache_gradient_input_dof(fx)
        if gradient_chunk_size_auto
            min(total_dof, _CHUNK_NFWD_MAX_LANES)
        else
            min(total_dof, requested_chunk_size)
        end
    end
    output_primal = nothing
    if config.friendly_tangents
        input_tangents = tuple_map(zero_tangent, fx)
        gradient_workspace = Ref{Union{Nothing,typeof(input_tangents)}}(nothing)
        return ForwardCache(
            rule,
            input_tangents,
            output_primal,
            _copy_output(fx),
            gradient_workspace,
            gradient_chunk_size,
            gradient_chunk_size_auto,
            chunkcache,
            input_specs,
        )
    end
    return ForwardCache(
        rule,
        nothing,
        output_primal,
        nothing,
        _fcache_gradient_lazy_workspace_ref(typeof(fx)),
        gradient_chunk_size,
        gradient_chunk_size_auto,
        chunkcache,
        input_specs,
    )
end

#
# `value_and_gradient!!` generic `_fcache_derivative_chunked!!` path
#
"""
    value_and_gradient!!(cache::ForwardCache, f, x...)

Compute the value and gradient of a scalar-returning function using the generic
`_fcache_derivative_chunked!!` path: seed standard-basis `NTangent`s,
call the batched forward interface, then accumulate the lane contributions into gradient
storage. Specialized backends behind `_fcache_derivative_chunked!!` may
pack/unpack those `NTangent`s into a different representation (for example NDual lanes),
but this generic path is expressed at the
`NTangent` boundary.

This overload exists so callers can prepare a forward cache once, then use it either for
directional derivatives via [`value_and_derivative!!`](@ref) or full gradients via chunked
forward mode.
"""
function _fcache_gradient_chunked!!(cache::ForwardCache, input_primals::Tuple)
    native_gradients = let workspace = cache.gradient_workspace[]
        if isnothing(workspace)
            workspace = tuple_map(zero_tangent, input_primals)
            cache.gradient_workspace[] = workspace
            workspace
        else
            zeroed = tuple_map(set_to_zero!!, workspace)
            cache.gradient_workspace[] = zeroed
            zeroed
        end
    end
    total_dof = _fcache_gradient_input_dof(input_primals)
    y = nothing

    if total_dof == 0
        output = cache.rule(
            tuple_map(_internal_forward_dual, input_primals, native_gradients)...
        )
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    else
        chunk_size = cache.gradient_chunk_size
        for start_slot in 1:chunk_size:total_dof
            chunk_width = min(chunk_size, total_dof - start_slot + 1)
            lane_tangents = ntuple(
                lane -> _fcache_gradient_seed_tangent(input_primals, start_slot + lane - 1),
                chunk_width,
            )
            input_tangents = ntuple(
                i -> begin
                    tangent_type(typeof(input_primals[i])) == NoTangent &&
                        return NoTangent()
                    return NTangent(ntuple(lane -> lane_tangents[lane][i], chunk_width))
                end,
                Val(fieldcount(typeof(input_primals))),
            )
            # `value_and_gradient!!` is a client of the batched forward interface: it seeds
            # standard-basis chunk tangents, calls `_fcache_derivative_chunked!!`, and
            # accumulates the resulting lane contributions into gradient storage.
            y_chunk, chunk_dy = _fcache_derivative_chunked!!(
                cache,
                Val(chunk_width),
                map(tuple, input_primals, input_tangents)...;
                friendly_tangents=false,
            )
            if isnothing(y)
                y = y_chunk
                y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
            end
            # A scalar output turns each derivative lane into one coefficient for the corresponding
            # seeded basis direction, so accumulate `coeff * lane_tangent` into the full gradient.
            for lane in 1:chunk_width
                coeff = Float64(_unwrap_unit_ntangent(chunk_dy[lane]))
                native_gradients = tuple_map(
                    (g, dx) -> begin
                        dx isa NoTangent && return g
                        lane_tangent = _ntangent_lane(dx, Val(lane))
                        lane_tangent isa NoTangent && return g
                        return increment!!(g, _scale(coeff, lane_tangent))
                    end,
                    native_gradients,
                    input_tangents,
                )
            end
        end
    end

    if isnothing(cache.input_tangents)
        return y::IEEEFloat, native_gradients
    end
    return y::IEEEFloat, _fcache_gradient_output(cache, input_primals, native_gradients)
end

@inline function _fcache_gradient_output(
    cache::ForwardCache, input_primals, native_gradients
)
    friendly_gradients = _copy_to_output!!(cache.friendly_gradients, input_primals)
    return tangent_to_primal_internal!!(
        friendly_gradients,
        native_gradients,
        isbitstype(typeof(friendly_gradients)) ? NoCache() : IdDict{Any,Any}(),
    )
end

#
# `value_and_gradient!!` fast paths
#
# ForwardCache path overview:
# - derivative machinery:
#   `value_and_derivative!!`, `_fcache_derivative_chunked!!`.
# - gradient machinery:
#   `value_and_gradient!!`, `_fcache_gradient_chunked!!`.
# - shared nfwd chunk machinery:
#   `_fcache_build_chunk_cache`,
#   `_is_ndual_unsupported_error`.
#
# Gradient dispatch summary for `value_and_gradient!!(cache, f, x...)`:
# - `x::IEEEFloat`: scalar width-1 path
# - `x::Vector{<:IEEEFloat}`: small-vector path
# - `x::Array{<:IEEEFloat}`: dense-array path
# - otherwise: generic vararg path, which seeds standard-basis `NTangent` chunks and
#   repeatedly calls `_fcache_derivative_chunked!!`

@inline function value_and_gradient!!(
    cache::ForwardCache, f::F, x::T
) where {F,T<:IEEEFloat}
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), (f, x))
    fastpath = cache.chunkcache
    if !isnothing(fastpath) && !isnothing(fastpath.frule_1)
        y, dy = value_and_derivative!!(fastpath.frule_1, (f, NoTangent()), (x, one(x)))
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        native_gradients = (NoTangent(), _unwrap_unit_ntangent(dy))
        if isnothing(cache.input_tangents)
            return y, native_gradients
        end
        return y, _fcache_gradient_output(cache, (f, x), native_gradients)
    end
    return _fcache_gradient_chunked!!(cache, (f, x))
end

@inline function value_and_gradient!!(
    cache::ForwardCache, f::F, x::V
) where {F,T<:IEEEFloat,V<:Vector{T}}
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), (f, x))
    fastpath = cache.chunkcache
    if !isnothing(fastpath) && !isnothing(fastpath.gradient_rrule)
        return value_and_gradient!!(fastpath.gradient_rrule, f, x)
    end
    if !isnothing(fastpath) && !isnothing(fastpath.small_vector_gradient_frule)
        rule = fastpath.small_vector_gradient_frule
        _fcache_small_vector_fill_identity!(fastpath.small_vector_gradient_buffer)
        y, output_tangent = value_and_derivative!!(
            rule, (f, NoTangent()), (x, fastpath.small_vector_gradient_buffer)
        )
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        native_gradients = fastpath.small_vector_gradient_workspace
        if output_tangent isa NTangent
            @inbounds for i in 1:NfwdMooncake.rule_chunk_size(typeof(rule))
                native_gradients[2][i] = output_tangent[i]
            end
        else
            native_gradients[2][1] = output_tangent
        end
        if isnothing(cache.input_tangents)
            return y, native_gradients
        end
        return y, _fcache_gradient_output(cache, (f, x), native_gradients)
    end
    return _fcache_gradient_chunked!!(cache, (f, x))
end

# Array `value_and_gradient!!` path: use the generic chunked gradient assembly.
@inline function value_and_gradient!!(
    cache::ForwardCache, f::F, x::A
) where {F,A<:Array{<:IEEEFloat}}
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), (f, x))
    fastpath = cache.chunkcache
    if !isnothing(fastpath) && !isnothing(fastpath.gradient_rrule)
        return value_and_gradient!!(fastpath.gradient_rrule, f, x)
    end
    return _fcache_gradient_chunked!!(cache, (f, x))
end

function value_and_gradient!!(cache::ForwardCache, f::F, x::Vararg{Any,N}) where {F,N}
    input_primals = (f, x...)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    return _fcache_gradient_chunked!!(cache, input_primals)
end

"""
    value_and_derivative!!(cache::ForwardCache, f::Dual, x::Vararg{Dual,N})

Returns a `Dual` containing the result of applying forward-mode AD to compute the (Frechet)
derivative of `primal(f)` at the primal values in `x` in the direction of the tangent values
in `f` and `x`.
"""
# Derivative dispatch summary for `value_and_derivative!!(cache, ...)`:
# - `value_and_derivative!!(cache, duals...)`: native/internal tangent interface;
#   calls the cached `frule` directly
# - `value_and_derivative!!(cache, (f, df), (x, dx), ...)`: tuple interface
# - tuple inputs whose tangents are `NTangent`s use the chunked forward path:
#   `_fcache_derivative_chunked!!(cache, Val(N), ...)`, where `N` is the
#   lane count
# - tuple inputs whose tangents are ordinary width-1 tangents are first wrapped into
#   `Dual(primal, tangent)` values, then passed to the cached `frule`
function value_and_derivative!!(cache::ForwardCache, fx::Vararg{Dual,N}) where {N}
    input_primals = map(primal, fx)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    if any(x -> tangent(x) isa NTangent, fx)
        y, dy = value_and_derivative!!(cache, map(extract, fx)...)
        return Dual(y, dy)
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
@inline function value_and_derivative!!(
    cache::ForwardCache, fx::Vararg{Tuple{Any,Any},M}
) where {M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    input_external_tangents = tuple_map(last, fx)
    friendly_tangents = !isnothing(cache.input_tangents)
    N_val = _fcache_derivative_ntangent_lane_count(input_external_tangents)
    !isnothing(N_val) && return _fcache_derivative_chunked!!(
        cache,
        N_val,
        map(tuple, input_primals, input_external_tangents)...;
        friendly_tangents,
    )
    input_tangents = if friendly_tangents
        tuple_map(primal_to_tangent!!, cache.input_tangents, input_external_tangents)
    else
        input_external_tangents
    end
    N_val = _fcache_derivative_ntangent_lane_count(input_tangents)
    !isnothing(N_val) && return _fcache_derivative_chunked!!(
        cache, N_val, map(tuple, input_primals, input_tangents)...; friendly_tangents
    )
    input_duals = tuple_map(_internal_forward_dual, input_primals, input_tangents)
    !friendly_tangents && error_if_incorrect_dual_types(input_duals...)
    output = __call_rule(cache.rule, input_duals)
    output_primal = primal(output)
    return if friendly_tangents
        output_friendly_tangent = tangent_to_friendly!!(
            friendly_tangent_cache(output_primal),
            output_primal,
            tangent(output),
            _friendly_cache((output_primal,)),
        )
        output_primal, output_friendly_tangent
    else
        output_primal, tangent(output)
    end
end

# `fwd_cache` is the derivative cache for `grad_f`. The compiled inner rrule is cached
# across `value_and_hvp!!` calls via a `LazyFoRRule` captured inside `fwd_cache`'s frule.
"""
    HVPCache

Cache type used by [`prepare_hvp_cache`](@ref) and [`prepare_hessian_cache`](@ref) for
repeated Hessian-vector product and Hessian evaluations.
"""
struct HVPCache{Tf,Tgrad_f,Tgrad_tangent,Tfwd_cache,TOS}
    f::Tf
    grad_f::Tgrad_f
    # Pre-computed zero tangent for grad_f; the function is never perturbed, only x is.
    # Safe to reuse because grad_f's closure environment is shape-stable for the lifetime
    # of the cache: grad_cache mutates stored values between calls but does not change the
    # closure/capture structure that zero_tangent depends on.
    grad_tangent::Tgrad_tangent
    fwd_cache::Tfwd_cache
    output_spec::TOS
end

function Base.show(io::IO, cache::HVPCache)
    print(
        io,
        "Mooncake.HVPCache(",
        "mode=:forward_over_reverse, ",
        "chunked=",
        !isnothing(getfield(getfield(cache, :fwd_cache), :chunkcache)),
        ", ",
        "inputs=",
        _cache_input_count(getfield(cache, :fwd_cache)),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::HVPCache)
    print(
        io,
        "Mooncake.HVPCache\n",
        "  mode: forward_over_reverse\n",
        "  chunked: ",
        !isnothing(getfield(getfield(cache, :fwd_cache), :chunkcache)),
        "\n",
        "  inputs: ",
        _cache_input_count(getfield(cache, :fwd_cache)),
    )
    _cache_print_io_summary(
        io,
        Base.tail(getfield(getfield(cache, :fwd_cache), :input_specs)),
        _cache_spec_summary(getfield(cache, :output_spec)),
    )
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

!!! note
    Calls `f(x...)` during cache preparation (via inner gradient and derivative caches).

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
        y -> begin
            val_and_grad = value_and_gradient!!(grad_cache, f, y)
            (val_and_grad[1], val_and_grad[2][2])
        end
    else
        function (ys...)
            val_and_grad = value_and_gradient!!(grad_cache, f, ys...)
            # Drop the gradient w.r.t. f itself (always index 1); return only x-arg gradients.
            (val_and_grad[1], Base.tail(val_and_grad[2]))
        end
    end
    fwd_cache = prepare_derivative_cache(grad_f, x...; config)
    return HVPCache(
        f, grad_f, zero_tangent(grad_f), fwd_cache, getfield(grad_cache, :output_spec)
    )
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
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
    _validate_prepared_cache_inputs(
        getfield(cache.fwd_cache, :input_specs), (cache.grad_f, x1)
    )
    _assert_matching_tangent_shape(x1, v, 1)
    (f_val, grad), output_tangent = value_and_derivative!!(
        cache.fwd_cache, (cache.grad_f, cache.grad_tangent), (x1, v)
    )
    _, hvp = _unwrap_unit_ntangent(output_tangent)
    return f_val, grad, hvp
end

@inline function value_and_hvp!!(
    cache::HVPCache, f::F, v::Tuple, x1::T1, xrest::Vararg{Any,N}
) where {F,T1,N}
    all_x = (x1, xrest...)
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
    input_primals = (cache.grad_f, all_x...)
    _validate_prepared_cache_inputs(getfield(cache.fwd_cache, :input_specs), input_primals)
    length(v) == length(all_x) ||
        throw(ArgumentError("Expected one tangent direction per primal argument"))
    for i in eachindex(all_x)
        _assert_matching_tangent_shape(all_x[i], v[i], i)
    end
    (f_val, grads), output_tangent = value_and_derivative!!(
        cache.fwd_cache, (cache.grad_f, cache.grad_tangent), map(tuple, all_x, v)...
    )
    _, hvps = _unwrap_unit_ntangent(output_tangent)
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

!!! note
    This path currently uses Mooncake's generic public forward cache over the captured
    reverse-mode gradient closure. It does not currently dispatch to the public
    `NfwdMooncake` fast path used by some `prepare_derivative_cache` /
    `value_and_gradient!!` calls.

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
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
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
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
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

# IT=Nothing specialisation: disambiguates against the Dual-vararg and Tuple-vararg
# zero-arg overloads (Aqua detects the ambiguity without this more-specific method).
function value_and_derivative!!(
    cache::ForwardCache{R,Nothing,OP,FG,GW,CF,S}
) where {R,OP,FG,GW,CF,S<:Tuple}
    _validate_prepared_cache_inputs(cache.input_specs, ())
    error("unreachable")
end

function value_and_derivative!!(cache::ForwardCache)
    _validate_prepared_cache_inputs(cache.input_specs, ())
    error("unreachable")
end
