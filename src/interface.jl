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

```jldoctest; setup = :(using Mooncake; import Mooncake: build_rrule, zero_tangent)
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
```jldoctest; setup = :(using Mooncake; import Mooncake: build_rrule)
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

Run a forward rule directly, without first constructing a `NfwdCache`.

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
    return __call_rule(rule, fx)
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Tuple{Any,Any},N}) where {R,N}
    input_primals = tuple_map(first, fx)
    input_tangents = tuple_map(last, fx)
    input_duals = tuple_map(Dual, input_primals, input_tangents)
    output = __call_rule(rule, input_duals)
    return primal(output), tangent(output)
end

# Cache types in this file:
# - `Cache`: reusable reverse-mode cache for repeated `value_and_pullback!!` and
#   `value_and_gradient!!` calls.
# - `HVPCache`: reusable forward-over-reverse cache for repeated `value_and_hvp!!` calls;
#   Hessian helpers reuse this cache rather than introducing a separate Hessian cache type.
# Internal helper cache types in this file:
# - `NfwdCache`: internal nfwd helper cache stored inside `NfwdCache` when the
#   prepared forward cache can use packed NDual execution.
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

    # Clear global caches if requested.
    config.empty_cache && empty_mooncake_caches!()

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
```jldoctest; setup = :(using Mooncake)
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
    config.empty_cache && empty_mooncake_caches!()
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
```jldoctest; setup = :(using Mooncake)
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

"""
    NTangent(lanes)

Explicit wrapper for chunked forward-mode tangents at the interface boundary.

Each element of `lanes` must itself be a valid width-1 tangent in the corresponding API
mode. Mooncake repacks chunked results in another `NTangent`, and uses an NDual-backed
single-pass fast path when the runtime values fit `nfwd`'s supported primal space.
"""
struct NTangent{L<:Tuple}
    lanes::L
end

Base.length(x::NTangent) = length(x.lanes)
Base.getindex(x::NTangent, i::Int) = x.lanes[i]
Base.iterate(x::NTangent, st...) = iterate(x.lanes, st...)

const _CHUNK_NFWD_MAX_LANES = 8

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

# Shared prepared-cache input validation for Cache, NfwdCache, and HVPCache entry points.
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

# ── Forward-mode NfwdCache ─────────────────────────────────────────────────────

"""
    NfwdCache

Forward-mode cache returned by [`prepare_derivative_cache`](@ref).

Holds pre-built nfwd rules for chunk widths 1–8 plus gradient workspace and input specs.

# Supported input types

The callable `f` must be a **singleton type** (zero fields): named functions like `sin` or
`MyModule.my_f` work; closures and callable structs with captured state do not.

Each positional argument `x` must be one of:

- `IEEEFloat` scalar (`Float16`, `Float32`, `Float64`)
- `Complex{<:IEEEFloat}` scalar
- `Array{T}` where `T` is one of the scalar types above
- `Tuple` of any combination of the above (recursively)

Everything else (structs, `NamedTuple`, `BitArray`, integer types, etc.) is rejected at
cache-construction time with an `UnsupportedInputError`.

The function output must also be one of the supported types above; an `UnsupportedOutputError`
is thrown at evaluation time otherwise.
"""
struct NfwdCache{R1,R2,R3,R4,R5,R6,R7,R8,PB,GW,S<:Tuple}
    friendly_tangents::Bool
    frule_1::R1
    frule_2::R2
    frule_3::R3
    frule_4::R4
    frule_5::R5
    frule_6::R6
    frule_7::R7
    frule_8::R8
    pack_buffers::PB
    gradient_workspace::GW
    gradient_chunk_size::Int
    input_specs::S
end

@inline _dual_primal_type(::Type) = Any
@inline _dual_primal_type(::Type{Dual{Y,T}}) where {Y,T} = Y

@inline function _nfwd_cache_output_summary(cache::NfwdCache)
    rule = getfield(cache, :frule_1)
    dual_arg_types = Tuple{
        map(spec -> dual_type(typeof(spec).parameters[1]), getfield(cache, :input_specs))...
    }
    output_type = Core.Compiler.return_type(rule, dual_arg_types)
    return _cache_type_summary(_dual_primal_type(output_type))
end

function Base.show(io::IO, cache::NfwdCache)
    print(
        io,
        "Mooncake.NfwdCache(",
        "mode=:forward, ",
        "chunk_size=",
        getfield(cache, :gradient_chunk_size),
        ", inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::NfwdCache)
    print(
        io,
        "Mooncake.NfwdCache\n",
        "  mode: forward\n",
        "  chunk_size: ",
        getfield(cache, :gradient_chunk_size),
        "\n",
        "  inputs: ",
        _cache_input_count(cache),
    )
    _cache_print_io_summary(
        io, Base.tail(getfield(cache, :input_specs)), _nfwd_cache_output_summary(cache)
    )
end

# For nfwd types, tangent_type === primal_type, so the only "friendly" conversion needed
# is swapping the function slot from NoTangent() to the function object itself.
@inline function _nfwd_maybe_friendly(
    cache::NfwdCache, input_primals::Tuple, native_grads::Tuple
)
    cache.friendly_tangents || return native_grads
    return (input_primals[1], Base.tail(native_grads)...)
end

@generated function _fcache_gradient_lazy_workspace_ref(::Type{T}) where {T<:Tuple}
    tangent_types = map(P -> :(tangent_type($P)), T.parameters)
    workspace_type = Expr(:curly, :Tuple, tangent_types...)
    return :(Ref{Union{Nothing,$workspace_type}}(nothing))
end

# ── fcache gradient bookkeeping ───────────────────────────────────────────────────
#
# Only nfwd-supported types: IEEEFloat, Complex{IEEEFloat}, Array of those, Tuple thereof.

@inline _fcache_gradient_input_dof(x) = _fcache_gradient_input_dof(x, IdDict{Any,Any}())
@inline _fcache_gradient_input_dof(::NoTangent, _seen::IdDict{Any,Any}) = 0
@inline _fcache_gradient_input_dof(x::IEEEFloat, _seen::IdDict{Any,Any}) = 1
@inline _fcache_gradient_input_dof(x::Complex{<:IEEEFloat}, _seen::IdDict{Any,Any}) = 2
@inline function _fcache_gradient_input_dof(x::Array{<:IEEEFloat}, seen::IdDict{Any,Any})
    haskey(seen, x) && return 0
    seen[x] = nothing
    return length(x)
end
@inline function _fcache_gradient_input_dof(
    x::Array{<:Complex{<:IEEEFloat}}, seen::IdDict{Any,Any}
)
    haskey(seen, x) && return 0
    seen[x] = nothing
    return 2 * length(x)
end
@inline function _fcache_gradient_input_dof(x::Tuple, seen::IdDict{Any,Any})
    total = 0
    for xi in x
        total += _fcache_gradient_input_dof(xi, seen)
    end
    return total
end
# Catch-all for non-differentiable types (function slots, etc.)
@inline function _fcache_gradient_input_dof(x, seen::IdDict{Any,Any})
    tangent_type(typeof(x)) == NoTangent && return 0
    error("Unsupported type for forward-mode gradient: $(typeof(x))")
end

# ── fcache gradient seeding ───────────────────────────────────────────────────────
#
# Only nfwd-supported types: IEEEFloat, Complex{IEEEFloat}, Array of those, Tuple thereof.

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
    x::Array{T}, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
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
    x::Array{Complex{T}}, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
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

@inline function _fcache_gradient_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:Tuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(
        n -> _fcache_gradient_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P))
    )
    return build_tangent(P, fields...)
end
# Catch-all for non-differentiable types (function slots, etc.)
@inline function _fcache_gradient_seed_tangent(
    x, _slot::Int, _cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    error("Unsupported type for forward-mode gradient seeding: $(typeof(x))")
end

# ── NfwdCache chunk building ──────────────────────────────────────────────────────
# Builds nfwd rules at widths 1–8 and the gradient fast-path helpers.
@inline function _fcache_build_nfwd_chunk_cache(fx::Tuple, config)
    config.debug_mode && return nothing
    getfield(config, :enable_nfwd) || return nothing
    sig = typeof(fx)
    params = Tuple(sig.parameters)
    requested_chunk_size = let requested = getfield(config, :chunk_size)
        isnothing(requested) ? 0 : Nfwd._nfwd_check_chunk_size(requested)
    end
    F = params[1]
    Base.issingletontype(F) || return nothing
    all(Base.tail(params)) do P
        P <: IEEEFloat && return true
        P <: Complex{<:IEEEFloat} && return true
        P <: Array || return false
        return Nfwd._nfwd_is_supported_scalar(P.parameters[1])
    end || return nothing
    frule_1 = NfwdMooncake.build_frule(
        sig; chunk_size=1, debug_mode=false, silence_debug_messages=true
    )
    frule_2 = NfwdMooncake.build_frule(
        sig; chunk_size=2, debug_mode=false, silence_debug_messages=true
    )
    frule_3 = NfwdMooncake.build_frule(
        sig; chunk_size=3, debug_mode=false, silence_debug_messages=true
    )
    frule_4 = NfwdMooncake.build_frule(
        sig; chunk_size=4, debug_mode=false, silence_debug_messages=true
    )
    frule_5 = NfwdMooncake.build_frule(
        sig; chunk_size=5, debug_mode=false, silence_debug_messages=true
    )
    frule_6 = NfwdMooncake.build_frule(
        sig; chunk_size=6, debug_mode=false, silence_debug_messages=true
    )
    frule_7 = NfwdMooncake.build_frule(
        sig; chunk_size=7, debug_mode=false, silence_debug_messages=true
    )
    frule_8 = NfwdMooncake.build_frule(
        sig; chunk_size=8, debug_mode=false, silence_debug_messages=true
    )
    pack_buffers = tuple_map(_nfwd_pack_buffer_for, fx)
    return (;
        frule_1,
        frule_2,
        frule_3,
        frule_4,
        frule_5,
        frule_6,
        frule_7,
        frule_8,
        pack_buffers,
    )
end

@inline _nfwd_pack_buffer_for(::IEEEFloat) = nothing
@inline _nfwd_pack_buffer_for(::Complex{<:IEEEFloat}) = nothing
@inline function _nfwd_pack_buffer_for(x::Array)
    Ref{Union{Nothing,Array{eltype(x)}}}(nothing)
end
@inline _nfwd_pack_buffer_for(x::Tuple) = tuple_map(_nfwd_pack_buffer_for, x)
@inline _nfwd_pack_buffer_for(_) = nothing  # function slots, non-differentiable types

# ── chunk pack tangent helpers ────────────────────────────────────────────────────

@inline function _chunk_pack_tangent(::IEEEFloat, dx::NTangent, _buf, ::Val{N}) where {N}
    return ntuple(k -> dx[k], Val(N))
end
@inline function _chunk_pack_tangent(::IEEEFloat, dx, _buf, ::Val{N}) where {N}
    return ntuple(_ -> dx, Val(N))
end

# Fallback for non-differentiable types (e.g. function slots).
@inline function _chunk_pack_tangent(
    _, dx::NTangent{<:Tuple{Vararg{NoTangent}}}, _buf, ::Val{N}
) where {N}
    return NoTangent()
end
@inline function _chunk_pack_tangent(_, dx::NoTangent, _buf, ::Val{N}) where {N}
    return NoTangent()
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
    buf = buf_ref[]
    wanted = (size(x)..., C)
    if !(buf isa Array{T} && size(buf) == wanted)
        buf = Array{T}(undef, wanted)
        buf_ref[] = buf
    end
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
    buf = buf_ref[]
    wanted = (size(x)..., C)
    if !(buf isa Array{T} && size(buf) == wanted)
        buf = Array{T}(undef, wanted)
        buf_ref[] = buf
    end
    @inbounds for I in CartesianIndices(x)
        idx = Tuple(I)
        value = dx[I]
        for lane in 1:C
            buf[idx..., lane] = value
        end
    end
    return buf
end

@inline function _chunk_pack_tangent(
    x::Tuple, dx::NTangent, bufs::Tuple, ::Val{N}
) where {N}
    return ntuple(
        i -> _chunk_pack_tangent(
            x[i], NTangent(ntuple(lane -> dx[lane][i], Val(N))), bufs[i], Val(N)
        ),
        Val(fieldcount(typeof(x))),
    )
end

@inline function _chunk_pack_tangent(x::Tuple, dx::Tuple, bufs::Tuple, ::Val{N}) where {N}
    return ntuple(
        i -> _chunk_pack_tangent(x[i], dx[i], bufs[i], Val(N)), Val(fieldcount(typeof(x)))
    )
end

# ── NTangent lane count and chunked derivative dispatch ───────────────────────────

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

    return isnothing(lane_count) ? :(nothing) : :(Val{$lane_count}())
end

@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{1})
    return cache.frule_1
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{2})
    return cache.frule_2
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{3})
    return cache.frule_3
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{4})
    return cache.frule_4
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{5})
    return cache.frule_5
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{6})
    return cache.frule_6
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{7})
    return cache.frule_7
end
@inline function _nfwd_select_rule(cache::NfwdCache, ::Val{8})
    return cache.frule_8
end

@noinline function _fcache_derivative_chunked!!(
    cache::NfwdCache, ::Val{N}, x_dx::Vararg{Tuple,M}
) where {N,M}
    N < 1 && throw(ArgumentError("NTangent inputs must contain at least one lane."))
    input_primals = map(first, x_dx)
    input_tangents = map(last, x_dx)
    if isnothing(cache.pack_buffers)
        # Fallback: lane-by-lane using width-1 rule (debug mode)
        return _fcache_derivative_lane_loop!!(
            cache.frule_1, Val(N), input_primals, input_tangents
        )
    end
    rule = _nfwd_select_rule(cache, Val(N))
    packed_tangents = ntuple(Val(M)) do i
        _chunk_pack_tangent(input_primals[i], input_tangents[i], cache.pack_buffers[i], Val(N))
    end
    output = rule(map(Dual, input_primals, packed_tangents)...)
    y = primal(output)
    dy = tangent(output)
    if dy isa Tuple
        output_tangent = NTangent(dy)
    elseif dy isa NoTangent
        output_tangent = NTangent(ntuple(_ -> NoTangent(), Val(N)))
    else
        output_tangent = NTangent(ntuple(_ -> dy, Val(N)))
    end
    return y, output_tangent
end

# Lane-by-lane fallback for when pack_buffers is unavailable (debug mode).
@noinline function _fcache_derivative_lane_loop!!(
    rule, ::Val{N}, input_primals::Tuple, input_tangents::Tuple
) where {N}
    function compute_lane(::Val{lane}) where {lane}
        lane_tangents = tuple_map(t -> t isa NTangent ? t[lane] : t, input_tangents)
        lane_duals = tuple_map(Dual, input_primals, lane_tangents)
        return rule(lane_duals...)
    end
    first_output = compute_lane(Val(1))
    y = primal(first_output)
    first_tangent = let t = tangent(first_output)
        t isa NoTangent ? t : _copy(t)
    end
    rest_tangents = ntuple(Val(N - 1)) do n
        lane_output = compute_lane(Val(n + 1))
        t = tangent(lane_output)
        return t isa NoTangent ? t : _copy(t)
    end
    return y, NTangent((first_tangent, rest_tangents...))
end

# ── prepare_derivative_cache ──────────────────────────────────────────────────────

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns a cache used with [`value_and_derivative!!`](@ref). See that function for more info.

See [`NfwdCache`](@ref) for the full list of supported input and output types.

!!! note
    Cache construction stays lazy and does not execute `f(x...)`, whether the prepared
    cache later runs through the generic lane loop or an nfwd fast path.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    config.empty_cache && empty_mooncake_caches!()
    fx = (f, x...)
    # Validate that f and all args are nfwd-supported, since nfwd is the only backend.
    F = typeof(f)
    Base.issingletontype(F) || throw(
        ArgumentError(
            "Forward-mode `prepare_derivative_cache` requires a singleton (stateless) " *
            "callable; got `$F` which has $(fieldcount(F)) field(s). " *
            "Closures and callable structs with captured state are not supported.",
        ),
    )
    for xi in x
        Nfwd._nfwd_is_supported_primal(xi) || Nfwd._nfwd_input_error(xi)
    end
    requested_chunk_size = getfield(config, :chunk_size)
    requested_chunk_size = if isnothing(requested_chunk_size)
        0
    else
        Nfwd._nfwd_check_chunk_size(requested_chunk_size)
    end
    chunk_fields = _fcache_build_nfwd_chunk_cache(fx, config)
    input_specs = map(fx) do x
        if x isa AbstractArray
            PreparedCacheInputSpec(typeof(x), size(x))
        else
            PreparedCacheInputSpec(typeof(x), ())
        end
    end
    gradient_chunk_size = let total_dof = _fcache_gradient_input_dof(fx)
        if requested_chunk_size == 0
            min(total_dof, _CHUNK_NFWD_MAX_LANES)
        else
            min(total_dof, requested_chunk_size)
        end
    end
    gradient_workspace = _fcache_gradient_lazy_workspace_ref(typeof(fx))
    friendly = config.friendly_tangents
    if isnothing(chunk_fields)
        # nfwd not available (debug mode or enable_nfwd=false). Build width-1 rule only.
        rule = NfwdMooncake.build_frule(
            typeof(fx); chunk_size=1, debug_mode=false, silence_debug_messages=true
        )
        if config.debug_mode
            rule = DebugFRule(rule)
        end
        return NfwdCache(
            friendly,
            rule,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            gradient_workspace,
            gradient_chunk_size,
            input_specs,
        )
    end
    frule_1 = chunk_fields.frule_1
    if config.debug_mode
        frule_1 = DebugFRule(frule_1)
    end
    return NfwdCache(
        friendly,
        frule_1,
        chunk_fields.frule_2,
        chunk_fields.frule_3,
        chunk_fields.frule_4,
        chunk_fields.frule_5,
        chunk_fields.frule_6,
        chunk_fields.frule_7,
        chunk_fields.frule_8,
        chunk_fields.pack_buffers,
        gradient_workspace,
        gradient_chunk_size,
        input_specs,
    )
end

# ── value_and_gradient!! generic chunked path ─────────────────────────────────────

function _fcache_gradient_chunked!!(cache::NfwdCache, input_primals::Tuple)
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

    rule = cache.frule_1
    if total_dof == 0
        output = rule(tuple_map(Dual, input_primals, native_gradients)...)
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        return y, _nfwd_maybe_friendly(cache, input_primals, native_gradients)
    end

    chunk_size = cache.gradient_chunk_size
    first_chunk_width = min(chunk_size, total_dof)
    first_lane_tangents = ntuple(
        lane -> _fcache_gradient_seed_tangent(input_primals, lane), first_chunk_width
    )
    first_input_tangents = ntuple(
        i -> NTangent(ntuple(lane -> first_lane_tangents[lane][i], first_chunk_width)),
        Val(fieldcount(typeof(input_primals))),
    )
    y, first_chunk_dy = _fcache_derivative_chunked!!(
        cache, Val(first_chunk_width), map(tuple, input_primals, first_input_tangents)...
    )
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    for lane in 1:first_chunk_width
        coeff = Float64(first_chunk_dy[lane])
        native_gradients = tuple_map(
            (g, dx) -> begin
                lane_tangent = dx[lane]
                lane_tangent isa NoTangent && return g
                return increment!!(g, _scale(coeff, lane_tangent))
            end,
            native_gradients,
            first_input_tangents,
        )
    end

    for start_slot in (1 + chunk_size):chunk_size:total_dof
        chunk_width = min(chunk_size, total_dof - start_slot + 1)
        lane_tangents = ntuple(
            lane -> _fcache_gradient_seed_tangent(input_primals, start_slot + lane - 1),
            chunk_width,
        )
        input_tangents = ntuple(
            i -> NTangent(ntuple(lane -> lane_tangents[lane][i], chunk_width)),
            Val(fieldcount(typeof(input_primals))),
        )
        _, chunk_dy = _fcache_derivative_chunked!!(
            cache, Val(chunk_width), map(tuple, input_primals, input_tangents)...
        )
        for lane in 1:chunk_width
            coeff = Float64(chunk_dy[lane])
            native_gradients = tuple_map(
                (g, dx) -> begin
                    lane_tangent = dx[lane]
                    lane_tangent isa NoTangent && return g
                    return increment!!(g, _scale(coeff, lane_tangent))
                end,
                native_gradients,
                input_tangents,
            )
        end
    end

    return y, _nfwd_maybe_friendly(cache, input_primals, native_gradients)
end

# Generic vararg
function value_and_gradient!!(cache::NfwdCache, f::F, x::Vararg{Any,N}) where {F,N}
    input_primals = (f, x...)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    return _fcache_gradient_chunked!!(cache, input_primals)
end

# ── value_and_derivative!! methods for NfwdCache ──────────────────────────────────

"""
    value_and_derivative!!(cache::NfwdCache, f::Dual, x::Vararg{Dual,N})

Returns a `Dual` containing the result of applying forward-mode AD to compute the (Frechet)
derivative of `primal(f)` at the primal values in `x` in the direction of the tangent values
in `f` and `x`.
"""
function value_and_derivative!!(cache::NfwdCache, fx::Vararg{Dual,N}) where {N}
    input_primals = map(primal, fx)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    if any(x -> tangent(x) isa NTangent, fx)
        throw(
            ArgumentError(
                "NTangent inputs are currently supported via the tuple interface " *
                "only. Use `value_and_derivative!!(cache, (f, df), (x, dx), ...)`.",
            ),
        )
    end
    return __call_rule(cache.frule_1, fx)
end

"""
    value_and_derivative!!(cache::NfwdCache, (f, df), (x, dx), ...)

Returns a tuple `(y, dy)` containing the result of applying forward-mode AD to compute the
(Frechet) derivative of `primal(f)` at the primal values in `x` in the direction of the
tangent values contained in `df` and `dx`.
"""
@inline function value_and_derivative!!(
    cache::NfwdCache, fx::Vararg{Tuple{Any,Any},M}
) where {M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    input_tangents = tuple_map(last, fx)
    N_val = _fcache_derivative_ntangent_lane_count(input_tangents)
    if !isnothing(N_val)
        return _fcache_derivative_chunked!!(
            cache, N_val, map(tuple, input_primals, input_tangents)...
        )
    end

    input_duals = tuple_map(Dual, input_primals, input_tangents)
    error_if_incorrect_dual_types(input_duals...)
    output = __call_rule(cache.frule_1, input_duals)
    return primal(output), tangent(output)
end

function value_and_derivative!!(cache::NfwdCache)
    _validate_prepared_cache_inputs(cache.input_specs, ())
    error("unreachable")
end

# ── HVP / Hessian — nfwd-over-reverse implementation ─────────────────────────────
#
# Strategy: compile a reverse-mode rule for NDual{T,1} inputs. A single forward+backward
# pass with seed (value=0, partials=(1,)) yields both the gradient (in the .partials
# field of the input fdata) and the HVP (in the .value field), from a single pass.
#
# NDual{T,1} acts as a dual number: value carries the primal, partials[1] carries the
# tangent direction. The reverse AD differentiates through NDual arithmetic, so the
# adjoint of the partials component of the output w.r.t. the value/partials components
# of the inputs naturally produces gradient and HVP information.

"""
    HVPCache

Cache for Hessian-vector products and Hessian evaluation using nfwd-over-reverse AD.
Stores a reverse-mode rule compiled for `NDual{T,1}` inputs, plus pre-allocated buffers.
"""
struct HVPCache{R,S<:Tuple,FB<:Tuple,XB<:Tuple,SD}
    ndual_rule::R           # DerivedRule compiled for NDual inputs
    input_specs::S          # PreparedCacheInputSpec for (f, x...)
    fdata_bufs::FB          # pre-allocated fdata buffers for NDual input tangents
    x_ndual_bufs::XB        # pre-allocated NDual input arrays
    seed::SD                # pre-allocated pullback seed rdata
end

function Base.show(io::IO, cache::HVPCache)
    print(
        io,
        "Mooncake.HVPCache(",
        "mode=:forward_over_reverse, ",
        "inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::HVPCache)
    n_inputs = _cache_input_count(cache)
    print(
        io, "Mooncake.HVPCache\n", "  mode: forward_over_reverse\n", "  inputs: ", n_inputs
    )
    input_specs = getfield(cache, :input_specs)
    _cache_print_io_summary(io, Base.tail(input_specs), "IEEEFloat scalar")
end

"""
    prepare_hvp_cache(f, x...; config=Mooncake.Config())

Prepare a cache for Hessian-vector products and Hessian evaluation.

Uses nfwd-over-reverse AD: compiles a reverse-mode rule for `NDual{T,1}` inputs so that
a single forward+backward pass produces both the gradient and the Hessian-vector product.

Only `Vector{<:IEEEFloat}` inputs are supported.
"""
function prepare_hvp_cache(f, x::Vararg{Any,N}; config=Config()) where {N}
    config.empty_cache && empty_mooncake_caches!()
    N == 0 && throw(ArgumentError("prepare_hvp_cache requires at least one x argument"))
    for (i, xi) in enumerate(x)
        xi isa Vector{<:IEEEFloat} || throw(
            ArgumentError(
                "HVP/Hessian requires `Vector{<:IEEEFloat}` inputs; " *
                "argument $i has type `$(typeof(xi))`.",
            ),
        )
    end

    # Build NDual versions of inputs and compile a reverse rule for NDual signature.
    x_ndual_bufs = map(x) do xi
        Nfwd.NDual{eltype(xi),1}.(xi, Ref(ntuple(_ -> zero(eltype(xi)), Val(1))))
    end
    ndual_rule = build_rrule(
        f,
        x_ndual_bufs...;
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )

    # Pre-allocate fdata buffers for NDual input tangents.
    fdata_bufs = map(x_ndual_bufs) do xb
        fdata(zero_tangent(xb))
    end

    # Build the pullback seed: rdata for NDual output with (value=0, partials=(1,)).
    T_elem = eltype(first(x))
    y_ndual_sample = Nfwd.NDual{T_elem,1}(zero(T_elem), ntuple(_ -> zero(T_elem), Val(1)))
    y_tangent_type = tangent_type(typeof(y_ndual_sample))
    seed = _hvp_make_seed(y_tangent_type, T_elem)

    specs = map((f, x...)) do xi
        if xi isa AbstractArray
            PreparedCacheInputSpec(typeof(xi), size(xi))
        else
            PreparedCacheInputSpec(typeof(xi), ())
        end
    end

    return HVPCache(ndual_rule, specs, fdata_bufs, x_ndual_bufs, seed)
end

# Construct the pullback seed rdata: (value=0, partials=(1,)) for NDual{T,1}.
function _hvp_make_seed(::Type{Tangent{NT}}, ::Type{T}) where {NT,T}
    return RData{NT}((value=zero(T), partials=(one(T),)))
end
function _hvp_make_seed(::Type{NoTangent}, ::Type{T}) where {T}
    return NoRData()
end

"""
    value_and_hvp!!(cache::HVPCache, f, directions, x...)

Compute the value, gradient, and Hessian-vector product `H * v` where `H = ∇² f(x)`
and `v = directions`.

Returns `(value, gradient, hvp)` for single-argument functions, or
`(value, (g1, g2, ...), (h1, h2, ...))` for multi-argument functions.
"""
function value_and_hvp!!(cache::HVPCache, f, v, x::Vararg{Any,N}) where {N}
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), (f, x...))
    if v isa AbstractVector
        N == 1 || throw(
            ArgumentError(
                "For multi-argument functions, `directions` must be a tuple of " *
                "vectors (one per argument), got a single vector for $N arguments.",
            ),
        )
        v = (v,)
    end
    length(v) == N || throw(
        ArgumentError(
            "Expected $N direction vector(s) (one per input argument), got $(length(v)).",
        ),
    )
    for (i, (vi, xi)) in enumerate(zip(v, x))
        length(vi) == length(xi) || throw(
            ArgumentError(
                "Direction vector $i has length $(length(vi)) but input $i " *
                "has length $(length(xi)).",
            ),
        )
    end

    # Lift x to NDual{T,1} with tangent direction v, reusing pre-allocated buffers.
    x_ndual_bufs = getfield(cache, :x_ndual_bufs)
    fdata_bufs = getfield(cache, :fdata_bufs)
    for k in 1:N
        xk, vk, buf = x[k], v[k], x_ndual_bufs[k]
        for i in eachindex(buf)
            buf[i] = Nfwd.NDual(xk[i], (vk[i],))
        end
    end

    # Zero the fdata buffers by copying fresh zero fdata into them.
    for k in 1:N
        fresh = fdata(zero_tangent(x_ndual_bufs[k]))
        copyto!(fdata_bufs[k], fresh)
    end

    # Build CoDuals and run the NDual reverse rule.
    f_codual = zero_fcodual(f)
    x_coduals = map(CoDual, x_ndual_bufs, fdata_bufs)
    out, pb = getfield(cache, :ndual_rule)(f_codual, x_coduals...)
    y_ndual = primal(out)

    # Run pullback with seed (value=0, partials=(1,)).
    pb(getfield(cache, :seed))

    # Extract gradient (from .partials) and HVP (from .value) of the fdata.
    val = y_ndual.value
    gradient_vectors = ntuple(Val(N)) do k
        fb = fdata_bufs[k]
        map(t -> t.fields.partials[1], fb)
    end
    hvp_vectors = ntuple(Val(N)) do k
        fb = fdata_bufs[k]
        map(t -> t.fields.value, fb)
    end

    return val,
    _unwrap_single(Val(N), gradient_vectors),
    _unwrap_single(Val(N), hvp_vectors)
end

@inline _unwrap_single(::Val{1}, t::Tuple) = t[1]
@inline _unwrap_single(::Val, t::Tuple) = t

"""
    prepare_hessian_cache(f, x...; config=Mooncake.Config())

Prepare a cache for Hessian evaluation.

Uses forward-over-reverse AD internally.

Only `Vector{<:IEEEFloat}` inputs are supported.
"""
function prepare_hessian_cache(f, x::Vararg{Any,N}; config=Config()) where {N}
    return prepare_hvp_cache(f, x...; config)
end

"""
    value_gradient_and_hessian!!(cache, f, x...)

Compute the value, gradient, and Hessian of `f` at `x`.

For single-argument `f(x::Vector)`, returns `(value, gradient, hessian_matrix)`.
For multi-argument `f(x1, x2, ...)`, returns
`(value, (g1, g2, ...), ((H11, H12, ...), (H21, H22, ...), ...))`.

Only `Vector{<:IEEEFloat}` inputs are supported. All inputs must have the same element type.
"""
function value_gradient_and_hessian!!(cache::HVPCache, f, x::Vararg{Any,N}) where {N}
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), (f, x...))

    # Validate inputs
    for (i, xi) in enumerate(x)
        xi isa Vector{<:IEEEFloat} || throw(
            ArgumentError(
                "Hessian requires `Vector{<:IEEEFloat}` inputs; " *
                "argument $i has type `$(typeof(xi))`.",
            ),
        )
    end
    T = eltype(first(x))
    for (i, xi) in enumerate(x)
        eltype(xi) === T || throw(
            ArgumentError(
                "All Hessian inputs must have the same element type; " *
                "argument 1 has `$T` but argument $i has `$(eltype(xi))`.",
            ),
        )
    end

    sizes = map(length, x)
    n = sum(sizes; init=0)

    if n == 0
        if N == 1
            return zero(T), zeros(T, 0), zeros(T, 0, 0)
        else
            grad_zeros = ntuple(k -> zeros(T, sizes[k]), Val(N))
            H_blocks = ntuple(Val(N)) do i
                ntuple(Val(N)) do j
                    zeros(T, sizes[i], sizes[j])
                end
            end
            return zero(T), grad_zeros, H_blocks
        end
    end

    # Build Hessian column by column using HVP with standard basis vectors.
    # The first call also gives us the value and gradient.
    if N == 1
        H = Matrix{T}(undef, n, n)
        local val, grad_vec
        for j in 1:n
            ej = zeros(T, n)
            ej[j] = one(T)
            v, g, hvp = value_and_hvp!!(cache, f, (ej,), x...)
            H[:, j] .= hvp
            if j == 1
                val = v
                grad_vec = copy(g)
            end
        end
        return val, grad_vec, H
    else
        H_full = Matrix{T}(undef, n, n)
        local val, grad_copies
        for j in 1:n
            ej_vecs = _hessian_basis_vectors(sizes, j, T)
            v, grads, hvp = value_and_hvp!!(cache, f, ej_vecs, x...)
            col_offset = 0
            for k in 1:N
                H_full[(col_offset + 1):(col_offset + sizes[k]), j] .= hvp[k]
                col_offset += sizes[k]
            end
            if j == 1
                val = v
                grad_copies = map(copy, grads)
            end
        end

        # Slice into blocks
        H_blocks = ntuple(N) do i
            row_start = sum(sizes[1:(i - 1)]; init=0) + 1
            row_end = row_start + sizes[i] - 1
            ntuple(N) do j2
                col_start = sum(sizes[1:(j2 - 1)]; init=0) + 1
                col_end = col_start + sizes[j2] - 1
                H_full[row_start:row_end, col_start:col_end]
            end
        end
        return val, grad_copies, H_blocks
    end
end

function _hessian_basis_vectors(sizes::Tuple, j::Int, ::Type{T}) where {T}
    N = length(sizes)
    vecs = Vector{Vector{T}}(undef, N)
    offset = 0
    for k in 1:N
        v = zeros(T, sizes[k])
        local_j = j - offset
        if 1 <= local_j <= sizes[k]
            v[local_j] = one(T)
        end
        vecs[k] = v
        offset += sizes[k]
    end
    return Tuple(vecs)
end
