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
    return _print_boxed_error(
        io, split("ValueAndGradientReturnTypeError: $(err.msg)", '\n')
    )
end

function Base.showerror(io::IO, err::ValueAndPullbackReturnTypeError)
    return _print_boxed_error(
        io, split("ValueAndPullbackReturnTypeError: $(err.msg)", '\n')
    )
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

Run a forward rule directly, without first constructing a `FCache`.

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
# - `FCache`: forward-mode cache returned by `prepare_derivative_cache`.
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
    return print(io, "\n  output: ", output_summary)
end

function Base.show(io::IO, cache::Cache)
    return print(
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
    return _cache_print_io_summary(
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
            InputSpec(typeof(x), size(x))
        else
            InputSpec(typeof(x), ())
        end
    end
    output_primal = primal(y)
    output_spec = if output_primal isa AbstractArray
        InputSpec(typeof(output_primal), size(output_primal))
    else
        InputSpec(typeof(output_primal), ())
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
    _validate_prepared_cache(getfield(cache, :input_specs), fx)
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
            InputSpec(typeof(x), size(x))
        else
            InputSpec(typeof(x), ())
        end
    end
    output_primal = primal(y)
    output_spec = if output_primal isa AbstractArray
        InputSpec(typeof(output_primal), size(output_primal))
    else
        InputSpec(typeof(output_primal), ())
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
    _validate_prepared_cache(getfield(cache, :input_specs), fx)
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
# encoded as a type parameter so that `_validate_prepared_cache` can read it at
# @generated specialisation time — eliminating the runtime `jl_types_equal` call that
# a `DataType`-valued field would require.
struct InputSpec{T,S}
    size::S
end

InputSpec(::Type{T}, s::S) where {T,S} = InputSpec{T,S}(s)

@inline function _cache_spec_size_summary(spec::InputSpec{T}) where {T}
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

@inline _cache_spec_summary(spec::InputSpec{T}) where {T} = "$(T) ($(_cache_spec_size_summary(spec)))"

struct PreparedCacheError <: Exception
    msg::String
end

function Base.showerror(io::IO, err::PreparedCacheError)
    return _print_boxed_error(io, split("PreparedCacheError:\n$(err.msg)", '\n'))
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
    throw(PreparedCacheError(msg))
end

# Shared prepared-cache input validation for Cache, FCache, and HVPCache entry points.
# The expected type T_i is extracted from the InputSpec{T_i,S_i} type parameter
# at @generated specialisation time, so the emitted `typeof(x_i) == T_i` comparison uses a
# compile-time constant type — eliminating the runtime jl_types_equal call.
@generated function _validate_prepared_cache(specs::Tuple, fx::Tuple)
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

# ── Forward-mode FCache ──────────────────────────────────────────────────

"""
    FCache

Forward-mode cache returned by [`prepare_derivative_cache`](@ref).

Holds a forward rule built by the primal-mode compiler, plus gradient and seed
buffers and input specs. `W` is `Nothing` for width-1 or `Val{N}` for width-N.

The `seed_buf` field lets `_gradient_width1`/`_gradient_widthN` reuse a single
seed-tangent tuple across slot iterations. The `lift_buf` field caches the
per-input `Container{NDual{T,W}}` lift target used by `_combine_to_ndual` in
the width-N path so that chunk evaluation doesn't allocate a fresh
`Array{NDual,...}` / `Memory{NDual,...}` lift container on every chunk.
"""
struct FCache{R,GW,SB,LB,S<:Tuple,W}
    friendly_tangents::Bool
    rule::R
    buf::GW
    seed_buf::SB
    lift_buf::LB
    input_specs::S
    width::W
end

@inline _dual_primal_type(::Type) = Any
@inline _dual_primal_type(::Type{Dual{Y,T}}) where {Y,T} = Y
@inline _dual_primal_type(::Type{NDual{T,W}}) where {T,W} = T

@inline function _output_summary(cache::FCache)
    rule = getfield(cache, :rule)
    width = getfield(cache, :width)
    dual_arg_types = Tuple{map(getfield(cache, :input_specs)) do spec
        P = typeof(spec).parameters[1]
        isnothing(width) ? dual_type(P) : dual_type(width, P)
    end...}
    output_type = Core.Compiler.return_type(rule, dual_arg_types)
    return _cache_type_summary(_dual_primal_type(output_type))
end

function Base.show(io::IO, cache::FCache)
    return print(
        io, "Mooncake.FCache(", "mode=:forward, ", "inputs=", _cache_input_count(cache), ")"
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::FCache)
    print(
        io,
        "Mooncake.FCache\n",
        "  mode: forward\n",
        "  inputs: ",
        _cache_input_count(cache),
    )
    return _cache_print_io_summary(
        io, Base.tail(getfield(cache, :input_specs)), _output_summary(cache)
    )
end

@inline function _maybe_friendly_tangent(
    cache::FCache, input_primals::Tuple, native_grads::Tuple
)
    cache.friendly_tangents || return native_grads
    friendly_gradients = _copy_output(input_primals)
    c = isbitstype(typeof(friendly_gradients)) ? NoCache() : IdDict{Any,Any}()
    return tangent_to_primal_internal!!(friendly_gradients, native_grads, c)
end

# ── fcache gradient bookkeeping ───────────────────────────────────────────────────

@noinline function _throw_uninit_field_error(::Type{P}, n::Int) where {P}
    throw(
        ArgumentError(
            "Forward-mode gradient seeding encountered an undefined field " *
            "`$(fieldname(P, n))` in a value of type `$P`, but that field is marked " *
            "always-initialised. This object is in a partially initialised state that " *
            "Mooncake cannot seed automatically.",
        ),
    )
end

# ── fcache gradient seeding ───────────────────────────────────────────────────────

@inline _make_seed_tangent(x, slot::Int) = _make_seed_tangent(
    x, slot, Ref(0), IdDict{Any,Any}()
)
@inline function _make_seed_tangent(
    ::NoTangent, _slot::Int, _cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    return NoTangent()
end
@inline function _make_seed_tangent(
    x::IEEEFloat, slot::Int, cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
)
    cursor[] += 1
    return cursor[] == slot ? one(x) : zero(x)
end
@inline function _make_seed_tangent(
    x::Complex{T}, slot::Int, cursor::Base.RefValue{Int}, _dict::IdDict{Any,Any}
) where {T<:IEEEFloat}
    cursor[] += 1
    real_part = cursor[] == slot ? one(T) : zero(T)
    cursor[] += 1
    imag_part = cursor[] == slot ? one(T) : zero(T)
    return complex(real_part, imag_part)
end

function _make_seed_tangent(
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

function _make_seed_tangent(
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

function _make_seed_tangent(
    x::AbstractArray, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
)
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    existing = get(dict, x, nothing)
    !isnothing(existing) && return existing
    dx = zero_tangent(x)
    dict[x] = dx
    for I in eachindex(x)
        dx[I] = _make_seed_tangent(x[I], slot, cursor, dict)
    end
    return dx
end

@inline function _make_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:Tuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(n -> _make_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P)))
    return build_tangent(P, fields...)
end

@inline function _make_seed_tangent(
    x::P, slot::Int, cursor::Base.RefValue{Int}, dict::IdDict{Any,Any}
) where {P<:NamedTuple}
    tangent_type(P) == NoTangent && return NoTangent()
    fields = ntuple(n -> _make_seed_tangent(x[n], slot, cursor, dict), Val(fieldcount(P)))
    return build_tangent(P, fields...)
end

function _make_seed_tangent(
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
                        tx, n, _make_seed_tangent(getfield(x, n), slot, cursor, dict)
                    )
                elseif inits[n]
                    _throw_uninit_field_error(P, n)
                end
            end
        end
        return tx
    end

    inits = always_initialised(P)
    fields = ntuple(Val(fieldcount(P))) do n
        if isdefined(x, n)
            return _make_seed_tangent(getfield(x, n), slot, cursor, dict)
        elseif inits[n]
            _throw_uninit_field_error(P, n)
        else
            return PossiblyUninitTangent{tangent_type(fieldtype(P, n))}()
        end
    end
    return build_tangent(P, fields...)
end

# ── Aliasing-aware allocation and gradient accumulation ──────────────────────
#
# When inputs can alias (a mutable buffer reaches multiple positions, e.g.
# `value_and_gradient!!(cache, f, x, x)`), `seen` is an `IdDict` so each unique
# buffer is seeded and accumulated once. Otherwise it stays `nothing`, keeping
# the all-bitstype path zero-alloc.

@generated function _inputs_can_alias(::Type{T}) where {T<:Tuple}
    return any(!isbitstype, T.parameters)
end

@inline _new_seen(input_primals::Tuple) =
    _inputs_can_alias(typeof(input_primals)) ? IdDict{Any,Nothing}() : nothing

@inline function _alloc_aliased_tangents(input_primals::Tuple)
    if _inputs_can_alias(typeof(input_primals))
        return zero_tangent_internal(input_primals, IdDict())
    end
    return tuple_map(zero_tangent, input_primals)
end

@inline _count_input_slots(input_primals::Tuple) = _count_slots(
    input_primals, _new_seen(input_primals)
)

@inline function _accumulate_gradient(native_gradients::Tuple, seed::Tuple, coeff, seen)
    seen isa IdDict{Any,Nothing} && empty!(seen)
    return tuple_map(
        (g, dx) -> _accumulate_one_gradient!!(g, dx, coeff, seen), native_gradients, seed
    )
end

@inline function _accumulate_one_gradient!!(g, dx, coeff, seen)
    dx isa NoTangent && return g
    if seen isa IdDict{Any,Nothing} && ismutabletype(typeof(g))
        haskey(seen, g) && return g
        seen[g] = nothing
    end
    return increment!!(g, _scale(coeff, dx))
end

# ── In-place seed-tangent updaters ────────────────────────────────────────────────
#
# `_seed_inplace!(buf, x, slot, cursor, seen)` mutates `buf` (a tangent-shaped
# value parallel to the primal `x`) so it represents the standard-basis seed
# for a given global slot index, returning the updated buf. For mutable
# containers (arrays) the buffer is updated in place and returned as-is. For
# immutable scalars the buffer is rebuilt and returned. Tuple/NamedTuple values
# rebuild the outer container around per-element results so the caller can
# drop the whole tuple back into the seed-tangent slot. The `seen` IdDict
# tracks mutable buffers already visited so aliased inputs are seeded once and
# `cursor` advances over each unique scalar dof exactly once.
#
# Used by `_gradient_width1` / `_gradient_widthN` to avoid re-allocating
# tangent arrays for each direction.

@inline function _seed_inplace!(
    ::NoTangent,
    _x,
    _slot::Int,
    _cursor::Base.RefValue{Int},
    _seen::Union{Nothing,IdDict{Any,Nothing}},
)
    return NoTangent()
end
@inline function _seed_inplace!(
    _buf::T,
    x::T,
    slot::Int,
    cursor::Base.RefValue{Int},
    _seen::Union{Nothing,IdDict{Any,Nothing}},
) where {T<:IEEEFloat}
    cursor[] += 1
    return cursor[] == slot ? one(x) : zero(x)
end
@inline function _seed_inplace!(
    _buf::Complex{T},
    x::Complex{T},
    slot::Int,
    cursor::Base.RefValue{Int},
    _seen::Union{Nothing,IdDict{Any,Nothing}},
) where {T<:IEEEFloat}
    cursor[] += 1
    re = cursor[] == slot ? one(T) : zero(T)
    cursor[] += 1
    im = cursor[] == slot ? one(T) : zero(T)
    return complex(re, im)
end
@inline function _seed_inplace!(
    buf::AbstractArray{T},
    x::AbstractArray{<:IEEEFloat},
    slot::Int,
    cursor::Base.RefValue{Int},
    seen::Union{Nothing,IdDict{Any,Nothing}},
) where {T<:IEEEFloat}
    if seen isa IdDict{Any,Nothing}
        haskey(seen, buf) && return buf
        seen[buf] = nothing
    end
    @inbounds for I in eachindex(x)
        cursor[] += 1
        buf[I] = cursor[] == slot ? one(T) : zero(T)
    end
    return buf
end
@inline function _seed_inplace!(
    buf::AbstractArray{Complex{T}},
    x::AbstractArray{Complex{<:IEEEFloat}},
    slot::Int,
    cursor::Base.RefValue{Int},
    seen::Union{Nothing,IdDict{Any,Nothing}},
) where {T<:IEEEFloat}
    if seen isa IdDict{Any,Nothing}
        haskey(seen, buf) && return buf
        seen[buf] = nothing
    end
    @inbounds for I in eachindex(x)
        cursor[] += 1
        re = cursor[] == slot ? one(T) : zero(T)
        cursor[] += 1
        im = cursor[] == slot ? one(T) : zero(T)
        buf[I] = complex(re, im)
    end
    return buf
end
@inline function _seed_inplace!(
    buf::Tuple,
    x::Tuple,
    slot::Int,
    cursor::Base.RefValue{Int},
    seen::Union{Nothing,IdDict{Any,Nothing}},
)
    return ntuple(
        i -> _seed_inplace!(buf[i], x[i], slot, cursor, seen), Val(fieldcount(typeof(x)))
    )
end
@inline function _seed_inplace!(
    buf::NamedTuple{names},
    x::NamedTuple{names},
    slot::Int,
    cursor::Base.RefValue{Int},
    seen::Union{Nothing,IdDict{Any,Nothing}},
) where {names}
    return NamedTuple{names}(_seed_inplace!(values(buf), values(x), slot, cursor, seen))
end
# Mutable struct buffer: walk fields and `set_tangent_field!` in place. The
# `seen` IdDict skips revisits when the same buffer is reached via aliased
# inputs (e.g. `value_and_gradient!!(cache, f, x, x)`).
@inline function _seed_inplace!(
    buf::MutableTangent,
    x::P,
    slot::Int,
    cursor::Base.RefValue{Int},
    seen::Union{Nothing,IdDict{Any,Nothing}},
) where {P}
    if seen isa IdDict{Any,Nothing}
        haskey(seen, buf) && return buf
        seen[buf] = nothing
    end
    inits = always_initialised(P)
    @inbounds for n in 1:fieldcount(P)
        if isdefined(x, n)
            field_buf = val(getfield(buf.fields, n))
            new_field = _seed_inplace!(field_buf, getfield(x, n), slot, cursor, seen)
            set_tangent_field!(buf, n, new_field)
        elseif inits[n]
            _throw_uninit_field_error(P, n)
        end
    end
    return buf
end

# Immutable struct buffer: rebuild the outer `Tangent` wrapper, reusing inner
# mutable buffers (arrays, MutableTangents) by reference.
@inline function _seed_inplace!(
    buf::Tangent,
    x::P,
    slot::Int,
    cursor::Base.RefValue{Int},
    seen::Union{Nothing,IdDict{Any,Nothing}},
) where {P}
    inits = always_initialised(P)
    fields = ntuple(Val(fieldcount(P))) do n
        if isdefined(x, n)
            field_buf = val(getfield(buf.fields, n))
            return _seed_inplace!(field_buf, getfield(x, n), slot, cursor, seen)
        elseif inits[n]
            return _throw_uninit_field_error(P, n)
        else
            return PossiblyUninitTangent{tangent_type(fieldtype(P, n))}()
        end
    end
    return build_tangent(P, fields...)
end

# Final fallback for shapes we can't mutate in place — falls back to allocating
# `_make_seed_tangent`. Slot semantics are preserved because `cursor` keeps
# walking through the global scalar dofs across calls.
@inline function _seed_inplace!(
    _buf,
    x,
    slot::Int,
    cursor::Base.RefValue{Int},
    _seen::Union{Nothing,IdDict{Any,Nothing}},
)
    return _make_seed_tangent(x, slot, cursor, IdDict{Any,Any}())
end

# ── _tangent_width ────────────────────────────────────────────────────────────────

@generated function _tangent_width(ts::T) where {T<:Tuple}
    width = nothing
    for entry in T.parameters
        entry <: NTangent || continue
        current = fieldcount(entry.parameters[1])
        if isnothing(width)
            width = current
        elseif width != current
            return quote
                throw(
                    ArgumentError(
                        "All NTangent inputs must have the same number of basis directions; " *
                        "found both $(width) and $(current).",
                    ),
                )
            end
        end
    end

    return isnothing(width) ? :(nothing) : :(Val{$width}())
end

# ── prepare_derivative_cache ──────────────────────────────────────────────────────

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns a [`FCache`](@ref) used with [`value_and_derivative!!`](@ref). See
[`Mooncake.Config`](@ref) for the `chunk_size` semantics; note that
`chunk_size=nothing` selects the legacy `Dual{P,T}` path, while
`chunk_size=1` selects the chunked NDual path at width 1 — these are
distinct internal paths even though both compute single-direction derivatives.

# Reuse contract

The cache is bound to the prep-time argument types, sizes, and aliasing
topology. Type and size mismatches surface as errors on the next call;
**aliasing mismatches do not** — aliasing relationships between array inputs
(`primal(a) === primal(b)`) are baked into the cache at prep time, and reusing
with a different aliasing relationship produces wrong gradients. Re-prepare
when any of these change.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    cs = config.chunk_size
    if cs !== nothing && cs < 1
        throw(ArgumentError("chunk_size must be at least 1, got $cs"))
    end
    config.empty_cache && empty_mooncake_caches!()
    fx = (f, x...)
    sig = typeof(fx)
    interp = get_interpreter(ForwardMode)
    width = cs !== nothing ? Val(cs) : nothing
    rule = build_frule(
        interp,
        sig,
        width;
        debug_mode=config.debug_mode,
        silence_debug_messages=config.silence_debug_messages,
    )
    input_specs = map(fx) do x
        if x isa AbstractArray
            InputSpec(typeof(x), size(x))
        else
            InputSpec(typeof(x), ())
        end
    end
    GW = Tuple{map(P -> tangent_type(P), sig.parameters)...}
    buf = Ref{Union{Nothing,GW}}(nothing)
    # Concrete-typed Ref keeps seed-buf access type-stable; required to keep
    # scalar `count_allocs(value_and_gradient!!, ...) == 0`.
    seed_buf = if width === nothing
        Ref{Union{Nothing,GW}}(nothing)
    else
        W = typeof(width).parameters[1]
        Ref{Union{Nothing,NTuple{W,GW}}}(nothing)
    end
    # `lift_buf` holds per-input `Container{NDual{T,W}}` lift targets reused by
    # `_gradient_widthN`. Tuple shape mirrors `input_primals`. Width-1 path
    # doesn't use it, so leave as Nothing in that branch.
    lift_buf = if width === nothing
        Ref{Nothing}(nothing)
    else
        W = typeof(width).parameters[1]
        LT = Tuple{map(P -> _ndual_lift_type(Val(W), P), sig.parameters)...}
        Ref{Union{Nothing,LT}}(nothing)
    end
    return FCache(
        config.friendly_tangents, rule, buf, seed_buf, lift_buf, input_specs, width
    )
end

# ── value_and_gradient!! (FCache) ───────────────────────────────────────────

function value_and_gradient!!(cache::FCache, f::F, x::Vararg{Any,N}) where {F,N}
    input_primals = (f, x...)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)

    native_gradients = let buf = cache.buf[]
        if isnothing(buf)
            buf = _alloc_aliased_tangents(input_primals)
            cache.buf[] = buf
            buf
        else
            zeroed = tuple_map(set_to_zero!!, buf)
            cache.buf[] = zeroed
            zeroed
        end
    end
    total_slots = _count_input_slots(input_primals)
    rule = cache.rule
    width = cache.width

    if total_slots == 0
        output = rule(tuple_map(Dual, input_primals, native_gradients)...)
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        return y, _maybe_friendly_tangent(cache, input_primals, native_gradients)
    end

    if width === nothing
        return _gradient_width1(rule, input_primals, native_gradients, total_slots, cache)
    else
        return _gradient_widthN(
            rule, input_primals, native_gradients, total_slots, cache, width
        )
    end
end

function _gradient_width1(rule, input_primals, native_gradients, total_slots, cache)
    local y
    seed_buf = let buf = cache.seed_buf[]
        if buf === nothing
            buf = _alloc_aliased_tangents(input_primals)
            cache.seed_buf[] = buf
        end
        buf
    end
    cursor = Ref(0)
    seed_seen = _new_seen(input_primals)
    accum_seen = _new_seen(native_gradients)
    for slot in 1:total_slots
        cursor[] = 0
        seed_seen isa IdDict{Any,Nothing} && empty!(seed_seen)
        seed = _seed_inplace!(seed_buf, input_primals, slot, cursor, seed_seen)
        output = rule(tuple_map(Dual, input_primals, seed)...)
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        coeff = Float64(tangent(output))
        native_gradients = _accumulate_gradient(native_gradients, seed, coeff, accum_seen)
    end
    return y, _maybe_friendly_tangent(cache, input_primals, native_gradients)
end

function _gradient_widthN(
    rule, input_primals, native_gradients, total_slots, cache, ::Val{W}
) where {W}
    local y
    slot = 1
    # `W` per-direction tangent buffers, lazily allocated and reused across chunks.
    seed_bufs = let cached = cache.seed_buf[]
        if cached === nothing
            buf = ntuple(_ -> _alloc_aliased_tangents(input_primals), Val(W))
            cache.seed_buf[] = buf
            buf
        else
            cached
        end
    end
    # Per-input NDual lift buffer; `Nothing` for scalar/struct inputs that
    # `_combine_to_ndual` doesn't allocate for. Allocated lazily once per cache
    # then mutated in place each chunk.
    lift_bufs = let cached = cache.lift_buf[]
        if cached === nothing
            allocated = tuple_map(input_primals) do x
                _alloc_lift_buf(Val(W), x)
            end
            cache.lift_buf[] = allocated
            allocated
        else
            cached
        end
    end
    cursor = Ref(0)
    seed_seen = _new_seen(input_primals)
    accum_seen = _new_seen(native_gradients)
    while slot <= total_slots
        chunk = min(W, total_slots - slot + 1)
        seeds = ntuple(Val(W)) do d
            cursor[] = 0
            seed_seen isa IdDict{Any,Nothing} && empty!(seed_seen)
            _seed_inplace!(seed_bufs[d], input_primals, slot + d - 1, cursor, seed_seen)
        end
        ndual_inputs = ntuple(Val(length(input_primals))) do i
            _combine_to_ndual_or_buffer(
                lift_bufs[i], input_primals[i], ntuple(d -> seeds[d][i], Val(W))
            )
        end
        output = rule(ndual_inputs...)
        y = output isa NDual ? output.value : primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        # Width-N IEEEFloat output must come back as NDual: dual_type(Val(N), ::IEEEFloat)
        # is NDual{T,N}, so a non-NDual output here means a custom rule returned a
        # width-1 Dual in a width-N pipeline. coeffs would then be a 1-tuple and the
        # `coeffs[d]` loop below would BoundsError for chunk>1; raise explicitly instead.
        coeffs = if output isa NDual
            output.partials
        elseif chunk == 1
            (Float64(tangent(output)),)
        else
            throw(
                ArgumentError(
                    "value_and_gradient!! width-$W path expected an NDual output " *
                    "for IEEEFloat primal but got `$(typeof(output))` — typically a " *
                    "custom frule!! returned a width-1 Dual or a scalar primitive " *
                    "lacks a width-N NDual overload. " *
                    "Use `Mooncake.Config(chunk_size=nothing)` as a workaround.",
                ),
            )
        end
        for d in 1:chunk
            coeff = Float64(coeffs[d])
            native_gradients = _accumulate_gradient(
                native_gradients, seeds[d], coeff, accum_seen
            )
        end
        slot += chunk
    end
    return y, _maybe_friendly_tangent(cache, input_primals, native_gradients)
end

# Per-input lift container type. Inputs without array storage (scalars, Complex,
# structs) don't need a buffer — `_combine_to_ndual` for those is allocation-free.
# For Array/Memory eltypes that the nfwd path supports, delegate to `dual_type`,
# which already returns the correct `Array{NDual{T,W},D}` / `Memory{NDual{T,W}}` shape.
const _LiftEltype = Union{IEEEFloat,Complex{<:IEEEFloat}}

@inline _ndual_lift_type(::Val, ::Type) = Nothing
@inline _ndual_lift_type(w::Val, ::Type{P}) where {P<:Array{<:_LiftEltype}} = dual_type(
    w, P
)

@inline _alloc_lift_buf(::Val, x) = nothing
@inline function _alloc_lift_buf(w::Val, x::Array{<:_LiftEltype})
    return dual_type(w, typeof(x))(undef, size(x))
end

@static if VERSION >= v"1.11-"
    @inline _ndual_lift_type(w::Val, ::Type{P}) where {P<:Memory{<:_LiftEltype}} = dual_type(
        w, P
    )
    @inline function _alloc_lift_buf(w::Val, x::Memory{<:_LiftEltype})
        return dual_type(w, typeof(x))(undef, length(x))
    end
end

# Pick between in-place (when a pre-allocated buffer is available) and
# allocating `_combine_to_ndual` based on the buffer slot.
@inline _combine_to_ndual_or_buffer(::Nothing, x, partials) = _combine_to_ndual(x, partials)
@inline function _combine_to_ndual_or_buffer(
    buf::AbstractArray{<:NDual}, x::AbstractArray{<:IEEEFloat}, partials::NTuple{W}
) where {W}
    return _combine_to_ndual!(buf, x, partials)
end
@inline function _combine_to_ndual_or_buffer(
    buf::AbstractArray{<:Complex{<:NDual}},
    x::AbstractArray{<:Complex{<:IEEEFloat}},
    partials::NTuple{W},
) where {W}
    return _combine_to_ndual!(buf, x, partials)
end

# In-place lift used by the FCache lift-buf path. `tangent_dirs` is typed
# tighter than `NTuple{W}` so the inner `ntuple` closure specialises (an
# untyped tuple boxes per index).
@inline function _combine_to_ndual!(
    result::AbstractArray{NDual{T,W}},
    x::AbstractArray{<:IEEEFloat},
    tangent_dirs::NTuple{W,<:AbstractArray{<:IEEEFloat}},
) where {T<:IEEEFloat,W}
    @inbounds for I in eachindex(x)
        result[I] = NDual{T,W}(x[I], ntuple(d -> tangent_dirs[d][I], Val(W)))
    end
    return result
end
@inline function _combine_to_ndual!(
    result::AbstractArray{Complex{NDual{T,W}}},
    x::AbstractArray{<:Complex{<:IEEEFloat}},
    tangent_dirs::NTuple{W,<:AbstractArray{<:Complex{<:IEEEFloat}}},
) where {T<:IEEEFloat,W}
    @inbounds for I in eachindex(x)
        re = NDual{T,W}(real(x[I]), ntuple(d -> real(tangent_dirs[d][I]), Val(W)))
        im = NDual{T,W}(imag(x[I]), ntuple(d -> imag(tangent_dirs[d][I]), Val(W)))
        result[I] = Complex(re, im)
    end
    return result
end

@inline function _combine_to_ndual(x::T, partials::NTuple{W,T}) where {T<:IEEEFloat,W}
    return NDual{T,W}(x, partials)
end

@inline function _combine_to_ndual(
    x::Complex{T}, partials::NTuple{W,Complex{T}}
) where {T<:IEEEFloat,W}
    re = NDual{T,W}(real(x), ntuple(d -> real(partials[d]), Val(W)))
    im = NDual{T,W}(imag(x), ntuple(d -> imag(partials[d]), Val(W)))
    return Complex(re, im)
end

function _combine_to_ndual(
    x::AbstractArray{T}, tangent_dirs::NTuple{W}
) where {T<:IEEEFloat,W}
    result = similar(x, NDual{T,W})
    @inbounds for I in eachindex(x)
        result[I] = NDual{T,W}(x[I], ntuple(d -> tangent_dirs[d][I], Val(W)))
    end
    return result
end

function _combine_to_ndual(
    x::AbstractArray{Complex{T}}, tangent_dirs::NTuple{W}
) where {T<:IEEEFloat,W}
    result = similar(x, Complex{NDual{T,W}})
    @inbounds for I in eachindex(x)
        re = NDual{T,W}(real(x[I]), ntuple(d -> real(tangent_dirs[d][I]), Val(W)))
        im = NDual{T,W}(imag(x[I]), ntuple(d -> imag(tangent_dirs[d][I]), Val(W)))
        result[I] = Complex(re, im)
    end
    return result
end

@inline _combine_to_ndual(x, ::NTuple{W,NoTangent}) where {W} = Dual(x, NoTangent())
@inline _combine_to_ndual(x::Tuple, ::Tuple{}) = Dual(x, NoTangent())

# Non-IEEEFloat aggregate primals (structs, NamedTuples, mutable types) lift
# through `Dual{P, NTangent{...}}` per AGENTS.md, with each NTangent lane
# carrying a per-direction `tangent_type(P)` payload (`MutableTangent`,
# `Tangent`, etc.). The IEEEFloat-specialised methods above take precedence
# for scalar/array/Complex primals.
#
# When the partials are structurally NoTangent (per-element-NoTangent containers
# mirroring a non-differentiable primal shape — `Vector{NoTangent}`,
# `Memory{NoTangent}`, `MemoryRef{NoTangent}`), all `W` lanes are byte-identical
# and the legacy memory rules expect `Dual{<:_MemTypes, <:_MemTypes}` directly,
# so we collapse to a single representative. This branches at compile time on
# `_is_structurally_no_tangent(T)` (a `where`-bound type parameter), so the
# `if` reduces to a single specialisation per concrete partial type.
@inline function _combine_to_ndual(x, partials::NTuple{W,T}) where {W,T}
    if _is_structurally_no_tangent(T)
        return Dual(x, partials[1])
    end
    return Dual(x, NTangent(partials))
end
@inline function _combine_to_ndual(
    x::AbstractArray{<:IEEEFloat}, ::NTuple{W,NoTangent}
) where {W}
    return Dual(x, NoTangent())
end
@inline function _combine_to_ndual(
    x::Complex{T}, ::NTuple{W,NoTangent}
) where {T<:IEEEFloat,W}
    return Dual(x, NoTangent())
end
@inline _combine_to_ndual(x::Complex{T}, ::Tuple{}) where {T<:IEEEFloat} = Dual(
    x, NoTangent()
)
@inline function _combine_to_ndual(
    x::AbstractArray{Complex{T}}, ::NTuple{W,NoTangent}
) where {T<:IEEEFloat,W}
    return Dual(x, NoTangent())
end
@inline function _combine_to_ndual(x::Tuple, tangent_dirs::NTuple{W,<:Tuple}) where {W}
    return ntuple(Val(length(x))) do i
        element_partials = ntuple(d -> tangent_dirs[d][i], Val(W))
        _combine_to_ndual(x[i], element_partials)
    end
end

# ── value_and_derivative!! (FCache) ─────────────────────────────────────────

"""
    value_and_derivative!!(cache::FCache, f::Dual, x::Vararg{Dual,N})

Forward-mode derivative via Dual inputs, returning a Dual output.
"""
function value_and_derivative!!(cache::FCache, fx::Vararg{Dual,N}) where {N}
    input_primals = map(primal, fx)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
    width = cache.width
    if width === nothing
        error_if_incorrect_dual_types(fx...)
        return __call_rule(cache.rule, fx)
    else
        padded = map(fx) do d
            t = tangent(d)
            p = primal(d)
            if t isa NoTangent
                ntuple(_ -> NoTangent(), width)
            else
                ntuple(i -> i == 1 ? t : zero_tangent(p), width)
            end
        end
        ndual_inputs = map(_combine_to_ndual, input_primals, padded)
        output = cache.rule(ndual_inputs...)
        return _ndual_output_to_width1(output)
    end
end

@inline function _remaining_dirs(::Val{N}, _eval_dir) where {N}
    return ntuple(Val(N - 1)) do n
        t = tangent(_eval_dir(Val(n + 1)))
        return t isa NoTangent ? t : _copy(t)
    end
end

@inline function _ndual_output_to_width1(output)
    _has_ndual(output) || return output
    return Dual(primal(output), _tangent_dir(output, 1))
end

"""
    value_and_derivative!!(cache::FCache, (f, df), (x, dx), ...)

Forward-mode derivative via tuple inputs, returning `(y, dy)`.
Plain tuple tangents represent a single direction. Multi-direction evaluation requires
`NTangent` for every differentiable tuple tangent; mixed chunked/plain differentiable
inputs are rejected.
"""
@inline function value_and_derivative!!(
    cache::FCache, fx::Vararg{Tuple{Any,Any},M}
) where {M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)

    raw_tangents = tuple_map(last, fx)
    # If any input is chunked, the rest must be NoTangent or a structural container
    # that recurses into NTangent leaves.
    ChunkCompatible = Union{NoTangent,NTangent,Tuple,NamedTuple,Tangent,MutableTangent}
    if any(t -> t isa NTangent, raw_tangents) &&
        !all(t -> t isa ChunkCompatible, raw_tangents)
        throw(
            ArgumentError(
                "Chunked tuple inputs must use NTangent consistently for all differentiable tangents.",
            ),
        )
    end
    width = _tangent_width(raw_tangents)
    if isnothing(width)
        N_val, input_tangents, single_dir = Val(1),
        tuple_map(t -> NTangent((t,)), raw_tangents),
        true
    else
        N_val, input_tangents, single_dir = width, raw_tangents, false
    end

    friendly = cache.friendly_tangents
    rule = cache.rule
    cache_width = cache.width

    function _eval_dir(::Val{dir}) where {dir}
        dir_tangents = tuple_map(t -> t isa NTangent ? t[dir] : t, input_tangents)
        canonical = if friendly
            tuple_map(input_primals, dir_tangents) do p, t
                T = tangent_type(typeof(p))
                t isa T && return t
                return primal_to_tangent!!(zero_tangent(p), t)
            end
        else
            dir_tangents
        end
        if cache_width === nothing
            dir_duals = tuple_map(Dual, input_primals, canonical)
            friendly || error_if_incorrect_dual_types(dir_duals...)
            return rule(dir_duals...)
        end
        friendly ||
            error_if_incorrect_dual_types(tuple_map(Dual, input_primals, canonical)...)
        rule_inputs = tuple_map(input_primals, canonical) do p, t
            padded = if t isa NoTangent
                ntuple(_ -> NoTangent(), cache_width)
            else
                ntuple(d -> d == 1 ? t : zero_tangent(p), cache_width)
            end
            _combine_to_ndual(p, padded)
        end
        # Rule is compiled at `cache_width`; convert its width-N output back to the
        # caller's per-direction width-1 view.
        return _ndual_output_to_width1(rule(rule_inputs...))
    end

    first_output = _eval_dir(Val(1))
    y = primal(first_output)

    if single_dir
        dy = tangent(first_output)
        if friendly
            dy = tangent_to_primal_internal!!(
                _copy_output(y), dy, isbitstype(typeof(y)) ? NoCache() : IdDict{Any,Any}()
            )
        end
        return y, dy
    end

    first_tangent = let t = tangent(first_output)
        t isa NoTangent ? t : _copy(t)
    end
    rest_tangents = _remaining_dirs(N_val, _eval_dir)
    nt = NTangent((first_tangent, rest_tangents...))
    if friendly
        dirs = ntuple(Val(length(nt))) do i
            tangent_to_primal_internal!!(
                _copy_output(y),
                nt[i],
                isbitstype(typeof(y)) ? NoCache() : IdDict{Any,Any}(),
            )
        end
        nt = NTangent(dirs)
    end
    return y, nt
end

function value_and_derivative!!(::FCache)
    throw(
        ArgumentError(
            "`value_and_derivative!!` with a prepared forward cache requires at least the cached function argument.",
        ),
    )
end

# ── HVP / Hessian ────────────────────────────────────────────────────────────────
#
# Two nesting strategies, controlled by `config.second_order_mode`:
#
# :reverse_over_forward — reverse-mode rule compiled for NDual{T,1} inputs.
#   A single forward+backward pass yields gradient (.partials fdata) and HVP (.value fdata).
#
# :forward_over_reverse — forward-mode derivative of a reverse-mode gradient closure.

"""
    HVPCache{M}

Cache for Hessian-vector products and Hessian evaluation. `M` is the nesting mode:
`:forward_over_reverse` or `:reverse_over_forward`, controlled by
`config.second_order_mode` in [`prepare_hvp_cache`](@ref).
"""
struct HVPCache{M,C,S<:Tuple}
    core::C
    input_specs::S
end

function Base.show(io::IO, cache::HVPCache{M}) where {M}
    return print(
        io,
        "Mooncake.HVPCache(",
        "mode=:",
        M,
        ", ",
        "inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::HVPCache{M}) where {M}
    n_inputs = _cache_input_count(cache)
    print(io, "Mooncake.HVPCache\n", "  mode: ", M, "\n", "  inputs: ", n_inputs)
    return _cache_print_io_summary(
        io, Base.tail(getfield(cache, :input_specs)), _hvp_output_summary(cache)
    )
end

function _hvp_output_summary(cache::HVPCache)
    spec = getfield(cache, :input_specs)[2]  # first x-input (index 2, after f)
    T = eltype(typeof(spec).parameters[1])    # Vector{Float64} → Float64
    return "$T (scalar)"
end

@inline _unwrap_single(::Val{1}, t::Tuple) = t[1]
@inline _unwrap_single(::Val, t::Tuple) = t

# ── prepare_hvp_cache ────────────────────────────────────────────────────────────

"""
    prepare_hvp_cache(f, x...; config=Mooncake.Config())

Prepare a cache for Hessian-vector products and Hessian evaluation.

The nesting strategy is controlled by `config.second_order_mode`:
- `:forward_over_reverse` (default): forward-mode derivative of a gradient closure.
- `:reverse_over_forward`: reverse-mode rule over `NDual{T,1}` inputs.

Only `Vector{<:IEEEFloat}` inputs are supported.

!!! note "Mutation semantics"
    Preparation never calls `f(x...)` directly. Both modes discover the output type
    by running a compiled AD rule and immediately unwinding it, so mutations are
    restored and side effects are not leaked. Callers may rely on `x` being unmodified
    after `prepare_hvp_cache` returns.
"""
function prepare_hvp_cache(f, x::Vararg{Any,N}; config=Config()) where {N}
    config.empty_cache && empty_mooncake_caches!()
    N == 0 && throw(ArgumentError("prepare_hvp_cache requires at least one x argument"))
    mode = config.second_order_mode
    return _prepare_hvp(Val(mode), f, x, config)
end

# ── :reverse_over_forward internals ──────────────────────────────────────────────

function _prepare_hvp(::Val{:reverse_over_forward}, f, x::Tuple, config)
    for (i, xi) in enumerate(x)
        xi isa Vector{<:IEEEFloat} || throw(
            ArgumentError(
                "HVP/Hessian requires `Vector{<:IEEEFloat}` inputs; " *
                "argument $i has type `$(typeof(xi))`.",
            ),
        )
    end
    x_ndual_bufs = map(x) do xi
        Nfwd.NDual{eltype(xi),1}.(xi, Ref(ntuple(_ -> zero(eltype(xi)), Val(1))))
    end
    ndual_rule = try
        build_rrule(
            f,
            x_ndual_bufs...;
            debug_mode=config.debug_mode,
            silence_debug_messages=config.silence_debug_messages,
        )
    catch e
        if e isa MooncakeRuleCompilationError
            throw(
                ArgumentError(
                    "`:reverse_over_forward` mode requires `f` to accept NDual-element " *
                    "vectors, but no method matches the widened signature. Use " *
                    "`:forward_over_reverse` (the default) or widen your function's " *
                    "argument types (e.g. `f(x::AbstractVector)` instead of " *
                    "`f(x::Vector{Float64})`).",
                ),
            )
        end
        rethrow()
    end
    fdata_bufs = map(x_ndual_bufs) do xb
        fdata(zero_tangent(xb))
    end

    # Discover the output type by running the compiled rule, NOT f(x...) directly.
    # This preserves mutation-restoration semantics (see docstring on prepare_hvp_cache).
    f_codual = zero_fcodual(f)
    x_coduals = map(CoDual, x_ndual_bufs, fdata_bufs)
    out, pb = ndual_rule(f_codual, x_coduals...)
    y_out = primal(out)
    is_ndual_out = y_out isa Nfwd.NDual
    T_out = is_ndual_out ? typeof(y_out.value) : typeof(y_out)
    T_out <: IEEEFloat || throw(
        ArgumentError("HVP/Hessian requires a scalar `IEEEFloat` output; got `$T_out`.")
    )

    # Warm the pullback once so its caches and stacks are populated. The seed
    # must match the output's lifted shape: `_hvp_make_seed` for NDual outputs
    # (RData carrying the `(value, partials)` named tuple), `zero_tangent` for
    # the rare non-NDual scalar fallback. fdata_bufs are zeroed unconditionally
    # below, so the prep call can't leak side effects.
    pb(is_ndual_out ? _hvp_make_seed(T_out) : zero_tangent(y_out))

    # Re-zero fdata_bufs after the preparatory run.
    for k in eachindex(fdata_bufs)
        fresh = fdata(zero_tangent(x_ndual_bufs[k]))
        copyto!(fdata_bufs[k], fresh)
    end

    seed = _hvp_make_seed(T_out)
    specs = _hvp_input_specs(f, x)
    core = (; ndual_rule, fdata_bufs, x_ndual_bufs, seed)
    return HVPCache{:reverse_over_forward,typeof(core),typeof(specs)}(core, specs)
end

function _hvp_make_seed(::Type{T}) where {T<:IEEEFloat}
    NT = @NamedTuple{value::T, partials::Tuple{T}}
    return RData{NT}((value=zero(T), partials=(one(T),)))
end

"""
    value_and_hvp!!(cache, f, v, x...)

Compute the value, gradient, and Hessian-vector product of scalar-valued `f` at `x`
along direction `v`.

For single-argument `f(x::Vector)`, returns `(value, gradient, hvp)`.
For multi-argument `f(x1, x2, ...)`, returns
`(value, (g1, g2, ...), (Hv1, Hv2, ...))`.

`v` must provide one direction vector per differentiable input, matching the shape
of `x`. Only `Vector{<:IEEEFloat}` inputs are supported.

For the single-argument case, `v` may be either an `AbstractVector` or a 1-tuple
containing one vector. For multi-argument cases, `v` must be a tuple of vectors
of length `length(x)`.
"""
function value_and_hvp!!(
    cache::HVPCache{:reverse_over_forward}, f, v, x::Vararg{Any,N}
) where {N}
    _validate_prepared_cache(getfield(cache, :input_specs), (f, x...))
    v = _normalise_directions(v, x)
    core = getfield(cache, :core)
    x_ndual_bufs = core.x_ndual_bufs
    fdata_bufs = core.fdata_bufs
    for k in 1:N
        xk, vk, buf = x[k], v[k], x_ndual_bufs[k]
        for i in eachindex(buf)
            buf[i] = Nfwd.NDual(xk[i], (vk[i],))
        end
    end
    for k in 1:N
        fresh = fdata(zero_tangent(x_ndual_bufs[k]))
        copyto!(fdata_bufs[k], fresh)
    end
    f_codual = zero_fcodual(f)
    x_coduals = map(CoDual, x_ndual_bufs, fdata_bufs)
    out, pb = core.ndual_rule(f_codual, x_coduals...)
    y_out = primal(out)
    if y_out isa Nfwd.NDual
        pb(core.seed)
        val = y_out.value
    else
        pb(one(typeof(y_out)))
        val = y_out
    end
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

# ── :forward_over_reverse internals ──────────────────────────────────────────────

function _prepare_hvp(::Val{:forward_over_reverse}, f::F, x::Tuple, config) where {F}
    for (i, xi) in enumerate(x)
        xi isa Vector{<:IEEEFloat} || throw(
            ArgumentError(
                "HVP/Hessian requires `Vector{<:IEEEFloat}` inputs; " *
                "argument $i has type `$(typeof(xi))`.",
            ),
        )
    end
    grad_cache = prepare_gradient_cache(f, x...; config)
    N = length(x)
    grad_f = _GradClosure{N,F,typeof(grad_cache)}(f, grad_cache)
    # The outer derivative cache must run width-1: it differentiates a single direction
    # `v` per HVP call. Forwarding the user's `chunk_size` would compile the gradient
    # closure for width-N NDual inputs and then fail when `value_and_gradient!!` reads
    # cache fields through an `NTangent`-typed input.
    fwd_cache = prepare_derivative_cache(
        grad_f, x...; config=Config(config; chunk_size=nothing, empty_cache=false)
    )
    specs = _hvp_input_specs(f, x)
    core = (; grad_cache, grad_f, grad_tangent=zero_tangent(grad_f), fwd_cache)
    return HVPCache{:forward_over_reverse,typeof(core),typeof(specs)}(core, specs)
end

struct _GradClosure{N,F,GC}
    f::F
    grad_cache::GC
end

function (gc::_GradClosure{1})(y)
    val_and_grad = value_and_gradient!!(gc.grad_cache, gc.f, y)
    return (val_and_grad[1], val_and_grad[2][2])
end

function (gc::_GradClosure{N})(ys...) where {N}
    val_and_grad = value_and_gradient!!(gc.grad_cache, gc.f, ys...)
    return (val_and_grad[1], Base.tail(val_and_grad[2]))
end

@inline function _assert_matching_tangent_shape(primal, tangent, arg_index::Int)
    if applicable(axes, primal) && applicable(axes, tangent)
        axes(primal) == axes(tangent) || throw(
            ArgumentError(
                "Tangent direction for argument $arg_index must match the primal axes; " *
                "got axes $(axes(tangent)) vs $(axes(primal))",
            ),
        )
    elseif applicable(length, primal) && applicable(length, tangent)
        length(primal) == length(tangent) || throw(
            ArgumentError(
                "Tangent direction for argument $arg_index must match the primal length; " *
                "got $(length(tangent)) vs $(length(primal))",
            ),
        )
    end
    return nothing
end

function value_and_hvp!!(cache::HVPCache{:forward_over_reverse}, f, v, x1)
    core = getfield(cache, :core)
    _validate_prepared_cache(getfield(cache, :input_specs), (f, x1))
    if v isa AbstractVector
        _assert_matching_tangent_shape(x1, v, 1)
    else
        length(v) == 1 || throw(
            ArgumentError("Expected 1 direction vector for single-argument function.")
        )
        _assert_matching_tangent_shape(x1, v[1], 1)
        v = v[1]
    end
    grad_f = _GradClosure{1,typeof(f),typeof(core.grad_cache)}(f, core.grad_cache)
    (f_val, grad), (_, hvp) = value_and_derivative!!(
        core.fwd_cache, (grad_f, core.grad_tangent), (x1, v)
    )
    return f_val, copy(grad), copy(hvp)
end

function value_and_hvp!!(
    cache::HVPCache{:forward_over_reverse}, f, v::Tuple, x1, xrest::Vararg{Any,N}
) where {N}
    all_x = (x1, xrest...)
    core = getfield(cache, :core)
    _validate_prepared_cache(getfield(cache, :input_specs), (f, all_x...))
    length(v) == length(all_x) || throw(
        ArgumentError("Expected $(length(all_x)) direction vector(s), got $(length(v))."),
    )
    for i in eachindex(all_x)
        _assert_matching_tangent_shape(all_x[i], v[i], i)
    end
    grad_f = _GradClosure{length(all_x),typeof(f),typeof(core.grad_cache)}(
        f, core.grad_cache
    )
    (f_val, grads), (_, hvps) = value_and_derivative!!(
        core.fwd_cache, (grad_f, core.grad_tangent), map(tuple, all_x, v)...
    )
    return f_val, map(copy, grads), map(copy, hvps)
end

# ── Shared helpers ───────────────────────────────────────────────────────────────

function _hvp_input_specs(f, x::Tuple)
    return map((f, x...)) do xi
        if xi isa AbstractArray
            InputSpec(typeof(xi), size(xi))
        else
            InputSpec(typeof(xi), ())
        end
    end
end

function _normalise_directions(v, x::Tuple)
    N = length(x)
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
    return v
end

"""
    prepare_hessian_cache(f, x...; config=Mooncake.Config())

Prepare a cache for Hessian evaluation. Delegates to [`prepare_hvp_cache`](@ref).

Only `Vector{<:IEEEFloat}` inputs are supported.
"""
function prepare_hessian_cache(f, x::Vararg{Any,N}; config=Config()) where {N}
    for (i, xi) in enumerate(x)
        xi isa Vector{<:IEEEFloat} || throw(
            ArgumentError(
                "Hessian requires `Vector{<:IEEEFloat}` inputs; " *
                "argument $i has type `$(typeof(xi))`.",
            ),
        )
    end
    return prepare_hvp_cache(f, x...; config)
end

function _validate_jacobian_argument(x)
    x isa AbstractVector || throw(
        ArgumentError(
            "value_and_jacobian!! only supports AbstractVector inputs; got $(typeof(x))"
        ),
    )
    T = eltype(x)
    T <: IEEEFloat || throw(
        ArgumentError(
            "value_and_jacobian!! only supports AbstractVector inputs with IEEEFloat element types; got eltype $T",
        ),
    )
    x isa DenseVector || throw(
        ArgumentError(
            "value_and_jacobian!! only supports dense vector inputs; got $(typeof(x))"
        ),
    )
    return T
end

function _throw_jacobian_eltype_mismatch(Tx, Ty)
    throw(
        ArgumentError(
            "value_and_jacobian!! requires input and output AbstractVector element types to match; got input eltype $Tx and output eltype $Ty",
        ),
    )
end

function _throw_jacobian_output_type_error(y)
    throw(
        ArgumentError(
            "value_and_jacobian!! only supports AbstractVector outputs; got $(typeof(y))"
        ),
    )
end

function _validate_jacobian_output(y, Tx)
    y isa AbstractVector || _throw_jacobian_output_type_error(y)
    Ty = eltype(y)
    Ty <: IEEEFloat || throw(
        ArgumentError(
            "value_and_jacobian!! only supports AbstractVector outputs with IEEEFloat element types; got eltype $Ty",
        ),
    )
    Ty == Tx || _throw_jacobian_eltype_mismatch(Tx, Ty)
    return Ty
end

"""
    value_and_jacobian!!(cache::FCache, f, x)
    value_and_jacobian!!(cache::Cache, f, x)

Using a pre-built cache, compute and return `(value, jacobian)` for a vector-valued
function `f` of a single vector input.

The current implementation supports a single dense vector input and an
`AbstractVector` output, both with the same `IEEEFloat` element type. The returned
Jacobian is a dense matrix whose columns correspond to input coordinates.

!!! info
    `cache` must be the output of [`prepare_derivative_cache`](@ref) or
    [`prepare_pullback_cache`](@ref), and `f` and `x` must match the types and shapes used
    to construct the cache.
"""
@unstable @inline function value_and_jacobian!!(
    cache::FCache, f::F, x::AbstractVector{<:IEEEFloat}
) where {F}
    _validate_jacobian_argument(x)
    _validate_prepared_cache(getfield(cache, :input_specs), (f, x))
    total_dof = length(x)
    total_dof > 0 ||
        throw(ArgumentError("value_and_jacobian!! requires a non-empty input vector"))
    width = cache.width
    if width === nothing
        return _jacobian_width1(cache.rule, (f, x), total_dof, eltype(x))
    else
        return _jacobian_widthN(cache, (f, x), total_dof, eltype(x), width)
    end
end

function _jacobian_width1(rule, input_primals, total_dof, Tx)
    seed1 = _make_seed_tangent(input_primals, 1)
    output1 = rule(tuple_map(Dual, input_primals, seed1)...)
    y = primal(output1)
    Ty = _validate_jacobian_output(y, Tx)
    J = zeros(Ty, length(y), total_dof)
    @inbounds J[:, 1] .= tangent(output1)
    for slot in 2:total_dof
        seed = _make_seed_tangent(input_primals, slot)
        output = rule(tuple_map(Dual, input_primals, seed)...)
        @inbounds J[:, slot] .= tangent(output)
    end
    return y, J
end

function _jacobian_widthN(
    cache::FCache, input_primals, total_dof, Tx, ::Val{W}
) where {W}
    rule = cache.rule
    # Lift / seed buffers are reused across calls (lazy on first invocation),
    # mirroring `_gradient_widthN`.
    lift_bufs = let cached = cache.lift_buf[]
        if cached === nothing
            allocated = tuple_map(p -> _alloc_lift_buf(Val(W), p), input_primals)
            cache.lift_buf[] = allocated
            allocated
        else
            cached
        end
    end
    seed_bufs = let cached = cache.seed_buf[]
        if cached === nothing
            buf = ntuple(_ -> _alloc_aliased_tangents(input_primals), Val(W))
            cache.seed_buf[] = buf
            buf
        else
            cached
        end
    end

    cursor = Ref(0)
    seed_seen = _new_seen(input_primals)
    # Run the first chunk to learn the output shape, then allocate J once and
    # fill it in place from this and all remaining chunks.
    output = _run_jacobian_chunk(
        rule, lift_bufs, seed_bufs, input_primals, 1, min(W, total_dof), cursor, seed_seen
    )
    y = primal(output)
    Ty = _validate_jacobian_output(y, Tx)
    J = zeros(Ty, length(y), total_dof)
    _write_jacobian_columns!(J, output, 1, min(W, total_dof))
    slot = W + 1
    while slot <= total_dof
        chunk = min(W, total_dof - slot + 1)
        output = _run_jacobian_chunk(
            rule, lift_bufs, seed_bufs, input_primals, slot, chunk, cursor, seed_seen
        )
        _write_jacobian_columns!(J, output, slot, chunk)
        slot += chunk
    end
    return y, J
end

@inline function _run_jacobian_chunk(
    rule, lift_bufs, seed_bufs, input_primals, slot, chunk, cursor, seed_seen
)
    seeds = ntuple(Val(length(seed_bufs))) do d
        cursor[] = 0
        seed_seen isa IdDict{Any,Nothing} && empty!(seed_seen)
        # Lanes beyond `chunk` reuse `slot` (their partials are written but
        # never read into J — see `_write_jacobian_columns!`).
        target_slot = d <= chunk ? slot + d - 1 : slot
        _seed_inplace!(seed_bufs[d], input_primals, target_slot, cursor, seed_seen)
    end
    ndual_inputs = ntuple(Val(length(input_primals))) do i
        _combine_to_ndual_or_buffer(
            lift_bufs[i], input_primals[i], ntuple(d -> seeds[d][i], Val(length(seed_bufs)))
        )
    end
    return rule(ndual_inputs...)
end

@inline function _write_jacobian_columns!(J, output, slot, chunk)
    for d in 1:chunk
        col = slot + d - 1
        @inbounds for i in eachindex(output)
            J[i, col] = output[i].partials[d]
        end
    end
    return nothing
end

@unstable @inline function value_and_jacobian!!(
    cache::Cache, f::F, x::AbstractVector{<:IEEEFloat}
) where {F}
    _validate_jacobian_argument(x)
    _validate_prepared_cache(getfield(cache, :input_specs), (f, x))
    total_dof = length(x)
    total_dof > 0 ||
        throw(ArgumentError("value_and_jacobian!! requires a non-empty input vector"))
    y_cache = cache.y_cache
    Ty = _validate_jacobian_output(y_cache, eltype(x))
    ȳ = zeros(Ty, length(y_cache))
    J = zeros(Ty, length(ȳ), total_dof)
    if isempty(ȳ)
        y, _ = value_and_pullback!!(cache, ȳ, f, x)
        return y, J
    end

    ȳ[1] = one(Ty)
    y, pb = value_and_pullback!!(cache, ȳ, f, x)
    @inbounds J[1, :] .= pb[2]
    ȳ[1] = zero(Ty)

    @inbounds for row in 2:length(ȳ)
        ȳ[row] = one(Ty)
        _, pb = value_and_pullback!!(cache, ȳ, f, x)
        J[row, :] .= pb[2]
        ȳ[row] = zero(Ty)
    end

    return y, J
end

@unstable function value_and_jacobian!!(cache::Union{Cache,FCache}, f::F, x) where {F}
    _validate_jacobian_argument(x)
    return _validate_prepared_cache(getfield(cache, :input_specs), (f, x))
end

@unstable function value_and_jacobian!!(cache, f::F, x) where {F}
    throw(ArgumentError("value_and_jacobian!! only supports cache types Cache and FCache"))
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
    _validate_prepared_cache(getfield(cache, :input_specs), (f, x...))

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
        val = f(x...)
        if N == 1
            return val, zeros(T, 0), zeros(T, 0, 0)
        else
            grad_zeros = ntuple(k -> zeros(T, sizes[k]), Val(N))
            H_blocks = ntuple(Val(N)) do i
                ntuple(Val(N)) do j
                    zeros(T, sizes[i], sizes[j])
                end
            end
            return val, grad_zeros, H_blocks
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
