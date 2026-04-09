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

Equivalent to `__value_and_pullback!!(rule, 1.0, f, x...)` -- assumes `f` returns an
`IEEEFloat`.

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

Equivalent to `value_and_pullback!!(rule, 1.0, f, x...)`, and assumes `f` returns an
`IEEEFloat`.

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
    value_and_derivative!!(rule, f, x...)
    value_and_derivative!!(rule, (f, df), (x, dx), ...)

Run a forward rule directly, without first constructing a cache.

The width-aware dual-type interface returns the rule output directly. The tuple interface returns
`(y, dy)` using the rule's native tangent representation. Specialized rule types may
add chunked `NTangent` support on top of this entrypoint.
"""
@inline function value_and_derivative!!(rule::R) where {R}
    throw(
        ArgumentError(
            "`value_and_derivative!!(rule, ...)` expects at least the function input, " *
            "either as a width-aware dual type or `(f, df)`.",
        ),
    )
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Any,N}) where {R,N}
    all(verify_dual_type, fx) || throw(
        ArgumentError(
            "`value_and_derivative!!(rule, ...)` expects width-aware dual-type inputs " *
            "or `(x, dx)` tuples.",
        ),
    )
    return __call_rule(rule, fx)
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
# - `NfwdCache`: temporary nfwd-backed forward-mode cache for prepared
#   `value_and_derivative!!`, `value_and_gradient!!`, and `value_and_pullback!!` calls.
# - `FoRCache`: internal forward cache backed by `build_primal`, used exclusively
#   by HVP/Hessian to differentiate reverse-mode gradient closures.
# - `HVPCache`: reusable forward-over-reverse cache for repeated `value_and_hvp!!` calls;
#   Hessian helpers reuse this cache rather than introducing a separate Hessian cache type.
# The `Cache` parameters are load-bearing: they keep the prepared reverse cache concrete
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
Ensures that `y` or the returned value of `fx::Tuple{Tf, Targs...}` contains no
aliasing, circular references, `Ptr`s, or non-differentiable datatypes.
In the forward pass `f(args...)`, the output can only return a tree-like
datastructure with primitive leaf nodes.
Refer to
https://github.com/chalk-lab/Mooncake.jl/issues/517#issuecomment-2715202789
and the related issue for details.
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
Performs a recursive depth-first search over the function output `y`, with an
`isbitstype()` check as the base case. The visited memory addresses are stored
inside `address_set`.
If the set already contains a newly visited address, it errors out indicating
an alias or circular reference.
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

Copy the contents of `src` to `dst`, with zero or minimal new memory
allocation. The type of `dst` and `src` must be the same.
Required because `Base.copy!()` does not work for all supported primal types.
For example, `Base.copy!` does not work for `Core.svec`.
For types with custom copy semantics, overload this function (see
`Core.SimpleVector` for an example).
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
Required because `Base.copy()` does not work for all supported primal types.
For example, `Base.copy` does not work for `Core.svec`.
For types with custom copy semantics, overload this function (see
`Core.SimpleVector` for an example).
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

As with all functionality in Mooncake, if `f` modifies itself or `x`,
`value_and_pullback!!` will return both to their original state as part of the
process of computing the pullback.

!!! info
    `cache` must be the output of [`prepare_pullback_cache`](@ref), and (fields of) `f` and
    `x` must be of the same size and shape as those used to construct the `cache`. This is
    to ensure that the pullback can be written to the memory allocated when the `cache`
    was built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable
    components of values returned by it will be mutated if you run this function again with
    different arguments. Therefore, if you need to keep the values returned by this function
    around over multiple calls to this function with the same `cache`, you should take a
    copy (using `copy` or `deepcopy`) of them before calling again.

The keyword argument `args_to_zero` is a tuple of boolean values specifying
which cotangents should be reset to zero before differentiation.
It contains one boolean for each element of `(f, x...)`.
It is used for performance optimizations if you can guarantee that the initial
cotangent allocated in `cache` (created by `zero_tangent`) never needs to be
zeroed out again.

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
If the cache was prepared with `config.friendly_tangents=true`, the gradient
uses the same types as those of `f` and `x`. Otherwise, it uses the tangent
types associated to `f` and `x`.

Assumes that `f` returns an `IEEEFloat`.

As with all functionality in Mooncake, if `f` modifies itself or `x`, `value_and_gradient!!`
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

The keyword argument `args_to_zero` is a tuple of boolean values specifying
which cotangents should be reset to zero before differentiation.
It contains one boolean for each element of `(f, x...)`.
It is used for performance optimizations if you can guarantee that the initial
cotangent allocated in `cache` (created by `zero_tangent`) never needs to be
zeroed out again.

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

@inline _dual_primal_type(::Type) = Any
@inline _dual_primal_type(::Type{Dual{Y,T}}) where {Y,T} = Y

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

@inline _internal_forward_dual(x, dx) = Dual(x, _canonical_forward_tangent(x, dx))

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

#
# Scalar pullbacks from cached forward rules
#
# These are the public scalar-output pullback / gradient entrypoints for width-1 and
# chunked forward rules. They stay on ordinary primals plus `NTangent` lanes,
# seed standard-basis chunk tangents internally, and accumulate scalar coefficients into
# full gradients. This keeps the public interface on top of forward rules rather than
# wrapping them in a separate reverse-rule object.
#
# MWE:
# `build_frule(sum, randn(8); chunk_size=4)` builds a cached frule that
# evaluates `sum` on ordinary primal arrays. `value_and_pullback!!` and
# `value_and_gradient!!` then seed four basis directions at a time and accumulate the
# resulting coefficients into the full input gradient.

@inline _irfwd_dof(x, seen::IdDict{Any,Any}=IdDict{Any,Any}()) = _primal_input_dof(x, seen)

@inline _irfwd_seed(x, slot::Int) = _primal_rebuild(
    _irfwd_seed,
    _primal_rebuild_ops(
        leaf=(x, cursor, _seen) -> if x isa IEEEFloat
            cursor[] += 1
            cursor[] == slot ? one(x) : zero(x)
        else
            T = typeof(real(x))
            cursor[] += 1
            real_part = cursor[] == slot ? one(T) : zero(T)
            cursor[] += 1
            imag_part = cursor[] == slot ? one(T) : zero(T)
            complex(real_part, imag_part)
        end,
        nodiff=(_x, _cursor, _seen) -> NoTangent(),
        make_tuple_tangent=(P, fields, _state, _seen) -> build_tangent(P, fields...),
        make_namedtuple_tangent=(P, fields, _state, _seen) -> build_tangent(P, fields...),
    ),
    x,
    Ref(0),
    IdDict{Any,Any}(),
)

function _irfwd_value_and_gradient!!(rule, chunk_size::Int, fx::Vararg{CoDual,M}) where {M}
    input_primals = map(primal, fx)
    native_gradients = tuple_map(set_to_zero!!, map(tangent, fx))
    total_dof = _irfwd_dof(input_primals)
    zero_input_tangents = zero_tangent(input_primals)
    y = nothing

    if total_dof == 0
        output = __call_rule(
            rule, tuple_map(_internal_forward_dual, input_primals, native_gradients)
        )
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        return y, native_gradients
    end

    for start_slot in 1:chunk_size:total_dof
        chunk_width = min(chunk_size, total_dof - start_slot + 1)
        lane_tangents = ntuple(
            lane -> if lane <= chunk_width
                _irfwd_seed(input_primals, start_slot + lane - 1)
            else
                zero_input_tangents
            end, Val(chunk_size)
        )
        input_tangents = ntuple(
            i -> begin
                tangent_type(typeof(input_primals[i])) == NoTangent && return NoTangent()
                return NTangent(ntuple(lane -> lane_tangents[lane][i], Val(chunk_size)))
            end,
            Val(fieldcount(typeof(input_primals))),
        )
        y_chunk, chunk_dy = value_and_derivative!!(
            rule, map(tuple, input_primals, input_tangents)...
        )
        if isnothing(y)
            y = y_chunk
            y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        end
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

    return y::IEEEFloat, native_gradients
end

@inline function __value_and_gradient!!(
    rule::Union{DebugFRule,typeof(frule!!)}, fx::Vararg{CoDual,N}
) where {N}
    return _irfwd_value_and_gradient!!(rule, 1, fx...)
end

@inline function __value_and_pullback!!(
    rule::Union{DebugFRule,typeof(frule!!)}, ȳ::T, fx::Vararg{CoDual,N}; y_cache=nothing
) where {T<:IEEEFloat,N}
    y, gradient = _irfwd_value_and_gradient!!(rule, 1, fx...)
    value = if y_cache === nothing
        _copy_output(y)
    else
        _copy_to_output!!(y_cache, y)
    end
    return value, tuple_map(g -> g isa NoTangent ? g : _scale(ȳ, g), gradient)
end

# ── NfwdCache — temporary nfwd-backed forward-mode prepared cache ─────────────────
#
# Narrow support boundary (IEEEFloat, Complex{<:IEEEFloat}, dense Array of those,
# tuples of supported types). No friendly tangents, no debug_mode.
# Uses true nfwd NDual arithmetic dispatch — no build_frule / build_primal.

"""
    NfwdCache{GW, S}

Temporary nfwd-backed prepared forward-mode cache.

  - `GW`: gradient workspace (`Ref` of pre-allocated tangent tuple), or `Nothing`.
  - `S`:  `Tuple` of [`PreparedCacheInputSpec`](@ref) for input validation.

The cache does not store compiled rules. At call time it lifts primals to
`NDual` arrays/scalars, calls `f` directly via Julia dispatch, and extracts
primal/tangent from the NDual result.
"""
struct NfwdCache{GW,S<:Tuple}
    chunk_size::Int
    gradient_workspace::GW
    input_specs::S
end

function Base.show(io::IO, cache::NfwdCache)
    print(
        io,
        "Mooncake.NfwdCache(",
        "mode=:forward, ",
        "inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::NfwdCache)
    print(
        io,
        "Mooncake.NfwdCache\n",
        "  mode: forward\n",
        "  inputs: ",
        _cache_input_count(cache),
        "\n",
    )
end

# ── NfwdCache boundary validation ─────────────────────────────────────────

# Validate that every *x* argument (not f) is within the narrow nfwd boundary.
@inline function _nfwd_cache_validate_inputs(x::Tuple)
    for xi in x
        Nfwd._nfwd_is_supported_primal(xi) || throw(
            ArgumentError(
                "NfwdCache only supports IEEEFloat, Complex{<:IEEEFloat}, and dense " *
                "Array{<:Union{IEEEFloat,Complex{<:IEEEFloat}}} inputs. Got $(typeof(xi)).",
            ),
        )
    end
    return nothing
end

# ── NfwdCache construction ──────────────────────────────────────────────

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns an [`NfwdCache`](@ref) for use with [`value_and_derivative!!`](@ref),
[`value_and_gradient!!`](@ref), and [`value_and_pullback!!`](@ref).

No rules are compiled at cache-construction time. The cache validates inputs,
pre-allocates gradient workspace, and stores a chunk size for the gradient path.

!!! note
    The temporary `NfwdCache` supports only the narrow nfwd boundary:
    `IEEEFloat`, `Complex{<:IEEEFloat}`, and dense `Array`s of those element types.
    `debug_mode=true` and `friendly_tangents=true` are not supported.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    config.debug_mode && throw(ArgumentError("NfwdCache does not support debug_mode=true."))
    config.friendly_tangents &&
        throw(ArgumentError("NfwdCache does not support friendly_tangents=true."))
    _nfwd_cache_validate_inputs(x)
    fx = (f, x...)
    chunk_size = something(config.chunk_size, 1)
    input_specs = map(fx) do xi
        if xi isa AbstractArray
            PreparedCacheInputSpec(typeof(xi), size(xi))
        else
            PreparedCacheInputSpec(typeof(xi), ())
        end
    end
    native_gradients = tuple_map(zero_tangent, fx)
    gradient_workspace = Ref(native_gradients)
    return NfwdCache(chunk_size, gradient_workspace, input_specs)
end

# ── NfwdCache derivative dispatch ─────────────────────────────────────────

"""
    value_and_derivative!!(cache::NfwdCache, (f, df), (x, dx), ...)

Returns a tuple `(y, dy)` containing the result of applying forward-mode AD to
compute the (Frechet) derivative of `f` at the primal values in `x` in
the direction of the tangent values contained in `df` and `dx`.

Uses true nfwd NDual arithmetic: lifts primals+tangents to `NDual`, calls `f`
directly, and extracts primal/tangent from the result.

!!! info
    `cache` must be the output of [`prepare_derivative_cache`](@ref), and (fields of) `f`
    and `x` must be of the same size and shape as those used to construct the `cache`.
"""
@inline function value_and_derivative!!(
    cache::NfwdCache, fx::Vararg{Tuple{Any,Any},M}
) where {M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache_inputs(getfield(cache, :input_specs), input_primals)
    input_tangents = tuple_map(last, fx)
    f_tangent = first(input_tangents)
    f_tangent isa NoTangent || throw(
        ArgumentError(
            "NfwdCache expects NoTangent for the function tangent, got $(typeof(f_tangent)).",
        ),
    )
    x_primals = Base.tail(input_primals)
    x_tangents = Base.tail(input_tangents)
    lifted = _nfwd_lift_primitive_args(Val(1), x_primals, x_tangents)
    y_ndual = first(input_primals)(lifted...)
    p, t = _nfwd_extract_primitive_parts(y_ndual, Val(1))
    return p, t
end

# ── NfwdCache gradient / pullback ─────────────────────────────────────────

function value_and_gradient!!(cache::NfwdCache, f::F, x::Vararg{Any,N}) where {F,N}
    input_primals = (f, x...)
    _validate_prepared_cache_inputs(cache.input_specs, input_primals)
    x_primals = x
    chunk_size = cache.chunk_size
    total_dof = Nfwd._nfwd_input_dof(x_primals)
    grads = tuple_map(set_to_zero!!, cache.gradient_workspace[])
    cache.gradient_workspace[] = grads
    y = nothing
    for start_slot in 1:chunk_size:total_dof
        seeded = _nfwd_seed_primitive_tangents(x_primals, Val(chunk_size), start_slot)
        lifted = _nfwd_lift_primitive_args(Val(chunk_size), x_primals, seeded)
        y_ndual = f(lifted...)
        y_p, dy = _nfwd_extract_primitive_parts(y_ndual, Val(chunk_size))
        if isnothing(y)
            y = y_p
            y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        end
        ȳ = one(typeof(y))
        lane_vals = _nfwd_contract_output(ȳ, dy)
        # Scatter into (f_grad, x_grads...) — f_grad is NoTangent, rest are x grads.
        grads = (
            first(grads),
            _nfwd_scatter_scalar_chunk(
                Base.tail(grads), x_primals, lane_vals, start_slot
            )...,
        )
    end
    cache.gradient_workspace[] = grads
    return y::IEEEFloat, grads
end

@inline function value_and_pullback!!(
    cache::NfwdCache, ȳ::T, f::F, x::Vararg{Any,N}
) where {T<:IEEEFloat,F,N}
    y, grads = value_and_gradient!!(cache, f, x...)
    pullback = tuple_map(g -> g isa NoTangent ? g : _scale(ȳ, g), grads)
    return y, pullback
end

# ── NfwdCache zero-arg disambiguation ───────────────────────────────────────

# Zero-arg disambiguation (Aqua ambiguity detection).
function value_and_derivative!!(cache::NfwdCache{GW,S}) where {GW,S<:Tuple{}}
    _validate_prepared_cache_inputs(cache.input_specs, ())
    error("unreachable")
end

# ── FoRCache ──────────────────────────────────────────────────────────── #

"""
    FoRCache{R, S}

Internal forward-over-reverse derivative cache backed by `build_primal`.

Used exclusively by [`prepare_hvp_cache`](@ref) / [`prepare_hessian_cache`](@ref) to
differentiate the gradient closure `grad_f`.  Unlike [`NfwdCache`](@ref), which lifts
primals to `NDual` and calls `f` directly via Julia dispatch, this cache compiles a
lifted-primal forward rule that handles arbitrary closures (including reverse-mode cache
closures) safely.

  - `R`: the compiled forward rule returned by `build_primal`.
  - `S`: `Tuple` of [`PreparedCacheInputSpec`](@ref) for input validation.
"""
struct FoRCache{R,S<:Tuple}
    rule::R
    input_specs::S
end

function Base.show(io::IO, cache::FoRCache)
    print(io, "Mooncake.FoRCache(inputs=", _cache_input_count(cache), ")")
end

@inline _cache_input_count(cache::FoRCache) = length(cache.input_specs) - 1

"""
    _prepare_for_primal_derivative_cache(f, x...; config=Config())

Build an internal [`FoRCache`](@ref) for differentiating `f` at `x...` using the
lifted-primal forward compiler (`build_primal`).  This is not part of the public API;
it exists to give HVP/Hessian a safe forward path over reverse-mode gradient closures.
"""
@unstable @inline function _prepare_for_primal_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    fx = (f, x...)
    interp = get_interpreter(ForwardMode)
    rule = build_primal(
        interp,
        typeof(fx);
        call_target=f,
        debug_mode=config.debug_mode,
        skip_world_age_check=false,
        tangent_mode=Val(1),
    )
    input_specs = map(fx) do xi
        if xi isa AbstractArray
            PreparedCacheInputSpec(typeof(xi), size(xi))
        else
            PreparedCacheInputSpec(typeof(xi), ())
        end
    end
    return FoRCache(rule, input_specs)
end

@inline function value_and_derivative!!(
    cache::FoRCache, fx::Vararg{Tuple{Any,Any},M}
) where {M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache_inputs(cache.input_specs, input_primals)
    input_tangents = tuple_map(last, fx)
    input_duals = tuple_map(_internal_forward_dual, input_primals, input_tangents)
    all(verify_dual_type, input_duals) || throw(
        ArgumentError(
            "FoRCache: input duals failed type verification. " *
            "All tangents must be width-1 compatible with their primals.",
        ),
    )
    output = cache.rule(input_duals...)
    return primal(output), tangent(output)
end

# ── HVPCache ───────────────────────────────────────────────────────────────── #

# `fwd_cache` is the derivative cache for `grad_f`. The compiled inner rrule is cached
# across `value_and_hvp!!` calls via the higher-order forward-over-reverse path captured
# inside `fwd_cache`'s lifted-primal rule.
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
    fwd_cache = _prepare_for_primal_derivative_cache(grad_f, x...; config)
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
    reverse-mode gradient closure. It does not currently dispatch to the NDual-lifted
    primitive path used by some `prepare_derivative_cache` / `value_and_gradient!!`
    calls.

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
