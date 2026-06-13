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
    # safe on Julia 1.10; the `(rule::Any)` dispatch barrier is not needed here.
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
    value_and_derivative!!(rule, f::Lifted, x::Lifted...)
    value_and_derivative!!(rule, (f, df), (x, dx), ...)

Run a forward rule directly, without first constructing a `FCache`.

The `Lifted` interface returns the rule output (a `Lifted`) directly. The tuple interface
returns `(y, dy)` using the rule's native tangent representation. Both compute a single
directional derivative (one tangent per input).
"""
@inline function value_and_derivative!!(rule::R) where {R}
    throw(
        ArgumentError(
            "`value_and_derivative!!(rule, ...)` expects at least the function input, " *
            "either as `f::Lifted` or `(f, df)`.",
        ),
    )
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Lifted,N}) where {R,N}
    return __call_rule(rule, fx)
end

@inline function value_and_derivative!!(rule::R, fx::Vararg{Tuple{Any,Any},N}) where {R,N}
    input_primals = tuple_map(first, fx)
    input_tangents = tuple_map(last, fx)
    input_lifteds = tuple_map(lift, input_primals, input_tangents)
    output = __call_rule(rule, input_lifteds)
    return primal(output), last(unlift(output))
end

# Cache types in this file:
# - `Cache`: reusable reverse-mode cache for repeated `value_and_pullback!!` and
#   `value_and_gradient!!` calls.
# - `FCache`: reusable forward-mode cache for repeated `value_and_derivative!!` and
#   `value_and_gradient!!` calls.
# - `HVPCache`: reusable forward-over-reverse cache for repeated `value_and_hvp!!` calls;
#   Hessian helpers reuse this cache rather than introducing a separate Hessian cache type.
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
# The two-argument methods are the allocation-free hot path (input restore on every
# autodiff pass); they recurse two-argument and stay byte-identical to the original
# acyclic implementation. Only at a cycle-capable node — a mutable struct or a
# reference-element array, which can be self-referential — do they re-dispatch to
# the three-argument family below, which threads an `IdDict` aliasing cache: each
# mutable `dst` is registered (keyed by its `src`) before its fields are restored,
# so a cycle returns the in-progress `dst` instead of recursing forever. Mirrors
# reverse-mode's `MaybeCache`.
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

# copy for Array, Memory. This acyclic method recurses two-argument; an array only
# participates in a cycle as a field of a cyclic mutable struct, which enters the
# three-argument family first (whose array method threads the cache), so this
# method never needs the cache itself and stays identical to the original.
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

    # Mutable structs can be self-referential — handle them via the cyclic family.
    ismutable(src) && return _copy_to_output!!(dst, src, IdDict{Any,Any}())

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

# ── Cyclic family: threads the `IdDict` aliasing cache `c` ─────────────────────
_copy_to_output!!(dst::Number, src::Number, ::IdDict) = src
_copy_to_output!!(::Type, src::Type, ::IdDict) = src
_copy_to_output!!(::Core.TypeName, src::Core.TypeName, ::IdDict) = src
_copy_to_output!!(::Module, src::Module, ::IdDict) = src
function _copy_to_output!!(dst::SimpleVector, src::SimpleVector, c::IdDict)
    return Core.svec(map((d, s) -> _copy_to_output!!(d, s, c), dst, src)...)
end
function _copy_to_output!!(dst::P, src::P, c::IdDict) where {P<:_BuiltinArrays}
    if !isbitstype(eltype(P))
        haskey(c, src) && return c[src]::P
        c[src] = dst
    end
    @inbounds for i in eachindex(src)
        if isassigned(src, i)
            dst[i] = if isassigned(dst, i)
                _copy_to_output!!(dst[i], src[i], c)
            else
                _copy_output(src[i], c)
            end
        end
    end
    return dst
end
function _copy_to_output!!(dst::P, src::P, c::IdDict) where {P<:Union{Tuple,NamedTuple}}
    isbitstype(P) && return src
    return map((d, s) -> _copy_to_output!!(d, s, c), dst, src)
end
function _copy_to_output!!(dst::P, src::P, c::IdDict) where {P}
    isbitstype(P) && return src
    nf = nfields(src)
    nf == 0 && return src
    if ismutable(src)
        haskey(c, src) && return c[src]::P
        c[src] = dst
        for src_sub in 1:nf
            if isdefined(src, src_sub)
                # using ccall as setfield! fails for const fields of a mutable struct.
                ccall(
                    :jl_set_nth_field,
                    Cvoid,
                    (Any, Csize_t, Any),
                    dst,
                    src_sub - 1,
                    _copy_to_output!!(getfield(dst, src_sub), getfield(src, src_sub), c),
                )
            end
        end
        return dst
    else
        flds = Vector{Any}(undef, nf)
        for src_sub in 1:nf
            if isdefined(src, src_sub)
                flds[src_sub] = _copy_to_output!!(
                    getfield(dst, src_sub), getfield(src, src_sub), c
                )
            else
                nf = src_sub - 1
                break
            end
        end
        !isassigned(flds, 1) && return src
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), P, flds, nf)::P
    end
end

"""
    _copy_output(x::T)

Returns a copy of `x`, of the same type `T`. Allocates new memory for the copy.
Required as Base.copy() does not work for all supported primal types. For example, `Base.copy` does not work for `Core.svec`.
For types with custom copy semantics, overload this function (see `Core.SimpleVector` for an example).
"""
# The optional aliasing cache `c::C` supports self-referential and aliased inputs:
# each cycle-capable node is registered before its fields are copied, so a cycle
# returns the in-progress copy rather than recursing forever. The cache is
# allocated lazily — only on first reaching a mutable struct or reference-element
# array, by re-dispatching with a fresh `IdDict`. `C` is a concrete type parameter
# (`Nothing` or `IdDict`) per call rather than a `Union`, which would force dynamic
# dispatch. Unlike the in-place `_copy_to_output!!` restore, `_copy_output` always
# allocates fresh copies and runs only at cache preparation, so it does not need
# `_copy_to_output!!`'s allocation-free two-family split. Mirrors `MaybeCache`.

# Type values (DataType, UnionAll, Union), Core.TypeName, and Modules
# cannot be deep-copied; return x as-is.
@unstable _copy_output(x::Type, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x
_copy_output(x::Core.TypeName, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x
_copy_output(x::Module, c::C=nothing) where {C<:Union{Nothing,IdDict}} = x

function _copy_output(x::SimpleVector, c::C=nothing) where {C<:Union{Nothing,IdDict}}
    return Core.svec([map(s -> _copy_output(s, c), x_sub) for x_sub in x]...)
end

# Array, Memory. Only reference-element arrays can participate in a cycle, so the
# isbits-element case skips the cache entirely.
function _copy_output(x::P, c::C=nothing) where {P<:_BuiltinArrays,C<:Union{Nothing,IdDict}}
    Tx = eltype(P)
    if !isbitstype(Tx)
        c === nothing && return _copy_output(x, IdDict{Any,Any}())
        haskey(c, x) && return c[x]::P
    end
    temp = similar(x)
    isbitstype(Tx) || (c[x] = temp)
    @inbounds for i in eachindex(temp)
        if isassigned(x, i)
            temp[i] = _copy_output(x[i], c)::Tx
        end
    end
    return temp::P
end

# Tuple, NamedTuple
function _copy_output(
    x::Union{Tuple,NamedTuple}, c::C=nothing
) where {C<:Union{Nothing,IdDict}}
    return map(s -> _copy_output(s, c), x)::typeof(x)
end

# mutable composite types, bitstype
function _copy_output(x::P, c::C=nothing) where {P,C<:Union{Nothing,IdDict}}
    isbitstype(P) && return x
    # nfields(x) not nfields(P): the latter counts fields of the
    # DataType object itself.
    nf = nfields(x)

    # No Julia-visible fields (e.g. Symbol, String): nothing to copy.
    # Overload _copy_output to customise.
    nf == 0 && return x

    if ismutable(x)
        c === nothing && return _copy_output(x, IdDict{Any,Any}())
        haskey(c, x) && return c[x]::P
        _copy_output_mutable_cartesian(x, Val(nf), c)
    else
        _copy_output_immutable_cartesian(x, Val(nf), c)
    end
end

@generated function _copy_output_mutable_cartesian(x::P, ::Val{nf}, c::IdDict) where {P,nf}
    quote
        temp = ccall(:jl_new_struct_uninit, Any, (Any,), P)::P
        # Register before copying fields so a self-reference resolves to `temp`.
        c[x] = temp
        Base.Cartesian.@nexprs(
            $nf,
            i -> if isdefined(x, i)
                ccall(
                    :jl_set_nth_field,
                    Cvoid,
                    (Any, Csize_t, Any),
                    temp,
                    i - 1,
                    _copy_output(getfield(x, i), c),
                )
            end
        )
        return temp::P
    end
end

@generated function _copy_output_immutable_cartesian(
    x::P, ::Val{nf}, c::C
) where {P,nf,C<:Union{Nothing,IdDict}}
    quote
        Base.Cartesian.@nif(
            $(nf + 1),
            # Assumes if a undefined field is found, all subsequent fields are undefined.
            i -> !isdefined(x, i),
            i -> _copy_output_immutable_cartesian_upto(x, Val(i - 1), c),
        )
    end
end
@generated function _copy_output_immutable_cartesian_upto(
    x::P, ::Val{idx}, c::C
) where {P,idx,C<:Union{Nothing,IdDict}}
    idx == 0 && return :(x)
    return quote
        flds = collect(
            Any, Base.Cartesian.@ntuple($idx, i -> _copy_output(getfield(x, i), c))
        )
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

struct FCache{R,IT<:Union{Nothing,Tuple},OP,FG,GW,CF,S<:Tuple,IS,GS}
    single_rule::R
    input_tangents::IT
    output_primal::OP
    friendly_gradients::FG
    gradient_workspace::GW
    gradient_chunk_size::Int
    gradient_chunk_size_auto::Bool
    chunk_rule::CF
    input_specs::S
    # Reusable buffer holding a copy of the input args `x...` (not `f`, which is never
    # mutated and may be uncopyable, e.g. the HVP `grad_f` closure), allocated once at cache
    # construction. The public API snapshots into it and restores from it (in place, via
    # `_copy_to_output!!`) around every call, so the inputs are never mutated even though
    # the forward rule aliases (and an in-place `f` mutates) the user's storage.
    input_snapshot::IS
    # Preallocated seeds for the zero-allocation packable gradient over one or more same-eltype
    # float vectors: `(f_seed, arg_seeds, grad_bufs)` — a `gradient_chunk_size`-wide width-`W`
    # `Lifted` per arg (over a cache-owned primal buffer whose partials are mutated in place per
    # chunk) plus a preallocated per-arg gradient buffer. `nothing` for every other input shape
    # (differentiable `f`, structs, tuples, complex, mixed eltypes, …), which uses the generic
    # chunked gradient path.
    gradient_seed::GS
end

@inline _dual_primal_type(::Type) = Any
@inline _dual_primal_type(::Type{<:Lifted{Y}}) where {Y} = Y

@inline function _forward_cache_output_summary(cache::FCache)
    output_primal = getfield(cache, :output_primal)
    return if !isnothing(output_primal)
        _cache_spec_summary(
            if output_primal isa AbstractArray
                InputSpec(typeof(output_primal), size(output_primal))
            else
                InputSpec(typeof(output_primal), ())
            end,
        )
    else
        lifted_arg_types = Tuple{
            map(
                spec -> lifted_type(Val(1), typeof(spec).parameters[1]),
                getfield(cache, :input_specs),
            )...,
        }
        output_type = Core.Compiler.return_type(
            getfield(cache, :single_rule), lifted_arg_types
        )
        _cache_type_summary(_dual_primal_type(output_type))
    end
end

function Base.show(io::IO, cache::FCache)
    chunk_size = getfield(cache, :gradient_chunk_size)
    print(
        io,
        "Mooncake.FCache(",
        "mode=:forward, ",
        "friendly_tangents=",
        !isnothing(getfield(cache, :input_tangents)),
        ", chunk=",
        !isnothing(getfield(cache, :chunk_rule)),
        ", chunk_size=",
        getfield(cache, :gradient_chunk_size_auto) ? "$(chunk_size) (auto)" : chunk_size,
        ", inputs=",
        _cache_input_count(cache),
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", cache::FCache)
    chunk_size = getfield(cache, :gradient_chunk_size)
    print(
        io,
        "Mooncake.FCache\n",
        "  mode: forward\n",
        "  friendly_tangents: ",
        !isnothing(getfield(cache, :input_tangents)),
        "\n",
        "  chunk: ",
        !isnothing(getfield(cache, :chunk_rule)),
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

const _MAX_CHUNK_WIDTH = 8

struct PreparedCacheError <: Exception
    msg::String
end

function Base.showerror(io::IO, err::PreparedCacheError)
    _print_boxed_error(io, split("PreparedCacheError:\n$(err.msg)", '\n'))
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

# fcache gradient bookkeeping
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

# `dof(t)` counts the differentiable scalar degrees of freedom of a TANGENT `t`, so the
# canonical non-differentiable `NoTangent` is 0 directly. Walk with an identity cache so
# aliased mutable tangents contribute once and cyclic tangents terminate locally. Dense leaf
# counts reuse the nfwd engine's slot vocabulary (`_nfwd_input_dof`, the single source of
# truth); the dedup wrapper around array/mutable nodes is the gradient-specific extension
# (nfwd never dedups). IEEEFloat/Complex array tangents are isbits and always assigned, so
# the count equals `length`/`2length`.
@inline dof(t) = dof(t, IdDict{Any,Any}())
@inline dof(::NoTangent, ::IdDict{Any,Any}) = 0
@inline dof(t::IEEEFloat, ::IdDict{Any,Any}) = Nfwd._nfwd_input_dof(t)
@inline dof(t::Complex{<:IEEEFloat}, ::IdDict{Any,Any}) = Nfwd._nfwd_input_dof(t)
# `Union{}`-eltype arrays (e.g. an empty `Memory{Union{}}` reached while walking a closure
# tangent like the HVP `grad_f`'s `MistyClosureTangent`) carry no differentiable content; 0.
# More specific than the float/complex-array and generic-array methods, so no ambiguity.
@inline dof(::AbstractArray{Union{}}, ::IdDict{Any,Any}) = 0
@inline function dof(
    t::AbstractArray{<:Union{IEEEFloat,Complex{<:IEEEFloat}}}, seen::IdDict{Any,Any}
)
    haskey(seen, t) && return 0
    seen[t] = nothing
    return Nfwd._nfwd_input_dof(t)
end
@inline function dof(t::AbstractArray, seen::IdDict{Any,Any})
    haskey(seen, t) && return 0
    seen[t] = nothing
    total = 0
    if t isa _BuiltinArrays
        for i in eachindex(t)
            isassigned(t, i) || continue
            total += dof(t[i], seen)
        end
    else
        for ti in t
            total += dof(ti, seen)
        end
    end
    return total
end
@inline function dof(t::Tuple, seen::IdDict{Any,Any})
    total = 0
    for ti in t
        total += dof(ti, seen)
    end
    return total
end
@inline function dof(t::NamedTuple, seen::IdDict{Any,Any})
    total = 0
    for ti in values(t)
        total += dof(ti, seen)
    end
    return total
end
@inline function dof(t::PossiblyUninitTangent, seen::IdDict{Any,Any})
    return is_init(t) ? dof(val(t), seen) : 0
end
# Generic fallback for any other tangent struct — `Tangent`/`MutableTangent` (whose single
# `fields` NamedTuple recurses), but also `MistyClosureTangent` and other custom/closure
# tangents from e.g. the HVP `grad_f`. Walk its fields with mutable-node dedup so aliased and
# cyclic tangents are handled uniformly.
@inline function dof(t::P, seen::IdDict{Any,Any}) where {P}
    if Base.ismutabletype(P)
        haskey(seen, t) && return 0
        seen[t] = nothing
    end
    total = 0
    for n in 1:fieldcount(P)
        isdefined(t, n) && (total += dof(getfield(t, n), seen))
    end
    return total
end

# Snapshot/restore for the chunked gradient/Jacobian sweeps. Forward slots alias the user's
# input (`primal(slot) === x`), and a sweep re-runs `f` once per chunk on that shared storage.
# An in-place-mutating `f` would otherwise compound its mutation across chunks and corrupt
# later chunks' derivatives. Each sweep snapshots the input args into the prepared
# `cache.input_snapshot` buffer (args only — `f` is never mutated and may be uncopyable) and
# restores from it (in place, via `_copy_to_output!!`) before each re-run and once at the end,
# leaving the inputs unchanged, consistent with reverse mode.

"""
    prepare_derivative_cache(fx...; config=Mooncake.Config())

Returns a cache used with [`value_and_derivative!!`](@ref). See that function for more info.

!!! note
    Cache construction stays lazy and does not execute `f(x...)`.
"""
@unstable @inline function prepare_derivative_cache(
    f, x::Vararg{Any,N}; config=Config()
) where {N}
    config.empty_cache && empty_mooncake_caches!()
    fx = (f, x...)
    requested_chunk_size = getfield(config, :chunk_size)
    requested_chunk_size = if isnothing(requested_chunk_size)
        0
    else
        Nfwd._nfwd_check_chunk_size(requested_chunk_size)
    end
    gradient_chunk_size_auto = requested_chunk_size == 0
    rule = build_frule(fx...; config.debug_mode, config.silence_debug_messages)
    input_specs = map(fx) do x
        if x isa AbstractArray
            InputSpec(typeof(x), size(x))
        else
            InputSpec(typeof(x), ())
        end
    end
    # Chunking only batches packable inputs (a non-differentiable `f` plus IEEEFloat
    # scalar/array args); other inputs (structs, tuples, complex, …) have no native chunk
    # rule and run width-1, so pin their chunk width to 1 here instead of re-deciding
    # packability per chunk at runtime.
    packable =
        tangent_type(typeof(first(fx))) === NoTangent &&
        all(p -> p isa IEEEFloat || p isa Array{<:IEEEFloat}, Base.tail(fx))
    gradient_chunk_size = let total_dof = dof(tuple_map(zero_tangent, fx))
        requested = gradient_chunk_size_auto ? _MAX_CHUNK_WIDTH : requested_chunk_size
        packable ? min(total_dof, requested) : 1
    end
    # The chunk cache is a native width-`W` `frule!!` that evaluates `W` directional
    # derivatives per pass (`W = gradient_chunk_size`). Width 1 carries no batching
    # benefit over `cache.single_rule`, so leave it unbuilt.
    chunk_rule = if gradient_chunk_size > 1
        build_frule(
            fx...;
            chunk_size=gradient_chunk_size,
            config.debug_mode,
            config.silence_debug_messages,
        )
    else
        nothing
    end
    output_primal = nothing
    # Preallocated seed for the zero-allocation single-float-vector gradient path: a width-W
    # `x_seed` over a cache-owned primal buffer (partials mutated in place per chunk) plus the
    # inert `f_seed`. `nothing` for every other shape, which uses the generic gradient path.
    gradient_seed = let args = Base.tail(fx)
        # The same-eltype requirement mirrors the packable method's dispatch
        # (`x1::AbstractVector{T}, xs_rest::Vararg{AbstractVector{T}}`): a mixed-eltype
        # seed would be dead cache weight that dispatch can never reach.
        if gradient_chunk_size >= 1 &&
            !isempty(args) &&
            first(args) isa AbstractVector{<:IEEEFloat} &&
            all(a -> a isa AbstractVector{eltype(first(args))}, args) &&
            packable
            W = gradient_chunk_size
            (
                zero_lifted(Val(W), fx[1]),
                map(a -> zero_lifted(Val(W), similar(a)), args),
                map(similar, args),
            )
        else
            nothing
        end
    end
    if config.friendly_tangents
        input_tangents = tuple_map(zero_tangent, fx)
        gradient_workspace = Ref{Union{Nothing,typeof(input_tangents)}}(nothing)
        return FCache(
            rule,
            input_tangents,
            output_primal,
            _copy_output(fx),
            gradient_workspace,
            gradient_chunk_size,
            gradient_chunk_size_auto,
            chunk_rule,
            input_specs,
            _copy_output(Base.tail(fx)),
            gradient_seed,
        )
    end
    return FCache(
        rule,
        nothing,
        output_primal,
        nothing,
        # Lazy gradient workspace, kept concretely typed (not `Ref{Any}`, which would make
        # cached forward gradients inference-opaque) without evaluating `zero_tangent` on the
        # runtime inputs here.
        Ref{Union{Nothing,Tuple{map(tangent_type, fieldtypes(typeof(fx)))...}}}(nothing),
        gradient_chunk_size,
        gradient_chunk_size_auto,
        chunk_rule,
        input_specs,
        _copy_output(Base.tail(fx)),
        gradient_seed,
    )
end

#
# `value_and_gradient!!` generic chunked path
#
"""
    value_and_gradient!!(cache::FCache, f, x...)

Compute the value and gradient of the scalar-returning `f` at `x...` using forward-mode AD
(the gradient is assembled from standard-basis directional derivatives, evaluated in chunks
of the cache's resolved chunk width). This overload exists so callers can prepare a forward
cache once, then use it either for directional derivatives via
[`value_and_derivative!!`](@ref) or for full gradients.

!!! warning
    `cache` owns any mutable state returned by this function: mutable components of the
    returned gradients will be overwritten if you call this function again with the same
    `cache`. Take a copy (`copy` / `deepcopy`) of anything you need to keep across calls.
"""
function value_and_gradient!!(cache::FCache, f::F, x::Vararg{Any,N}) where {F,N}
    input_primals = (f, x...)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
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
    # `dof` walks the tangent; reuse the freshly-built/zeroed workspace tangent.
    total_dof = dof(native_gradients)

    if total_dof == 0
        output = __call_rule(
            cache.single_rule, tuple_map(lift, input_primals, native_gradients)
        )
        y = primal(output)
        y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
        if isnothing(cache.input_tangents)
            return y, native_gradients
        end
        friendly_gradients = _copy_to_output!!(cache.friendly_gradients, input_primals)
        return y,
        tangent_to_primal_internal!!(
            friendly_gradients,
            native_gradients,
            isbitstype(typeof(friendly_gradients)) ? NoCache() : IdDict{Any,Any}(),
        )
    end

    # Per chunk of `W` standard-basis directions starting at `start_slot`:
    #  - seed the forward direction by basis-seeding the whole input tuple's `zero_lifted` V
    #    (`basis_lifted!!` walks all inputs' dofs with one global cursor; slots past
    #    `total_dof` give zero lanes), then split that tuple V into per-input width-`W` slots;
    #  - run the width-dispatched `value_and_derivative!!` (chunk rule for `W > 1`, single rule
    #    for `W == 1`) and read lane `k`'s directional derivative as `coeff = tangent(out, k)`;
    #  - scatter `coeff * reverse_tangent` into the gradient, where the reverse basis tangent
    #    per lane is the width-1 `basis_lifted!!` seed at that scalar dof, `unlift`ed back to a
    #    reverse tangent (a scalar output makes each lane's derivative the coefficient for its
    #    seeded basis direction).
    # `W == gradient_chunk_size ≤ total_dof`, so the first chunk is always full width; it is
    # peeled out to keep the scalar `y` concretely typed.
    W = cache.gradient_chunk_size
    nfields = Val(fieldcount(typeof(input_primals)))
    P = typeof(input_primals)
    # Snapshot the inputs into the cache buffer before any chunk runs `f`; restore from it
    # before each subsequent chunk (so an in-place `f` does not compound) and once at the end.
    _copy_to_output!!(cache.input_snapshot, Base.tail(input_primals))
    first_lanes = ntuple(
        lane -> last(unlift(basis_lifted!!(zero_lifted(Val(1), input_primals), (lane,)))), W
    )
    first_tangents = ntuple(i -> ntuple(lane -> first_lanes[lane][i], W), nfields)
    first_vs = tangent(
        basis_lifted!!(zero_lifted(Val(W), input_primals), ntuple(identity, W))
    )
    first_lifted = ntuple(
        i -> Lifted{fieldtype(P, i),W}(input_primals[i], first_vs[i]), nfields
    )
    first_out = value_and_derivative!!(cache, first_lifted...)
    y = primal(first_out)
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    for lane in 1:W
        coeff = Float64(tangent(first_out, lane))
        native_gradients = tuple_map(
            (g, dx) -> begin
                lt = dx[lane]
                lt isa NoTangent && return g
                return increment!!(g, _scale(coeff, lt))
            end,
            native_gradients,
            first_tangents,
        )
    end
    for start_slot in (1 + W):W:total_dof
        _copy_to_output!!(Base.tail(input_primals), cache.input_snapshot)
        slots = ntuple(lane -> start_slot + lane - 1, W)
        lanes = ntuple(
            lane -> last(
                unlift(basis_lifted!!(zero_lifted(Val(1), input_primals), (slots[lane],))),
            ),
            W,
        )
        input_tangents = ntuple(i -> ntuple(lane -> lanes[lane][i], W), nfields)
        vs = tangent(basis_lifted!!(zero_lifted(Val(W), input_primals), slots))
        lifted = ntuple(i -> Lifted{fieldtype(P, i),W}(input_primals[i], vs[i]), nfields)
        output = value_and_derivative!!(cache, lifted...)
        for lane in 1:W
            slot = start_slot + lane - 1
            slot <= total_dof || break
            coeff = Float64(tangent(output, lane))
            native_gradients = tuple_map(
                (g, dx) -> begin
                    lt = dx[lane]
                    lt isa NoTangent && return g
                    return increment!!(g, _scale(coeff, lt))
                end,
                native_gradients,
                input_tangents,
            )
        end
    end
    # Final restore so the inputs are left unchanged (each chunk ran on the original).
    _copy_to_output!!(Base.tail(input_primals), cache.input_snapshot)

    if isnothing(cache.input_tangents)
        return y, native_gradients
    end
    friendly_gradients = _copy_to_output!!(cache.friendly_gradients, input_primals)
    return y,
    tangent_to_primal_internal!!(
        friendly_gradients,
        native_gradients,
        isbitstype(typeof(friendly_gradients)) ? NoCache() : IdDict{Any,Any}(),
    )
end

#
# `value_and_gradient!!` fast paths
#
# FCache path overview:
# - derivative machinery: `value_and_derivative!!` (width-dispatched single/chunk rule).
# - gradient machinery: `value_and_gradient!!` (scalar / packable-vector / generic chunked).
#
# Gradient dispatch summary for `value_and_gradient!!(cache, f, x...)`:
# - `x::IEEEFloat`: scalar width-1 path
# - all-`AbstractVector{<:IEEEFloat}`: zero-allocation packable path (preallocated seeds)
# - otherwise: generic chunked path, which per chunk seeds `gradient_chunk_size` standard-basis
#   directions and runs the width-dispatched `value_and_derivative!!`

# Scalar `value_and_gradient!!` fast path: a single width-1 forward evaluation through
# `cache.single_rule`. A scalar input has one degree of freedom, so there is nothing to chunk;
# this avoids the generic path's standard-basis seeding and lane accumulation.
@inline function value_and_gradient!!(cache::FCache, f::F, x::T) where {F,T<:IEEEFloat}
    # A differentiable `f` carries its own degrees of freedom: the width-1 single-seed run below
    # cannot represent them (and `lift(f, NoTangent())` would seed uninitialised tangent storage),
    # so fall back to the generic chunked path, which sweeps `f`'s dofs too.
    tangent_type(F) === NoTangent ||
        return invoke(value_and_gradient!!, Tuple{FCache,Any,Vararg{Any}}, cache, f, x)
    _validate_prepared_cache(getfield(cache, :input_specs), (f, x))
    output = __call_rule(cache.single_rule, (lift(f, NoTangent()), lift(x, one(x))))
    y = primal(output)
    y isa IEEEFloat || throw_val_and_grad_ret_type_error(y)
    native_gradients = (NoTangent(), last(unlift(output)))
    if isnothing(cache.input_tangents)
        return y, native_gradients
    end
    friendly_gradients = _copy_to_output!!(cache.friendly_gradients, (f, x))
    return y,
    tangent_to_primal_internal!!(
        friendly_gradients,
        native_gradients,
        isbitstype(typeof(friendly_gradients)) ? NoCache() : IdDict{Any,Any}(),
    )
end

# Zero-allocation packable gradient for one or more same-eltype float vectors. Reuse the
# preallocated width-`W` seeds (`cache.gradient_seed = (f_seed, arg_seeds, grad_bufs)`):
# per chunk, mutate each arg seed's `NDualArray` partials in place to set standard-basis
# directions (mapping the global slot to the owning arg via running offsets), run the
# width-dispatched `value_and_derivative!!`, and scatter each lane's directional derivative
# straight into the preallocated per-arg gradient buffer. The seed primals are restored from
# `xs` and the partials zeroed at the top of every chunk, so an in-place `f` neither touches the
# user's arrays nor compounds across chunks. A differentiable `f`, mixed
# element types, or any other shape has no preallocated seed (`gradient_seed === nothing`) and
# falls back to the generic chunked path.
function value_and_gradient!!(
    cache::FCache, f::F, x1::AbstractVector{T}, xs_rest::Vararg{AbstractVector{T},Nm1}
) where {F,T<:IEEEFloat,Nm1}
    # Leading `x1` binds `T` directly (a bare `Vararg{AbstractVector{T},N}` leaves `T` unbound at N=0;
    # Aqua). Gradient always has >=1 input. Reconstruct `xs`/`N` to leave the body below unchanged.
    xs = (x1, xs_rest...)
    N = Nm1 + 1
    input_primals = (f, xs...)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
    seed = cache.gradient_seed
    # A differentiable `f` (or any non-packable shape) has no preallocated seed; fall back to
    # the generic chunked method via `invoke` (the args are float vectors, so a plain call
    # would re-dispatch here).
    seed === nothing &&
        return invoke(value_and_gradient!!, Tuple{FCache,Any,Vararg{Any}}, cache, f, xs...)
    f_seed_stored, arg_seeds, grad_bufs = seed
    # Re-wrap the CALL-time `f` (the stored seed holds the prepare-time instance, and a
    # non-differentiable callable can still carry primal-visible state). `V === NoDual` is
    # guaranteed by the packability gate, so this is a free isbits rewrap.
    f_seed = typeof(f_seed_stored)(f, tangent(f_seed_stored))
    z = zero(T)
    for i in 1:N
        fill!(grad_bufs[i], z)
    end
    total_dof = sum(length, xs)
    W = _lifted_width(f_seed)
    y = z
    s = 1
    while s <= total_dof
        # Re-seed every chunk: an in-place `f` mutates the seed primals (and its rule scales the
        # partials) during the previous chunk's run, so restore the primals from the user's
        # arrays and zero the partials before setting this chunk's basis directions — otherwise
        # the mutation compounds across chunks and the gradient is silently wrong.
        off = 0
        @inbounds for i in 1:N
            nda = arg_seeds[i].value
            copyto!(nda.primal, xs[i])
            len = length(xs[i])
            for lane in 1:W
                fill!(nda.partials[lane], z)
                slot = s + lane - 1
                (slot <= total_dof && off < slot <= off + len) &&
                    (nda.partials[lane][slot - off] = one(T))
            end
            off += len
        end
        output = value_and_derivative!!(cache, f_seed, arg_seeds...)
        yv = primal(output)
        yv isa IEEEFloat || throw_val_and_grad_ret_type_error(yv)
        y = yv
        off = 0
        @inbounds for i in 1:N
            nda = arg_seeds[i].value
            gb = grad_bufs[i]
            len = length(xs[i])
            for lane in 1:W
                slot = s + lane - 1
                if slot <= total_dof && off < slot <= off + len
                    gb[slot - off] += tangent(output, lane)
                end
            end
            off += len
        end
        s += W
    end
    native_gradients = (NoTangent(), grad_bufs...)
    if isnothing(cache.input_tangents)
        return y, native_gradients
    end
    friendly_gradients = _copy_to_output!!(cache.friendly_gradients, input_primals)
    return y,
    tangent_to_primal_internal!!(
        friendly_gradients,
        native_gradients,
        isbitstype(typeof(friendly_gradients)) ? NoCache() : IdDict{Any,Any}(),
    )
end

"""
    value_and_derivative!!(cache::FCache, f::Lifted, x::Vararg{Lifted,N})

Returns a `Lifted` containing the result of applying forward-mode AD to compute the (Frechet)
derivative of `primal(f)` at the primal values in `x` in the direction of the tangent values
in `f` and `x`.
"""
# Derivative dispatch summary for `value_and_derivative!!(cache, ...)`. Both compute a
# single directional derivative (one tangent per input); chunking is internal to
# `value_and_gradient!!` / `value_and_jacobian!!`.
# - `value_and_derivative!!(cache, lifteds...)`: native/internal tangent interface;
#   calls the cached `frule` directly
# - `value_and_derivative!!(cache, (f, df), (x, dx), ...)`: tuple interface; lifts each
#   width-1 tangent and runs the cached `frule`
# Width dispatch on the `Lifted{P,N,V}` width parameter: all-width-1 slots are a single
# directional derivative through `single_rule`; width-`W` slots are a `W`-lane chunk through
# `chunk_rule` (built at that width). `Lifted{<:Any,1}` is strictly more specific, so the
# first method serves single directions and the second serves chunks.
function value_and_derivative!!(cache::FCache, fx::Vararg{Lifted{<:Any,1},N}) where {N}
    input_primals = map(primal, fx)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
    return __call_rule(cache.single_rule, fx)
end
function value_and_derivative!!(cache::FCache, fx::Vararg{Lifted,N}) where {N}
    input_primals = map(primal, fx)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
    rule = cache.chunk_rule
    rule === nothing && throw(
        PreparedCacheError(
            "This cache holds no chunk rule: width-N Lifted inputs require the cache to " *
            "have been prepared with chunk_size > 1 (resolved chunk width " *
            "$(getfield(cache, :gradient_chunk_size))).",
        ),
    )
    W = getfield(cache, :gradient_chunk_size)
    _lifted_width(first(fx)) == W || throw(
        PreparedCacheError(
            "Lifted inputs have chunk width $(_lifted_width(first(fx))), but this " *
            "cache's chunk rule was built at width $W.",
        ),
    )
    return __call_rule(rule, fx)
end

"""
    value_and_derivative!!(cache::FCache, (f, df), (x, dx), ...)

Returns a tuple `(y, dy)` containing the result of applying forward-mode AD to compute the (Frechet) derivative of `primal(f)` at the primal values in `x` in the direction of the tangent values contained in `df` and `dx`.

Tuples are used as inputs and outputs instead of a combined value/tangent wrapper to accommodate the case where internal Mooncake tangent types do not coincide with tangents provided by the user (in which case we translate between "friendly tangents" and internal tangents using cache storage).

The arguments in `x` are returned to their original state: if `f` mutates them in place, they are restored from a cache-owned snapshot, so the inputs are not mutated. `f` itself is not snapshotted — a callable that mutates its own fields is not restored.

!!! info
    `cache` must be the output of [`prepare_derivative_cache`](@ref), and (fields of) `f` and `x` must be of the same size and shape as those used to construct the `cache`. This is to ensure that the gradient can be written to the memory allocated when the `cache` was built.

!!! warning
    `cache` owns any mutable state returned by this function, meaning that mutable components of values returned by it will be mutated if you run this function again with different arguments. Therefore, if you need to keep the values returned by this function around over multiple calls to this function with the same `cache`, you should take a copy (using `copy` or `deepcopy`) of them before calling again.
"""
@inline function value_and_derivative!!(
    cache::FCache{R,IT,OP,FG,GW,CF,S}, fx::Vararg{Tuple{Any,Any},M}
) where {R,IT<:Tuple,OP,FG,GW,CF,S,M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
    input_friendly_tangents = tuple_map(last, fx)
    input_tangents = tuple_map(
        primal_to_tangent!!, cache.input_tangents, input_friendly_tangents
    )

    # Snapshot the inputs into the cache buffer and restore from it after the rule runs, so
    # an in-place-mutating `f` does not mutate the user's inputs.
    _copy_to_output!!(cache.input_snapshot, Base.tail(input_primals))
    output = __call_rule(cache.single_rule, tuple_map(lift, input_primals, input_tangents))
    output_primal = primal(output)
    _, output_internal_tangent = unlift(output)
    output_friendly_tangent = tangent_to_friendly!!(
        friendly_tangent_cache(output_primal),
        output_primal,
        output_internal_tangent,
        _friendly_cache((output_primal,)),
    )
    _copy_to_output!!(Base.tail(input_primals), cache.input_snapshot)
    return output_primal, output_friendly_tangent
end

@inline function value_and_derivative!!(
    cache::FCache{R,Nothing,OP,FG,GW,CF,S}, fx::Vararg{Tuple{Any,Any},M}
) where {R,OP,FG,GW,CF,S<:Tuple,M}
    input_primals = tuple_map(first, fx)
    _validate_prepared_cache(getfield(cache, :input_specs), input_primals)
    input_tangents = tuple_map(last, fx)

    # An unfriendly cache (`friendly_tangents=false`) does not translate friendly,
    # primal-shaped tangents, so each supplied tangent must already be the internal tangent
    # for its primal; otherwise `lift` below would fail with an opaque `MethodError`. The
    # `typeof(t) <: tangent_type(typeof(p))` check folds away when it holds.
    tuple_map(input_primals, input_tangents) do p, t
        typeof(t) <: tangent_type(typeof(p)) || throw(
            ArgumentError(
                "Tangent types do not match primal types: tangent $(typeof(t)) is not a " *
                "$(tangent_type(typeof(p))) for primal $(typeof(p)). With " *
                "`friendly_tangents=false`, supply internal tangents (e.g. " *
                "`Mooncake.zero_tangent(x)`) or rebuild the cache with `friendly_tangents=true`.",
            ),
        )
        nothing
    end

    # One aliasing cache scoped to this input lift: a reverse rule captured in
    # `grad_f` shares its `fwds_oc`/`pb_oc` captures, so the forward tangent of
    # that shared mutable state must be shared too (see `lift(::MistyClosure)`).
    c = IdDict()
    input_lifted = tuple_map((p, t) -> lift(p, t, c), input_primals, input_tangents)
    # Snapshot/restore around the rule so an in-place `f` does not mutate the user's inputs.
    _copy_to_output!!(cache.input_snapshot, Base.tail(input_primals))
    output = __call_rule(cache.single_rule, input_lifted)
    result = (primal(output), last(unlift(output)))
    _copy_to_output!!(Base.tail(input_primals), cache.input_snapshot)
    return result
end

# `fwd_cache` is the derivative cache for `grad_f`. The compiled inner rrule is cached
# across `value_and_hvp!!` calls via a `LazyFoRRule` captured inside `fwd_cache`'s frule.
"""
    HVPCache

Cache type used by [`prepare_hvp_cache`](@ref) and [`prepare_hessian_cache`](@ref) for
repeated Hessian-vector product and Hessian evaluations.
"""
struct HVPCache{Tf,Tgrad_f,Tgrad_tangent,Tfwd_cache,TOS,THB}
    f::Tf
    grad_f::Tgrad_f
    # Pre-computed zero tangent for grad_f; the function is never perturbed, only x is.
    # Safe to reuse because grad_f's closure environment is shape-stable for the lifetime
    # of the cache: grad_cache mutates stored values between calls but does not change the
    # closure/capture structure that zero_tangent depends on.
    grad_tangent::Tgrad_tangent
    fwd_cache::Tfwd_cache
    output_spec::TOS
    # Hessian-assembly buffers populated by `prepare_hessian_cache`, `nothing` for caches
    # built via `prepare_hvp_cache`. `value_gradient_and_hessian!!` writes into these.
    # Single-arg layout: `(; H::Matrix, grad::Vector, v::Vector)`.
    # Multi-arg layout:  `(; H_blocks::Tuple, grads::Tuple, vs::Tuple)`.
    hess_buffers::THB
end

function Base.show(io::IO, cache::HVPCache)
    print(
        io,
        "Mooncake.HVPCache(",
        "mode=:forward_over_reverse, ",
        "chunk=",
        !isnothing(getfield(getfield(cache, :fwd_cache), :chunk_rule)),
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
        "  chunk: ",
        !isnothing(getfield(getfield(cache, :fwd_cache), :chunk_rule)),
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

```jldoctest; setup = :(using Mooncake)
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
        f,
        grad_f,
        zero_tangent(grad_f),
        fwd_cache,
        getfield(grad_cache, :output_spec),
        nothing,
    )
end

function _make_hessian_buffers(::Type{T}, xs::Tuple) where {T}
    if length(xs) == 1
        n = length(xs[1])
        return (; H=zeros(T, n, n), grad=zeros(T, n), v=zeros(T, n))
    end
    ns = tuple_map(length, xs)
    nargs = length(xs)
    # H_blocks[k][j] = ∂²f/∂xk∂xj, shape ns[k] × ns[j]
    H_blocks = ntuple(k -> ntuple(j -> zeros(T, ns[k], ns[j]), nargs), nargs)
    grads = tuple_map(ni -> zeros(T, ni), ns)
    vs = tuple_map(ni -> zeros(T, ni), ns)
    return (; H_blocks, grads, vs)
end

@noinline _throw_not_hessian_cache() = throw(
    ArgumentError(
        "`cache` was not built with `prepare_hessian_cache`; rebuild via `prepare_hessian_cache(f, x...)` to use `value_gradient_and_hessian!!`",
    ),
)

@noinline _throw_hessian_arity_mismatch(cached::Int, got::Int) = throw(
    ArgumentError(
        "cache was prepared for $cached argument$(cached == 1 ? "" : "s") but called with $got; rebuild via `prepare_hessian_cache`",
    ),
)

"""
    value_and_hvp!!(cache::HVPCache, f, v, x...)

Given a cache prepared by [`prepare_hvp_cache`](@ref), compute the gradient of `f` at
`x...` and the Hessian-vector product `H v`.

**Single argument:** `v` is the tangent direction; returns `(f(x), ∇f(x), H(x)v)`. For
`f: Rⁿ → R` with `x::Vector{Float64}`, the gradient and HVP are `Vector{Float64}`.

**Multiple arguments:** `v` must be a tuple of tangent directions (one per argument);
returns `(f(x...), (∇f_x1, ∇f_x2, ...), (h1, h2, ...))` where
`hk = ∑_j (∂²f/∂xk∂xj) v[j]` is the joint Hessian-vector product for argument `xk`.

As with all functionality in Mooncake, `x` is returned to its original state: if `f`
mutates `x` in place, it is restored, so the input is not mutated.

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

```jldoctest; setup = :(using Mooncake)
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
    _validate_prepared_cache(getfield(cache.fwd_cache, :input_specs), (cache.grad_f, x1))
    _assert_matching_tangent_shape(x1, v, 1)
    (f_val, grad), (_, hvp) = value_and_derivative!!(
        cache.fwd_cache, (cache.grad_f, cache.grad_tangent), (x1, v)
    )
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
    _validate_prepared_cache(getfield(cache.fwd_cache, :input_specs), input_primals)
    length(v) == length(all_x) ||
        throw(ArgumentError("Expected one tangent direction per primal argument"))
    for i in eachindex(all_x)
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
which is also accepted by [`value_and_hvp!!`](@ref).

The `x...` inputs must be `AbstractVector`s of a single IEEE-float element type;
validation is eager and raises `ArgumentError` here rather than at evaluation time.
The cache pre-allocates the Hessian, gradient, and basis-direction buffers that
[`value_gradient_and_hessian!!`](@ref) writes into, so subsequent calls do not allocate
fresh outputs. The returned `gradient` and Hessian alias cache storage; copy them if
you need to retain previous results.

Hessian computation uses forward-over-reverse AD: one forward-mode pass per input
dimension over the reverse-mode gradient function.

!!! note
    This path uses Mooncake's generic public forward cache over the captured
    reverse-mode gradient closure.

```jldoctest; setup = :(using Mooncake)
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
    N == 0 && throw(ArgumentError("prepare_hessian_cache requires at least one x argument"))
    T = _validate_hessian_arguments(x...)
    base = prepare_hvp_cache(f, x...; config)
    return HVPCache(
        base.f,
        base.grad_f,
        base.grad_tangent,
        base.fwd_cache,
        base.output_spec,
        _make_hessian_buffers(T, x),
    )
end

function _validate_hessian_argument(x, i::Int)
    x isa AbstractVector || throw(
        ArgumentError(
            "Hessian computation only supports AbstractVector inputs; argument $i has type $(typeof(x))",
        ),
    )
    T = eltype(x)
    T <: IEEEFloat || throw(
        ArgumentError(
            "Hessian computation only supports AbstractVector inputs with IEEEFloat element types; argument $i has eltype $T",
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
                "Hessian computation requires all arguments to share the same IEEEFloat element type; argument 1 has eltype $T but argument $i has eltype $Ti",
            ),
        )
    end
    return T
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

As with all functionality in Mooncake, `x` is returned to its original state: if `f`
mutates `x` in place, it is restored, so the input is not mutated.

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
    # `f` non-differentiable and `x::Vector{<:IEEEFloat}` are always packable, so
    # `gradient_chunk_size == min(total_dof, requested)` and the per-chunk width `W` always
    # equals the built `chunk_rule` width — width-dispatched `value_and_derivative!!` routes
    # to `chunk_rule` (W > 1) or `single_rule` (W == 1, no chunk rule). Each chunk seeds `W`
    # standard-basis columns starting at `start_col` via `basis_lifted!!` (slots past
    # `total_dof` map to `0`, an all-zero lane) and reads one Jacobian column per lane.
    W = cache.gradient_chunk_size
    f_seed = zero_lifted(Val(W), f)
    x_seed = zero_lifted(Val(W), x)        # `NDualArray` partials reseeded in place per chunk
    cols(start_col) = ntuple(lane -> let slot = start_col + lane - 1
        slot <= total_dof ? slot : 0
    end, W)
    # Snapshot `x` into the cache buffer (the args copy, so `x` is element 1) before any
    # chunk runs `f`; restore before each subsequent chunk (so an in-place `f` does not
    # compound) and once at the end, leaving `x` unchanged.
    x_snapshot = _copy_to_output!!(cache.input_snapshot[1], x)
    output = value_and_derivative!!(cache, f_seed, basis_lifted!!(x_seed, cols(1)))
    y = primal(output)
    Ty = _validate_jacobian_output(y, eltype(x))
    J = zeros(Ty, length(y), total_dof)
    # First chunk is never short (`W == gradient_chunk_size ≤ total_dof`); only trailing
    # chunks can run past `total_dof` and need the per-lane bound below.
    @inbounds for lane in 1:W
        J[:, lane] .= tangent(output, lane)
    end
    for start_col in (W + 1):W:total_dof
        _copy_to_output!!(x, x_snapshot)
        output = value_and_derivative!!(
            cache, f_seed, basis_lifted!!(x_seed, cols(start_col))
        )
        @inbounds for lane in 1:W
            col = start_col + lane - 1
            col <= total_dof || break
            J[:, col] .= tangent(output, lane)
        end
    end
    # Final restore so the input is left unchanged (each chunk ran on the original).
    _copy_to_output!!(x, x_snapshot)
    return y, J
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
    # Reached only for inputs the methods above reject (`x` is not a dense
    # `AbstractVector{<:IEEEFloat}`). `_validate_jacobian_argument` always throws
    # a specific message here; the explicit throw documents that this fallback
    # never returns a value (previously a dead `_validate_prepared_cache`
    # call left the nominal return as `nothing`).
    _validate_jacobian_argument(x)
    return throw(
        ArgumentError(
            "value_and_jacobian!! only supports dense AbstractVector{<:IEEEFloat} inputs; got $(typeof(x))",
        ),
    )
end

@unstable function value_and_jacobian!!(cache, f::F, x) where {F}
    throw(ArgumentError("value_and_jacobian!! only supports cache types Cache and FCache"))
end

"""
    value_gradient_and_hessian!!(cache::HVPCache, f, x...)

Using a pre-built `cache` from [`prepare_hessian_cache`](@ref), compute and return
`(value, gradient, hessian)` of `f`.

**Single argument:** returns `(f(x), ∇f(x), ∇²f(x))` — value, gradient vector, Hessian
matrix.

**Multiple arguments:** returns `(f(x1,...), (∇_x1 f, ∇_x2 f, ...), H_blocks)` where
`H_blocks[k][j]` is the `nk × nj` matrix `∂²f/∂xk∂xj`. The return structure differs
from the single-argument case.

Uses forward-over-reverse AD: one forward-mode pass per total input dimension.

!!! info
    `cache` must be the output of [`prepare_hessian_cache`](@ref), and `f` must be the
    same function object used to construct `cache`. All `x` arguments must have the
    same sizes and element types as used to construct the cache. The implementation
    supports only `AbstractVector`s of IEEE floats, with all arguments sharing the same
    element type. For non-vector inputs, use [`value_and_hvp!!`](@ref) to obtain
    second-order directional derivatives without forming a full Hessian.

!!! warning
    The returned `gradient` and Hessian alias buffers owned by `cache` and are
    overwritten on the next call with the same cache. Copy them (`copy`/`deepcopy`)
    before mutating or if you need to retain previous results.

!!! warning
    `HVPCache` is not safe for concurrent reuse across threads. Use a separate cache per
    task/thread if calls may overlap in time.

# Example
```jldoctest; setup = :(using Mooncake)
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
    buf = cache.hess_buffers
    buf === nothing && _throw_not_hessian_cache()
    if buf isa NamedTuple{(:H_blocks, :grads, :vs)}
        _throw_hessian_arity_mismatch(length(buf.vs), 1)
    end
    T = _validate_hessian_argument(x1, 1)
    H = buf.H
    g = buf.grad
    v = buf.v
    n = length(x1)
    # Buffer sizes are fixed at cache build time; reject mismatched inputs before
    # indexing `v`/`H`, otherwise the sweep below raises a raw `BoundsError`.
    n == length(v) || throw(
        ArgumentError(
            "input vector has length $n but cache was prepared for length $(length(v)); rebuild via `prepare_hessian_cache`",
        ),
    )
    # Reset `v` in case a prior call threw between `v[i] = one(T)` and `v[i] = zero(T)`.
    fill!(v, zero(T))
    if n == 0
        fval, _, _ = value_and_hvp!!(cache, f, v, x1)
        return fval, g, H
    end
    local value
    for i in 1:n
        v[i] = one(T)
        fval, grad_alias, hvp = value_and_hvp!!(cache, f, v, x1)
        if i == 1
            value = fval
            g .= grad_alias
        end
        @inbounds @views H[:, i] .= hvp
        v[i] = zero(T)
    end
    return value, g, H
end

@unstable @inline function value_gradient_and_hessian!!(
    cache::HVPCache, f::F, x1::T1, xrest::Vararg{Any,N}
) where {F,T1,N}
    cache.f === f || throw(
        ArgumentError("`f` must be the same function object used to construct `cache`")
    )
    buf = cache.hess_buffers
    nargs = N + 1
    buf === nothing && _throw_not_hessian_cache()
    if buf isa NamedTuple{(:H, :grad, :v)}
        _throw_hessian_arity_mismatch(1, nargs)
    end
    all_xs = (x1, xrest...)
    T = _validate_hessian_arguments(all_xs...)
    ns = tuple_map(length, all_xs)
    H_blocks = buf.H_blocks
    grads = buf.grads
    v = buf.vs
    # Buffer arity/sizes are fixed at cache build time; reject mismatched inputs
    # before indexing `v[k]`/`H_blocks`, otherwise the sweep below raises a raw
    # `BoundsError`.
    nargs == length(v) || _throw_hessian_arity_mismatch(length(v), nargs)
    for k in 1:nargs
        ns[k] == length(v[k]) || throw(
            ArgumentError(
                "argument $k has length $(ns[k]) but cache was prepared for length $(length(v[k])); rebuild via `prepare_hessian_cache`",
            ),
        )
    end
    # Reset each `v[k]` in case a prior call threw between `v[k][i] = one(T)` and
    # `v[k][i] = zero(T)`.
    tuple_map(vk -> fill!(vk, zero(T)), v)
    if all(==(0), ns)
        fval, _, _ = value_and_hvp!!(cache, f, v, all_xs...)
        return fval, grads, H_blocks
    end
    local value
    first_iter = true
    for argidx in 1:nargs
        v_i = v[argidx]
        for i in 1:ns[argidx]
            v_i[i] = one(T)
            fval, gs_alias, hvps = value_and_hvp!!(cache, f, v, all_xs...)
            if first_iter
                value = fval
                tuple_map((g, a) -> (g .= a), grads, gs_alias)
                first_iter = false
            end
            tuple_map((Hk, hk) -> (@inbounds @views Hk[argidx][:, i] .= hk), H_blocks, hvps)
            v_i[i] = zero(T)
        end
    end
    return value, grads, H_blocks
end

# IT=Nothing specialisation: disambiguates against the Lifted-vararg and Tuple-vararg
# zero-arg overloads (Aqua detects the ambiguity without this more-specific method).
function value_and_derivative!!(
    cache::FCache{R,Nothing,OP,FG,GW,CF,S}
) where {R,OP,FG,GW,CF,S<:Tuple}
    _validate_prepared_cache(cache.input_specs, ())
    error("unreachable")
end

function value_and_derivative!!(cache::FCache)
    _validate_prepared_cache(cache.input_specs, ())
    error("unreachable")
end
