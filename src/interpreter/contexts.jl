"""
    struct MinimalCtx end

Functions should only be primitives in this context if not making them so would cause AD to
fail. In particular, do not add primitives to this context if you are writing them for
performance only -- instead, make these primitives in the DefaultCtx.
"""
struct MinimalCtx end

"""
    struct DefaultCtx end

Context for all usually used AD primitives. Anything which is a primitive in a MinimalCtx is
a primitive in the DefaultCtx automatically. If you are adding a rule for the sake of
performance, it should be a primitive in the DefaultCtx, but not the MinimalCtx.
"""
struct DefaultCtx end

"""
    abstract type Mode end

Subtypes of this signify which mode of AD is being considered.
"""
abstract type Mode end

"""
    struct ForwardMode end

Used primarily as the second argument to [`is_primitive`](@ref) to determine whether a
function is a primitive in forwards-mode AD.
"""
struct ForwardMode <: Mode end

"""
    struct ReverseMode end

Used primarily as the second argument to [`is_primitive`](@ref) to determine whether a
function is a primitive in reverse-mode AD.
"""
struct ReverseMode <: Mode end

"""
    _is_primitive(context::Type, mode::Type{<:Mode}, sig::Type{<:Tuple})

This function is an internal implementation detail. It is used only by
[`is_primitive`](@ref).

Generally speaking, you ought not to add methods to this function
yourself, but make use of [`@is_primitive`](@ref).
"""
function _is_primitive end

"""
    @is_primitive context_type [mode_type] signature

Declares that calls with signature `signature` are primitives in `context_type` and
`mode_type`. For example
```jldoctest
julia> using Mooncake: DefaultCtx, @is_primitive, is_primitive, ForwardMode, ReverseMode

julia> foo(x::Float64) = 2x
foo (generic function with 1 method)

julia> @is_primitive DefaultCtx Tuple{typeof(foo),Float64}

julia> is_primitive(DefaultCtx, ForwardMode, Tuple{typeof(foo),Float64}, Base.get_world_counter())
true

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Float64}, Base.get_world_counter())
true
```
Observe that this means that a rule is a primitive in all AD modes.

Optionally, you can specify that a rule is only a primitive in a particular mode, eg.
```jldoctest
julia> using Mooncake: DefaultCtx, @is_primitive, is_primitive, ForwardMode, ReverseMode

julia> bar(x::Float64) = 2x
bar (generic function with 1 method)

julia> @is_primitive DefaultCtx ForwardMode Tuple{typeof(bar),Float64}

julia> is_primitive(DefaultCtx, ForwardMode, Tuple{typeof(bar),Float64}, Base.get_world_counter())
true

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(bar),Float64}, Base.get_world_counter())
false
```
"""
macro is_primitive(Tctx, sig)
    return _is_primitive_expression(Tctx, :(Mooncake.Mode), sig)
end

macro is_primitive(Tctx, Tmode, sig)
    return _is_primitive_expression(Tctx, esc(Tmode), sig)
end

function _is_primitive_expression(Tctx, Tmode, sig)
    return quote
        function Mooncake._is_primitive(
            ::Type{$(esc(Tctx))}, ::Type{<:$(Tmode)}, ::Type{<:$(esc(sig))}
        )
            return true
        end
    end
end

"""
    is_primitive(ctx::Type, mode::Type{<:Mode}, sig::Type{<:Tuple}, world::UInt)

Returns a `Bool` specifying whether the methods specified by `sig` are considered primitives
in the context of context `ctx` in mode `mode` at world age `world`.

```jldoctest
julia> using Mooncake: is_primitive, DefaultCtx, ReverseMode

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(sin), Float64}, Base.get_world_counter())
true
```

`world` is needed as rules which Mooncake derives are associated to a particular Julia world
age. As a result, anything declared a primitive after the construction of a rule ought not
to be considered a primitive by that rule. One can explicitly derive a new rule (eg. via
[`build_frule`](@ref), [`build_rrule`](@ref), or a function from the higher-level interface
such as [`prepare_derivative_cache`](@ref), [`prepare_pullback_cache`](@ref) or
[`prepare_gradient_cache`](@ref)) after new `@is_primitive` declarations, should it be
needed in cases where a rule has been previously derived. To see how this works, consider
the following:
```jldoctest
julia> using Mooncake: is_primitive, DefaultCtx, ReverseMode, @is_primitive

julia> foo(x::Float64) = 5x
foo (generic function with 1 method)

julia> old_world_age = Base.get_world_counter();

julia> @is_primitive DefaultCtx ReverseMode Tuple{typeof(foo),Float64}

julia> new_world_age = Base.get_world_counter();

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Float64}, old_world_age)
false

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Float64}, new_world_age)
true
```
Observe that `is_primitive` returns `false` for the world age prior to declaring `foo` a
primitive, but `true` afterwards. For more information on Julia's world age mechanism, see
https://docs.julialang.org/en/v1/manual/methods/#Redefining-Methods .
"""
function is_primitive(
    ctx::Type{MinimalCtx}, mode::Type{<:Mode}, sig::Type{Tsig}, world::UInt
) where {Tsig<:Tuple}
    @nospecialize sig
    try
        Base.invoke_in_world(world, _is_primitive, ctx, mode, sig)::Bool
    catch e
        # Sometimes there are ambiguous `_is_primitive` declarations, especially ones where
        # `f` is a type such as some rules for `Array` and `CuArray`. In these cases we'll
        # just assume that a `MethodError` in `_is_primitive` meant that the method was
        # ambiguous, so we'll assume that the call was meant to be a primitive, and thus
        # should not be inlined.
        e isa MethodError ? true : rethrow(e)
    end
end

function is_primitive(
    ctx::Type{DefaultCtx}, mode::Type{<:Mode}, sig::Type{Tsig}, world::UInt
) where {Tsig<:Tuple}
    @nospecialize sig
    # This function returns `true` if the method is a primitive in either 
    # `DefaultCtx` _or_ `MinimalCtx`.
    try
        Base.invoke_in_world(world, _is_primitive, ctx, mode, sig)::Bool
    catch e
        # Sometimes there are ambiguous `_is_primitive` declarations, especially ones where
        # `f` is a type such as some rules for `Array` and `CuArray`. In these cases we'll
        # just assume that a `MethodError` in `_is_primitive` meant that the method was
        # ambiguous, so we'll assume that the call was meant to be a primitive, and thus
        # should not be inlined.
        e isa MethodError ? true : rethrow(e)
    end
end

_is_primitive(::Type{MinimalCtx}, args...) = false
_is_primitive(::Type{DefaultCtx}, args...) = _is_primitive(MinimalCtx, args...)
