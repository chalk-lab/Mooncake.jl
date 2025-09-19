"""
    struct DefaultCtx end

Context for all usually used AD primitives. Anything which is a primitive in a MinimalCtx is
a primitive in the DefaultCtx automatically. If you are adding a rule for the sake of
performance, it should be a primitive in the DefaultCtx, but not the MinimalCtx.
"""
abstract type DefaultCtx end

"""
    struct MinimalCtx end

Functions should only be primitives in this context if not making them so would cause AD to
fail. In particular, do not add primitives to this context if you are writing them for
performance only -- instead, make these primitives in the DefaultCtx.
"""
abstract type MinimalCtx <: DefaultCtx end

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
[`is_primitive`](@ref) and [`maybe_primitive`](@ref), and is used by these two functions in
a very non-standard way. In particular, the value these functions return depends on the
signatures of methods of this function, not what the methods do when invoked.

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

const _IS_PRIMITIVE_CACHE = IdDict{Any,Bool}()

"""
    is_primitive(ctx::Type, mode::Type{<:Mode}, sig::Type{<:Tuple}, world::UInt)

Returns a `Bool` specifying whether the methods specified by `sig` are considered primitives
in the context of context `ctx` in mode `mode` at world age `world`.

```jldoctest
julia> using Mooncake: is_primitive, DefaultCtx, ReverseMode

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(sin), Float64}, Base.get_world_counter())
true
```
"""
function is_primitive(ctx::Type, mode::Type, sig::Type{<:Tuple}, world::UInt)
    @nospecialize sig

    # We don't ever need to evaluate this function for abstract `mode`s, and there is a
    # performance penalty associated with doing so, so exclude the possiblity.
    isconcretetype(mode) || throw(ArgumentError("mode $mode is not a concrete type."))

    # Check to see whether any methods of `_is_primitive` exist which apply to this
    # ctx-mode-signature triple in world age `world`. If we have looked this up before,
    # return the answer from the cache.
    tt = Tuple{typeof(_is_primitive),Type{<:ctx},Type{mode},Type{sig}}
    return get!(_IS_PRIMITIVE_CACHE, (world, tt)) do
        return !isempty(Base._methods_by_ftype(tt, -1, world))
    end
end

const _MAYBE_PRIMITIVE_CACHE = IdDict{Any,Bool}()

"""
    maybe_primitive(ctx::Type, mode::Type, sig::Type{<:Tuple}, world::UInt)

`true` if there exists `C<:ctx`, `M<:mode`, and `S<:sig` such that
`is_primitive(C, M, S, world)` returns `true`.

This functionality is used to determine whether or not it is safe to inline away a call
site when performing abstract interpretation using a `MooncakeInterpreter`, which is only
safe to do if the inferred argument types at the call site preclude the call being to a
primitive.

For example, consider the following:
```jldoctest is_prim_example
julia> using Mooncake: Mooncake, @is_primitive, DefaultCtx, ReverseMode

julia> foo(x) = 5x;

julia> @is_primitive DefaultCtx ReverseMode Tuple{typeof(foo),Float64}

```
This function agrees with [`is_primitive`](@ref) for fully inferred call sites:
```jldoctest is_prim_example
julia> world = Base.get_world_counter();

julia> Mooncake.maybe_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Float64}, world)
true

julia> Mooncake.maybe_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Int}, world)
false
```
However, it differs for call sites containing arguments whose types are not fully inferred.
For example:
```jldoctest is_prim_example
julia> Mooncake.is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Real}, world)
false

julia> Mooncake.maybe_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Real}, world)
true
```
Per the definition at the top of this docstring, this function returns `true` because
`Tuple{typeof(foo),Float64} <: Tuple{typeof(foo),Real}`.
"""
function maybe_primitive(ctx::Type, mode::Type{<:Mode}, sig::Type{<:Tuple}, world::UInt)
    @nospecialize sig

    # We don't ever need to evaluate this function for abstract `mode`s, and there is a
    # performance penalty associated with doing so, so exclude the possiblity.
    isconcretetype(mode) || throw(ArgumentError("mode $mode is not a concrete type."))

    # Check to see whether any methods of `_is_primitive` exist which apply to any subtypes
    # of this ctx-mode-signature triple in world age `world`. If we have looked this up
    # before, return the answer from the cache.
    tt = Tuple{typeof(_is_primitive),Type{<:ctx},Type{mode},Type{<:sig}}
    return get!(_MAYBE_PRIMITIVE_CACHE, (world, tt)) do
        return !isempty(Base._methods_by_ftype(tt, -1, world))
    end
end
