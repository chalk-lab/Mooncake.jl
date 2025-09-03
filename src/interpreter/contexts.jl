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

function _is_primitive end

struct PrimitiveCacheKey{tt}
    world::UInt
end

const _IS_PRIMITIVE_CACHE = IdDict{Any,Bool}()

"""
    is_primitive(ctx::Type, mode::Type{<:Mode}, sig::Type{<:Tuple}, world::UInt)

Returns a `Bool` specifying whether the methods specified by `sig` are considered primitives
in the context of contexts of type `ctx` in mode `mode` at world age `world`.

```jldoctest
julia> using Mooncake: is_primitive, DefaultCtx, ReverseMode

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(sin), Float64}, Base.get_world_counter())
true
```
will return if calling `sin(5.0)` should be treated as primitive when the context is a
`DefaultCtx`.

Observe that this information means that whether or not something is a primitive in a
particular context depends only on static information, not any run-time information that
might live in a particular instance of `Ctx`.
"""
function is_primitive(ctx::Type, mode::Type, sig::Type{<:Tuple}, world::UInt)
    @nospecialize sig
    isconcretetype(mode) || throw(ArgumentError("mode $mode is not a concrete type."))
    tt = Tuple{typeof(_is_primitive),Type{<:ctx},Type{mode},Type{sig}}
    return get!(_IS_PRIMITIVE_CACHE, (world, tt)) do
        return !isempty(Base._methods_by_ftype(tt, -1, world))
    end
end

const _MAYBE_PRIMITIVE_CACHE = IdDict{Any,Bool}()

"""
    maybe_primitive(ctx::Type, mode::Type, sig::Type{<:Tuple}, world::UInt)

`true` if there exists at least one method of `_is_primitive` (typically created using
[`@is_primitive`](@ref)) such that ...
"""
function maybe_primitive(ctx::Type, mode::Type{<:Mode}, sig::Type{<:Tuple}, world::UInt)
    @nospecialize sig
    isconcretetype(mode) || throw(ArgumentError("mode $mode is not a concrete type."))
    tt = Tuple{typeof(_is_primitive),Type{<:ctx},Type{mode},Type{<:sig}}
    return get!(_MAYBE_PRIMITIVE_CACHE, (world, tt)) do
        return !isempty(Base._methods_by_ftype(tt, -1, world))
    end
end

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
