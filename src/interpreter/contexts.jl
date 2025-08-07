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
    is_primitive(::Type{Ctx}, ::Type{M}, sig) where {Ctx,M}

Returns a `Bool` specifying whether the methods specified by `sig` are considered primitives
in the context of contexts of type `Ctx` in mode `M`.

```julia
is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(sin), Float64})
```
will return if calling `sin(5.0)` should be treated as primitive when the context is a
`DefaultCtx`.

Observe that this information means that whether or not something is a primitive in a
particular context depends only on static information, not any run-time information that
might live in a particular instance of `Ctx`.
"""
is_primitive(::Type{MinimalCtx}, ::Type{<:Mode}, sig::Type{<:Tuple}) = false
function is_primitive(::Type{DefaultCtx}, ::Type{M}, sig) where {M<:Mode}
    return is_primitive(MinimalCtx, M, sig)
end

"""
    @is_primitive context_type [mode_type] signature

Creates a method of [`is_primitive`](@ref) which always returns `true` for the
`context_type`, and `signature` provided. For example
```jldoctest
julia> using Mooncake: DefaultCtx, @is_primitive, is_primitive, ForwardMode, ReverseMode

julia> foo(x::Float64) = 2x
foo (generic function with 1 method)

julia> @is_primitive DefaultCtx Tuple{typeof(foo),Float64}

julia> is_primitive(DefaultCtx, ForwardMode, Tuple{typeof(foo),Float64})
true

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(foo),Float64})
true
```
Observe that this means that a rule is a primitive in all AD modes.

You should implement more complicated methods of [`is_primitive`](@ref) in the usual way.

Optionally, you can specify that a rule is only a primitive in a particular mode, eg.
```jldoctest
julia> using Mooncake: DefaultCtx, @is_primitive, is_primitive, ForwardMode, ReverseMode

julia> bar(x::Float64) = 2x
bar (generic function with 1 method)

julia> @is_primitive DefaultCtx ForwardMode Tuple{typeof(bar),Float64}

julia> is_primitive(DefaultCtx, ForwardMode, Tuple{typeof(bar),Float64})
true

julia> is_primitive(DefaultCtx, ReverseMode, Tuple{typeof(bar),Float64})
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
        function Mooncake.is_primitive(
            ::Type{$(esc(Tctx))}, ::Type{<:$(Tmode)}, ::Type{<:$(esc(sig))}
        )
            return true
        end
    end
end
