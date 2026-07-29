#
# Primal-shaped ("mirror") tangents
#
# A mirror tangent has the same shape as its primal: every field replaced by that field's
# tangent, so a gradient for `P` comes back as a `P` rather than a `Tangent`. `Array`,
# `Memory` and `IdDict` are hand-written instances of this pattern, each repeating the same
# operations; this file derives that interface from one declaration.
#
# Every operation has the same shape — map over the children, which are the primal's own
# fields — so all logic lives in the ordinary functions below and `@mirror_tangent` emits
# nothing but signatures. That split is deliberate: there is no logic inside macro expansion
# to get wrong, and these functions are testable without expanding a macro. It also buys the
# thing a predicate cannot give, since `tangent_type(P)` alone can select a branch in a body
# but cannot make an existing method's signature match a new type.

##
## Tangent operations
##

@inline _mirror_children(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))
@inline _mirror_rebuild(::Type{T}, children::Tuple) where {T} = _new_(T, children...)

function _mirror_zero(x, d::MaybeCache)
    children = map(c -> zero_tangent_internal(c, d), _mirror_children(x))
    return _mirror_rebuild(tangent_type(typeof(x)), children)
end

function _mirror_randn(rng::AbstractRNG, x, d::MaybeCache)
    children = map(c -> randn_tangent_internal(rng, c, d), _mirror_children(x))
    return _mirror_rebuild(tangent_type(typeof(x)), children)
end

function _mirror_increment(c::IncCache, x::T, y::T) where {T}
    children = map(
        (a, b) -> increment_internal!!(c, a, b), _mirror_children(x), _mirror_children(y)
    )
    return _mirror_rebuild(T, children)
end

function _mirror_set_to_zero(c::SetToZeroCache, x)
    return _mirror_rebuild(
        typeof(x), map(t -> set_to_zero_internal!!(c, t), _mirror_children(x))
    )
end

function _mirror_scale(c::MaybeCache, a::Float64, t)
    return _mirror_rebuild(
        typeof(t), map(x -> _scale_internal(c, a, x), _mirror_children(t))
    )
end

function _mirror_dot(c::MaybeCache, t::T, s::T) where {T}
    children = map(
        (a, b) -> _dot_internal(c, a, b)::Float64, _mirror_children(t), _mirror_children(s)
    )
    return sum(children; init=0.0)
end

function _mirror_add_to_primal(c::MaybeCache, x, t, unsafe::Bool)
    children = map(
        (a, b) -> _add_to_primal_internal(c, a, b, unsafe),
        _mirror_children(x),
        _mirror_children(t),
    )
    return _mirror_rebuild(typeof(x), children)
end

function _mirror_to_primal(x, tx, c::MaybeCache)
    children = map(
        (a, b) -> tangent_to_primal_internal!!(a, b, c),
        _mirror_children(x),
        _mirror_children(tx),
    )
    return _mirror_rebuild(typeof(x), children)
end

function _mirror_to_tangent(tx, x, c::MaybeCache)
    children = map(
        (a, b) -> primal_to_tangent_internal!!(a, b, c),
        _mirror_children(tx),
        _mirror_children(x),
    )
    return _mirror_rebuild(typeof(tx), children)
end

# A mirror is its own fdata only when no field carries reverse data. A field that did would
# need the mirror split into an fdata-shaped and an rdata-shaped copy, which is not generally
# constructible, so refuse rather than silently report the wrong gradient. This runs during
# rule compilation, so a bad declaration fails there rather than at the call site.
@unstable function _mirror_rdata_type(::Type{T}) where {T}
    for F in fieldtypes(T)
        rdata_type(F) === NoRData || throw(
            ArgumentError(
                "@mirror_tangent is not applicable to $T: field type $F carries reverse " *
                "data, so the tangent cannot be shaped like the primal. Leave this type to " *
                "the generic `Tangent` representation.",
            ),
        )
    end
    return NoRData
end

##
## Field access and construction rules
##

# The generic field rules are bound to Mooncake's own fdata layouts, so a mirror needs its
# own. These bodies mirror the generic ones; only the field accessors differ.
function _mirror_field_rrule(x, name, y_primal)
    dx_r = lazy_zero_rdata(primal(x))
    pb!!(dy) = (NoRData(), increment_field!!(instantiate(dx_r), dy, name), NoRData())
    return CoDual(y_primal, _get_fdata_field(primal(x), tangent(x), name)), pb!!
end

function _mirror_getfield(x, name)
    n = primal(name)
    return _mirror_field_rrule(x, n, getfield(primal(x), n))
end

function _mirror_getfield(x, name, order)
    n = primal(name)
    y, pb!! = _mirror_field_rrule(x, n, getfield(primal(x), n, primal(order)))
    return y, dy -> (pb!!(dy)..., NoRData())
end

_mirror_lgetfield(x, ::Val{f}) where {f} = _mirror_field_rrule(x, f, getfield(primal(x), f))

function _mirror_lgetfield(x, name::Val, ::Val)
    y, pb!! = _mirror_lgetfield(x, name)
    return y, dy -> (pb!!(dy)..., NoRData())
end

# Construction. The tangent is built from the field fdatas and aliases them, so nothing flows
# back as reverse data.
function _mirror_new(f, P::CoDual{Type{T}}, xs::CoDual...) where {T}
    y = _new_(T, map(primal, xs)...)
    dy = _new_(tangent_type(T), map(tangent, xs)...)
    return CoDual(y, dy), NoPullback(f, P, xs...)
end

##
## Registration
##

"""
    @mirror_tangent P{T} => tangent_type_expression

Declare that the tangent of `P` is shaped like `P` itself, with every field replaced by that
field's tangent, and derive the tangent interface for it. Gradients for `P` are then returned
as a `P` rather than a `Tangent`, so no further conversion is needed to present them.

```julia
struct Poly{T}
    coeffs::Vector{T}
end

Mooncake.@mirror_tangent Poly{T} => Poly{tangent_type(T)}
```

Applicable when no field of the tangent carries reverse data; a field that did would need the
mirror split in two, which is not generally constructible, and is rejected with an error when
rules are compiled.
"""
macro mirror_tangent(declaration)
    is_pair =
        declaration isa Expr &&
        declaration.head === :call &&
        length(declaration.args) == 3 &&
        declaration.args[1] === :(=>)
    is_pair || throw(
        ArgumentError(
            "@mirror_tangent expects `P{T} => tangent type`, got `$declaration`. See the " *
            "docstring for an example.",
        ),
    )
    primal_type, tangent_expr = declaration.args[2], declaration.args[3]
    P = primal_type isa Expr ? primal_type.args[1] : primal_type
    type_vars =
        primal_type isa Expr ? filter(a -> a isa Symbol, primal_type.args[2:end]) : Symbol[]

    # Every method below is a single call forwarding to one of the functions above.
    return esc(
        quote
            function Mooncake.tangent_type(::Type{$primal_type}) where {$(type_vars...)}
                $tangent_expr
            end
            Mooncake.tangent_type(::Type{T}, ::Type{Mooncake.NoRData}) where {T<:$P} = T

            function Mooncake.zero_tangent_internal(x::$P, d::Mooncake.MaybeCache)
                $(_mirror_zero)(x, d)
            end
            function Mooncake.randn_tangent_internal(
                rng::Mooncake.Random.AbstractRNG, x::$P, d::Mooncake.MaybeCache
            )
                $(_mirror_randn)(rng, x, d)
            end
            function Mooncake.increment_internal!!(
                c::Mooncake.IncCache, x::T, y::T
            ) where {T<:$P}
                $(_mirror_increment)(c, x, y)
            end
            function Mooncake.set_to_zero_internal!!(c::Mooncake.SetToZeroCache, x::$P)
                $(_mirror_set_to_zero)(c, x)
            end
            function Mooncake._scale_internal(c::Mooncake.MaybeCache, a::Float64, t::$P)
                $(_mirror_scale)(c, a, t)
            end
            function Mooncake._dot_internal(
                c::Mooncake.MaybeCache, t::T, s::T
            ) where {T<:$P}
                $(_mirror_dot)(c, t, s)
            end
            function Mooncake._add_to_primal_internal(
                c::Mooncake.MaybeCache, x::$P, t::$P, unsafe::Bool
            )
                $(_mirror_add_to_primal)(c, x, t, unsafe)
            end
            function Mooncake.tangent_to_primal_internal!!(
                x::$P, tx::$P, c::Mooncake.MaybeCache
            )
                $(_mirror_to_primal)(x, tx, c)
            end
            function Mooncake.primal_to_tangent_internal!!(
                tx::$P, x::$P, c::Mooncake.MaybeCache
            )
                $(_mirror_to_tangent)(tx, x, c)
            end

            # The fdata/rdata split. `_mirror_rdata_type` rejects fields carrying rdata.
            Mooncake.fdata_type(::Type{T}) where {T<:$P} = T
            Mooncake.rdata_type(::Type{T}) where {T<:$P} = $(_mirror_rdata_type)(T)
            Mooncake.fdata(t::$P) = t
            Mooncake.rdata(::$P) = Mooncake.NoRData()
            Mooncake.tangent(f::$P, ::Mooncake.NoRData) = f

            # A mirror's fields are its own, not wrapped in `.fields` or `.data`.
            Mooncake._get_fdata_field(t::$P, name) = getfield(t, name)
            Mooncake._get_fdata_field(_, t::$P, name) = getfield(t, name)
            Mooncake._get_tangent_field(t::$P, name) = getfield(t, name)
            Mooncake.TestUtils.__get_data_field(t::$P, name) = getfield(t, name)

            function Mooncake.rrule!!(
                ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
                x::Mooncake.CoDual{<:$P},
                name::Mooncake.CoDual{<:Val},
            )
                $(_mirror_lgetfield)(x, Mooncake.primal(name))
            end
            function Mooncake.rrule!!(
                ::Mooncake.CoDual{typeof(Mooncake.lgetfield)},
                x::Mooncake.CoDual{<:$P},
                name::Mooncake.CoDual{<:Val},
                order::Mooncake.CoDual{<:Val},
            )
                $(_mirror_lgetfield)(x, Mooncake.primal(name), Mooncake.primal(order))
            end
            function Mooncake.rrule!!(
                ::Mooncake.CoDual{typeof(getfield)},
                x::Mooncake.CoDual{<:$P},
                name::Mooncake.CoDual,
            )
                $(_mirror_getfield)(x, name)
            end
            function Mooncake.rrule!!(
                ::Mooncake.CoDual{typeof(getfield)},
                x::Mooncake.CoDual{<:$P},
                name::Mooncake.CoDual,
                order::Mooncake.CoDual,
            )
                $(_mirror_getfield)(x, name, order)
            end
            function Mooncake.rrule!!(
                f::Mooncake.CoDual{typeof(Mooncake._new_)},
                P::Mooncake.CoDual{Type{T}},
                xs::Mooncake.CoDual...,
            ) where {T<:$P}
                $(_mirror_new)(f, P, xs...)
            end
        end,
    )
end
