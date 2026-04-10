"""
    tangent_type(::Val{N}, P::Type)

Public chunked forward tangent representation used by the default `dual_type`
implementation.

This is the forward-mode width-aware overload of `tangent_type`. The ordinary
`tangent_type(P)` query still returns the width-1 tangent type for `P`;
`tangent_type(Val(N), P)` returns the width-`N` chunked representation. Overloads may
return `NTangent`, NDual-style packed tangent storage, or another width-aware tangent
representation.
"""
@unstable @foldable function tangent_type(::Val{N}, ::Type{P}) where {N,P}
    T = tangent_type(P)
    return T == NoTangent ? NoTangent : NTangent{NTuple{N,T}}
end

@inline function _merge_ntangent_lane_counts(a, b)
    a === nothing && return b
    b === nothing && return a
    a == b || throw(ArgumentError("All NTangent lanes must agree; found both $a and $b."))
    return a
end

@inline _ntangent_lane_count(x) = _ntangent_lane_count(x, nothing)

@inline _ntangent_lane_count(::NoTangent, ::Nothing) = nothing
@inline _ntangent_lane_count(::NoTangent, _seen::IdDict{Any,Nothing}) = nothing
@inline _ntangent_lane_count(x::NTangent, ::Nothing) = length(x)
@inline _ntangent_lane_count(x::NTangent, _seen::IdDict{Any,Nothing}) = length(x)
@inline _ntangent_lane_count(x::Tangent, seen::Union{Nothing,IdDict{Any,Nothing}}) = _ntangent_lane_count(
    x.fields, seen
)
@inline function _ntangent_lane_count(
    x::MutableTangent, seen::Union{Nothing,IdDict{Any,Nothing}}
)
    seen = isnothing(seen) ? IdDict{Any,Nothing}() : seen
    haskey(seen, x) && return nothing
    seen[x] = nothing
    return _ntangent_lane_count(x.fields, seen)
end
@inline function _ntangent_lane_count(
    x::PossiblyUninitTangent, seen::Union{Nothing,IdDict{Any,Nothing}}
)
    return is_init(x) ? _ntangent_lane_count(val(x), seen) : nothing
end
@generated function _ntangent_lane_count(
    x::Tuple{Vararg{Any,N}}, seen::Union{Nothing,IdDict{Any,Nothing}}
) where {N}
    lane_count = :(nothing)
    for i in 1:N
        lane_count = :(_merge_ntangent_lane_counts(
            $lane_count, _ntangent_lane_count(getfield(x, $i), seen)
        ))
    end
    return lane_count
end
@generated function _ntangent_lane_count(
    x::NamedTuple{names,Tuple{Vararg{Any,N}}}, seen::Union{Nothing,IdDict{Any,Nothing}}
) where {names,N}
    lane_count = :(nothing)
    for i in 1:N
        lane_count = :(_merge_ntangent_lane_counts(
            $lane_count, _ntangent_lane_count(getfield(x, $i), seen)
        ))
    end
    return lane_count
end
@inline _ntangent_lane_count(::Any, _seen) = nothing

@inline _ntangent_lane(::NoTangent, ::Val) = NoTangent()
@inline _ntangent_lane(x::NTangent, ::Val{lane}) where {lane} = x[lane]
@inline _ntangent_lane(x::Tangent, lane::Val) = Tangent(_ntangent_lane(x.fields, lane))
@inline _ntangent_lane(x::MutableTangent, lane::Val) = MutableTangent(
    _ntangent_lane(x.fields, lane)
)
@inline function _ntangent_lane(x::PossiblyUninitTangent, lane::Val)
    return is_init(x) ? PossiblyUninitTangent(_ntangent_lane(val(x), lane)) : x
end
@generated function _ntangent_lane(x::Tuple{Vararg{Any,N}}, lane::Val) where {N}
    return Expr(:tuple, [:(_ntangent_lane(getfield(x, $i), lane)) for i in 1:N]...)
end
@generated function _ntangent_lane(
    x::NamedTuple{names,Tuple{Vararg{Any,N}}}, lane::Val
) where {names,N}
    fields = Expr(:tuple, [:(_ntangent_lane(getfield(x, $i), lane)) for i in 1:N]...)
    return :(NamedTuple{names}($fields))
end
@inline _ntangent_lane(x, ::Val) = x

@inline _tuple_is_chunked_forward_tangent(::Type{P}, ::Type{<:Tuple}) where {P} =
    !(tangent_type(P) <: Tuple)
@inline function _array_chunk_lane_count(x::AbstractArray, dx::AbstractArray)
    ndims(dx) == ndims(x) + 1 || return nothing
    size(dx)[1:ndims(x)] == size(x) || return nothing
    return size(dx, ndims(dx))
end
@inline _array_chunk_lane_count(::Any, ::Any) = nothing
@inline function _array_chunked_forward_tangent(
    x::AbstractArray, dx::AbstractArray, ::Val{N}
) where {N}
    lane_count = _array_chunk_lane_count(x, dx)
    isnothing(lane_count) && return nothing
    lane_count == N ||
        throw(ArgumentError("Chunked forward tangent expected $N lanes, got $lane_count."))
    return NTangent(ntuple(lane -> copy(selectdim(dx, ndims(dx), lane)), Val(N)))
end
@inline _array_chunked_forward_tangent(::Any, ::Any, ::Val) = nothing

@inline function _canonical_forward_tangent(x::P, dx) where {P}
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    dx isa NTangent && return dx
    array_lane_count = _array_chunk_lane_count(x, dx)
    !isnothing(array_lane_count) &&
        return _array_chunked_forward_tangent(x, dx, Val(array_lane_count))
    dx isa Tuple && _tuple_is_chunked_forward_tangent(P, typeof(dx)) && return NTangent(dx)
    lane_count = _ntangent_lane_count(dx)
    if isnothing(lane_count)
        lane = dx isa NoTangent ? zero_tangent(x) : dx
        return NTangent((lane,))
    end
    lane_count == 1 && return NTangent((_ntangent_lane(dx, Val(1)),))
    return NTangent(ntuple(lane -> _ntangent_lane(dx, Val(lane)), Val(lane_count)))
end

@inline _canonical_forward_tangent(x::Type, dx) = NoTangent()

@inline function canonicalize_chunked_tangent(x::P, dx, ::Val{N}) where {P,N}
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    array_chunked = _array_chunked_forward_tangent(x, dx, Val(N))
    !isnothing(array_chunked) && return array_chunked
    dx isa Tuple &&
        _tuple_is_chunked_forward_tangent(P, typeof(dx)) &&
        return if length(dx) == N
            NTangent(dx)
        else
            throw(
                ArgumentError(
                    "Chunked forward tangent expected $N lanes, got $(length(dx))."
                ),
            )
        end
    lane_count = _ntangent_lane_count(dx)
    if isnothing(lane_count)
        return if dx isa NoTangent
            NTangent(ntuple(_ -> zero_tangent(x), Val(N)))
        elseif N == 1
            # Single lane: no inter-lane aliasing, so skip _copy to preserve any
            # internal aliasing the tangent relies on (e.g. shared Stack memory
            # between fwds_oc and pb_oc in DerivedRule tangents).
            NTangent((dx,))
        else
            # _copy rather than a bare reference: dx may be a mutable tangent (e.g.
            # Array, MutableTangent). Sharing the same object across N lanes would
            # alias them — in-place accumulation on one lane would corrupt the others.
            NTangent(ntuple(_ -> _copy(dx), Val(N)))
        end
    end
    tx = _canonical_forward_tangent(x, dx)
    length(tx) == N || throw(
        ArgumentError("Chunked forward tangent expected $N lanes, got $(length(tx)).")
    )
    return tx
end

@inline canonicalize_chunked_tangent(x::Type, dx, ::Val) = NoTangent()

@inline function zero_tangent(x, ::Val{N}) where {N}
    zx = zero_tangent(x)
    zx isa NoTangent && return NoTangent()
    return NTangent(ntuple(_ -> zero_tangent(x), Val(N)))
end

@inline function zero_tangent(x::Ptr, ::Val{N}) where {N}
    zx = zero_tangent(x, uninit_tangent(x))
    return NTangent(ntuple(_ -> zx, Val(N)))
end

@inline function uninit_tangent(x, ::Val{N}) where {N}
    tx = uninit_tangent(x)
    tx isa NoTangent && return NoTangent()
    return NTangent(ntuple(_ -> uninit_tangent(x), Val(N)))
end

@inline _unwrap_unit_ntangent(dx::NTangent{Tuple{T}}) where {T} = dx[1]
@inline _unwrap_unit_ntangent(dx) = dx

@generated function _ntangent_all_notangent(::Type{NT}) where {L,NT<:NTangent{L}}
    return all(==(NoTangent), L.parameters) ? :(true) : :(false)
end

"""
    Dual(primal::P, tangent::T)

Used to pair together a `primal` value and a `tangent` to it. In the context of forward mode
AD (aka computing Frechet derivatives), `primal` governs the point at which the derivative
is computed, and `tangent` the direction in which it is computed.

The default `dual_type(Val(N), P)` uses `tangent_type(Val(N), P)`, and `dual_type(P)` is
the width-1 shorthand. Explicit hand-written rules may still pass width-1 tangents to
`Dual(x, dx)`, which canonicalises them to Mooncake's internal `NTangent` representation
automatically.
"""
struct Dual{P,T}
    primal::P
    tangent::T
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
# Dual can be safely shared without copying
_copy(x::P) where {P<:Dual} = x

"""
    extract(x::Dual)

Helper function. Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::Dual) = primal(x), tangent(x)

function zero_dual(::Val{N}, x) where {N}
    dual_type(Val(N), typeof(x))(x, zero_tangent(x, Val(N)))
end
zero_dual(x) = dual_type(typeof(x))(x, zero_tangent(x))
zero_dual(::Val, x::Type) = Dual(x, NoTangent())
zero_dual(x::Type) = Dual(x, NoTangent())
function uninit_dual(::Val{N}, x::P) where {N,P}
    dual_type(Val(N), P)(x, uninit_tangent(x, Val(N)))
end
function randn_dual(::Val{N}, rng::AbstractRNG, x) where {N}
    tx = canonicalize_chunked_tangent(x, randn_tangent(rng, x), Val(N))
    return dual_type(Val(N), typeof(x))(x, tx)
end
randn_dual(rng::AbstractRNG, x) = dual_type(typeof(x))(x, randn_tangent(rng, x))

"""
    dual_type(::Val{N}, P::Type)
    dual_type(P::Type)

Returns the forward-mode width-aware dual type for primal type `P`.

`dual_type(P)` is the width-1 default. `dual_type(Val(N), P)` returns the width-`N`
dual type used by chunked forward mode. The default implementation remains
`Dual{P,tangent_type(Val(N), P)}`, but width-aware callers should dispatch through this
interface rather than hard-coding `Dual{...,NTangent...}`. Overloads may return `Dual`,
`NDual`, or another width-aware dual type. Together with `primal`, `tangent`,
`zero_dual(::Val{N}, x)`, and `uninit_dual(::Val{N}, x)`, this forms the minimal
construction / extraction protocol for a custom width-aware dual type.
"""
@unstable @foldable function dual_type(::Val{N}, ::Type{P}) where {N,P}
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(Val(N), P.a),dual_type(Val(N), P.b)}
    # Use `isa` not `<:`: generators like `NTuple{N,Int} where N` are instances of
    # UnionAll but not subtypes of it (`NTuple{N,Int} where N <: UnionAll` is false).
    # `P == UnionAll` handles the UnionAll metatype itself (`UnionAll isa UnionAll` is false).
    (P isa UnionAll || P == UnionAll) && return Dual # P is abstract, tangent type unknown.

    # Union Splitting
    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)

        # If there is exactly one Union field, split it to help the compiler
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{dual_type(Val(N), P_split.a),dual_type(Val(N), P_split.b)}
        end
    end

    return isconcretetype(P) ? Dual{P,tangent_type(Val(N), P)} : Dual
end

@unstable @foldable function dual_type(::Type{P}) where {P}
    return dual_type(Val(1), P)
end

@unstable @foldable function dual_type(::Val, p::Type{Type{P}}) where {P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

_primal(x) = x
_primal(x::Dual) = primal(x)

@inline _dual_width(x) = something(_ntangent_lane_count(tangent(x)), 1)

@inline function _canonicalize_width_aware_dual(x)
    p = primal(x)
    N = _dual_width(x)
    tx = canonicalize_chunked_tangent(p, tangent(x), Val(N))
    return dual_type(Val(N), typeof(p))(p, tx)
end

@inline function _width_aware_dual(x, p, dx)
    N = _dual_width(x)
    return dual_type(Val(N), typeof(p))(p, canonicalize_chunked_tangent(p, dx, Val(N)))
end

"""
    verify_dual_type(x)

Check that the type of `tangent(x)` matches a supported forward tangent type for the type
of `primal(x)`. This is auto-derived from `primal`, `tangent`, `dual_type`,
`tangent_type`, and the current width-detection logic for `NTangent`-compatible tangent
representations. Custom width-aware dual types only need to overload it when they
intentionally diverge from that default shape.
"""
function verify_dual_type(x)
    p = try
        primal(x)
    catch
        return false
    end
    t = try
        tangent(x)
    catch
        return false
    end
    P = typeof(p)
    N = something(_ntangent_lane_count(t), 1)
    expected_dual_type = dual_type(Val(N), P)
    expected_tangent_type = tangent_type(Val(N), P)
    tangent_matches =
        typeof(t) == expected_tangent_type || (N == 1 && typeof(t) == tangent_type(P))
    dual_matches = typeof(x) == expected_dual_type || (x isa Dual && tangent_matches)
    return dual_matches && tangent_matches
end

function error_if_incorrect_dual_types(duals::Vararg{Any,N}) where {N}
    correct_types = map(verify_dual_type, duals)
    if !all(correct_types)
        primal_types = map(duals) do x
            try
                typeof(primal(x))
            catch
                typeof(x)
            end
        end
        tangent_types = map(duals) do x
            try
                typeof(tangent(x))
            catch
                :unavailable
            end
        end
        throw(
            ArgumentError(
                """
Tangent types do not match primal types:
  - primal types:           $(primal_types)
  - provided tangent types: $(tangent_types)
  - supported dual/tangent pairs are derived from `dual_type(Val(N), P)` and
    `tangent_type(Val(N), P)` for the inferred width `N`, with the width-1 fallback
    `tangent_type(P)` still accepted
"""
            ),
        )
    end
end

@inline uninit_dual(x::P) where {P} = dual_type(P)(x, uninit_tangent(x))

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end

function Dual(x::Type{P}, dx::NTangent) where {P}
    _ntangent_all_notangent(typeof(dx)) || throw(
        ArgumentError(
            "forward-mode Duals do not support differentiating with respect to type objects.",
        ),
    )
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, NoTangent())
end

const CanonicalForwardTangent = Union{
    NoTangent,
    IEEEFloat,
    Complex{<:IEEEFloat},
    AbstractArray,
    Tuple,
    NamedTuple,
    Tangent,
    MutableTangent,
    PossiblyUninitTangent,
    Ptr,
}

function Dual(x::F, dx::T) where {F<:Function,T<:CanonicalForwardTangent}
    # Design choice: reject non-`NoTangent` tangents for zero-field `Function` singletons
    # such as `sin`, because Mooncake does not support differentiating with respect to the
    # function object itself. Captured closures are stateful and must fall through to the
    # generic constructor so internal cache-carrying callables can keep their structured
    # tangents.
    fieldcount(F) == 0 || return invoke(Dual, Tuple{Any,T}, x, dx)
    dx isa NoTangent || throw(
        ArgumentError(
            "forward-mode Duals do not support differentiating with respect to `f`."
        ),
    )
    return Dual{F,NoTangent}(x, NoTangent())
end

function Dual(x::F, dx::NTangent) where {F<:Function}
    fieldcount(F) == 0 || return invoke(Dual, Tuple{Any,NTangent}, x, dx)
    _ntangent_all_notangent(typeof(dx)) || throw(
        ArgumentError(
            "forward-mode Duals do not support differentiating with respect to `f`."
        ),
    )
    return Dual{F,NoTangent}(x, NoTangent())
end

function Dual(x::P, dx::NTangent) where {P}
    Px = isconcretetype(P) ? P : typeof(x)
    _ntangent_all_notangent(typeof(dx)) && return Dual{Px,NoTangent}(x, NoTangent())
    return Dual{Px,typeof(dx)}(x, dx)
end

function Dual(x::P, dx::T) where {P,T<:CanonicalForwardTangent}
    # Use the runtime primal type when the call site is abstract. Otherwise forward-mode
    # can build an unsound `Dual` that keeps an abstract source type even after control
    # flow has proved a concrete runtime value.
    #
    # MWE:
    #     pi_node_tester(y::Ref{Any}) = isa(y[], Int) ? sin(y[]) : y[]
    #
    # If `y[] == 5`, then the transformed program reaches the `Int` branch with a concrete
    # runtime primal `5::Int`, even though the source path came through `Any`. Preserving
    # `P == Any` here makes later PiNode refinement act on an abstractly-typed dual rather
    # than on the actual runtime primal, which is incorrect. See the matching PiNode note
    # in `src/interpreter/primal_mode.jl` for the control-flow side of the same fix.
    Px = isconcretetype(P) ? P : typeof(x)
    if (dx isa Tuple && _tuple_is_chunked_forward_tangent(typeof(x), typeof(dx))) ||
        !isnothing(_array_chunk_lane_count(x, dx))
        throw(
            ArgumentError(
                "Chunked forward tangents must be passed explicitly as `NTangent(...)`. " *
                "Do not pass raw tuple lanes or raw trailing-lane array layouts to `Dual(x, dx)`.",
            ),
        )
    end
    dx isa NoTangent && return Dual{Px,T}(x, dx)
    dx_canonical = _canonical_forward_tangent(x, dx)
    return Dual{Px,typeof(dx_canonical)}(x, dx_canonical)
end

function (::Type{Dual{P,NTangent{Tuple{T}}}})(x::P, dx::T) where {P,T}
    return Dual{P,NTangent{Tuple{T}}}(x, NTangent((dx,)))
end
