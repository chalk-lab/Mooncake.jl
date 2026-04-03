"""
    NTangent_type(P::Type)

Width-1 public tangent representation used by `dual_type(P)`.
"""
@unstable function NTangent_type(::Val{N}, ::Type{P}) where {N,P}
    T = tangent_type(P)
    return T == NoTangent ? NoTangent : NTangent{NTuple{N,T}}
end

@unstable function NTangent_type(::Type{P}) where {P}
    return NTangent_type(Val(1), P)
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
@inline _is_raw_chunked_tuple_tangent(x, dx) =
    dx isa Tuple && _tuple_is_chunked_forward_tangent(typeof(x), typeof(dx))
@inline function _array_chunk_lane_count(x::AbstractArray, dx::AbstractArray)
    ndims(dx) == ndims(x) + 1 || return nothing
    size(dx)[1:ndims(x)] == size(x) || return nothing
    return size(dx, ndims(dx))
end
@inline _array_chunk_lane_count(::Any, ::Any) = nothing
@inline _is_raw_chunked_array_tangent(x, dx) = !isnothing(_array_chunk_lane_count(x, dx))
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

@inline function _chunked_forward_tangent(x::P, dx, ::Val{N}) where {P,N}
    tangent_type(typeof(x)) == NoTangent && return NoTangent()
    array_chunked = _array_chunked_forward_tangent(x, dx, Val(N))
    !isnothing(array_chunked) && return array_chunked
    dx isa Tuple &&
        _tuple_is_chunked_forward_tangent(P, typeof(dx)) &&
        return if length(dx) == N
            NTangent(dx)
        else
            throw(
            ArgumentError("Chunked forward tangent expected $N lanes, got $(length(dx)).")
        )
        end
    lane_count = _ntangent_lane_count(dx)
    if isnothing(lane_count)
        return if dx isa NoTangent
            NTangent(ntuple(_ -> zero_tangent(x), Val(N)))
        else
            NTangent(ntuple(_ -> dx, Val(N)))
        end
    end
    tx = _canonical_forward_tangent(x, dx)
    length(tx) == N || throw(
        ArgumentError("Chunked forward tangent expected $N lanes, got $(length(tx)).")
    )
    return tx
end

@inline function _chunked_zero_tangent(x, ::Val{N}) where {N}
    zx = zero_tangent(x)
    zx isa NoTangent && return NoTangent()
    return NTangent(ntuple(_ -> zero_tangent(x), Val(N)))
end

@inline function _chunked_zero_tangent(x::Ptr, ::Val{N}) where {N}
    zx = zero_tangent(x, uninit_tangent(x))
    return NTangent(ntuple(_ -> zx, Val(N)))
end

@inline function _chunked_uninit_tangent(x, ::Val{N}) where {N}
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

Used to pair together a `primal` value and a `tangent` to it. In the context of foward mode
AD (aka computing Frechet derivatives), `primal` governs the point at which the derivative
is computed, and `tangent` the direction in which it is computed.

The default `dual_type(P)` uses `NTangent_type(P)`, but explicit hand-written rules may
still pass width-1 tangents to `Dual(x, dx)`, which canonicalises them to Mooncake's
internal `NTangent` representation automatically.
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

zero_dual(x) = dual_type(typeof(x))(x, zero_tangent(x))
randn_dual(rng::AbstractRNG, x) = dual_type(typeof(x))(x, randn_tangent(rng, x))

@unstable function dual_type(::Type{P}) where {P}
    P == Union{} && return Union{}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(P.a),dual_type(P.b)}
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
            return Union{dual_type(P_split.a),dual_type(P_split.b)}
        end
    end

    return isconcretetype(P) ? Dual{P,NTangent_type(P)} : Dual
end

function dual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

_primal(x) = x
_primal(x::Dual) = primal(x)

@generated function _verify_ntangent_type(::Type{P}, ::Type{NT}) where {P,L,NT<:NTangent{L}}
    T = tangent_type(P)
    checks = map(==(T), L.parameters)
    return all(checks) ? :(true) : :(false)
end

"""
    verify_dual_type(x::Dual)

Check that the type of `tangent(x)` is the tangent type of the type of `primal(x)`.
"""
function verify_dual_type(x::Dual)
    P = typeof(primal(x))
    T = typeof(tangent(x))
    return T == tangent_type(P) ||
           T == NTangent_type(P) ||
           (T <: NTangent && _verify_ntangent_type(P, T))
end

function error_if_incorrect_dual_types(duals::Vararg{Dual,N}) where {N}
    correct_types = map(verify_dual_type, duals)
    if !all(correct_types)
        primals = map(primal, duals)
        tangents = map(tangent, duals)
        throw(
            ArgumentError(
                """
Tangent types do not match primal types:
  - primal types:           $(map(typeof, primals))
  - provided tangent types: $(map(typeof, tangents))
  - supported tangent types: $(map(P -> (tangent_type(P), NTangent_type(P)), map(typeof, primals)))
""",
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
    # in `src/interpreter/forward_mode.jl` for the control-flow side of the same fix.
    Px = isconcretetype(P) ? P : typeof(x)
    if _is_raw_chunked_tuple_tangent(x, dx) || _is_raw_chunked_array_tangent(x, dx)
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
