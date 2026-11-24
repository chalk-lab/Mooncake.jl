# Loaded after nfwd/Nfwd.jl for the NDual carriers. Forward V types encode differentiability,
# wrappers and layout, so frules may need separate methods for shapes one rrule handles.
# When porting a reverse rule, preserve its conventions as well as its formula.

"""
    ImmutableDual{T<:NamedTuple}

Single-field immutable wrapper used as the canonical V for *immutable struct*
primals under the forward-mode structural lift. Its `fields::T` holds
the recursive `NamedTuple{fieldnames(P), Tuple{V_i...}}` of canonical field
Vs, where each `V_i = dual_type(Val(N), fieldtype(P, i))`.
"""
struct ImmutableDual{T<:NamedTuple}
    fields::T
end

Base.:(==)(x::ImmutableDual, y::ImmutableDual) = x.fields == y.fields

"""
    MutableDual{T<:NamedTuple}

Mutable counterpart to `ImmutableDual`.
"""
mutable struct MutableDual{T<:NamedTuple}
    fields::T
end

Base.:(==)(x::MutableDual, y::MutableDual) = x.fields == y.fields

# Type caches keep the source type alive. Its aligned address leaves the tag's low bit
# free to mark a source whose primal and partial layouts match.
struct PointerDual{N}
    lanes::NTuple{N,Ptr{Cvoid}}
    source::UInt
    function PointerDual{N}(lanes::NTuple{N,Ptr{Cvoid}}, ::Type{T}) where {N,T}
        if T !== Union{} && T <: PointerDual
            T === PointerDual{N} && all(p -> p == lanes[1], lanes) || throw(
                ArgumentError(
                    "Pointer metadata must have matching width and one shared buffer."
                ),
            )
        end
        return new{N}(lanes, _pointer_source(T))
    end
    function PointerDual(lanes::NTuple{N,Ptr{Cvoid}}, v::PointerDual{N}) where {N}
        v.source === _pointer_source(PointerDual{N}) &&
            return PointerDual{N}(lanes, PointerDual{N})
        return new{N}(lanes, v.source)
    end
end

@inline function _pointer_source(::Type{T}) where {T}
    T === NoDual ||
        T === Nothing ||
        T === NoTangent ||
        _carries_partials(T) ||
        return _pointer_source(NoTangent)
    source = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), T))
    return source | UInt(isbitstype(T) && T === tangent_type(T) && sizeof(T) > 0)
end

@inline PointerDual(lanes::Tuple{Ptr{T},Vararg{Ptr{T},N}}) where {N,T} = PointerDual{N+1}(
    map(p -> bitcast(Ptr{Cvoid}, p), lanes), T
)
Base.:(==)(x::PointerDual, y::PointerDual) = x.source === y.source && x.lanes == y.lanes

"""
    Lifted{P, N, V}

Forward-mode slot holding a primal of type `P` and its width-`N` forward representation `V`.

- `primal::P`: the user's primal value; a mutable primal is aliased, not copied. Where `rep`
  holds the primal itself (`NDualArray`, `NDualMemoryRef`), the constructor takes it from
  `rep`, so the two cannot disagree.
- `rep::V`: the forward representation. A caller of `Lifted{P, N, V}` must ensure that `V` is
  `dual_type(Val(N), P)` when that is concrete, and a subtype of it otherwise. Only a `Ptr`
  slot is checked: its `rep` must be a `PointerDual{N}`.

Rules dispatch on `Lifted{P, N}` and read a slot with `primal` and `tangent`. They build
outputs with `Lifted{P, N}(primal, rep)`, which infers `V`; given a type `X` as its primal,
it sets `P` to `Type{X}`, whatever `P` the caller wrote.

Width `N == 1` is ordinary forward mode; `N >= 2` is chunked forward mode.
"""
struct Lifted{P,N,V}
    primal::P
    rep::V
    # Inner, so every construction path takes the `Ptr` check and `_slot_primal`.
    function Lifted{P,N,V}(primal, rep) where {P,N,V}
        P <: Ptr &&
            !(rep isa PointerDual{N}) &&
            throw(
                ArgumentError(
                    "Pointer lanes must have width $N and one source element type."
                ),
            )
        return new{P,N,V}(_slot_primal(primal, rep), rep)
    end
end

# NDualArray / NDualMemoryRef name their primal storage; take the slot's primal from there.
@unstable @inline _slot_primal(primal, rep) = primal
@inline _slot_primal(::Any, rep::NDualArray) = getfield(rep, :primal)
@static if VERSION >= v"1.11-rc4"
    @inline _slot_primal(::Any, rep::NDualMemoryRef) = getfield(rep, :primal)
end

"""
    NoDual

Forward-mode sentinel for "this slot has no derivative". Used as the
`V` of `Lifted{P, N, V}` and as the return of `dual_type(Val(N), P)` for
primal types without a meaningful tangent space (integers, booleans,
symbols, modules, types, …).

Parallels reverse-mode's `NoTangent` but lives in the forward-mode V
layer so the dual_type return shape is self-documenting. The two are
intentionally distinct types so a code path's mode (forward vs reverse)
is visible from its argument types.
"""
struct NoDual end

@inline function Lifted{P,N}(primal::P, rep::V) where {P,N,V}
    return Lifted{P,N,V}(primal, rep)
end

@inline function Lifted{P,N}(p::P, lanes::Tuple{Ptr{T},Vararg{Ptr{T}}}) where {P<:Ptr,N,T}
    return Lifted{P,N}(p, PointerDual(lanes))
end
@inline function Lifted{P,N}(p::P, ::NoDual) where {P<:Ptr,N}
    return Lifted{P,N}(p, PointerDual{N}(ntuple(_ -> Ptr{Cvoid}(C_NULL), Val(N)), NoDual))
end

# Sharpen `P` for a `Type{X}` primal: type-constructor rules dispatch on the type slot, and
# `Lifted` is invariant, so a wide `Lifted{DataType,N}` slot would miss them. `P_wide` is what the
# caller type-applied and is ignored.
@inline function Lifted{P_wide,N}(primal::Type{P_sharp}, rep::V) where {P_wide,P_sharp,N,V}
    # A phantom `TypeVar` leaves `P_sharp` unbound, as in the `CoDual(::Type{P})` constructor.
    return Lifted{(@isdefined(P_sharp) ? Type{P_sharp} : typeof(primal)),N,V}(primal, rep)
end

# Accessors — mirror the `CoDual` API.
primal(d::Lifted) = d.primal
tangent(d::Lifted) = d.rep

extract(d::Lifted) = (primal(d), tangent(d))

function Base.:(==)(a::Lifted, b::Lifted)
    return primal(a) == primal(b) && tangent(a) == tangent(b)
end

# Whole-array accessors — O(1) by aliasing.
@inline primal(a::NDualArray) = a.primal
@inline tangent(a::NDualArray) = Nfwd._lane_views(a)

# `NDualMemoryRef` (and its constructor) lives in `src/nfwd/Nfwd.jl`.
# Mooncake-namespace method extensions follow.

@static if VERSION >= v"1.11-rc4"
    @inline primal(a::NDualMemoryRef) = a.primal
    # Bulk/interface access reconstructs the shared (N, ncols) block from its backing ref.
    @inline tangent(a::NDualMemoryRef) = Nfwd._reconstruct_block(a)
end

struct NumericStorage end
struct StructuralStorage end

# Split numeric storage requires a concrete element type; unions (including bottom)
# use structural storage so each runtime element keeps its own forward representation.
@foldable @inline function _storage_kind(::Type{T}) where {T}
    return isconcretetype(T) && T <: FloatOrComplex ? NumericStorage() : StructuralStorage()
end

"""
    dual_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `N`-width forward-mode value type for a primal of
type `P`. This is the type of the inner `V` carried by a
`Lifted{P, N, V}` slot.

Shapes: `NDual{P,N}` for a float, `Complex{NDual{R,N}}` for a complex float; `NDualArray` for a
float or complex `Array` (and `Memory` on 1.11+), whose primal aliases user storage and whose
partials are a slot-local `(N, size...)` block; `NDualMemoryRef` for a `MemoryRef`; element-wise
for tuples, named tuples and other arrays; `ImmutableDual`/`MutableDual` over the per-field
`NamedTuple` for other structs; `NoDual` where `tangent_type(P) === NoTangent`.
"""
@foldable @inline function dual_type(::Val{N}, ::Type{P}) where {N,P<:IEEEFloat}
    return isconcretetype(P) ? NDual{P,N} : Any
end
# Each member mirrors a `tangent_type` method returning `NoTangent`. Other `Integer` subtypes
# can hold differentiable fields, so they take the structural method.
@foldable @inline dual_type(
    ::Val{N},
    ::Type{
        <:Union{
            Bool,
            Base.BitInteger,
            BigInt,
            Char,
            Symbol,
            Nothing,
            Type,
            TypeVar,
            Module,
            Expr,
            Cstring,
            Cwstring,
        },
    },
) where {N} = NoDual
# Bottom type, mirroring `tangent_type(Union{}) === Union{}` and
# `lifted_type(Val(N), Union{}) === Union{}` (e.g. a Bottom-typed IR node after a
# guaranteed-throw call). Without this, `dual_type(Val(N), Union{})` MethodErrors.
@foldable @inline dual_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
@foldable @inline function dual_type(::Val{N}, ::Type{Complex{R}}) where {N,R<:IEEEFloat}
    return Complex{NDual{R,N}}
end
@foldable @inline function dual_type(w::Val, ::Type{Array{T,D}}) where {T,D}
    return _dual_type(_storage_kind(T), w, Array{T,D})
end
@foldable @inline function _dual_type(
    ::NumericStorage, w::Val, ::Type{Array{T,D}}
) where {T,D}
    return Nfwd._ndual_array_V(Array{T,D}, w)
end
@foldable @inline function _dual_type(
    ::StructuralStorage, w::Val, ::Type{Array{T,D}}
) where {T,D}
    return Array{dual_type(w, T),D}
end
@foldable @inline function dual_type(w::Val, ::Type{Base.RefValue{P}}) where {P}
    return _dual_type(_storage_kind(P), w, Base.RefValue{P})
end
@foldable @inline function _dual_type(
    ::NumericStorage, ::Val{N}, ::Type{Base.RefValue{P}}
) where {N,P}
    return NDualRef{P,N}
end
@foldable @inline function dual_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    # A phantom free-TypeVar `Tuple` leaves `P` unbound.
    @isdefined(P) || return Any
    # First, so a non-concrete but non-differentiable tuple collapses as in reverse.
    tangent_type(P) === NoTangent && return NoDual
    # A union of tuples has no single element list; `fieldcount` reports one member's.
    Base.unwrap_unionall(P) isa Union && return Any
    # A `Vararg` tuple has no finite element list.
    Base.isvatuple(P) && return Any
    V = _dual_tuple_v(Val(N), P)
    # A non-concrete tuple can have wholly non-differentiable concretisations, which collapse to
    # `NoDual`, so its V must admit `NoDual` too, as `tangent_type` does. A concrete one cannot.
    isconcretetype(P) && return V
    return Tuple{Vararg{NoDual,fieldcount(P)}} <: V ? Union{V,NoDual} : V
end
# Element-wise tuple V without the collapse gate. Generated, as `tangent_type(::Type{<:Tuple})` is:
# head/tail recursion overflows inference on long tuples, which can SIGSEGV (julia#17109).
@foldable @generated function _dual_tuple_v(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    Vs = map(i -> :(dual_type(Val(N), fieldtype(P, $i))), 1:fieldcount(P))
    return :(Tuple{$(Vs...)})
end
@foldable @inline function dual_type(
    ::Val{N}, ::Type{NamedTuple{names,T}}
) where {N,names,T<:Tuple}
    # Non-differentiable NamedTuples collapse to NoDual, matching reverse and kwargs seeds.
    tangent_type(NamedTuple{names,T}) === NoTangent && return NoDual
    # As in `tangent_type`: `Any` unless `DT` is concrete, which an abstract field prevents.
    DT = dual_type(Val(N), T)
    return isconcretetype(DT) ? NamedTuple{names,DT} : Any
end
@foldable @inline dual_type(::Val{N}, ::Type{Ptr{T}}) where {N,T} = PointerDual{N}
# MemoryRef canonical V (Julia 1.11+); paired with NDualMemoryRef above.
# Memory itself is `<: AbstractArray{T, 1}` on 1.11+ — its canonical V is
# an NDualArray over `Memory{T}`: `Memory{T} → NDualArray{T, N, 1, Memory{T}}`.
@static if VERSION >= v"1.11-rc4"
    for C in (:Memory, :MemoryRef)
        @eval @foldable @inline function dual_type(w::Val, ::Type{$C{T}}) where {T}
            return _dual_type(_storage_kind(T), w, $C{T})
        end
        @eval @foldable @inline function _dual_type(
            ::StructuralStorage, w::Val, ::Type{$C{T}}
        ) where {T}
            return $C{dual_type(w, T)}
        end
    end
    @foldable @inline function _dual_type(
        ::NumericStorage, w::Val, ::Type{Memory{T}}
    ) where {T}
        return Nfwd._ndual_array_V(Memory{T}, w)
    end
    @foldable @inline function _dual_type(
        ::NumericStorage, ::Val{N}, ::Type{MemoryRef{T}}
    ) where {N,T}
        return NDualMemoryRef{T,N,Memory{T}}
    end
end

# Structural fallback: abstract P widens to Any; non-differentiable P gives NoDual.
# Fields recurse uniformly, with seed factories coercing into the declared backing NamedTuple.
@foldable @inline dual_type(w::Val, P::Type) = _dual_type(StructuralStorage(), w, P)
@foldable @generated function _dual_type(
    ::StructuralStorage, ::Val{N}, ::Type{P}
) where {N,P}
    # No union splitting, unlike `tangent_type`: `Lifted` is invariant in `P`, so a
    # union-distributed V would let `PiNode`s lower a valid branch to `unreachable`.
    isconcretetype(P) || return Any
    # Resolve extension-overloadable tangent_type in the returned expression at the call
    # world. Only the world-independent structural skeleton belongs in the generator.
    if fieldcount(P) == 0
        return :(tangent_type($P) === NoTangent ? NoDual : NTuple{$N,tangent_type($P)})
    end
    field_names = fieldnames(P)
    n_fields = fieldcount(P)
    inits = always_initialised(P)
    # Non-always-initialised fields are wrapped in `PossiblyUninitTangent`,
    # exactly as reverse-mode `tangent_type`, so the two stay coherent (e.g. a
    # lazily-built field like `LazyDerivedRule.rule`).
    field_dual_exprs = map(1:n_fields) do i
        base = :(dual_type(Val($N), $(fieldtype(P, i))))
        inits[i] ? base : :(PossiblyUninitTangent{$base})
    end
    inner_nt_type = :(NamedTuple{$field_names,Tuple{$(field_dual_exprs...)}})
    wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
    # Only a structural reverse tangent has a structural lift; a custom one would otherwise fail
    # late at the reverse bridge. Deferred, so a later extension overload takes effect.
    Tt = ismutabletype(P) ? :MutableTangent : :Tangent
    msg =
        "`$P` has a custom reverse tangent type but no `dual_type` method, and the structural " *
        "forward representation applies only where the reverse tangent is a `$Tt`. " *
        "Define `dual_type(::Val{N}, ::Type{$P})`."
    return :(
        if tangent_type($P) === NoTangent
            NoDual
        elseif tangent_type($P) <: $Tt
            $wrapper{$inner_nt_type}
        else
            error($msg)
        end
    )
end

"""
    lifted_type(::Val{N}, ::Type{P}) -> Type

Return the canonical `Lifted{P, N, V}` slot type for a primal of type `P`
at width `N`. For concrete `P`, equals `Lifted{P, N, dual_type(Val(N), P)}`.

Abstract `P` and the metatype kinds give a `UnionAll` slot, since `Lifted` is invariant in `P`.
"""
# Metatype kinds are concrete, but their runtime slots are `Lifted{Type{X}}`, which a bounded
# slot cannot hold.
@inline _is_metatype_kind(@nospecialize(T)) =
    T === DataType || T === UnionAll || T === Union || T === Core.TypeofBottom

# A widened `dual_type` is an upper bound, and `Lifted` is invariant in `V`, so the slot must
# leave `V` free rather than claim `Lifted{P,N,Any}`.
@inline function _concrete_lifted_type(::Val{N}, ::Type{P}) where {N,P}
    V = dual_type(Val(N), P)
    return isconcretetype(V) ? Lifted{P,N,V} : (Lifted{P,N,W} where {W})
end

@foldable @inline lifted_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
# An abstract tuple `P` widens to a UnionAll slot, since `Lifted` is invariant in `P`. `@foldable`
# keeps folding under coverage and exempts the `@isdefined(P)` guard from DispatchDoctor.
@foldable @inline function lifted_type(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    @isdefined(P) || return Lifted  # phantom free-TypeVar Tuple — broad `Lifted` slot, as the generic `lifted_type` guard below
    return if isconcretetype(P)
        _concrete_lifted_type(Val(N), P)
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# True when every member of `U` is non-differentiable (`tangent_type === NoTangent`), so its
# `lifted_type` is a concrete `NoDual`-V `Lifted` that union-splits safely (see `lifted_type` below).
function _all_nodual_union_members(@nospecialize(U))
    if U isa Union
        _all_nodual_union_members(U.a) && _all_nodual_union_members(U.b)
    else
        (U === Union{} || tangent_type(U) === NoTangent)
    end
end

# Abstract P and metatypes need a UnionAll slot: Lifted is invariant in P.
# @foldable preserves folding under coverage and exempts the phantom-P guard from
# DispatchDoctor's unconditional static-parameter reads, as in the Tuple overload.
@foldable @inline function lifted_type(::Val{N}, ::Type{P}) where {N,P}
    # A phantom `TypeVar` leaves `P` unbound.
    @isdefined(P) || return Lifted
    # A union of non-differentiable members splits into concrete slots, which do not box; any
    # other union keeps the UnionAll, as a split tripped an OpaqueClosure return assertion.
    P isa Union &&
        _all_nodual_union_members(P) &&
        return Union{lifted_type(Val(N), P.a),lifted_type(Val(N), P.b)}
    return if isconcretetype(P) && !_is_metatype_kind(P)
        _concrete_lifted_type(Val(N), P)
    elseif P <: Type
        # A bounded slot trips an OpaqueClosure return assertion on the sharpened `Lifted{Type{X}}`.
        Lifted{T,N,V} where {T,V}
    else
        (Lifted{T,N,V} where {T<:P,V})
    end
end
# `Type{X}` has one inhabitant, so its concrete slot is exact; the generic method would widen it
# (`isconcretetype(Type{X})` is false) and box it.
@foldable @inline function lifted_type(::Val{N}, ::Type{Type{X}}) where {N,X}
    return Lifted{Type{X},N,dual_type(Val(N), Type{X})}
end
