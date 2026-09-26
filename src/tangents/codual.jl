struct CoDual{Tx,Tdx}
    x::Tx
    dx::Tdx
end

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function CoDual(x::Type{P}, dx::NoFData) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoFData}(P, dx)
end

function CoDual(x::Type{P}, dx::NoTangent) where {P}
    return CoDual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(P, dx)
end

primal(x::CoDual) = x.x
tangent(x::CoDual) = x.dx
Base.copy(x::CoDual) = CoDual(copy(primal(x)), copy(tangent(x)))
# CoDual can be safely shared without copying
_copy(x::P) where {P<:CoDual} = x

"""
    extract(x::CoDual)

Helper function. Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::CoDual) = primal(x), tangent(x)

"""
    zero_codual(x)

Equivalent to `CoDual(x, zero_tangent(x))`.

For `Ptr` types, constructing a true zero tangent would require allocating new derivative
storage and returning a pointer to it, which has unclear ownership and lifetime. Instead,
`zero_codual(x::Ptr{P})` falls back to `uninit_codual(x)`, which uses the bitcast
convention: the tangent pointer is produced by reinterpreting the primal address as a
`Ptr{tangent_type(P)}`. The result must not be dereferenced as valid derivative storage —
it is a type-correct structural placeholder. See the comment on `uninit_tangent(x::Ptr)`
in `tangents.jl` for the full explanation of the Ptr tangent convention.
"""
zero_codual(x) = CoDual(x, zero_tangent(x))
zero_codual(x::Ptr{P}) where {P} = uninit_codual(x)

"""
    uninit_codual(x)

Equivalent to `CoDual(x, uninit_tangent(x))`.
"""
uninit_codual(x) = CoDual(x, uninit_tangent(x))

function _codual_internal(::Type{P}, f::F, extractor::E) where {P,F,E}
    P == Union{} && return Union{}
    P == DataType && return CoDual
    P isa Union && return Union{f(P.a),f(P.b)}
    # Use `isa` not `<:`: generators like `NTuple{N,Int} where N` are instances of
    # UnionAll but not subtypes of it (`NTuple{N,Int} where N <: UnionAll` is false).
    # `P == UnionAll` handles the UnionAll metatype itself (`UnionAll isa UnionAll` is false).
    (P isa UnionAll || P == UnionAll) && return CoDual # P is abstract, tangent type unknown.

    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{f(P_split.a),f(P_split.b)}
        end
    end

    return isconcretetype(P) ? CoDual{P,extractor(P)} : CoDual
end

"""
    codual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and associated tangents.
"""
@unstable function codual_type(::Type{P}) where {P}
    # The static parameter is unbound for e.g. `UnionAll(A, AbstractArray{T, A})`, whose
    # body has a free `TypeVar` `T`; touching `P` would then throw
    # `UndefVarError(:P, :static_parameter)`. The overloads below and `dual_type` need it too.
    @isdefined(P) || return CoDual
    return _codual_internal(P, codual_type, tangent_type)
end

@unstable function codual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoTangent} : CoDual{_typeof(p),NoTangent}
end

"""
    fcodual_type(P::Type)

The type of the `CoDual` which contains instances of `P` and its fdata.
"""
@unstable function fcodual_type(::Type{P}) where {P}
    @isdefined(P) || return CoDual
    return _codual_internal(P, fcodual_type, P -> fdata_type(tangent_type(P)))
end

@unstable function fcodual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? CoDual{Type{P},NoFData} : CoDual{_typeof(p),NoFData}
end

to_fwds(x::CoDual) = CoDual(primal(x), fdata(tangent(x)))

to_fwds(x::CoDual{Type{P}}) where {P} = CoDual{Type{P},NoFData}(primal(x), NoFData())

"""
    zero_fcodual(x)

Equivalent to `CoDual(x, fdata(zero_tangent(x)))`.

For `Ptr` types, falls back to `uninit_fcodual(x)` for the same reason `zero_codual`
does: constructing a true zero tangent requires allocating derivative storage, which has
unclear ownership. Since `fdata_type(Ptr{P}) == Ptr{tangent_type(P)}` (the full tangent
is fdata for Ptr), the fdata is produced via bitcast - same address, reinterpreted as
`Ptr{tangent_type(P)}`. Not safe to dereference as valid derivatives. See the comment
on `uninit_tangent(x::Ptr)` in `tangents.jl` for the full explanation.
"""
zero_fcodual(p) = to_fwds(zero_codual(p))
zero_fcodual(p::Ptr{P}) where {P} = uninit_fcodual(p)

"""
    uninit_fcodual(x)

Like `zero_fcodual`, but doesn't guarantee that the value of the fdata is initialised.
See implementation for details, as this function is subject to change.
"""
@inline uninit_fcodual(x::P) where {P} = CoDual(x, uninit_fdata(x))

struct NoPullback{R<:Tuple}
    r::R
end

# Recursively copy the contained reverse data
_copy(x::P) where {P<:NoPullback} = P(_copy(x.r))

"""
    NoPullback(args::CoDual...)

Construct a `NoPullback` from the arguments passed to an `rrule!!`. For each argument,
extracts the primal value, and constructs a `LazyZeroRData`. These are stored in a
`NoPullback` which, in the reverse-pass of AD, instantiates these `LazyZeroRData`s and
returns them in order to perform the reverse-pass of AD.

The advantage of this approach is that if it is possible to construct the zero rdata element
for each of the arguments lazily, the `NoPullback` generated will be a singleton type. This
means that AD can avoid generating a stack to store this pullback, which can result in
significant performance improvements.
"""
function NoPullback(args::Vararg{CoDual,N}) where {N}
    return NoPullback(tuple_map(lazy_zero_rdata ∘ primal, args))
end

@inline (pb::NoPullback)(_) = tuple_map(instantiate, pb.r)

"""
    ConstAliasSet(primals::Vector{Any} = Any[])

Constants and globals with derivative storage created at rule-build time. Rules refuse
arguments identical to these objects because their derivative storage is separate.

Use a concrete, non-differentiable field of this type: parameterising rules by constants
breaks the `Core.Compiler.return_type(build_derived_rrule, ...)` inference used by
`__build_primitive_frule` to key its cache.
"""
struct ConstAliasSet
    primals::Vector{Any}
end
ConstAliasSet() = ConstAliasSet(Any[])

tangent_type(::Type{ConstAliasSet}) = NoTangent

"""
    _check_constant_aliasing(consts::ConstAliasSet, args)

Refuse arguments identical to a rule's constants. Most rules have an empty set.
Matching only checks root identity; nesting on either side can hide aliasing and produce
wrong derivatives. See [`record_const_alias!`](@ref) for this boundary's rationale.
"""
@inline function _check_constant_aliasing(consts::ConstAliasSet, args)
    isempty(consts.primals) && return nothing
    return _check_constant_aliasing_slow(consts.primals, args)
end

# Recurse over heterogeneous `args` to avoid boxing `primal(a)` during iteration.
# A closure over `c::Any` would heap-allocate even if `any` unrolled the tuple.
@noinline function _check_constant_aliasing_slow(consts::Vector{Any}, args)
    for c in consts
        _aliases_any_arg(c, args) && _throw_constant_alias_error(c)
    end
    return nothing
end

@inline _aliases_any_arg(@nospecialize(c), ::Tuple{}) = false
@inline function _aliases_any_arg(@nospecialize(c), args::Tuple)
    return _alias_target(c) === primal(first(args)) || _aliases_any_arg(c, Base.tail(args))
end

"""
    GlobalBinding(mod, name)

A non-const global recorded in a [`ConstAliasSet`](@ref) by binding. Resolve its value at
call time so rebinding cannot bypass the alias guard. Const globals are recorded directly.
"""
struct GlobalBinding
    mod::Module
    name::Symbol
end

# The binding's call-time value can have any type; `@unstable` exempts this check
# from DispatchDoctor without narrowing which objects it can detect.
@unstable @inline _alias_target(@nospecialize(c)) = c
@unstable @inline function _alias_target(b::GlobalBinding)
    return isdefined(b.mod, b.name) ? getglobal(b.mod, b.name) : nothing
end

@noinline function _throw_constant_alias_error(@nospecialize(c))
    c = _alias_target(c)
    throw(
        ArgumentError(
            "An argument is the same object as a constant or global read inside the function " *
            "being differentiated (a $(typeof(c))). Their derivative storage is separate — the " *
            "constant's is created once when the rule is built — so the contribution through " *
            "the constant would be silently dropped and the derivative returned would be wrong. " *
            "Pass a copy of the argument, or read the value through an argument instead of a " *
            "global.",
        ),
    )
end

# `tangent_type` needs custom methods for recursive types. Check reachability:
# `IOContext` contains a self-referential `Base.ImmutableDict{Symbol,Any}`.
# Exceeding the node budget also reports true, so callers record without asking.
function _reaches_recursive_type(
    @nospecialize(P::Type),
    on_path::IdDict{Type,Bool}=IdDict{Type,Bool}(),
    nodes::Ref{Int}=Ref(0),
)
    # Follow unions like `next::Union{Nothing,LNode}`, as `tangent_type` does.
    if P isa Union
        return _reaches_recursive_type(P.a, on_path, nodes) ||
               _reaches_recursive_type(P.b, on_path, nodes)
    end
    (isconcretetype(P) && !isprimitivetype(P)) || return false
    haskey(on_path, P) && return on_path[P]
    (nodes[] += 1) > 600 && return true
    on_path[P] = true
    for F in fieldtypes(P)
        _reaches_recursive_type(F, on_path, nodes) && return true
    end
    on_path[P] = false
    return false
end

"""
    record_const_alias!(consts::Vector{Any}, @nospecialize(v))

Record `v` when an identical argument could clash with its mutable derivative storage.
Values with `NoFData` are skipped, avoiding spurious matches on equal scalars but also
missing differentiable immutable constants such as `const C = ("a", 1.0)` in both modes.
Unknown storage is recorded: an unnecessary refusal is preferable to a wrong derivative.

Only root identity is checked, so nesting on either side can hide aliasing. Walking each
constant's reachable objects reaches modules and method tables: recursive traversal
stack-overflows, while iterative traversal pushed precompilation from 40s past 600s.
Bounding that walk would silently under-collect; detecting nested aliases requires a way
to enumerate derivative storage without traversing arbitrary objects.
"""
function record_const_alias!(consts::Vector{Any}, @nospecialize(v))
    record_const_alias!(consts, v, v)
end

# `stored` is what lands in the set: the value itself, or a `GlobalBinding` to resolve at call
# time. `v` is always the value, since only it can answer whether there is storage to clash over.
function record_const_alias!(consts::Vector{Any}, @nospecialize(v), @nospecialize(stored))
    # Isbits values have no shared fdata (`Ptr` constants arrive as IR literals).
    # Check first to avoid asking `tangent_type` about unsupported primitives.
    isbits(v) && return nothing
    # Record recursive types and exhausted walks without asking `tangent_type`.
    if !_reaches_recursive_type(_typeof(v))
        fdata_type(tangent_type(_typeof(v))) === NoFData && return nothing
    end
    any(c -> c === stored, consts) || push!(consts, stored)
    return nothing
end
