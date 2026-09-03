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

The constant and global primals a rule built derivative storage for at rule-build time.
`DerivedRule`, `DerivedFRule` and `NfwdFRule` each carry one, and refuse a call whose arguments
include one of these objects: that storage is shared with nothing, so the contribution through the
constant would be dropped.

A field of this fixed, concrete, non-differentiable type rather than a type parameter. A parameter
would make a rule's type depend on whether its function happens to read a differentiable constant,
which defeats the `Core.Compiler.return_type(build_derived_rrule, ...)` inference that
`__build_primitive_frule` relies on to key its cache.
"""
struct ConstAliasSet
    primals::Vector{Any}
end
ConstAliasSet() = ConstAliasSet(Any[])

tangent_type(::Type{ConstAliasSet}) = NoTangent

"""
    _check_constant_aliasing(consts::ConstAliasSet, args)

Refuse `args` if any of them is one of the rule's build-time constants. Every rule kind calls this
at entry; the set is empty for most rules, which is the branch that matters for cost.
"""
@inline function _check_constant_aliasing(consts::ConstAliasSet, args)
    isempty(consts.primals) && return nothing
    return _check_constant_aliasing_slow(consts.primals, args)
end

@noinline function _check_constant_aliasing_slow(consts::Vector{Any}, args)
    for a in args, c in consts
        c === primal(a) && _throw_constant_alias_error(c)
    end
    return nothing
end

@noinline function _throw_constant_alias_error(@nospecialize(c))
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

"""
    record_const_alias!(consts::Vector{Any}, @nospecialize(v))

Add the constant primal `v` to `consts` if an argument that is the same object would clash with it.

Only a value owning mutable derivative storage can clash, and those are exactly the ones whose
fdata is not `NoFData`. A scalar, or an immutable aggregate of scalars, has none: there is nothing
for an argument to share, and excluding it is also what stops `===` matching a constant `2.0`
against an argument that happens to be `2.0`. The cost is that a differentiable immutable constant
passed as an argument — `const C = ("a", 1.0)` — is not caught, in either mode.
"""
function record_const_alias!(consts::Vector{Any}, @nospecialize(v))
    # An isbits value has no fdata to share (`Ptr` aside, and a `Ptr` constant is embedded as an IR
    # literal rather than reaching here). Testing that first also keeps `tangent_type` off the
    # primitive types it deliberately refuses, which the constants of an arbitrary walked body
    # include.
    isbits(v) && return nothing
    fdata_type(tangent_type(_typeof(v))) === NoFData && return nothing
    any(c -> c === v, consts) || push!(consts, v)
    return nothing
end
