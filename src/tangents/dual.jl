# ── NTangent: width-aware tangent container ───────────────────────────────────

"""
    NTangent(lanes::Tuple)

Width-aware tangent container for forward-mode AD. Holds one tangent per basis
direction (lane). Each element of `lanes` must be a valid width-1 tangent for the
corresponding primal type.

Width 1 is ordinary forward mode; width N > 1 is chunked forward mode. The canonical
public forward tangent at any width is always an `NTangent`, never a bare tangent type.
"""
struct NTangent{L<:Tuple}
    lanes::L
end

Base.length(x::NTangent) = length(x.lanes)
Base.getindex(x::NTangent, i::Int) = x.lanes[i]
Base.iterate(x::NTangent, st...) = iterate(x.lanes, st...)

# Per-lane copy so callers (rules that copy a `Dual{P, NTangent}` value
# whole, like `Base.copy(::Memory{<:Struct})` via the
# `:jl_genericmemory_copy` foreigncall path) get independent lane
# tangents rather than aliased references to the original.
Base.copy(t::NTangent) = NTangent(map(copy, t.lanes))

# Singleton-NTangent unwrap: many bare-Dual rules sit on top of inputs
# whose tangent has been canonicalised to `NTangent{Tuple{T}}` (width-1
# canonical form), but the rule body operates on the bare `T`. This
# helper extracts the single lane; for non-NTangent tangents (already
# bare), it is the identity.
@inline _ntangent_unwrap_singleton(t::NTangent{Tuple{T}}) where {T} = t.lanes[1]
@inline _ntangent_unwrap_singleton(t) = t

# ── Width-aware tangent_type ──────────────────────────────────────────────────

"""
    tangent_type(::Val{N}, ::Type{P})

Width-aware tangent type query. Returns the canonical tangent representation for
primal type `P` at width `N`.

- `Val(0)` → `NoTangent` (primal passthrough, no tangent needed)
- `Val(N)` where `tangent_type(P) == NoTangent` → `NoTangent`
- `Val(N)` otherwise → `NTangent{NTuple{N, tangent_type(P)}}`
"""
# `Union{}` is the bottom type and is `<:` every primal; specialise both the
# tangent and dual queries so they win against the IEEEFloat / Complex / etc.
# overloads (which would otherwise produce nonsensical
# `NDual{Union{}, N}` etc.).
tangent_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
dual_type(::Val{N}, ::Type{Union{}}) where {N} = Union{}
tangent_type(::Val{0}, ::Type{P}) where {P} = NoTangent
function tangent_type(::Val{N}, ::Type{P}) where {N,P}
    T = tangent_type(P)
    T === NoTangent && return NoTangent
    # Width 1 wraps once (`NTangent{Tuple{T}}`). `dual_type(Val(1),
    # generic_P)` also returns the NTangent-wrapped form `Dual{P,
    # NTangent{Tuple{T}}}` for generic concrete `P`, so `tangent_type` and
    # `dual_type` agree on shape at every positive width.
    return NTangent{NTuple{N,T}}
end

# ── Width-aware dual_type ─────────────────────────────────────────────────────

"""
    dual_type(::Val{N}, ::Type{P})

Width-aware forward value type query.

- `Val(0)` → `P` (primal passthrough)
- `Val(N)`, concrete `P` → one of the canonical-V shapes below, or the
  `Dual{P, tangent_type(Val(N), P)}` fallback for primals outside the
  canonical coverage.
- abstract/union `P` → `Dual` (bare, for compiler flexibility)

## Canonical-V coverage

Each category below has a fused canonical V where primal and tangent
share storage (the runtime invariant the wrapper-exception removal
plan targets).

NDual leaf (scalar interleaving)
  IEEEFloat                  → NDual{T, N}
  Complex{<:IEEEFloat}       → Complex{NDual{T, N}}

Array of NDual (top-level only)
  Array{<:IEEEFloat, D}      → Array{NDual{T, N}, D}
  Array{<:Complex{T}, D}     → Array{Complex{NDual{T, N}}, D}
  Memory{...}, MemoryRef{...}→ same template, NDual element

NDual-element wrapper view (Phases 1+2)
  Wrapper{<:IEEEFloat, P_parent}
                             → Wrapper{NDual{T, N}, V_parent}
  Wrapper ∈ {Transpose, Adjoint, SubArray, Diagonal,
             Symmetric, Hermitian, *Triangular, UpperHessenberg,
             ReshapedArray, ReinterpretArray}
  (parent V from recursive `dual_type`; wrapper's structural shape is
   preserved, only its element type is lifted to NDual)

Structural Tuple/NamedTuple (element-wise recursive)
  Tuple{T1, T2, ...}         → Tuple{V_1, V_2, ...}
  NamedTuple{names, T}       → NamedTuple{names, T'}

Structural lift for concrete immutable struct
  (default `Tangent`, all fields always-initialised, lift-safe types)
                             → NamedTuple{fieldnames, Tuple{V_i, ...}}

StepRangeLen
  StepRangeLen{T<:IEEEFloat, TWP{T}, TWP{T}, Int}
                             → structural lift

Nested-Array canonical
  Vector{Vector{<:IEEEFloat}} → Array{Array{NDual{T, N}, K}, D}
  (and Matrix-of-Vector, Complex variants)

SplitDual for mutable struct
  Mutable struct with at least one canonical-NDual-eligible field
  (top-level Array-of-IEEEFloat, nested-Array, or `PossiblyUninit`
   field whose V's primal type is concrete)
                             → SplitDual{NamedTuple{...}}

## Residual `Dual` at runtime

Primals outside the canonical coverage still produce the
`Dual{P, tangent_type(Val(N), P)}` fallback. Primary categories:

  - Custom-`tangent_type` primals       e.g. `TwicePrecision{Float64}`
  - Heterogeneous containers            `Vector{Any}`, `SimpleVector`
  - `NoTangent`-element arrays          `Vector{Int}`, `Memory{Int}`
  - Mutable structs with `Any` fields   `MutableFoo`, `TypeUnstableMutableStruct`
  - Mutable structs, scalar-only        `TypeStableMutableStruct{Float64}`
"""
# `@unstable`: return type depends on the type-domain shape of `P` (Union
# splitting, Tuple field concreteness). Callers force-specialise via
# `Val(N)` constants or accept a bare `Dual`.
@unstable function dual_type(::Val{N}, ::Type{P}) where {N,P}
    P == Union{} && return Union{}
    # `DataType` primals carry `tangent_type === NoTangent` (types are
    # non-differentiable), so the canonical inner V is the concrete
    # `Dual{DataType, NoTangent}`. Returning the abstract `Dual` here
    # produced `Lifted{P, N, Dual}` slots whose subsequent `frule!!` calls
    # failed to dispatch (the wrapper-exception rule requires `V <: Dual{P,T}`).
    P == DataType && return Dual{DataType,NoTangent}
    P isa Union && return Union{dual_type(Val(N), P.a),dual_type(Val(N), P.b)}
    (P isa UnionAll || P == UnionAll) && return Dual

    if P <: Tuple && !all(isconcretetype, (P.parameters...,))
        field_types = (P.parameters...,)
        union_fields = _findall(Base.Fix2(isa, Union), field_types)
        if length(union_fields) == 1 &&
            all(p -> p isa Union || isconcretetype(p), field_types)
            P_split = split_union_tuple_type(field_types)
            return Union{dual_type(Val(N), P_split.a),dual_type(Val(N), P_split.b)}
        end
    end

    # Statically-fielded Tuple: element-wise lifting — each field type is
    # individually lifted. `Base.datatype_fieldcount(P)` returns a definite
    # field count for any Tuple whose field types are statically resolvable
    # (covers `Tuple{Float64, Int}`, `Tuple{Type{Float64}, Type{Int}}`, and
    # `Tuple{Val{1}, DataType, Float64, Vector{Float64}}` — none of which
    # are `Base.isdispatchtuple` but all have well-defined `fieldtype(P, i)`
    # for the @generated `_dual_type_tuple_inner` to recurse on). Returns
    # `nothing` for Vararg/abstract tuples (`Tuple{Vararg{Float64}}`,
    # `Tuple`); those fall through to the abstract `Dual` fallback. The
    # per-field `dual_type` calls unroll at @generated expansion time so
    # Julia can statically infer the result.
    if P <: Tuple && Base.datatype_fieldcount(P) !== nothing
        return _dual_type_tuple_inner(Val(N), P)
    end

    # Concrete NamedTuple: element-wise lifting symmetric to Tuple. Same
    # generated-helper unrolling as the Tuple branch for the same reason.
    if isconcretetype(P) && P <: NamedTuple
        return _dual_type_named_tuple_inner(Val(N), P)
    end

    # Concrete IMMUTABLE struct with `tangent_type(P) <: Tangent`, all fields
    # always initialised, and only "lift-safe" field types: recursive
    # NamedTuple lift. Each field's `dual_type` is the canonical V for that
    # field's primal type; the inner V mirrors the struct's field structure
    # as a `NamedTuple{names, Tuple{Vᵢ…}}`. This generalises the per-wrapper
    # structural lift (Diagonal/Adjoint/SubArray) to arbitrary immutable
    # structs and closes the silent-corruption gap for in-place mutation
    # through struct fields. See `notes/mooncake/dual-types.md` §13.
    #
    # Excluded:
    # - Mutable structs (`MutableTangent` tangent_type): their `lsetfield!`
    #   rules need a mutable inner V to support `s.field = x`, but
    #   `NamedTuple` is immutable. Keep the existing parallel
    #   `Dual{P, MutableTangent}` form.
    # - Structs with potentially-undef fields (`PossiblyUninitTangent` in
    #   their `tangent_field_types`): the lift would call
    #   `getfield(primal, name)` on undef fields. Keep the legacy form.
    # - Structs with nested-struct fields (e.g. `Broadcasted`'s `args` is a
    #   `Tuple` containing an `Extruded` struct, which would itself recurse
    #   through this lift, losing its struct identity inside the Tuple V).
    #   `_uses_structural_dual_type` walks Tuple/NamedTuple containers to
    #   detect any non-flat struct lurking inside. Keep the legacy form.
    #
    # Specific per-wrapper `dual_type` overloads (e.g. `Diagonal{T,Vector{T}}`
    # in `nfwd/NfwdMooncake.jl`) are more specific and dispatch first, so
    # this branch only fires for immutable structs without an explicit lift.
    if N >= 1 && _uses_structural_dual_type(P)
        return _dual_type_structural_struct(Val(N), P)
    end

    # Mutable struct with at least one Array-of-IEEEFloat field: use SplitDual
    # so each field's V is the canonical NDual-element form and `dual_type`
    # recurses coherently. Falls through to the `Dual` fallback below
    # for mutable structs without canonical-V Array fields.
    if N >= 1 && _split_dual_eligible(P)
        return _dual_type_split_dual(Val(N), P)
    end

    return isconcretetype(P) ? Dual{P,tangent_type(Val(N), P)} : Dual
end

@inline function _split_dual_eligible(::Type{P}) where {P}
    return isconcretetype(P) &&
           ismutabletype(P) &&
           fieldcount(P) > 0 &&
           tangent_type(P) <: MutableTangent &&
           _all_split_dual_field_types_concrete(P) &&
           _has_split_dual_field(P)
end
# Per-field concreteness check: SplitDual's per-field canonical V slot needs
# `dual_type(Val(N), field_type)` to be concrete so the SplitDual generator
# can produce a typed NamedTuple. Non-concrete (e.g. Any-typed) field types
# would yield abstract V slots that the inner ctors can't dispatch on.
# Initialised fields need a concrete field type directly; PossiblyUninit
# fields wrap the V in `PossiblyUninitTangent{Vi}`, which still needs `Vi`
# concrete for the wrapper to be useful. `_all_split_dual_field_types_concrete`
# replaces the old `all(always_initialised(P))` guard — PossiblyUninit
# fields are now supported via the `PossiblyUninitTangent{Vi}` wrapper
# pattern (mirroring `MutableTangent`'s field-wise convention) as long as
# each field's primal type is concrete.
@generated function _all_split_dual_field_types_concrete(::Type{P}) where {P}
    for i in 1:fieldcount(P)
        ft = fieldtype(P, i)
        isconcretetype(ft) || return :(false)
    end
    return :(true)
end
# Predicate per field type — true if the field's primal canonicalises to an
# NDual-element form anywhere (top-level Array/Memory of IEEEFloat /
# Complex{IEEEFloat}, or recursively through a nested mutable struct).
# Trait-style dispatch + recursion so eligibility broadens beyond top-level
# `Array{<:IEEEFloat}` to `Memory{<:IEEEFloat}` / `Memory{<:Complex{<:IEEEFloat}}`
# (Julia 1.11+) and nested mutable structs containing such fields.
#
# Policy: SplitDual eligibility is intentionally closed over Base/Core
# container shapes whose canonical NDual-element `dual_type` is built in.
# Extensions that introduce their own `dual_type` overloads do not yet
# trigger SplitDual via this predicate — they would need an extension-
# visible trait. If/when an extension needs SplitDual eligibility for a
# new container shape, prefer adding a trait method
# (`_field_canonical_ndual_eligible(::Type{<:MyContainer{<:IEEEFloat}}) =
# true`) rather than widening the recursion in the generator body, so
# precompile expansion stays bounded.
@inline _field_canonical_ndual_eligible(::Type{T}) where {T} = false
@inline _field_canonical_ndual_eligible(::Type{<:DenseArray{<:IEEEFloat}}) = true
@inline _field_canonical_ndual_eligible(::Type{<:DenseArray{<:Complex{<:IEEEFloat}}}) = true
# Nested Array canonical NDual: `DenseArray{<:DenseArray{<:IEEEFloat}}`
# now has a canonical form, so a mutable struct field of that shape can also
# participate in SplitDual.
@inline _field_canonical_ndual_eligible(::Type{<:DenseArray{<:DenseArray{<:IEEEFloat}}}) =
    true
@inline _field_canonical_ndual_eligible(
    ::Type{<:DenseArray{<:DenseArray{<:Complex{<:IEEEFloat}}}}
) = true
@static if VERSION >= v"1.11-"
    @inline _field_canonical_ndual_eligible(::Type{<:Memory{<:IEEEFloat}}) = true
    @inline _field_canonical_ndual_eligible(::Type{<:Memory{<:Complex{<:IEEEFloat}}}) = true
end
@generated function _has_split_dual_field(::Type{P}) where {P}
    n = fieldcount(P)
    for i in 1:n
        ft = fieldtype(P, i)
        # Direct Array/Memory of IEEEFloat element.
        if _field_canonical_ndual_eligible(ft)
            return :(true)
        end
        # Nested struct (mutable or immutable) with its own canonical-NDual
        # field; recurse so SplitDual covers structs whose differentiable
        # leaves live inside immutable wrappers.
        if isconcretetype(ft) && fieldcount(ft) > 0 && ft !== P
            try
                _has_split_dual_field(ft) && return :(true)
            catch e
                # Recursive generator expansion can fail mid-precompile when
                # `fieldtype` / `fieldcount` hit world-age boundaries on
                # extension-loaded types. Treat as "not eligible" rather
                # than throw so the outer `dual_type` falls back to the
                # default Dual representation. Re-throw on `InterruptException`
                # so Ctrl-C still works; otherwise swallow and continue.
                e isa InterruptException && rethrow()
            end
        end
    end
    return :(false)
end
# `_dual_type_split_dual` defers the per-field `dual_type` calls to the
# returned expression's runtime so any extension-defined `dual_type`
# overloads loaded after this generator first expands are still honored
# (per AGENTS.md world-age guidance).
@generated function _dual_type_split_dual(::Val{N}, ::Type{P}) where {N,P}
    names = fieldnames(P)
    n = fieldcount(P)
    inits = always_initialised(P)
    # PossiblyUninit fields wrap their canonical V in `PossiblyUninitTangent{Vi}`
    # so a field that may or may not be assigned in the primal has the same
    # may-or-may-not-have-tangent semantics in the SplitDual NamedTuple.
    # Mirrors the `MutableTangent`'s per-field convention.
    field_dual_type_exprs = map(1:n) do i
        Vi_expr = :(dual_type(Val($N), $(fieldtype(P, i))))
        inits[i] ? Vi_expr : :(PossiblyUninitTangent{$Vi_expr})
    end
    return quote
        SplitDual{NamedTuple{$names,Tuple{$(field_dual_type_exprs...)}}}
    end
end

# Generated helpers — emit per-field `dual_type` calls in the returned
# expression so they evaluate at runtime. Calling `dual_type` directly
# from the generator body would freeze the field-wise types at the
# first expansion's world age, which excludes any extension-loaded
# `dual_type` overloads added later (per AGENTS.md world-age guidance).
@generated function _dual_type_tuple_inner(::Val{N}, ::Type{P}) where {N,P<:Tuple}
    field_dual_type_exprs = [
        :(dual_type(Val($N), $(fieldtype(P, i)))) for i in 1:fieldcount(P)
    ]
    return :(Tuple{$(field_dual_type_exprs...)})
end
@generated function _dual_type_named_tuple_inner(
    ::Val{N}, ::Type{P}
) where {N,P<:NamedTuple}
    names = fieldnames(P)
    field_dual_type_exprs = [
        :(dual_type(Val($N), $(fieldtype(P, i)))) for i in 1:fieldcount(P)
    ]
    return :(NamedTuple{$names,Tuple{$(field_dual_type_exprs...)}})
end
@generated function _dual_type_structural_struct(::Val{N}, ::Type{P}) where {N,P}
    names = fieldnames(P)
    field_dual_type_exprs = [
        :(dual_type(Val($N), $(fieldtype(P, i)))) for i in 1:fieldcount(P)
    ]
    return :(NamedTuple{$names,Tuple{$(field_dual_type_exprs...)}})
end

@inline function _uses_structural_dual_type(::Type{P}) where {P}
    # Structural `dual_type` mirrors structural `tangent_type`. There is no
    # broad ownership/package gate — any type whose `tangent_type` is the
    # default structural `Tangent{NamedTuple{names, ...}}` shape with all
    # fields always initialised gets the recursive `NamedTuple` lift.
    # Specific Base/Core/LinearAlgebra wrappers that need a different
    # representation (e.g. `Diagonal`, `Adjoint`, `SubArray`, `Broadcasted`)
    # register explicit `dual_type` overloads in `src/nfwd/NfwdMooncake.jl`
    # (and per-extension files), which dispatch first by Julia's method
    # specificity. Failing types should fail loudly (via missing overloads
    # or explicit local failures), not via a silent ownership check.
    return isconcretetype(P) &&
           !ismutabletype(P) &&
           fieldcount(P) > 0 &&
           tangent_type(P) <: Tangent &&
           _uses_structural_tangent_type(P) &&
           all(always_initialised(P))
end

@inline function _uses_structural_tangent_type(::Type{P}) where {P}
    # Generated callers use this predicate, so avoid reflection (`which` is
    # forbidden there). Match the default struct tangent by shape instead.
    names = fieldnames(P)
    field_tangent_types = Tuple{
        ntuple(i -> tangent_type(fieldtype(P, i)), Val(fieldcount(P)))...
    }
    return tangent_type(P) === Tangent{NamedTuple{names,field_tangent_types}}
end

dual_type(::Val{0}, ::Type{P}) where {P} = P

dual_type(::Val{0}, ::Type{Type{P}}) where {P} = Type{P}

function dual_type(::Val{N}, p::Type{Type{P}}) where {N,P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

# ── Dual ──────────────────────────────────────────────────────────────────────

"""
    Dual(primal::P, tangent::T)

Used to pair together a `primal` value and a `tangent` to it. In the context of foward mode
AD (aka computing Frechet derivatives), `primal` governs the point at which the derivative
is computed, and `tangent` the direction in which it is computed.

Must satisfy `tangent_type(P) == T`.
"""
struct Dual{P,T}
    primal::P
    tangent::T
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent

# `SplitDual{V}` — coherent inner V for mutable struct primals with canonical
# NDual-element Array fields. Wraps a `NamedTuple` whose values are the
# canonical V for each field (`dual_type(Val(N), field_type_i)`), so the
# `dual_type` recursion invariant holds field-wise. Primal values live
# inside the NDual elements' `.value` slots — no separate primal-half
# storage, avoiding the duplication of the `Dual{Struct, MutableTangent}`
# form.
#
# `V` is constrained to `NamedTuple` so the field-by-field recursion is
# structurally typed. `SplitDual` itself is mutable so that
# whole-field replacement (`s.field = new_value`) on the primal lifts
# to `setfield!(d, :canonical, new_nt)` on the SplitDual, preserving
# the primal's aliasing semantics: aliased references to the same
# mutable struct continue to see in-place mutations.
mutable struct SplitDual{V<:NamedTuple}
    canonical::V
end

# Mooncake-protocol accessors. `primal` reconstructs the original primal
# struct's field values from each field's V (NDual `.value` slots for
# canonical-NDual Array fields; `primal(field_V)` recursively for
# non-Array fields). `tangent` builds the matching `MutableTangent` from
# each field's tangent. Per-lane `tangent(_, i)` projects lane i.
@inline _unlift(d::SplitDual) = d.canonical
# 2-arg ctor: build the canonical NamedTuple from a (primal, MutableTangent)
# pair, field by field. Each field's V is constructed via its own 2-arg
# constructor (`Vector{NDual{T,1}}(primal_field, tangent_field)` etc.).
# `@generated` so the field-wise calls unroll statically.
@generated function (::Type{SplitDual{NamedTuple{names,Vs}}})(
    primal::P, tangent::MutableTangent
) where {names,Vs<:Tuple,P}
    n = fieldcount(P)
    field_exprs = map(1:n) do i
        Vi = fieldtype(Vs, i)
        fname = names[i]
        if Vi <: PossiblyUninitTangent
            # PossiblyUninit field: check `isdefined(primal, fname)`. If
            # assigned, unwrap the tangent's `PossiblyUninitTangent` slot
            # (via `val`) and build the inner V; otherwise produce an empty
            # `PossiblyUninitTangent{Vi_inner}()`.
            Vi_inner = fieldtype(Vi, :tangent)
            quote
                if isdefined(primal, $(QuoteNode(fname)))
                    inner = _inner_dual_for_field(
                        $Vi_inner,
                        getfield(primal, $(QuoteNode(fname))),
                        val(getfield(tangent.fields, $(QuoteNode(fname)))),
                    )
                    $Vi(inner)
                else
                    $Vi()
                end
            end
        else
            :(_inner_dual_for_field(
                $Vi,
                getfield(primal, $(QuoteNode(fname))),
                getfield(tangent.fields, $(QuoteNode(fname))),
            ))
        end
    end
    return quote
        SplitDual($(NamedTuple{names})(($(field_exprs...),)))
    end
end
# NTangent-wrapped tangent variant — the generic `Lifted{P,N}(::P,
# ::NTangent)` dispatch routes here when `dual_type(Val(N), P) <:
# SplitDual`. Per-lane `MutableTangent`s combine field-wise into the
# canonical V's `(primal, ::NTangent{NTuple{N, …}})` ctor (defined per
# leaf type — `Array{NDual}`, `Memory{NDual}`, etc.).
@generated function (::Type{SplitDual{NamedTuple{names,Vs}}})(
    primal::P, tangent::NTangent{NTup}
) where {names,Vs<:Tuple,P,NTup<:Tuple}
    n = fieldcount(P)
    N = fieldcount(NTup)
    field_exprs = map(1:n) do i
        Vi = fieldtype(Vs, i)
        fname = names[i]
        if Vi <: PossiblyUninitTangent
            Vi_inner = fieldtype(Vi, :tangent)
            lane_exprs = map(
                d -> :(val(getfield(tangent.lanes[$d].fields, $(QuoteNode(fname))))),
                1:N,
            )
            quote
                if isdefined(primal, $(QuoteNode(fname)))
                    inner = _inner_dual_for_field(
                        $Vi_inner,
                        getfield(primal, $(QuoteNode(fname))),
                        NTangent(($(lane_exprs...),)),
                    )
                    $Vi(inner)
                else
                    $Vi()
                end
            end
        else
            lane_exprs = map(
                d -> :(val(getfield(tangent.lanes[$d].fields, $(QuoteNode(fname))))),
                1:N,
            )
            :(_inner_dual_for_field(
                $Vi,
                getfield(primal, $(QuoteNode(fname))),
                NTangent(($(lane_exprs...),)),
            ))
        end
    end
    return quote
        SplitDual($(NamedTuple{names})(($(field_exprs...),)))
    end
end

# Mooncake-protocol accessors for Lifted{P, N, SplitDual{V}} live below
# the `Lifted` type definition (see "SplitDual Lifted accessors" block).

# `tangent(x, dir)` — per-lane tangent accessor. Per-type fast paths for
# NDual, Complex{NDual}, Array{<:NDual}, NTangent, Memory, MemoryRef, etc.
# live in `nfwd/NfwdMooncake.jl`. The two-argument overload must not
# materialise all lanes before selecting one — extracting a single
# direction from a width-N container should remain O(container size), not
# O(N × container size). Per-type lane-extraction methods live directly on
# `tangent(x, ::Integer)` rather than a private `_tangent_dir` helper. The
# untyped fallback below returns `zero_tangent(x)` for primals that aren't
# NDual-bearing.
@inline tangent(x, _::Integer) = zero_tangent(x)

# `primal` / `tangent` on a bare element-wise tuple of inner duals (the inner
# `V` of a `Lifted{<:Tuple, N}`). Recursive map so nested Tuple-of-Dual works.
_field_primal(x) = x
_field_primal(x::Dual) = primal(x)
_field_tangent(x) = zero_tangent(x)
_field_tangent(x::Dual) = tangent(x)
primal(t::Tuple) = map(_field_primal, t)
tangent(t::Tuple) = map(_field_tangent, t)
# Bare NamedTuple inner V (struct-primal recursive lift, see §13 of
# notes/mooncake/dual-types.md): per-field `primal` / `tangent`. Mirrors
# the Tuple bare-V conventions above.
primal(t::NamedTuple) = map(_field_primal, t)
tangent(t::NamedTuple) = map(_field_tangent, t)
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
# Dual can be safely shared without copying
_copy(x::P) where {P<:Dual} = x

"""
    extract(x::Dual)

Helper function. Returns the 2-tuple `x.x, x.dx`.
"""
extract(x::Dual) = primal(x), tangent(x)

zero_dual(x) = Dual(x, zero_tangent(x))
randn_dual(rng::AbstractRNG, x) = Dual(x, randn_tangent(rng, x))

# Generic width-N fallback. The result must match the canonical
# `dual_type(Val(N), P)` shape for primal `P`. Branches, in order:
#
# 1. Concrete `Tuple` / `NamedTuple` primals: element-wise recursion. Also
#    handles empty tuples and tuples whose fields all have `NoTangent`
#    tangent — those still need the bare element-wise tuple form rather than
#    a Dual{Tuple, NoTangent} wrap.
# 2. Structural-immutable struct primals whose `dual_type` is the structural
#    `NamedTuple` lift: build the lift via `_structural_zero_dual_struct`.
# 3. `NoTangent` primals: `Dual(x, NoTangent())` matches the canonical
#    `Dual{P, NoTangent}`.
# 4. Otherwise consult `dual_type(Val(N), P)`:
#    - `Dual{P, NTangent{...}}` (the canonical positive-width form for
#      non-leaf primals, and also the canonical width-1 form for generic
#      concrete `P`): wrap N independent `zero_tangent(x)` lanes.
#    - `Dual{P, T}` (bare-T width-1 wrapper exceptions like `Diagonal`,
#      `Adjoint{T,Matrix{T}}`, `SubArray`, the triangulars, etc.): return
#      `Dual(x, zero_tangent(x))` (the legacy form).
# 5. Anything not matching falls through to the no-width form (Val(0)
#    callers without a specialised overload, etc.).
#
# Specialised IEEEFloat / Complex / array / Memory overloads in
# `nfwd/NfwdMooncake.jl` dispatch first by Julia method specificity and so
# don't reach this fallback.
@inline function zero_dual(w::Val{N}, x) where {N}
    P = _typeof(x)
    if N >= 1
        isconcretetype(P) && P <: Tuple && return _structural_zero_dual_tuple(w, x)
        isconcretetype(P) &&
            P <: NamedTuple &&
            return _structural_zero_dual_namedtuple(w, x)
        _uses_structural_dual_type(P) &&
            dual_type(w, P) <: NamedTuple &&
            return _structural_zero_dual_struct(w, x)
        tangent_type(P) === NoTangent && return Dual(x, NoTangent())
        DT = dual_type(w, P)
        if DT isa DataType && DT <: SplitDual
            return DT(x, zero_tangent(x))
        end
        if DT isa DataType && DT <: Dual
            T = DT.parameters[2]
            if T <: NTangent
                return Dual(x, NTangent(ntuple(_ -> zero_tangent(x), Val(N))))
            end
            return Dual(x, zero_tangent(x))
        end
    end
    tangent_type(P) === NoTangent && return Dual(x, NoTangent())
    return zero_dual(x)
end

@inline _structural_zero_dual_tuple(w::Val, x::Tuple) = ntuple(
    i -> zero_dual(w, getfield(x, i)), Val(fieldcount(typeof(x)))
)
@inline function _structural_zero_dual_namedtuple(
    w::Val, x::NamedTuple{names}
) where {names}
    return NamedTuple{names}(
        ntuple(i -> zero_dual(w, getfield(x, i)), Val(fieldcount(typeof(x))))
    )
end
@inline function _structural_zero_dual_struct(w::Val, x)
    P = typeof(x)
    return NamedTuple{fieldnames(P)}(
        ntuple(i -> zero_dual(w, getfield(x, i)), Val(fieldcount(P)))
    )
end

# No-`Val` `dual_type(P)` delegates to `dual_type(Val(1), P)` so the two
# queries agree by construction. The IEEEFloat / Complex / Array specialised
# overloads return `NDual`-shaped forms; generic concrete `P` returns
# `Dual{P, NTangent{Tuple{tangent_type(P)}}}`.
@unstable dual_type(::Type{P}) where {P} = dual_type(Val(1), P)

function dual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

_primal(x) = x
_primal(x::Dual) = primal(x)

"""
    verify_dual_type(x::Dual)

Check that the type of `tangent(x)` is the tangent type of the type of `primal(x)`.
Accepts both legacy bare tangents and width-aware NTangent-wrapped tangents.
"""
function verify_dual_type(x::Dual)
    P = typeof(primal(x))
    T = typeof(tangent(x))
    T === NoTangent && return tangent_type(P) === NoTangent
    if T <: NTangent
        N = fieldcount(T.parameters[1])
        return T === tangent_type(Val(N), P) ||
               (N == 1 && fieldtype(T.parameters[1], 1) === tangent_type(P))
    end
    # Legacy width-1 path: bare tangent without NTangent wrapper
    return tangent_type(P) == T
end

function error_if_incorrect_dual_types(duals::Vararg{Dual,N}) where {N}
    correct_types = map(verify_dual_type, duals)
    if !all(correct_types)
        primals = map(primal, duals)
        tangents = map(tangent, duals)
        throw(ArgumentError("""
        Tangent types do not match primal types:
          - primal types:           $(map(typeof, primals))
          - provided tangent types: $(map(typeof, tangents))
          - required tangent types: $(map(tangent_type, map(typeof, primals)))
        """))
    end
end

@inline uninit_dual(x::P) where {P} = Dual(x, uninit_tangent(x))

# Width-aware `uninit_dual` and `randn_dual` fallbacks, matching the structural
# branches in `zero_dual(w::Val, x)` above. Specialised IEEEFloat / Complex /
# array / Memory overloads live in `nfwd/NfwdMooncake.jl`.
@inline function uninit_dual(w::Val{N}, x) where {N}
    P = _typeof(x)
    if N >= 1
        isconcretetype(P) &&
            P <: Tuple &&
            return ntuple(i -> uninit_dual(w, getfield(x, i)), Val(fieldcount(P)))
        if isconcretetype(P) && P <: NamedTuple
            return NamedTuple{fieldnames(P)}(
                ntuple(i -> uninit_dual(w, getfield(x, i)), Val(fieldcount(P)))
            )
        end
        if _uses_structural_dual_type(P) && dual_type(w, P) <: NamedTuple
            return NamedTuple{fieldnames(P)}(
                ntuple(i -> uninit_dual(w, getfield(x, i)), Val(fieldcount(P)))
            )
        end
        tangent_type(P) === NoTangent && return Dual(x, NoTangent())
        DT = dual_type(w, P)
        if DT isa DataType && DT <: SplitDual
            return DT(x, uninit_tangent(x))
        end
        if DT isa DataType && DT <: Dual
            T = DT.parameters[2]
            if T <: NTangent
                return Dual(x, NTangent(ntuple(_ -> uninit_tangent(x), Val(N))))
            end
            return Dual(x, uninit_tangent(x))
        end
    end
    tangent_type(P) === NoTangent && return Dual(x, NoTangent())
    return uninit_dual(x)
end

@inline function randn_dual(w::Val{N}, rng::AbstractRNG, x) where {N}
    P = _typeof(x)
    if N >= 1
        isconcretetype(P) &&
            P <: Tuple &&
            return ntuple(i -> randn_dual(w, rng, getfield(x, i)), Val(fieldcount(P)))
        if isconcretetype(P) && P <: NamedTuple
            return NamedTuple{fieldnames(P)}(
                ntuple(i -> randn_dual(w, rng, getfield(x, i)), Val(fieldcount(P)))
            )
        end
        if _uses_structural_dual_type(P) && dual_type(w, P) <: NamedTuple
            return NamedTuple{fieldnames(P)}(
                ntuple(i -> randn_dual(w, rng, getfield(x, i)), Val(fieldcount(P)))
            )
        end
        tangent_type(P) === NoTangent && return Dual(x, NoTangent())
        DT = dual_type(w, P)
        if DT isa DataType && DT <: SplitDual
            return DT(x, randn_tangent(rng, x))
        end
        if DT isa DataType && DT <: Dual
            T = DT.parameters[2]
            if T <: NTangent
                return Dual(x, NTangent(ntuple(_ -> randn_tangent(rng, x), Val(N))))
            end
            return Dual(x, randn_tangent(rng, x))
        end
    end
    tangent_type(P) === NoTangent && return Dual(x, NoTangent())
    return randn_dual(rng, x)
end

# Always sharpen the first thing if it's a type so static dispatch remains possible.
function Dual(x::Type{P}, dx::NoTangent) where {P}
    return Dual{@isdefined(P) ? Type{P} : typeof(x),NoTangent}(x, dx)
end

# ── Inner-type constructors (Layer-2 dual targets) ────────────────────────────
# These let the `Lifted{P, N}(primal, tangent)` 2-arg constructor delegate to
# `dual_type(Val(N), P)(primal, tangent)` without rule bodies needing to choose
# the inner shape. Each inner type accepts the canonical tangent shapes a rule
# body may produce: scalar broadcast, pre-computed lanes, or NTangent-wrapped.

# 1-tuple convenience for the bare `Dual{P, T}` width-1 inner: lets per-lane rule
# bodies use the same `ntuple(closure, Val(N))` pattern at any width — at width 1
# the closure produces a 1-tuple `(t,)` which this method unwraps to scalar `t`.
Dual{P,T}(value, t::NTuple{1,T}) where {P,T} = Dual{P,T}(value, t[1])
# NTangent-wrapped singleton unwraps to the bare tangent for the bare-T
# form — an explicit width-1 compatibility boundary. The canonical width-1
# inner is `Dual{P, NTangent{Tuple{T}}}`; this method is kept for callers
# that explicitly request the legacy bare-T `Dual{P, T}` shape (e.g.
# `_ndual_output_to_width1`'s public-boundary normalisation,
# `from_chainrules` adapters, hand-written legacy rules).
Dual{P,T}(value, t::NTangent{Tuple{T}}) where {P,T} = Dual{P,T}(value, t.lanes[1])

# Chunked structured `Dual{P, NTangent{NTuple{N, T}}}`: pre-computed lanes wrap
# in NTangent; scalar tangent broadcasts across N lanes.
# Inner-tuple shape is bound through `V` (rather than `NTuple{N,T}`) so the
# zero-lane edge case does not leave `T` unbound (Aqua `test_unbound_args`).
function (::Type{Dual{P,NTangent{V}}})(value::P, lanes::V) where {P,V<:Tuple}
    return Dual(value, NTangent(lanes))
end
function (::Type{Dual{P,NTangent{NTuple{N,T}}}})(value::P, tangent::T) where {P,N,T}
    return Dual(value, NTangent(ntuple(_ -> tangent, Val(N))))
end

# Bare-T `Dual{P, T}` → canonical width-1 `Dual{P, NTangent{Tuple{T}}}` convert.
# Friendly-tangent inputs (e.g. `Dual{Core.Box, MutableTangent}` from user-facing
# `value_and_derivative!!` callers) flow into width-1 cache slots typed as
# `Dual{P, NTangent{Tuple{T}}}`. Julia's auto-generated `convert` between
# parametric `Dual{P, T1}` and `Dual{P, T2}` fails when T1 != T2; this explicit
# convert wraps the bare tangent in a singleton NTangent.
function Base.convert(::Type{Dual{P,NTangent{Tuple{T}}}}, x::Dual{P,T}) where {P,T}
    return Dual(primal(x), NTangent((tangent(x),)))
end

# Bare-tangent 2-arg ctor for the canonical width-1 `Dual{P, NTangent{Tuple{T}}}`
# slot. Callers crossing the AD boundary with a paired `Dual{P, T}` (e.g.
# `test_frule_interface` lifting `Dual{CuArray, CuArray}` to
# `Lifted{CuArray, 1, Dual{CuArray, NTangent{Tuple{CuArray}}}}` via the
# `Lifted{P,N}(primal, tangent)` ctor) reach this constructor with a raw
# tangent rather than the NTangent-wrapped form. Wrap it here so the
# downstream auto-generated `Dual{P, NTangent{Tuple{T}}}(::P, ::NTangent{...})`
# dispatch succeeds.
function Dual{P,NTangent{Tuple{T}}}(value::P, tangent::T) where {P,T}
    return Dual{P,NTangent{Tuple{T}}}(value, NTangent((tangent,)))
end

# `lsetfield!` rule bodies can produce a bare new value (e.g. an Int when
# updating a `Vector`'s `:size` field). The slot type expects
# `Dual{Int64, NoTangent}`; this convert lets Julia's `setfield!` /
# typed-Tuple slot writes succeed.
function Base.convert(::Type{Dual{P,NoTangent}}, x::P) where {P}
    return Dual{P,NoTangent}(x, NoTangent())
end

# NTangent{Tuple{T}} → NoTangent convert: some IR paths wrap a
# NoTangent-leaf primal's tangent in NTangent (e.g. tangent of a Tuple
# element that turned out to be Int64) before it reaches a
# `Dual{P, NoTangent}` slot. The slot's NoTangent tangent type means
# the lane content is meaningless for AD purposes — drop and return
# NoTangent.
Base.convert(::Type{NoTangent}, ::NTangent) = NoTangent()

# ── Lifted: Layer-3 wrapper struct ───────────────────────────────────────────

"""
    Lifted{P, N, V}

Forward-mode slot wrapper. `P` is the primal type, `N` is the AD width
(`1` ordinary, `N >= 2` chunked), and `V === dual_type(Val(N), P)` is the
inner dual shape. Slot identity `(P, N)` is dispatch-visible at the wrapper
level; the inner shape `V` is hidden.

Mirrors `CoDual{Tx, Tdx}` for forward mode. The extra `N` parameter and the
fact that `V` varies (NDual, Complex{<:NDual}, Array{<:NDual}, …) reflect
forward mode supporting multiple inner shapes; `V` is fully determined by
`(P, N)` via `dual_type`.

`Lifted` exists for every `P` at `N >= 1`, including `Tuple` primals (which
lift to a single outer `Lifted` whose `V` is a `Tuple` of bare inner duals —
`Lifted` never nests inside another `Lifted`'s `V`). `Val(0)` slots are
unwrapped (primal passthrough) and are not represented by `Lifted`.
"""
struct Lifted{P,N,V}
    value::V
end

"""
    verify_lifted_type(x::Lifted{P, N, V})

Check the canonical Lifted-slot invariant: `V === dual_type(Val(N), P)`, the
stored `value::V` matches that type, and the inner dual value itself
validates via `verify_dual_type`. Rejects nested `Lifted` inside `V`.

Returns `true` if `x` is canonical, `false` otherwise. Companion to
`verify_dual_type`, which validates only the inner dual value; this
function additionally validates the outer slot wrapper.
"""
function verify_lifted_type(x::Lifted{P,N,V}) where {P,N,V}
    # Reject nested `Lifted` — slot wrappers must not appear inside V.
    _contains_lifted(V) && return false
    # Canonical V for this (P, N) must match.
    V === dual_type(Val(N), P) || return false
    # Stored value must actually be of declared V.
    x.value isa V || return false
    # And the inner value must itself be a valid dual.
    return verify_dual_type(x.value)
end

# Type-level walker: detect any `Lifted` nested in T (including inside Tuple /
# NamedTuple V shapes). Fully type-domain, no runtime cost in the foldable
# generic case.
@inline _contains_lifted(::Type{<:Lifted}) = true
@inline _contains_lifted(::Type{T}) where {T<:Tuple} = any(_contains_lifted, fieldtypes(T))
@inline _contains_lifted(::Type{NamedTuple{names,T}}) where {names,T} = _contains_lifted(T)
@inline _contains_lifted(::Type) = false

# Lifted slot wrappers: the canonical V invariant guarantees
# `V === dual_type(Val(N), P)`, so the slot is well-typed by construction.
verify_dual_type(::Lifted) = true
# Bare canonical-V Tuple (e.g. `Tuple{Dual{Int,NoTangent}, Dual{Float64,...}}`)
# from `_unlift(::Lifted{<:Tuple, 1})` — verify each element.
verify_dual_type(t::Tuple) = all(verify_dual_type, t)
verify_dual_type(t::NamedTuple) = all(verify_dual_type, values(t))
# `SplitDual{V}` is a structural inner V for mutable struct lifts; its
# canonical NamedTuple is field-wise coherent by construction, so
# delegating to the NamedTuple overload validates each field's V.
verify_dual_type(d::SplitDual) = verify_dual_type(d.canonical)
# `PossiblyUninitTangent{Vi}` slots in a SplitDual NamedTuple wrap the
# canonical V for PossiblyUninit primal fields. Unwrap via `val` and
# recurse if assigned; an uninitialised slot is trivially valid.
verify_dual_type(p::PossiblyUninitTangent) = is_init(p) ? verify_dual_type(val(p)) : true
# Bare canonical-V leaf-scalar shapes (`NDual`, `Complex{<:NDual}`,
# `<:NTangent`) leak through helper-API boundaries but still represent valid
# inner dual values. Their concrete overloads are added in `nfwd/NfwdMooncake.jl`
# once `NDual` is defined; the fallback below accepts everything else as
# user-typed primitive values flowing through outside the strict slot path.

# 1-arg: wrap an already-built inner. V is inferred from typeof(value).
@inline Lifted{P,N}(value) where {P,N} = Lifted{P,N,typeof(value)}(value)

# 2-arg: build the inner via the inner type's own constructor methods. Mirrors
# `CoDual(x, dx)` — pass `(primal, tangent)` and the wrapper takes care of the
# rest. The dispatch on inner shape lives in the inner type's constructors.
#
# When `P` is abstract, sharpen to `typeof(primal)` so the runtime wrapper remains
# canonical (`V === dual_type(Val(N), Q)` for concrete `Q = typeof(primal)`).
# Abstract slot compatibility is checked separately via `isa lifted_type(Val(N),
# P_static)` — the concrete runtime wrapper is a subtype of the abstract
# `Lifted{Q,N,V} where {Q<:P_static, V}` annotation. This eliminates the dead
# `dual_type(Val(N), abstract_P) === Dual` (abstract) path, which previously
# produced unconstructable abstract `V` slots.
@inline function Lifted{P,N}(primal, tangent) where {P,N}
    if !isconcretetype(P)
        Q = _typeof(primal)
        InnerT = dual_type(Val(N), Q)
        # Tuple-shaped V (e.g. `Tuple{Dual{Type{Float64},NoTangent}, ...}` from
        # the element-wise lift of `Tuple{Type{Float64},...}`): the Tuple type
        # has no `(::Tuple, ::Tangent)` ctor — build element-wise via
        # `_build_tuple_v_from_pair`. Same shape as the existing Tuple+NoTangent
        # `@generated` ctor below, but tolerant of `typeof(primal) ⊊ Q` (the
        # dispatch-tuple widening that happens for `Tuple{Type{...}, ...}`).
        if InnerT isa DataType && InnerT <: Tuple
            return Lifted{Q,N,InnerT}(_build_tuple_v_from_pair(InnerT, primal, tangent))
        end
        return Lifted{Q,N,InnerT}(InnerT(primal, tangent))
    end
    InnerT = dual_type(Val(N), P)
    if InnerT isa DataType && InnerT <: Tuple
        return Lifted{P,N,InnerT}(_build_tuple_v_from_pair(InnerT, primal, tangent))
    end
    return Lifted{P,N,InnerT}(InnerT(primal, tangent))
end

# Element-wise build of a Tuple-V from a `(primal_tuple, tangent)` pair. Used
# when `dual_type(Val(N), P)` produces a `Tuple{V_i…}` shape and the value
# 2-arg path in `Lifted{P,N}(primal, tangent)` would otherwise call
# `Tuple{V_i…}(primal, tangent)` (which has no such constructor).
@inline @generated function _build_tuple_v_from_pair(
    ::Type{InnerT}, primal::Tuple, tangent::NoTangent
) where {InnerT<:Tuple}
    n = fieldcount(InnerT)
    exprs = [
        :(_inner_dual_for_field($(fieldtype(InnerT, i)), primal[$i], NoTangent())) for
        i in 1:n
    ]
    return :(($(exprs...),))
end
# Tuple-tangent variant: per-element `(primal[i], tangent[i])` pair feeds the
# matching field-V ctor. Reached when `Lifted{P<:Tuple, N}(primal::P,
# tangent::Tuple)` is invoked at a width where each element's tangent has
# already been lifted (e.g. NTangent per element from a mixed Val/Type/Float64
# primal Tuple inside `_apply_iterate_equivalent`).
@inline @generated function _build_tuple_v_from_pair(
    ::Type{InnerT}, primal::Tuple, tangent::Tuple
) where {InnerT<:Tuple}
    n = fieldcount(InnerT)
    exprs = [
        :(_inner_dual_for_field($(fieldtype(InnerT, i)), primal[$i], tangent[$i])) for
        i in 1:n
    ]
    return :(($(exprs...),))
end

# Type-slot specialisation: `dual_type(Val(N), Type{P_user})` may substitute the
# inner type parameter (e.g. `Type{Memory{Float64}}` → V-primal
# `Type{Memory{NDual{Float64,N}}}` per the override in `nfwd/NfwdMooncake.jl`)
# so that the OC slot's inner V matches what IR-emit produces at runtime. The
# user-facing path (test framework, direct callers) still passes the
# unsubstituted `P_user`, which would fail the auto-generated
# `Dual{Type{P_lifted}, NoTangent}` constructor. Detect the Type-slot shape and
# substitute to `P_lifted` here so both paths converge. Safe when `P_lifted ==
# P_user` (e.g. `Type{Float64}` lifts to itself); the substitution is a no-op.
@inline function Lifted{P,N}(primal::Type, tangent::NoTangent) where {P<:Type,N}
    InnerT = dual_type(Val(N), P)
    # Only substitute when the inner primal slot is a specific `Type{P_lifted}`
    # (one type parameter). For a bare `DataType` slot (no parameters), the
    # primal *is* the user-supplied value, so the substitution is skipped and
    # the runtime primal flows through to the inner Dual ctor directly.
    if InnerT isa DataType &&
        InnerT <: Dual &&
        InnerT.parameters[1] <: Type &&
        length(InnerT.parameters[1].parameters) > 0
        P_lifted = InnerT.parameters[1].parameters[1]
        return Lifted{P,N,InnerT}(InnerT(P_lifted, NoTangent()))
    end
    return Lifted{P,N,InnerT}(InnerT(primal, tangent))
end

# 3-param 2-arg ctor: callers that obtained `T = lifted_type(Val(N), P)` (the
# fully-parameterised slot type) and want to build it with `(primal, tangent)`.
# Forwards to the 2-param 2-arg form.
@inline Lifted{P,N,V}(primal, tangent) where {P,N,V} = Lifted{P,N}(primal, tangent)

# Tuple-primal special case: `dual_type(Val(N), P<:Tuple)` is a bare
# element-wise `Tuple{...}` of inner duals, which has no user-defined 2-arg
# constructor. Build the inner tuple here element-wise. Nested tuple fields
# recurse via `_inner_dual_for_field` so that
# `Tuple{NDual, Tuple{NDual, Vector{NDual}}, ...}` is built leaf-by-leaf.
#
# `@generated` so the per-field `_inner_dual_for_field` calls unroll
# statically — a runtime `ntuple` closure body would leave the inner result
# Union-typed (each Vᵢ varies) and force a heap allocation. The tangent shape
# is read from the dispatched type parameter `Tt`; the per-field accessor
# differs (`tangent[i]`, `NoTangent()`, or `NTuple{N, Tᵢ}` for NTangent
# lanes) but the rest of the body is shared.
@inline @generated function Lifted{P,N}(
    primal::P, tangent::Tt
) where {P<:Tuple,N,Tt<:Union{Tuple,NoTangent,NTangent}}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: Tuple)
        Tt <: NTangent && return :(
            if _all_notangent_lanes(tangent)
                Lifted{$P,$N}(primal, NoTangent())
            else
                Lifted{$P,$N,$InnerT}($InnerT(primal, tangent))
            end
        )
        return :(invoke(Lifted{$P,$N}, Tuple{Vararg{Any}}, primal, tangent))
    end
    n = fieldcount(P)
    inner_exprs = if Tt <: NoTangent
        map(1:n) do i
            :(_inner_dual_for_field($(fieldtype(InnerT, i)), primal[$i], NoTangent()))
        end
    elseif Tt <: Tuple
        map(1:n) do i
            :(_inner_dual_for_field($(fieldtype(InnerT, i)), primal[$i], tangent[$i]))
        end
    else  # Tt <: NTangent
        map(1:n) do i
            Vi = fieldtype(InnerT, i)
            lane_exprs = map(d -> :(_lane_field(tangent.lanes[$d], $i)), 1:N)
            quote
                let partials = ($(lane_exprs...),)
                    $Vi(primal[$i], _all_notangent_lane(partials) ? NoTangent() : partials)
                end
            end
        end
    end
    return quote
        return Lifted{$P,$N,$InnerT}(($(inner_exprs...),))
    end
end

# Build a single field's inner dual value. For non-Tuple fields, defer to the
# field type's own 2-arg constructor (NDual, Vector{NDual}, Dual, etc.). For
# Tuple-typed fields, recurse element-wise so a nested Tuple-of-Dual builds
# without trying `Tuple{...}(::Tuple, ::Tuple)` (which has no ctor).
@inline function _inner_dual_for_field(::Type{V}, primal::P, tangent::T) where {V,P,T}
    # Do NOT add an "escape hatch" that returns a bare `Dual(primal, tangent)`
    # (= `Dual{P, T}`) when V is a concrete Dual whose declared tangent type
    # differs from `T`. Past attempts bypassed the NTangent wrap, producing
    # element-wise bare-T Tuple-of-Duals that didn't match the canonical
    # width-1 OC slot (`Dual{P, NTangent{Tuple{T}}}`) and surfaced as
    # typeassert failures in high_order_derivative_patches /
    # misty_closures. The canonical V's 2-arg ctor (see §"Inner-type
    # constructors" above) already adapts the tangent shape — route through it.
    return V(primal, tangent)
end
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Base.Broadcast.Extruded, tangent::Tangent
) where {V<:Base.Broadcast.Extruded}
    inner_x = _inner_dual_for_field(fieldtype(V, 1), primal.x, val(tangent.fields.x))
    # Broadcast metadata is non-differentiable; only `x` carries tangent data.
    return V(inner_x, primal.keeps, primal.defaults)
end
# When the tangent is already the canonical V (e.g. test_rule passes a `Dual`
# whose tangent slot carries an NDual or Array{NDual} directly), pass it
# through — re-wrapping would invoke a non-existent 2-arg ctor.
@inline _inner_dual_for_field(::Type{V}, primal, tangent::V) where {V} = tangent
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Tuple, tangent::Tuple
) where {V<:Tuple}
    return ntuple(
        i -> _inner_dual_for_field(fieldtype(V, i), primal[i], tangent[i]),
        Val(fieldcount(V)),
    )
end
# `Complex{<:Dual-like}` inner V with `NTangent{NTuple{N, Complex{T}}}` tangent
# (the multi-lane NTangent shape produced by chunked forward-mode for a Complex
# scalar primal). Build the inner Complex element-wise: per-lane real/imag
# partials split into two inner duals via the inner V's own field type ctor.
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Complex{T}, tangent::NTangent
) where {V<:Complex,T<:IEEEFloat}
    DualT = fieldtype(V, 1)  # the inner NDual / Dual element type
    N = length(tangent.lanes)
    re_partials = ntuple(lane -> real(tangent.lanes[lane]), N)
    im_partials = ntuple(lane -> imag(tangent.lanes[lane]), N)
    return Complex(DualT(real(primal), re_partials), DualT(imag(primal), im_partials))
end
# NoTangent broadcast for Tuple/NamedTuple V — recurse element-wise with NoTangent
# at each leaf. Required when a parent `Lifted{<:Tuple, N}(primal, ::NoTangent)`
# constructor visits a nested Tuple field.
@inline function _inner_dual_for_field(
    ::Type{V}, primal::Tuple, ::NoTangent
) where {V<:Tuple}
    return ntuple(
        i -> _inner_dual_for_field(fieldtype(V, i), primal[i], NoTangent()),
        Val(fieldcount(V)),
    )
end
@inline function _inner_dual_for_field(
    ::Type{V}, primal::NamedTuple, ::NoTangent
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(fieldtype(InnerTup, i), values(primal)[i], NoTangent()),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end
# NamedTuple-typed nested fields recurse element-wise like Tuple.
@inline function _inner_dual_for_field(
    ::Type{V}, primal::NamedTuple, tangent::NamedTuple
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(
            fieldtype(InnerTup, i), values(primal)[i], values(tangent)[i]
        ),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end
# Nested-struct case: V is the recursive NamedTuple lift of an immutable
# struct field whose tangent is `Tangent`. Recurse element-wise, extracting
# per-field primals from the struct and per-field tangents from the
# Tangent's fields NamedTuple (with `_get_tangent_field_for_lift` unwrapping
# `PossiblyUninitTangent`).
@inline function _inner_dual_for_field(
    ::Type{V}, primal, tangent::Tangent
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    inner_tup = ntuple(
        i -> _inner_dual_for_field(
            fieldtype(InnerTup, i),
            getfield(primal, names[i]),
            _get_tangent_field_for_lift(tangent, names[i]),
        ),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end

# Width-N nested-struct case: V is the recursive NamedTuple lift; tangent
# is an `NTangent` of per-lane Tangents. Recurse element-wise: for each
# field, gather per-lane tangent values into an NTangent and recurse with
# the (struct-primal-field, per-lane-NTangent) pair. Parallels the
# `tangent::Tangent` overload above but for the multi-lane wrap.
@inline function _inner_dual_for_field(
    ::Type{V}, primal, tangent::NTangent
) where {names,V<:NamedTuple{names}}
    InnerTup = V.parameters[2]
    N = length(tangent.lanes)
    inner_tup = ntuple(
        i -> _inner_dual_for_field(
            fieldtype(InnerTup, i),
            getfield(primal, names[i]),
            NTangent(
                ntuple(
                    lane -> _get_tangent_field_for_lift(tangent.lanes[lane], names[i]),
                    Val(N),
                ),
            ),
        ),
        Val(fieldcount(V)),
    )
    return NamedTuple{names}(inner_tup)
end

# NoTangent / NTangent variants for Tuple-primal are handled by the unified
# Tuple-primal @generated ctor above (dispatched via `Tt`).
@inline _all_notangent_lanes(t::NTangent) = all(_all_notangent_lane, t.lanes)
@inline _all_notangent_lane(t::NoTangent) = true
@inline _all_notangent_lane(t::Tuple) = all(_all_notangent_lane, t)
@inline _all_notangent_lane(t::NamedTuple) = all(_all_notangent_lane, values(t))
@inline _all_notangent_lane(_) = false

# Per-lane element accessor used by the unified width-1/N inner-V constructor.
# A `NoTangent` lane has no per-element structure, so `lane[i]` would error —
# return `NoTangent()` instead and let the per-element guard collapse to
# `NoTangent()` when every lane is `NoTangent`.
@inline _lane_field(lane::NoTangent, i::Integer) = NoTangent()
@inline _lane_field(lane, i::Integer) = lane[i]

# Unified NamedTuple-primal Lifted ctor — parallels the Tuple-primal one above.
# Dispatches on tangent shape `Tt` at @generated expansion time so the per-field
# `_inner_dual_for_field` calls unroll statically with the right tangent
# accessor. `Tt` excludes `NTangent` here to avoid a dispatch ambiguity with the
# generic struct-primal `(P, ::NTangent)` ctor below; the NTangent case for
# NamedTuple primals is handled by a separate disambiguation method.
@inline @generated function Lifted{P,N}(
    primal::P, tangent::Tt
) where {names,P<:NamedTuple{names},N,Tt<:Union{NamedTuple,NoTangent}}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        return :(invoke(Lifted{$P,$N}, Tuple{Vararg{Any}}, primal, tangent))
    end
    InnerTup = InnerT.parameters[2]
    n = fieldcount(P)
    inner_exprs = if Tt <: NoTangent
        map(1:n) do i
            :($(fieldtype(InnerTup, i))(primal[$i], NoTangent()))
        end
    else  # Tt <: NamedTuple
        map(1:n) do i
            :(_inner_dual_for_field($(fieldtype(InnerTup, i)), primal[$i], values(tangent)[$i]))
        end
    end
    return quote
        return Lifted{$P,$N,$InnerT}($(NamedTuple{names})(($(inner_exprs...),)))
    end
end
# NamedTuple-primal + NTangent disambiguation: per-field NTuple-of-lane-partials
# build. Resolves the dispatch ambiguity between the unified NamedTuple-primal
# ctor above and the generic struct-primal `(P, ::NTangent)` ctor below.
@inline function Lifted{P,N}(primal::P, tangent::NTangent) where {P<:NamedTuple,N}
    InnerT = dual_type(Val(N), P)
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        _all_notangent_lanes(tangent) && return Lifted{P,N}(primal, NoTangent())
        return Lifted{P,N,InnerT}(InnerT(primal, tangent))
    end
    names = fieldnames(P)
    InnerTup = InnerT.parameters[2]
    lanes = tangent.lanes
    inner = ntuple(Val(fieldcount(P))) do i
        Vi = fieldtype(InnerTup, i)
        partials = ntuple(d -> _lane_field(lanes[d], i), Val(N))
        Vi(primal[i], _all_notangent_lane(partials) ? NoTangent() : partials)
    end
    return Lifted{P,N,InnerT}(NamedTuple{names}(inner))
end

# Struct-primal with `Tangent` tangent — recursive lift: the inner V is a
# `NamedTuple{fieldnames(S), Tuple{Vᵢ…}}` mirroring the struct's field
# structure with each field already in canonical V form. Each field is built
# via `_inner_dual_for_field`, which routes nested Tuple / NamedTuple /
# struct fields to their respective recursive constructors and leaf
# canonical-V types (`NDual`, `Vector{NDual}`, …) to their 2-arg ctors.
# `_get_tangent_field_for_lift` unwraps `PossiblyUninitTangent` slots so
# field tangents pass through unwrapped. Mutable structs keep the existing
# `Dual` path (their `dual_type` does not return `<: NamedTuple`).
@inline @generated function Lifted{P,N}(primal::P, tangent::Tangent) where {P,N}
    # `try` to resolve InnerT at expansion time. The static path emits a
    # tight ctor (recursive struct lift). On expansion failure (ext-typed
    # primitive-leaf world-age error inside `tangent_type`), defer the
    # whole construction to runtime.
    InnerT = try
        dual_type(Val(N), P)
    catch
        return :(_lifted_struct_runtime_fallback($P, Val($N), primal, tangent))
    end
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        # Non-struct lift: defer to the inner type's own 2-arg constructor.
        return :(Lifted{$P,$N,$InnerT}($InnerT(primal, tangent)))
    end
    names = fieldnames(P)
    InnerTup = InnerT.parameters[2]
    n = fieldcount(P)
    inner_exprs = map(1:n) do i
        :(_inner_dual_for_field(
            $(fieldtype(InnerTup, i)),
            getfield(primal, $(QuoteNode(names[i]))),
            _get_tangent_field_for_lift(tangent, $(QuoteNode(names[i]))),
        ))
    end
    return quote
        return Lifted{$P,$N,$InnerT}($(NamedTuple{names})(($(inner_exprs...),)))
    end
end

@inline function Lifted{P,N}(
    primal::P, tangent::Tangent
) where {P<:Base.Broadcast.Extruded,N}
    InnerT = dual_type(Val(N), P)
    XInnerT = fieldtype(InnerT, 1)
    inner_x = _inner_dual_for_field(XInnerT, getfield(primal, :x), val(tangent.fields.x))
    # Broadcast metadata is non-differentiable; only `x` carries tangent data.
    return Lifted{P,N,InnerT}(InnerT(inner_x, primal.keeps, primal.defaults))
end

# Helper: extract a field's tangent value from a `Tangent`, unwrapping
# `PossiblyUninitTangent` wrappers so the value passes to the leaf constructor
# in its bare form. Mirrors `_get_tangent_field` in `rules/misc.jl` but lives
# here to avoid a load-order dependency.
@inline _get_tangent_field_for_lift(t::Tangent, name) = val(getfield(t.fields, name))

# Struct-primal NTangent ctor: parallel to the `tangent::Tangent` form at the
# generated function above (line ~806), but combines per-lane tangents from an
# `NTangent{NTuple{N, Tangent}}` into per-field width-N partials. Required for
# rule bodies that return a Lifted with struct primal at width N≥2 (e.g.
# `lmemoryrefget` on `MemoryRef{<:Struct}`). The single-Tangent form at
# line ~806 broadcasts one tangent across all N lanes, which suffices for
# constants/seeds but NOT for rule outputs where each lane has independent
# tangent data.
#
# Specificity: `P<:Tuple` (line ~720) and `P<:NamedTuple` (line ~740) have
# their own `NTangent` ctors that are strictly more specific, so this generic
# variant only fires for "other" concrete struct primals.
#
# Disambiguator for `P<:Tuple, tangent::NTangent`: the Tuple-primal generated
# ctor above (line ~988) takes `Tt<:Union{Tuple,NoTangent,NTangent}` so its P
# is narrower but its tangent type is broader. This generic body has the
# narrower tangent type. Neither dominates → ambiguous. Pin both `P<:Tuple`
# and `tangent::NTangent` so this method is strictly more specific than both
# and route through the Tuple-primal body via `invoke`.
@inline function Lifted{P,N}(primal::P, tangent::NTangent) where {P<:Tuple,N}
    return invoke(Lifted{P,N}, Tuple{P,Union{Tuple,NoTangent,NTangent}}, primal, tangent)
end
@inline @generated function Lifted{P,N}(primal::P, tangent::NTangent) where {P,N}
    InnerT = try
        dual_type(Val(N), P)
    catch
        return :(_lifted_struct_runtime_fallback($P, Val($N), primal, tangent))
    end
    if !(InnerT isa DataType) || !(InnerT <: NamedTuple)
        # For Complex / NDual / other non-struct inner Vs, route through
        # `_inner_dual_for_field` so a missing direct `InnerT(primal, tangent)`
        # ctor (e.g. `Complex{NDual{T,N}}(::Complex, ::NTangent)` at N≥2)
        # is handled by a specialised method below rather than the default
        # `Complex(::Real, ::Real)` path that triggers `InexactError`.
        return :(Lifted{$P,$N,$InnerT}(_inner_dual_for_field($InnerT, primal, tangent)))
    end
    names = fieldnames(P)
    InnerTup = InnerT.parameters[2]
    n = fieldcount(P)
    inner_exprs = map(1:n) do i
        fname = QuoteNode(names[i])
        lane_exprs = map(
            d -> :(_get_tangent_field_for_lift(tangent.lanes[$d], $fname)), 1:N
        )
        :(_inner_dual_for_field(
            $(fieldtype(InnerTup, i)), getfield(primal, $fname), ($(lane_exprs...),)
        ))
    end
    return quote
        return Lifted{$P,$N,$InnerT}($(NamedTuple{names})(($(inner_exprs...),)))
    end
end

# Runtime fallback for `Lifted{P, N}(primal::P, tangent::Tangent|NTangent)` when
# the `@generated` expansion can't resolve `InnerT = dual_type(Val(N), P)` —
# typically because the recursive `tangent_type` descent through `P`'s fields
# hits a primitive-leaf world-age boundary (e.g. CuArray's nested
# `CuPtr{Nothing}` chain). At runtime the call uses the latest world.
# Dispatches on tangent shape to choose the per-field accessor:
# `_get_tangent_field_for_lift(tangent, name)` for a single Tangent vs.
# `ntuple(d -> _get_tangent_field_for_lift(tangent.lanes[d], name), Val(N))`
# for the multi-lane NTangent case.
@inline function _lifted_struct_runtime_fallback(
    ::Type{P}, ::Val{N}, primal, tangent
) where {P,N}
    InnerT = dual_type(Val(N), P)
    if InnerT isa DataType && InnerT <: NamedTuple
        names = fieldnames(P)
        InnerTup = InnerT.parameters[2]
        fields = ntuple(Val(fieldcount(P))) do i
            _inner_dual_for_field(
                fieldtype(InnerTup, i),
                getfield(primal, names[i]),
                _lifted_struct_field_tangent(tangent, names[i], Val(N)),
            )
        end
        return Lifted{P,N,InnerT}(NamedTuple{names}(fields))
    end
    return Lifted{P,N,InnerT}(InnerT(primal, tangent))
end
@inline _lifted_struct_field_tangent(t::Tangent, name, ::Val) = _get_tangent_field_for_lift(
    t, name
)
@inline _lifted_struct_field_tangent(t::NTangent, name, ::Val{N}) where {N} = ntuple(
    d -> _get_tangent_field_for_lift(t.lanes[d], name), Val(N)
)

# Accessors: delegate to the inner's own primal/tangent. Tuple primals need
# field-type-aware reconstruction: a field can have primal type `CoDual` while
# its canonical inner V is a structural `NamedTuple`, so a plain `map(primal, ...)`
# would erase the field's original type.
primal(d::Lifted) = primal(d.value)
tangent(d::Lifted) = tangent(d.value)

# SplitDual Lifted accessors: reconstruct the primal mutable struct from
# canonical Vs' `.value`s; build the matching `MutableTangent` from each
# field's `tangent(_, i)`. Reconstruction yields a fresh primal struct;
# SplitDual deliberately does not carry the user's struct identity.
#
# PossiblyUninitTangent-wrapped fields (whose primal counterpart may be
# undef) take the `jl_new_struct_uninit + jl_set_nth_field` path so that
# uninitialized fields stay undef in the reconstructed primal. With all
# fields always-initialised the generated body delegates to `P(args...)`.
@generated function primal(d::Lifted{P,N,V}) where {P,N,V<:SplitDual}
    nt_type = V.parameters[1]
    names = nt_type.parameters[1]
    field_types = nt_type.parameters[2].parameters
    has_put = any(T -> T <: PossiblyUninitTangent, field_types)
    if !has_put
        field_exprs = map(enumerate(names)) do (i, f)
            :(_field_primal_for_type(
                fieldtype(P, $i), getfield(d.value.canonical, $(QuoteNode(f)))
            ))
        end
        return :(P($(field_exprs...)))
    end
    set_exprs = map(enumerate(names)) do (i, f)
        Ti = field_types[i]
        if Ti <: PossiblyUninitTangent
            quote
                v = getfield(d.value.canonical, $(QuoteNode(f)))
                if is_init(v)
                    ccall(
                        :jl_set_nth_field,
                        Cvoid,
                        (Any, Csize_t, Any),
                        temp,
                        $(i - 1),
                        _field_primal_for_type(fieldtype(P, $i), val(v)),
                    )
                end
            end
        else
            quote
                ccall(
                    :jl_set_nth_field,
                    Cvoid,
                    (Any, Csize_t, Any),
                    temp,
                    $(i - 1),
                    _field_primal_for_type(
                        fieldtype(P, $i), getfield(d.value.canonical, $(QuoteNode(f)))
                    ),
                )
            end
        end
    end
    return quote
        temp = ccall(:jl_new_struct_uninit, Any, (Any,), P)::P
        $(set_exprs...)
        return temp::P
    end
end
# Type-guided field-primal extraction. Delegate to `_new_field_primal`
# (defined in `rules/new.jl`) which handles the canonical structural
# inner-V shapes — bare NDual / Complex{NDual} / Array{NDual} leaves,
# Tuple/NamedTuple structural inner Vs, and nested immutable struct
# fields. SplitDual fields (nested mutable struct primals) require
# extra reconstruction below.
@inline _field_primal_for_type(::Type{T}, val) where {T} = _new_field_primal(T, val)
@inline _field_primal_for_type(::Type{T}, val::SplitDual) where {T} = _construct_from_split_dual(
    T, val
)
@generated function _construct_from_split_dual(
    ::Type{T}, sd::SplitDual{V}
) where {T,V<:NamedTuple}
    names = V.parameters[1]
    field_exprs = map(enumerate(names)) do (i, f)
        :(_field_primal_for_type(fieldtype(T, $i), getfield(sd.canonical, $(QuoteNode(f)))))
    end
    return :(T($(field_exprs...)))
end
@generated function tangent(d::Lifted{P,N,V}) where {P,N,V<:SplitDual}
    lane_exprs = [:(tangent(d, $lane)) for lane in 1:N]
    return :(NTangent(($(lane_exprs...),)))
end
@generated function tangent(d::Lifted{P,N,V}, i::Integer) where {P,N,V<:SplitDual}
    return :(_build_mutable_tangent(P, d.value, i))
end
# Type-guided field-tangent extraction. Delegate to `_new_field_tangent`
# (defined in `rules/new.jl`) which produces the right tangent shape for
# the canonical structural inner Vs — `NDual`/`Complex{NDual}`/`Array{NDual}`
# leaves return their per-lane partials; immutable struct fields are
# rebuilt as `Tangent{...}`. SplitDual fields (nested mutable struct
# primals) recurse via `_build_mutable_tangent` below.
@inline _field_tangent_for_type(::Type{T}, val, i) where {T} = _new_field_tangent(T, val, i)
@inline _field_tangent_for_type(::Type{T}, val::SplitDual, i) where {T} = _build_mutable_tangent(
    T, val, i
)
# PossiblyUninitTangent-wrapped V: rebuild the matching PUT-wrapped tangent so
# `MutableTangent{NT}` (whose field types are `PUT{tangent_type(field)}`) gets
# the right shape. Uninit V → uninit PUT; init V → init PUT carrying the
# lane-extracted tangent of the inner canonical value.
@inline function _field_tangent_for_type(::Type{T}, val::PossiblyUninitTangent, i) where {T}
    Tput = PossiblyUninitTangent{tangent_type(T)}
    return is_init(val) ? Tput(_new_field_tangent(T, val.tangent, i)) : Tput()
end
@generated function _build_mutable_tangent(
    ::Type{T}, sd::SplitDual{V}, i
) where {T,V<:NamedTuple}
    names = V.parameters[1]
    tt = tangent_type(T)
    nt_inner = tt.parameters[1]
    field_exprs = map(enumerate(names)) do (idx, f)
        :(
            $(f) = _field_tangent_for_type(
                fieldtype(T, $idx), getfield(sd.canonical, $(QuoteNode(f))), i
            )
        )
    end
    return :(MutableTangent{$nt_inner}((; $(field_exprs...))))
end
# Type-singleton primal: a `Lifted{Type{P}, N, V}` slot stores the substituted
# `Type{P_lifted}` inside `V` (e.g. `Type{Array{Float64,D}}` → V-primal
# `Type{Array{NDual{Float64,N},D}}` per the `dual_type` override in
# `nfwd/NfwdMooncake.jl`). The substitution is for OC-internal use so
# `Array{...}(undef, n)` allocates `NDual` storage. Callers querying the
# user-facing primal type — e.g. `@zero_derivative` rules invoking
# `Base.allocatedinline(primal(p))` — should see the original `Type{P}`. Use the
# slot's `P` parameter directly; the inner V substitution stays available via
# `_unlift(slot)` for AD-internal callsites that need the lifted type.
primal(::Lifted{Type{P},N,V}) where {P,N,V} = P
@generated function primal(d::Lifted{P,N,V}) where {P<:Tuple,N,V<:Tuple}
    isconcretetype(P) || return :(map(primal, d.value))
    exprs = map(1:fieldcount(P)) do i
        Pi = fieldtype(P, i)
        Vi = fieldtype(V, i)
        if (Pi <: Tuple && Vi <: Tuple) || (Vi <: NamedTuple && !(Pi <: NamedTuple))
            :(primal(Lifted{$Pi,$N,$Vi}(d.value[$i])))
        else
            :(primal(d.value[$i]))
        end
    end
    return :(($(exprs...),))
end
@generated function tangent(d::Lifted{P,N,V}) where {P<:Tuple,N,V<:Tuple}
    isconcretetype(P) || return :(map(tangent, d.value))
    # Delegate per-lane to `tangent(d, lane)`, which recurses with
    # primal-type awareness and rebuilds inner `Tangent{...}` wrappers for
    # struct elements.
    lane_exprs = [:(tangent(d, $lane)) for lane in 1:N]
    return :(NTangent(($(lane_exprs...),)))
end
function primal(d::Lifted{P,N,V}) where {names,P<:NamedTuple,N,V<:NamedTuple{names}}
    return map(primal, d.value)
end
@generated function tangent(
    d::Lifted{P,N,V}
) where {names,P<:NamedTuple{names},N,V<:NamedTuple{names}}
    # Delegate per-lane so that NamedTuple-primal fields whose primal type
    # is a struct rebuild `Tangent{...}` correctly.
    lane_exprs = [:(tangent(d, $lane)) for lane in 1:N]
    return :(NTangent(($(lane_exprs...),)))
end

# Struct-primal accessors: the inner V is a `NamedTuple{fieldnames(P), Tuple{Vᵢ…}}`
# (recursive lift), but `P` itself is a struct, not a NamedTuple. Reconstruct
# the struct via `_new_` (Mooncake's bypass-constructor primitive) from the
# per-field primals; build a `Tangent` / `MutableTangent` whose fields carry
# the per-field tangent (NTangent-bearing for IEEEFloat-leaf fields, mirroring
# the existing `tangent(::Array{<:NDual})` convention). This shape is used
# for address-map tracking; `tangent(slot, i)` produces the bare-tangent
# shape used by `_dot` for FD comparison.
@generated function primal(d::Lifted{P,N,V}) where {P,N,V<:NamedTuple{names}} where {names}
    P <: NamedTuple && return :(map(primal, d.value))   # earlier method handles this
    # Nested struct fields need type-aware reconstruction, so delegate to
    # `_new_field_primal` (defined in rules/new.jl) which recurses with
    # primal field types and rebuilds inner structs.
    return :(_new_field_primal($P, d.value))
end
@generated function tangent(d::Lifted{P,N,V}) where {P,N,V<:NamedTuple{names}} where {names}
    P <: NamedTuple && return :(map(tangent, d.value))   # earlier method handles this
    # Delegate per-lane to `tangent(d, lane)`, which routes through
    # `_build_struct_tangent_dir` for struct primals so nested struct fields
    # are rebuilt as `Tangent{...}` rather than leaking raw
    # `NamedTuple{...}`.
    lane_exprs = [:(tangent(d, $lane)) for lane in 1:N]
    return :(NTangent(($(lane_exprs...),)))
end

"""
    extract(d::Lifted)

Return the 2-tuple `(primal(d), tangent(d))`.
"""
extract(d::Lifted) = (primal(d), tangent(d))

# Wrap / unwrap mechanics: `_lift(Val(N), P, _unlift(d))` is a typed identity.
# `Val(0)` is the primal passthrough — `lifted_type(Val(0), P) = P`, so the
# value passes through without wrapping.
_lift(::Val{0}, ::Type{P}, value) where {P} = value
_lift(::Val{N}, ::Type{P}, inner) where {N,P} = Lifted{P,N}(inner)
_unlift(d::Lifted) = d.value

Base.copy(d::Lifted{P,N}) where {P,N} = Lifted{P,N}(copy(d.value))
# Lifted can be safely shared without copying (same as Dual).
_copy(d::L) where {L<:Lifted} = d

# ── lifted_type: Layer-3 slot type query ─────────────────────────────────────

"""
    lifted_type(::Val{N}, ::Type{P})

Width-aware Layer-3 slot type query. Returns the wrapped slot type for primal
`P` at width `N`.

- `Val(0)` → `P` (primal passthrough)
- `Val(N)` → `Lifted{P, N, dual_type(Val(N), P)}`

Companion to `tangent_type` (Layer 1) and `dual_type` (Layer 2). The single
top-level slot-type query — used both by rule bodies (computing the type of
a result slot) and by IR-emit at lift sites (computing the type of an OC
slot). Symmetric with reverse mode's `codual_type`.
"""
lifted_type(::Val{0}, ::Type{P}) where {P} = P
function lifted_type(::Val{N}, ::Type{P}) where {N,P}
    V = dual_type(Val(N), P)
    # Concrete `(P, V)`: produce the fully-parameterised slot type — except
    # when `P` is a leaf "kind" (DataType etc.), where concrete V would
    # mismatch the singleton-V runtime values it admits. See
    # `_has_type_singleton_storage_mismatch`.
    if isconcretetype(V) && !_has_type_singleton_storage_mismatch(P, V)
        return Lifted{P,N,V}
    end
    # A `Tuple{Type{T1}, Type{T2}, ...}` primal lifts to a
    # `Tuple{Dual{Type{T1}, NoTangent}, Dual{Type{T2}, NoTangent}, ...}` V.
    # `isconcretetype` returns false for these (Type-of-Type fields), but
    # the constructor path can still build a concrete Lifted directly — the
    # alternative is an unrepresentable UnionAll-ctor call. Treat such Vs
    # as "instantiable" and return the fully-parametrised slot.
    V isa DataType && V <: Tuple && _is_instantiable_tuple_V(V) && return Lifted{P,N,V}
    # Abstract `P` (e.g. `Any`, `Real`): keep the width `N` bound and let `P`
    # widen via the existing `Q<:P` constraint and `V` widen freely. Returning
    # the bare `Lifted` UnionAll (the previous behaviour) was unsound — a
    # width-2 abstract slot would accept a width-1 lifted runtime value, since
    # `Lifted{Float64, 1, NDual{Float64, 1}} <: Lifted`. Preserving `N`
    # rejects cross-width substitution while still accepting any concrete
    # subtype of `P` at the bound width.
    return Lifted{Q,N,V_inner} where {Q<:P,V_inner}
end
# Leaf-kind `P` (`DataType`, `UnionAll`, `Union`, `Core.TypeofBottom`) — i.e.
# `P <: Type` with zero type-parameters — has `isconcretetype(P) == true` but
# its inhabitants are `Type{X}` singletons (0 bytes). A concrete slot
# `Lifted{P, N, Dual{P, NoTangent}}` declares `primal::P` as an 8-byte boxed
# pointer; passing a singleton `Dual{Type{X}, NoTangent}` value leaves that
# slot uninitialised — segfault in `jl_valid_type_param` from the
# `Core.apply_type` frule (e.g. CUDA `view → fieldtypes → ntupleany`). Widen
# V to UnionAll for these P so the OC emits a generic boxed access.
@inline function _has_type_singleton_storage_mismatch(::Type{P}, ::Type{V}) where {P,V}
    P isa DataType && P <: Type && length(P.parameters) == 0 || return false
    V isa DataType && V <: Dual || return false
    length(V.parameters) == 2 || return false
    Vp = V.parameters[1]
    Vp isa DataType && Vp <: Type && length(Vp.parameters) == 0 || return false
    return true
end
# A Tuple V is "instantiable" if every field is either concrete or a
# `Dual{Type{T}, NoTangent}` (which has a single inhabitant per T).
@inline function _is_instantiable_tuple_V(::Type{V}) where {V<:Tuple}
    for i in 1:fieldcount(V)
        ft = fieldtype(V, i)
        if isconcretetype(ft)
            continue
        elseif ft isa DataType &&
            ft <: Dual &&
            ft.parameters[1] isa DataType &&
            ft.parameters[1] <: Type
            continue
        else
            return false
        end
    end
    return true
end

# ── Layer-3 seed factories: return wrapped Lifted slot ───────────────────────
# One-line delegations to the Layer-2 factories. Both routes consult the same
# `dual_type` table for the inner shape; Layer 3 just adds the outer wrap.
# `Val(0)` is the primal passthrough — `lifted_type(Val(0), P) === P`.

"""
    zero_lifted(::Val{N}, x)

Width-aware Layer-3 seed factory. Returns a `Lifted{typeof(x), N}` wrapping
the canonical zero dual at width `N`, or the bare primal `x` at `Val(0)`.

Companion to `zero_dual` (Layer 2). The result type matches
`lifted_type(Val(N), typeof(x))`.
"""
# Val(0) is the primal passthrough — bare `x` flows unchanged at every
# primal shape. Explicit Tuple/NamedTuple/Type overloads disambiguate
# against the matching width-N variants below (Aqua flagged these as
# `zero_lifted(::Val{0}, ::Tuple)` etc.). Keep the generic `Val(0)` last
# so it acts as the catch-all.
@inline zero_lifted(::Val{0}, x::Tuple) = x
@inline zero_lifted(::Val{0}, x::NamedTuple) = x
@inline zero_lifted(::Val{0}, x::Type) = x
@inline zero_lifted(::Val{0}, x) = x
# When `dual_type(Val(N), P)` matches `typeof(zero_dual(w, x))` (the
# IEEEFloat / Ptr / NDual paths), the 1-arg `Lifted` ctor uses the
# canonical inner V directly. Otherwise route through the 2-arg ctor so
# the inner V is rebuilt from `(primal, tangent)` via
# `dual_type(Val(N), P)(primal, tangent)`. Avoids the generic
# `zero_tangent` safety check that rejects bare `Ptr` primals.
@inline function zero_lifted(w::Val{N}, x) where {N}
    zd = zero_dual(w, x)
    InnerT = dual_type(w, typeof(x))
    return if typeof(zd) === InnerT
        Lifted{typeof(x),N,InnerT}(zd)
    else
        Lifted{typeof(x),N}(primal(zd), tangent(zd))
    end
end
@inline function zero_lifted(w::Val{N}, x::Type{P}) where {N,P}
    P_slot = @isdefined(P) ? Type{P} : typeof(x)
    return Lifted{P_slot,N}(zero_dual(w, x))
end
# Tuple/NamedTuple primal: produce canonical V (a bare element-wise tuple of
# inner duals) per AGENTS.md tuple-lifting. Without this overload, the generic
# `zero_dual(w, ::Tuple)` returns `Dual(tuple, NoTangent)` which violates the
# canonical V invariant and breaks `tangent(::Lifted{<:Tuple})` (which calls
# `map(tangent, .value)` expecting a bare Tuple).
@inline function zero_lifted(w::Val{N}, x::Tuple) where {N}
    inner = ntuple(i -> zero_lifted_inner(w, x[i]), Val(length(x)))
    return Lifted{typeof(x),N}(inner)
end
@inline function zero_lifted(w::Val{N}, x::NamedTuple{names}) where {N,names}
    inner = ntuple(i -> zero_lifted_inner(w, values(x)[i]), Val(length(x)))
    return Lifted{typeof(x),N}(NamedTuple{names}(inner))
end
# Same disambiguation pattern for the recursive helper.
@inline zero_lifted_inner(::Val{0}, x::Tuple) = x
@inline zero_lifted_inner(::Val{0}, x::NamedTuple) = x
@inline zero_lifted_inner(::Val{0}, x) = x
@inline zero_lifted_inner(w::Val{N}, x) where {N} = zero_dual(w, x)
@inline zero_lifted_inner(w::Val{N}, x::Tuple) where {N} = ntuple(
    i -> zero_lifted_inner(w, x[i]), Val(length(x))
)
@inline zero_lifted_inner(w::Val{N}, x::NamedTuple{names}) where {N,names} = NamedTuple{
    names
}(
    ntuple(i -> zero_lifted_inner(w, values(x)[i]), Val(length(x)))
)

"""
    uninit_lifted(::Val{N}, x)

Layer-3 seed factory for uninitialised slots. See [`zero_lifted`](@ref).
"""
@inline uninit_lifted(::Val{0}, x) = x
@inline uninit_lifted(::Val{0}, x::Type) = x
@inline function uninit_lifted(w::Val{N}, x) where {N}
    zd = uninit_dual(w, x)
    InnerT = dual_type(w, typeof(x))
    return if typeof(zd) === InnerT
        Lifted{typeof(x),N,InnerT}(zd)
    else
        Lifted{typeof(x),N}(primal(zd), tangent(zd))
    end
end
@inline function uninit_lifted(w::Val{N}, x::Type{P}) where {N,P}
    P_slot = @isdefined(P) ? Type{P} : typeof(x)
    return Lifted{P_slot,N}(uninit_dual(w, x))
end

"""
    randn_lifted(::Val{N}, rng, x)

Layer-3 seed factory with random partials. See [`zero_lifted`](@ref).
"""
@inline randn_lifted(::Val{0}, ::AbstractRNG, x) = x
@inline randn_lifted(::Val{0}, ::AbstractRNG, x::Type) = x
@inline function randn_lifted(w::Val{N}, rng::AbstractRNG, x) where {N}
    zd = randn_dual(w, rng, x)
    InnerT = dual_type(w, typeof(x))
    return if typeof(zd) === InnerT
        Lifted{typeof(x),N,InnerT}(zd)
    else
        Lifted{typeof(x),N}(primal(zd), tangent(zd))
    end
end
@inline function randn_lifted(w::Val{N}, rng::AbstractRNG, x::Type{P}) where {N,P}
    P_slot = @isdefined(P) ? Type{P} : typeof(x)
    return Lifted{P_slot,N}(randn_dual(w, rng, x))
end
