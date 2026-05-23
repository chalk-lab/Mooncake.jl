#
# Performance-only rules. These should be able to be removed, and everything still works,
# just a bit slower. The effect of these is typically to remove many nodes from the tape.
# Ideally, it would be the case that acitivty analysis eliminates any run-time improvements
# that these rules provide. Possibly they would still be useful in order to avoid having to
# deduce that these bits of code are inactive though.
#

@zero_derivative DefaultCtx Tuple{typeof(in),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(iszero),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(isempty),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(isbitstype),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(sizeof),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(promote_type),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.promote_op),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.elsize),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Core.Compiler.sizeof_nothrow),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_haspadding),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_nfields),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_pointerfree),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_alignment),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.datatype_fielddesc_type),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(LinearAlgebra.chkstride1),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Threads.nthreads),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.depwarn),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.reduced_indices),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.check_reducedims),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.throw_boundserror),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.Broadcast.eltypes),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(Base.eltype),Vararg}
# Debug verification helpers are diagnostic-only. They should execute at primal time, but
# AD should never propagate derivatives through them in any context.
@zero_derivative MinimalCtx Tuple{typeof(verify_args),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(verify_dual_inputs),Tuple}
@zero_derivative MinimalCtx Tuple{typeof(verify_dual_output),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(verify_dual_value),Dual}
@zero_derivative MinimalCtx Tuple{typeof(verify_rvs_input),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(verify_rvs_output),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(verify_fwds_inputs),Any,Tuple}
@zero_derivative MinimalCtx Tuple{typeof(verify_fwds_output),Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(verify_fwds),CoDual}
@zero_derivative MinimalCtx Tuple{typeof(Base.padding),DataType}
@zero_derivative MinimalCtx Tuple{typeof(Base.padding),DataType,Int}
@zero_derivative MinimalCtx Tuple{Type,TypeVar,Type}

# Required to avoid an ambiguity.
@zero_derivative MinimalCtx Tuple{Type{Symbol},TypeVar,Type}

@static if VERSION >= v"1.11-"
    @zero_derivative MinimalCtx Tuple{typeof(Random.hash_seed),Vararg}
    @zero_derivative MinimalCtx Tuple{typeof(Base.dataids),Memory}
end

# Workaround: compiling `Base.unalias` through the primal-mode forward AD produces
# an OpaqueClosure that segfaults non-deterministically inside broadcast over NDual
# containers. Root cause undiagnosed; making it a forward-mode primitive that runs
# `mightalias` on primals sidesteps the broken codegen path.
@is_primitive MinimalCtx ForwardMode Tuple{typeof(Base.unalias),Any,Any}
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(Base.unalias),Any,Any}}) = true
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.unalias),N},
    dest::Mooncake.Lifted{<:Any,N},
    src::Mooncake.Lifted{S,N},
) where {S,N}
    d = primal(dest)
    s = primal(src)
    if d isa AbstractArray && s isa AbstractArray && Base.mightalias(d, s)
        return Mooncake.Lifted{S,N}(copy(_unlift(src)))
    end
    return src
end

"""
    stop_gradient(x)

Returns `x` with zero gradient. Gradients will not propagate through `x` in the reverse
pass. In the forward pass, `x` is returned unchanged.

To stop gradients through multiple values at once, pack them into a tuple:
`stop_gradient((x, y, z))`.

This is analogous to `tf.stop_gradient` in TensorFlow and `jax.lax.stop_gradient` in JAX.

!!! warning
    Mooncake requires that aliased primals have aliased fdatas (the "aliasing invariant"):
    `primal(a) === primal(b)` implies `fdata(a) === fdata(b)`. `stop_gradient`
    deliberately breaks this — the returned CoDual has `primal(y) === primal(x)` but
    `fdata(y) = _copy(fdata(x))` — so that downstream gradient accumulation into `y` does
    not affect `x`. This will produce incorrect gradients if the output is mutated in-place
    and `x` is subsequently read (or vice versa), because the two fdata buffers diverge.
    For example:
    ```julia
    function f(x)
        y = stop_gradient(x)  # primal(y) === x, but fdata(y) ≠ fdata(x)
        y[1] = 2.0            # mutates x[1], but tangent goes into fdata(y)
        return x[1] + x[2]   # reads fdata(x), which is now out of sync with fdata(y)
    end
    ```
    See https://github.com/chalk-lab/Mooncake.jl/issues/1081 for more details.

# Examples

```jldoctest
julia> using Mooncake

julia> f(x) = x[1] * Mooncake.stop_gradient(x)[2]
f (generic function with 1 method)

julia> cache = Mooncake.prepare_gradient_cache(f, [3.0, 4.0]);

julia> _, (_, g) = Mooncake.value_and_gradient!!(cache, f, [3.0, 4.0]);

julia> g  # g[2] == 0: gradient through x[2] inside stop_gradient is blocked
2-element Vector{Float64}:
 4.0
 0.0
```
"""
stop_gradient(x) = x

@is_primitive MinimalCtx Tuple{typeof(stop_gradient),Any}

@inline function frule!!(
    ::Mooncake.Lifted{typeof(stop_gradient),N}, x::Mooncake.Lifted
) where {N}
    return zero_lifted(Val(N), primal(x))
end
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(stop_gradient),Any}}) = true

function rrule!!(::CoDual{typeof(stop_gradient)}, x::CoDual)
    # Copy fdata so that in-place gradient accumulation into the output does not
    # affect the input's fdata (i.e., avoids aliasing of tangent storage).
    y = CoDual(primal(x), _copy(tangent(x)))
    lzr = lazy_zero_rdata(primal(x))
    stop_gradient_pb!!(_) = (NoRData(), instantiate(lzr))
    return y, stop_gradient_pb!!
end

"""
    lgetfield(x, f::Val)

An implementation of `getfield` in which the field `f` is specified statically via a
`Val`. This enables the implementation to be type-stable even when it is not
possible to constant-propagate `f`. Moreover, it enables the pullback to also be type-stable.

It will always be the case that
```julia
getfield(x, :f) === lgetfield(x, Val(:f))
getfield(x, 2) === lgetfield(x, Val(2))
```

This approach is identical to the one taken by `Zygote.jl` to circumvent the same problem.
`Zygote.jl` calls the function `literal_getfield`, while we call it `lgetfield`.
"""
lgetfield(x, ::Val{f}) where {f} = getfield(x, f)

@is_primitive MinimalCtx Tuple{typeof(lgetfield),Any,Val}
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(lgetfield),Any,Any}}) = true
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(lgetfield),Any,Any,Any}}) = true

# `lgetfield(x, Val(f), Val(order))` lifts the order arg into `Val`; the
# downstream `getfield(x, f, order)` call expects the raw value
# (`Symbol` for ordering, `Bool` for boundscheck). `_lgetfield_extras`
# unwraps each Lifted-wrapped Val back to its bare value for the
# arity-3 path; arity-2 returns an empty tuple.
@inline _unwrap_lifted_val(::Mooncake.Lifted{Val{x}}) where {x} = x
@inline _lgetfield_extras(extras::Vararg{Mooncake.Lifted,M}) where {M} = ntuple(
    i -> _unwrap_lifted_val(extras[i]), Val(M)
)

# Split-by-V Lifted-typed lgetfield bodies. Each body matches one inner-V
# shape and handles both arity-2 (name only) and arity-3 (name + order) via
# Vararg-tail unification. Each body extracts the field name from
# `Val{f}` in the where clause so `getfield(x, f, ...)` receives the bare
# Int/Symbol.
#
# Wrapper-exception V Dual{P,NoTangent}: parent has no tangent.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f}},
    extras::Vararg{Mooncake.Lifted,M},
) where {N,P,V_x<:Dual{P,NoTangent},f,M}
    return zero_lifted(Val(N), getfield(primal(x), f, _lgetfield_extras(extras...)...))
end
# Wrapper-exception V Dual{P,T<:NTangent}: standard primal+tangent field access.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f}},
    extras::Vararg{Mooncake.Lifted,M},
) where {N,P,T<:NTangent,V_x<:Dual{P,T},f,M}
    p_extras = _lgetfield_extras(extras...)
    y = getfield(primal(x), f, p_extras...)
    if tangent_type(_typeof(y)) === NoTangent
        return zero_lifted(Val(N), y)
    end
    return Mooncake.Lifted{_typeof(y),N}(y, _get_tangent_field(tangent(x), f))
end
# Tuple / NamedTuple bare element-wise V: field is itself canonical inner V.
# Specialised for Tuple/NamedTuple primal P to propagate fieldtype(P, field).
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f},N},
    extras::Vararg{Mooncake.Lifted,M},
) where {P<:Union{Tuple,NamedTuple},N,V_x<:Union{Tuple,NamedTuple},f,M}
    field_val = getfield(Mooncake._unlift(x), f, _lgetfield_extras(extras...)...)
    return _wrap_rule_result(fieldtype(P, f), Val(N), field_val)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f}},
    extras::Vararg{Mooncake.Lifted,M},
) where {N,P,V_x<:Union{Tuple,NamedTuple},f,M}
    field_val = getfield(Mooncake._unlift(x), f, _lgetfield_extras(extras...)...)
    return _wrap_rule_result(Val(N), field_val)
end
# SplitDual V: project canonical V's field directly.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f}},
    ::Vararg{Mooncake.Lifted,M},
) where {N,P,V_x<:Mooncake.SplitDual,f,M}
    return _wrap_rule_result(Val(N), getfield(Mooncake._unlift(x).canonical, f))
end
# AbstractArray{<:NDual} wrappers (Diagonal, Adjoint, SubArray, Memory, Array,
# …): structural lift places canonical V at differentiable leaf fields;
# non-differentiable fields are bare values. The `MemoryRef{<:NDual}` shape
# (returned by `getfield(::Memory, :ref)` and `getfield(::Array, :ref)`)
# is also canonical V — `MemoryRef` is not `<:AbstractArray`, so we list
# it explicitly in the post-`getfield` type check.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f}},
    extras::Vararg{Mooncake.Lifted,M},
) where {N,P,V_x<:AbstractArray{<:Union{NDual,Complex{<:NDual}}},f,M}
    field_val = getfield(Mooncake._unlift(x), f, _lgetfield_extras(extras...)...)
    if field_val isa Union{
        NDual,
        Complex{<:NDual},
        AbstractArray{<:NDual},
        AbstractArray{<:Complex{<:NDual}},
        MemoryRef{<:NDual},
        MemoryRef{<:Complex{<:NDual}},
    }
        return _wrap_rule_result(Val(N), field_val)
    end
    return zero_lifted(Val(N), field_val)
end
# Broadcasted / Extruded V.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N,V_x},
    ::Mooncake.Lifted{Val{f}},
    extras::Vararg{Mooncake.Lifted,M},
) where {N,P,V_x<:Union{Base.Broadcast.Extruded,Base.Broadcast.Broadcasted},f,M}
    field_val = getfield(Mooncake._unlift(x), f, _lgetfield_extras(extras...)...)
    return if _has_ndual(field_val)
        _wrap_rule_result(Val(N), field_val)
    else
        zero_lifted(Val(N), field_val)
    end
end
_get_tangent_field(f::Union{NamedTuple,Tuple}, name) = getfield(f, name)
_get_tangent_field(f::Union{NamedTuple,Tuple}, name, inbounds) = getfield(f, name, inbounds)
_get_tangent_field(f::Union{Tangent,MutableTangent}, name) = val(getfield(f.fields, name))
function _get_tangent_field(f::Union{Tangent,MutableTangent}, name, inbounds)
    return val(getfield(f.fields, name, inbounds))
end
# When the struct tangent is NoTangent (e.g. a non-differentiable type captured inside
# another struct), field access also contributes no derivative.
_get_tangent_field(::NoTangent, _) = NoTangent()
_get_tangent_field(::NoTangent, _, _) = NoTangent()
# Raw-array tangent shape (`Vector{Any}`, `Vector{NoTangent}` etc.) arises
# when `_lgetfield_impl` recurses into the NTangent lanes of a non-NDual
# array primal. The tangent has the same struct fields as the primal Array,
# so `getfield(tangent, name)` produces the matching per-field tangent shape
# directly (e.g. `tangent.ref::MemoryRef{T}` matches `tangent_type(::Array
# .ref) === MemoryRef{T}`).
_get_tangent_field(t::AbstractArray, name) = getfield(t, name)
_get_tangent_field(t::AbstractArray, name, inbounds) = getfield(t, name, inbounds)
# MemoryRef tangent shape arises from `lgetfield(::Array, :ref)` paths. The
# `:mem` field returns the underlying `Memory` tangent (matching shape). The
# `:ptr_or_offset` field is a `Ptr{Nothing}` in the primal — but its tangent
# is `Ptr{NoTangent}` per the Mooncake convention (mirrors the bare-Dual
# `lgetfield(::Dual{<:MemoryRef, <:MemoryRef}, :ptr_or_offset)` rule which
# also `bitcast`s the value).
@inline _get_tangent_field(t::MemoryRef, name::Symbol) =
    name === :ptr_or_offset ? bitcast(Ptr{NoTangent}, t.ptr_or_offset) : getfield(t, name)
@inline _get_tangent_field(t::MemoryRef, name::Symbol, inbounds) =
    if name === :ptr_or_offset
        bitcast(Ptr{NoTangent}, t.ptr_or_offset)
    else
        getfield(t, name, inbounds)
    end
function _get_tangent_field(f::NTangent, name)
    return NTangent(map(t -> _get_tangent_field(t, name), f.lanes))
end
function _get_tangent_field(f::NTangent, name, inbounds)
    return NTangent(map(t -> _get_tangent_field(t, name, inbounds), f.lanes))
end

# NTangent wraps the inner tangent; propagate `set_tangent_field!` into each
# lane. When the new value `x` is itself NTangent-wrapped, apply per-lane;
# otherwise broadcast the same `x` across all lanes. Lives here rather than
# in `tangents.jl` because `NTangent` is defined later (in
# `tangents/dual.jl`).
@inline function set_tangent_field!(t::NTangent, i::Union{Int,Symbol}, x)
    for lane in t.lanes
        set_tangent_field!(lane, i, x)
    end
    return x
end
@inline function set_tangent_field!(
    t::NTangent{Vt}, i::Union{Int,Symbol}, x::NTangent{Vx}
) where {Vt<:Tuple,Vx<:Tuple}
    @inbounds for n in eachindex(t.lanes)
        set_tangent_field!(t.lanes[n], i, x.lanes[n])
    end
    return x
end

@inline function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{P,F}, ::CoDual{Val{f}}
) where {P,F<:StandardFDataType,f}
    pb!! = if ismutabletype(P)
        dx = tangent(x)
        function mutable_lgetfield_pb!!(dy)
            increment_field_rdata!(dx, dy, Val{f}())
            return NoRData(), NoRData(), NoRData()
        end
    else
        dx_r = lazy_zero_rdata(primal(x))
        field = Val{f}()
        function immutable_lgetfield_pb!!(dy)
            return NoRData(), increment_field!!(instantiate(dx_r), dy, field), NoRData()
        end
    end
    y = CoDual(getfield(primal(x), f), _get_fdata_field(primal(x), tangent(x), f))
    return y, pb!!
end

@unstable @inline _get_fdata_field(_, t::Union{Tuple,NamedTuple}, f) = getfield(t, f)
@unstable @inline _get_fdata_field(_, data::FData, f) = val(getfield(data.data, f))
@unstable @inline _get_fdata_field(primal, ::NoFData, f) = uninit_fdata(getfield(primal, f))
@unstable @inline _get_fdata_field(_, t::MutableTangent, f) = fdata(
    val(getfield(t.fields, f))
)

increment_field_rdata!(dx::MutableTangent, ::NoRData, ::Val) = dx
increment_field_rdata!(dx::NoFData, ::NoRData, ::Val) = dx
function increment_field_rdata!(dx::T, dy_rdata, ::Val{f}) where {T<:MutableTangent,f}
    set_tangent_field!(dx, f, increment_rdata!!(get_tangent_field(dx, f), dy_rdata))
    return dx
end

#
# lgetfield with order argument
#

# This is largely copy + pasted from the above. Attempts were made to refactor to avoid
# code duplication, but it wound up not being any cleaner than this copy + pasted version.

@is_primitive MinimalCtx Tuple{typeof(lgetfield),Any,Val,Val}

@inline function rrule!!(
    ::CoDual{typeof(lgetfield)}, x::CoDual{P,F}, ::CoDual{Val{f}}, ::CoDual{Val{order}}
) where {P,F<:StandardFDataType,f,order}
    pb!! = if ismutabletype(P)
        dx = tangent(x)
        function mutable_lgetfield_pb!!(dy)
            increment_field_rdata!(dx, dy, Val{f}())
            return NoRData(), NoRData(), NoRData(), NoRData()
        end
    else
        dx_r = lazy_zero_rdata(primal(x))
        function immutable_lgetfield_pb!!(dy)
            tmp = increment_field!!(instantiate(dx_r), dy, Val{f}())
            return NoRData(), tmp, NoRData(), NoRData()
        end
    end
    y = CoDual(getfield(primal(x), f, order), _get_fdata_field(primal(x), tangent(x), f))
    return y, pb!!
end

@is_primitive MinimalCtx Tuple{typeof(lsetfield!),Any,Any,Any}
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(lsetfield!),Any,Any,Any}}) = true
@inline _ndual_arg_unwrap(x::Dual) = primal(x)
@inline _ndual_arg_unwrap(x::Tuple) = map(_ndual_arg_unwrap, x)
@inline _ndual_arg_unwrap(x) = x
# Direct Lifted bodies per inner V shape. Wrapper-exception value V (slot
# inner V is `Dual{P, T<:StandardTangentType}`) splits on T: NoTangent
# does primal-only setfield!; NTangent additionally updates the tangent
# field. The per-lane tangent of `x` is recovered from public accessors
# (`tangent(x::Lifted, lane)`) — no Dual{P_x, NTangent} bridge needed.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lsetfield!),N},
    value::Mooncake.Lifted{P,N,V},
    ::Mooncake.Lifted{Val{f}},
    x::Mooncake.Lifted,
) where {N,P,V<:Dual{P,NoTangent},f}
    setfield!(primal(value), f, primal(x))
    return x
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lsetfield!),N},
    value::Mooncake.Lifted{P,N,V},
    ::Mooncake.Lifted{Val{f}},
    x::Mooncake.Lifted,
) where {N,P,T<:NTangent,V<:Dual{P,T},f}
    setfield!(primal(value), f, primal(x))
    tx = Mooncake.NTangent(ntuple(lane -> Mooncake.tangent(x, lane), Val(N)))
    set_tangent_field!(tangent(value), f, tx)
    return x
end
# Canonical-NDual value V (AbstractArray{<:NDual}): bare lsetfield! on
# the array; NDual elements carry tangent so no separate tangent update.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lsetfield!),N},
    value::Mooncake.Lifted{P,N,V},
    name::Mooncake.Lifted,
    x::Mooncake.Lifted,
) where {N,P,V<:AbstractArray{<:Mooncake.Nfwd.NDual}}
    bare_value = Mooncake._unlift(value)
    bare_name = Mooncake._unlift(name)
    bare_x = Mooncake._unlift(x)
    result = lsetfield!(bare_value, primal(bare_name), _ndual_arg_unwrap(bare_x))
    return _wrap_rule_result(Val(N), result)
end
# SplitDual value V: rebuild the canonical NamedTuple in place via an
# `ntuple` rebuild that preserves zero allocation.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lsetfield!),N},
    value::Mooncake.Lifted{P,N,V},
    name::Mooncake.Lifted{<:Val{f}},
    x::Mooncake.Lifted,
) where {N,P,NT<:NamedTuple,V<:Mooncake.SplitDual{NT},f}
    bare_value = Mooncake._unlift(value)
    bare_x = Mooncake._unlift(x)
    nt = getfield(bare_value, :canonical)
    i = Base.fieldindex(NT, f)
    new_nt = NT(ntuple(n -> n == i ? bare_x : nt[n], fieldcount(NT)))
    setfield!(bare_value, :canonical, new_nt)
    return _wrap_rule_result(Val(N), bare_x)
end
@inline function rrule!!(
    ::CoDual{typeof(lsetfield!)}, value::CoDual{P,F}, name::CoDual, x::CoDual
) where {P,F<:StandardFDataType}
    return lsetfield_rrule(value, name, x)
end

# `lsetfield_frule` split by tangent shape — after Phase 6 StandardTangentType
# narrowing, the reachable T are `NoTangent` and `NTangent`. `NoTangent` means
# the primal has no tangent to update, so we skip `set_tangent_field!`; `NTangent`
# (and other non-NoTangent T) needs both the primal mutation and the tangent
# update. Dispatch on T eliminates the runtime `T !== NoTangent` check.
function lsetfield_frule(
    value::Dual{P,NoTangent}, ::Dual{Val{name}}, x::Dual
) where {P,name}
    setfield!(primal(value), name, primal(x))
    return x
end
function lsetfield_frule(value::Dual{P,T}, ::Dual{Val{name}}, x::Dual) where {P,T,name}
    setfield!(primal(value), name, primal(x))
    set_tangent_field!(tangent(value), name, tangent(x))
    return x
end

# Bare-Dual lsetfield! with a structural-lift NamedTuple value. Arises in
# HVP cache-update paths (e.g. `lazy_rule.rule = build_rrule(...)`) where
# the inner build_rrule output is structurally lifted as a NamedTuple of
# field Duals rather than wrapped in a single `Dual{P_field, T_field}`.
# Build the field's primal via `_new_`, materialise the structural tangent
# from each field's lane-1 / wrapper-exception tangent, and delegate to
# the standard `lsetfield_frule`.
@inline function frule!!(
    ::Dual{typeof(lsetfield!),NoTangent},
    value::Dual{P,T},
    name::Dual{Val{f},NoTangent},
    x::NamedTuple,
) where {P,T,f}
    P_field = fieldtype(P, f)
    field_primals = map(primal, Tuple(x))
    new_value = Mooncake._new_(P_field, field_primals...)
    # Lane-1 tangent extraction — fields whose tangent is `NTangent{Tuple{T}}`
    # (single-lane chunk-mode wrapper) unwrap to their lane-1 element so the
    # structural tangent has the canonical per-field tangent_type.
    field_tangents = map(d -> begin
        t = tangent(d)
        t isa Mooncake.NTangent ? t.lanes[1] : t
    end, Tuple(x))
    T_field = Mooncake.tangent_type(P_field)
    new_tangent = if T_field === NoTangent || all(t -> t isa NoTangent, field_tangents)
        NoTangent()
    else
        T_field(NamedTuple{fieldnames(P_field)}(field_tangents))
    end
    new_x = Dual(new_value, new_tangent)
    return lsetfield_frule(value, name, new_x)
end

function lsetfield_rrule(
    value::CoDual{P,F}, ::CoDual{Val{name}}, x::CoDual
) where {P,F,name}
    save = isdefined(primal(value), name)
    old_x = save ? getfield(primal(value), name) : nothing
    old_dx = if F == NoFData
        NoFData()
    else
        save ? get_tangent_field(tangent(value), name) : nothing
    end
    dvalue = tangent(value)
    pb!! = if F == NoFData
        function __setfield!_pullback(dy)
            old_x !== nothing && lsetfield!(primal(value), Val(name), old_x)
            return NoRData(), NoRData(), NoRData(), dy
        end
    else
        function setfield!_pullback(dy)
            new_dx = increment!!(dy, rdata(get_tangent_field(dvalue, name)))
            old_x !== nothing && lsetfield!(primal(value), Val(name), old_x)
            old_x !== nothing && set_tangent_field!(dvalue, name, old_dx)
            return NoRData(), NoRData(), NoRData(), new_dx
        end
    end
    yf = if F == NoFData
        NoFData()
    else
        fdata(set_tangent_field!(dvalue, name, zero_tangent(primal(x), tangent(x))))
    end
    y = CoDual(lsetfield!(primal(value), Val(name), primal(x)), yf)
    return y, pb!!
end

@static if VERSION < v"1.11"
    @is_primitive MinimalCtx Tuple{typeof(copy),Dict}
    # The Lifted-typed body computes the result independently from the inner V.
    @inline function frule!!(
        ::Mooncake.Lifted{typeof(copy),N}, a::Mooncake.Lifted{<:Dict}
    ) where {N}
        inner = Mooncake._unlift(a)
        return Mooncake.Lifted{_typeof(primal(inner)),N}(
            copy(primal(inner)), _copy_dict_tangent(tangent(inner))
        )
    end
    @inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(copy),<:Dict}}) = true
    function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Dict})
        dx = tangent(a)
        t = dx.fields
        new_fields = typeof(t)((
            copy(t.slots), copy(t.keys), copy(t.vals), tuple_fill(NoTangent(), Val(5))...
        ))
        dy = MutableTangent(new_fields)
        y = CoDual(copy(primal(a)), dy)
        function copy_pullback!!(::NoRData)
            increment!!(dx, dy)
            return NoRData(), NoRData()
        end
        return y, copy_pullback!!
    end
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:misc})

    # Data which needs to not be GC'd.
    _x = Ref(5.0)
    _dx = Ref(4.0)
    memory = Any[_x, _dx]

    specific_test_cases = Any[
        # stop_gradient: value passes through, gradients are zeroed out.
        # interface_only=true because the rule intentionally returns zero gradient,
        # which does not match the finite-difference Jacobian of the primal (identity).
        (true, :none, nothing, stop_gradient, 5.0),
        (true, :none, nothing, stop_gradient, randn(4)),
        (true, :none, nothing, stop_gradient, (3.0, 4.0)),

        # Rules to avoid pointer type conversions.
        (
            true,
            :stability,
            nothing,
            +,
            CoDual(
                bitcast(Ptr{Float64}, pointer_from_objref(_x)),
                bitcast(Ptr{Float64}, pointer_from_objref(_dx)),
            ),
            2,
        ),

        # Lack of activity-analysis rules:
        (false, :stability_and_allocs, nothing, Base.elsize, randn(5, 4)),
        (false, :stability_and_allocs, nothing, Base.elsize, view(randn(5, 4), 1:2, 1:2)),
        (false, :stability_and_allocs, nothing, Core.Compiler.sizeof_nothrow, Float64),
        (false, :stability_and_allocs, nothing, Base.datatype_haspadding, Float64),

        # Performance-rules that would ideally be completely removed.
        (false, :stability_and_allocs, nothing, in, 5.0, randn(4)),
        (false, :stability_and_allocs, nothing, iszero, 5.0),
        (false, :stability_and_allocs, nothing, isempty, randn(5)),
        (false, :stability_and_allocs, nothing, isbitstype, Float64),
        (false, :stability_and_allocs, nothing, sizeof, Float64),
        (false, :stability_and_allocs, nothing, promote_type, Float64, Float64),
        (false, :stability_and_allocs, nothing, LinearAlgebra.chkstride1, randn(3, 3)),
        (
            false,
            :stability_and_allocs,
            nothing,
            LinearAlgebra.chkstride1,
            randn(3, 3),
            randn(2, 2),
        ),
        (false, :allocs, nothing, Threads.nthreads),
        (false, :none, nothing, Base.eltype, randn(1)),
        (false, :none, nothing, Base.padding, @NamedTuple{a::Float64}),
        (false, :none, nothing, Base.padding, @NamedTuple{a::Float64}, 1),

        # Literal replacement for setfield!.
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            MutableFoo(5.0, [1.0, 2.0]),
            Val(:a),
            4.0,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            FullyInitMutableStruct(5.0, [1.0, 2.0]),
            Val(:y),
            [1.0, 3.0, 4.0],
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            NonDifferentiableFoo(5, false),
            Val(:x),
            4,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            lsetfield!,
            NonDifferentiableFoo(5, false),
            Val(:y),
            true,
        ),
    ]

    # Some specific test cases for lgetfield to test the basics.
    specific_lgetfield_test_cases = Any[

        # Tuple
        (false, :stability_and_allocs, nothing, lgetfield, (5.0, 4), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (5.0, 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (1, 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, ((), 4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (randn(2),), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (randn(2), 5), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (randn(2), 5), Val(2)),

        # NamedTuple
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(:a)),
        (false, :stability_and_allocs, nothing, lgetfield, (a=5.0, b=4), Val(:b)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2),), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2),), Val(:y)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(1)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(2)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(:y)),
        (false, :stability_and_allocs, nothing, lgetfield, (y=randn(2), x=5), Val(:x)),

        # structs
        (false, :stability_and_allocs, nothing, lgetfield, 1:5, Val(:start)),
        (false, :stability_and_allocs, nothing, lgetfield, 1:5, Val(:stop)),
        # `getfield` primal is ~1–2 ns; rule overhead is ~100–500 ns. ub=750 gives margin.
        (true, :none, (lb=1e-3, ub=750), lgetfield, StructFoo(5.0), Val(:a)),
        (false, :none, (lb=1e-3, ub=750), lgetfield, StructFoo(5.0, randn(5)), Val(:a)),
        (false, :none, (lb=1e-3, ub=200), lgetfield, StructFoo(5.0, randn(5)), Val(:b)),
        (true, :none, (lb=1e-3, ub=750), lgetfield, StructFoo(5.0), Val(1)),
        (false, :none, (lb=1e-3, ub=750), lgetfield, StructFoo(5.0, randn(5)), Val(1)),
        (false, :none, (lb=1e-3, ub=750), lgetfield, StructFoo(5.0, randn(5)), Val(2)),

        # mutable structs
        (true, :none, (lb=1e-3, ub=350), lgetfield, MutableFoo(5.0), Val(:a)),
        (false, :none, (lb=1e-3, ub=350), lgetfield, MutableFoo(5.0, randn(5)), Val(:b)),
        (false, :none, nothing, lgetfield, UInt8, Val(:name)),
        (false, :none, nothing, lgetfield, UInt8, Val(:super)),
        (true, :none, nothing, lgetfield, UInt8, Val(:layout)),
        (false, :none, nothing, lgetfield, UInt8, Val(:hash)),
        (false, :none, nothing, lgetfield, UInt8, Val(:flags)),
    ]

    # Create `lgetfield` tests for each type in TestTypes for broader coverage.
    general_lgetfield_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        _, primal = TestTypes.instantiate((interface_only, P, args))
        names = fieldnames(P)[1:length(args)] # only query fields which get initialised
        return Any[
            (interface_only, :none, nothing, lgetfield, primal, Val(name)) for name in names
        ]
    end

    # lgetfield has both 3 and 4 argument forms. Create test cases for both scenarios.
    all_lgetfield_test_cases = Any[
        (case..., order...) for
        case in vcat(specific_lgetfield_test_cases, general_lgetfield_test_cases...) for
        order in Any[(), (Val(false),)]
    ]

    # Create `lsetfield!` tests for each type in TestTypes for broader coverage.
    general_lsetfield_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        ismutabletype(P) || return Any[]
        _, primal = TestTypes.instantiate((interface_only, P, args))
        names = fieldnames(P)[1:length(args)] # only query fields which get initialised
        return Any[
            (interface_only, :none, nothing, lsetfield!, primal, Val(name), args[n]) for
            (n, name) in enumerate(names)
        ]
    end

    test_cases = vcat(
        specific_test_cases, all_lgetfield_test_cases..., general_lsetfield_test_cases...
    )
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:misc})
    test_cases = Any[
        (false, :none, nothing, x -> copy(Dict("A" => x[1], "B" => x[2]))["A"], (5.0, 5.0)),
        (false, :none, nothing, copy, Dict{Any,Any}("A" => [5.0], [3.0] => 5.0)),
        (false, :none, nothing, () -> copy(Set())),
    ]
    return test_cases, Any[]
end
