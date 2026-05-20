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

# `lgetfield` implementation kernels (no `Dual{typeof(lgetfield)}` arg). The
# `Lifted`-typed body below dispatches the runtime inner V (`Dual{P, T}`,
# `Tuple`, `NamedTuple`) into the matching kernel.
@inline function _lgetfield_impl(x::Dual{P,T}, ::Val{f}) where {P,T<:StandardTangentType,f}
    primal_field = getfield(primal(x), f)
    # Short-circuit on either parent-NoTangent or field-NoTangent (see the
    # 4-arg overload below for rationale).
    if tangent_type(P) === NoTangent || tangent_type(_typeof(primal_field)) === NoTangent
        return uninit_dual(primal_field)
    else
        return _dual_or_ndual(primal_field, _get_tangent_field(tangent(x), f))
    end
end
# Bare Tuple/NamedTuple with NDual elements — tangent info lives inside each element.
@inline _lgetfield_impl(x::T, ::Val{f}) where {T<:Union{Tuple,NamedTuple},f} = getfield(
    x, f
)
# `SplitDual{V}` mutable-struct canonical lift: project the canonical V's
# field directly. The field's value is the canonical V of the primal
# field type (e.g., `Vector{NDual{T,1}}` for an Array field), so its
# tangent info lives inside the NDual elements.
@inline _lgetfield_impl(x::Mooncake.SplitDual, ::Val{f}) where {f} = getfield(
    x.canonical, f
)
# Bare `AbstractArray{<:NDual}` wrappers (Diagonal, Adjoint, SubArray, …) — the
# structural lift in `nfwd/NfwdMooncake.jl` places `NDual`/`Array{NDual}` at
# differentiable leaf fields (which are returned as-is, already canonical V).
# Non-differentiable fields (e.g. `SubArray`'s `:indices`, `:offset1`,
# `:stride1`) are bare values; lift them via `uninit_dual` so the slot V matches
# `dual_type(Val(1), fieldtype)` — mirrors the `Dual{P, NoTangent}` field path
# in the `Dual{P, T}` overload above.
@inline function _lgetfield_impl(
    x::AbstractArray{<:Union{NDual,Complex{<:NDual}}}, ::Val{f}
) where {f}
    field_val = getfield(x, f)
    return if field_val isa Union{
        NDual,Complex{<:NDual},AbstractArray{<:NDual},AbstractArray{<:Complex{<:NDual}}
    }
        field_val
    else
        uninit_dual(field_val)
    end
end
@inline function _lgetfield_impl(
    x::Union{Base.Broadcast.Extruded,Base.Broadcast.Broadcasted}, ::Val{f}
) where {f}
    field_val = getfield(x, f)
    return _has_ndual(field_val) ? field_val : uninit_dual(field_val)
end

@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N}, x::Mooncake.Lifted, name::Mooncake.Lifted{<:Val}
) where {N}
    bare_result = _lgetfield_impl(Mooncake._unlift(x), primal(name))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted{P,N},
    name::Mooncake.Lifted{Val{field},N},
) where {P<:Union{Tuple,NamedTuple},N,field}
    bare_result = _lgetfield_impl(Mooncake._unlift(x), primal(name))
    return _wrap_rule_result(fieldtype(P, field), Val(N), bare_result)
end
# Mixed dispatch fallback: Tuple/NamedTuple primal arrives as a `Lifted` slot
# while the function/index arrive as bare `Dual` (e.g. via the IR-emit constant
# path that uses `zero_dual` rather than `zero_lifted`). Unlift the slot to
# its bare inner V and delegate to the implementation kernel.
@inline function frule!!(
    ::Dual{typeof(lgetfield)}, x::Mooncake.Lifted{P}, ::Dual{Val{f}}
) where {P<:Union{Tuple,NamedTuple},f}
    return _lgetfield_impl(Mooncake._unlift(x), Val(f))
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
# Implementation kernels for `lgetfield(x, Val(f), Val(order))`.
@inline function _lgetfield_impl(
    x::Dual{P,<:StandardTangentType}, ::Val{f}, ::Val{order}
) where {P,f,order}
    primal_field = getfield(primal(x), f, order)
    # Short-circuit when *either* the parent `P` has no tangent, *or* the
    # specific field is non-differentiable
    # (`tangent_type(typeof(primal_field)) === NoTangent`). The narrower
    # field check covers struct/array primals like `Vector{Any}.size` where
    # the parent has a tangent shape but the specific field doesn't — avoids
    # recursing into `_get_tangent_field` on a bulk-array tangent that
    # doesn't structurally decompose.
    if tangent_type(P) === NoTangent || tangent_type(_typeof(primal_field)) === NoTangent
        return uninit_dual(primal_field)
    else
        return _dual_or_ndual(primal_field, _get_tangent_field(tangent(x), f))
    end
end
@inline _lgetfield_impl(x::Mooncake.SplitDual, ::Val{f}, ::Val{order}) where {f,order} = getfield(
    x.canonical, f
)
@inline _lgetfield_impl(x::T, ::Val{f}, ::Val{order}) where {T<:Union{Tuple,NamedTuple},f,order} = getfield(
    x, f, order
)

@inline function frule!!(
    ::Mooncake.Lifted{typeof(lgetfield),N},
    x::Mooncake.Lifted,
    name::Mooncake.Lifted{<:Val},
    order::Mooncake.Lifted{<:Val},
) where {N}
    bare_result = _lgetfield_impl(Mooncake._unlift(x), primal(name), primal(order))
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
# Bare-Dual 4-arg lgetfield: paired with the 3-arg `frule!!(::Dual{lgetfield},
# ::Lifted{P}, ::Dual{Val{f}})` form above. The IR-emit's mixed-dispatch
# path (bare-Dual function + Lifted struct + bare-Dual Val constants) can
# also produce a bare-Dual struct value at width 1 for wrapper-exception
# primals. `_lgetfield_impl(::Dual{P, T<:StandardTangentType}, ::Val{f},
# ::Val{order})` already handles both NTangent-wrapped and bare-T forms.
@inline function frule!!(
    ::Dual{typeof(lgetfield)}, x::Dual{P,T}, ::Dual{Val{f}}, ::Dual{Val{order}}
) where {P,T<:StandardTangentType,f,order}
    return _lgetfield_impl(x, Val(f), Val(order))
end
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
@inline function frule!!(
    ::Dual{typeof(lsetfield!)}, value::Dual{P,T}, name::Dual, x::Dual
) where {P,T<:StandardTangentType}
    return lsetfield_frule(value, name, x)
end
# IR-emit unwraps `Lifted` slot values to bare V's; for IEEEFloat fields
# this is `NDual{T, N}` (or `Vector{NDual}` / `Complex{NDual}` for container
# fields). Wrap into a width-N `Dual{P, NTangent{NTuple{N, ...}}}` form via
# `_ndual_to_dual_widthN` so `set_tangent_field!` hits the per-lane
# `NTangent{NTuple{N,T}}, name, NTangent{NTuple{N,Tx}}` overload — not
# the width-1 broadcast overload that duplicates lane 1.
@inline function frule!!(
    ::Dual{typeof(lsetfield!)},
    value::Dual{P,T},
    name::Dual,
    x::Union{
        NDual,Complex{<:NDual},AbstractArray{<:NDual},AbstractArray{<:Complex{<:NDual}}
    },
) where {P,T<:StandardTangentType}
    return lsetfield_frule(value, name, _ndual_to_dual_widthN(x))
end
@inline function frule!!(
    ::Dual{typeof(lsetfield!)}, value::Dual{P,T}, name::Dual, x::Tuple
) where {P,T<:StandardTangentType}
    return lsetfield_frule(value, name, _tuple_duals_to_dual(x))
end
# A bare-NDual-container `value` can arrive when the unlifted form is the
# canonical V (e.g. `Lifted{Vector{T<:IEEEFloat}, 1, Vector{NDual{T,1}}}`
# unlifts to bare `Vector{NDual}`). The Vector IS the lifted form — setting
# its `:ref` or `:size` field on the bare canonical V just modifies the
# lifted form in-place. No separate tangent update needed.
@inline function frule!!(
    ::Dual{typeof(lsetfield!)}, value::AbstractArray{<:Mooncake.Nfwd.NDual}, name::Dual, x
)
    return lsetfield!(value, primal(name), _ndual_arg_unwrap(x))
end
# `SplitDual{V}` lsetfield!: replace the canonical NamedTuple's field with
# the bare canonical-V form of `x`, mutating the SplitDual in place so
# aliased Lifted slots share the update (preserving primal aliasing
# semantics for mutable structs).
@inline function frule!!(
    ::Dual{typeof(lsetfield!)}, value::Mooncake.SplitDual{V}, name::Dual{Val{f}}, x
) where {V<:NamedTuple,f}
    nt = getfield(value, :canonical)
    new_nt = Base.setindex(nt, x, f)
    setfield!(value, :canonical, new_nt)
    return x
end
@inline _ndual_arg_unwrap(x::Dual) = primal(x)
@inline _ndual_arg_unwrap(x::Tuple) = map(_ndual_arg_unwrap, x)
@inline _ndual_arg_unwrap(x) = x
@inline function frule!!(
    f::Mooncake.Lifted{typeof(lsetfield!),N},
    value::Mooncake.Lifted,
    name::Mooncake.Lifted,
    x::Mooncake.Lifted,
) where {N}
    bare_result = frule!!(
        Mooncake._unlift(f),
        Mooncake._unlift(value),
        Mooncake._unlift(name),
        Mooncake._unlift(x),
    )
    P_out = __primal_type(_typeof(bare_result))
    return _wrap_rule_result(P_out, Val(N), bare_result)
end
# `_ndual_to_dual_lane1` bridges canonical width-1 `NDual` / `Complex{NDual}` /
# array containers into the legacy `Dual{P, T}` form expected by
# `lsetfield_frule`. For width-N (N >= 2) callers, `_ndual_to_dual_widthN`
# below produces an `NTangent`-wrapped tangent so the per-lane
# `set_tangent_field!(::NTangent{NTuple{N, T}}, name, ::NTangent{NTuple{N, Tx}})`
# overload fires (otherwise `set_tangent_field!` broadcasts lane-1 across
# all N lanes — wrong with distinct seeds).
@inline _ndual_to_dual_lane1(x::Dual) = x
@inline _ndual_to_dual_lane1(x::NDual) = Dual(primal(x), x.partials[1])
@inline _ndual_to_dual_lane1(x::Complex{<:NDual}) = Dual(
    complex(x.re.value, x.im.value), complex(x.re.partials[1], x.im.partials[1])
)
@inline _ndual_to_dual_lane1(x::AbstractArray{<:NDual}) = Dual(
    map(d -> d.value, x), map(d -> d.partials[1], x)
)
@inline _ndual_to_dual_lane1(x::AbstractArray{<:Complex{<:NDual}}) = Dual(
    map(z -> complex(z.re.value, z.im.value), x),
    map(z -> complex(z.re.partials[1], z.im.partials[1]), x),
)
# Width-N bridge: returns `Dual{P, NTangent{NTuple{N, T}}}` so
# `set_tangent_field!` dispatches to the per-lane overload.
@inline _ndual_to_dual_widthN(x::Dual) = x
@inline function _ndual_to_dual_widthN(x::NDual{T,N}) where {T,N}
    return Dual(primal(x), Mooncake.NTangent(x.partials))
end
@inline function _ndual_to_dual_widthN(x::Complex{NDual{T,N}}) where {T<:IEEEFloat,N}
    p = Complex(x.re.value, x.im.value)
    ts = ntuple(n -> Complex(x.re.partials[n], x.im.partials[n]), Val(N))
    return Dual(p, Mooncake.NTangent(ts))
end
@inline function _ndual_to_dual_widthN(x::AbstractArray{NDual{T,N}}) where {T,N}
    p = map(d -> d.value, x)
    ts = ntuple(n -> map(d -> d.partials[n], x), Val(N))
    return Dual(p, Mooncake.NTangent(ts))
end
@inline function _ndual_to_dual_widthN(
    x::AbstractArray{Complex{NDual{T,N}}}
) where {T<:IEEEFloat,N}
    p = map(c -> Complex(c.re.value, c.im.value), x)
    ts = ntuple(Val(N)) do n
        map(c -> Complex(c.re.partials[n], c.im.partials[n]), x)
    end
    return Dual(p, Mooncake.NTangent(ts))
end
@inline function _tuple_duals_to_dual(x::Tuple)
    ts = map(tangent, x)
    return Dual(map(primal, x), ts isa Tuple{Vararg{NoTangent}} ? NoTangent() : ts)
end
@inline function rrule!!(
    ::CoDual{typeof(lsetfield!)}, value::CoDual{P,F}, name::CoDual, x::CoDual
) where {P,F<:StandardFDataType}
    return lsetfield_rrule(value, name, x)
end

function lsetfield_frule(value::Dual{P,T}, ::Dual{Val{name}}, x::Dual) where {P,T,name}
    setfield!(primal(value), name, primal(x))
    T !== NoTangent && set_tangent_field!(tangent(value), name, tangent(x))
    return x
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
