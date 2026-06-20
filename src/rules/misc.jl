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

# `stop_gradient` zeros all derivative information. The result must still carry the canonical
# `dual_type(Val(Nw), P)` V (a `NoDual` paired with a differentiable primal violates coherence and
# breaks downstream field-access frules); `zero_lifted` builds that V with zero partials, which is
# exactly a blocked gradient.
function frule!!(::Lifted{typeof(stop_gradient),Nw}, x::Lifted) where {Nw}
    return zero_lifted(Val(Nw), primal(x))
end

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
# Extract the field's slot-level V from the parent V via `_get_lifted_field`.
# Forward-mode lift wrappers (`ImmutableDual`/`MutableDual`) and plain
# `Tuple`/`NamedTuple` V's are handled by the `_get_lifted_field` dispatch
# table below.
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw}, x::Lifted, ::Lifted{Val{f}}
) where {Nw,f}
    primal_field = getfield(primal(x), f)
    # A non-differentiable parent (`tangent(x) === NoDual()`) yields a non-differentiable field,
    # but its forward V is the field's *canonical* zero V — `NoDual` for the usual scalars, yet
    # `Vector{Any}` for a `SimpleVector` field (e.g. `lgetfield(::DataType, Val(:parameters))`).
    # `uninit_lifted` builds that canonical slot (mirrors the reverse non-diff `getfield` path).
    tangent(x) isa NoDual && return uninit_lifted(Val(Nw), primal_field)
    V_i = _get_lifted_field(tangent(x), f)
    return Lifted{typeof(primal_field),Nw}(primal_field, V_i)
end

# `Ref{P<:NDualEltype}` field read (`r[]`): rebuild the scalar inner V (`NDual` real /
# `Complex{NDual}` complex, via `_scalar_ndual`) from the primal value and the `NDualRef` parallel
# partials buffer. The generic frule above routes ImmutableDual/MutableDual/Tuple/NamedTuple V's;
# `NDualRef` is the `Ref`-specific V, so it needs its own branch.
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    ::Lifted{Val{:x}},
) where {Nw,P<:NDualEltype}
    v = getfield(primal(x), :x)
    return Lifted{P,Nw}(v, _scalar_ndual(v, tangent(x).partials[]))
end

@inline _get_lifted_field(V::Union{NamedTuple,Tuple}, name) = getfield(V, name)
# A `PossiblyUninitTangent` backing field is unwrapped: the caller has already
# read the primal field (so it is defined), hence the PUT is initialised.
@inline _maybe_val(x::PossiblyUninitTangent) = val(x)
@inline _maybe_val(x) = x
@inline _coerce_backing_field(::Type{F}, v) where {F<:PossiblyUninitTangent} = F(v)
@inline _coerce_backing_field(::Type, v) = v
@inline _get_lifted_field(V::Union{ImmutableDual,MutableDual}, name) = _maybe_val(
    getfield(getfield(V, :value), name)
)
@inline _get_lifted_field(::NoDual, _) = NoDual()
# NDualArray is the parallel-arrays wrapper for `Array{T,D}` slots (not a struct lift).
# Sub-field access into the underlying primal struct happens in inlined
# `Array` / `Memory` code; project to the matching parallel-arrays per-lane V so the
# downstream rule (`memoryrefnew`, etc.) keeps a coherent V chain.
@static if VERSION >= v"1.11-rc4"
    # An `Array{T,D}` (any rank D) has a `.ref::MemoryRef{T}` field over flat
    # `Memory{T}` storage. Project the NDualArray V to the matching memory-ref V
    # so the downstream `memoryrefnew`/`memoryrefget` chain stays coherent.
    # Accept the integer field index too: `getfield(arr, 1)` is `.ref` — without
    # this the integer form silently fell through to a `NoDual`, zeroing the
    # forward derivative of `.ref` in forward-over-reverse.
    @inline function _get_lifted_field(
        V::Nfwd.NDualArray{T,N,D,A}, name::Union{Symbol,Int}
    ) where {T<:Nfwd.NDualEltype,N,D,A<:Array{T,D}}
        name = name isa Int ? fieldname(typeof(V.primal), name) : name
        if name === :ref
            primal_ref = getfield(V.primal, :ref)
            partial_refs = ntuple(k -> getfield(V.partials[k], :ref), Val(N))
            return Nfwd.NDualMemoryRef{T,N,Memory{T}}(primal_ref, partial_refs)
        end
        return NoDual()
    end
    # `.mem` of a `MemoryRef` is the underlying `Memory`; project the parallel-arrays
    # memory-ref V to the matching `NDualArray` over those memories (the
    # `.ptr_or_offset` field is a non-diff Ptr → `NoDual`). Mirrors the reverse
    # `rrule!!`, which returns `x.dx.mem`.
    @inline function _get_lifted_field(
        V::Nfwd.NDualMemoryRef{T,N,M}, name::Union{Symbol,Int}
    ) where {T,N,M}
        name = name isa Int ? fieldname(typeof(V.primal), name) : name
        if name === :mem
            primal_mem = getfield(V.primal, :mem)
            partial_mems = ntuple(k -> getfield(V.partials[k], :mem), Val(N))
            return Nfwd.NDualArray{T,N,1,M}(primal_mem, partial_mems)
        elseif name === :ptr_or_offset
            # Per-lane raw pointers (one into each partial memory); the downstream
            # `bitcast` re-types them, landing an `NTuple{N,Ptr{T}}` for a foreigncall.
            return ntuple(k -> getfield(V.partials[k], :ptr_or_offset), Val(N))
        end
        return NoDual()
    end
    # Element-wise array V (a plain `Array` of per-element forward Vs, for differentiable
    # non-float-element arrays): its `.ref` is a `MemoryRef` into the V array,
    # parallel to the primal's `.ref`. Other fields (`.size`) are non-diff. Accept the integer
    # field index too (`getfield(arr, 1)` occurs in forward-over-reverse), like the
    # `NDualArray` method above.
    @inline function _get_lifted_field(V::Array, name::Union{Symbol,Int})
        name = name isa Int ? fieldname(typeof(V), name) : name
        return name === :ref ? getfield(V, :ref) : NoDual()
    end
    # Element-wise memory-ref V (a plain `MemoryRef` into an element-wise V `Memory`): `.mem` projects to the
    # element-wise `Memory` V; `.ptr_or_offset` is the raw data pointer into that V `Memory`, typed as its
    # element-wise dual element (e.g. `Ptr{NDualArray}`) so a downstream `unsafe_copyto!` copies the correct
    # per-element stride. Wrapped in a 1-tuple — the canonical per-lane `Ptr` V (width-1, element-wise).
    # Mirrors the 1.10 `jl_array_ptr` element-wise frule, which emits `Ptr{NDualArray}` directly; without
    # this the 1.12 `pointer(::Vector{<diff non-float>})` path drops the tangent (`NoDual`).
    @inline function _get_lifted_field(V::MemoryRef, name::Union{Symbol,Int})
        name = name isa Int ? fieldname(typeof(V), name) : name
        name === :mem && return getfield(V, :mem)
        # Only a differentiable element (element-wise dual `Memory`, not `Memory{NoDual}`) carries a
        # tangent pointer; a non-differentiable element's pointer stays `NoDual`.
        if name === :ptr_or_offset
            E = eltype(getfield(V, :mem))
            E === NoDual && return NoDual()
            return (Base.bitcast(Ptr{E}, getfield(V, :ptr_or_offset)),)
        end
        return NoDual()
    end
    # Element-wise `Memory` V: its fields (`.length`, `.ptr`, by name OR position) are all
    # non-diff metadata; element access goes through `memoryrefget`, not here.
    @inline _get_lifted_field(::Memory, _) = NoDual()
end
# Generic NDualArray fall-through (older Julia, non-Vector storage, etc.).
@inline _get_lifted_field(::Mooncake.Nfwd.NDualArray, _) = NoDual()

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
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw}, x::Lifted, ::Lifted{Val{f}}, ::Lifted{Val{order}}
) where {Nw,f,order}
    primal_field = getfield(primal(x), f, order)
    # See the 2-arg `lgetfield` frule: canonical zero V for a non-differentiable parent.
    tangent(x) isa NoDual && return uninit_lifted(Val(Nw), primal_field)
    V_i = _get_lifted_field(tangent(x), f)
    return Lifted{typeof(primal_field),Nw}(primal_field, V_i)
end
# `Ref{P<:NDualEltype}` field read with an order argument: the `NDualRef` V needs the same
# rebuild as the 2-arg branch (the order arg controls atomicity, not the value/derivative).
# Without this the generic 3-arg frule above routes through `_get_lifted_field(::NDualRef, ...)`,
# which has no method.
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    ::Lifted{Val{:x}},
    ::Lifted{Val{order}},
) where {Nw,P<:NDualEltype,order}
    v = getfield(primal(x), :x, order)
    return Lifted{P,Nw}(v, _scalar_ndual(v, tangent(x).partials[]))
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
# Write the field's per-lane V_i back into the parent's MutableDual via the
# central writeback helper. Mutable structs only (immutable structs go through
# reverse-mode rebuild paths and don't reach lsetfield!).
@inline function frule!!(
    ::Lifted{typeof(lsetfield!),Nw},
    value::Lifted{P,Nw,<:MutableDual},
    ::Lifted{Val{name}},
    x::Lifted,
) where {Nw,P,name}
    setfield!(primal(value), name, primal(x))
    # Normalise an integer field index to its symbol name: the V's backing `NamedTuple` is
    # symbol-keyed, so `NamedTuple{(name,)}` with an `Int` `name` would throw (mirrors the
    # integer-index normalisation in `_get_lifted_field`).
    nm = name isa Int ? fieldname(P, name) : name
    # Write the field's per-lane V back through the shared `MutableDual` writeback helper, which
    # coerces the backing field type and `convert`s the merged `NamedTuple` (needed for abstract
    # fields — see `_setfield_tangent!`). Sharing the helper keeps this in lockstep with the
    # runtime-name `setfield!` frule rather than duplicating the merge logic.
    _setfield_tangent!(tangent(value), nm, tangent(x))
    return x
end
# Non-differentiable struct (V === NoDual): set the primal field; there is no
# tangent to update. Mirrors the reverse `F == NoFData` branch of `lsetfield_rrule`.
@inline function frule!!(
    ::Lifted{typeof(lsetfield!),Nw},
    value::Lifted{P,Nw,NoDual},
    ::Lifted{Val{name}},
    x::Lifted,
) where {Nw,P,name}
    setfield!(primal(value), name, primal(x))
    return x
end
# `Ref{P<:NDualEltype}` field write (`r[] = v`): set the primal value and the `NDualRef` partials
# shadow (`_nfwd_dual_partial` extracts the per-lane partials from `NDual`/`Complex{NDual}`). The
# MutableDual frule above handles generic mutable structs; `NDualRef` is the Ref-specific V.
@inline function frule!!(
    ::Lifted{typeof(lsetfield!),Nw},
    value::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    ::Lifted{Val{:x}},
    x::Lifted{P,Nw},
) where {Nw,P<:NDualEltype}
    setfield!(primal(value), :x, primal(x))
    tangent(value).partials[] = ntuple(k -> _nfwd_dual_partial(tangent(x), k), Val(Nw))
    return x
end
@inline function rrule!!(
    ::CoDual{typeof(lsetfield!)}, value::CoDual{P,F}, name::CoDual, x::CoDual
) where {P,F<:StandardFDataType}
    return lsetfield_rrule(value, name, x)
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
    # Copy a Dict-field forward V to alias the copied primal array. A float-element field
    # (e.g. `Vector{Float64}` vals → `NDualArray`) rebuilds over the new primal array with
    # copied partials; an element-wise field (`Vector{UInt8}` slots → `Vector{NoDual}`, `Vector{Any}`
    # keys/vals) is a shallow array copy whose elements alias the shallow-shared key/value
    # objects, matching `Base.copy(::Dict)`. The old `map(copy, …)` form errored on
    # `Vector{NoDual}` (`copy(::NoDual)`) and assumed every field was an `NDualArray`.
    _copy_dict_field_v(new_arr, v::NDualArray) = typeof(v)(new_arr, map(copy, v.partials))
    _copy_dict_field_v(::Any, v::AbstractArray) = copy(v)
    function frule!!(
        ::Lifted{typeof(copy),Nw}, a::Lifted{D,Nw,<:MutableDual}
    ) where {Nw,D<:Dict}
        new_primal = copy(primal(a))
        old_nt = getfield(tangent(a), :value)
        new_nt = (
            slots=_copy_dict_field_v(new_primal.slots, old_nt.slots),
            keys=_copy_dict_field_v(new_primal.keys, old_nt.keys),
            vals=_copy_dict_field_v(new_primal.vals, old_nt.vals),
            ndel=NoDual(),
            count=NoDual(),
            age=NoDual(),
            idxfloor=NoDual(),
            maxprobe=NoDual(),
        )
        return Lifted{D,Nw}(new_primal, MutableDual(new_nt))
    end
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
        # Always-initialised ABSTRACT field (`Foo.x::Real`): its backing V field is bare `Any`, so the
        # merged `NamedTuple` narrows to a concrete dual type and (NamedTuple invariance) would fail a
        # bare `setfield!` writeback — the `convert` in the shared `_setfield_tangent!` is required.
        # `:none` perf flag: an abstract field legitimately boxes, so don't assert stability/allocs.
        (false, :none, nothing, lsetfield!, TestResources.Foo(5.0), Val(:x), 4.0),
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

        # Ref{<:NDualEltype} carries the `NDualRef` V. Regression for the 3-arg (order-arg) form:
        # the generic frule routed it through a missing `_get_lifted_field(::NDualRef, ...)`. The
        # `order` loop below generates both the 2-arg and 3-arg variants from this single entry.
        (false, :none, nothing, lgetfield, Ref(5.0), Val(:x)),
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
