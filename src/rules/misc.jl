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

# Zero partials must retain the canonical V; NoDual would break downstream field access.
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

# `rethrow` is a `ccall(:jl_rethrow)`, which has no foreigncall rule, so a `finally` or `catch`
# block's exception path would raise `MissingForeigncallRuleError` in place of the exception
# itself. Forward only: reverse refuses try/catch before any rule is reached.
@is_primitive MinimalCtx ForwardMode Tuple{typeof(rethrow)}
@is_primitive MinimalCtx ForwardMode Tuple{typeof(rethrow),Any}
frule!!(::Lifted{typeof(rethrow)}) = rethrow()
frule!!(::Lifted{typeof(rethrow)}, e::Lifted) = rethrow(primal(e))

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
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw}, x::Lifted, ::Lifted{Val{f}}
) where {Nw,f}
    primal_field = getfield(primal(x), f)
    # A NoDual parent can have a differentiable field (e.g. DataType.parameters);
    # construct the field's canonical V.
    # TODO(#1295): those partials do not alias the storage the pass seeded for the field.
    tangent(x) isa NoDual && return uninit_lifted(Val(Nw), primal_field)
    V_i = _get_lifted_field(tangent(x), f)
    _check_lifted_field_ptr_lanes(V_i, Val(Nw))
    return Lifted{typeof(primal_field),Nw}(primal_field, V_i)
end

# NDualRef stores partials separately, so field reads must rebuild the scalar V.
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    ::Lifted{<:Union{Val{:x},Val{1}}},
) where {Nw,P<:NDualEltype}
    v = getfield(primal(x), :x)
    return Lifted{P,Nw}(v, _scalar_ndual(v, tangent(x).partials[]))
end

@inline _get_lifted_field(V::Union{NamedTuple,Tuple}, name) = getfield(V, name)
@inline _coerce_backing_field(::Type{F}, v) where {F<:PossiblyUninitTangent} = F(v)
@inline _coerce_backing_field(::Type, v) = v
# A `PossiblyUninitTangent` backing field is unwrapped via `val` (the caller has already read the
# primal field, so the PUT is initialised); any other field passes through unchanged.
@inline _get_lifted_field(V::Union{ImmutableDual,MutableDual}, name) = val(
    getfield(getfield(V, :fields), name)
)
@inline _get_lifted_field(::NoDual, _) = NoDual()
# Complex{NDual} fields already hold their canonical scalar V.
@inline _get_lifted_field(V::Complex, name) = getfield(V, name)
@static if VERSION >= v"1.11-rc4"
    # Project Array.ref by name or index, preserving the memoryref chain's canonical V.
    @inline function _get_lifted_field(
        V::Nfwd.NDualArray{T,N,D,A}, name::Union{Symbol,Int}
    ) where {T<:Nfwd.NDualEltype,N,D,A<:Array{T,D}}
        name = name isa Int ? fieldname(typeof(V.primal), name) : name
        if name === :ref
            # Share block storage so mutations alias. Use its flat parent's ref to avoid
            # allocating a reshape header on every element access; array.ref starts at column 1.
            return Nfwd.NDualMemoryRef{T,N,Memory{T}}(
                getfield(V.primal, :ref),
                getfield(getfield(getfield(V, :partials_block), :parent), :ref),
                length(V.primal),
                1,
            )
        end
        return NoDual()
    end
    # The ref may cover only a window (e.g. a grown Vector with capacity slack).
    # Mem slot 1 is at refoff + (col - o)*N; project the whole Memory over the
    # same backing so mutations through the ref and Memory V still alias.
    @inline function _get_lifted_field(
        V::Nfwd.NDualMemoryRef{T,N,M}, name::Union{Symbol,Int}
    ) where {T,N,M}
        name = name isa Int ? fieldname(typeof(V.primal), name) : name
        if name === :mem
            p = getfield(V, :primal)
            primal_mem = getfield(p, :mem)
            len = length(primal_mem)
            bref = getfield(V, :partials_ref)
            backing = getfield(bref, :mem)
            start =
                Core.memoryrefoffset(bref) +
                (getfield(V, :col) - Core.memoryrefoffset(p)) * N
            if start < 1 || start + N * len - 1 > length(backing)
                throw(
                    ArgumentError(
                        "Cannot project `.mem` of a lifted `MemoryRef{$T}`: its partials " *
                        "block does not cover the whole backing `Memory` (length $len; " *
                        "block backing length $(length(backing)), start offset $start). " *
                        "This ref's block was created for a smaller container.",
                    ),
                )
            end
            # One header construction, branching on the REF rather than on the header: LLVM
            # will not promote an allocation that reaches a phi node, so building the empty
            # case separately keeps both headers alive. An empty window has no slot to address,
            # so it takes the block's own ref, which `_new_` never dereferences at length 0.
            ref = if len == 0
                bref
            else
                Core.memoryrefnew(Core.memoryrefnew(backing), start, true)
            end
            flat = _new_(Vector{T}, ref, (N * len,))
            return Nfwd.NDualArray{T,N,1,M}(primal_mem, NDualBlock{T,2}(flat, (N, len)))
        elseif name === :ptr_or_offset
            # Per-lane raw pointers require dense per-lane storage; in the element-major block
            # a lane is strided (stride N), so only width 1 has an addressable lane. The
            # downstream `bitcast` re-types the pointer, landing `NTuple{1,Ptr{T}}`.
            N == 1 || throw(
                ArgumentError(
                    "Forward-mode raw pointer (`ptr_or_offset`) of a lifted `MemoryRef{$T}` " *
                    "is unsupported at chunk width $N > 1: the element-major partials block " *
                    "stores each lane with stride $N, so there is no dense per-lane buffer " *
                    "a raw pointer could address. Differentiate at chunk width 1.",
                ),
            )
            lane_ref = Nfwd._block_column_ref(
                getfield(V, :partials_ref), getfield(V, :col), N
            )
            return (getfield(lane_ref, :ptr_or_offset),)
        end
        return NoDual()
    end
    # Element-wise arrays project .ref by name or index; .size is non-differentiable.
    @inline function _get_lifted_field(V::Array, name::Union{Symbol,Int})
        name = name isa Int ? fieldname(typeof(V), name) : name
        return name === :ref ? getfield(V, :ref) : NoDual()
    end
    # Element-wise refs project .mem and a width-1 pointer typed for the dual element,
    # so unsafe_copyto! uses the correct stride (as in the 1.10 jl_array_ptr rule).
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

# Element-wise pointer tuples support only width 1: no dense per-lane buffer exists
# at wider widths. Every _get_lifted_field caller building a slot must check this;
# the projection itself cannot see Nw. Coherent NTuple{Nw,Ptr} values pass through.
@inline function _check_lifted_field_ptr_lanes(V_i, ::Val{Nw}) where {Nw}
    if Nw > 1 && V_i isa Tuple{Vararg{Ptr}} && length(V_i) != Nw
        throw(
            ArgumentError(
                "Forward-mode raw pointer of a lifted nested array is unsupported at chunk " *
                "width $Nw > 1: the per-element dual has no dense per-lane buffer a single raw " *
                "pointer could address, so the derivative would be silently dropped. " *
                "Differentiate at chunk width 1.",
            ),
        )
    end
    return nothing
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

# TODO(#1295): the `NoFData` method below mints fresh fdata for a differentiable field of a
# `NoTangent` parent, breaking the aliasing invariant. This is the site a literal field name
# reaches, via `lgetfield`; the dynamic-name spelling is in `builtins.jl`.
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

@is_primitive MinimalCtx Tuple{typeof(lgetfield),Any,Val,Val}
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw}, x::Lifted, ::Lifted{Val{f}}, ::Lifted{Val{order}}
) where {Nw,f,order}
    primal_field = getfield(primal(x), f, order)
    # See the 2-arg `lgetfield` frule: canonical zero V for a non-differentiable parent.
    tangent(x) isa NoDual && return uninit_lifted(Val(Nw), primal_field)
    V_i = _get_lifted_field(tangent(x), f)
    _check_lifted_field_ptr_lanes(V_i, Val(Nw))
    return Lifted{typeof(primal_field),Nw}(primal_field, V_i)
end
# Ordered Ref reads also need to rebuild the scalar V from NDualRef partials.
@inline function frule!!(
    ::Lifted{typeof(lgetfield),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    ::Lifted{<:Union{Val{:x},Val{1}}},
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
@inline function frule!!(
    ::Lifted{typeof(lsetfield!),Nw},
    value::Lifted{P,Nw,<:MutableDual},
    ::Lifted{Val{name}},
    x::Lifted,
) where {Nw,P,name}
    setfield!(primal(value), name, primal(x))
    # The backing NamedTuple requires symbol keys.
    nm = name isa Int ? fieldname(P, name) : name
    # Share runtime-name writeback, including conversion for abstract backing fields.
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
# NDualRef keeps scalar partials in a separate shadow buffer.
@inline function frule!!(
    ::Lifted{typeof(lsetfield!),Nw},
    value::Lifted{<:Base.RefValue{P},Nw,<:NDualRef},
    ::Lifted{<:Union{Val{:x},Val{1}}},
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
    # Rebind float-array V to the copied primal; shallow-copy element-wise V so
    # its elements alias the shallow-shared keys/values, matching Base.copy(::Dict).
    _copy_dict_field_v(new_arr, v::NDualArray) = typeof(v)(
        new_arr, copy(getfield(v, :partials_block))
    )
    _copy_dict_field_v(::Any, v::AbstractArray) = copy(v)
    function frule!!(
        ::Lifted{typeof(copy),Nw}, a::Lifted{D,Nw,<:MutableDual}
    ) where {Nw,D<:Dict}
        new_primal = copy(primal(a))
        old_nt = getfield(tangent(a), :fields)
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

@zero_derivative MinimalCtx Tuple{typeof(sortperm),Vector{<:IEEEFloat}}
@is_primitive MinimalCtx Tuple{typeof(sort),Vector{<:IEEEFloat}}

function frule!!(::Lifted{typeof(sort),N}, x::Lifted{Vector{T},N}) where {N,T<:IEEEFloat}
    p = sortperm(primal(x))
    y = primal(x)[p]
    # Element `i` of `y` is element `p[i]` of `x`, so its lane column is column `p[i]` of `x`'s block.
    dy = NDualArray{T,N,1,Vector{T}}(y)
    copyto!(
        getfield(dy, :partials_block), view(getfield(tangent(x), :partials_block), :, p)
    )
    return Lifted{Vector{T},N}(y, dy)
end

function rrule!!(::CoDual{typeof(sort)}, x::CoDual{<:Vector{<:IEEEFloat}})
    p = sortperm(primal(x))
    dx = tangent(x)
    y = primal(x)[p]
    dy = zero(y)
    function sort_pb!!(::NoRData)
        for i in eachindex(p)
            dx[p[i]] += dy[i]
        end
        return NoRData(), NoRData()
    end
    return CoDual(y, dy), sort_pb!!
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
        # Foo.x::Real requires converting the merged backing NamedTuple; abstract fields
        # legitimately box, so use :none rather than asserting stability/allocations.
        (false, :none, nothing, lsetfield!, TestResources.Foo(5.0), Val(:x), 4.0),
        # Positional access on a single-field Ref: setfield!(r, 1, v) === setfield!(r, :x, v), so the
        # lsetfield! frule must accept Val(1) as well as Val(:x).
        (false, :none, nothing, lsetfield!, Ref(5.0), Val(1), 4.0),
        (false, :none, nothing, lsetfield!, Ref(5.0), Val(:x), 4.0),
    ]

    for T in (Float16, Float32, Float64), f in (sort, sortperm)
        # Float16 needs small inputs for the finite-difference step grid.
        x = T === Float16 ? T[0.01, -0.01] : T[3, 1, 11, 2, 10, 4, 9, 5, 8, 6, 7]
        push!(specific_test_cases, (false, :stability, nothing, f, x))
    end

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

        # NDualRef reads by name and position, with both arities from the order loop.
        (false, :none, nothing, lgetfield, Ref(5.0), Val(:x)),
        (false, :none, nothing, lgetfield, Ref(5.0), Val(1)),
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
