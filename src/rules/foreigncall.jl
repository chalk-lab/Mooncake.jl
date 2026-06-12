# Fallback rule for foreigncall which gives an interpretable error message.
struct MissingForeigncallRuleError <: Exception
    msg::String
end

function Base.showerror(io::IO, err::MissingForeigncallRuleError)
    _print_boxed_error(io, split(err.msg, '\n'))
end

# Fallback foreigncall rules. This is a sufficiently common special case, that it's worth
# creating an informative error message, so that users have some chance of knowing why
# they're not able to differentiate a piece of code.
function frule!!(::Lifted{typeof(_foreigncall_),Nw}, args...) where {Nw}
    return throw_missing_foreigncall_rule_error(:frule!!, args...)
end
function rrule!!(::CoDual{typeof(_foreigncall_)}, args...)
    return throw_missing_foreigncall_rule_error(:rrule!!, args...)
end

function throw_missing_foreigncall_rule_error(rule_name::Symbol, args...)
    throw(
        MissingForeigncallRuleError(
            "No $rule_name available for foreigncall with primal argument types " *
            "$(typeof(map(primal, args))). " *
            "This problem has most likely arisen because there is a ccall somewhere in the " *
            "function you are trying to differentiate, for which an $rule_name has not been " *
            "explicitly written. " *
            "You have three options: write an $rule_name for this foreigncall, write an $rule_name " *
            "for a Julia function that calls this foreigncall, or re-write your code to " *
            "avoid this foreigncall entirely. " *
            "If you believe that this error has arisen for some other reason than the above, " *
            "or the above does not help you to workaround this problem, please open an issue.",
        ),
    )
end

_get_arg_type(::Type{Val{T}}) where {T} = T

"""
    function _foreigncall_(
        ::Val{name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x...
    ) where {name, RT, nreq, calling_convention}

:foreigncall nodes get translated into calls to this function.
For example,
```julia
Expr(:foreigncall, :foo, Tout, (A, B), nreq, :ccall, args...)
```
becomes
```julia
_foreigncall_(Val(:foo), Val(Tout), (Val(A), Val(B)), Val(nreq), Val(:ccall), args...)
```
Please consult the Julia documentation for more information on how foreigncall nodes work,
and consult this package's tests for examples.

Credit: Umlaut.jl has the original implementation of this function. This is largely copied
over from there.
"""
@generated function _foreigncall_(
    ::Val{name},
    ::Val{RT},
    AT::Tuple,
    ::Val{nreq},
    ::Val{calling_convention},
    x::Vararg{Any,N},
) where {name,RT,nreq,calling_convention,N}
    return Expr(
        :foreigncall,
        QuoteNode(name),
        :($(RT)),
        Expr(:call, :(Core.svec), map(_get_arg_type, AT.parameters)...),
        :($nreq),
        QuoteNode(calling_convention),
        map(n -> :(x[$n]), 1:length(x))...,
    )
end

@is_primitive MinimalCtx Tuple{typeof(_foreigncall_),Vararg}

#
# Rules to handle / avoid foreigncall nodes
#

@zero_derivative MinimalCtx Tuple{typeof(Base.allocatedinline),Type}

@zero_derivative MinimalCtx Tuple{typeof(objectid),Any}

@is_primitive MinimalCtx Tuple{typeof(pointer_from_objref),Any}
# Generic fallback (a real/complex-scalar `Ref` is handled by the more-specific `NDualRef` overload
# below). V is `NTuple{Nw, Ptr{tangent_type(Nothing)}}` — the tangent OBJECT's address, typed as the
# reverse rule below threads it, and deliberately distinct from any per-lane-partial pointer V
# (`Ptr{<:IEEEFloat}` / `Ptr{Nothing}`). `bitcast` preserves that element type, so the address
# round-trips through `unsafe_pointer_to_objref` (recovering the full slot) but a raw scalar load
# (`pointerref` after `bitcast` to a float ptr) stays incoherent and hits the loud `pointerref`
# throw — a mutable struct's tangent interleaves value and partials with no parallel partials buffer
# to address. Only a proven-non-differentiable (`NoDual`) tangent gets the NULL sentinel mapped back
# to `NoDual`; an immutable differentiable V (e.g. `NDualArray`) has no tangent-object address to
# thread, so fail loudly rather than silently dropping the derivative. The primal address
# (identity/hashing/`objectid`) is unchanged.
function frule!!(::Lifted{typeof(pointer_from_objref),Nw}, x::Lifted) where {Nw}
    y = pointer_from_objref(primal(x))
    tx = tangent(x)
    taddr = if tx isa NoDual
        Ptr{tangent_type(Nothing)}(0)
    elseif ismutable(tx)
        bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(tx))
    else
        throw(
            ArgumentError(
                "Forward-mode AD cannot take `pointer_from_objref` of a `$(typeof(primal(x)))` " *
                "whose forward tangent is the immutable `$(typeof(tx))`: there is no " *
                "tangent-object address to thread through the pointer, so the derivative would " *
                "be silently dropped.",
            ),
        )
    end
    return Lifted{typeof(y),Nw}(y, ntuple(_ -> taddr, Val(Nw)))
end
# `Ref{P<:NDualEltype}` (`NDualRef`): unlike the interleaved `MutableDual` fallback above, the
# partials live in a parallel buffer with primal-identical scalar layout. Thread per-lane pointers
# into that buffer's contiguous `NTuple` (lane `k` at offset `(k-1)*sizeof(P)`), so the existing
# `bitcast` (re-types each `Ptr{P}` lane) and `pointerref` (`NTuple{Nw,Ptr{P}}` → scalar dual)
# frules reconstruct the derivative. Forward raw-pointer reads of a `Ref` (real or complex) are
# correct; `unsafe_pointer_to_objref` reads the partials back through these pointers to round-trip.
function frule!!(
    ::Lifted{typeof(pointer_from_objref),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef{P,Nw}},
) where {Nw,P<:NDualEltype}
    y = pointer_from_objref(primal(x))
    base = UInt(pointer_from_objref(tangent(x).partials))
    return Lifted{typeof(y),Nw}(y, ntuple(k -> Ptr{P}(base + (k - 1) * sizeof(P)), Val(Nw)))
end
function rrule!!(f::CoDual{typeof(pointer_from_objref)}, x)
    y = CoDual(
        pointer_from_objref(primal(x)),
        bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(tangent(x))),
    )
    return y, NoPullback(f, x)
end

@zero_derivative MinimalCtx Tuple{typeof(CC.return_type),Vararg}

@is_primitive MinimalCtx Tuple{typeof(Base.unsafe_pointer_to_objref),Ptr}
# Inverse of the `pointer_from_objref` rule: recover the primal object from the primal address, and
# — when the lane carries a non-NULL tangent-object address — recover that tangent object so the
# forward derivative survives the round-trip. The recovered object is dynamically typed, so the
# slot type and its V are only known at runtime; the `Lifted{typeof(y),Nw}` constructor sharpens
# through `typeof(y)`, widening the inferred return type (the cost of this dynamic boundary).
function frule!!(
    ::Lifted{typeof(Base.unsafe_pointer_to_objref),Nw}, x::Lifted{<:Ptr}
) where {Nw}
    y = unsafe_pointer_to_objref(primal(x))
    tx = tangent(x)
    (tx isa NoDual || tx[1] == Ptr{Nothing}(0)) && return Lifted{typeof(y),Nw}(y, NoDual())
    return Lifted{typeof(y),Nw}(y, unsafe_pointer_to_objref(tx[1]))
end
# Round-trip of a `Ref{<:NDualEltype}` pointer (V `NTuple{Nw,Ptr{<:NDualEltype}}`, from the `NDualRef`
# rule above): recover the primal `Ref` from the primal pointer, and rebuild its `NDualRef` by reading
# the per-lane partials back through the V's pointers. Forward tangent storage is slot-local, so a
# fresh partials buffer holding the same values is correct (the JVP through identity is identity).
# The V's `Ptr{<:NDualEltype}` element distinguishes a `Ref`-pointer from the generic objref tag
# (`Ptr{tangent_type(Nothing)}`, handled above), and is the only such V — arrays are not round-tripped
# this way. The scalar element type is read at runtime (`eltype`) rather than bound as a type
# parameter, since `P` in `NTuple{Nw,Ptr{P}}` would be unbound at the degenerate `Nw == 0`.
function frule!!(
    ::Lifted{typeof(Base.unsafe_pointer_to_objref),Nw},
    x::Lifted{<:Ptr,Nw,<:NTuple{Nw,Ptr{<:NDualEltype}}},
) where {Nw}
    ref = unsafe_pointer_to_objref(primal(x))
    P = eltype(eltype(tangent(x)))
    partials = ntuple(k -> unsafe_load(tangent(x)[k]), Val(Nw))
    return Lifted{typeof(ref),Nw}(
        ref, NDualRef{P,Nw}(Base.RefValue{NTuple{Nw,P}}(partials))
    )
end
function rrule!!(f::CoDual{typeof(Base.unsafe_pointer_to_objref)}, x::CoDual{<:Ptr})
    y = CoDual(unsafe_pointer_to_objref(primal(x)), unsafe_pointer_to_objref(tangent(x)))
    return y, NoPullback(f, x)
end

@zero_derivative MinimalCtx Tuple{typeof(Threads.threadid)}
@zero_derivative MinimalCtx Tuple{typeof(typeintersect),Any,Any}

function _increment_pointer!(x::Ptr{T}, y::Ptr{T}, N::Integer) where {T}
    increment!!(unsafe_wrap(Vector{T}, x, N), unsafe_wrap(Vector{T}, y, N))
    return x
end

# unsafe_copyto! is the only function in Julia that appears to rely on a ccall to `memmove`.
# Since we can't differentiate `memmove` (due to a lack of type information), it is
# necessary to work with `unsafe_copyto!` instead.
@is_primitive MinimalCtx Tuple{typeof(unsafe_copyto!),Ptr{T},Ptr{T},Any} where {T}
# A differentiable Ptr's V is `NTuple{N, Ptr}` (per-lane partial pointers). Copy the primal
# data and each lane's tangent data through its own per-lane pair of source/destination
# pointers. The partial-pointer element type is element-type-agnostic, so this covers both
# the scalar parallel-arrays case (`Ptr{T<:NDualEltype}`, V `NTuple{N,Ptr{T}}`) and the nested-array case
# (`Ptr{Vector{Float64}}` from `pointer(::Vector{Vector{Float64}})`, V `NTuple{N,Ptr{NDualArray}}`),
# matching the `rrule!!`'s `Ptr{T}` breadth below.
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    dest::Lifted{P,Nw,<:NTuple{Nw,Ptr}},
    src::Lifted{P,Nw,<:NTuple{Nw,Ptr}},
    n::Lifted,
) where {Nw,P<:Ptr}
    _n = primal(n)
    unsafe_copyto!(primal(dest), primal(src), _n)
    dest_partials = tangent(dest)
    src_partials = tangent(src)
    @inbounds for lane in 1:Nw
        unsafe_copyto!(dest_partials[lane], src_partials[lane], _n)
    end
    return dest
end
# Non-differentiable pointers (V === NoDual, e.g. `Ptr{UInt8}` / `Ptr{Int}` — the
# element type is non-differentiable, `tangent_type(T) === NoTangent`): copy the primal
# data; no tangent to copy. (A `Ptr{Vector{Float64}}` is differentiable — its V is
# `NTuple{Nw, Ptr}`, handled by the V<:NTuple frule above, not this NoDual overload.)
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    dest::Lifted{Ptr{T},Nw,NoDual},
    src::Lifted{Ptr{T},Nw,NoDual},
    n::Lifted,
) where {Nw,T}
    unsafe_copyto!(primal(dest), primal(src), primal(n))
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)}, dest::CoDual{Ptr{T}}, src::CoDual{Ptr{T}}, n::CoDual
) where {T}
    _n = primal(n)

    # Record values that will be overwritten.
    dest_copy = Vector{T}(undef, _n)
    ddest_copy = Vector{T}(undef, _n)
    pdest = primal(dest)
    ddest = tangent(dest)
    unsafe_copyto!(pointer(dest_copy), pdest, _n)
    unsafe_copyto!(pointer(ddest_copy), ddest, _n)

    # Run primal computation.
    dsrc = tangent(src)
    unsafe_copyto!(primal(dest), primal(src), _n)
    unsafe_copyto!(tangent(dest), dsrc, _n)

    function unsafe_copyto!_pb!!(::NoRData)

        # Increment dsrc.
        _increment_pointer!(dsrc, ddest, _n)

        # Restore initial state.
        unsafe_copyto!(pdest, pointer(dest_copy), _n)
        unsafe_copyto!(ddest, pointer(ddest_copy), _n)

        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return dest, unsafe_copyto!_pb!!
end

# Reshape both primal and per-lane partials via ccall.
function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_reshape_array},Nw},
    ::Lifted{Val{Array{P,M}},Nw},
    ::Lifted{Tuple{Val{Any},Val{Any},Val{Any}},Nw},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    ::Lifted{Type{Array{P,M}},Nw},
    a::Lifted{Array{P,D},Nw,NDualArray{P,Nw,D,Array{P,D},NDual{P,Nw}}},
    dims::Lifted,
) where {Nw,P<:IEEEFloat,M,D}
    d = primal(dims)
    y = ccall(:jl_reshape_array, Array{P,M}, (Any, Any, Any), Array{P,M}, primal(a), d)
    new_partials = ntuple(
        k -> ccall(
            :jl_reshape_array,
            Array{P,M},
            (Any, Any, Any),
            Array{P,M},
            tangent(a).partials[k],
            d,
        ),
        Val(Nw),
    )
    return Lifted{Array{P,M},Nw}(y, NDualArray{P,Nw,M,Array{P,M}}(y, new_partials))
end
# Non-differentiable element arrays (element-wise `Array{NoDual}` V, e.g. the
# `Matrix{Tuple{Int,Colon}}` index buffer reshaped inside `sortslices`): reshape primal and V in
# lockstep. Mirrors the element-type-generic reverse rrule below.
function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_reshape_array},Nw},
    ::Lifted{Val{Array{P,M}},Nw},
    ::Lifted{Tuple{Val{Any},Val{Any},Val{Any}},Nw},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    ::Lifted{Type{Array{P,M}},Nw},
    a::Lifted{<:Array,Nw,<:AbstractArray{NoDual}},
    dims::Lifted,
) where {Nw,P,M}
    d = primal(dims)
    y = ccall(:jl_reshape_array, Array{P,M}, (Any, Any, Any), Array{P,M}, primal(a), d)
    v = ccall(
        :jl_reshape_array, Array{NoDual,M}, (Any, Any, Any), Array{NoDual,M}, tangent(a), d
    )
    return Lifted{Array{P,M},Nw}(y, v)
end
function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
    ::CoDual{Val{:jl_reshape_array}},
    ::CoDual{Val{Array{P,M}}},
    ::CoDual{Tuple{Val{Any},Val{Any},Val{Any}}},
    ::CoDual, # nreq
    ::CoDual, # calling convention
    x::CoDual{Type{Array{P,M}}},
    a::CoDual{Array{P,N},Array{T,N}},
    dims::CoDual,
) where {P,T,M,N}
    d = primal(dims)
    y = CoDual(
        ccall(:jl_reshape_array, Array{P,M}, (Any, Any, Any), Array{P,M}, primal(a), d),
        ccall(:jl_reshape_array, Array{T,M}, (Any, Any, Any), Array{T,M}, tangent(a), d),
    )
    return y, NoPullback(ntuple(_ -> NoRData(), 9))
end

function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_array_isassigned},Nw},
    ::Lifted{RT,Nw},
    ::Lifted{AT,Nw},
    ::Lifted{nreq,Nw},
    ::Lifted{calling_convention,Nw},
    a::Lifted,
    ii::Lifted,
    args...,
) where {Nw,RT,AT,nreq,calling_convention}
    GC.@preserve args begin
        y = ccall(:jl_array_isassigned, Cint, (Any, UInt), primal(a), primal(ii))
    end
    return Lifted{Cint,Nw}(y, NoDual())
end

function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
    ::CoDual{Val{:jl_array_isassigned}},
    ::CoDual{RT}, # return type is Int32
    arg_types::CoDual{AT}, # arg types are (Any, UInt64)
    ::CoDual{nreq}, # nreq
    ::CoDual{calling_convention}, # calling convention
    a::CoDual{<:Array},
    ii::CoDual{UInt},
    args...,
) where {RT,AT,nreq,calling_convention}
    GC.@preserve args begin
        y = ccall(:jl_array_isassigned, Cint, (Any, UInt), primal(a), primal(ii))
    end
    return zero_fcodual(y), NoPullback(ntuple(_ -> NoRData(), length(args) + 8))
end

function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_type_unionall},Nw},
    ::Lifted{Val{Any},Nw},
    ::Lifted{Tuple{Val{Any},Val{Any}},Nw},
    ::Lifted{Val{0},Nw},
    ::Lifted{Val{:ccall},Nw},
    a::Lifted,
    b::Lifted,
) where {Nw}
    y = ccall(:jl_type_unionall, Any, (Any, Any), primal(a), primal(b))
    return Lifted{typeof(y),Nw}(y, NoDual())
end
function rrule!!(
    ::CoDual{typeof(_foreigncall_)},
    ::CoDual{Val{:jl_type_unionall}},
    ::CoDual{Val{Any}}, # return type
    ::CoDual{Tuple{Val{Any},Val{Any}}}, # arg types
    ::CoDual{Val{0}}, # number of required args
    ::CoDual{Val{:ccall}},
    a::CoDual,
    b::CoDual,
)
    y = ccall(:jl_type_unionall, Any, (Any, Any), primal(a), primal(b))
    return zero_fcodual(y), NoPullback(ntuple(_ -> NoRData(), 8))
end

@zero_derivative MinimalCtx Tuple{typeof(Base.has_free_typevars),Any}

@is_primitive MinimalCtx Tuple{typeof(deepcopy),Any}
# Copy primal and V through one shared `IdDict` walk: independent `deepcopy` calls would sever
# the internal aliasing (e.g. `NDualArray.primal === primal(slot)`), so the copy's inner `.value`
# would read a stale third array after the copied primal is mutated. (`deepcopy(x::Lifted)` of
# the whole slot would also work, but `deepcopy_internal(::Lifted, ...)` defeats inference.)
function frule!!(::Lifted{typeof(deepcopy),Nw}, x::Lifted{P,Nw,V}) where {Nw,P,V}
    d = IdDict()
    p = Base.deepcopy_internal(primal(x), d)::P
    return Lifted{P,Nw}(p, Base.deepcopy_internal(tangent(x), d)::V)
end
function rrule!!(::CoDual{typeof(deepcopy)}, x::CoDual)
    fdx = tangent(x)
    dx = zero_rdata(primal(x))
    y = deepcopy(x)
    fdy = tangent(y)
    function deepcopy_pb!!(dy)
        increment!!(fdx, fdy)
        return NoRData(), increment!!(dx, dy)
    end
    return y, deepcopy_pb!!
end

# `Type`, not `DataType`: the type-value arg lifts to `Lifted{Type{T}}`, and `@zero_derivative`
# emits the forward sig `Lifted{<:Type}`. Under `@nospecialize` (e.g. in `test_frule_correctness`)
# inference widens that slot to the existential `Lifted{Type{S}} where S`, which is `<: Lifted{<:Type}`
# but NOT `<: Lifted{<:DataType}` (`Lifted` is invariant; `Type{S} <: DataType` only for concrete `S`).
# With `DataType` the existential matched no method, so the call inferred `Union{}` → `unreachable` →
# SIGILL once the concrete arg dispatched at runtime. This also broadens the REVERSE primitive from
# `DataType` to `Type` — deliberate and harmless, since the derivative is zero either way.
@zero_derivative MinimalCtx Tuple{typeof(fieldoffset),Type,Integer}
@zero_derivative MinimalCtx Tuple{Type{UnionAll},TypeVar,Any}
@zero_derivative MinimalCtx Tuple{Type{UnionAll},TypeVar,Type}
@zero_derivative MinimalCtx Tuple{typeof(hash),Vararg}

function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_string_ptr},Nw},
    args::Vararg{Lifted,M},
) where {Nw,M}
    y = _foreigncall_(Val(:jl_string_ptr), tuple_map(primal, args)...)
    # Returns a `Ptr{UInt8}` — tangent is structurally non-differentiable.
    return Lifted{typeof(y),Nw}(y, NoDual())
end

function rrule!!(
    f::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{:jl_string_ptr}}, args::Vararg{CoDual,N}
) where {N}
    x = tuple_map(primal, args)
    pb!! = NoPullback((NoRData(), NoRData(), tuple_map(_ -> NoRData(), args)...))
    return uninit_fcodual(_foreigncall_(Val(:jl_string_ptr), x...)), pb!!
end

for name in (:jl_get_world_counter, :jl_matching_methods)
    @eval function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{$(QuoteNode(name))},Nw},
        args::Vararg{Lifted,M},
    ) where {Nw,M}
        y = _foreigncall_(Val($(QuoteNode(name))), tuple_map(primal, args)...)
        return Lifted{typeof(y),Nw}(y, NoDual())
    end
    @eval function rrule!!(
        f::CoDual{typeof(_foreigncall_)},
        n::CoDual{Val{$(QuoteNode(name))}},
        args::Vararg{CoDual,N},
    ) where {N}
        return zero_adjoint(f, n, args...)
    end
end

for (name, P) in
    ((Symbol("llvm.powi.f32.i32"), Float32), (Symbol("llvm.powi.f64.i32"), Float64))
    @eval function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw},
        ::Lifted{Val{$(QuoteNode(name))},Nw},
        ::Lifted{Val{$P},Nw},
        ::Lifted{Tuple{Val{$P},Val{Int32}},Nw},
        ::Lifted{Val{0},Nw},
        ::Lifted{Val{:llvmcall},Nw},
        x::Lifted{$P,Nw,NDual{$P,Nw}},
        n::Lifted{Int32,Nw},
        ::Lifted{Int32,Nw},
        ::Lifted{$P,Nw,NDual{$P,Nw}},
    ) where {Nw}
        _x = primal(x)
        _n = primal(n)
        y = Base.FastMath.pow_fast(_x, _n)
        # Scale only the partials and set V.value to `y`. A naive `grad * tangent(x)` multiplies
        # the inner NDual, scaling `.value` to `grad * x_p` and breaking the V.value === primal
        # invariant (mirrors the `pow_fast` frule in rules_via_nfwd.jl).
        grad = Nfwd._nfwd_pow_grad_x(_x, $P(_n), float(y))
        return Lifted{$P,Nw}(y, NDual{$P,Nw}(y, Nfwd._pt_scale(tangent(x).partials, grad)))
    end

    @eval function rrule!!(
        ::CoDual{typeof(_foreigncall_)},
        ::CoDual{Val{$(QuoteNode(name))}},
        ::CoDual{Val{$P}},
        ::CoDual{Tuple{Val{$P},Val{Int32}}},
        ::CoDual{Val{0}},
        ::CoDual{Val{:llvmcall}},
        x::CoDual{$P},
        n::CoDual{Int32},
        n_dup::CoDual{Int32},
        x_dup::CoDual{$P},
    )
        _x = primal(x)
        _n = primal(n)
        y = Base.FastMath.pow_fast(_x, _n)
        function llvm_powi_pb!!(dy::$P)
            dx = Nfwd._nfwd_pow_grad_x(_x, $P(_n), float(y)) * dy
            return (
                NoRData(),
                NoRData(),
                NoRData(),
                NoRData(),
                NoRData(),
                dx,
                NoRData(),
                NoRData(),
                zero_rdata(primal(x_dup)),
            )
        end
        return zero_fcodual(y), llvm_powi_pb!!
    end
end

function unexpected_foreigncall_error(name)
    throw(
        error(
            "AD has hit a :($name) ccall. This should not happen. " *
            "Please open an issue with a minimal working example in order to reproduce. ",
            "This is true unless you have intentionally written a ccall to :$(name), ",
            "in which case you must write a :foreigncall rule. It may not be possible ",
            "to implement a :foreigncall rule if too much type information has been lost ",
            "in which case your only recourse is to write a rule for whichever Julia ",
            "function calls this one (and retains enough type information).",
        ),
    )
end

for name in [
    :(:jl_alloc_array_1d),
    :(:jl_alloc_array_2d),
    :(:jl_alloc_array_3d),
    :(:jl_new_array),
    :(:jl_array_grow_end),
    :(:jl_array_del_end),
    :(:jl_array_copy),
    :(:jl_object_id),
    :(:jl_type_intersection),
    :(:memset),
    :(:jl_get_tls_world_age),
    :(:memmove),
    :(:jl_array_sizehint),
    :(:jl_array_del_at),
    :(:jl_array_grow_at),
    :(:jl_array_del_beg),
    :(:jl_array_grow_beg),
    :(:jl_value_ptr),
    :(:jl_type_unionall),
    :(:jl_threadid),
    :(:memhash_seed),
    :(:memhash32_seed),
    :(:jl_get_field_offset),
]
    @eval function _foreigncall_(
        ::Val{$name}, ::Val{RT}, AT::Tuple, ::Val{nreq}, ::Val{calling_convention}, x...
    ) where {RT,nreq,calling_convention}
        return unexpected_foreigncall_error($name)
    end
    @eval function frule!!(
        ::Lifted{typeof(_foreigncall_),Nw}, ::Lifted{Val{$name},Nw}, args...
    ) where {Nw}
        return unexpected_foreigncall_error($name)
    end
    @eval function rrule!!(::CoDual{typeof(_foreigncall_)}, ::CoDual{Val{$name}}, args...)
        return unexpected_foreigncall_error($name)
    end
end

function hand_written_rule_test_cases(rng_ctor, ::Val{:foreigncall})
    _x = Ref(5.0)
    _dx = randn_tangent(Xoshiro(123456), _x)

    _a, _da = randn(5), randn(5)
    _b, _db = randn(4), randn(4)
    ptr_a, ptr_da = pointer(_a), pointer(_da)
    ptr_b, ptr_db = pointer(_b), pointer(_db)

    test_cases = Any[
        (false, :stability, nothing, Base.allocatedinline, Float64),
        (false, :stability, nothing, Base.allocatedinline, Vector{Float64}),
        (false, :stability, nothing, objectid, 5.0),
        (true, :stability, nothing, objectid, randn(5)),
        (true, :stability, nothing, pointer_from_objref, _x),
        (
            true,
            :none, # primal is unstable
            (lb=1e-3, ub=250),
            unsafe_pointer_to_objref,
            CoDual(
                pointer_from_objref(_x),
                bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(_dx)),
            ),
        ),
        (false, :none, nothing, Core.Compiler.return_type, sin, Tuple{Float64}),
        (
            false,
            :none,
            (lb=1e-3, ub=100.0),
            Core.Compiler.return_type,
            Tuple{typeof(sin),Float64},
        ),
        (false, :stability, nothing, Threads.threadid),
        (false, :stability, nothing, typeintersect, Float64, Int),
        (
            true,
            :stability,
            nothing,
            unsafe_copyto!,
            CoDual(ptr_a, ptr_da),
            CoDual(ptr_b, ptr_db),
            4,
        ),
        (false, :stability, nothing, deepcopy, 5.0),
        (false, :stability, nothing, deepcopy, randn(5)),
        (false, :none, nothing, deepcopy, TestResources.MutableFoo(5.0, randn(5))),
        (false, :none, nothing, deepcopy, TestResources.StructFoo(5.0, randn(5))),
        (false, :stability, nothing, deepcopy, (5.0, randn(5))),
        (false, :stability, nothing, deepcopy, (a=5.0, b=randn(5))),
        (false, :none, nothing, fieldoffset, @NamedTuple{a::Float64, b::Int}, 1),
        (false, :none, nothing, fieldoffset, @NamedTuple{a::Float64, b::Int}, 2),
        (false, :none, nothing, UnionAll, TypeVar(:a), Real),
        (false, :none, nothing, hash, "5", UInt(3)),
        (false, :none, nothing, hash, Float64, UInt(5)),
        (false, :none, nothing, hash, Float64),
    ]
    memory = Any[_x, _dx, _a, _da, _b, _db]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:foreigncall})
    _x = Ref(5.0)

    function unsafe_copyto_tester(x::Vector{T}, y::Vector{T}, n::Int) where {T}
        GC.@preserve x y unsafe_copyto!(pointer(x), pointer(y), n)
        return x
    end

    _a, _da = randn(5), randn(5)
    _b, _db = randn(4), randn(4)
    ptr_a, ptr_da = pointer(_a), pointer(_da)
    ptr_b, ptr_db = pointer(_b), pointer(_db)
    memory = Any[_x, _a, _da, _b, _db]

    test_cases = [
        (false, :none, nothing, reshape, randn(5, 4), (4, 5)),
        (false, :none, nothing, reshape, randn(5, 4), (2, 10)),
        (false, :none, nothing, reshape, randn(5, 4), (10, 2)),
        (false, :none, nothing, reshape, randn(5, 4), (5, 4, 1)),
        (false, :none, nothing, reshape, randn(5, 4), (2, 10, 1)),
        (false, :none, nothing, unsafe_copyto_tester, randn(5), randn(3), 2),
        (false, :none, nothing, unsafe_copyto_tester, randn(5), randn(6), 4),
        (
            # Raw-pointer round-trip through `unsafe_copyto!` on a `Vector{Vector}` cannot yet
            # preserve the canonical dual at width N>1 (a Cluster-C forward limitation); the
            # width-1 path is correct, so skip only the chunked check here.
            false,
            :none,
            (skip_chunked=true,),
            unsafe_copyto_tester,
            [randn(3) for _ in 1:5],
            [randn(4) for _ in 1:6],
            4,
        ),
        (
            false,
            :none,
            (lb=0.1, ub=150),
            x -> unsafe_pointer_to_objref(pointer_from_objref(x)),
            _x,
        ),
        (false, :none, nothing, isassigned, randn(5), 4),
        (false, :none, nothing, copy, Dict{Any,Any}("A" => [5.0], [3.0] => 5.0)),
        (false, :none, nothing, x -> (Base._growbeg!(x, 2); x[1:2].=2.0), randn(5)),
        (
            false,
            :none,
            nothing,
            (t, v) -> ccall(:jl_type_unionall, Any, (Any, Any), t, v),
            TypeVar(:a),
            Real,
        ),
        (false, :none, nothing, Base.has_free_typevars, Float64),
        (false, :none, nothing, Base.has_free_typevars, Vector{Float64}),
        (
            true,
            :none,
            nothing,
            unsafe_copyto!,
            CoDual(ptr_a, ptr_da),
            CoDual(ptr_b, ptr_db),
            4,
        ),
        (
            true,
            :none,
            nothing,
            unsafe_copyto!,
            CoDual(ptr_a, ptr_da),
            CoDual(ptr_b, ptr_db),
            4,
        ),
        (false, :none, nothing, Base.get_world_counter), # jl_get_world_counter
        (
            false,
            :none,
            nothing,
            Base._methods_by_ftype, # jl_matching_methods
            Tuple{typeof(sin),Float64},
            -1,
            Base.get_world_counter(),
        ),
    ]
    return test_cases, memory
end

function throwing_rule_test_cases(::Val{:foreigncall})
    # pointer_from_objref of a value whose forward V is immutable but differentiable
    # (e.g. `NDualArray`) has no tangent-object address and must fail loudly rather than
    # emit NULL lanes that silently drop the derivative downstream.
    cases = Any[(
        ArgumentError, pointer_from_objref, (randn_lifted(Val(1), Xoshiro(123456), [1.0]),)
    )]
    return cases, Any[]
end
