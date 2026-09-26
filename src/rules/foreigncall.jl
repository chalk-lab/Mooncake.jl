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
function frule!!(::Lifted{typeof(_foreigncall_)}, args...)
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
# Mutable V uses the tangent object's address, tagged Ptr{tangent_type(Nothing)} rather
# than a per-lane partial pointer. This permits object round-trips but makes scalar loads
# fail the pointerref coherence check: value and partials have no parallel buffer.
# Only NoDual gets NULL; immutable differentiable V has no object address and must throw.
# The primal address (identity/hashing/objectid) is unchanged.
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
# Ref{<:NDualEltype} is refused at every width: a raw pointer read is invisible to the
# optimiser, which may elide the primal Ref's store. Parallel partials would preserve
# derivatives but not the primal; apparent correctness at width 1 depends on inlining.
function frule!!(
    ::Lifted{typeof(pointer_from_objref),Nw},
    x::Lifted{<:Base.RefValue{P},Nw,<:NDualRef{P,Nw}},
) where {Nw,P<:NDualEltype}
    throw(
        ArgumentError(
            "Forward-mode AD cannot take `pointer_from_objref` of a `Ref{$P}`: a read through " *
            "that address is invisible to the optimiser, which may elide the store into the " *
            "primal object, so the PRIMAL itself (not merely the derivative) can come back " *
            "wrong. Hold the value in an `Array`, or use reverse mode.",
        ),
    )
end
# Tangent-object pointers are indexed at PRIMAL field offsets, so layouts must agree.
# A zero-size non-differentiable field can shift later cotangents out of bounds.
# Admit layout-identical structs/Refs and objects with no differentiable payload.
@inline function _objref_tangent_layout_matches(::Type{P}) where {P}
    T = tangent_type(P)
    T <: MutableTangent || return false
    NT = fieldtype(T, :fields)
    # Empty payloads may supply an address to foreigncalls, but must be tagged NoTangent
    # by _objref_tangent_elt so later differentiable loads cannot access absent storage.
    sizeof(NT) === 0 && return true
    fieldcount(NT) === fieldcount(P) || return false
    for i in 1:fieldcount(P)
        fieldoffset(NT, i) === fieldoffset(P, i) || return false
        _field_slot(fieldtype(NT, i)) === _field_slot(fieldtype(P, i)) || return false
    end
    return sizeof(NT) === sizeof(P)
end

# Compare slot layout, not boxed-value size: abstract fields occupy a pointer slot.
# Size alone cannot distinguish an inline Float64 from a boxed pointer.
@inline function _field_slot(::Type{T}) where {T}
    return isbitstype(T) ? (true, sizeof(T)) : (false, sizeof(Ptr{Nothing}))
end

# What lies behind a `pointer_from_objref` tangent address: `Nothing` when the tangent is an object
# whose layout matches the primal's (checked above, so a later load is sound), `NoTangent` when the
# tangent has no payload at all.
@inline function _objref_tangent_elt(::Type{P}) where {P}
    T = tangent_type(P)
    return sizeof(fieldtype(T, :fields)) === 0 ? NoTangent : Nothing
end

function rrule!!(f::CoDual{typeof(pointer_from_objref)}, x)
    P = _typeof(primal(x))
    if !_objref_tangent_layout_matches(P)
        throw(
            ArgumentError(
                "Cannot take `pointer_from_objref` of a `$P` during reverse-mode AD: the address " *
                "would point at the tangent object, but a load or store through it indexes at " *
                "`$P`'s own field offsets, and `$(tangent_type(P))` does not share that layout. " *
                "A non-differentiable field has a zero-size tangent counterpart, so every later " *
                "field shifts and the access lands outside the tangent object. Use a struct whose " *
                "fields are all differentiable and of the same size, or reach the field directly " *
                "instead of through a raw pointer.",
            ),
        )
    end
    # Nothing tags a layout-checked object, not a uniform buffer; NoTangent tags an empty
    # payload whose address must never serve a differentiable load.
    y = CoDual(
        pointer_from_objref(primal(x)),
        VoidPtrTangent(pointer_from_objref(tangent(x)), _objref_tangent_elt(P)),
    )
    return y, NoPullback(f, x)
end

@zero_derivative MinimalCtx Tuple{typeof(CC.return_type),Vararg}

@is_primitive MinimalCtx Tuple{typeof(Base.unsafe_pointer_to_objref),Ptr}
# Recover the full tangent object for non-NULL lanes. This dynamic boundary widens
# inference: the primal and canonical V types are known only at runtime.
function frule!!(
    ::Lifted{typeof(Base.unsafe_pointer_to_objref),Nw}, x::Lifted{<:Ptr}
) where {Nw}
    y = unsafe_pointer_to_objref(primal(x))
    tx = tangent(x)
    # NULL proves non-differentiability. An uninit_* placeholder instead aliases the
    # primal address: recovering it as a tangent would let derivatives mutate user data.
    (tx isa NoDual || tx[1] == Ptr{Nothing}(0)) && return Lifted{typeof(y),Nw}(y, NoDual())
    UInt(tx[1]) == UInt(primal(x)) && throw(
        ArgumentError(
            "Forward-mode AD cannot recover a tangent object from a `Ptr` with no tangent " *
            "storage behind it: the lane pointer is the primal's own address, so the recovered " *
            "tangent would be the primal object itself and a derivative written into it would " *
            "mutate the primal. Take `pointer_from_objref` of a value that carries a tangent.",
        ),
    )
    return Lifted{typeof(y),Nw}(y, unsafe_pointer_to_objref(tx[1]))
end
# Numeric per-lane pointers distinguish Ref partial storage from the generic objref tag.
# Read P at runtime: NTuple{0,Ptr{P}} erases P, causing an unbound signature parameter.
# Width 0 is rejected by _nfwd_check_chunk_size, so eltype needs no degenerate case.
function frule!!(
    ::Lifted{typeof(Base.unsafe_pointer_to_objref),Nw},
    x::Lifted{<:Ptr,Nw,<:NTuple{Nw,Ptr{<:NDualEltype}}},
) where {Nw}
    ref = unsafe_pointer_to_objref(primal(x))
    P = eltype(eltype(tangent(x)))
    # Lane 1 is the original partials object's base address. Recover it rather than copy:
    # writes through the recovered Ref must reach the original slot's partials.
    partials = unsafe_pointer_to_objref(
        Ptr{Nothing}(UInt(tangent(x)[1]))
    )::Base.RefValue{NTuple{Nw,P}}
    return Lifted{typeof(ref),Nw}(ref, NDualRef{P,Nw}(partials))
end
# The tangent of a `Ptr{Nothing}` carries its address alongside the erased element width; every other
# pointer tangent IS the address.
@inline _void_ptr_addr(p::VoidPtrTangent) = p.p
@inline _void_ptr_addr(p::Ptr) = p

function rrule!!(f::CoDual{typeof(Base.unsafe_pointer_to_objref)}, x::CoDual{<:Ptr})
    y = CoDual(
        unsafe_pointer_to_objref(primal(x)),
        unsafe_pointer_to_objref(_void_ptr_addr(tangent(x))),
    )
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
# Per-lane pointers cover both scalar partial buffers and nested-array element duals;
# their element type need not match the primal pointer's.
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    dest::Lifted{P,Nw,<:NTuple{Nw,Ptr}},
    src::Lifted{P,Nw,<:NTuple{Nw,Ptr}},
    n::Lifted,
) where {Nw,P<:Ptr}
    _n = primal(n)
    dest_partials = tangent(dest)
    src_partials = tangent(src)
    @inbounds for lane in 1:Nw
        IntrinsicsWrappers._check_fwd_tangent_ptr_addressable(
            primal(dest), dest_partials[lane]
        )
        IntrinsicsWrappers._check_fwd_tangent_ptr_addressable(
            primal(src), src_partials[lane]
        )
    end
    unsafe_copyto!(primal(dest), primal(src), _n)
    @inbounds for lane in 1:Nw
        unsafe_copyto!(dest_partials[lane], src_partials[lane], _n)
    end
    return dest
end
# Mixed V needs an explicit diagnostic under the broad primitive declaration. NoDual
# proves only absence of partial storage, not that copying constants in would be sound.
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    ::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    ::Lifted{Ptr{T},Nw,NoDual},
    ::Lifted,
) where {Nw,T}
    throw(ArgumentError(IntrinsicsWrappers._NODUAL_DIFF_PTR_MSG))
end
function frule!!(
    ::Lifted{typeof(unsafe_copyto!),Nw},
    ::Lifted{Ptr{T},Nw,NoDual},
    ::Lifted{Ptr{T},Nw,<:NTuple{Nw,Ptr}},
    ::Lifted,
) where {Nw,T}
    throw(ArgumentError(IntrinsicsWrappers._NODUAL_DIFF_PTR_MSG))
end
# NoDual pointers have non-differentiable elements; nested differentiable arrays instead
# carry per-lane pointers and use the NTuple overload.
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
    # Both pointers are dereferenced below, so both must address real tangent
    # bytes. Same guard as the load/store rules and reverse `unsafe_wrap`; it lives
    # in `IntrinsicsWrappers`, not the top-level module.
    IntrinsicsWrappers._check_tangent_ptr(primal(dest), tangent(dest))
    IntrinsicsWrappers._check_tangent_ptr(primal(src), tangent(src))
    _n = primal(n)

    # Exact self-copy is identity: accumulating then restoring the same buffer would
    # erase downstream cotangents. Partial overlap still uses snapshot-and-restore.
    if primal(dest) === primal(src)
        return dest, NoPullback(ntuple(_ -> NoRData(), 4))
    end

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

# Reshape primal and block without copying: their aliasing must agree. E covers real
# and complex NDualEltype; the free V element type admits both corresponding duals.
function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_reshape_array},Nw},
    ::Lifted{Val{Array{E,M}},Nw},
    ::Lifted{Tuple{Val{Any},Val{Any},Val{Any}},Nw},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    ::Lifted{Type{Array{E,M}},Nw},
    a::Lifted{Array{E,D},Nw,<:NDualArray{E,Nw,D,Array{E,D}}},
    dims::Lifted,
) where {Nw,E<:NDualEltype,M,D}
    d = primal(dims)
    y = ccall(:jl_reshape_array, Array{E,M}, (Any, Any, Any), Array{E,M}, primal(a), d)
    # A new header over the same parent preserves the original partials storage.
    new_block = reshape(getfield(tangent(a), :partials_block), (Nw, d...))
    return Lifted{Array{E,M},Nw}(y, NDualArray{E,Nw,M,Array{E,M}}(y, new_block))
end
# Element-wise Array V covers non-differentiable and structured differentiable elements;
# reshape it in lockstep with the primal. Numeric leaves use NDualArray above.
function frule!!(
    ::Lifted{typeof(_foreigncall_),Nw},
    ::Lifted{Val{:jl_reshape_array},Nw},
    ::Lifted{Val{Array{P,M}},Nw},
    ::Lifted{Tuple{Val{Any},Val{Any},Val{Any}},Nw},
    ::Lifted, # nreq
    ::Lifted, # calling convention
    ::Lifted{Type{Array{P,M}},Nw},
    a::Lifted{<:Array,Nw,<:Array{VE}},
    dims::Lifted,
) where {Nw,P,M,VE}
    d = primal(dims)
    y = ccall(:jl_reshape_array, Array{P,M}, (Any, Any, Any), Array{P,M}, primal(a), d)
    v = ccall(:jl_reshape_array, Array{VE,M}, (Any, Any, Any), Array{VE,M}, tangent(a), d)
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
    ::Lifted,  # return type (Cint)
    ::Lifted,  # arg types
    ::Lifted,  # nreq
    ::Lifted,  # calling convention
    a::Lifted,
    ii::Lifted,
    args...,
) where {Nw}
    GC.@preserve args begin
        y = ccall(:jl_array_isassigned, Cint, (Any, UInt), primal(a), primal(ii))
    end
    return zero_lifted(Val(Nw), y)
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
    return zero_lifted(Val(Nw), y)
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

# Type admits inferred existential Lifted{Type{S}} slots; DataType does not, since
# Lifted is invariant. Missing that dispatch can infer Union{} and emit unreachable.
# Reverse also admits Type: fieldoffset has zero derivative for every type value.
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
    # Canonical zero V: UInt uses NoDual, but Vector{Any} uses Vector{Any}.
    @eval function frule!!(
        f::Lifted{typeof(_foreigncall_),Nw},
        n::Lifted{Val{$(QuoteNode(name))},Nw},
        args::Vararg{Lifted,M},
    ) where {Nw,M}
        return zero_derivative(f, n, args...)
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
        # Preserve V.value === primal; scale only partials.
        grad = Nfwd._nfwd_pow_grad_x(_x, $P(_n), float(y))
        # Inactive lanes stay zero even at infinite gradients (e.g. x=0, n<0).
        return Lifted{$P,Nw}(
            y, NDual{$P,Nw}(y, Nfwd._fwd_guarded_scale(tangent(x).partials, grad))
        )
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
            # Zero cotangents must stay zero at infinite local gradients.
            dx = nan_tangent_guard(dy, Nfwd._nfwd_pow_grad_x(_x, $P(_n), float(y)) * dy)
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
        # Refused in forward mode: a read through the address is invisible to the optimiser,
        # which may elide the store into the primal `Ref`. Reverse keeps its ordinary test.
        (
            true,
            :stability,
            (throws=(ArgumentError, "invisible to the optimiser"), mode=ForwardMode),
            pointer_from_objref,
            _x,
        ),
        (true, :stability, (mode=ReverseMode,), pointer_from_objref, _x),
        (
            # _dx is a reverse tangent, not canonical forward V, so skip_forward is required.
            true,
            :none, # primal is unstable
            (lb=1e-3, ub=250, skip_forward=true),
            unsafe_pointer_to_objref,
            CoDual(
                pointer_from_objref(_x), VoidPtrTangent(pointer_from_objref(_dx), Nothing)
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
        # A threading foreigncall whose result is a non-differentiable `Cint`: its V must be
        # `NoDual`. The ABI mirrors the normalized call, which carries no call arguments.
        # Forward only: reverse refuses these outright, which the throwing case below pins.
        (
            false,
            :none,
            (skip_reverse=true,),
            _foreigncall_,
            Val(:jl_in_threaded_region),
            Val{Cint}(),
            (),
            Val{0}(),
            Val{:ccall}(),
        ),
    ]
    throwing_rows, throwing_memory = _foreigncall_throwing_rows()
    test_cases = vcat(Any[test_cases...], Any[_throwing_row(c) for c in throwing_rows])
    memory = vcat(Any[_x, _dx, _a, _da, _b, _db], throwing_memory)
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:foreigncall})
    _x = Ref(5.0)

    function unsafe_copyto_tester(x::Vector{T}, y::Vector{T}, n::Int) where {T}
        GC.@preserve x y unsafe_copyto!(pointer(x), pointer(y), n)
        return x
    end

    # One argument avoids the forward repeated-argument refusal and exercises self-copy:
    # snapshot-and-restore must not erase cotangents accumulated in the shared buffer.
    function unsafe_self_copy_tester(x::Vector{T}, n::Int) where {T}
        GC.@preserve x unsafe_copyto!(pointer(x), pointer(x), n)
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
        # Complex reshape: the forward frule must be element-type-agnostic across `NDualEltype`
        # (the V is `NDualArray{Complex{R}, …}`). On Julia 1.10 this lowers to a
        # `jl_reshape_array` foreigncall, which the frule must handle for complex element types too.
        (false, :none, nothing, reshape, randn(ComplexF64, 5, 4), (4, 5)),
        # Reshape of an array of differentiable struct / tuple elements (Array{FloatPair},
        # Array{Tuple{Float64,Float64}}): forward mode must reshape primal and V in lockstep.
        (
            false,
            :none,
            nothing,
            x -> (
                v=[TestResources.FloatPair(x, 2x), TestResources.FloatPair(3x, 4x)];
                r=reshape(v, 2, 1);
                r[1, 1].a + r[2, 1].b
            ),
            1.0,
        ),
        (
            false,
            :none,
            nothing,
            x -> (v=[(x, 2x), (3x, 4x)]; r=reshape(v, 2, 1); r[1, 1][1] + r[2, 1][2]),
            1.0,
        ),
        # `skip_chunked`: these take a raw pointer to a float array, whose element-major partials
        # block stores each lane with stride N, so no dense per-lane buffer exists for a pointer to
        # address. The guard fires loudly at width > 1; the width-1 path is correct.
        (false, :none, (skip_chunked=true,), unsafe_copyto_tester, randn(5), randn(3), 2),
        (false, :none, (skip_chunked=true,), unsafe_self_copy_tester, randn(5), 3),
        (false, :none, (skip_chunked=true,), unsafe_copyto_tester, randn(5), randn(6), 4),
        (
            # Nested duals have no dense per-lane pointer buffer at width > 1. The guard
            # throws (missing frule on 1.10); width 1 remains supported.
            false,
            :none,
            (skip_chunked=true,),
            unsafe_copyto_tester,
            [randn(3) for _ in 1:5],
            [randn(4) for _ in 1:6],
            4,
        ),
        (
            # Abstract elements occupy reference slots; the pointer guard must not call
            # sizeof(Any), which throws instead of checking storage.
            false,
            :none,
            (skip_chunked=true,),
            unsafe_copyto_tester,
            Any[randn(3) for _ in 1:5],
            Any[randn(4) for _ in 1:6],
            4,
        ),
        # Forward also refuses sound object round-trips: distinguishing raw byte reads
        # would require an objref tag outside the canonical NTuple{N,Ptr{T}} contract.
        # This conservative refusal is intentional until that representation changes.
        (
            (false, :none, opts, f, _x) for f in (
                x -> unsafe_pointer_to_objref(pointer_from_objref(x)),
                # Writes through the recovered alias must reach the original partials storage.
                x -> (
                    r=unsafe_pointer_to_objref(pointer_from_objref(x));
                    r[]=r[] * 3.0;
                    x[]
                ),
            ) for opts in (
                (throws=(ArgumentError, "invisible to the optimiser"), mode=ForwardMode),
                (lb=0.1, ub=150, mode=ReverseMode),
            )
        )...,
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

function unsafe_copyto_retyped_bytes(x, b)
    y = similar(x)
    GC.@preserve x y b unsafe_copyto!(pointer(y), Ptr{Float64}(pointer(b)), length(y))
    return sum(x) + sum(y)
end

# A primal whose tangent layout differs from its own: `a`'s tangent is `NoTangent` (zero-size), so
# `b`'s cotangent sits at offset 0 while `b` itself sits at offset 8.
mutable struct MismatchedLayout
    a::Int
    b::Float64
end

function _foreigncall_throwing_rows()
    # pointer_from_objref of a value whose forward V is immutable but differentiable
    # (e.g. `NDualArray`) has no tangent-object address and must fail loudly rather than
    # emit NULL lanes that silently drop the derivative downstream.
    cases = Any[(ArgumentError, pointer_from_objref, ([1.0],), (; mode=ForwardMode))]
    memory = Any[]
    # Tangent-object loads use primal offsets; a non-differentiable field shifts them.
    push!(
        cases,
        (
            (ArgumentError, "does not share that layout"),
            pointer_from_objref,
            (MismatchedLayout(7, 3.0),),
            (; mode=ReverseMode),
        ),
    )
    # A ready-made slot expresses the uninit_* placeholder: recovering the primal's own
    # address as a tangent would corrupt it; a raw Ptr seed cannot express this shape.
    objref_target = [1.0, 2.0]
    push!(memory, objref_target)
    let p = pointer_from_objref(objref_target)
        push!(
            cases,
            (
                (ArgumentError, "the lane pointer is the primal's own address"),
                Base.unsafe_pointer_to_objref,
                (Lifted{typeof(p),1}(p, (p,)),),
                (; mode=ForwardMode),
            ),
        )
    end
    # Empty tangent objects may supply foreigncall addresses, but differentiable loads
    # must refuse them rather than read/store beyond the zero-byte payload.
    function zero_payload_objref_load(r::Base.RefValue{NTuple{8,Int}}, x::Float64)
        return GC.@preserve r x * unsafe_load(Ptr{Float64}(pointer_from_objref(r)))
    end
    push!(
        cases,
        (
            (ArgumentError, "no tangent storage"),
            zero_payload_objref_load,
            (Base.RefValue(ntuple(_ -> 0, Val(8))), 2.0),
            (; mode=ReverseMode),
        ),
    )
    @static if VERSION >= v"1.11-rc4"
        # Nested MemoryRef duals have no dense per-lane buffer at width > 1;
        # ptr_or_offset must refuse rather than drop derivatives.
        nested = [randn(2), randn(2)]
        push!(memory, nested)
        push!(
            cases,
            (
                ArgumentError,
                lgetfield,
                (getfield(nested, :ref), Val(:ptr_or_offset)),
                (; mode=ForwardMode, chunk_size=2),
            ),
        )
        # Dynamic Symbol names avoid rewriting to lgetfield; exercise both arities.
        for extra in ((), (false,))
            push!(
                cases,
                (
                    ArgumentError,
                    getfield,
                    (getfield(nested, :ref), :ptr_or_offset, extra...),
                    (; mode=ForwardMode, chunk_size=2),
                ),
            )
        end
    else
        # Julia 1.10 has no `MemoryRef`; the same guard sits on the legacy-array raw pointer
        # (`jl_array_ptr`), and must fail loudly at width > 1 for the same reason.
        push!(
            cases, (ArgumentError, pointer, (randn(2),), (; mode=ForwardMode, chunk_size=2))
        )
    end
    copy_bytes = zeros(UInt8, 24)
    push!(memory, copy_bytes)
    # Reverse `unsafe_copyto!` dereferences BOTH tangent pointers, and the source has none: it is
    # re-typed off a non-differentiable buffer, which the `bitcast` rrule refuses on every version.
    push!(
        cases,
        (
            (ArgumentError, "no tangent storage"),
            unsafe_copyto_retyped_bytes,
            (randn(3), copy_bytes),
            (; mode=ReverseMode),
        ),
    )
    # Mixed V: real destination partials but NoDual source, on every Julia version.
    push!(
        cases,
        (
            (ArgumentError, "forward representation is `NoDual`"),
            unsafe_copyto_retyped_bytes,
            (randn(3), copy_bytes),
            (; mode=ForwardMode),
        ),
    )
    # Ready-made slots express a coherent destination and an uninit_* source placeholder
    # pointing at primal storage; raw Ptr seeding cannot express this shape.
    cp_dest, cp_dest_t, cp_src = randn(3), randn(3), randn(3)
    append!(memory, (cp_dest, cp_dest_t, cp_src))
    push!(
        cases,
        (
            (ArgumentError, "forward representation is `NoDual`"),
            unsafe_copyto!,
            (
                Lifted{Ptr{Float64},1}(pointer(cp_dest), (pointer(cp_dest_t),)),
                Lifted{Ptr{Float64},1}(pointer(cp_src), (pointer(cp_src),)),
                3,
            ),
            (; mode=ForwardMode),
        ),
    )
    # Reverse refuses threading rather than returning an incorrect gradient.
    push!(
        cases,
        (
            (ErrorException, "Differentiating through threading is not safe"),
            _foreigncall_,
            (Val(:jl_in_threaded_region), Val{Cint}(), (), Val{0}(), Val{:ccall}()),
            (; mode=ReverseMode),
        ),
    )
    for name in [
        :jl_alloc_array_1d,
        :jl_alloc_array_2d,
        :jl_alloc_array_3d,
        :jl_new_array,
        :jl_array_copy,
        :jl_type_intersection,
        :memset,
        :jl_get_tls_world_age,
        :memmove,
        :jl_object_id,
        :jl_array_sizehint,
        :jl_array_grow_beg,
        :jl_array_grow_end,
        :jl_array_grow_at,
        :jl_array_del_beg,
        :jl_array_del_end,
        :jl_array_del_at,
        :jl_value_ptr,
        :jl_threadid,
        :memhash_seed,
        :memhash32_seed,
        :jl_get_field_offset,
    ]
        push!(
            cases,
            (
                (ErrorException, "AD has hit a :($name) ccall"),
                _foreigncall_,
                (Val(name),),
                (;),
            ),
        )
    end
    return cases, memory
end
