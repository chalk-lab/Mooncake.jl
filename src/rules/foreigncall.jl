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
function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N}, args::Vararg{Mooncake.Lifted,M}
) where {N,M}
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
            "explicitly written." *
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

# Broader `Any` slot rather than `Type`: `Base.allocatedinline` only accepts
# Type at runtime, but inference can widen the slot to `Any` (e.g. when used
# inside `all(allocatedinline, fieldtypes(P))` checks), producing a
# `Lifted{Any, ...}` slot that doesn't match `Lifted{<:Type}`.
@zero_derivative MinimalCtx Tuple{typeof(Base.allocatedinline),Any}

@zero_derivative MinimalCtx Tuple{typeof(objectid),Any}

@is_primitive MinimalCtx Tuple{typeof(pointer_from_objref),Any}
# `tangent(::Dual{P, NTangent{Tuple{T}}})` returns the NTangent wrapper which
# is itself immutable, so `pointer_from_objref(NTangent)` fails. Unwrap the
# singleton lane to get the underlying (mutable) tangent object before
# taking its pointer.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(pointer_from_objref),N}, x::Mooncake.Lifted
) where {N}
    y = pointer_from_objref(primal(x))
    dys = ntuple(Val(N)) do k
        bitcast(Ptr{tangent_type(Nothing)}, pointer_from_objref(Mooncake.tangent(x, k)))
    end
    return Mooncake.Lifted{Ptr{Nothing},N}(y, Mooncake.NTangent(dys))
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
# `tangent(::Dual{Ptr, NTangent{Tuple{Ptr}}})` returns the NTangent wrapper.
# Reuse `Mooncake._ntangent_unwrap_singleton` (defined in tangents/dual.jl) so
# `unsafe_pointer_to_objref` receives the bare Ptr it expects.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(Base.unsafe_pointer_to_objref),N}, x::Mooncake.Lifted{<:Ptr}
) where {N}
    y = unsafe_pointer_to_objref(primal(x))
    dys = ntuple(k -> unsafe_pointer_to_objref(Mooncake.tangent(x, k)), Val(N))
    return Mooncake.Lifted{_typeof(y),N}(y, Mooncake.NTangent(dys))
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(unsafe_copyto!),N},
    dest::Mooncake.Lifted{Ptr{T}},
    src::Mooncake.Lifted{Ptr{T}},
    n::Mooncake.Lifted,
) where {N,T}
    pn = primal(n)
    unsafe_copyto!(primal(dest), primal(src), pn)
    # Unwrap NTangent-wrapped Ptr tangent at this boundary.
    unsafe_copyto!(
        Mooncake._ntangent_unwrap_singleton(tangent(dest)),
        Mooncake._ntangent_unwrap_singleton(tangent(src)),
        pn,
    )
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

# Direct Lifted bodies per inner V shape of `a`. Wrapper-exception V
# (`Dual{Array{P,N_a}, Array{T,N_a}}`): reshape primal and tangent
# separately via two ccalls.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N},
    ::Mooncake.Lifted{Val{:jl_reshape_array}},
    ::Mooncake.Lifted{Val{Array{P,M}}},
    ::Mooncake.Lifted{Tuple{Val{Any},Val{Any},Val{Any}}},
    ::Mooncake.Lifted, # nreq
    ::Mooncake.Lifted, # calling convention
    ::Mooncake.Lifted{Type{Array{P,M}}},
    a::Mooncake.Lifted{<:Array{P},N,V_a},
    dims::Mooncake.Lifted,
) where {N,P,M,N_a,T,V_a<:Dual{<:Array{P,N_a},<:Array{T,N_a}}}
    d = primal(dims)
    bare_a = Mooncake._unlift(a)
    y = ccall(:jl_reshape_array, Array{P,M}, (Any, Any, Any), Array{P,M}, primal(bare_a), d)
    dy = ccall(
        :jl_reshape_array, Array{T,M}, (Any, Any, Any), Array{T,M}, tangent(bare_a), d
    )
    return Mooncake.Lifted{Array{P,M},N}(y, dy)
end
# Canonical width-1 NDual-array V: reshape Array{NDual{P,1}} directly —
# NDual elements carry primal+tangent so one ccall suffices.
@inline function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N},
    ::Mooncake.Lifted{Val{:jl_reshape_array}},
    ::Mooncake.Lifted{Val{Array{P,M}}},
    ::Mooncake.Lifted{Tuple{Val{Any},Val{Any},Val{Any}}},
    ::Mooncake.Lifted, # nreq
    ::Mooncake.Lifted, # calling convention
    ::Mooncake.Lifted{Type{Array{P,M}}},
    a::Mooncake.Lifted{<:Array{P},N,V_a},
    dims::Mooncake.Lifted,
) where {N,P<:IEEEFloat,M,N_a,V_a<:AbstractArray{NDual{P,1},N_a}}
    bare_a = Mooncake._unlift(a)
    result = ccall(
        :jl_reshape_array,
        Array{NDual{P,1},M},
        (Any, Any, Any),
        Array{NDual{P,1},M},
        bare_a,
        primal(dims),
    )
    return Mooncake.Lifted{Array{P,M},N,typeof(result)}(result)
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

@inline function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N},
    ::Mooncake.Lifted{Val{:jl_array_isassigned}},
    ::Mooncake.Lifted{RT},
    ::Mooncake.Lifted{AT},
    ::Mooncake.Lifted{nreq},
    ::Mooncake.Lifted{calling_convention},
    a::Mooncake.Lifted{<:Array},
    ii::Mooncake.Lifted{UInt},
    args::Vararg{Mooncake.Lifted,M},
) where {N,RT,AT,nreq,calling_convention,M}
    arr = primal(a)
    GC.@preserve args begin
        y = ccall(:jl_array_isassigned, Cint, (Any, UInt), arr, primal(ii))
    end
    return zero_lifted(Val(N), y)
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

@inline function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N},
    ::Mooncake.Lifted{Val{:jl_type_unionall}},
    ::Mooncake.Lifted{Val{Any}},
    ::Mooncake.Lifted{Tuple{Val{Any},Val{Any}}},
    ::Mooncake.Lifted{Val{0}},
    ::Mooncake.Lifted{Val{:ccall}},
    a::Mooncake.Lifted,
    b::Mooncake.Lifted,
) where {N}
    return zero_lifted(
        Val(N), ccall(:jl_type_unionall, Any, (Any, Any), primal(a), primal(b))
    )
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
@inline function frule!!(
    ::Mooncake.Lifted{typeof(deepcopy),N}, x::Mooncake.Lifted
) where {N}
    inner = Mooncake._unlift(x)
    return Mooncake.Lifted{_typeof(primal(inner)),N}(deepcopy(inner))
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

@zero_derivative MinimalCtx Tuple{typeof(fieldoffset),DataType,Integer}
@zero_derivative MinimalCtx Tuple{Type{UnionAll},TypeVar,Any}
@zero_derivative MinimalCtx Tuple{Type{UnionAll},TypeVar,Type}
@zero_derivative MinimalCtx Tuple{typeof(hash),Vararg}

@inline function frule!!(
    ::Mooncake.Lifted{typeof(_foreigncall_),N},
    ::Mooncake.Lifted{Val{:jl_string_ptr}},
    args::Vararg{Mooncake.Lifted,M},
) where {N,M}
    bare_args = ntuple(i -> Mooncake._unlift(args[i]), Val(M))
    # Use `__get_primal` (recurses into tuples) instead of a flat `isa Dual`
    # check: the AT slot lifts to `Tuple{Dual{Val{Any}, NoTangent}, ...}`, and
    # the @generated `_foreigncall_` body indexes `AT.parameters` for raw `Val`
    # types — Dual-wrapped Vals trip `_get_arg_type`.
    primal_args = tuple_map(Mooncake.__get_primal, bare_args)
    return zero_lifted(Val(N), _foreigncall_(Val(:jl_string_ptr), primal_args...))
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
        f::Lifted{typeof(_foreigncall_),N},
        n::Lifted{Val{$(QuoteNode(name))}},
        args::Vararg{Lifted,M},
    ) where {N,M}
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
        ::Lifted{typeof(_foreigncall_),N},
        ::Lifted{Val{$(QuoteNode(name))}},
        ::Lifted{Val{$P}},
        ::Lifted{Tuple{Val{$P},Val{Int32}}},
        ::Lifted{Val{0}},
        ::Lifted{Val{:llvmcall}},
        x::Lifted{$P,N},
        n::Lifted{Int32},
        ::Lifted{Int32},
        ::Lifted{$P},
    ) where {N}
        _x = primal(x)
        _n = primal(n)
        y = Base.FastMath.pow_fast(_x, _n)
        coeff = Nfwd._nfwd_pow_grad_x(_x, $P(_n), float(y))
        lanes = ntuple(lane -> coeff * tangent(x, lane), Val(N))
        return Lifted{$P,N}(y, lanes)
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
        ::Lifted{typeof(_foreigncall_),N}, ::Lifted{Val{$name}}, args::Vararg{Lifted,M}
    ) where {N,M}
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
            false,
            :none,
            nothing,
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
