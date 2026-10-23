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
function frule!!(::Dual{typeof(_foreigncall_)}, args...)
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

# A null tangent pointer also refuses recovering an object whose tangent layout differs.
@is_primitive MinimalCtx Tuple{typeof(pointer_from_objref),Any}
function frule!!(::Dual{typeof(pointer_from_objref)}, x)
    y = pointer_from_objref(primal(x))
    dx = tangent(x)
    dy = Ptr{NoTangent}(C_NULL)
    if !(dx isa NoTangent)
        dy = bitcast(Ptr{NoTangent}, pointer_from_objref(dx))
        dy = _build_tangent_ptr(dy, typeof(primal(x)), typeof(dx), Val(true))
    end
    return Dual(y, dy)
end
function rrule!!(f::CoDual{typeof(pointer_from_objref)}, x)
    dx = tangent(x)
    dy = Ptr{NoTangent}(C_NULL)
    if !(dx isa NoFData)
        dy = bitcast(Ptr{NoTangent}, pointer_from_objref(dx))
        dy = _build_tangent_ptr(dy, typeof(primal(x)), typeof(dx), Val(true))
    end
    y = CoDual(pointer_from_objref(primal(x)), dy)
    return y, NoPullback(f, x)
end

@zero_derivative MinimalCtx Tuple{typeof(CC.return_type),Vararg}

@is_primitive MinimalCtx Tuple{typeof(Base.unsafe_pointer_to_objref),Ptr}
function _objref_tangent(y, dx::Ptr)
    tangent_type(typeof(y)) == NoTangent && return NoTangent()
    _check_tangent_ptr(bitcast(Ptr{tangent_type(typeof(y))}, dx), 1)
    return unsafe_pointer_to_objref(dx)
end
function frule!!(::Dual{typeof(Base.unsafe_pointer_to_objref)}, x::Dual{<:Ptr})
    y = unsafe_pointer_to_objref(primal(x))
    return Dual(y, _objref_tangent(y, tangent(x)))
end
function rrule!!(f::CoDual{typeof(Base.unsafe_pointer_to_objref)}, x::CoDual{<:Ptr})
    py = unsafe_pointer_to_objref(primal(x))
    y = CoDual(py, fdata(_objref_tangent(py, tangent(x))))
    return y, NoPullback(f, x)
end

@zero_derivative MinimalCtx Tuple{typeof(Threads.threadid)}
@zero_derivative MinimalCtx Tuple{typeof(typeintersect),Any,Any}

function _swap_array_entries!(x::Array, y::Vector, xinds)
    @inbounds for (i, j) in enumerate(xinds)
        if isassigned(x, j)
            tmp = x[j]
            if isassigned(y, i)
                x[j] = y[i]
            else
                Base._unsetindex!(x, j)
            end
            y[i] = tmp
        elseif isassigned(y, i)
            x[j] = y[i]
            Base._unsetindex!(y, i)
        end
    end
    return nothing
end

# unsafe_copyto! is the only function in Julia that appears to rely on a ccall to `memmove`.
# Since we can't differentiate `memmove` (due to a lack of type information), it is
# necessary to work with `unsafe_copyto!` instead.
@is_primitive MinimalCtx Tuple{typeof(unsafe_copyto!),Ptr{T},Ptr{T},Any} where {T}
function frule!!(
    ::Dual{typeof(unsafe_copyto!)}, dest::Dual{Ptr{T}}, src::Dual{Ptr{T}}, n::Dual
) where {T}
    has_tangent =
        _check_tangent_ptr(tangent(dest), primal(n)) &
        _check_tangent_ptr(tangent(src), primal(n))
    unsafe_copyto!(primal(dest), primal(src), primal(n))
    has_tangent && unsafe_copyto!(tangent(dest), tangent(src), primal(n))
    return dest
end
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)}, dest::CoDual{Ptr{T}}, src::CoDual{Ptr{T}}, n::CoDual
) where {T}
    has_tangent =
        _check_tangent_ptr(tangent(dest), primal(n)) &
        _check_tangent_ptr(tangent(src), primal(n))
    _n = primal(n)

    # Record values that will be overwritten.
    dest_copy = Vector{T}(undef, _n)
    pdest = primal(dest)
    ddest = tangent(dest)
    # Must be an empty `Vector`, not `nothing`: a union-typed snapshot makes the pullback
    # unstable.
    ddest_copy = Vector{eltype(ddest)}(undef, has_tangent ? _n : 0)
    unsafe_copyto!(pointer(dest_copy), pdest, _n)
    has_tangent && unsafe_copyto!(pointer(ddest_copy), ddest, _n)

    # Run primal computation.
    dsrc = tangent(src)
    unsafe_copyto!(primal(dest), primal(src), _n)
    has_tangent && unsafe_copyto!(ddest, dsrc, _n)

    function unsafe_copyto!_pb!!(::NoRData)
        unsafe_copyto!(pdest, pointer(dest_copy), _n)
        if has_tangent
            ddest_array = unsafe_wrap(Vector{eltype(ddest)}, ddest, _n)
            if max(ddest, dsrc) - min(ddest, dsrc) >= _n * Base.elsize(ddest_array)
                increment!!(unsafe_wrap(Vector{eltype(dsrc)}, dsrc, _n), ddest_array)
                unsafe_copyto!(ddest, pointer(ddest_copy), _n)
            else
                # Overlapping ranges require restoring ddest before accumulating into dsrc.
                _swap_array_entries!(ddest_array, ddest_copy, eachindex(ddest_copy))
                increment!!(unsafe_wrap(Vector{eltype(dsrc)}, dsrc, _n), ddest_copy)
            end
        end

        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return dest, unsafe_copyto!_pb!!
end

function frule!!(
    ::Dual{typeof(_foreigncall_)},
    ::Dual{Val{:jl_reshape_array}},
    ::Dual{Val{Array{P,M}}},
    ::Dual{Tuple{Val{Any},Val{Any},Val{Any}}},
    ::Dual, # nreq
    ::Dual, # calling convention
    x::Dual{Type{Array{P,M}}},
    a::Dual{Array{P,N},Array{T,N}},
    dims::Dual,
) where {P,T,M,N}
    d = primal(dims)
    y = ccall(:jl_reshape_array, Array{P,M}, (Any, Any, Any), Array{P,M}, primal(a), d)
    dy = ccall(:jl_reshape_array, Array{T,M}, (Any, Any, Any), Array{T,M}, tangent(a), d)
    return Dual(y, dy)
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
    ::Dual{typeof(_foreigncall_)},
    ::Dual{Val{:jl_array_isassigned}},
    ::Dual{RT}, # return type is Int32
    arg_types::Dual{AT}, # arg types are (Any, UInt64)
    ::Dual{nreq}, # nreq
    ::Dual{calling_convention}, # calling convention
    a::Dual{<:Array},
    ii::Dual{UInt},
    args...,
) where {RT,AT,nreq,calling_convention}
    GC.@preserve args begin
        y = ccall(:jl_array_isassigned, Cint, (Any, UInt), primal(a), primal(ii))
    end
    return zero_dual(y)
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
    ::Dual{typeof(_foreigncall_)},
    ::Dual{Val{:jl_type_unionall}},
    ::Dual{Val{Any}}, # return type
    ::Dual{Tuple{Val{Any},Val{Any}}}, # arg types
    ::Dual{Val{0}}, # number of required args
    ::Dual{Val{:ccall}},
    a::Dual,
    b::Dual,
)
    return zero_dual(ccall(:jl_type_unionall, Any, (Any, Any), primal(a), primal(b)))
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
frule!!(::Dual{typeof(deepcopy)}, x::Dual) = Dual(deepcopy(primal(x)), deepcopy(tangent(x)))
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

function frule!!(
    ::Dual{typeof(_foreigncall_)}, ::Dual{Val{:jl_string_ptr}}, args::Vararg{Dual,N}
) where {N}
    return uninit_dual(_foreigncall_(Val(:jl_string_ptr), tuple_map(primal, args)...))
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
        f::Dual{typeof(_foreigncall_)},
        n::Dual{Val{$(QuoteNode(name))}},
        args::Vararg{Dual,N},
    ) where {N}
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
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{$(QuoteNode(name))}},
        ::Dual{Val{$P}},
        ::Dual{Tuple{Val{$P},Val{Int32}}},
        ::Dual{Val{0}},
        ::Dual{Val{:llvmcall}},
        x::Dual{$P},
        n::Dual{Int32},
        ::Dual{Int32},
        ::Dual{$P},
    )
        _x, dx = extract(x)
        _n = primal(n)
        y = Base.FastMath.pow_fast(_x, _n)
        return Dual(y, Nfwd._nfwd_pow_grad_x(_x, $P(_n), float(y)) * dx)
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
    @eval function frule!!(::Dual{typeof(_foreigncall_)}, ::Dual{Val{$name}}, args...)
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
    _c, _d = [1, 2], [3, 4]

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
        (
            true,
            :stability,
            nothing,
            unsafe_copyto!,
            CoDual(pointer(_c), Ptr{NoTangent}(C_NULL)),
            CoDual(pointer(_d), Ptr{NoTangent}(C_NULL)),
            2,
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
    memory = Any[_x, _dx, _a, _da, _b, _db, _c, _d]
    return test_cases, memory
end

function derived_rule_test_cases(rng_ctor, ::Val{:foreigncall})
    _x = Ref(5.0)

    function unsafe_copyto_tester(x::Vector{T}, y::Vector{T}, n::Int) where {T}
        GC.@preserve x y unsafe_copyto!(pointer(x), pointer(y), n)
        return x
    end

    function unsafe_copyto_overlap_tester(x, doffs, soffs, n)
        GC.@preserve x unsafe_copyto!(pointer(x, doffs), pointer(x, soffs), n)
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
        (false, :none, nothing, unsafe_copyto_overlap_tester, randn(4), 1, 1, 4),
        (false, :none, nothing, unsafe_copyto_overlap_tester, randn(4), 2, 1, 3),
        (false, :none, nothing, unsafe_copyto_overlap_tester, randn(4), 1, 2, 3),
        (false, :none, nothing, unsafe_copyto_overlap_tester, randn(4), 3, 1, 2),
        (false, :none, nothing, unsafe_copyto_overlap_tester, randn(4), 1, 3, 2),
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
        (
            false,
            :none,
            nothing,
            x -> unsafe_pointer_to_objref(pointer_from_objref(x)),
            [TestResources.TypeStableMutableStruct{Float64}(1.0, 2.0)],
        ),
        (
            false,
            :none,
            nothing,
            (
                function objptr_isa(x)
                    v = ["a", "bb"]
                    s = GC.@preserve v unsafe_pointer_to_objref(
                        unsafe_load(Ptr{Ptr{Cvoid}}(pointer(v, 2)))
                    )
                    return s === v[2] ? 2x : x
                end
            ),
            2.0,
        ),
        (
            false,
            :none,
            nothing,
            x -> begin
                r = TestResources.TypeStableMutableStruct{typeof(x)}(x, 2x)
                return GC.@preserve r unsafe_load(
                    Ptr{typeof(x)}(pointer_from_objref(r)) + sizeof(x)
                )
            end,
            1.0,
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
        (
            false,
            :none,
            nothing,
            x -> begin
                r = TestResources.NonDifferentiableFoo(1, false)
                return GC.@preserve r (pointer_from_objref(r) != C_NULL ? 2x : x)
            end,
            1.0,
        ),
    ]
    @static if VERSION >= v"1.11.0-rc4"
        push!(
            test_cases,
            (
                false,
                :none,
                nothing,
                x -> begin
                    m = Memory{Int}(undef, 1)
                    return GC.@preserve m (pointer_from_objref(m) != C_NULL ? 2x : x)
                end,
                1.0,
            ),
        )
    end
    return test_cases, memory
end
