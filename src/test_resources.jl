"""
    module TestResources

A collection of functions and types which should be tested. The intent is to get this module
to a state in which if we can successfully AD everything in it, we know we can successfully
AD anything.
"""
module TestResources

using ..Mooncake
using ..Mooncake:
    CoDual,
    Dual,
    Tangent,
    MutableTangent,
    NoTangent,
    PossiblyUninitTangent,
    ircode,
    @is_primitive,
    MinimalCtx,
    val,
    primal,
    tangent

using LinearAlgebra, Random

#
# Types used for testing purposes
#

function equal_field(a, b, f)
    (!isdefined(a, f) || !isdefined(b, f)) && return true
    return getfield(a, f) == getfield(b, f)
end

mutable struct Foo
    x::Real
end

Base.:(==)(a::Foo, b::Foo) = equal_field(a, b, :x)

struct StructFoo
    a::Real
    b::Vector{Float64}
    StructFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    StructFoo(a::Float64) = new(a)
end

Base.:(==)(a::StructFoo, b::StructFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

mutable struct MutableFoo
    a::Float64
    b::AbstractVector
    MutableFoo(a::Float64, b::Vector{Float64}) = new(a, b)
    MutableFoo(a::Float64) = new(a)
end

Base.:(==)(a::MutableFoo, b::MutableFoo) = equal_field(a, b, :a) && equal_field(a, b, :b)

mutable struct NonDifferentiableFoo
    x::Int
    y::Bool
end

mutable struct TypeStableMutableStruct{T}
    a::Float64
    b::T
    TypeStableMutableStruct{T}(a::Float64) where {T} = new{T}(a)
    TypeStableMutableStruct{T}(a::Float64, b::T) where {T} = new{T}(a, b)
end

function Base.:(==)(a::TypeStableMutableStruct, b::TypeStableMutableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end

mutable struct TypeUnstableMutableStruct
    a::Float64
    b
    TypeUnstableMutableStruct(a::Float64) = new(a)
    TypeUnstableMutableStruct(a::Float64, b) = new(a, b)
end

mutable struct TypeUnstableMutableStruct2
    a
    b
end

struct TypeStableStruct{T}
    a::Int
    b::T
    TypeStableStruct{T}(a::Int) where {T} = new{T}(a)
    TypeStableStruct{T}(a::Int, b::T) where {T} = new{T}(a, b)
end

struct TypeUnstableStruct2
    a
    b
end

struct TypeUnstableStruct
    a::Float64
    b
    TypeUnstableStruct(a::Float64) = new(a)
    TypeUnstableStruct(a::Float64, b) = new(a, b)
end

function Base.:(==)(a::TypeUnstableStruct, b::TypeUnstableStruct)
    return equal_field(a, b, :a) && equal_field(a, b, :b)
end

mutable struct FullyInitMutableStruct
    x::Float64
    y::Vector{Float64}
end

function Base.:(==)(a::FullyInitMutableStruct, b::FullyInitMutableStruct)
    return equal_field(a, b, :x) && equal_field(a, b, :y)
end

struct StructNoFwds
    x::Float64
end

struct StructNoRvs
    x::Vector{Float64}
end

struct FiveFields{A,B,C,D,E}
    a::A
    b::B
    c::C
    d::D
    e::E
end

struct FourFields{A,B,C,D}
    a::A
    b::B
    c::C
    d::D
end

struct OneField{A}
    a::A
end

# Test for unions involving `Nothing`. See, 
# https://github.com/chalk-lab/Mooncake.jl/issues/597 for the reason.
struct P_union_nothing_float{T<:Base.IEEEFloat}
    x::Union{T,Nothing}
    x2::Union{T,Nothing}
    y::T
    z::Union{Array{T,1},Nothing}
    z2::Union{Array{T,1},Nothing}
    w::Union{Array{T,2},Nothing}
    w2::Union{Array{T,2},Nothing}
end
function make_P_union_nothing(T=Float32)
    return P_union_nothing_float{T}(
        T(1.0),
        nothing,
        T(1.0),
        randn(Xoshiro(1), T, 2),
        nothing,
        randn(Xoshiro(1), T, 2, 2),
        nothing,
    )
end

# https://github.com/chalk-lab/Mooncake.jl/issues/598
struct P_union_nothing_array{T}
    w::T
    w2::Union{Vector{Tuple{Int,Int,Vector{Tuple{Int,Int}}}},Nothing}
end
function make_P_union_array(T=Float32)
    return P_union_nothing_array{T}(T(1.0), nothing)
end

# https://github.com/chalk-lab/Mooncake.jl/issues/631
struct P_adam_like
    alphas::Vector{Float64}
    values::Vector{Float64}
    slopes::Vector{Float64}
end
const P_adam_like_union = Union{Nothing,P_adam_like}

function build_big_isbits_struct()
    return FourFields(
        FiveFields(
            FourFields(OneField(5.0), OneField(5.0), OneField(3), OneField(nothing)),
            OneField(4),
            OneField(3.0),
            OneField(nothing),
            OneField(false),
        ),
        OneField(5.0),
        OneField(nothing),
        OneField(4),
    )
end

#
# generate test cases for circular references
#

function make_circular_reference_struct()
    c = TypeUnstableMutableStruct(1.0, nothing)
    c.b = c
    return c
end

function make_indirect_circular_reference_struct()
    c = TypeUnstableMutableStruct(1.0)
    _c = TypeUnstableMutableStruct(1.0, c)
    c.b = _c
    return c
end

function make_circular_reference_array()
    a = Any[1.0, 2.0, 3.0]
    a[1] = a
    return a
end

function make_indirect_circular_reference_array()
    a = Any[1.0, 2.0, 3.0]
    b = Any[a, 4.0]
    a[1] = b
    return a
end

#
# Tests for AD. There are not rules defined directly on these functions, and they require
# that most language primitives have rules defined.
#

@noinline function foo(x)
    y = sin(x)
    z = cos(y)
    return z
end

# A function in which everything is non-differentiable and has no branching. Ideally, the
# reverse-pass of this function would be a no-op, and there would be no use of the block
# stack anywhere.
function non_differentiable_foo(x::Int)
    y = 5x
    z = y + x
    return 10z
end

function bar(x, y)
    x1 = sin(x)
    x2 = cos(y)
    x3 = foo(x2)
    x4 = foo(x3)
    x5 = x2 + x4
    return x5
end

function unused_expression(x, n)
    y = getfield((Float64,), n)
    return x
end

const_tester_non_differentiable() = 1

const_tester() = cos(5.0)

intrinsic_tester(x) = 5x

function goto_tester(x)
    if x > cos(x)
        @goto aha
    end
    x = sin(x)
    @label aha
    return cos(x)
end

struct StableFoo
    x::Float64
    y::Symbol
end

new_tester(x, y) = StableFoo(x, y)

new_tester_2(x) = StableFoo(x, :symbol)

@eval function new_tester_3(x::Ref{Any})
    y = x[]
    return $(Expr(:new, :y, 5.0))
end

@eval splatnew_tester(x::Ref{Tuple}) = $(Expr(:splatnew, StableFoo, :(x[])))

type_stable_getfield_tester_1(x::StableFoo) = x.x
type_stable_getfield_tester_2(x::StableFoo) = x.y

const __x_for_gref_test = 5.0
@eval globalref_tester() = $(GlobalRef(@__MODULE__, :__x_for_gref_test))

const __y_for_gref_test = false
@eval globalref_tester_bool() = $(GlobalRef(@__MODULE__, :__y_for_gref_test))

function globalref_tester_2(use_gref::Bool)
    v = use_gref ? __x_for_gref_test : 1
    return sin(v)
end

const __x_for_gref_tester_3 = 5.0
@eval globalref_tester_3() = $(GlobalRef(@__MODULE__, :__x_for_gref_tester_3))

const __x_for_gref_tester_4::Float64 = 3.0
@eval globalref_tester_4() = $(GlobalRef(@__MODULE__, :__x_for_gref_tester_4))

__x_for_gref_tester_5 = 5.0
@eval globalref_tester_5() = $(GlobalRef(@__MODULE__, :__x_for_gref_tester_5))

# See https://github.com/chalk-lab/Mooncake.jl/issues/329 .
const __x = randn(10)
@noinline globalref_tester_6_inner(x) = sum(x)
globalref_tester_6() = globalref_tester_6_inner(__x)

type_unstable_tester_0(x::Ref{Any}) = x[]

type_unstable_tester(x::Ref{Any}) = cos(x[])

type_unstable_tester_2(x::Ref{Real}) = cos(x[])

type_unstable_tester_3(x::Ref{Any}) = Foo(x[])

type_unstable_function_eval(f::Ref{Any}, x::Float64) = f[](x)

type_unstable_argument_eval(@nospecialize(f), x::Float64) = f(x)

abstractly_typed_unused_container(::StructFoo, x::Float64) = 5x

function phi_const_bool_tester(x)
    if x > 0
        a = true
    else
        a = false
    end
    return cos(a)
end

function phi_node_with_undefined_value(x::Bool, y::Float64)
    if x
        v = sin(y)
    end
    z = cos(y)
    if x
        z += v
    end
    return z
end

function pi_node_tester(y::Ref{Any})
    x = y[]
    return isa(x, Int) ? sin(x) : x
end

Base.@nospecializeinfer arg_in_pi_node(@nospecialize(x)) = x isa Bool ? x : false

function avoid_throwing_path_tester(x)
    if x < 0
        Base.throw_boundserror(1:5, 6)
    end
    return sin(x)
end

simple_foreigncall_tester(s::String) = ccall(:jl_string_ptr, Ptr{UInt8}, (Any,), s)

function simple_foreigncall_tester_2(a::TypeVar, b::Type)
    return ccall(:jl_type_unionall, Any, (Any, Any), a, b)
end

function no_primitive_inlining_tester(x)
    X = Matrix{Float64}(undef, 5, 5) # contains a foreigncall which should never be hit
    for n in eachindex(X)
        X[n] = x
    end
    return X
end

@noinline varargs_tester(x::Vararg{Any,N}) where {N} = x

varargs_tester_2(x) = varargs_tester(x)
varargs_tester_2(x, y) = varargs_tester(x, y)
varargs_tester_2(x, y, z) = varargs_tester(x, y, z)

@noinline varargs_tester_3(x, y::Vararg{Any,N}) where {N} = sin(x), y

varargs_tester_4(x) = varargs_tester_3(x...)
varargs_tester_4(x, y) = varargs_tester_3(x...)
varargs_tester_4(x, y, z) = varargs_tester_3(x...)

splatting_tester(x) = varargs_tester(x...)
unstable_splatting_tester(x::Ref{Any}) = varargs_tester(x[]...)

function inferred_const_tester(x::Base.RefValue{Any})
    y = x[]
    y === nothing && return y
    return 5y
end
inferred_const_tester(x::Int) = x == 5 ? x : 5x

getfield_tester(x::Tuple) = x[1]
getfield_tester_2(x::Tuple) = getfield(x, 1)

function setfield_tester_left!(x::FullyInitMutableStruct, new_field)
    x.x = new_field
    return new_field
end

function setfield_tester_right!(x::FullyInitMutableStruct, new_field)
    x.y = new_field
    return new_field
end

function datatype_slot_tester(n::Int)
    return (Float64, Int)[n]
end

@noinline test_sin(x) = sin(x)

test_cos_sin(x) = cos(sin(x))

test_isbits_multiple_usage(x::Float64) = Core.Intrinsics.mul_float(x, x)

function test_isbits_multiple_usage_2(x::Float64)
    y = Core.Intrinsics.mul_float(x, x)
    return Core.Intrinsics.mul_float(y, y)
end

function test_isbits_multiple_usage_3(x::Float64)
    y = sin(x)
    z = Core.Intrinsics.mul_float(y, y)
    a = Core.Intrinsics.mul_float(z, z)
    b = cos(a)
    return b
end

function test_isbits_multiple_usage_4(x::Float64)
    y = x > 0.0 ? cos(x) : sin(x)
    return Core.Intrinsics.mul_float(y, y)
end

function test_isbits_multiple_usage_5(x::Float64)
    y = Core.Intrinsics.mul_float(x, x)
    return x > 0.0 ? cos(y) : sin(y)
end

function test_isbits_multiple_usage_phi(x::Bool, y::Float64)
    z = x ? y : 1.0
    return z * y
end

function test_multiple_call_non_primitive(x::Float64)
    for _ in 1:2
        x = test_sin(x)
    end
    return x
end

test_getindex(x::AbstractArray{<:Real}) = x[1]

function test_mutation!(x::AbstractVector{<:Real})
    x[1] = sin(x[2])
    return x[1]
end

function test_while_loop(x)
    n = 3
    while n > 0
        x = cos(x)
        n -= 1
    end
    return x
end

function test_for_loop(x)
    for _ in 1:500
        x = sin(x)
    end
    return x
end

# This catches the case where there are multiple phi nodes at the start of the block, and
# they refer to one another. It is in this instance that the distinction between phi nodes
# acting "instanteneously" and "in sequence" becomes apparent.
function test_multiple_phinode_block(x::Float64, N::Int)
    a = 1.0
    b = x
    i = 1
    while i < N
        temp = a
        a = b
        b = 2temp
        i += 1
    end
    return (a, b)
end

test_mutable_struct_basic(x) = Foo(x).x

test_mutable_struct_basic_sin(x) = sin(Foo(x).x)

function test_mutable_struct_setfield(x)
    foo = Foo(1.0)
    foo.x = x
    return foo.x
end

function test_mutable_struct(x)
    foo = Foo(x)
    foo.x = sin(foo.x)
    return foo.x
end

test_struct_partial_init(a::Float64) = StructFoo(a).a

test_mutable_partial_init(a::Float64) = MutableFoo(a).a

function test_naive_mat_mul!(C::Matrix{T}, A::Matrix{T}, B::Matrix{T}) where {T<:Real}
    for p in 1:size(C, 1)
        for q in 1:size(C, 2)
            C[p, q] = zero(T)
            for r in 1:size(A, 2)
                C[p, q] += A[p, r] * B[r, q]
            end
        end
    end
    return C
end

test_diagonal_to_matrix(D::Diagonal) = Matrix(D)

relu(x) = max(x, zero(x))

test_mlp(x, W1, W2) = W2 * tanh.(W1 * x)

function test_multiple_pi_nodes(x::Base.RefValue{Any})
    v = x[]
    return (v::Float64, v::Float64) # PiNode applied to the same SSAValue
end

function test_multi_use_pi_node(x::Base.RefValue{Any})
    v = x[]
    for _ in 1:2
        v = sin(v)::Float64
    end
    return v
end

function test_union_of_arrays(x::Vector{Float64}, b::Bool)
    y = randn(Xoshiro(1), Float32, 4)
    z = b ? x : y
    return 2z
end

function test_union_of_types(x::Ref{Union{Type{Float64},Type{Int}}})
    return x[]
end

function test_small_union(x::Ref{Union{Float64,Vector{Float64}}})
    v = x[]
    return v isa Float64 ? v : v[1]
end

# Only one of these is a primitive. Lots of methods to prevent the compiler from
# over-specialising.
@noinline edge_case_tester(x::Float64) = 5x
@noinline edge_case_tester(x::Any) = 5.0
@noinline edge_case_tester(x::Float32) = 6.0
@noinline edge_case_tester(x::Int) = 10
@noinline edge_case_tester(x::String) = "hi"
@is_primitive MinimalCtx Tuple{typeof(edge_case_tester),Float64}
function Mooncake.frule!!(::Dual{typeof(edge_case_tester)}, x::Dual{Float64})
    return Dual(5 * primal(x), 5 * tangent(x))
end
function Mooncake.rrule!!(::CoDual{typeof(edge_case_tester)}, x::CoDual{Float64})
    edge_case_tester_pb!!(dy) = Mooncake.NoRData(), 5 * dy
    return Mooncake.zero_fcodual(5 * primal(x)), edge_case_tester_pb!!
end

# To test the edge case properly, call this with x = Any[5.0, false]
function test_primitive_dynamic_dispatch(x::Vector{Any})
    i = 0
    y = 0.0
    while i < 2
        i += 1
        y += edge_case_tester(x[i])
    end
    return y
end

sr(n) = Xoshiro(n)

@noinline function test_self_reference(a, b)
    return a < b ? a * b : test_self_reference(b, a) + a
end

# See https://github.com/withbayes/Mooncake.jl/pull/84 for info
@noinline function test_recursive_sum(x::Vector{Float64})
    isempty(x) && return 0.0
    return @inbounds x[1] + test_recursive_sum(x[2:end])
end

# Copied over from https://github.com/TuringLang/Turing.jl/issues/1140
function _sum(x)
    z = 0 # this intentionally causes a type instability -- do not make this type stable.
    for i in eachindex(x)
        z += x[i]
    end
    return z
end

function test_handwritten_sum(x::AbstractArray{<:Real})
    y = 0.0
    n = 0
    @inbounds while n < length(x)
        n += 1
        y += x[n]
    end
    return y
end

function test_map(x::Vector{Float64}, y::Vector{Float64})
    return map((x, y) -> sin(cos(exp(x)) + exp(y) * sin(y)), x, y)
end

test_getfield_of_tuple_of_types(n::Int) = getfield((Float64, Float64), n)

test_for_invoke(x) = 5x

inlinable_invoke_call(x::Float64) = invoke(test_for_invoke, Tuple{Float64}, x)

vararg_test_for_invoke(n::Tuple{Int,Int}, x...) = sum(x) + n[1]

function inlinable_vararg_invoke_call(
    rows::Tuple{Vararg{Int}}, n1::N, ns::Vararg{N}
) where {N}
    return invoke(vararg_test_for_invoke, Tuple{typeof(rows),Vararg{N}}, rows, n1, ns...)
end

# build_rrule should error for this function, because it references a non-const global ref.
__x_for_non_const_global_ref::Float64 = 5.0
function non_const_global_ref(y::Float64)
    global __x_for_non_const_global_ref = y
    return __x_for_non_const_global_ref
end

# The inferred type of `TypeVar(...)` is `CC.PartialTypeVar`. Thanks to Jameson Nash for
# pointing out this pleasantly simple test case.
partial_typevar_tester() = TypeVar(:a, Union{}, Any)

function typevar_tester()
    tv = Core._typevar(:a, Union{}, Any)
    t = Core.apply_type(AbstractArray, tv, 1)
    return UnionAll(tv, t)
end

tuple_with_union(x::Bool) = (x ? 5.0 : 5, nothing)
tuple_with_union_2(x::Bool) = (x ? 5.0 : 5, x ? 5 : 5.0)
tuple_with_union_3(x::Bool, y::Bool) = (x ? 5.0 : (y ? 5 : nothing), nothing)

struct NoDefaultCtor{T}
    x::T
    NoDefaultCtor(x::T) where {T} = new{T}(x)
end

@noinline function __inplace_function!(x::Vector{Float64})
    x .= cos.(x)
    return nothing
end

function inplace_invoke!(x::Vector{Float64})
    __inplace_function!(x)
    return nothing
end

highly_nested_tuple(x) = ((((x,),), x), x)

# Regression test: https://github.com/chalk-lab/Mooncake.jl/issues/450
sig_argcount_mismatch(x) = vcat(x[1], x[2:2], x[3:3], x[4:4])

# Regression test: https://github.com/chalk-lab/Mooncake.jl/issues/473
large_tuple_inference(x::NTuple{1_000,Float64}) = sum(cos, x)

# Regression test: https://github.com/chalk-lab/Mooncake.jl/issues/319
function regression_319(θ)
    d = [0.0, 0.0]
    x = θ[1:2]
    return d
end

function generate_test_functions()
    return Any[
        (false, :allocs, nothing, const_tester),
        (false, :allocs, nothing, const_tester_non_differentiable),
        (false, :allocs, nothing, identity, 5.0),
        (false, :allocs, nothing, foo, 5.0),
        (false, :allocs, nothing, non_differentiable_foo, 5),
        (false, :allocs, nothing, bar, 5.0, 4.0),
        (false, :allocs, nothing, unused_expression, 5.0, 1),
        (false, :none, nothing, type_unstable_argument_eval, sin, 5.0),
        (
            false,
            :none,
            nothing,
            abstractly_typed_unused_container,
            StructFoo(5.0, [4.0]),
            5.0,
        ),
        (false, :none, (lb=1, ub=1_000), pi_node_tester, Ref{Any}(5.0)),
        (false, :none, (lb=1, ub=1_000), pi_node_tester, Ref{Any}(5)),
        (false, :none, nothing, arg_in_pi_node, false),
        (false, :allocs, nothing, intrinsic_tester, 5.0),
        (false, :allocs, nothing, goto_tester, 5.0),
        (false, :allocs, nothing, new_tester, 5.0, :hello),
        (false, :allocs, nothing, new_tester_2, 4.0),
        (false, :none, nothing, new_tester_3, Ref{Any}(StructFoo)),
        (false, :none, nothing, splatnew_tester, Ref{Tuple}((5.0, :a))),
        (false, :allocs, nothing, type_stable_getfield_tester_1, StableFoo(5.0, :hi)),
        (false, :allocs, nothing, type_stable_getfield_tester_2, StableFoo(5.0, :hi)),
        (false, :none, nothing, globalref_tester),
        (false, :none, nothing, globalref_tester_bool),
        (false, :none, nothing, globalref_tester_2, true),
        (false, :none, nothing, globalref_tester_2, false),
        (false, :allocs, nothing, globalref_tester_3),
        (false, :allocs, nothing, globalref_tester_4),
        (false, :none, nothing, globalref_tester_5),
        (false, :allocs, nothing, globalref_tester_6),
        (false, :none, (lb=1, ub=1_000), type_unstable_tester_0, Ref{Any}(5.0)),
        (false, :none, nothing, type_unstable_tester, Ref{Any}(5.0)),
        (false, :none, nothing, type_unstable_tester_2, Ref{Real}(5.0)),
        (false, :none, (lb=1, ub=500), type_unstable_tester_3, Ref{Any}(5.0)),
        (false, :none, (lb=1, ub=500), test_primitive_dynamic_dispatch, Any[5.0, false]),
        (false, :none, nothing, type_unstable_function_eval, Ref{Any}(sin), 5.0),
        (false, :allocs, nothing, phi_const_bool_tester, 5.0),
        (false, :allocs, nothing, phi_const_bool_tester, -5.0),
        (false, :allocs, nothing, phi_node_with_undefined_value, true, 4.0),
        (false, :allocs, nothing, phi_node_with_undefined_value, false, 4.0),
        (false, :allocs, nothing, test_multiple_phinode_block, 3.0, 3),
        (
            false,
            :none,
            nothing,
            Base._unsafe_getindex,
            IndexLinear(),
            randn(5),
            1,
            Base.Slice(Base.OneTo(1)),
        ), # fun PhiNode example
        (false, :allocs, nothing, avoid_throwing_path_tester, 5.0),
        (true, :allocs, nothing, simple_foreigncall_tester, "hello"),
        (
            false,
            :none,
            nothing,
            simple_foreigncall_tester_2,
            TypeVar(:T, Union{}, Any),
            Vector{T} where {T},
        ),
        (false, :none, nothing, no_primitive_inlining_tester, 5.0),
        (false, :allocs, nothing, varargs_tester, 5.0),
        (false, :allocs, nothing, varargs_tester, 5.0, 4),
        (false, :allocs, nothing, varargs_tester, 5.0, 4, 3.0),
        (false, :allocs, nothing, varargs_tester_2, 5.0),
        (false, :allocs, nothing, varargs_tester_2, 5.0, 4),
        (false, :allocs, nothing, varargs_tester_2, 5.0, 4, 3.0),
        (false, :allocs, nothing, varargs_tester_3, 5.0),
        (false, :allocs, nothing, varargs_tester_3, 5.0, 4),
        (false, :allocs, nothing, varargs_tester_3, 5.0, 4, 3.0),
        (false, :allocs, nothing, varargs_tester_4, 5.0),
        (false, :allocs, nothing, varargs_tester_4, 5.0, 4),
        (false, :allocs, nothing, varargs_tester_4, 5.0, 4, 3.0),
        (false, :allocs, nothing, splatting_tester, 5.0),
        (false, :allocs, nothing, splatting_tester, (5.0, 4.0)),
        (false, :allocs, nothing, splatting_tester, (5.0, 4.0, 3.0)),
        # (false, :stability, nothing, unstable_splatting_tester, Ref{Any}(5.0)), # known failure case -- no rrule for _apply_iterate
        # (false, :stability, nothing, unstable_splatting_tester, Ref{Any}((5.0, 4.0))), # known failure case -- no rrule for _apply_iterate
        # (false, :stability, nothing, unstable_splatting_tester, Ref{Any}((5.0, 4.0, 3.0))), # known failure case -- no rrule for _apply_iterate
        (false, :none, (lb=1, ub=1_000), inferred_const_tester, Ref{Any}(nothing)),
        (false, :none, (lb=1, ub=1_000), datatype_slot_tester, 1),
        (false, :none, (lb=1, ub=1_000), datatype_slot_tester, 2),
        (false, :none, (lb=1, ub=100_000_000), test_union_of_arrays, randn(5), true),
        (
            false,
            :none,
            nothing,
            test_union_of_types,
            Ref{Union{Type{Float64},Type{Int}}}(Float64),
        ),
        (false, :allocs, nothing, test_self_reference, 1.1, 1.5),
        (false, :allocs, nothing, test_self_reference, 1.5, 1.1),
        (false, :none, nothing, test_recursive_sum, randn(2)),
        (
            false,
            :none,
            nothing,
            LinearAlgebra._modify!,
            LinearAlgebra.MulAddMul(5.0, 4.0),
            5.0,
            randn(5, 4),
            (5, 4),
        ), # for Bool comma,
        (false, :allocs, nothing, getfield_tester, (5.0, 5)),
        (false, :allocs, nothing, getfield_tester_2, (5.0, 5)),
        (
            false,
            :allocs,
            nothing,
            setfield_tester_left!,
            FullyInitMutableStruct(5.0, randn(3)),
            4.0,
        ),
        (
            false,
            :none,
            nothing,
            setfield_tester_right!,
            FullyInitMutableStruct(5.0, randn(3)),
            randn(5),
        ),
        (false, :none, nothing, mul!, randn(3, 5)', randn(5, 5), randn(5, 3), 4.0, 3.0),
        (false, :none, nothing, Random.SHA.digest!, Random.SHA.SHA2_256_CTX()),
        (false, :none, nothing, Xoshiro, 123456),
        (false, :none, nothing, *, randn(250, 500), randn(500, 250)),
        (false, :allocs, nothing, test_sin, 1.0),
        (false, :allocs, nothing, test_cos_sin, 2.0),
        (false, :allocs, nothing, test_isbits_multiple_usage, 5.0),
        (false, :allocs, nothing, test_isbits_multiple_usage_2, 5.0),
        (false, :allocs, nothing, test_isbits_multiple_usage_3, 4.1),
        (false, :allocs, nothing, test_isbits_multiple_usage_4, 5.0),
        (false, :allocs, nothing, test_isbits_multiple_usage_5, 4.1),
        (false, :allocs, nothing, test_isbits_multiple_usage_phi, false, 1.1),
        (false, :allocs, nothing, test_isbits_multiple_usage_phi, true, 1.1),
        (false, :allocs, nothing, test_multiple_call_non_primitive, 5.0),
        (false, :none, (lb=1, ub=1500), test_multiple_pi_nodes, Ref{Any}(5.0)),
        (false, :none, (lb=1, ub=500), test_multi_use_pi_node, Ref{Any}(5.0)),
        (false, :allocs, nothing, test_getindex, [1.0, 2.0]),
        (false, :allocs, nothing, test_mutation!, [1.0, 2.0]),
        (false, :allocs, nothing, test_while_loop, 2.0),
        (false, :allocs, nothing, test_for_loop, 3.0),
        (false, :none, nothing, test_mutable_struct_basic, 5.0),
        (false, :none, nothing, test_mutable_struct_basic_sin, 5.0),
        (false, :none, nothing, test_mutable_struct_setfield, 4.0),
        (false, :none, (lb=1, ub=500), test_mutable_struct, 5.0),
        (false, :none, nothing, test_struct_partial_init, 3.5),
        (false, :none, nothing, test_mutable_partial_init, 3.3),
        (
            false,
            :allocs,
            nothing,
            test_naive_mat_mul!,
            randn(100, 50),
            randn(100, 30),
            randn(30, 50),
        ),
        (
            false,
            :allocs,
            nothing,
            (A, C) -> test_naive_mat_mul!(C, A, A),
            randn(25, 25),
            randn(25, 25),
        ),
        (false, :allocs, nothing, sum, randn(32)),
        (false, :none, nothing, test_diagonal_to_matrix, Diagonal(randn(30))),
        (
            false,
            :allocs,
            nothing,
            ldiv!,
            randn(20, 20),
            Diagonal(rand(20) .+ 1),
            randn(20, 20),
        ),
        (
            false,
            :allocs,
            nothing,
            LinearAlgebra._kron!,
            randn(25, 25),
            randn(5, 5),
            randn(5, 5),
        ),
        (false, :allocs, nothing, kron!, randn(25, 25), Diagonal(randn(5)), randn(5, 5)),
        (
            false,
            :none,
            nothing,
            test_mlp,
            randn(sr(1), 50, 20),
            randn(sr(2), 70, 50),
            randn(sr(3), 30, 70),
        ),
        (false, :allocs, nothing, test_handwritten_sum, randn(128, 128)),
        (false, :allocs, nothing, _naive_map_sin_cos_exp, randn(1024), randn(1024)),
        (false, :allocs, nothing, _naive_map_negate, randn(1024), randn(1024)),
        (false, :allocs, nothing, test_from_slack, randn(10_000)),
        (false, :none, nothing, _sum, randn(1024)),
        (false, :none, nothing, test_map, randn(1024), randn(1024)),
        (false, :none, nothing, _broadcast_sin_cos_exp, randn(10, 10)),
        (false, :none, nothing, _map_sin_cos_exp, randn(10, 10)),
        (false, :none, nothing, ArgumentError, "hi"),
        (false, :none, nothing, test_small_union, Ref{Union{Float64,Vector{Float64}}}(5.0)),
        (
            false,
            :none,
            nothing,
            test_small_union,
            Ref{Union{Float64,Vector{Float64}}}([1.0]),
        ),
        (false, :allocs, nothing, inlinable_invoke_call, 5.0),
        (false, :none, nothing, inlinable_vararg_invoke_call, (2, 2), 5.0, 4.0, 3.0, 2.0),
        (false, :none, nothing, hvcat, (2, 2), 3.0, 2.0, 0.0, 1.0),
        (false, :none, nothing, partial_typevar_tester),
        (false, :none, nothing, typevar_tester),
        (false, :allocs, nothing, inplace_invoke!, randn(1_024)),
        (false, :allocs, nothing, highly_nested_tuple, 5.0),
        (false, :none, nothing, sig_argcount_mismatch, ones(4)),
        (false, :allocs, (lb=2, ub=1500), large_tuple_inference, Tuple(zeros(1_000))),
        (false, :none, nothing, regression_319, randn(3)),
    ]
end

_broadcast_sin_cos_exp(x::AbstractArray{<:Real}) = sum(sin.(cos.(exp.(x))))

_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(x -> sin(cos(exp(x))), x))

function _naive_map_sin_cos_exp(y::AbstractArray{<:Real}, x::AbstractArray{<:Real})
    n = 1
    while n <= length(x)
        y[n] = sin(cos(exp(x[n])))
        n += 1
    end
    return y
end

function _naive_map_negate(y::AbstractArray{<:Real}, x::AbstractArray{<:Real})
    n = 1
    while n <= length(x)
        y[n] = -x[n]
        n += 1
    end
    return y
end

function test_from_slack(x::AbstractVector{T}) where {T}
    y = zero(T)
    n = 1
    while n <= length(x)
        if iseven(n)
            y += sin(x[n])
        else
            y += cos(x[n])
        end
        n += 1
    end
    return y
end

#
# This is a version of setfield! in which there is an issue with the address map.
# The method of setfield! is incorrectly implemented, so it errors. This is intentional,
# and is used to ensure that the tests correctly pick up on this mistake.
#

my_setfield!(args...) = setfield!(args...)

_setfield!(value::MutableTangent, name, x) = x

function Mooncake.rrule!!(::Mooncake.CoDual{typeof(my_setfield!)}, value, name, x)
    _name = primal(name)
    y = Mooncake.CoDual(
        setfield!(primal(value), _name, primal(x)),
        _setfield!(tangent(value), _name, tangent(x)),
    )
    return y, nothing
end

export MutableFoo, StructFoo, NonDifferentiableFoo, FullyInitMutableStruct

end

using .TestResources

function derived_rule_test_cases(rng_ctor, ::Val{:test_resources})
    return TestResources.generate_test_functions(), Any[]
end
