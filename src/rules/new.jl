@is_primitive MinimalCtx Tuple{typeof(_new_),Vararg}

# Bare-Dual `_new_` kernel — invoked by other rules (e.g. legacy
# `getindex`/`_new_`-via-MemoryRef paths in the test framework) that build
# composite results from canonical V components. Kept as a kernel rather than
# deleted because direct callers exist outside the Lifted dispatch path.
function frule!!(f::Dual{typeof(_new_)}, p::Dual{Type{P}}, x::Vararg{Any,N}) where {P,N}
    primals = map(__get_primal, x)
    # For NDual containers (Array, etc.), the type P is the primal type but the fields
    # have NDual elements. Construct the NDual container type instead.
    # Check `_find_ndual_memref(x...)` on the ORIGINAL args, not on `primals`:
    # `__get_primal(::MemoryRef{<:NDual})` strips the NDual element type back to
    # the primal type, which would prevent `_find_ndual_memref` from matching.
    if P <: Array
        ref = _find_ndual_memref(x...)
        if ref !== nothing
            P_ndual = Array{eltype(ref),ndims(P)}
            return _new_(P_ndual, ref, primals[2:end]...)
        end
        # No NDual content: build the bare array and produce a `Dual{P, T}` with
        # `zero_tangent(y)`. The struct-output `build_output_tangent` path expects
        # a NamedTuple-shaped struct ctor, which Array does not have — calling it
        # would fail on `Vector{T}(::@NamedTuple{ref, size})`.
        y = _new_(P, primals...)
        T = tangent_type(P)
        T == NoTangent && return Dual(y, NoTangent())
        return Dual(y, zero_tangent(y))
    end
    # Complex{NDual} is bare — construct directly from NDual fields.
    if P <: Complex && _has_ndual(x...)
        return Complex(primals...)
    end
    if _has_ndual(x...)
        primals_extracted = map(_ndual_primal, x)
        y = _new_(P, primals_extracted...)
        T = tangent_type(P)
        T == NoTangent && return Dual(y, NoTangent())
        return _ndual_new_result(P, y, x, primals_extracted)
    end
    y = _new_(P, primals...)
    T = tangent_type(P)
    T == NoTangent && return Dual(y, NoTangent())
    return Dual(y, build_output_tangent(P, primals, map(tangent, x)))
end

# Lifted-aware `_new_`. The static `P` from `Lifted{Type{P}, N}` is the
# source of truth for the result wrap — the generic Lifted-aware adapter
# can't be used because the Array branch returns `Array{NDual{T,N}, D}`
# whose primal type isn't recoverable from the value alone.
@inline function frule!!(
    f::Lifted{typeof(_new_),N}, p::Lifted{Type{P},N}, x::Vararg{Lifted,M}
) where {N,P,M}
    bare_x = ntuple(i -> _unlift(x[i]), Val(M))

    # Array primal: with an NDual MemoryRef arg, build the bare element-wise
    # lifted Array; otherwise build the bare Array and wrap as `Dual{P, T}`
    # (the struct-output `build_output_tangent` path expects a NamedTuple-
    # shaped tangent constructor, which Array tangents don't have). `bare_x`
    # may carry Tuple-of-Dual / Dual{Tuple} dims args; `__get_primal` extracts
    # them while leaving `MemoryRef` and other shapes untouched.
    if P <: Array
        # `bare_x` may carry Tuple-of-Dual or Dual{Tuple} dims args from the
        # legacy non-IEEEFloat Tuple wrap; extract those, but leave MemoryRef
        # and other V shapes untouched (the NDual branch below needs the V
        # MemoryRef directly to construct `Array{NDual{T,N}}` via `:new`).
        new_args = map(bare_x) do v
            v isa Tuple && return primal(v)
            v isa Dual && return primal(v)
            return v
        end
        ref = _find_ndual_memref(bare_x...)
        if ref !== nothing
            P_ndual = Array{eltype(ref),ndims(P)}
            return Lifted{P,N}(_new_(P_ndual, new_args...))
        end
        y = _new_(P, new_args...)
        T = tangent_type(P)
        T == NoTangent && return Lifted{P,N}(Dual(y, NoTangent()))
        return Lifted{P,N}(Dual(y, zero_tangent(y)))
    end

    # Complex primal with NDual fields: bare Complex{NDual}.
    if P <: Complex && _has_ndual(bare_x...)
        return Lifted{P,N}(Complex(bare_x...))
    end

    # Array-wrapper primal (Diagonal, Adjoint, SubArray, …) with a structural-
    # lift `dual_type(Val(N), P)` defined: build the lifted wrapper directly
    # via `:new` on the lifted type. Extracts bare primals from non-NDual args
    # (e.g. SubArray's `indices::Tuple{StepRange}`, `offset1::Int`) and leaves
    # NDual arrays (`Vector{NDual{T,N}}` etc.) intact so tangent data stays in
    # element form, mirroring the `P <: Array` branch above.
    if P <: AbstractArray && _has_ndual(bare_x...)
        DT = _static_dual_type(Val(N), P)
        if DT isa DataType && !(DT <: Dual) && DT <: AbstractArray
            new_args = map(bare_x) do v
                v isa Tuple && return primal(v)
                v isa Dual && return primal(v)
                return v
            end
            return Lifted{P,N,DT}(_new_(DT, new_args...))
        end
    end

    # Tuple primal: canonical V is the element-wise tuple-of-inner-duals
    # (`dual_type(Val(N), Tuple{...})` lifts each field). `bare_x` is already
    # the element-wise inner tuple, so wrap it directly. Tuple primals do not
    # use the `Dual{Tuple, NTangent}` shape.
    if P <: Tuple
        InnerT = _static_dual_type(Val(N), P)
        if InnerT isa DataType && InnerT <: Tuple
            return Lifted{P,N,InnerT}(bare_x)
        end
    end

    # NamedTuple primal: parallel to Tuple — canonical V is
    # `NamedTuple{names, Tuple{V_i...}}` of bare inner duals. `bare_x` carries
    # the element-wise per-field inner duals; wrap into the named tuple.
    if P <: NamedTuple && fieldcount(P) > 0
        InnerT = _static_dual_type(Val(N), P)
        if InnerT isa DataType && InnerT <: NamedTuple
            names = fieldnames(P)
            return Lifted{P,N,InnerT}(NamedTuple{names}(bare_x))
        end
    end

    struct_result = _lifted_new_struct_namedtuple(P, Val(N), bare_x)
    struct_result === nothing || return struct_result

    # Base.Broadcast helper structs have wrapper-shaped canonical V around
    # NDual array leaves.
    if _is_base_broadcast_struct(P) && _has_ndual(bare_x...)
        InnerT = _static_dual_type(Val(N), P)
        if InnerT isa DataType && !(InnerT <: Dual)
            return Lifted{P,N,InnerT}(_new_(InnerT, _lifted_new_field_args(P, bare_x)...))
        end
    end

    # Struct with NDual content (legacy parallel-Dual path): at width=1
    # produce a bare-tangent Dual (matches `dual_type(Val(1), P) = Dual{P, T}`);
    # at width N>=2 wrap the per-direction tangents in `NTangent`. This
    # branch is only reached for structs whose `dual_type` returns a `Dual`
    # (e.g. types where `tangent_type(P)` is not `<: Tangent`, or which
    # have an explicit per-type `dual_type` overload pointing at the
    # parallel form).
    if _has_ndual(bare_x...)
        primals_extracted = _new_field_primals(P, bare_x)
        y = _new_(P, primals_extracted...)
        T = tangent_type(P)
        T == NoTangent && return Lifted{P,N}(Dual(y, NoTangent()))
        if N == 1
            return Lifted{P,N}(
                Dual(
                    y,
                    build_output_tangent(
                        P, primals_extracted, _new_field_tangents(P, bare_x, 1)
                    ),
                ),
            )
        end
        tangent_dirs = ntuple(Val(N)) do dir
            dir_tangents = _new_field_tangents(P, bare_x, dir)
            build_output_tangent(P, primals_extracted, dir_tangents)
        end
        return Lifted{P,N}(Dual(y, NTangent(tangent_dirs)))
    end

    # No NDual content (e.g. all-non-differentiable args): pure struct construction.
    primals = map(_new_field_primal_width1, bare_x)
    y = _new_(P, primals...)
    T = tangent_type(P)
    T == NoTangent && return Lifted{P,N}(Dual(y, NoTangent()))
    field_tangents = map(_new_field_tangent_width1, bare_x)
    return Lifted{P,N}(Dual(y, build_output_tangent(P, primals, field_tangents)))
end

@inline _new_field_primal_width1(v) = __get_primal(v)
@inline _new_field_primal_width1(v::NamedTuple) = primal(v)

@inline function _new_field_tangent_width1(v)
    t = tangent(v)
    return t isa Tuple{Vararg{NoTangent}} ? NoTangent() : t
end
@inline _new_field_tangent_width1(v::NamedTuple) = _tangent_dir(v, 1)

@generated function _lifted_new_struct_namedtuple(
    ::Type{P}, ::Val{N}, bare_x::Tx
) where {P,N,Tx<:Tuple}
    # `_new_` creates struct values in primal-mode IR, so its lifted result
    # must use the same structural V chosen by `dual_type(Val(N), P)`.
    if N >= 1 && _uses_structural_dual_type(P) && fieldcount(P) == fieldcount(Tx)
        names = fieldnames(P)
        InnerTup = Tuple{
            map(i -> _static_dual_type_value(Val(N), fieldtype(P, i)), 1:fieldcount(P))...
        }
        InnerT = NamedTuple{names,InnerTup}
        return :(Lifted{$P,$N,$InnerT}(NamedTuple{$names}(bare_x)))
    end
    return :(nothing)
end

@inline function frule!!(
    f::Lifted{typeof(_new_),N}, p::Lifted{<:Type,N}, x::Vararg{Lifted,M}
) where {N,M}
    P = primal(p)
    return frule!!(f, Lifted{Type{P},N}(P, NoTangent()), x...)
end

@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(_new_),Vararg}}) = true

@inline _is_base_broadcast_struct(::Type{P}) where {P} =
    P <: Union{Base.Broadcast.Extruded,Base.Broadcast.Broadcasted}

@inline function _lifted_new_field_args(::Type{<:Base.Broadcast.Extruded}, bare_x::Tuple)
    return map(_lifted_new_field_arg_deep, bare_x)
end
@inline function _lifted_new_field_args(::Type{<:Base.Broadcast.Broadcasted}, bare_x::Tuple)
    style, f, args, axes = bare_x
    return (
        _lifted_new_field_arg_shallow(style),
        _lifted_new_field_arg_shallow(f),
        args,
        _lifted_new_field_arg_deep(axes),
    )
end
@inline _lifted_new_field_arg_shallow(v) = v
@inline _lifted_new_field_arg_shallow(v::Dual) = primal(v)
@inline _lifted_new_field_arg_deep(v) = v
@inline _lifted_new_field_arg_deep(v::Dual) = primal(v)
@inline _lifted_new_field_arg_deep(v::Tuple) = map(_lifted_new_field_arg_deep, v)
@inline _lifted_new_field_arg_deep(v::NamedTuple{names}) where {names} = NamedTuple{names}(
    map(_lifted_new_field_arg_deep, values(v))
)

@generated function _new_field_primals(::Type{P}, bare_x::Tx) where {P,Tx<:Tuple}
    exprs = [:(_new_field_primal($(fieldtype(P, i)), bare_x[$i])) for i in 1:fieldcount(Tx)]
    return :(($(exprs...),))
end
@inline _new_field_primal(::Type, x) = _ndual_primal(x)
@inline _new_field_primal(::Type{P}, x::NamedTuple) where {P} = _new_(
    P, _ndual_primal(x)...
)

@generated function _new_field_tangents(::Type{P}, bare_x::Tx, dir) where {P,Tx<:Tuple}
    exprs = [
        :(_new_field_tangent($(fieldtype(P, i)), bare_x[$i], dir)) for i in 1:fieldcount(Tx)
    ]
    return :(($(exprs...),))
end
@inline _new_field_tangent(::Type, x, dir) = _tangent_dir(x, dir)
@inline function _new_field_tangent(::Type{P}, x::NamedTuple, dir) where {P}
    return build_output_tangent(P, Tuple(_ndual_primal(x)), Tuple(_tangent_dir(x, dir)))
end

# `_find_ndual_memref`, `_ndual_primal`, `_ndual_width`, `_tangent_dir`, and
# `_tangent_dir_elem` are defined in `nfwd/NfwdMooncake.jl` so that all NDual
# container dispatch lives in one file.

@inline function _ndual_new_result(::Type{P}, y, x::Tuple, primals::Tuple) where {P}
    return _ndual_new_result(P, y, x, primals, _ndual_width(x...))
end
@inline function _ndual_new_result(
    ::Type{P}, y, x::Tuple, primals::Tuple, ::Val{1}
) where {P}
    # Width 1 produces a bare-tangent `Dual{P, T}` matching
    # `dual_type(Val(1), P) = Dual{P, tangent_type(P)}`. Wrapping in
    # `NTangent{Tuple{T}}` here is the chunked-N shape and would mismatch
    # the OC slot.
    dir_tangents = map(xi -> _tangent_dir(xi, 1), x)
    return Dual(y, build_output_tangent(P, primals, dir_tangents))
end
@inline function _ndual_new_result(
    ::Type{P}, y, x::Tuple, primals::Tuple, ::Val{W}
) where {P,W}
    tangent_dirs = ntuple(Val(W)) do i
        dir_tangents = map(xi -> _tangent_dir(xi, i), x)
        build_output_tangent(P, primals, dir_tangents)
    end
    return Dual(y, NTangent(tangent_dirs))
end

function rrule!!(
    f::CoDual{typeof(_new_)}, p::CoDual{Type{P}}, x::Vararg{CoDual,N}
) where {P,N}
    y = _new_(P, tuple_map(primal, x)...)
    F = fdata_type(tangent_type(P))
    R = rdata_type(tangent_type(P))
    dy = if F == NoFData
        NoFData()
    else
        build_fdata(P, tuple_map(primal, x), tuple_map(tangent, x))
    end
    pb!! = if ismutabletype(P)
        if F == NoFData
            NoPullback(f, p, x...)
        else
            function _mutable_new_pullback!!(::NoRData)
                rdatas = tuple_map(rdata ∘ val, Tuple(dy.fields)[1:N])
                return NoRData(), NoRData(), rdatas...
            end
        end
    else
        if R == NoRData
            NoPullback(f, p, x...)
        else
            function _new_pullback_for_immutable!!(dy::T) where {T}
                data = Tuple(T <: NamedTuple ? dy : dy.data)[1:N]
                return NoRData(), NoRData(), map(val, data)...
            end
        end
    end
    return CoDual(y, dy), pb!!
end

@inline function build_output_tangent(::Type{P}, x::Tuple, t::Tuple) where {P}
    return _build_output_tangent_cartesian(P, x, t, Val(fieldcount(P)), Val(fieldnames(P)))
end
@generated function _build_output_tangent_cartesian(
    ::Type{P}, x::Tuple, t::Tt, ::Val{nfield}, ::Val{names}
) where {P,nfield,names,Tt<:Tuple}
    N = length(Tt.parameters)
    quote
        # Compute tangent_field_types and tangent_type at runtime to avoid world-age
        # issues with user-defined tangent_type methods. See #893, #1008.
        processed_tangent = Base.Cartesian.@ntuple(
            $nfield, n -> let
                F = tangent_field_types(P)[n]
                if n <= $N
                    data = __get_data(P, x, t, n)
                    F <: PossiblyUninitTangent ? F(data) : data
                else
                    F()
                end
            end
        )
        T_out = tangent_type(P)
        return T_out(NamedTuple{$names}(processed_tangent))
    end
end

@inline function build_fdata(::Type{P}, x::Tuple, fdata::Tuple) where {P}
    return _build_fdata_cartesian(P, x, fdata, Val(fieldcount(P)), Val(fieldnames(P)))
end
@generated function _build_fdata_cartesian(
    ::Type{P}, x::Tuple, fdata::Tfdata, ::Val{nfield}, ::Val{names}
) where {P,nfield,names,Tfdata<:Tuple}
    N = length(Tfdata.parameters)
    quote
        processed_fdata = Base.Cartesian.@ntuple(
            $nfield, n -> let
                F = fdata_field_type(P, n)
                if n <= $N
                    data = __get_data(P, x, fdata, n)
                    F <: PossiblyUninitTangent ? F(data) : data
                else
                    F()
                end
            end
        )
        F_out = fdata_type(tangent_type(P))
        return F_out(NamedTuple{$names}(processed_fdata))
    end
end

# Helper for build_fdata
@unstable @inline function __get_data(::Type{P}, x, f, n) where {P}
    tmp = getfield(f, n)
    return ismutabletype(P) ? zero_tangent(getfield(x, n), tmp) : tmp
end

@inline function build_fdata(::Type{P}, x::Tuple, fdata::Tuple) where {P<:NamedTuple}
    return fdata_type(tangent_type(P))(fdata)
end

"""
    _splat_new_(::Type{P}, x::Tuple) where {P}

Function which replaces instances of `:splatnew`.
"""
_splat_new_(::Type{P}, x::Tuple) where {P} = _new_(P, x...)

@is_primitive MinimalCtx Tuple{typeof(_splat_new_),Type,Tuple}

@inline @generated function frule!!(
    f::Lifted{typeof(_splat_new_),N}, p::Lifted{Type{P},N}, x::Lifted{Tx,N}
) where {N,P,Tx<:Tuple}
    args = [
        :(Lifted{$(fieldtype(Tx, i)),$N}(
            _canonical_splat_new_arg(Val($N), _unlift(x)[$i])
        )) for i in 1:fieldcount(Tx)
    ]
    return :(frule!!(Lifted{typeof(_new_),$N}(_new_, tangent(f)), p, $(args...)))
end

@inline _canonical_splat_new_arg(::Val, x) = x
@inline _canonical_splat_new_arg(::Val{N}, x::Dual{P,T}) where {N,P<:IEEEFloat,T<:IEEEFloat} = _combine_to_ndual(
    primal(x), ntuple(_ -> tangent(x), Val(N))
)

@inline _splat_new_field_fdata(::NoFData, _) = NoFData()
@inline _splat_new_field_fdata(fdata::Tuple, n) = fdata[n]
@inline function _splat_new_tuple_rdata(::Type{Tx}, field_rdatas) where {Tx<:Tuple}
    return rdata_type(tangent_type(Tx)) == NoRData ? NoRData() : field_rdatas
end

function rrule!!(
    f::CoDual{typeof(_splat_new_)}, p::CoDual{Type{P}}, x::CoDual{Tx}
) where {P,Tx<:Tuple}
    x_primal = primal(x)
    xs = ntuple(
        n -> CoDual(x_primal[n], _splat_new_field_fdata(tangent(x), n)), Val(fieldcount(Tx))
    )
    y, new_pb!! = rrule!!(zero_fcodual(_new_), p, xs...)
    function splat_new_pb!!(dy)
        parts = new_pb!!(dy)
        field_rdatas = Base.tail(Base.tail(parts))
        return NoRData(), NoRData(), _splat_new_tuple_rdata(Tx, field_rdatas)
    end
    return y, splat_new_pb!!
end

@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(_splat_new_),Type,Tuple}}) = true

function hand_written_rule_test_cases(rng_ctor, ::Val{:new})

    # Specialised test cases for _new_.
    specific_test_cases = Any[
        (false, :stability_and_allocs, nothing, _new_, @NamedTuple{}),
        (false, :stability_and_allocs, nothing, _new_, @NamedTuple{y::Float64}, 5.0),
        (false, :stability_and_allocs, nothing, _new_, @NamedTuple{y::Int, x::Int}, 5, 4),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            @NamedTuple{y::Float64, x::Int},
            5.0,
            4,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            @NamedTuple{y::Vector{Float64}, x::Int},
            randn(2),
            4,
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            @NamedTuple{y::Vector{Float64}},
            randn(2),
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            TestResources.TypeStableStruct{Float64},
            5,
            4.0,
        ),
        (false, :stability_and_allocs, nothing, _new_, UnitRange{Int64}, 5, 4),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            TestResources.TypeStableMutableStruct{Float64},
            5.0,
            4.0,
        ),
        (
            false,
            :none,
            nothing,
            _new_,
            TestResources.TypeStableMutableStruct{Any},
            5.0,
            4.0,
        ),
        (false, :none, nothing, _new_, TestResources.StructFoo, 6.0, [1.0, 2.0]),
        (false, :none, nothing, _new_, TestResources.StructFoo, 6.0),
        (false, :none, nothing, _new_, TestResources.MutableFoo, 6.0, [1.0, 2.0]),
        (false, :none, nothing, _new_, TestResources.MutableFoo, 6.0),
        (false, :stability_and_allocs, nothing, _new_, TestResources.StructNoFwds, 5.0),
        (false, :stability_and_allocs, nothing, _new_, TestResources.StructNoRvs, [5.0]),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            LowerTriangular{Float64,Matrix{Float64}},
            randn(2, 2),
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            UpperTriangular{Float64,Matrix{Float64}},
            randn(2, 2),
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            UnitLowerTriangular{Float64,Matrix{Float64}},
            randn(2, 2),
        ),
        (
            false,
            :stability_and_allocs,
            nothing,
            _new_,
            UnitUpperTriangular{Float64,Matrix{Float64}},
            randn(2, 2),
        ),
    ]
    general_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        return (interface_only, :none, nothing, _new_, P, args...)
    end
    test_cases = vcat(specific_test_cases, general_test_cases)
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:new}) = Any[], Any[]
