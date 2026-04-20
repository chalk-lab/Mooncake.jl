@is_primitive MinimalCtx Tuple{typeof(_new_),Vararg}

function frule!!(f::Dual{typeof(_new_)}, p::Dual{Type{P}}, x::Vararg{Any,N}) where {P,N}
    primals = map(__get_primal, x)
    # For NDual containers (Array, etc.), the type P is the primal type but the fields
    # have NDual elements. Construct the NDual container type instead.
    if P <: Array
        ref = _find_ndual_memref(primals...)
        if ref !== nothing
            P_ndual = Array{eltype(ref),ndims(P)}
            return _new_(P_ndual, primals...)
        end
    end
    # Complex{NDual} is bare — construct directly from NDual fields.
    if P <: Complex && _has_ndual(x...)
        raw = map(x_i -> x_i isa Dual ? primal(x_i) : x_i, x)
        return Complex(raw...)
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

@inline _find_ndual_memref(_, rest...) = _find_ndual_memref(rest...)
@inline _find_ndual_memref() = nothing

# Extract primal from any representation: Dual-wrapped, bare NDual, or plain value.
@inline _ndual_primal(x::Dual) = primal(x)
@inline _ndual_primal(x::NDual) = primal(x)
@inline _ndual_primal(x::Complex{<:NDual}) = primal(x)
@inline _ndual_primal(x::AbstractArray{<:NDual}) = map(d -> d.value, x)
@inline _ndual_primal(x::AbstractArray{<:Complex{<:NDual}}) = map(
    z -> complex(z.re.value, z.im.value), x
)
@inline _ndual_primal(x::Tuple) = map(_ndual_primal, x)
@inline _ndual_primal(x) = x

@inline _ndual_width(x::Tuple, rest...) = _ndual_width(x..., rest...)
@inline _ndual_width(::NDual{T,W}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::Complex{NDual{T,W}}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::AbstractArray{NDual{T,W}}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::AbstractArray{Complex{NDual{T,W}}}, rest...) where {T,W} = Val(W)
@inline _ndual_width(::Dual{<:Any,NTangent{L}}, rest...) where {L<:Tuple} = Val(
    fieldcount(L)
)
@inline _ndual_width(x::Dual, rest...) = _ndual_width(tangent(x), rest...)
@inline _ndual_width(_, rest...) = _ndual_width(rest...)
@inline _ndual_width() = error("_ndual_width called with no NDual arguments")

@inline _tangent_dir(x::NDual, i) = x.partials[i]
@inline _tangent_dir(x::Complex{<:NDual}, i) = complex(x.re.partials[i], x.im.partials[i])
@inline _tangent_dir(x::Dual{<:Any,<:NTangent}, i) = tangent(x).lanes[i]
@inline _tangent_dir(x::Dual{<:Any,<:Tuple}, i) = map(
    t -> _tangent_dir_elem(t, i), tangent(x)
)
@inline _tangent_dir(x::Dual, _) = tangent(x)
@inline _tangent_dir(x::AbstractArray{NDual{T,N}}, i) where {T,N} = map(
    d -> d.partials[i], x
)
@inline _tangent_dir(x::AbstractArray{Complex{NDual{T,N}}}, i) where {T,N} = map(
    z -> complex(z.re.partials[i], z.im.partials[i]), x
)
@inline _tangent_dir(x::Tuple, i) = map(xi -> _tangent_dir(xi, i), x)
@inline _tangent_dir(x, _) = zero_tangent(x)

@inline _tangent_dir_elem(t::NTangent, i) = t.lanes[i]
@inline _tangent_dir_elem(t, _) = t

@inline function _ndual_new_result(::Type{P}, y, x::Tuple, primals::Tuple) where {P}
    W = _ndual_width(x...)
    tangent_dirs = ntuple(W) do i
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
