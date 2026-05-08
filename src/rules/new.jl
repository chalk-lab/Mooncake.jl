@is_primitive MinimalCtx Tuple{typeof(_new_),Vararg}

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

    # Tuple primal: canonical V is the element-wise tuple-of-inner-duals
    # (`dual_type(Val(N), Tuple{...})` lifts each field). `bare_x` is already
    # the element-wise inner tuple, so wrap it directly. Tuple primals do not
    # use the `Dual{Tuple, NTangent}` shape.
    if P <: Tuple
        InnerT = dual_type(Val(N), P)
        if InnerT isa DataType && InnerT <: Tuple
            return Lifted{P,N,InnerT}(bare_x)
        end
    end

    # NamedTuple primal: parallel to Tuple — canonical V is
    # `NamedTuple{names, Tuple{V_i...}}` of bare inner duals. `bare_x` carries
    # the element-wise per-field inner duals; wrap into the named tuple.
    if P <: NamedTuple
        InnerT = dual_type(Val(N), P)
        if InnerT isa DataType && InnerT <: NamedTuple
            names = fieldnames(P)
            return Lifted{P,N,InnerT}(NamedTuple{names}(bare_x))
        end
    end

    # Struct with NDual content: at width=1 produce a bare-tangent Dual
    # (matches `dual_type(Val(1), P) = Dual{P, T}`); at width N>=2 wrap the
    # per-direction tangents in `NTangent`.
    if _has_ndual(bare_x...)
        primals_extracted = map(_ndual_primal, bare_x)
        y = _new_(P, primals_extracted...)
        T = tangent_type(P)
        T == NoTangent && return Lifted{P,N}(Dual(y, NoTangent()))
        if N == 1
            dir_tangents = map(v -> _tangent_dir(v, 1), bare_x)
            return Lifted{P,N}(
                Dual(y, build_output_tangent(P, primals_extracted, dir_tangents))
            )
        end
        tangent_dirs = ntuple(Val(N)) do dir
            dir_tangents = map(v -> _tangent_dir(v, dir), bare_x)
            build_output_tangent(P, primals_extracted, dir_tangents)
        end
        return Lifted{P,N}(Dual(y, NTangent(tangent_dirs)))
    end

    # No NDual content (e.g. all-non-differentiable args): pure struct construction.
    primals = map(__get_primal, bare_x)
    y = _new_(P, primals...)
    T = tangent_type(P)
    T == NoTangent && return Lifted{P,N}(Dual(y, NoTangent()))
    # Fold all-NoTangent `Tuple` field tangents to a single `NoTangent`, mirroring
    # `tangent_type(Tuple{...})`'s all-NoTangent fold so `build_output_tangent`
    # can place the result in a `NoTangent` struct field.
    field_tangents = map(bare_x) do v
        t = tangent(v)
        t isa Tuple{Vararg{NoTangent}} ? NoTangent() : t
    end
    return Lifted{P,N}(Dual(y, build_output_tangent(P, primals, field_tangents)))
end
@inline Mooncake._is_lifted_aware(::Type{<:Tuple{typeof(_new_),Vararg}}) = true

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
