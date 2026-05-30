@is_primitive MinimalCtx Tuple{typeof(_new_),Vararg}

# Lifted-arg `_new_` — branches on P shape inside the @generated body and
# returns a per-shape construction expression. All sub-function calls
# (`_new_`, `tuple_map`, `Lifted`, `ImmutableDual`, `MutableDual`) live in
# the returned expression per AGENTS.md; the generator body uses only
# introspection. Specific overloads in other files (e.g.
# `_new_(Complex{P}, ::P, ::P)` in `complex.jl`) are more specific and
# take precedence.
#
# Assumes `M == fieldcount(P)` for the struct-lift case — padding for
# default-initialized fields via `PossiblyUninitTangent` (matching
# `build_output_tangent`'s behaviour) is deferred to a follow-up.
@generated function frule!!(
    ::Lifted{typeof(_new_),Nw}, ::Lifted{Type{P},Nw}, x::Vararg{Lifted,M}
) where {P,Nw,M}
    if !isconcretetype(P)
        msg = "_new_ Lifted: P=$P is not concrete"
        return :(error($msg))
    end
    if P <: Tuple
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            return Lifted{P,Nw}(y, tuple_map(tangent, x))
        end
    elseif P <: NamedTuple
        # An all-non-differentiable NamedTuple has `dual_type(P) === NoDual`, so
        # build whole `NoDual` rather than an element-wise V (which the generic
        # `tangent_type === NoTangent` branch below would also do for structs).
        if tangent_type(P) === NoTangent
            return quote
                y = _new_(P, tuple_map(primal, x)...)
                return Lifted{P,Nw}(y, NoDual())
            end
        end
        names = (P.parameters[1])::Tuple
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            return Lifted{P,Nw}(y, NamedTuple{$names}(tuple_map(tangent, x)))
        end
    elseif isprimitivetype(P) || fieldcount(P) == 0 || tangent_type(P) === NoTangent
        # Non-differentiable structs (e.g. `Base.OneTo`, all-Int fields) have
        # `dual_type(P) === NoDual`, mirroring `tangent_type(P) === NoTangent`.
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            return Lifted{P,Nw}(y, NoDual())
        end
    else
        wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
        inits = always_initialised(P)
        # Coerce the field-V tuple into the *declared* backing NamedTuple
        # `fieldtype(dual_type(Val(Nw), P), 1)`: a field declared abstract is
        # stored as `Any`; a non-always-init field as `PossiblyUninitTangent`,
        # initialised from its backing field type when the arg is supplied
        # (`i <= M`) or left uninit when `_new_` omits it (`i > M`, e.g.
        # `StructFoo(a)` leaving `b` undefined). Keeps V `=== dual_type(Val(Nw), P)`.
        field_exprs = map(1:fieldcount(P)) do i
            i > M && return :(fieldtype(backing, $i)())
            base = :(tangent(x[$i]))
            inits[i] ? base : :(fieldtype(backing, $i)($base))
        end
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            backing = fieldtype(dual_type(Val(Nw), P), 1)
            return Lifted{P,Nw}(y, $wrapper(backing(($(field_exprs...),))))
        end
    end
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
