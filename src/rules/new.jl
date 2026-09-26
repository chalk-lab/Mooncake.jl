@is_primitive MinimalCtx Tuple{typeof(_new_),Vararg}

# Keep construction and dual_type calls in returned code to respect call-world overloads.
# Constructor-omitted fields use uninitialised backing, as in build_output_tangent.
@generated function frule!!(
    ::Lifted{typeof(_new_),Nw}, ::Lifted{Type{P},Nw}, x::Vararg{Lifted,M}
) where {P,Nw,M}
    if !isconcretetype(P)
        msg = "_new_ Lifted: P=$P is not concrete"
        return :(error($msg))
    end
    if P <: Union{Tuple,NamedTuple}
        fields = :(tuple_map(tangent, x))
        if P <: NamedTuple
            fields = :(NamedTuple{$(P.parameters[1]::Tuple)}($fields))
        end
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            # Non-differentiable tuples, including Tuple{}, collapse to whole NoDual.
            dual_type(Val(Nw), P) === NoDual && return Lifted{P,Nw}(y, NoDual())
            return Lifted{P,Nw}(y, $fields)
        end
    elseif fieldcount(P) == 0
        # Fieldless types (including primitives) have no differentiable content.
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            return Lifted{P,Nw}(y, NoDual())
        end
    else
        wrapper = ismutabletype(P) ? :MutableDual : :ImmutableDual
        inits = always_initialised(P)
        # Use declared backing types: abstract fields store Any, and possibly uninitialised
        # fields wrap supplied values or remain uninitialised when omitted. This keeps V canonical.
        field_exprs = map(1:fieldcount(P)) do i
            i > M && return :(fieldtype(backing, $i)())
            base = :(tangent(x[$i]))
            inits[i] ? base : :(fieldtype(backing, $i)($base))
        end
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            # Non-differentiable structs also collapse to whole NoDual.
            V = dual_type(Val(Nw), P)
            V === NoDual && return Lifted{P,Nw}(y, NoDual())
            # Dedicated containers need specific rules; struct backing construction is invalid for them.
            V <: Union{ImmutableDual,MutableDual} || error(
                "forward _new_($P, ...): the canonical forward representation is $V, not a " *
                "struct-lift Immutable/MutableDual, so the generic struct construction does " *
                "not apply. Construct the value via its dedicated primitive (e.g. " *
                "`memoryrefnew` for `MemoryRef`), or add a specific `frule!!` for this signature.",
            )
            backing = fieldtype(V, 1)
            return Lifted{P,Nw}(y, $wrapper(backing(($(field_exprs...),))))
        end
    end
end

# Real/complex RefValue uses NDualRef, not struct lift; seed its per-lane partials.
function frule!!(
    ::Lifted{typeof(_new_),Nw}, ::Lifted{Type{Base.RefValue{P}},Nw}, x::Lifted{P,Nw}
) where {Nw,P<:NDualEltype}
    pr = Base.RefValue{P}(primal(x))
    parts = ntuple(k -> _nfwd_dual_partial(tangent(x), k), Val(Nw))
    return Lifted{Base.RefValue{P},Nw}(
        pr, NDualRef{P,Nw}(Base.RefValue{NTuple{Nw,P}}(parts))
    )
end
# Uninitialised real/complex RefValue needs zero-init NDualRef, not MutableDual.
function frule!!(
    ::Lifted{typeof(_new_),Nw}, ::Lifted{Type{Base.RefValue{P}},Nw}
) where {Nw,P<:NDualEltype}
    return Lifted{Base.RefValue{P},Nw}(Base.RefValue{P}(), NDualRef{P,Nw}())
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
# A `_new_`ed IdDict has an empty tangent (itself an IdDict, which `fdata_type` maps to
# itself), and a pullback that returns nothing, every field it is built from having
# `NoRData` rdata. The generic method above instead reads the raw `ht::Memory` field as an
# iterable of `(k, v)` pairs, and builds a pullback expecting a `MutableTangent`.
function rrule!!(
    f::CoDual{typeof(_new_)}, p::CoDual{Type{P}}, x::Vararg{CoDual,N}
) where {P<:IdDict,N}
    y = _new_(P, tuple_map(primal, x)...)
    return CoDual(y, tangent_type(P)()), NoPullback(f, p, x...)
end
# IdDict uses a dedicated dual container; its non-differentiable constructor fields
# produce an empty dual dict.
function frule!!(
    ::Lifted{typeof(_new_),Nw}, ::Lifted{Type{P},Nw}, x::Vararg{Lifted,N}
) where {P<:IdDict,Nw,N}
    y = _new_(P, tuple_map(primal, x)...)
    return Lifted{P,Nw}(y, dual_type(Val(Nw), P)())
end

@inline function build_output_tangent(::Type{P}, x::Tuple, t::Tuple) where {P}
    return _build_output_tangent_cartesian(P, x, t, Val(fieldcount(P)), Val(fieldnames(P)))
end
@inline function build_output_tangent(::Type{P}, ::Tuple, ::Tuple) where {P<:IdDict}
    return tangent_type(P)()   # see the IdDict `rrule!!` above
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
        # `Ref(x)` / `RefValue{P}(x)` construction (real + complex): the canonical V is `NDualRef`.
        # `:none` perf — a mutable `Ref` allocates, so the alloc check does not apply.
        (false, :none, nothing, _new_, Base.RefValue{Float64}, 5.0),
        (false, :none, nothing, _new_, Base.RefValue{ComplexF64}, 1.0 + 2.0im),
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
        # `ht` is `Memory{Any}` on 1.11+ but `Vector{Any}` on 1.10. Omitting the fields
        # entirely leaves `ht` undefined, which segfaults for reasons of its own.
        (
            false,
            :none,
            nothing,
            _new_,
            IdDict{Int,Float64},
            fieldtype(IdDict{Int,Float64}, :ht)(undef, 0),
            0,
            0,
        ),
    ]
    general_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        return (interface_only, :none, nothing, _new_, P, args...)
    end
    # Dedicated containers must report the coherence error and supported primitive.
    # MemoryRef is unavailable before Julia 1.11.
    coherence_cases, coherence_memory = @static if VERSION >= v"1.11-"
        let mem = fill!(Memory{Float64}(undef, 3), 1.0), ref = memoryref(mem)
            Any[(
                false,
                :none,
                (throws="memoryrefnew", mode=ForwardMode),
                _new_,
                MemoryRef{Float64},
                zero_lifted(Val(1), ref.ptr_or_offset),
                mem,
            )],
            Any[mem, ref]
        end
    else
        Any[], Any[]
    end
    test_cases = vcat(specific_test_cases, general_test_cases, coherence_cases)
    memory = coherence_memory
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:new}) = Any[], Any[]
