@is_primitive MinimalCtx Tuple{typeof(_new_),Vararg}

# Lifted-arg `_new_` — branches on P shape inside the @generated body and
# returns a per-shape construction expression. All sub-function calls
# (`_new_`, `tuple_map`, `Lifted`, `ImmutableDual`, `MutableDual`) live in
# the returned expression per AGENTS.md; the generator body uses only
# introspection. In particular each branch's `dual_type(Val(Nw), P) === NoDual` collapse test
# (whole-`NoDual` vs element-wise V) is emitted into the returned expression and evaluated at the
# call world, never the generator body, so a more-specific or extension `dual_type` overload (e.g.
# CUDA `CuArray`/`CuPtr`) is respected. Specific overloads in other files (e.g.
# `_new_(Complex{P}, ::P, ::P)` in `complex.jl`) are more specific and
# take precedence.
#
# For the struct-lift case, constructor-omitted fields (`i > M` of `fieldcount(P)`)
# are padded with uninitialised backing (`fieldtype(backing, i)()`), matching
# `build_output_tangent`'s `PossiblyUninitTangent` behaviour.
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
            # An all-non-differentiable tuple (incl. the empty `Tuple{}`) has
            # `dual_type(P) === NoDual`; build whole `NoDual`, not an element-wise V.
            dual_type(Val(Nw), P) === NoDual && return Lifted{P,Nw}(y, NoDual())
            return Lifted{P,Nw}(y, tuple_map(tangent, x))
        end
    elseif P <: NamedTuple
        # An all-non-differentiable NamedTuple has `dual_type(P) === NoDual`, so build whole
        # `NoDual` rather than an element-wise V (same collapse as the Tuple/struct branches).
        names = (P.parameters[1])::Tuple
        return quote
            y = _new_(P, tuple_map(primal, x)...)
            dual_type(Val(Nw), P) === NoDual && return Lifted{P,Nw}(y, NoDual())
            return Lifted{P,Nw}(y, NamedTuple{$names}(tuple_map(tangent, x)))
        end
    elseif isprimitivetype(P) || fieldcount(P) == 0
        # Fieldless / primitive: no differentiable content, so V is `NoDual`.
        # (`isprimitivetype` / `fieldcount` are world-independent — fine in the body.)
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
            # A non-differentiable struct collapses to `NoDual` (same collapse as the
            # Tuple/NamedTuple branches).
            V = dual_type(Val(Nw), P)
            V === NoDual && return Lifted{P,Nw}(y, NoDual())
            # This branch can only build a struct-lift wrapper. When P's canonical V is a
            # dedicated container instead (e.g. `NDualMemoryRef` for `MemoryRef`), the backing
            # construction below would throw a baffling MethodError — fail clearly instead.
            # (Runtime guard, so a more-specific `frule!!` for such a P wins before reaching this.)
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

# `Ref(x)` / `RefValue{P}(x)` with `P<:NDualEltype` (real or complex scalar): the canonical V is
# `NDualRef`, not the generic struct lift. Its parallel partials buffer takes the seed's per-lane
# partials (`_nfwd_dual_partial` handles `NDual` and `Complex{NDual}`). More specific than the
# @generated `_new_` above, so it wins.
function frule!!(
    ::Lifted{typeof(_new_),Nw}, ::Lifted{Type{Base.RefValue{P}},Nw}, x::Lifted{P,Nw}
) where {Nw,P<:NDualEltype}
    pr = Base.RefValue{P}(primal(x))
    parts = ntuple(k -> _nfwd_dual_partial(tangent(x), k), Val(Nw))
    return Lifted{Base.RefValue{P},Nw}(
        pr, NDualRef{P,Nw}(Base.RefValue{NTuple{Nw,P}}(parts))
    )
end
# Zero-arg `RefValue{P}()` (uninitialised — no value to seed): canonical V is a zero-init
# `NDualRef`. Without this, M=0 falls into the @generated struct-lift branch above, which builds
# a `MutableDual` backing incoherent with `dual_type === NDualRef` and throws.
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
    ]
    general_test_cases = map(TestTypes.PRIMALS) do (interface_only, P, args)
        return (interface_only, :none, nothing, _new_, P, args...)
    end
    test_cases = vcat(specific_test_cases, general_test_cases)
    memory = Any[]
    return test_cases, memory
end

derived_rule_test_cases(rng_ctor, ::Val{:new}) = Any[], Any[]

@static if VERSION >= v"1.11"
    function throwing_rule_test_cases(::Val{:new})
        # Forward `_new_` on a type whose canonical V is a dedicated container (not a
        # struct-lift wrapper) must fail with the clear coherence error pointing at the
        # supported primitive (`memoryrefnew` for `MemoryRef`), not a baffling MethodError
        # from the backing construction.
        mem = fill!(Memory{Float64}(undef, 3), 1.0)
        ref = memoryref(mem)
        cases = Any[(
            "memoryrefnew",
            _new_,
            (
                zero_lifted(Val(1), MemoryRef{Float64}),
                zero_lifted(Val(1), ref.ptr_or_offset),
                zero_lifted(Val(1), mem),
            ),
        )]
        return cases, Any[mem, ref]
    end
end
