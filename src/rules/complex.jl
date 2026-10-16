# NOTE: Type aliasing does not reliably preserve parameter constraints.
# For example:
#   const ComplexFloat{P<:IEEEFloat} = Complex{P}
#   const CF{P} = ComplexFloat{P}
#
# Methods like
#   tangent_type(::Type{CF{P}}) where {P}
#   tangent_type(::Type{NoFData}, ::Type{CF{P}}) where {P}
#
# and even uses of `CF{P}` itself will accept `P = Int`, despite the original
# `P<:IEEEFloat` constraint on `ComplexFloat{P}`. Aliases do not enforce the
# bounds of the types they refer to.
#
# Therefore, we use forms like `Complex{<:IEEEFloat}` or a non-parametric alias
# `const CF = Complex{<:IEEEFloat}`.

const CF = Complex{<:IEEEFloat}

@foldable tangent_type(::Type{Complex{P}}) where {P<:IEEEFloat} = Complex{tangent_type(P)}
@foldable tangent_type(::Type{NoFData}, ::Type{Complex{P}}) where {P<:IEEEFloat} = Complex{
    P
}
@foldable fdata_type(::Type{T}) where {T<:CF} = NoFData
@foldable rdata_type(::Type{T}) where {T<:CF} = T

tangent(::NoFData, t::CF) = t

zero_tangent_internal(p::Complex{P}, ::MaybeCache) where {P<:IEEEFloat} = zero(p)

zero_rdata(p::CF) = zero_tangent(p)
zero_rdata_from_type(P::Type{<:CF}) = zero(P)

set_to_zero_internal!!(::SetToZeroCache, p::CF) = zero(p)

randn_tangent_internal(rng::AbstractRNG, p::CF, ::MaybeCache) = randn(rng, typeof(p))

_verify_rdata_value(::P, ::P) where {P<:CF} = nothing

increment_internal!!(::IncCache, t::T, s::T) where {T<:CF} = t + s

function increment_field!!(
    x::Complex{P}, re_or_im::P, ::Val{FieldName}
) where {P<:IEEEFloat,FieldName}
    return if (FieldName === :re) || (FieldName == 1)
        complex(real(x) + re_or_im, imag(x))
    elseif (FieldName === :im) || (FieldName == 2)
        complex(real(x), re_or_im + imag(x))
    else
        throw(ArgumentError(lazy"Unknown field `$FieldName` for type `$(Complex{P})`)"))
    end
end

tangent_to_primal_internal!!(x::P, t::P, ::MaybeCache) where {P<:CF} = t
primal_to_tangent_internal!!(t, x::CF, ::MaybeCache) = x

# Complex tangents are scalar values, not field-wise Tangent wrappers.
friendly_tangent_cache(::CF) = FriendlyTangentCache{AsRaw}(nothing)

_add_to_primal_internal(::MaybeCache, x::T, t::T, ::Bool) where {T<:CF} = x + t

function _dot_internal(::MaybeCache, t::T, s::T) where {T<:CF}
    _dot(real(t), real(s)) + _dot(imag(t), imag(s))
end

_scale_internal(::MaybeCache, a::Float64, t::T) where {T<:CF} = T(a * t)

TestUtils.populate_address_map_internal(m::TestUtils.AddressMap, ::P, ::P) where {P<:CF} = m

# Generic lgetfield rules use _get_lifted_field forward and increment_field!! reverse.
# _new_ is already primitive via the generic declaration in new.jl.
function frule!!(
    ::Lifted{typeof(_new_),N},
    ::Lifted{Type{Complex{P}},N},
    re::Lifted{P,N,NDual{P,N}},
    im::Lifted{P,N,NDual{P,N}},
) where {N,P<:IEEEFloat}
    x = _new_(Complex{P}, primal(re), primal(im))
    dx = _new_(Complex{NDual{P,N}}, tangent(re), tangent(im))
    return Lifted{Complex{P},N}(x, dx)
end
function rrule!!(
    ::CoDual{typeof(_new_)}, ::CoDual{Type{Complex{P}}}, re::CoDual{P}, im::CoDual{P}
) where {P<:IEEEFloat}
    _new_complex_pb(dy::Complex{P}) = NoRData(), NoRData(), real(dy), imag(dy)
    return zero_fcodual(_new_(Complex{P}, re.x, im.x)), _new_complex_pb
end

# Keep the structural wrapper: these views share the parent's derivative storage.
const FloatReinterpretArray = Union{
    Base.ReinterpretArray{Complex{T},D,T,P,false} where {T<:IEEEFloat,D,P},
    Base.ReinterpretArray{T,D,Complex{T},P,false} where {T<:IEEEFloat,D,P},
}
@inline function _check_reinterpret_layout(x, ::Type{T}, f) where {T}
    E = eltype(x)
    storage = x isa Array
    @static if VERSION >= v"1.11-"
        storage |= x isa Memory
    end
    if !(E === T || E === Complex{T}) || parent(x) === x && !storage
        throw(
            ArgumentError(
                "$f on ReinterpretArray requires matching $T/Complex{$T} storage; got $(typeof(x))",
            ),
        )
    end
    parent(x) === x || _check_reinterpret_layout(parent(x), T, f)
    return nothing
end
@inline _reinterpret_storage(p, d) = d isa typeof(p) ? d : last(arrayify(p, d))
@inline function _reinterpret_storage(p::SubArray, d::TangentOrFData)
    return view(_reinterpret_storage(parent(p), _fields(d).parent), p.indices...)
end
@inline function _reinterpret_storage(p::Base.ReshapedArray, d::TangentOrFData)
    return reshape(_reinterpret_storage(parent(p), _fields(d).parent), size(p))
end
@inline function _reinterpret_storage(p::Base.ReinterpretArray, d::TangentOrFData)
    return reinterpret(eltype(p), _reinterpret_storage(parent(p), _fields(d).parent))
end
@inline function _reinterpret_tangent(x::CoDual{<:FloatReinterpretArray}, f=getindex)
    p = primal(x).parent
    _check_reinterpret_layout(p, real(eltype(primal(x))), f)
    return reinterpret(
        eltype(primal(x)), _reinterpret_storage(p, _fields(tangent(x)).parent)
    )
end
@inline function _reinterpret_lanes(
    x::Lifted{<:FloatReinterpretArray,N}, mode::Val=Val(false), f=getindex
) where {N}
    _check_reinterpret_layout(primal(x).parent, real(eltype(primal(x))), f)
    return ntuple(Val(N)) do k
        reinterpret(
            eltype(primal(x)),
            _arrayify_lane(primal(x).parent, tangent(x).fields.parent, k, mode),
        )
    end
end

@is_primitive MinimalCtx Tuple{typeof(getindex),FloatReinterpretArray,Vararg{Int}}
function rrule!!(
    ::CoDual{typeof(getindex)}, x::CoDual{<:FloatReinterpretArray}, ii::CoDual{Int}...
)
    inds = map(primal, ii)
    dx = _reinterpret_tangent(x)
    y = getindex(primal(x), inds...)
    function reinterpret_getindex_pb(dy)
        dx[inds...] += dy
        return (NoRData(), NoRData(), map(_ -> NoRData(), ii)...)
    end
    return zero_fcodual(y), reinterpret_getindex_pb
end
function frule!!(
    ::Lifted{typeof(getindex),N}, x::Lifted{<:FloatReinterpretArray,N}, ii::Lifted{Int}...
) where {N}
    inds = map(primal, ii)
    y = getindex(primal(x), inds...)
    return _lift_from_lanes(y, map(d -> d[inds...], _reinterpret_lanes(x, Val(:read))))
end

@is_primitive MinimalCtx Tuple{
    typeof(setindex!),FloatReinterpretArray,Union{IEEEFloat,CF,Integer},Vararg{Int}
}
function rrule!!(
    ::CoDual{typeof(setindex!)},
    x::CoDual{<:FloatReinterpretArray},
    v::CoDual{V},
    ii::CoDual{Int}...,
) where {V<:Union{IEEEFloat,CF,Integer}}
    inds = map(primal, ii)
    p, dx = primal(x), _reinterpret_tangent(x, setindex!)
    old, old_d = p[inds...], dx[inds...]
    p[inds...] = primal(v)
    dx[inds...] = zero(old_d)
    function reinterpret_setindex_pb(::NoRData)
        dv = dx[inds...]
        p[inds...] = old
        dx[inds...] = old_d
        dv = V <: Integer ? NoRData() : V(V <: Real ? real(dv) : dv)
        return (NoRData(), NoRData(), dv, map(_ -> NoRData(), ii)...)
    end
    return x, reinterpret_setindex_pb
end
function frule!!(
    ::Lifted{typeof(setindex!),N},
    x::Lifted{<:FloatReinterpretArray,N},
    v::Lifted{V,N},
    ii::Lifted{Int}...,
) where {N,V<:Union{IEEEFloat,CF,Integer}}
    inds = map(primal, ii)
    ds = _reinterpret_lanes(x, Val(false), setindex!)
    p = primal(x)
    p[inds...] = primal(v)
    for k in 1:N
        dv = V <: Integer ? zero(eltype(p)) : tangent(v, k)
        ds[k][inds...] = eltype(p) <: Real ? real(dv) : dv
    end
    return x
end

# Inference may inline the public accessors before the target element type is known.
# Catch the same logical operations at Base's remaining array-level boundary.
@is_primitive MinimalCtx Tuple{
    typeof(Base._getindex_ra),FloatReinterpretArray,Int,Tuple{Vararg{Int}}
}
function rrule!!(
    ::CoDual{typeof(Base._getindex_ra)},
    x::CoDual{<:FloatReinterpretArray},
    i::CoDual{Int},
    tail::CoDual{<:Tuple{Vararg{Int}}},
)
    y, pb = rrule!!(zero_fcodual(getindex), x, i, map(zero_fcodual, primal(tail))...)
    function reinterpret_getindex_ra_pb(dy)
        pb(dy)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return y, reinterpret_getindex_ra_pb
end
function frule!!(
    ::Lifted{typeof(Base._getindex_ra),N},
    x::Lifted{<:FloatReinterpretArray,N},
    i::Lifted{Int,N},
    tail::Lifted{<:Tuple{Vararg{Int}},N},
) where {N}
    return frule!!(
        zero_lifted(Val(N), getindex),
        x,
        i,
        map(j -> zero_lifted(Val(N), j), primal(tail))...,
    )
end
@is_primitive MinimalCtx Tuple{
    typeof(Base._setindex_ra!),
    FloatReinterpretArray,
    Union{IEEEFloat,CF,Integer},
    Int,
    Tuple{Vararg{Int}},
}
function rrule!!(
    ::CoDual{typeof(Base._setindex_ra!)},
    x::CoDual{<:FloatReinterpretArray},
    v::CoDual{<:Union{IEEEFloat,CF,Integer}},
    i::CoDual{Int},
    tail::CoDual{<:Tuple{Vararg{Int}}},
)
    y, pb = rrule!!(zero_fcodual(setindex!), x, v, i, map(zero_fcodual, primal(tail))...)
    function reinterpret_setindex_ra_pb(dy)
        dv = pb(dy)[3]
        return NoRData(), NoRData(), dv, NoRData(), NoRData()
    end
    return y, reinterpret_setindex_ra_pb
end
function frule!!(
    ::Lifted{typeof(Base._setindex_ra!),N},
    x::Lifted{<:FloatReinterpretArray,N},
    v::Lifted{<:Union{IEEEFloat,CF,Integer},N},
    i::Lifted{Int,N},
    tail::Lifted{<:Tuple{Vararg{Int}},N},
) where {N}
    return frule!!(
        zero_lifted(Val(N), setindex!),
        x,
        v,
        i,
        map(j -> zero_lifted(Val(N), j), primal(tail))...,
    )
end

# Widen half precision only for the value mask, preserving its live outer perturbation.
function _reinterpret_dot_zero(x::T, zero_seed::Bool) where {T}
    y = T <: Union{Float16,ComplexF16} ? widen(x) : x
    return T(_rvs_zero(y, zero_seed))
end

function _reinterpret_dot_add!(dx, x, a)
    iszero(a) && return nothing
    if eltype(dx) <: BlasFloat && dx isa StridedVector && x isa StridedVector && isfinite(a)
        BLAS.axpy!(a, x, dx)
    else
        @inbounds for i in eachindex(dx, x)
            dx[i] += _reinterpret_dot_zero(x[i] * a, iszero(x[i]))
        end
    end
    return nothing
end
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(_reinterpret_dot_add!),AbstractVector{T},AbstractVector{T},T
} where {T<:Union{IEEEFloat,CF}}
function frule!!(
    ::Lifted{typeof(_reinterpret_dot_add!),N}, dx::Lifted, x::Lifted, a::Lifted
) where {N}
    for k in 1:N
        ddx = _arrayify_lane(primal(dx), tangent(dx), k)
        tx = _arrayify_lane(primal(x), tangent(x), k, Val(:read))
        _reinterpret_dot_add!(ddx, tx, primal(a))
        _reinterpret_dot_add!(ddx, primal(x), tangent(a, k))
    end
    _reinterpret_dot_add!(primal(dx), primal(x), primal(a))
    return zero_lifted(Val(N), nothing)
end

# Only direct, matching-component operands are adapted. Wrapped BLAS operands retain
# their existing refusal; the logical dot never retypes a derivative pointer.
for (E, S) in ((:T, :(Complex{T})), (:(Complex{T}), :T))
    R = :(Base.ReinterpretArray{$E,1,$S,P,false} where {P})
    A = :(AbstractVector{$E})
    for (X, Y) in ((R, A), (A, R), (R, R))
        @eval begin
            @is_primitive MinimalCtx Tuple{typeof(dot),$X,$Y} where {T<:IEEEFloat}
            function rrule!!(
                ::CoDual{typeof(dot)}, x::CoDual{<:$X}, y::CoDual{<:$Y}
            ) where {T<:IEEEFloat}
                xp, yp = primal(x), primal(y)
                xp isa FloatReinterpretArray || _check_blas_operand(xp)
                yp isa FloatReinterpretArray || _check_blas_operand(yp)
                dx = if xp isa FloatReinterpretArray
                    _reinterpret_tangent(x, dot)
                else
                    (tangent(x) isa typeof(xp) ? tangent(x) : last(arrayify(x)))
                end
                dy = if yp isa FloatReinterpretArray
                    _reinterpret_tangent(y, dot)
                else
                    (tangent(y) isa typeof(yp) ? tangent(y) : last(arrayify(y)))
                end
                v = dot(xp, yp)
                function reinterpret_dot_pb(dv)
                    _reinterpret_dot_add!(dx, yp, conj(dv))
                    _reinterpret_dot_add!(dy, xp, dv)
                    return NoRData(), NoRData(), NoRData()
                end
                return zero_fcodual(v), reinterpret_dot_pb
            end
            function frule!!(
                ::Lifted{typeof(dot),N}, x::Lifted{<:$X,N}, y::Lifted{<:$Y,N}
            ) where {T<:IEEEFloat,N}
                xp, yp = primal(x), primal(y)
                xp isa FloatReinterpretArray || _check_blas_operand(xp)
                yp isa FloatReinterpretArray || _check_blas_operand(yp)
                dx = if xp isa FloatReinterpretArray
                    _reinterpret_lanes(x, Val(:read), dot)
                else
                    ntuple(k -> _arrayify_lane(xp, tangent(x), k, Val(:read)), Val(N))
                end
                dy = if yp isa FloatReinterpretArray
                    _reinterpret_lanes(y, Val(:read), dot)
                else
                    ntuple(k -> _arrayify_lane(yp, tangent(y), k, Val(:read)), Val(N))
                end
                v = dot(xp, yp)
                ds = ntuple(Val(N)) do k
                    acc = zero(v)
                    @inbounds for i in eachindex(xp, yp)
                        acc +=
                            _reinterpret_dot_zero(
                                only(_fwd_guarded_scale((conj(dx[k][i]),), yp[i])),
                                iszero(yp[i]),
                            ) + _reinterpret_dot_zero(
                                only(_fwd_guarded_scale((dy[k][i],), conj(xp[i]))),
                                iszero(xp[i]),
                            )
                    end
                    acc
                end
                return _lift_from_lanes(v, ds)
            end
        end
    end
end

# Register primitives separately for chunked checks (derived cases use is_primitive=false).
function hand_written_rule_test_cases(rng_ctor, ::Val{:complex})
    cases = Any[
        (false, :stability_and_allocs, nothing, lgetfield, 1.5 - 0.5im, Val(:re)),
        (false, :stability_and_allocs, nothing, lgetfield, 1.5 - 0.5im, Val(:im)),
        (false, :stability_and_allocs, nothing, _new_, ComplexF64, 1.5, -0.5),
    ]
    for T in (Float32, Float64), (E, S) in ((T, Complex{T}), (Complex{T}, T))
        x = reinterpret(E, S[1, 2, 3, 4])
        for inds in ((1,), (1, 2))
            a = length(inds) == 1 ? x : reinterpret(E, reshape(parent(x), 2, 2))
            push!(cases, (false, :none, (unsafe_perturb=true,), getindex, a, inds...))
            push!(
                cases,
                (
                    false,
                    :none,
                    (unsafe_perturb=true,),
                    Base._getindex_ra,
                    a,
                    first(inds),
                    Base.tail(inds),
                ),
            )
        end
        for v in (E(2), T === Float32 ? 2.5 : 2.0f0, 2)
            push!(cases, (false, :none, (unsafe_perturb=true,), setindex!, x, v, 1))
            push!(
                cases,
                (false, :none, (unsafe_perturb=true,), Base._setindex_ra!, x, v, 1, ()),
            )
        end
        for (a, b) in ((x, ones(E, length(x))), (ones(E, length(x)), x), (x, x))
            push!(cases, (false, :none, (unsafe_perturb=true,), dot, a, b))
        end
        push!(
            cases,
            (
                false,
                :none,
                (unsafe_perturb=true, mode=ForwardMode),
                _reinterpret_dot_add!,
                x,
                ones(E, length(x)),
                E(2),
            ),
        )
    end
    # Reads must project structural constants instead of using writable lane wrappers.
    push!(
        cases,
        (
            false,
            :none,
            (unsafe_perturb=true, mode=ForwardMode),
            getindex,
            reinterpret(Float64, UnitUpperTriangular(ComplexF64[2+3im 4+5im; 6+7im 8+9im])),
            1,
            1,
        ),
    )
    return cases, Any[]
end

function derived_rule_test_cases(rng_ctor, ::Val{:complex})
    test_cases = Any[
        (false, :none, nothing, real, 1.0 + 2.0im),
        (false, :none, nothing, imag, 1.0 + 2.0im),
        (false, :none, nothing, z -> z.re * z.im, 1.5 - 0.5im),
        (false, :none, nothing, (a, b) -> Complex(a, b), 1.0, 2.0),
        (false, :none, nothing, (a, b) -> abs2(Complex(a, b)), 1.0, 2.0),
    ]
    for T in (Float32, Float64), (E, S) in ((T, Complex{T}), (Complex{T}, T))
        x = S[1, 2, 3, 4]
        for f in (
            v -> sum(abs2, reinterpret(E, v)),
            v -> sum(abs2, reinterpret(E, v)) + real(sum(v)),
            v -> real(dot(ones(E, length(reinterpret(E, v))), reinterpret(E, v))),
            v -> begin
                r = reinterpret(E, v)
                r[1] = 2r[2]
                sum(abs2, r) + real(sum(v))
            end,
            v -> sum(abs2, reinterpret(E, view(v, 1:2:3))),
            v -> sum(abs2.(reinterpret(E, v))),
            v -> sum(abs2, reinterpret(S, view(reinterpret(E, v), :))),
        )
            push!(test_cases, (false, :none, nothing, f, x))
        end
        @static if VERSION >= v"1.11-"
            push!(
                test_cases,
                (false, :none, nothing, v -> sum(abs2, reinterpret(E, v)), Memory{S}(x)),
            )
        end
    end
    # Half-precision rounding defeats infinitesimal finite differences; use exact seeds.
    for E in (Float16, Complex{Float16}),
        (op, d) in ((:sum, Float16(4)), (:alias, Float16(5)), (:dot, Float16(1)))

        f = x -> begin
            p = E <: Real ? [complex(x, Float16(1))] : [x, Float16(1)]
            r = reinterpret(E, p)
            op === :dot && return real(dot(ones(E, length(r)), r))
            sum(abs2, r) + (op === :alias ? real(sum(p)) : Float16(0))
        end
        opts = (oracle=(deriv=(fwd=d, rvs=(NoRData(), d)),), output_tangent=Float16(1))
        push!(test_cases, (false, :none, opts, f, CoDual(Float16(2), Float16(1))))
    end
    for E in (Float16, ComplexF16), memory in (false, true)
        memory && VERSION < v"1.11-" && continue
        f = x -> begin
            p = E <: Real ? [complex(x, Float16(1))] : [x, Float16(1)]
            @static if VERSION >= v"1.11-"
                memory && (p = Memory{eltype(p)}(p))
            end
            sum(abs2, reinterpret(E, view(p, :)))
        end
        opts = (
            oracle=(deriv=(fwd=Float16(4), rvs=(NoRData(), Float16(4))),),
            output_tangent=Float16(1),
        )
        push!(test_cases, (false, :none, opts, f, CoDual(Float16(2), Float16(1))))
    end
    for E in (Float16, ComplexF16)
        f = x -> begin
            p = if E <: Real
                [complex(Float16(1), x), complex(Float16(3), x)]
            else
                [x, Float16(1), x, Float16(3)]
            end
            inds = E <: Real ? (2:2:4) : (1:2)
            real(dot(ones(E, 2), view(reinterpret(E, p), inds)))
        end
        opts = (
            oracle=(deriv=(fwd=Float16(2), rvs=(NoRData(), Float16(2))),),
            output_tangent=Float16(1),
        )
        push!(test_cases, (false, :none, opts, f, CoDual(Float16(2), Float16(1))))
    end
    for wrap in (x -> view(x, :), x -> reshape(x, 2, 1))
        f = v -> sum(abs2, reinterpret(ComplexF64, wrap(reinterpret(Float64, v))))
        push!(
            test_cases,
            (
                false,
                :none,
                (throws=(ArgumentError, "getindex on ReinterpretArray requires matching"),),
                f,
                Float32[1, 2, 3, 4],
            ),
        )
    end
    for f in (
        z -> sum(reinterpret(Float64, z) * [2.0 0.0; 0.0 3.0]),
        z -> sum(cholesky(Symmetric(reinterpret(Float64, z))).U),
    )
        push!(test_cases, (false, :none, nothing, f, ComplexF64[2+0im 0+3im]))
    end
    push!(
        test_cases,
        (
            false,
            :none,
            (throws=(ArgumentError, "BLAS does not support operand"),),
            z -> sum(abs2, BLAS.scal!(2.0, reinterpret(Float64, z))),
            ComplexF64[1 + 2im, 3 + 4im],
        ),
    )
    for (E, x) in (
            (ComplexF64, reshape(collect(1.0:8.0), 2, 4)),
            (Float64, ComplexF64[1 + 2im, 3 + 4im]),
        ),
        mode in (ReverseMode, ForwardMode),
        w in (1, 8)

        mode === ReverseMode && w == 8 && continue
        msg = if mode === ReverseMode || (E === ComplexF64 && VERSION >= v"1.11-" && w == 1)
            "to another element type is not supported"
        elseif E === ComplexF64 && VERSION >= v"1.11-"
            "is unsupported at chunk width 8 > 1"
        else
            "cannot take `pointer_from_objref` of a `Ref{ComplexF64}`"
        end
        push!(
            test_cases,
            (
                false,
                :none,
                (throws=(ArgumentError, msg), mode=mode, chunk_size=w),
                v -> sum(abs2, reinterpret(reshape, E, v)),
                x,
            ),
        )
    end
    push!(
        test_cases,
        (
            false,
            :none,
            (throws="BLAS does not support operand",),
            (w, z) -> dot(w, view(reinterpret(Float64, z), 2:2:4)),
            ones(2),
            ComplexF64[1 + 2im, 3 + 4im],
        ),
    )
    return test_cases, Any[]
end
