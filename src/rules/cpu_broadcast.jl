# Reverse-mode rule for broadcasting over CPU arrays.
#
# Without this rule Mooncake differentiates the broadcast loop one element at a time, which
# is an order of magnitude slower than the primal. Instead, `materialize` of a supported
# broadcast is a primitive: the whole (fused) broadcast is evaluated once on `NDual`s, with
# one partial per differentiable leaf, and the pullback contracts those partials against the
# output cotangent. This is the CPU counterpart of the `materialize` rule in MooncakeCUDAExt.
#
# Only broadcasts that this can handle exactly are claimed. Everything else keeps the
# existing derived rule. A broadcast is supported if
#   - its style is `DefaultArrayStyle{N}` with `N > 0` (so the result is an `Array`),
#   - every leaf is an `Array{<:IEEEFloat}`, an `IEEEFloat` scalar, or a value whose
#     tangent type is `NoTangent` and whose elements are not `IEEEFloat`s (this includes
#     stateless functions such as the `^` in `x .^ 3`),
#   - no broadcast function carries differentiable state, or is a `Type`,
#   - there are between 1 and `_CPU_BCAST_MAX_SLOTS` differentiable leaves, and
#   - the flattened function, applied to `NDual` elements, is inferred to return a concrete
#     `NDual`. A function that has no method for `NDual` inputs therefore falls back, and
#   - the flattened function cannot reach `Nfwd`'s strong-zero guard (see below).
#
# The result must be the one the derived rule gives, edge cases included. Two places differ:
#   - `Nfwd` masks `0 * Inf` to `0` in forward mode at the singular points of `sqrt`, `cbrt`,
#     `log` and real powers, where reverse mode gives `NaN`. Broadcasts that can reach this
#     guard are not claimed; this is decided by inference, conservatively.
#   - Any other non-finite derivative (or value) may be combined in a different order, or
#     masked by a different rule, than in reverse mode. If any element of the output value
#     or partials is not finite, the rule hands the whole broadcast to the derived rule.
#
# Forward mode is not affected. Forward-over-reverse differentiates the body of this rule
# like any other Julia code, so second-order derivatives are still available.

using Base.Broadcast: Broadcasted

const _CPUBroadcasted = Broadcasted{<:Base.Broadcast.DefaultArrayStyle}

const _CPU_BCAST_MAX_SLOTS = 8

# Element type that a leaf contributes to the broadcast, or `nothing` if unsupported.
_cpu_bcast_leaf_eltype(::Type{<:Array{T}}) where {T<:IEEEFloat} = T
_cpu_bcast_leaf_eltype(::Type{T}) where {T<:IEEEFloat} = T
function _cpu_bcast_leaf_eltype(::Type{L}) where {L}
    isconcretetype(L) || return nothing
    # Broadcasting reads a `Ref` only through its contents (`x .^ 3` wraps `^` and `Val(3)`).
    L <: Base.RefValue && return _cpu_bcast_leaf_eltype(eltype(L))
    tangent_type(L) === NoTangent || return nothing
    E = if L <: AbstractArray
        eltype(L)
    elseif L <: Number ||
        L <: Symbol ||
        L <: Val ||
        (L <: Function && Base.issingletontype(L))
        L
    else
        return nothing
    end
    # A non-differentiable leaf that still has float elements would be given a slot.
    return E <: Union{IEEEFloat,Complex} ? nothing : E
end

_cpu_bcast_is_diff_leaf(::Type{<:Array{<:IEEEFloat}}) = true
_cpu_bcast_is_diff_leaf(::Type{<:IEEEFloat}) = true
_cpu_bcast_is_diff_leaf(::Type) = false

# Walk the (nested) broadcast type: are all functions and leaves supported, and how many
# differentiable leaves are there?
function _cpu_bcast_scan(::Type{B}) where {B<:Broadcasted}
    isconcretetype(B) || return (false, 0)
    F = fieldtype(B, :f)
    (F <: Type || tangent_type(F) !== NoTangent) && return (false, 0)
    n = 0
    for A in fieldtypes(fieldtype(B, :args))
        if A <: Broadcasted
            ok, m = _cpu_bcast_scan(A)
            ok || return (false, 0)
            n += m
        elseif _cpu_bcast_leaf_eltype(A) === nothing
            return (false, 0)
        else
            n += _cpu_bcast_is_diff_leaf(A)
        end
    end
    return (true, n)
end

function _cpu_bcast_supported(::Type{B}) where {B}
    B <: Broadcasted{<:Base.Broadcast.DefaultArrayStyle} || return false
    B <: Broadcasted{Base.Broadcast.DefaultArrayStyle{0}} && return false
    ok, n = _cpu_bcast_scan(B)
    (ok && 1 <= n <= _CPU_BCAST_MAX_SLOTS) || return false
    FB = Core.Compiler.return_type(Base.Broadcast.flatten, Tuple{B})
    (isconcretetype(FB) && FB <: Broadcasted) || return false
    Ts = map(_cpu_bcast_leaf_eltype, fieldtypes(fieldtype(FB, :args)))
    any(isnothing, Ts) && return false
    R = Core.Compiler.return_type(_cpu_apply_with_duals, Tuple{fieldtype(FB, :f),Ts...})
    (isconcretetype(R) && R <: Nfwd.NDual) || return false
    return !_cpu_bcast_may_reach_guard(fieldtype(FB, :f), Ts)
end

# Can `f` reach `Nfwd._pt_guarded_scale` on some path? `NDual{T,0}` is never built outside
# this check, and its guard is replaced by one with a side effect, so the answer is whether
# inference finds `f` free of side effects on width-0 duals. If inference cannot tell, the
# answer is yes.
@noinline _cpu_bcast_guard_reached() = (_CPU_BCAST_GUARD_SINK[]=true; nothing)
const _CPU_BCAST_GUARD_SINK = Ref(false)
@inline function Nfwd._pt_guarded_scale(p::Tuple{}, ::T) where {T}
    _cpu_bcast_guard_reached()
    return p
end
@generated function _cpu_apply_with_probe_duals(f::F, args...) where {F}
    wrapped = map(enumerate(args)) do (i, A)
        A <: IEEEFloat ? :(Nfwd.NDual{$A,0}(args[$i], ())) : :(args[$i])
    end
    return :(f($(wrapped...)))
end
function _cpu_bcast_may_reach_guard(F::Type, Ts)
    effects = Base.infer_effects(_cpu_apply_with_probe_duals, Tuple{F,Ts...})
    return !Core.Compiler.is_effect_free(effects)
end

# Anything this rule does not claim (GPU broadcasts included) keeps whatever `MinimalCtx`
# decides, so existing `materialize` primitives are not shadowed.
function _is_primitive(
    ::Type{DefaultCtx}, M::Type{<:ReverseMode}, ::Type{S}
) where {S<:Tuple{typeof(Base.Broadcast.materialize),Broadcasted}}
    return _cpu_bcast_supported(fieldtype(S, 2)) || _is_primitive(MinimalCtx, M, S)
end
@static if VERSION < v"1.11-"
    # Julia 1.10 can expose the `copy` inside `materialize` as the call boundary.
    function _is_primitive(
        ::Type{DefaultCtx}, M::Type{<:ReverseMode}, ::Type{S}
    ) where {S<:Tuple{typeof(copy),Broadcasted}}
        return _cpu_bcast_supported(fieldtype(S, 2)) || _is_primitive(MinimalCtx, M, S)
    end
end

# Seed each differentiable element with a one-hot partial and call `f`. The slots are laid
# out in leaf order, one per `IEEEFloat` argument.
@generated function _cpu_apply_with_duals(f::F, args...) where {F}
    N = count(A -> A <: IEEEFloat, args)
    N == 0 && return :(f(args...))
    slot = 0
    wrapped = map(enumerate(args)) do (i, A)
        A <: IEEEFloat || return :(args[$i])
        slot += 1
        partials = Expr(:tuple, (A(j == slot) for j in 1:N)...)
        return :(Nfwd.NDual{$A,$N}(args[$i], $partials))
    end
    return :(f($(wrapped...)))
end

struct _CPUApplyWithDuals{F}
    f::F
end
@inline (a::_CPUApplyWithDuals)(args...) = _cpu_apply_with_duals(a.f, args...)

# Leaves of a broadcast tree, and the matching fdata, in the order `flatten` uses.
function _cpu_bcast_leaves(bc::Broadcasted, fd)
    return _cpu_bcast_leaves_args(bc.args, _cpu_args_fdata(bc, fd))
end
_cpu_args_fdata(bc::Broadcasted, ::NoFData) = map(_ -> NoFData(), bc.args)
_cpu_args_fdata(::Broadcasted, fd::FData) = _fields(fd).args
_cpu_bcast_leaves_args(::Tuple{}, ::Tuple{}) = ((), ())
function _cpu_bcast_leaves_args(args::Tuple, fds::Tuple)
    rest_p, rest_f = _cpu_bcast_leaves_args(Base.tail(args), Base.tail(fds))
    a, fd = first(args), first(fds)
    if a isa Broadcasted
        p, f = _cpu_bcast_leaves(a, fd)
        return (p..., rest_p...), (f..., rest_f...)
    end
    return (a, rest_p...), (fd, rest_f...)
end

# Sum `dx` down to `sz`, undoing broadcasting's extension of singleton and missing dims.
function _cpu_unbroadcast(dx::Array, sz::Tuple)
    size(dx) == sz && return dx
    dims = filter(!iszero, ntuple(d -> (d > length(sz) || sz[d] == 1) ? d : 0, ndims(dx)))
    return reshape(sum(dx; dims), sz)
end

# rdata of the broadcast, with the gradients of its scalar leaves filled in. `grads` holds
# them in leaf order; `idx` is the position of the next one.
function _cpu_bcast_rdata(bc::Broadcasted, grads::Tuple)
    r = zero_rdata(bc)
    r isa NoRData && return r
    return first(_cpu_bcast_fill_rdata(r, bc, grads, 1))
end
function _cpu_bcast_fill_rdata(r::RData, bc::Broadcasted, grads::Tuple, idx::Int)
    args = r.data.args
    new_args = ()
    for i in 1:length(bc.args)
        a = bc.args[i]
        ra = args[i]
        if a isa Broadcasted
            ra, idx = if ra isa NoRData
                (ra, idx + _cpu_bcast_nleaves(a))
            else
                _cpu_bcast_fill_rdata(ra, a, grads, idx)
            end
        else
            if a isa IEEEFloat
                ra = grads[idx]
            end
            idx += 1
        end
        new_args = (new_args..., ra)
    end
    data = r.data
    return RData((; style=data.style, f=data.f, args=new_args, axes=data.axes)), idx
end
function _cpu_bcast_nleaves(bc::Broadcasted)
    return sum(a -> a isa Broadcasted ? _cpu_bcast_nleaves(a) : 1, bc.args; init=0)
end

# The derived rule of the same broadcast, used when the result has a non-finite element.
# It mirrors `copy(::Broadcasted)` without calling `materialize` or `copy`, which are claimed
# by this rule.
function _cpu_bcast_generic(bc::Broadcasted)
    bci = Base.Broadcast.instantiate(bc)
    return copyto!(similar(bci, Base.Broadcast.combine_eltypes(bci.f, bci.args)), bci)
end

struct CPUBroadcastRule{R}
    fallback::R
end
CPUBroadcastRule() = CPUBroadcastRule(DynamicDerivedRule(false, Base.get_world_counter()))
_copy(r::CPUBroadcastRule) = CPUBroadcastRule(_copy(r.fallback))
function build_primitive_rrule(
    ::Type{S}
) where {S<:Tuple{typeof(Base.Broadcast.materialize),_CPUBroadcasted}}
    return _cpu_bcast_supported(fieldtype(S, 2)) ? CPUBroadcastRule() : rrule!!
end

function (rule::CPUBroadcastRule)(
    ::CoDual{typeof(Base.Broadcast.materialize)}, bc::CoDual{<:Broadcasted}
)
    bc_primal = primal(bc)
    flat = Base.Broadcast.flatten(bc_primal)
    pargs, fdatas = _cpu_bcast_leaves(bc_primal, tangent(bc))
    # One pass: evaluate on `NDual`s, writing values and each leaf's partials to plain arrays.
    bci = Base.Broadcast.instantiate(
        Broadcasted{typeof(flat.style)}(_CPUApplyWithDuals(flat.f), flat.args, flat.axes)
    )
    D = Base.Broadcast.combine_eltypes(bci.f, bci.args)
    y = similar(bci, Nfwd._nfwd_dual_primal_type(D))
    parts = ntuple(_ -> similar(y), Val(_cpu_ndual_width(D)))
    finite = _cpu_bcast_eval!(y, parts, Base.Broadcast.preprocess(y, bci))
    if !finite
        z, pb = rule.fallback(zero_fcodual(_cpu_bcast_generic), bc)
        return z, pb
    end
    dy = zero(y)
    function cpu_materialize_pb!!(::NoRData)
        grads = _cpu_bcast_accumulate!(pargs, fdatas, parts, dy, Val(1))
        return NoRData(), _cpu_bcast_rdata(bc_primal, grads)
    end
    return CoDual(y, dy), cpu_materialize_pb!!
end

function rrule!!(
    f::CoDual{typeof(Base.Broadcast.materialize)}, bc::CoDual{<:_CPUBroadcasted}
)
    return CPUBroadcastRule()(f, bc)
end

_cpu_ndual_width(::Type{<:Nfwd.NDual{T,N}}) where {T,N} = N

# Write values and partials; return whether every one is finite.
function _cpu_bcast_eval!(y::Array, parts::NTuple{N,Array}, bc) where {N}
    finite = true
    @inbounds @simd for I in eachindex(bc)
        o = bc[I]
        y[I] = o.value
        finite &= isfinite(o.value)
        for k in 1:N
            parts[k][I] = o.partials[k]
            finite &= isfinite(o.partials[k])
        end
    end
    return finite
end

# Accumulate each leaf's gradient. Array leaves are incremented in place; the gradients of
# scalar leaves are returned (one entry per leaf, `nothing` for non-scalar leaves).
_cpu_bcast_accumulate!(::Tuple{}, ::Tuple{}, parts, dy, ::Val) = ()
function _cpu_bcast_accumulate!(pargs::Tuple, fdatas::Tuple, parts, dy, ::Val{k}) where {k}
    pa, fd = first(pargs), first(fdatas)
    is_diff = _cpu_bcast_is_diff_leaf(typeof(pa))
    g = if is_diff && pa isa Array
        contrib = Base.broadcasted(*, dy, parts[k])
        if size(pa) == size(dy)
            fd .+= contrib
        else
            fd .+= _cpu_unbroadcast(Base.Broadcast.materialize(contrib), size(pa))
        end
        nothing
    elseif is_diff
        oftype(pa, sum(Base.broadcasted(*, dy, parts[k])))
    else
        nothing
    end
    rest = _cpu_bcast_accumulate!(
        Base.tail(pargs), Base.tail(fdatas), parts, dy, Val(is_diff ? k + 1 : k)
    )
    return (g, rest...)
end

@static if VERSION < v"1.11-"
    function build_primitive_rrule(::Type{S}) where {S<:Tuple{typeof(copy),_CPUBroadcasted}}
        return _cpu_bcast_supported(fieldtype(S, 2)) ? CPUBroadcastRule() : rrule!!
    end
    function (rule::CPUBroadcastRule)(::CoDual{typeof(copy)}, bc::CoDual{<:Broadcasted})
        return rule(zero_fcodual(Base.Broadcast.materialize), bc)
    end
    function rrule!!(::CoDual{typeof(copy)}, bc::CoDual{<:_CPUBroadcasted})
        return CPUBroadcastRule()(zero_fcodual(Base.Broadcast.materialize), bc)
    end
end
