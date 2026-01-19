# Forward-mode primitive for _build_rule! on LazyDerivedRule.
# This avoids differentiating through get_interpreter which has a ccall to jl_get_world_counter.
# The tangent propagation happens through the fwds_oc MistyClosure call, not the rule building.
# Reverse-over-reverse is not supported; an rrule!! that throws is provided below.
@is_primitive MinimalCtx Tuple{typeof(_build_rule!),LazyDerivedRule,Tuple}

function frule!!(
    ::Dual{typeof(_build_rule!)},
    lazy_rule_dual::Dual{<:LazyDerivedRule{sig}},
    args_dual::Dual{<:Tuple},
) where {sig}
    lazy_rule = primal(lazy_rule_dual)
    lazy_tangent = tangent(lazy_rule_dual)
    primal_args = primal(args_dual)
    tangent_args = tangent(args_dual)

    # Build rrule if not built (primal operation, no differentiation needed)
    if !isdefined(lazy_rule, :rule)
        interp = get_interpreter(ReverseMode)
        lazy_rule.rule = build_rrule(interp, lazy_rule.mi; debug_mode=lazy_rule.debug_mode)
    end
    derived_rule = lazy_rule.rule

    # Initialize the tangent of the derived rule if needed
    rule_tangent_field = lazy_tangent.fields.rule
    if !isdefined(rule_tangent_field, :tangent)
        # Need to update the MutableTangent's fields with a new PossiblyUninitTangent
        new_rule_tangent = PossiblyUninitTangent(zero_tangent(derived_rule))
        lazy_tangent.fields = merge(lazy_tangent.fields, (; rule=new_rule_tangent))
        rule_tangent_field = new_rule_tangent
    end
    derived_tangent = rule_tangent_field.tangent

    # Forward-differentiate through the DerivedRule call.
    # DerivedRule(args...) internally calls fwds_oc(args...) and returns (CoDual, Pullback)
    fwds_oc = derived_rule.fwds_oc
    fwds_oc_tangent = derived_tangent.fields.fwds_oc

    # Handle varargs unflattening
    isva = _isva(derived_rule)
    nargs = derived_rule.nargs
    N = length(primal_args)
    uf_primal_args = __unflatten_codual_varargs(isva, primal_args, nargs)
    uf_tangent_args = __unflatten_tangent_varargs(isva, tangent_args, nargs)

    # Create dual args for frule!! call
    dual_args = map(Dual, uf_primal_args, uf_tangent_args)

    # Call frule!! on fwds_oc to get forward-differentiated result
    dual_fwds_oc = Dual(fwds_oc, fwds_oc_tangent)
    codual_result_dual = frule!!(dual_fwds_oc, dual_args...)

    # Create Pullback and its tangent
    pb_oc_ref = derived_rule.pb_oc_ref
    pb_primal = Pullback(sig, pb_oc_ref, isva, N)
    pb_tangent = Tangent((; pb_oc=derived_tangent.fields.pb_oc_ref))

    # Return Dual of (CoDual, Pullback)
    primal_result = (primal(codual_result_dual), pb_primal)
    tangent_result = (tangent(codual_result_dual), pb_tangent)
    return Dual(primal_result, tangent_result)
end

# Helper to unflatten tangent args similar to __unflatten_codual_varargs
function __unflatten_tangent_varargs(isva::Bool, tangent_args, ::Val{nargs}) where {nargs}
    isva || return tangent_args
    group_tangent = tangent_args[nargs:end]
    return (tangent_args[1:(nargs - 1)]..., group_tangent)
end

# Reverse-over-reverse is not supported. Throw an informative error.
function rrule!!(
    ::CoDual{typeof(_build_rule!)}, ::CoDual{<:LazyDerivedRule}, ::CoDual{<:Tuple}
)
    throw(
        ArgumentError(
            "Reverse-over-reverse differentiation is not supported. " *
            "Encountered attempt to differentiate _build_rule! in reverse mode.",
        ),
    )
end

# TODO: This is a workaround for forward-over-reverse. Primitives in reverse mode can get
# inlined when building the forward rule, exposing internal ccalls that lack an frule!!.
# For example, `dataids` is a reverse-mode primitive, but inlining it exposes
# `jl_genericmemory_owner`. The proper fix is to prevent primitive inlining during
# forward-over-reverse by forwarding `inlining_policy` through `BugPatchInterpreter` to
# `MooncakeInterpreter` during `optimise_ir!`, but this causes allocation regressions.
# See https://github.com/chalk-lab/Mooncake.jl/pull/878 for details.
# @static if VERSION >= v"1.11-"
#     function frule!!(
#         ::Dual{typeof(_foreigncall_)},
#         ::Dual{Val{:jl_genericmemory_owner}},
#         ::Dual{Val{Any}},
#         ::Dual{Tuple{Val{Any}}},
#         ::Dual{Val{0}},
#         ::Dual{Val{:ccall}},
#         a::Dual{<:Memory},
#     )
#         return zero_dual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
#     end
#     function rrule!!(
#         ::CoDual{typeof(_foreigncall_)},
#         ::CoDual{Val{:jl_genericmemory_owner}},
#         ::CoDual{Val{Any}},
#         ::CoDual{Tuple{Val{Any}}},
#         ::CoDual{Val{0}},
#         ::CoDual{Val{:ccall}},
#         a::CoDual{<:Memory},
#     )
#         y = zero_fcodual(ccall(:jl_genericmemory_owner, Any, (Any,), primal(a)))
#         return y, NoPullback(ntuple(_ -> NoRData(), 7))
#     end
# end

# Avoid differentiating through AD infrastructure during second-order differentiation.
@zero_derivative MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(prepare_gradient_cache),Vararg
}
@zero_derivative MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(prepare_derivative_cache),Vararg
}
@zero_derivative MinimalCtx Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(prepare_pullback_cache),Vararg
}
@zero_derivative MinimalCtx Tuple{typeof(zero_tangent),Any}

@static if VERSION < v"1.11-"
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_1d}},
        ::Dual{Val{Vector{P}}},
        ::Dual{Tuple{Val{Any},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Vector{P}}},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _n = primal(n)
            y = ccall(:jl_alloc_array_1d, Vector{$P}, (Any, Int), Vector{$P}, _n)
            dy = ccall(:jl_alloc_array_1d, Vector{$T}, (Any, Int), Vector{$T}, _n)
            return Dual(y, dy)
        end
    end
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_2d}},
        ::Dual{Val{Matrix{P}}},
        ::Dual{Tuple{Val{Any},Val{Int},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Matrix{P}}},
        m::Dual{Int},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _m, _n = primal(m), primal(n)
            y = ccall(:jl_alloc_array_2d, Matrix{$P}, (Any, Int, Int), Matrix{$P}, _m, _n)
            dy = ccall(:jl_alloc_array_2d, Matrix{$T}, (Any, Int, Int), Matrix{$T}, _m, _n)
            return Dual(y, dy)
        end
    end
    @generated function frule!!(
        ::Dual{typeof(_foreigncall_)},
        ::Dual{Val{:jl_alloc_array_3d}},
        ::Dual{Val{Array{P,3}}},
        ::Dual{Tuple{Val{Any},Val{Int},Val{Int},Val{Int}}},
        ::Dual{Val{0}},
        ::Dual{Val{:ccall}},
        ::Dual{Type{Array{P,3}}},
        l::Dual{Int},
        m::Dual{Int},
        n::Dual{Int},
        args::Vararg{Dual},
    ) where {P}
        T = tangent_type(P)
        return quote
            _l, _m, _n = primal(l), primal(m), primal(n)
            y = ccall(
                :jl_alloc_array_3d,
                Array{$P,3},
                (Any, Int, Int, Int),
                Array{$P,3},
                _l,
                _m,
                _n,
            )
            dy = ccall(
                :jl_alloc_array_3d,
                Array{$T,3},
                (Any, Int, Int, Int),
                Array{$T,3},
                _l,
                _m,
                _n,
            )
            return Dual(y, dy)
        end
    end
end
