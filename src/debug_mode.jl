"""
    DebugFRule(rule)

Construct a callable equivalent to `rule` but with additional type checking for forward-mode
AD. Checks:
- Each `Dual` argument has tangent type matching `tangent_type(typeof(primal))`
- The returned `Dual` has a correctly-typed tangent
- Deep structural validation (array sizes, field types, etc.)

Forward-mode counterpart to [`DebugRRule`](@ref).

*Note:* Debug mode significantly slows execution (10-100x) and should only be used for
diagnosing problems, not production runs.
```
"""
struct DebugFRule{Trule}
    rule::Trule
end

# Recursively copy the wrapped rule
_copy(x::P) where {P<:DebugFRule} = P(_copy(x.rule))

_rule_primal_sig(::Type{T}) where {T} = T.parameters[1]
function _rule_isva(::Type{T}) where {T}
    return T <: DerivedFRule ? T.parameters[3] : T.parameters[6]
end
function _rule_nargs(::Type{T}) where {T}
    return T <: DerivedFRule ? T.parameters[4] : T.parameters[7].parameters[1]
end

"""
    (rule::DebugFRule)(x::Vararg{Dual,N}) where {N}

Apply pre- and post-condition type checking. See [`DebugFRule`](@ref).
"""
@static if VERSION < v"1.11-"
    # On Julia 1.10, we encounter a segfault when an OpaqueClosure/MistyClosure is
    # called with a specialised signature whose declared return type disagrees with
    # what the IR actually returns (JuliaLang/julia#51016). Using @generated we
    # check tangent types at compile time; for a bad specialisation we return
    # :(error(...)) so the compiler never generates code for rule.rule(x...) with
    # mismatched types. For well-typed calls the normal body is emitted.
    #
    # NOTE: @generated alone (without the type check) does NOT prevent the segfault -
    # returning an unconditional quote generates the same code as a plain function.
    # The early-return on mismatch is the critical part.
    
    # See https://github.com/JuliaLang/julia/issues/61368
    @generated function (rule::DebugFRule{Trule})(x::Vararg{Dual,N}) where {Trule,N}
        # First, check tangent type consistency for all Dual inputs at compile time.
        # This prevents the compiler from generating code for rule.rule(x...) with
        # mismatched Dual types (e.g., Dual{Float64,Float32} instead of Dual{Float64,Float64}).
        for dt in x
            P = dt.parameters[1]  # primal type
            T = dt.parameters[2]  # tangent type
            T_expected = tangent_type(P)
            if T !== T_expected
                msg = "Error in inputs to rule with input types $(Tuple{x...})"
                return :(error($msg))
            end
        end

        # Check primal types match rule signature
        if Trule <: DerivedFRule && isconcretetype(Trule)
            sig = _rule_primal_sig(Trule)
            isva = _rule_isva(Trule)
            nargs_val = _rule_nargs(Trule)

            # Extract primal types
            primal_types = [dt.parameters[1] for dt in x]

            # Handle varargs unflattening
            if isva
                regular_types = primal_types[1:(nargs_val - 1)]
                vararg_types = primal_types[nargs_val:end]
                grouped_type = Tuple{vararg_types...}
                final_types = [regular_types..., grouped_type]
            else
                final_types = primal_types
            end

            Tx = Tuple{final_types...}
            if !(Tx <: sig)
                msg = "Error in inputs to rule with input types $(Tuple{x...})"
                return :(error($msg))
            end
        end

        return quote
            verify_dual_inputs(x)
            y = rule.rule(x...)
            verify_dual_output(x, y)
            return y
        end
    end
else
    @noinline function (rule::DebugFRule)(x::Vararg{Dual,N}) where {N}
        try
            verify_args(rule.rule, x)
        catch
            error("Error in inputs to rule with input types $(_typeof(x))")
        end
        verify_dual_inputs(x)
        y = rule.rule(x...)
        verify_dual_output(x, y)
        return y::Dual
    end
end

@noinline function verify_dual_inputs(@nospecialize(x::Tuple))
    try
        for _x in x
            _x isa Dual || error("Expected Dual, got $(typeof(_x))")
            verify_dual_value(_x)
        end
    catch e
        error("Error in inputs to rule with input types $(_typeof(x))")
    end
end

@noinline function verify_dual_output(@nospecialize(x), @nospecialize(y))
    try
        y isa Dual || error("frule!! must return a Dual, got $(typeof(y))")
        verify_dual_value(y)
    catch e
        error("Error in outputs of rule with input types $(_typeof(x))")
    end
end

@noinline function verify_dual_value(d::Dual{P,T}) where {P,T}
    # Fast path: type-level check using the Dual type parameters to enforce T == tangent_type(P)
    T_expected = tangent_type(P)
    if T !== T_expected
        throw(
            InvalidFDataException(
                "Dual tangent type mismatch: primal $P requires tangent type " *
                "$T_expected, but got $T",
            ),
        )
    end

    # Slow path: deep structural validation
    p, t = primal(d), tangent(d)
    # We validate fdata and rdata separately so these helpers stay in sync with reverse-mode checks.
    verify_fdata_value(p, fdata(t))
    verify_rdata_value(p, rdata(t))

    return nothing
end

"""
    DebugPullback(pb, y, x)

Construct a callable which is equivalent to `pb`, but which enforces type-based pre- and
post-conditions to `pb`. Let `dx = pb.pb(dy)`, for some rdata `dy`, then this function
- checks that `dy` has the correct rdata type for `y`, and
- checks that each element of `dx` has the correct rdata type for `x`.

Reverse pass counterpart to [`DebugRRule`](@ref)
"""
struct DebugPullback{Tpb,Ty,Tx}
    pb::Tpb
    y::Ty
    x::Tx
end

"""
    (pb::DebugPullback)(dy)

Apply type checking to enforce pre- and post-conditions on `pb.pb`. See the docstring for
`DebugPullback` for details.
"""
@inline function (pb::DebugPullback)(dy)
    verify_rvs_input(pb.y, dy)
    dx = pb.pb(dy)
    verify_rvs_output(pb.x, dx)
    return dx
end

@noinline verify_rvs_input(y, dy) = verify_rdata_value(y, dy)

@noinline function verify_rvs_output(x, dx)

    # Number of arguments and number of elements in pullback must match. Have to check this
    # because `zip` doesn't require equal lengths for arguments.
    l_pb = length(x)
    l_dx = length(dx)
    if l_pb != l_dx
        error("Number of args = $l_pb but number of rdata = $l_dx. They must to be equal.")
    end

    # Use for-loop to keep stack trace as simple as possible.
    for (_x, _dx) in zip(x, dx)
        verify_rdata_value(_x, _dx)
    end
end

"""
    DebugRRule(rule)

Construct a callable which is equivalent to `rule`, but inserts additional type checking.
In particular:
- check that the fdata in each argument is of the correct type for the primal
- check that the fdata in the `CoDual` returned from the rule is of the correct type for the
    primal.

This happens recursively.
For example, each element of a `Vector{Any}` is compared against each element of the
associated fdata to ensure that its type is correct, as this cannot be guaranteed from the
static type alone.

Some additional dynamic checks are also performed (e.g. that an fdata array of the same size
as its primal).

Let `rule` return `y, pb!!`, then `DebugRRule(rule)` returns `y, DebugPullback(pb!!)`.
`DebugPullback` inserts the same kind of checks as `DebugRRule`, but on the reverse-pass. See
the docstring for details.

*Note:* at any given point in time, the checks performed by this function constitute a
necessary but insufficient set of conditions to ensure correctness. If you find that an
error isn't being caught by these tests, but you believe it ought to be, please open an
issue or (better still) a PR.
"""
struct DebugRRule{Trule}
    rule::Trule
end

# Recursively copy the wrapped rule
_copy(x::P) where {P<:DebugRRule} = P(_copy(x.rule))

"""
    (rule::DebugRRule)(x::CoDual...)

Apply type checking to enforce pre- and post-conditions on `rule.rule`. See the docstring
for `DebugRRule` for details.
"""
@static if VERSION < v"1.11-"
    # Julia 1.10 can segfault while codegen'ing an OpaqueClosure call for an invalid CoDual
    # specialization before these runtime debug checks execute. Reject bad specializations at
    # compile time so the compiler never generates `rule.rule(x...)` for them.
    #
    # See https://github.com/JuliaLang/julia/issues/61368
    @generated function (rule::DebugRRule{Trule})(x::Vararg{CoDual,N}) where {Trule,N}
        for dt in x
            P = dt.parameters[1]
            T_expected = fcodual_type(P)
            if !(dt <: T_expected)
                msg = "error in inputs to rule with input types $(Tuple{x...})"
                return :(error($msg))
            end
        end

        if Trule <: DerivedRule && isconcretetype(Trule)
            sig = _rule_primal_sig(Trule)
            isva = _rule_isva(Trule)
            nargs_val = _rule_nargs(Trule)

            primal_types = [dt.parameters[1] for dt in x]
            if isva
                regular_types = primal_types[1:(nargs_val - 1)]
                vararg_types = primal_types[nargs_val:end]
                grouped_type = Tuple{vararg_types...}
                final_types = [regular_types..., grouped_type]
            else
                final_types = primal_types
            end

            Tx = Tuple{final_types...}
            if !(Tx <: sig)
                msg = "error in inputs to rule with input types $(Tuple{x...})"
                return :(error($msg))
            end
        end

        return quote
            verify_fwds_inputs(rule.rule, x)
            y, pb = rule.rule(x...)
            verify_fwds_output(x, y)
            return y::CoDual, DebugPullback(pb, primal(y), map(primal, x))
        end
    end
else
    @noinline function (rule::DebugRRule)(x::Vararg{CoDual,N}) where {N}
        verify_fwds_inputs(rule.rule, x)
        y, pb = rule.rule(x...)
        verify_fwds_output(x, y)
        return y::CoDual, DebugPullback(pb, primal(y), map(primal, x))
    end
end

# DerivedRule adds a method to this function.
verify_args(_, x) = nothing

@noinline function verify_fwds_inputs(rule, @nospecialize(x::Tuple))
    try
        # Check that the input types are correct. If this check is not present, the passing
        # in arguments of the wrong type can result in a segfault.
        verify_args(rule, x)

        # Use for-loop to keep the stack trace as simple as possible.
        for _x in x
            verify_fwds(_x)
        end
    catch e
        error("error in inputs to rule with input types $(_typeof(x))")
    end
end

@noinline function verify_fwds_output(@nospecialize(x), @nospecialize(y))
    try
        verify_fwds(y)
    catch e
        error("error in outputs of rule with input types $(_typeof(x))")
    end
end

@noinline verify_fwds(x::CoDual) = verify_fdata_value(primal(x), tangent(x))
