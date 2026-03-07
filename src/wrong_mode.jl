"""
    @is_forward signature

Declare that a given `signature` (function type + argument types) must be differentiated in forward mode, even when the user tries to call Mooncake in reverse mode.

# Example

    @is_forward Tuple{typeof(f), Float64}

This forces function calls `f(::Float64)` to be differentiated in forward mode.

!!! warning
    This macro is still experimental and has very strict limitations:
        - The signature is limited to function with one argument
        - The function must have scalar input and output of type `<:Base.IEEEFloat`
        - The function must not close over any data (its tangent type must be `NoTangent`)
"""
macro is_forward(sig)
    @assert Meta.isexpr(sig, :curly)
    @assert sig.args[1] == :Tuple
    # first two arguments are :Tuple and :(typeof(f))
    nargs = length(sig.args) - 2
    return quote
        @is_primitive DefaultCtx ReverseMode $(esc(sig))
        $(_is_forward(sig, Val(nargs)))
    end
end

function _is_forward(sig::Expr, ::Val{N}) where {N}
    throw(
        ArgumentError(
            "`Mooncake.@is_forward` does not yet support functions with $N arguments."
        ),
    )
end

function _is_forward(sig::Expr, ::Val{1})
    F = sig.args[2]
    X = sig.args[3]
    return quote
        function Mooncake.rrule!!(
            f_codual::CoDual{<:$(esc(F))}, x_codual::CoDual{<:$(esc(X))}
        )
            if tangent_type($(esc(F))) != NoTangent
                throw(
                    ArgumentError(
                        "`Mooncake.@is_forward` does not support functions which close over data.",
                    ),
                )
            end
            return _forward_mode_rrule!!(f_codual, x_codual)
        end
    end
end

function _forward_mode_rrule!!(f_codual::CoDual, x_codual::CoDual{<:IEEEFloat})
    f = primal(f_codual)
    x = primal(x_codual)

    f_dual = Dual(f, NoTangent())
    x_dual = Dual(x, one(x))
    y_dual = frule!!(f_dual, x_dual)

    y = primal(y_dual)

    function forward_mode_scalar_pullback(dy::IEEEFloat)
        f_rdata = NoRData()
        y_rdata = tangent(y_dual) * dy
        return f_rdata, y_rdata
    end

    if y isa IEEEFloat
        y_fdata = fdata(zero_tangent(y))
        return CoDual(y, y_fdata), forward_mode_scalar_pullback
    else
        throw(
            ArgumentError(
                "`Mooncake.@is_forward` does not support output type $(typeof(y))."
            ),
        )
    end
end
