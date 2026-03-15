struct ForwardModeRRule!!{FR}
    _frule!!::FR
end

function __verify_sig(rule::ForwardModeRRule!!, fx)
    # TODO: modify `fx`
    return __verify_sig(rule._frule!!, fx)
end

function (forward_mode_rrule!!::ForwardModeRRule!!)(
    f_codual::CoDual{F},
    args_codual::Vararg{<:CoDual{<:Union{IEEEFloat,<:Array{<:IEEEFloat}}},N},  # TODO: relax
) where {F,N}
    (; _frule!!) = forward_mode_rrule!!
    f = primal(f_codual)
    args = map(primal, args_codual)

    if tangent_type(F) != NoTangent
        throw(
            ArgumentError(
                "`Mooncake.@reverse_from_forward` does not support functions which close over data.",
            ),
        )
    end

    f_dual = Dual(f, NoTangent())
    args_dual_zero = map(zero_dual, args)
    y_dual = _frule!!(f_dual, args_dual_zero...)

    y = primal(y_dual)
    y_codual = zero_fcodual(y)

    function forward_mode_pullback(dy_rdata)
        @assert dy_rdata isa rdata_type(typeof(y)) # TODO: check this at compile time
        # compute args rdata
        dargs_rdata = ntuple(Val(N)) do i
            # ntuple ensures unrolling with statically inferrable integers (hopefully)
            if rdata_type(typeof(args[i])) == NoRData
                return NoRData()
            else
                @assert args[i] isa IEEEFloat # TODO: relax
                # create perturbation of scalar argument i
                args_dual_one_i = (
                    args_dual_zero[begin:(i - 1)]...,
                    Dual(args[i], one(args[i])),
                    args_dual_zero[(i + 1):end]...,
                )
                # compute partial derivative with respect to argument i
                y_dual_one_i = _frule!!(f_dual, args_dual_one_i...)
                partial_derivative_i = tangent(y_dual_one_i)
                # deduce one component of the pullback using a dot product
                rdata_i =
                    dot(fdata(partial_derivative_i), tangent(y_codual)) +
                    dot(rdata(partial_derivative_i), dy_rdata)
                return rdata_i
            end
        end

        # update args fdata
        ntuple(Val(N)) do i
            # ntuple ensures unrolling with statically inferrable integers (hopefully)
            if tangent(args_codual[i]) isa NoFData
                return nothing
            else
                @assert args[i] isa Array{<:IEEEFloat} # TODO: relax
                # iterate over input dimensions
                tangent(args_codual[i]) .+=
                    map(eachindex(args[i])) do j
                        # create perturbation of scalar dimension j of argument i
                        args_dual_one_ij = (
                            args_dual_zero[begin:(i - 1)]...,
                            Dual(args[i], basis(args[i], j)),
                            args_dual_zero[(i + 1):end]...,
                        )
                        # compute partial derivative with respect to dimension j of argument i
                        y_dual_one_ij = _frule!!(f_dual, args_dual_one_ij...)
                        partial_derivative_ij = tangent(y_dual_one_ij)
                        # deduce one component of the pullback using a dot product
                        rdata_ij =
                            dot(fdata(partial_derivative_ij), tangent(y_codual)) +
                            dot(rdata(partial_derivative_ij), dy_rdata)
                        return rdata_ij
                    end
                return nothing
            end
        end

        # return function and args rdata
        df_rdata = NoRData()
        return (df_rdata, dargs_rdata...)
    end

    return y_codual, forward_mode_pullback
end

function basis(a::Array{<:IEEEFloat}, i)
    # TODO: fix for immutable arrays
    # TODO: fix for GPU arrays
    b = zero(a)
    b[i] = oneunit(eltype(b))
    return b
end

"""
    @reverse_from_forward signature

Define a reverse rule for a given `signature` (function type + argument types) from an existing (primitive or derived) forward rule.

# Example

    @reverse_from_forward Tuple{typeof(f), Float64, Vector{Float64}}

This forces function calls `f(::Float64, ::Vector{Float64})` to be differentiated in forward mode inside any reverse-mode procedure.

!!! warning
    This macro is still experimental and has strict limitations:
        - The function must have all its arguments and its output of type `<:Base.IEEEFloat` or `Array{<:Base.IEEEFloat}`
        - The function must not close over any data (its own tangent type must be `NoTangent`)
        - The function must not mutate any of its arguments. That's because we need to run several forward passes for each argument to mimick a reverse pass, so the state of the arguments must remain unchanged.
"""
macro reverse_from_forward(sig)
    if !(Meta.isexpr(sig, :curly) && sig.args[1] == :Tuple)
        throw(
            ArgumentError(
                "The provided signature must be of the form `Tuple{typeof(f), ...}`."
            ),
        )
    end
    return quote
        @is_primitive DefaultCtx ReverseMode $(esc(sig))
        function Mooncake.build_primitive_rrule(concrete_sig::Type{<:$(esc(sig))})
            interp = get_interpreter(ForwardMode)
            _frule!! = build_frule(interp, concrete_sig)
            return ForwardModeRRule!!(_frule!!)
        end
    end
end
