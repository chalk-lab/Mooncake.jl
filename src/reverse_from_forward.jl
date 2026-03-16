# Build a tuple of pre-allocated buffers for ForwardModeRRule!!, one entry per
# argument type in `Sig` (excluding the leading function type).
# Entry i is `nothing` for scalar (IEEEFloat) arguments, and a `Ref` holding an
# empty placeholder array for array arguments.  Using a `@generated` function
# gives a precisely typed NTuple (e.g. Tuple{Nothing, Ref{Vector{Float64}}}),
# so that indexing with a compile-time constant inside the caller's ntuple(Val(N))
# is fully type-stable with no union splitting.
@generated function _make_fwd_bufs(::Type{Sig}) where {Sig<:Tuple}
    N = length(Sig.parameters) - 1  # exclude the function type
    exprs = map(1:N) do i
        T = Sig.parameters[i + 1]
        if T <: IEEEFloat
            :(nothing)
        elseif T <: Array{<:IEEEFloat}
            dims = ntuple(_ -> 0, ndims(T))
            :(Ref{$T}($T(undef, $(dims...))))  # empty placeholder; sized on first use, resized if shape changes
        else
            throw(
                ArgumentError(
                    "unsupported argument type $T in `_make_fwd_bufs`: expected IEEEFloat or Array{<:IEEEFloat}",
                ),
            )
        end
    end
    return :(($(exprs...),))
end

# `_zero_bufs[i]` and `_basis_bufs[i]` are typed tuples produced by
# _make_fwd_bufs: `nothing` for scalar args, `Ref{Array{T,D}}` for array
# args.  The Refs hold empty arrays initially and are lazily sized on the
# first call, then reused on every subsequent call to avoid per-call heap
# allocation.  If the array size changes between calls the buffer is resized.
struct ForwardModeRRule!!{FR,ZB<:Tuple,BB<:Tuple}
    _frule!!::FR
    _zero_bufs::ZB
    _basis_bufs::BB
end

function __verify_sig(rule::ForwardModeRRule!!, fx)
    # TODO: modify `fx`
    return __verify_sig(rule._frule!!, fx)
end

function (fmr::ForwardModeRRule!!)(
    f_codual::CoDual{F},
    args_codual::Vararg{<:CoDual{<:Union{IEEEFloat,<:Array{<:IEEEFloat}}},N},  # TODO: relax
) where {F,N}
    (; _frule!!) = fmr
    f = primal(f_codual)
    args = map(primal, args_codual)

    if tangent_type(F) != NoTangent
        throw(
            ArgumentError(
                "`Mooncake.@reverse_from_forward` does not support functions which close over data.",
            ),
        )
    end

    # Build args_dual_zero, reusing pre-allocated buffers for array args.
    # For each i, fmr._zero_bufs[i] is statically typed (Nothing or Ref{<:Array}),
    # so the branch below is resolved at compile time inside ntuple(Val(N)).
    args_dual_zero = ntuple(Val(N)) do i
        buf_ref = fmr._zero_bufs[i]
        if buf_ref === nothing
            zero_dual(args[i])  # scalar: no heap allocation
        else
            if size(buf_ref[]) != size(args[i])
                buf_ref[] = zero(args[i])  # first call or size change: (re)allocate
            else
                fill!(buf_ref[], zero(eltype(buf_ref[])))  # same size: reuse
            end
            Dual(args[i], buf_ref[])
        end
    end

    # Zero-tangent forward pass: run the primal to get y.
    # Tangents are all zero so this pass does not contribute to any derivative.
    f_dual = Dual(f, NoTangent())
    y_dual = _frule!!(f_dual, args_dual_zero...)

    y = primal(y_dual)
    y_codual = zero_fcodual(y)

    function forward_mode_pullback(dy_rdata)
        @assert dy_rdata isa rdata_type(typeof(y)) # TODO: check this at compile time
        # The full output seed ȳ is split: its fdata was accumulated into tangent(y_codual)
        # by increment!! in __value_and_pullback!!, and its rdata is passed as dy_rdata.
        # Reconstruct the full output tangent for use in dot products below.
        # Note: `tangent(y_codual)` (not `fdata(y_codual)`) is correct here — `fdata` is not
        # defined on CoDuals. For a forward-mode CoDual, `tangent(c) == c.dx` stores the fdata
        # directly, so `tangent(tangent(y_codual), dy_rdata)` is `tangent(fdata, rdata)`.
        dy_full = tangent(tangent(y_codual), dy_rdata)
        # compute args rdata
        dargs_rdata = ntuple(Val(N)) do i
            if rdata_type(typeof(args[i])) == NoRData
                return NoRData()
            else
                @assert args[i] isa IEEEFloat # TODO: relax
                # create perturbation of scalar argument i
                args_dual_one_i = ntuple(Val(N)) do k
                    k == i ? Dual(args[i], one(args[i])) : args_dual_zero[k]
                end
                # Forward pass with unit perturbation on arg i gives ∂y/∂xᵢ.
                y_dual_one_i = _frule!!(f_dual, args_dual_one_i...)
                partial_derivative_i = tangent(y_dual_one_i)
                # VJP: rdata_i = <∂y/∂xᵢ, ȳ>. Use the full output seed (fdata + rdata).
                rdata_i = _dot(partial_derivative_i, dy_full)
                return rdata_i
            end
        end

        # update args fdata
        ntuple(Val(N)) do i
            if tangent(args_codual[i]) isa NoFData
                return nothing
            else
                @assert args[i] isa Array{<:IEEEFloat} # TODO: relax
                b_ref = fmr._basis_bufs[i]  # statically typed: Ref{Array{T,D}}
                if size(b_ref[]) != size(args[i])
                    b_ref[] = zero(args[i])  # first call or size change: (re)allocate
                end
                # Resize the gradient accumulator if the input size changed.
                t = tangent(args_codual[i])
                if length(t) != length(args[i])
                    resize!(t, length(args[i]))
                    fill!(t, zero(eltype(t)))
                end
                b = b_ref[]
                for j in eachindex(args[i])
                    b[j] = oneunit(eltype(b))
                    args_dual_one_ij = ntuple(Val(N)) do k
                        k == i ? Dual(args[i], b) : args_dual_zero[k]
                    end
                    # Forward pass with unit perturbation on element j of arg i gives column j of ∂y/∂xᵢ.
                    y_dual_one_ij = _frule!!(f_dual, args_dual_one_ij...)
                    partial_derivative_ij = tangent(y_dual_one_ij)
                    # VJP: accumulate <∂y/∂xᵢ[j], ȳ> into fdata of arg i.
                    t[j] += _dot(partial_derivative_ij, dy_full)
                    b[j] = zero(eltype(b))
                end
                return nothing
            end
        end

        # The function has no tangent (NoTangent), so its rdata is NoRData.
        return (NoRData(), dargs_rdata...)
    end

    return y_codual, forward_mode_pullback
end

"""
    @reverse_from_forward signature

Define a reverse rule for a given `signature` (function type + argument types) from an existing (primitive or derived) forward rule.

# Example

    @reverse_from_forward Tuple{typeof(f), Float64, Vector{Float64}}

Registers a reverse-mode `rrule!!` for `f(::Float64, ::Vector{Float64})` that implements
the pullback via repeated forward passes (one per input dimension).

!!! warning
    This macro is still experimental and has strict limitations:
        - The function must have all its arguments and its output of type `<:Base.IEEEFloat` or `Array{<:Base.IEEEFloat}`
        - The function must not close over any data (its own tangent type must be `NoTangent`)
        - The function must not mutate any of its arguments, because the pullback runs several forward passes and relies on the arguments remaining unchanged between them
        - The rule is **not thread-safe**: it holds shared mutable buffers that are reused across calls. Do not call the same cached rule concurrently from multiple threads.
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
            zero_bufs = _make_fwd_bufs(concrete_sig)
            basis_bufs = _make_fwd_bufs(concrete_sig)
            return ForwardModeRRule!!(_frule!!, zero_bufs, basis_bufs)
        end
    end
end
