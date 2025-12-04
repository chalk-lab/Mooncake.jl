module MooncakeDifferentiationInterfaceExt

using Mooncake:
    Mooncake, @is_primitive, MinimalCtx, ForwardMode, Dual, primal, tangent, NoTangent
import DifferentiationInterface as DI

# Mark shuffled_gradient as forward-mode primitive to avoid expensive type inference hang.
# This prevents build_frule from trying to derive rules for the complex gradient closure.
@is_primitive MinimalCtx ForwardMode Tuple{typeof(DI.shuffled_gradient),Vararg}
@is_primitive MinimalCtx ForwardMode Tuple{typeof(DI.shuffled_gradient!),Vararg}

# Helper to create Dual array from primal and tangent arrays
_make_dual_array(x::AbstractArray, dx::AbstractArray) = Dual.(x, dx)
_make_dual_array(x, dx) = Dual(x, dx)

# Helper to extract primal and tangent from Dual array
_extract_primals(arr::AbstractArray{<:Dual}) = primal.(arr)
_extract_primals(d::Dual) = primal(d)
_extract_tangents(arr::AbstractArray{<:Dual}) = tangent.(arr)
_extract_tangents(d::Dual) = tangent(d)

# frule for shuffled_gradient without prep
# shuffled_gradient(x, f, backend, rewrap, contexts...) -> gradient(f, backend, x, contexts...)
function Mooncake.frule!!(
    ::Dual{typeof(DI.shuffled_gradient)},
    x_dual::Dual,
    f_dual::Dual,
    backend_dual::Dual,
    rewrap_dual::Dual,
    context_duals::Vararg{Dual},
)
    # Extract primals and tangents
    x = primal(x_dual)
    dx = tangent(x_dual)
    f = primal(f_dual)
    backend = primal(backend_dual)
    rewrap = primal(rewrap_dual)
    contexts = map(d -> primal(d), context_duals)

    # Create Dual inputs: each element is Dual(x[i], dx[i])
    # This allows the Hvp to be computed via forward-over-reverse
    x_with_duals = _make_dual_array(x, dx)

    # Call gradient with Dual inputs
    # Since Dual{Float64,Float64} is self-tangent, reverse mode handles it correctly
    grad_duals = DI.shuffled_gradient(x_with_duals, f, backend, rewrap, contexts...)

    # Extract primal (gradient) and tangent (Hvp) from the Dual outputs
    grad_primal = _extract_primals(grad_duals)
    grad_tangent = _extract_tangents(grad_duals)

    return Dual(grad_primal, grad_tangent)
end

# frule for shuffled_gradient with prep
function Mooncake.frule!!(
    ::Dual{typeof(DI.shuffled_gradient)},
    x_dual::Dual,
    f_dual::Dual,
    prep_dual::Dual,
    backend_dual::Dual,
    rewrap_dual::Dual,
    context_duals::Vararg{Dual},
)
    x = primal(x_dual)
    dx = tangent(x_dual)
    f = primal(f_dual)
    prep = primal(prep_dual)
    backend = primal(backend_dual)
    rewrap = primal(rewrap_dual)
    contexts = map(d -> primal(d), context_duals)

    x_with_duals = _make_dual_array(x, dx)
    grad_duals = DI.shuffled_gradient(x_with_duals, f, prep, backend, rewrap, contexts...)

    grad_primal = _extract_primals(grad_duals)
    grad_tangent = _extract_tangents(grad_duals)

    return Dual(grad_primal, grad_tangent)
end

# frule for shuffled_gradient! (in-place version)
function Mooncake.frule!!(
    ::Dual{typeof(DI.shuffled_gradient!)},
    grad_dual::Dual,
    x_dual::Dual,
    f_dual::Dual,
    backend_dual::Dual,
    rewrap_dual::Dual,
    context_duals::Vararg{Dual},
)
    grad = primal(grad_dual)
    dgrad = tangent(grad_dual)  # Tangent storage for gradient (where Hvp goes)
    x = primal(x_dual)
    dx = tangent(x_dual)
    f = primal(f_dual)
    backend = primal(backend_dual)
    rewrap = primal(rewrap_dual)
    contexts = map(d -> primal(d), context_duals)

    x_with_duals = _make_dual_array(x, dx)
    # Allocate Dual buffer for in-place gradient
    grad_duals = _make_dual_array(grad, similar(grad))
    DI.shuffled_gradient!(grad_duals, x_with_duals, f, backend, rewrap, contexts...)

    # Copy primal (gradient) back to grad
    grad .= _extract_primals(grad_duals)
    # Copy tangent (Hvp) back to dgrad
    dgrad .= _extract_tangents(grad_duals)

    return Dual(nothing, NoTangent())
end

# frule for shuffled_gradient! with prep
function Mooncake.frule!!(
    ::Dual{typeof(DI.shuffled_gradient!)},
    grad_dual::Dual,
    x_dual::Dual,
    f_dual::Dual,
    prep_dual::Dual,
    backend_dual::Dual,
    rewrap_dual::Dual,
    context_duals::Vararg{Dual},
)
    grad = primal(grad_dual)
    dgrad = tangent(grad_dual)  # Tangent storage for gradient (where Hvp goes)
    x = primal(x_dual)
    dx = tangent(x_dual)
    f = primal(f_dual)
    prep = primal(prep_dual)
    backend = primal(backend_dual)
    rewrap = primal(rewrap_dual)
    contexts = map(d -> primal(d), context_duals)

    x_with_duals = _make_dual_array(x, dx)
    grad_duals = _make_dual_array(grad, similar(grad))
    DI.shuffled_gradient!(grad_duals, x_with_duals, f, prep, backend, rewrap, contexts...)

    # Copy primal (gradient) back to grad
    grad .= _extract_primals(grad_duals)
    # Copy tangent (Hvp) back to dgrad
    dgrad .= _extract_tangents(grad_duals)

    return Dual(nothing, NoTangent())
end

end
