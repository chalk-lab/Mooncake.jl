module MooncakeOrdinaryDiffEqCoreExt

using OrdinaryDiffEqCore
import Mooncake: DefaultCtx, @zero_derivative

# These functions handle bookkeeping, logging, and error checking rather than
# numerical computations that need gradient information. Marking them as
# zero-derivative prevents unnecessary differentiation through non-differentiable code.
# This mirrors the inactive_noinl rules in the Enzyme extension.

@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.increment_nf!),Vararg}
@zero_derivative DefaultCtx Tuple{
    typeof(OrdinaryDiffEqCore.fixed_t_for_floatingpoint_error!),Vararg
}
@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.increment_accept!),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.increment_reject!),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.check_error!),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.log_step!),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.final_progress),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(OrdinaryDiffEqCore.ode_determine_initdt),Vararg}

end
