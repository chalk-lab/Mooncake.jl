module DispatchDoctorRules
# This lives here rather than in DispatchDoctor
# due to the circular dependency. However,
# the logic here is the same as other DispatchDoctor extensions
# for, e.g., Enzyme and ChainRulesCore.

import ..@zero_derivative
import ..DefaultCtx

import DispatchDoctor._RuntimeChecks: is_precompiling, checking_enabled
import DispatchDoctor._Stabilization: _show_warning, _construct_pairs
import DispatchDoctor._Utils:
    specializing_typeof,
    map_specializing_typeof,
    _promote_op,
    type_instability,
    type_instability_limit_unions

@zero_derivative DefaultCtx Tuple{typeof(_show_warning),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(_construct_pairs),Vararg}

@zero_derivative DefaultCtx Tuple{typeof(specializing_typeof),Any}
@zero_derivative DefaultCtx Tuple{typeof(map_specializing_typeof),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(_promote_op),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(type_instability),Vararg}
@zero_derivative DefaultCtx Tuple{typeof(type_instability_limit_unions),Vararg}

@zero_derivative DefaultCtx Tuple{typeof(is_precompiling)}
@zero_derivative DefaultCtx Tuple{typeof(checking_enabled)}

end
