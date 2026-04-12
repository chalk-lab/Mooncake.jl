module DispatchDoctorRules
# This lives here rather than in DispatchDoctor
# due to the circular dependency. However,
# the logic here is the same as other DispatchDoctor extensions
# for, e.g., Enzyme and ChainRulesCore.

import ..@zero_derivative
import ..@is_primitive
import ..DefaultCtx
import .._foreigncall_
import ..CoDual
import ..Dual
import ..NoTangent
import ..NoPullback
import ..zero_fcodual

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

# is_precompiling() is @inline, so Julia inlines its ccall body directly into
# DispatchDoctor's @stable wrappers. The @zero_derivative rules above apply only to the
# named-function call sites, not to the resulting inlined foreigncall node. Add explicit
# rules for the raw foreigncall so that nested AD through @stable-wrapped functions works.
#! format: off
@is_primitive DefaultCtx Tuple{
    typeof(_foreigncall_),
    Val{:jl_generating_output},
    Val{Cint},
    Tuple{},
    Val{0},
    Val{:ccall},
}
#! format: on
function rrule!!(
    f::CoDual{typeof(_foreigncall_)},
    name::CoDual{Val{:jl_generating_output}},
    rt::CoDual{Val{Cint}},
    at::CoDual{Tuple{}},
    nreq::CoDual{Val{0}},
    cc::CoDual{Val{:ccall}},
)
    pb = NoPullback(f, name, rt, at, nreq, cc)
    return zero_fcodual(ccall(:jl_generating_output, Cint, ())), pb
end

end
