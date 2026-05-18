module MooncakeAllocCheckExt

using AllocCheck, Mooncake
import Mooncake.TestUtils: check_allocs_internal, Shim

@check_allocs check_allocs_internal(::Shim, f::F, x) where {F} = f(x)
@check_allocs check_allocs_internal(::Shim, f::F, x, y) where {F} = f(x, y)
@check_allocs check_allocs_internal(::Shim, f::F, x, y, z) where {F} = f(x, y, z)

# Extend AllocCheck's "known non-allocating" list with runtime pgcstack helpers that
# upstream doesn't yet whitelist:
#
# - "get_pgcstack_static": arm64 variant used on Apple Silicon.
# - "get_pgcstack_fallback": runtime helper emitted on Julia 1.13+.
#
# Neither allocates in steady state. TODO: remove once upstream AllocCheck catches up
# (see https://github.com/JuliaLang/AllocCheck.jl/pull/100).
#
# Known 1.13 caveat we deliberately do NOT patch: on Julia 1.13+, BLAS ccalls go
# through `Base.Libc.Libdl.LazyLibrary`. AllocCheck can't resolve the runtime library
# operand at LLVM-rewrite time, renames the call to "jl_unknown_fptr", and raises a
# DynamicDispatch error — even though the resolved handle is cached and the actual
# call doesn't allocate. This is the same upstream LazyLibrary regression that causes
# the runtime `sum(abs2)` perf-test failure documented in `rules/performance_patches.jl`
# (https://github.com/JuliaLang/julia/pull/61735 backports a JLL revert; a proper fix
# is planned for 1.14). Until then, `@check_allocs` calls that hit a BLAS-backed path
# are expected to fail on 1.13.
function __init__()
    orig = AllocCheck.fn_may_allocate
    orig_world = Base.get_world_counter()
    @eval AllocCheck function fn_may_allocate(name::AbstractString; ignore_throw::Bool)
        name in ("get_pgcstack_static", "get_pgcstack_fallback") && return false
        return Base.invoke_in_world($orig_world, $orig, name; ignore_throw)
    end
end

end
