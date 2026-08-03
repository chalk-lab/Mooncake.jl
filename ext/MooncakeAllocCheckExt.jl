module MooncakeAllocCheckExt

using AllocCheck, Mooncake
import Mooncake.TestUtils: check_allocs_internal, Shim

@check_allocs check_allocs_internal(::Shim, f::F, x) where {F} = f(x)
@check_allocs check_allocs_internal(::Shim, f::F, x, y) where {F} = f(x, y)
@check_allocs check_allocs_internal(::Shim, f::F, x, y, z) where {F} = f(x, y, z)

# Known 1.13 caveat: on Julia 1.13+, BLAS ccalls go through
# `Base.Libc.Libdl.LazyLibrary`. AllocCheck can't resolve the runtime library
# operand at LLVM-rewrite time, renames the call to "jl_unknown_fptr", and raises a
# DynamicDispatch error — even though the resolved handle is cached and the actual
# call doesn't allocate. This is the same upstream LazyLibrary regression that causes
# the runtime `sum(abs2)` perf-test failure documented in `rules/performance_patches.jl`
# (https://github.com/JuliaLang/julia/pull/61735 backports a JLL revert; a proper fix
# is planned for 1.14). Until then, `@check_allocs` calls that hit a BLAS-backed path
# are expected to fail on 1.13.
end
