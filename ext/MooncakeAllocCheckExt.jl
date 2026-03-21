module MooncakeAllocCheckExt

using AllocCheck, Mooncake
import Mooncake.TestUtils: check_allocs_internal, Shim

@check_allocs check_allocs_internal(::Shim, f::F, x) where {F} = f(x)
@check_allocs check_allocs_internal(::Shim, f::F, x, y) where {F} = f(x, y)
@check_allocs check_allocs_internal(::Shim, f::F, x, y, z) where {F} = f(x, y, z)

function __init__()
    # AllocCheck's allowlist includes "get_pgcstack" but not "get_pgcstack_static",
    # which is the arm64-specific variant used on Apple Silicon. Patch fn_may_allocate
    # to recognise it as non-allocating. See: AllocCheck.jl/src/classify.jl
    #
    # MWE reproducing the false positive on arm64:
    #   using Mooncake, AllocCheck
    #   T = Tuple{Vector{Float64}, Vector{Float64}}
    #   AllocCheck.compile_callable(Mooncake.fdata_type, Tuple{Type{T}}; ignore_throw=true).analysis
    #   # => ["Allocating runtime call to \"jl_get_pgcstack_static\" in unknown location"]
    orig = AllocCheck.fn_may_allocate
    orig_world = Base.get_world_counter()
    @eval AllocCheck function fn_may_allocate(name::AbstractString; ignore_throw::Bool)
        name == "get_pgcstack_static" && return false
        return Base.invoke_in_world($orig_world, $orig, name; ignore_throw)
    end
end

end
