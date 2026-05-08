# Test-resource forward-mode rules whose signatures reference
# `Mooncake.Nfwd.NDual` — included after `nfwd/NfwdMooncake.jl` so the type
# is available. Each `@is_primitive` declaration stays in `test_resources.jl`
# (which loads earlier); only the frule body is here.

# Accept both the legacy `Dual{Float64}` shape (scaffold-path bare value) and
# the canonical `NDual{Float64, 1}` (width-1 lifted slot inner V) so the
# primitive survives canonicalisation through the generic Lifted-aware
# adapter when invoked via `DynamicPrimal` from polymorphic memory rules.
function Mooncake.frule!!(
    ::Dual{typeof(Mooncake.TestResources.edge_case_tester)},
    x::Union{Dual{Float64},Mooncake.Nfwd.NDual{Float64,1}},
)
    return if x isa Dual
        Dual(5 * primal(x), 5 * tangent(x))
    else
        Mooncake.Nfwd.NDual{Float64,1}(5 * x.value, (5 * x.partials[1],))
    end
end

@inline function Mooncake.frule!!(
    ::Mooncake.Lifted{typeof(Mooncake.TestResources.edge_case_tester),N},
    x::Mooncake.Lifted{Float64,N},
) where {N}
    inner = Mooncake._unlift(x)
    bare = Mooncake.frule!!(
        Dual(Mooncake.TestResources.edge_case_tester, NoTangent()), inner
    )
    return Mooncake.Lifted{Float64,N}(bare)
end
