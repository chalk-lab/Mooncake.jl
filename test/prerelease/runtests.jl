# Run from the repository root: julia --project=test/prerelease test/prerelease/runtests.jl basic
# Do not use Pkg.test: Mooncake's normal test target installs AllocCheck.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

ENV["MOONCAKE_TEST_PROFILE"] = "prerelease"
@info "Prerelease profile: AllocCheck analysis is skipped; JET analysis and runtime allocation checks remain enabled."
include(joinpath(@__DIR__, "..", "runtests.jl"))
