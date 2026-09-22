# Run from the repository root: julia --project=test/nightly test/nightly/runtests.jl basic
# Do not use Pkg.test: Mooncake's normal test target installs JET and AllocCheck.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

ENV["MOONCAKE_TEST_PROFILE"] = "nightly"
@info "Nightly correctness profile: JET and AllocCheck analysis is skipped; runtime allocation checks remain enabled."
include(joinpath(@__DIR__, "..", "runtests.jl"))
