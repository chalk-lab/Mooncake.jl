# Formatting is outside the standard test env due to some dependency clashes with other
# test dependencies. Putting it in its own environment ensures that there is no possibility
# of such dependency clashes causing development issues in the future.
#
# Locating this amongst the integration tests is purely a matter of convenience.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaFormatter

@testset "quality" begin
    path = joinpath(@__DIR__, "..", "..")
    @test JuliaFormatter.format(path; verbose=false, overwrite=false)
end
