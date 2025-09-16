using Pkg
Pkg.activate(@__DIR__)

using JuliaFormatter, Test

@testset "quality" begin
    path = joinpath(@__DIR__, "..", "..")
    @test JuliaFormatter.format(path; verbose=false, overwrite=false)
end
