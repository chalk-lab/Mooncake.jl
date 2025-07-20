using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using Distributions, DynamicPPL, Mooncake, StableRNGs, Test
using Mooncake.TestUtils: test_rule

@model function simple_model()
    return y ~ Normal()
end

@model function demo()
    # Assumptions
    σ2 ~ LogNormal() # tweaked from InverseGamma due to control flow issues.
    σ = sqrt(σ2 + 1e-3)
    μ ~ Normal(0.0, σ)

    # Observations
    x ~ Normal(μ, σ)
    return y ~ Normal(μ, σ)
end

@model broadcast_demo(x) = begin
    μ ~ truncated(Normal(1, 2), 0.1, 10)
    σ ~ truncated(Normal(1, 2), 0.1, 10)
    x .~ LogNormal(μ, σ)
end

# LDA example -- copied over from
# https://github.com/TuringLang/Turing.jl/issues/668#issuecomment-1153124051
function _make_data(D, K, V, N, α, η)
    β = Matrix{Float64}(undef, V, K)
    for k in 1:K
        β[:, k] .= rand(Dirichlet(η))
    end

    θ = Matrix{Float64}(undef, K, D)
    z = Vector{Int}(undef, D * N)
    w = Vector{Int}(undef, D * N)
    doc = Vector{Int}(undef, D * N)
    i = 0
    for d in 1:D
        θ[:, d] .= rand(Dirichlet(α))
        for n in 1:N
            i += 1
            z[i] = rand(Categorical(θ[:, d]))
            w[i] = rand(Categorical(β[:, z[i]]))
            doc[i] = d
        end
    end
    return (D=D, K=K, V=V, N=N, α=α, η=η, z=z, w=w, doc=doc, θ=θ, β=β)
end

data = let D = 2, K = 2, V = 160, N = 290
    _make_data(D, K, V, N, ones(K), ones(V))
end

# LDA with vectorization and manual log-density accumulation
@model function LatentDirichletAllocationVectorizedCollapsedMannual(D, K, V, α, η, w, doc)
    β ~ filldist(Dirichlet(η), K)
    θ ~ filldist(Dirichlet(α), D)

    log_product = log.(β * θ)
    DynamicPPL.@addlogprob! sum(log_product[CartesianIndex.(w, doc)])
    # Above is equivalent to below
    #product = β * θ
    #dist = [Categorical(product[:,i]) for i in 1:D]
    #w ~ arraydist([dist[doc[i]] for i in 1:length(doc)])
end

function make_large_model()
    num_tildes = 50
    expr = Base.remove_linenums!(:(function $(Symbol(:demo, num_tildes))() end))
    mainbody = last(expr.args)
    append!(mainbody.args, [:($(Symbol("x", j)) ~ Normal()) for j in 1:num_tildes])
    f = @eval $(DynamicPPL.model(:Main, LineNumberNode(1), expr, false))
    return invokelatest(f)
end

# Run this once in order to avoid world age problems in testset.
make_large_model()

function build_turing_problem(rng, model, example=nothing)
    ctx = DynamicPPL.DefaultContext()
    vi = DynamicPPL.SimpleVarInfo(example === nothing ? model : example)
    vi_linked = DynamicPPL.link(vi, model)
    ldp = DynamicPPL.LogDensityFunction(model, vi_linked, ctx)
    test_function = Base.Fix1(DynamicPPL.LogDensityProblems.logdensity, ldp)
    d = DynamicPPL.LogDensityProblems.dimension(ldp)
    return test_function, randn(rng, d)
end

@testset "turing" begin
    @testset "$(typeof(model))" for (interface_only, name, model, ex) in vcat(
        Any[
            (false, "simple_model", simple_model(), nothing),
            (false, "demo", demo(), nothing),
            (
                false,
                "broadcast_demo",
                broadcast_demo(rand(LogNormal(1.5, 0.5), 1_000)),
                nothing,
            ),
            (false, "large model", make_large_model(), nothing),
            # (
            #     false,
            #     "CollapsedLDA",
            #     LatentDirichletAllocationVectorizedCollapsedMannual(
            #         data.D, data.K, data.V, data.α, data.η, data.w, data.doc,
            #     ),
            # ), doesn't currently work with SimpleVarInfo
        ],
        Any[
            (false, "demo_$n", m, DynamicPPL.TestUtils.rand_prior_true(m)) for
            (n, m) in enumerate(DynamicPPL.TestUtils.DEMO_MODELS)
        ],
    )
        @info name
        f, x = build_turing_problem(StableRNG(123), model, ex)
        rng = StableRNG(123456)
        test_rule(rng, f, x; interface_only, is_primitive=false, unsafe_perturb=true)
    end
end
