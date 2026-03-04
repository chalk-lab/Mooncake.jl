module MooncakeDistributionsExt

using Distributions, Mooncake, LinearAlgebra
using PrecompileTools: @setup_workload, @compile_workload

#! format: off

# Skip precompilation on GitHub Actions for Julia versions earlier than 1.11
@static if !haskey(ENV, "GITHUB_ACTIONS") || VERSION ≥ v"1.11-"

# Precompile the AD machinery for the most common `logpdf` patterns so that users
# who load Distributions.jl get a much smaller time-to-first-gradient when
# differentiating through distribution log-densities.
#
# The workload exercises `prepare_gradient_cache` → `value_and_gradient!!` for a
# representative subset of distribution families drawn from
# `test/integration_testing/distributions/distributions.jl`:
#   • a simple univariate distribution  (Normal)
#   • a simple multivariate distribution (MvNormal with diagonal covariance)

@setup_workload begin
    @compile_workload begin
        # --- reverse-mode, univariate logpdf ---
        d_uni = Normal(0.0, 1.0)
        cache_uni = Mooncake.prepare_gradient_cache(logpdf, d_uni, 0.1)
        Mooncake.value_and_gradient!!(cache_uni, logpdf, d_uni, 0.1)

        # --- reverse-mode, multivariate logpdf ---
        d_mv = MvNormal(Diagonal([1.0, 1.0]))
        x_mv = [0.1, -0.1]
        cache_mv = Mooncake.prepare_gradient_cache(logpdf, d_mv, x_mv)
        Mooncake.value_and_gradient!!(cache_mv, logpdf, d_mv, x_mv)
    end
end

end # @static if

#! format: on

end
