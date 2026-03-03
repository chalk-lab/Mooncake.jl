using PrecompileTools: @setup_workload, @compile_workload

#! format: off

# Skip precompilation on GitHub Actions for Julia versions earlier than 1.11
@static if !haskey(ENV, "GITHUB_ACTIONS") || VERSION ≥ v"1.11-"

# Precompile the core AD machinery for the most common patterns so that the
# time-to-first-gradient is reduced for users.  The workload exercises the full
# `prepare_gradient_cache` → `value_and_gradient!!` and
# `prepare_derivative_cache` → `value_and_derivative!!` pipelines (which internally call
# `build_rrule`/`build_frule`, `generate_ir`, and all the IR-transformation infrastructure)
# for both a simple scalar and a simple vector function.  Because the IR-manipulation
# methods (`normalise!`, `BBCode`, `make_ad_stmts!`, …) work on `IRCode`/`BBCode` objects
# whose *Julia type* is the same regardless of which function is being differentiated, one
# call through the pipeline is enough to pre-warm the bulk of the compilation work.

@setup_workload begin
    # A non-primitive scalar function: exercises the derived-rule code path end-to-end.
    _precompile_f(x::Float64) = sin(x) + cos(x) * exp(x)
    # A non-primitive vector function: exercises array tangent/fdata handling as well.
    _precompile_g(x::Vector{Float64}) = sum(abs2, x)

    xs = [1.0, 2.0, 3.0]

    @compile_workload begin
        # --- reverse-mode, scalar Float64 ---
        cache = prepare_gradient_cache(_precompile_f, 1.0)
        value_and_gradient!!(cache, _precompile_f, 1.0)

        # --- reverse-mode, vector Float64 ---
        cache2 = prepare_gradient_cache(_precompile_g, xs)
        value_and_gradient!!(cache2, _precompile_g, xs)

        # --- forward-mode, scalar Float64 ---
        dcache = prepare_derivative_cache(_precompile_f, 1.0)
        value_and_derivative!!(dcache, Dual(_precompile_f, NoTangent()), Dual(1.0, 1.0))

        # --- forward-mode, vector Float64 ---
        dcache2 = prepare_derivative_cache(_precompile_g, xs)
        value_and_derivative!!(
            dcache2, Dual(_precompile_g, NoTangent()), Dual(xs, ones(3))
        )
    end
end

end # @static if

#! format: on
