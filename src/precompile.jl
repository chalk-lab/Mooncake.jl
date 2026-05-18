using PrecompileTools: @setup_workload, @compile_workload

#! format: off

# Skip precompilation on GitHub Actions for Julia versions earlier than 1.11.
# On Julia LTS (1.10), precompilation can cause certain Mooncake allocation tests to fail.
@static if !haskey(ENV, "GITHUB_ACTIONS") || VERSION ≥ v"1.11-"

# Precompile the core AD machinery for the most common patterns so that the
# time-to-first-gradient is reduced for users.  The workload exercises the full
# `prepare_gradient_cache` → `value_and_gradient!!` and
# `prepare_derivative_cache` → `value_and_derivative!!` pipelines (which internally call
# `build_rrule`/`build_frule`, `generate_ir`, and all the IR-transformation infrastructure)
# for both a simple scalar and a simple vector function.  Because the IR-manipulation
# methods (`normalise!`, `make_ad_stmts!`, CFG lowering, …) work on a stable `IRCode`/CFG
# pipeline whose *Julia type* is the same regardless of which function is being
# differentiated, one call through the pipeline is enough to pre-warm the bulk of the
# compilation work.

@setup_workload begin
    # A non-primitive scalar function: exercises the derived-rule code path end-to-end.
    _precompile_f(x) = sin(x) + cos(x) * exp(x)
    # A non-primitive vector function: exercises array tangent/fdata handling as well.
    _precompile_g(x) = sum(abs2, x)
    # A non-primitive Complex scalar function: exercises complex primitive dispatch.
    _precompile_h(z) = abs2(sin(z) + cos(z) * exp(z))
    _precompile_h32(z) = abs2(sin(z) + cos(z) * exp(z))

    xs = [1.0, 2.0, 3.0]
    z = 1.0 + 2.0im
    z32 = ComplexF32(1.0f0, 2.0f0)

    @compile_workload begin
        # Reverse-mode: scalar Float64
        cache = prepare_gradient_cache(_precompile_f, 1.0)
        value_and_gradient!!(cache, _precompile_f, 1.0)

        # Reverse-mode: vector Float64
        cache2 = prepare_gradient_cache(_precompile_g, xs)
        value_and_gradient!!(cache2, _precompile_g, xs)

        # Reverse-mode: scalar ComplexF64
        cache3 = prepare_gradient_cache(_precompile_h, z)
        value_and_gradient!!(cache3, _precompile_h, z)

        # Reverse-mode: scalar ComplexF32
        cache4 = prepare_gradient_cache(_precompile_h32, z32)
        value_and_gradient!!(cache4, _precompile_h32, z32)

        # Forward-mode: scalar Float64
        dcache = prepare_derivative_cache(_precompile_f, 1.0)
        value_and_derivative!!(dcache, Dual(_precompile_f, NoTangent()), Dual(1.0, 1.0))

        # Forward-mode: vector Float64
        dcache2 = prepare_derivative_cache(_precompile_g, xs)
        value_and_derivative!!(
            dcache2, Dual(_precompile_g, NoTangent()), Dual(xs, ones(3))
        )

        # Forward-mode: scalar ComplexF64
        dcache3 = prepare_derivative_cache(_precompile_h, z)
        value_and_derivative!!(dcache3, Dual(_precompile_h, NoTangent()), Dual(z, one(z)))

        # Forward-mode: scalar ComplexF32
        dcache4 = prepare_derivative_cache(_precompile_h32, z32)
        value_and_derivative!!(
            dcache4, Dual(_precompile_h32, NoTangent()), Dual(z32, one(z32))
        )
    end
end

end # @static if

#! format: on
