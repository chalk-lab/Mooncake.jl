using StableRNGs: StableRNG
using Base: IEEEFloat
using Mooncake: ForwardMode, _typeof, _add_to_primal, _scale, _diff

function finite_diff_jvp(func, x, dx, unsafe_perturb::Bool=false)
    ε_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    results = map(ε_list) do ε
        x_plus = _add_to_primal(x, _scale(ε, dx), unsafe_perturb)
        x_minus = _add_to_primal(x, _scale(-ε, dx), unsafe_perturb)
        y_plus = func(x_plus...)
        y_minus = func(x_minus...)
        return _scale(1 / (2ε), _diff(y_plus, y_minus))
    end

    return results
end

function low_level_gradient(rule, f, args...)
    return Base.tail(Mooncake.value_and_gradient!!(rule, f, args...)[2])
end

_as_tuple(x::Tuple) = x
_as_tuple(x) = (x,)

function _isapprox_nested(a, b; atol=1e-8, rtol=1e-6)
    if a isa Number && b isa Number
        return isapprox(a, b; atol, rtol)
    elseif a isa AbstractArray && b isa AbstractArray
        return isapprox(a, b; atol, rtol)
    elseif a isa Tuple && b isa Tuple
        return length(a) == length(b) &&
               all(_isapprox_nested(ai, bi; atol, rtol) for (ai, bi) in zip(a, b))
    else
        return a == b
    end
end

@testset "forward-over-reverse Hessian AD" begin
    @testset "$(_typeof((f, x...)))" for (n, (interface_only, perf_flag, bnds, f, x...)) in
                                         collect(
        enumerate(TestResources.generate_test_functions())
    )
        # Skip interface-only tests as they don't have implementations
        interface_only && continue

        @info "$n: $(_typeof((f, x...)))"

        interp = Mooncake.get_interpreter(ForwardMode)
        args = map(TestUtils._deepcopy, x)
        rule = Mooncake.build_rrule(f, x...)
        sig = Tuple{
            typeof(low_level_gradient),typeof(rule),_typeof(f),map(_typeof, args)...
        }
        grad_rule = Mooncake.build_frule(interp, sig)
        rng = StableRNG(0xF0 + n)
        dirs = map(arg -> Mooncake.randn_tangent(rng, arg), args)
        if any(dir -> dir isa Mooncake.NoTangent, dirs)
            @test_broken true
            continue
        end

        dual_inputs = (
            Mooncake.Dual(low_level_gradient, Mooncake.zero_tangent(low_level_gradient)),
            Mooncake.Dual(rule, Mooncake.zero_tangent(rule)),
            Mooncake.Dual(f, Mooncake.zero_tangent(f)),
            map((arg, dir) -> Mooncake.Dual(arg, dir), args, dirs)...,
        )
        dual_result = grad_rule(dual_inputs...)
        pushforward = _as_tuple(Mooncake.tangent(dual_result))

        # Use our own finite difference JVP implementation
        grad_func = x -> _as_tuple(low_level_gradient(rule, f, x...))
        fd_results_all = finite_diff_jvp(grad_func, args, dirs)

        # Check if any epsilon value gives a close match (following test_utils.jl pattern)
        # Convert each result to tuple form for comparison
        fd_results_tuples = map(res -> _as_tuple(res), fd_results_all)

        # Check which epsilon values give close results
        isapprox_results = map(fd_results_tuples) do fd_ref
            return all(_isapprox_nested(pf, fd)  # uses default atol=1e-8, rtol=1e-6
                for (pf, fd) in zip(pushforward, fd_ref))
        end

        @test length(pushforward) == length(first(fd_results_tuples))
        # At least one epsilon value should give a close result
        if !any(isapprox_results)
            # If none match, display values for debugging (like test_utils.jl does)
            println("No epsilon gave close result. AD vs FD for each epsilon:")
            for (i, fd_ref) in enumerate(fd_results_tuples)
                println("  ε[$(i)]: AD=$pushforward, FD=$fd_ref")
            end
        end
        @test any(isapprox_results)
    end
end
