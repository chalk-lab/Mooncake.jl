using StableRNGs: StableRNG
using Base: IEEEFloat
using FiniteDifferences
using Mooncake: ForwardMode, _typeof

const HESSIAN_CASE_IDS = Set([3, 4, 13, 34])
const HESSIAN_RTOL = 1e-6
const HESSIAN_ATOL = 1e-8
const HESSIAN_FDM = FiniteDifferences.central_fdm(5, 1)

function low_level_gradient(rule, f, args...)
    return Base.tail(Mooncake.value_and_gradient!!(rule, f, args...)[2])
end

function _scalar_output(f, args...)
    copied = map(TestUtils._deepcopy, args)
    return f(copied...) isa IEEEFloat
end

function _hessian_supported_arg(x)
    x isa Number && return true
    x isa AbstractArray && return eltype(typeof(x)) <: Number
    x isa Tuple && return all(_hessian_supported_arg, x)
    x isa NamedTuple && return all(_hessian_supported_arg, x)
    return false
end

function _select_hessian_cases()
    selected = Vector{Tuple{Int,Tuple}}()
    for (n, case) in enumerate(TestResources.generate_test_functions())
        n in HESSIAN_CASE_IDS || continue
        interface_only, _, _, fx... = case
        interface_only && continue
        f = fx[1]
        args = fx[2:end]
        _scalar_output(f, args...) || continue
        any(!_hessian_supported_arg(arg) for arg in args) && continue
        push!(selected, (n, case))
    end
    return selected
end

_as_tuple(x::Tuple) = x
_as_tuple(x) = (x,)

function _isapprox_nested(a, b; atol, rtol)
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
    cases = _select_hessian_cases()
    @info "forward-over-reverse Hessian cases" total = length(cases)
    @test !isempty(cases)
    for (n, (interface_only, perf_flag, _, fx...)) in cases
        interp = Mooncake.get_interpreter(ForwardMode)
        f = fx[1]
        args = map(TestUtils._deepcopy, fx[2:end])
        rule = Mooncake.build_rrule(fx...)
        sig = Tuple{
            typeof(low_level_gradient),typeof(rule),_typeof(f),map(_typeof, args)...
        }
        grad_rule = Mooncake.build_frule(interp, sig)
        rng = StableRNG(0xF0 + n)
        dirs = map(arg -> Mooncake.randn_tangent(rng, arg), args)
        if any(dir -> dir isa Mooncake.NoTangent, dirs)
            @testset "$n - $(_typeof((fx)))" begin
                @test_broken true
            end
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

        fd_ref = FiniteDifferences.jvp(
            HESSIAN_FDM,
            x -> _as_tuple(low_level_gradient(rule, f, x...)),
            (Tuple(args), Tuple(dirs)),
        )
        reference = _as_tuple(fd_ref)

        @testset "$n - $(_typeof((fx)))" begin
            @test length(pushforward) == length(reference)
            for (pf, fd) in zip(pushforward, reference)
                @test _isapprox_nested(pf, fd; atol=HESSIAN_ATOL, rtol=HESSIAN_RTOL)
            end
        end
    end
end
