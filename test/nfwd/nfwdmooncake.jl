# Run this file directly with:
# julia --project=bench -e 'using Test, Mooncake; include("test/nfwd/nfwdmooncake.jl")'

# Test support
# Shared helpers and primitive registrations used by the structured sections below.

using LinearAlgebra, Random
using Random: Xoshiro

struct NfwdRRuleTestFunc{N,F}
    f::F
end

(f::NfwdRRuleTestFunc)(x...) = f.f(x...)

function nfwd_test_rule(
    rng::AbstractRNG, chunk_size::Int, f, args...; perf_flag=:none, kwargs...
)
    nf = NfwdRRuleTestFunc{chunk_size,typeof(f)}(f)
    return Mooncake.TestUtils.test_rule(
        rng,
        nf,
        args...;
        is_primitive=false,
        perf_flag,
        mode=Mooncake.ReverseMode,
        kwargs...,
    )
end

function Mooncake.build_rrule(
    f::NfwdRRuleTestFunc{N}, x...; debug_mode=false, silence_debug_messages=true
) where {N}
    return Mooncake.NfwdMooncake.build_rrule(
        f.f, x...; chunk_size=N, debug_mode, silence_debug_messages
    )
end

function _nfwd_primitive_rrule(sig::Type{<:Tuple}, chunk_size::Int)
    Mooncake.NfwdMooncake.build_rrule(sig; chunk_size)
end

function steady_state_allocations_rrule(rule, fx::Tuple)
    for _ in 1:3
        rule(fx...)
    end
    GC.gc()
    return @allocated rule(fx...)
end

# Bug fix note: measuring `@allocated value_and_gradient!!(...)` through a generic helper
# counted wrapper/capture overhead on Julia 1.10 and produced a false 192-byte regression.
@noinline _allocated_value_and_gradient(cache::C, f::F, x::X) where {C,F,X} = @allocated Mooncake.value_and_gradient!!(
    cache, f, x
)

function steady_state_allocations_value_and_gradient(cache, f, x)
    for _ in 1:3
        Mooncake.value_and_gradient!!(cache, f, x)
    end
    GC.gc()
    return _allocated_value_and_gradient(cache, f, x)
end

function nfwd_safe_log(x)
    try
        return log(x)
    catch
        return -Inf
    end
end

function nfwd_safe_log_vec(x)
    try
        return [log(x), x^2]
    catch
        return [-Inf, x^2]
    end
end

nfwd_outer(x) = nfwd_safe_log(x) + x^2
vector_sum_sq(x) = sum(abs2, x)

struct NfwdDerivedMultiplier
    a::Float64
end

(m::NfwdDerivedMultiplier)(x) = m.a * x

Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{
    typeof(nfwd_safe_log),Float64
}

function Mooncake.build_primitive_rrule(::Type{<:Tuple{typeof(nfwd_safe_log),Float64}})
    return _nfwd_primitive_rrule(Tuple{typeof(nfwd_safe_log),Float64}, 1)
end

Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{
    typeof(nfwd_safe_log_vec),Float64
}

function Mooncake.build_primitive_rrule(::Type{<:Tuple{typeof(nfwd_safe_log_vec),Float64}})
    return _nfwd_primitive_rrule(Tuple{typeof(nfwd_safe_log_vec),Float64}, 1)
end

Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{
    typeof(vector_sum_sq),Vector{Float64}
}

function Mooncake.build_primitive_rrule(
    ::Type{<:Tuple{typeof(vector_sum_sq),Vector{Float64}}}
)
    return _nfwd_primitive_rrule(Tuple{typeof(vector_sum_sq),Vector{Float64}}, 4)
end

@testset "nfwd API" begin
    f = (x, y) -> x * y + cos(x)
    x, y = 5.0, 4.0
    dx, dy = 3.0, 2.0
    z = f(x, y)
    dz = dx * y + x * dy + dx * (-sin(x))
    scalar_cases = (
        (
            name="chunk_size=1",
            chunk_size=1,
            dual_inputs=(Mooncake.Dual(x, dx), Mooncake.Dual(y, dy)),
            expected_tangent=dz,
        ),
        (
            name="chunk_size=2 scalar lanes",
            chunk_size=2,
            dual_inputs=(Mooncake.Dual(x, (dx, 0.0)), Mooncake.Dual(y, (0.0, dy))),
            expected_tangent=(dx * y + dx * (-sin(x)), x * dy),
        ),
    )

    @testset "public rule and cache entrypoints" begin
        @testset "scalar rule construction and caches" begin
            @testset "$case.name" for case in scalar_cases
                rule = Mooncake.NfwdMooncake.build_frule(
                    f, x, y; chunk_size=case.chunk_size
                )
                out = rule(Mooncake.zero_dual(f), case.dual_inputs...)
                @test out isa Mooncake.Dual
                @test Mooncake.primal(out) == z
                @test Mooncake.tangent(out) == case.expected_tangent

                rrule = Mooncake.NfwdMooncake.build_rrule(
                    f, x, y; chunk_size=case.chunk_size
                )
                ȳ, pb!! = rrule(
                    Mooncake.zero_fcodual(f),
                    Mooncake.zero_fcodual(x),
                    Mooncake.zero_fcodual(y),
                )
                @test Mooncake.primal(ȳ) == z
                @test pb!!(1.0) == (Mooncake.NoRData(), y - sin(x), x)

                cache = Mooncake.NfwdMooncake.prepare_cache(
                    f, x, y; chunk_size=case.chunk_size
                )
                z_and_dz_dual = Mooncake.value_and_derivative!!(
                    cache, Mooncake.zero_dual(f), case.dual_inputs...
                )
                @test z_and_dz_dual isa Mooncake.Dual
                @test Mooncake.primal(z_and_dz_dual) == z
                @test Mooncake.tangent(z_and_dz_dual) == case.expected_tangent

                case.chunk_size == 1 && continue
                z_and_grad = Mooncake.value_and_gradient!!(cache, f, x, y)
                @test first(z_and_grad) == z
                @test last(z_and_grad) == (Mooncake.NoTangent(), y - sin(x), x)
            end
        end

        @testset "scalar cache edge cases" begin
            square(x) = x^2
            constant_scalar(x) = 1.0

            @testset "single-input scalar cache with chunk_size>1" begin
                x_single = 1.5
                cache = Mooncake.NfwdMooncake.prepare_cache(square, x_single; chunk_size=2)
                value, grad = Mooncake.value_and_gradient!!(cache, square, x_single)
                @test value == square(x_single)
                @test grad == (Mooncake.NoTangent(), 2 * x_single)
            end

            @testset "constant outputs synthesize zero tangents" begin
                x_const = 2.0
                cache_grad = Mooncake.NfwdMooncake.prepare_cache(
                    constant_scalar, x_const; chunk_size=1
                )
                value, grad = Mooncake.value_and_gradient!!(
                    cache_grad, constant_scalar, x_const
                )
                @test value == 1.0
                @test grad == (Mooncake.NoTangent(), 0.0)

                cache_deriv = Mooncake.NfwdMooncake.prepare_cache(
                    constant_scalar, x_const; chunk_size=2
                )
                out = Mooncake.value_and_derivative!!(
                    cache_deriv,
                    Mooncake.zero_dual(constant_scalar),
                    Mooncake.Dual(x_const, (1.0, 0.0)),
                )
                @test Mooncake.primal(out) == 1.0
                @test Mooncake.tangent(out) == (0.0, 0.0)
            end

            @testset "automatic chunk_size selection" begin
                cache = Mooncake.NfwdMooncake.prepare_cache(f, x, y)
                value, grad = Mooncake.value_and_gradient!!(cache, f, x, y)
                @test value == z
                @test grad == (Mooncake.NoTangent(), y - sin(x), x)

                # NfwdMooncake.build_frule / NfwdMooncake.build_rrule omitting chunk_size
                auto_frule = Mooncake.NfwdMooncake.build_frule(f, x, y)
                @test auto_frule isa Mooncake.NfwdMooncake.Rule  # compiles without chunk_size
                auto_rrule = Mooncake.NfwdMooncake.build_rrule(f, x, y)
                @test auto_rrule isa Mooncake.NfwdMooncake.RRule
                value2, grad2 = Mooncake.value_and_gradient!!(auto_rrule, f, x, y)
                @test value2 == z
                @test grad2 == (Mooncake.NoTangent(), y - sin(x), x)
            end

            @testset "_nfwd_sig_dof and _nfwd_sig_default_chunk_size" begin
                # scalar-only sigs: DOF is known exactly at type level
                @test Mooncake.Nfwd._nfwd_sig_dof(Tuple{typeof(nfwd_safe_log),Float64}) == 1
                @test Mooncake.Nfwd._nfwd_sig_dof(Tuple{typeof(f),Float64,Float64}) == 2
                @test Mooncake.Nfwd._nfwd_sig_dof(Tuple{typeof(f),ComplexF64,Float64}) == 3

                # array input: DOF is unknown (nothing)
                @test isnothing(
                    Mooncake.Nfwd._nfwd_sig_dof(Tuple{typeof(f),Vector{Float64}})
                )

                # default chunk: min(DOF, preferred) for scalar; preferred for array
                pref = Mooncake.Nfwd._NFWD_PREFERRED_CHUNK_SIZE
                @test Mooncake.Nfwd._nfwd_sig_default_chunk_size(
                    Tuple{typeof(nfwd_safe_log),Float64}
                ) == 1
                @test Mooncake.Nfwd._nfwd_sig_default_chunk_size(
                    Tuple{typeof(f),Float64,Float64}
                ) == 2
                @test Mooncake.Nfwd._nfwd_sig_default_chunk_size(
                    Tuple{typeof(f),Vector{Float64}}
                ) == pref
            end
        end

        @testset "array and complex forward evaluation" begin
            @testset "chunk_size=2 array lanes" begin
                g(x) = sin.(x)
                x_vec = [1.0, 2.0]
                dx_vec = reshape([1.0, 0.0, 0.0, 1.0], 2, 2)
                rrule = Mooncake.NfwdMooncake.build_rrule(g, x_vec; chunk_size=2)
                value, pullback = Mooncake.value_and_pullback!!(rrule, [3.0, 4.0], g, x_vec)
                @test value == sin.(x_vec)
                @test pullback ==
                    (Mooncake.NoTangent(), [3.0 * cos(x_vec[1]), 4.0 * cos(x_vec[2])])

                cache = Mooncake.NfwdMooncake.prepare_cache(g, x_vec; chunk_size=2)
                y_and_dy = Mooncake.value_and_derivative!!(
                    cache, Mooncake.zero_dual(g), Mooncake.Dual(x_vec, dx_vec)
                )
                @test Mooncake.primal(y_and_dy) == sin.(x_vec)
                @test Mooncake.tangent(y_and_dy) ≈ [cos(x_vec[1]) 0.0; 0.0 cos(x_vec[2])]

                @test_throws Mooncake.ValueAndGradientReturnTypeError Mooncake.value_and_gradient!!(
                    cache, g, x_vec
                )
            end

            @testset "complex inputs" begin
                fc(z) = real(z * z + cos(z))
                zc = ComplexF64(1.2, -0.3)
                dzc = ComplexF64(0.5, -0.25)
                expected_dzc = real((2zc - sin(zc)) * dzc)
                rule = Mooncake.NfwdMooncake.build_frule(fc, zc; chunk_size=1)
                out = rule(Mooncake.zero_dual(fc), Mooncake.Dual(zc, dzc))
                @test Mooncake.primal(out) == fc(zc)
                @test Mooncake.tangent(out) ≈ expected_dzc

                rrule = Mooncake.NfwdMooncake.build_rrule(fc, zc; chunk_size=2)
                ȳ, pb!! = rrule(Mooncake.zero_fcodual(fc), Mooncake.zero_fcodual(zc))
                @test Mooncake.primal(ȳ) == fc(zc)
                @test pb!!(1.0) == (Mooncake.NoRData(), conj(2zc - sin(zc)))

                cache_scalar = Mooncake.NfwdMooncake.prepare_cache(fc, zc; chunk_size=2)
                z_and_grad = Mooncake.value_and_gradient!!(cache_scalar, fc, zc)
                @test first(z_and_grad) == fc(zc)
                @test last(z_and_grad) == (Mooncake.NoTangent(), conj(2zc - sin(zc)))

                gc(z) = sum(abs2, z)
                z_vec = ComplexF64[1.0 + 2.0im, -3.0 + 0.5im]
                dz_vec = reshape(
                    ComplexF64[1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 1.0im], 2, 2
                )
                cache = Mooncake.NfwdMooncake.prepare_cache(gc, z_vec; chunk_size=2)
                out_vec = Mooncake.value_and_derivative!!(
                    cache, Mooncake.zero_dual(gc), Mooncake.Dual(z_vec, dz_vec)
                )
                @test Mooncake.primal(out_vec) == gc(z_vec)
                @test Mooncake.tangent(out_vec) == (2.0, 1.0)

                hc(z) = z .* z
                ȳ_vec = ComplexF64[2.0 - 1.0im, -0.5 + 0.25im]
                rrule_vec = Mooncake.NfwdMooncake.build_rrule(hc, z_vec; chunk_size=2)
                value_vec, pullback_vec = Mooncake.value_and_pullback!!(
                    rrule_vec, ȳ_vec, hc, z_vec
                )
                @test value_vec == hc(z_vec)
                @test pullback_vec == (Mooncake.NoTangent(), 2 .* conj.(z_vec) .* ȳ_vec)
            end

            @testset "multi-argument and matrix array inputs" begin
                h(x, y) = sum(x .* y)
                x_vec = [1.0, 2.0]
                y_vec = [3.0, 4.0]
                cache = Mooncake.NfwdMooncake.prepare_cache(h, x_vec, y_vec; chunk_size=2)
                dx = reshape([1.0, 0.0, 0.0, 0.0], 2, 2)
                dy = reshape([0.0, 0.0, 1.0, 0.0], 2, 2)
                out = Mooncake.value_and_derivative!!(
                    cache,
                    Mooncake.zero_dual(h),
                    Mooncake.Dual(x_vec, dx),
                    Mooncake.Dual(y_vec, dy),
                )
                @test Mooncake.primal(out) == h(x_vec, y_vec)
                @test Mooncake.tangent(out) == (3.0, 1.0)

                hm(X) = sin.(X)
                X = reshape([1.0, 2.0, 3.0, 4.0], 2, 2)
                dX = zeros(2, 2, 2)
                dX[1, 1, 1] = 1.0
                dX[2, 2, 2] = 1.0
                cache_matrix = Mooncake.NfwdMooncake.prepare_cache(hm, X; chunk_size=2)
                y_and_dy = Mooncake.value_and_derivative!!(
                    cache_matrix, Mooncake.zero_dual(hm), Mooncake.Dual(X, dX)
                )
                @test Mooncake.primal(y_and_dy) == sin.(X)
                expected = zeros(2, 2, 2)
                expected[1, 1, 1] = cos(X[1, 1])
                expected[2, 2, 2] = cos(X[2, 2])
                @test Mooncake.tangent(y_and_dy) ≈ expected
            end
        end

        @testset "tuple outputs" begin
            @testset "sincos chunk_size=1 frule and rrule" begin
                x_sc = 1.2
                frule = Mooncake.NfwdMooncake.build_frule(sincos, x_sc; chunk_size=1)
                out = frule(Mooncake.zero_dual(sincos), Mooncake.Dual(x_sc, 1.0))
                @test Mooncake.primal(out) == sincos(x_sc)
                s_t, c_t = Mooncake.tangent(out)
                @test s_t ≈ cos(x_sc)
                @test c_t ≈ -sin(x_sc)

                rrule = Mooncake.NfwdMooncake.build_rrule(sincos, x_sc; chunk_size=1)
                ȳ_sc, pb!! = rrule(
                    Mooncake.zero_fcodual(sincos), Mooncake.zero_fcodual(x_sc)
                )
                @test Mooncake.primal(ȳ_sc) == sincos(x_sc)
                ȳ_sin, ȳ_cos = 3.0, 2.0
                @test pb!!((ȳ_sin, ȳ_cos)) ==
                    (Mooncake.NoRData(), ȳ_sin * cos(x_sc) + ȳ_cos * (-sin(x_sc)))
            end

            @testset "sincos chunk_size=2 via prepare_cache" begin
                x_sc = 0.7
                cache = Mooncake.NfwdMooncake.prepare_cache(sincos, x_sc; chunk_size=2)
                value, pullback = Mooncake.value_and_pullback!!(
                    cache.rrule, (3.0, 2.0), sincos, x_sc
                )
                @test value == sincos(x_sc)
                @test pullback[2] ≈ 3.0 * cos(x_sc) + 2.0 * (-sin(x_sc))
            end

            @testset "modf derivative: frac=1, int=0" begin
                x_mf = 3.7
                frule = Mooncake.NfwdMooncake.build_frule(modf, x_mf; chunk_size=1)
                out = frule(Mooncake.zero_dual(modf), Mooncake.Dual(x_mf, 1.0))
                @test Mooncake.primal(out) == modf(x_mf)
                frac_t, int_t = Mooncake.tangent(out)
                @test frac_t ≈ 1.0
                @test int_t ≈ 0.0
            end

            @testset "tuple output rejects value_and_gradient!!" begin
                cache = Mooncake.NfwdMooncake.prepare_cache(sincos, 1.0; chunk_size=1)
                @test_throws Mooncake.ValueAndGradientReturnTypeError Mooncake.value_and_gradient!!(
                    cache, sincos, 1.0
                )
            end
        end

        @testset "test_rule integration" begin
            nfwd_test_rule(Xoshiro(123), 2, f, x, y)
        end
    end

    @testset "validation and layout helpers" begin
        @testset "stateful callables are rejected" begin
            a = 2.0
            f_stateful = x -> a * x
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_frule(
                f_stateful, 5.0; chunk_size=1
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_rrule(
                f_stateful, 5.0; chunk_size=1
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.prepare_cache(
                f_stateful, 5.0; chunk_size=1
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_rrule(
                Tuple{NfwdDerivedMultiplier,Float64}; chunk_size=1
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_rrule(
                Tuple{Nothing,Float64}; chunk_size=1
            )
        end

        @testset "unsupported config is rejected" begin
            @test_throws ArgumentError Mooncake.NfwdMooncake.prepare_cache(
                f, x, y; chunk_size=1, config=Mooncake.Config(; friendly_tangents=true)
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.prepare_cache(
                f, x, y; chunk_size=1, config=Mooncake.Config(; debug_mode=true)
            )
            @test_throws Mooncake.NfwdMooncake.UnsupportedInputError Mooncake.NfwdMooncake.prepare_cache(
                f, view([x, y], 1:2); chunk_size=1
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_frule(
                f, x, y; chunk_size=1, debug_mode=true
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_rrule(
                f, x, y; chunk_size=1, debug_mode=true
            )
        end

        @testset "invalid chunk_size is rejected" begin
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_frule(
                f, x, y; chunk_size=0
            )
            @test_throws ArgumentError Mooncake.NfwdMooncake.prepare_cache(
                f, x, y; chunk_size=-1
            )
        end

        @testset "function tangent rejection" begin
            rule = Mooncake.NfwdMooncake.build_frule(f, x, y; chunk_size=1)
            @test_throws ArgumentError rule(
                Mooncake.Dual(f, 1.0), Mooncake.Dual(x, dx), Mooncake.Dual(y, dy)
            )

            rrule = Mooncake.NfwdMooncake.build_rrule(f, x, y; chunk_size=1)
            @test_throws ArgumentError rrule(
                Mooncake.CoDual(f, 1.0), Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(y)
            )
        end

        @testset "array tangent validation" begin
            g(x) = sin.(x)
            x_vec = [1.0, 2.0]
            cache = Mooncake.NfwdMooncake.prepare_cache(g, x_vec; chunk_size=2)
            @test_throws ArgumentError Mooncake.value_and_derivative!!(
                cache, Mooncake.zero_dual(g), Mooncake.Dual(x_vec, [1.0, 2.0, 3.0])
            )
        end

        @testset "args_to_zero validation" begin
            cache = Mooncake.NfwdMooncake.prepare_cache(f, x, y; chunk_size=2)
            bad_args_to_zero_cases = ((true, true), (true, false, true))
            for args_to_zero in bad_args_to_zero_cases
                @test_throws ArgumentError Mooncake.value_and_gradient!!(
                    cache, f, x, y; args_to_zero
                )
            end
        end

        @testset "cache size mismatch is rejected" begin
            g(x) = sum(x)
            cache = Mooncake.NfwdMooncake.prepare_cache(g, [1.0, 2.0]; chunk_size=1)
            @test_throws ArgumentError Mooncake.value_and_gradient!!(
                cache, g, [1.0, 2.0, 3.0]
            )
        end

        @testset "cache specs and chunk fallback helpers" begin
            @test Mooncake.NfwdMooncake._nfwd_sig(sin) == (type=typeof(sin), size=())
            @test Mooncake.NfwdMooncake._nfwd_sig([1.0, 2.0]) ==
                (type=Vector{Float64}, size=(2,))
            @test Mooncake.NfwdMooncake._nfwd_sig(1.0) == (type=Float64, size=())

            @test Mooncake._chunk_should_fallback_to_lane_loop(
                Mooncake.NfwdMooncake.UnsupportedInputError("unsupported input")
            )
            @test Mooncake._chunk_should_fallback_to_lane_loop(
                Mooncake.NfwdMooncake.UnsupportedOutputError("unsupported output")
            )
            @test Mooncake._chunk_should_fallback_to_lane_loop(
                Mooncake.Nfwd.NDualUnsupportedError(:floor)
            )
            @test !Mooncake._chunk_should_fallback_to_lane_loop(ArgumentError("other"))
        end
    end

    @testset "primitive reverse-mode integration" begin
        @testset "signature-based build returns nfwd reverse rule" begin
            interp = Mooncake.get_interpreter(Mooncake.ReverseMode)
            rule = Mooncake.build_rrule(interp, Tuple{typeof(nfwd_safe_log),Float64})
            @test rule isa Mooncake.NfwdMooncake.RRule
        end

        primitive_scalar_cases = (
            (
                name="gradient cache works for try/catch scalar function",
                x=2.0,
                expected_value=log(2.0),
                expected_grad=inv(2.0),
            ),
            (
                name="gradient cache handles try/catch fallback branch",
                x=-1.0,
                expected_value=(-Inf),
                expected_grad=0.0,
            ),
        )

        @testset "$case.name" for case in primitive_scalar_cases
            cache = Mooncake.prepare_gradient_cache(nfwd_safe_log, case.x)
            value, grad = Mooncake.value_and_gradient!!(cache, nfwd_safe_log, case.x)
            @test value == case.expected_value || value ≈ case.expected_value
            @test grad == (Mooncake.NoTangent(), case.expected_grad)
        end

        @testset "pullback cache uses the nfwd primitive backend" begin
            x = 2.0
            ȳ = [3.0, 4.0]
            cache = Mooncake.prepare_pullback_cache(nfwd_safe_log_vec, x)
            value, pullback = Mooncake.value_and_pullback!!(cache, ȳ, nfwd_safe_log_vec, x)
            @test value ≈ [log(x), x^2]
            @test pullback == (Mooncake.NoTangent(), ȳ[1] / x + ȳ[2] * 2x)
        end

        @testset "nested derived calls reuse the backend" begin
            x = 2.0
            cache = Mooncake.prepare_gradient_cache(nfwd_outer, x)
            value, grad = Mooncake.value_and_gradient!!(cache, nfwd_outer, x)
            @test value ≈ log(x) + x^2
            @test grad == (Mooncake.NoTangent(), inv(x) + 2x)
        end

        @testset "cached vector gradients use the nfwd backend" begin
            x = randn(16)
            cache = Mooncake.prepare_gradient_cache(vector_sum_sq, x)
            value, grad = Mooncake.value_and_gradient!!(cache, vector_sum_sq, x)
            @test value ≈ sum(abs2, x)
            @test grad == (Mooncake.NoTangent(), 2 .* x)
        end

        @testset "allocation regressions" begin
            allocation_cases = (
                (
                    name="scalar primitive path stays allocation-free",
                    f=nfwd_safe_log,
                    x=2.0,
                ),
                (
                    name="vector primitive path stays allocation-free",
                    f=vector_sum_sq,
                    x=randn(16),
                ),
            )

            @testset "$case.name" for case in allocation_cases
                rule = Mooncake.build_rrule(case.f, case.x)
                fx = (Mooncake.zero_fcodual(case.f), Mooncake.zero_fcodual(case.x))
                cache = Mooncake.prepare_gradient_cache(case.f, case.x)
                @test steady_state_allocations_rrule(rule, fx) == 0
                @test steady_state_allocations_value_and_gradient(cache, case.f, case.x) ==
                    0
            end

            @testset "multi-argument array pullback pins allocations" begin
                # The generic (multi-arg) pullback path allocates for temporaries; pin it so
                # regressions are caught.  This does not assert zero: it asserts the value
                # does not grow unexpectedly between runs (steady-state == 3rd-run value).
                h(x, y) = sum(x .* y)
                x_a = randn(8)
                y_a = randn(8)
                rrule = Mooncake.NfwdMooncake.build_rrule(h, x_a, y_a; chunk_size=4)
                fx = (
                    Mooncake.zero_fcodual(h),
                    Mooncake.zero_fcodual(x_a),
                    Mooncake.zero_fcodual(y_a),
                )
                allocs = steady_state_allocations_rrule(rrule, fx)
                # Re-measure to confirm steady-state (two identical runs).
                @test steady_state_allocations_rrule(rrule, fx) == allocs
            end
        end

        @testset "unsupported callable-state and function tangents fail explicitly" begin
            @test_throws ArgumentError Mooncake.NfwdMooncake.prepare_cache(
                NfwdDerivedMultiplier(2.0), 5.0
            )
            x = randn(16)
            rule = Mooncake.build_rrule(vector_sum_sq, x)
            bad_f = Mooncake.CoDual(vector_sum_sq, 1.0)
            bad_x = Mooncake.CoDual(x, zero(x))
            @test_throws ArgumentError Mooncake.__value_and_gradient!!(rule, bad_f, bad_x)
        end

        @testset "debug mode warns and runs with outer debug checks" begin
            x = 2.0
            cache = Mooncake.prepare_gradient_cache(
                nfwd_safe_log,
                x;
                config=Mooncake.Config(; debug_mode=true, silence_debug_messages=true),
            )
            @test_logs (:warn, r"ignore `debug_mode=true`") begin
                value, grad = Mooncake.value_and_gradient!!(cache, nfwd_safe_log, x)
                @test value ≈ log(x)
                @test grad == (Mooncake.NoTangent(), inv(x))
            end
        end
    end
end

# ── Integration test helpers ──────────────────────────────────────────────────
# Module-level constants and wrappers so functions are singleton callables.

const _nfw_A5 = randn(Xoshiro(99), 5, 5)
const _nfw_M5 = _nfw_A5 * _nfw_A5' + 5LinearAlgebra.I
const _nfw_A35 = randn(Xoshiro(88), 3, 5)   # for mul! wrapper
_nfw_sum_matvec(x) = sum(_nfw_A5 * x)
_nfw_sum_linsolve(x) = sum(_nfw_M5 \ x)
_nfw_sum_matmat(x) = sum(reshape(x, 5, 5) * reshape(x, 5, 5))
_nfw_sum_view(x) = sum(view(x, 1:5))
_nfw_sum_getindex(x) = x[1] + x[3] + x[5] + x[7] + x[9]

# Wrappers for previously-excluded cases: kwarg lambdas, vector outputs, mutating.
# All use sum/dot to reduce to a scalar so value_and_gradient!! applies.
# Mutating wrappers use similar(x, ...) so the buffer element type follows x
# (Float64 in primal runs, NDual{Float64,N} during the forward sweep).
_nfw_lse_dims1(x) = sum(logsumexp(x; dims=1))       # kwarg → named function
_nfw_lse_dims2(x) = sum(logsumexp(x; dims=2))
_nfw_softmax_dot(x) = dot(softmax(x), x)               # non-trivial scalar from softmax
_nfw_sum_adjoint_v(x) = sum(adjoint(x))                  # Adjoint output → wrapped
_nfw_sum_adjoint_m(x) = sum(adjoint(reshape(x, 4, 3)))
_nfw_sum_map_sin(x) = sum(map(sin, x))                 # map → vector output → sum
_nfw_getindex_4(x) = x[4]                             # hardcoded index avoids Int input
_nfw_view_23_1(x) = sum(view(x, 2:3, 1))            # hardcoded UnitRange/Int
_nfw_setindex_sum(x) = (y=copy(x); setindex!(y, 2.0, 3); sum(y))  # mutating via copy
_nfw_mul_sum(x) = (C=similar(x, 3); mul!(C, _nfw_A35, x); sum(C))  # similar(x) = right eltype
_nfw_push_sum(x) = (y=copy(x[1:(end - 1)]); push!(y, x[end]); sum(y))
function _nfw_lse_bang_sum(x)
    (r=similar(x, size(x, 1)); logsumexp!(r, reshape(x, size(x, 1), :)); sum(r))
end

# ── logexpfunctions integration ───────────────────────────────────────────────
# All singleton scalar and vector/matrix functions from
# test/integration_testing/logexpfunctions/logexpfunctions.jl.
#
# Excluded (API limitations, not missing NDual rules):
#   logsumexp(x; dims=...) — kwarg lambda, not singleton callable
#   logsumexp!             — mutating output argument
#   softmax                — vector output (value_and_gradient!! rejects)
@testset "logexpfunctions integration" begin
    using LogExpFunctions
    rng = Xoshiro(1)

    scalar1_cases = Any[
        (xlogx, 1.1),
        (xexpx, -0.5),
        (logistic, 0.5),
        (logit, 0.3),
        (logcosh, 1.5),
        (logabssinh, 0.3),
        (log1psq, 0.3),
        (log1pexp, 0.1),
        (log1mexp, -0.5),
        (log2mexp, 0.1),
        (logexpm1, 0.1),
        (log1pmx, -0.95),
        (logmxp1, 0.02),
        (cloglog, 0.5),
        (cexpexp, -0.3),
        (loglogistic, 0.5),
        (logitexp, -0.3),
        (log1mlogistic, -0.9),
        (logit1mexp, -0.6),
    ]
    @testset "$f" for (f, x) in scalar1_cases
        nfwd_test_rule(rng, 1, f, x)
    end

    scalar2_cases = Any[
        (xlogy, 0.3, 1.2),
        (xlog1py, 0.3, -0.5),
        (xexpy, 1.0, -0.7),
        (logaddexp, -0.5, 0.4),
        (logaddexp, 1.5, 1.5),   # equal-input edge case, see #881
        (logsubexp, -0.5, -5.0),
    ]
    @testset "$f($a, $b)" for (f, a, b) in scalar2_cases
        nfwd_test_rule(rng, 2, f, a, b)
    end

    @testset "logsumexp($desc)" for (desc, x) in [
        ("vector", randn(rng, 5)),
        ("matrix", randn(rng, 5, 4)),
        ("view", view(randn(rng, 5), 1:4)),
        ("[1.0,1.0]", [1.0, 1.0]),   # equal-input edge case, see #881
    ]
        nfwd_test_rule(rng, 8, logsumexp, x)
    end

    # Float32 — verify LogExpFunctions' precision-generic branches work
    @testset "$f (Float32)" for (f, x) in Any[
        (xlogx, Float32(1.1)),
        (logistic, Float32(0.5)),
        (log1pexp, Float32(0.1)),
        (logcosh, Float32(1.5)),
    ]
        nfwd_test_rule(rng, 1, f, x)
    end
    @testset "logaddexp (Float32)" begin
        nfwd_test_rule(rng, 2, logaddexp, -Float32(0.5), Float32(0.4))
    end
end

# ── array integration ─────────────────────────────────────────────────────────
# Representative LA cases from test/integration_testing/array/array.jl.
# Uses module-level constant matrices (_nfw_A5, _nfw_M5) for singleton wrappers.
#
# Excluded (API limitations, not missing rules):
#   mul! / setindex! / push! — mutating
#   adjoint / Transpose outputs — non-dense output type rejected
@testset "array integration" begin
    rng = Xoshiro(2)
    @testset "$name" for (name, f, x, C) in [
        ("sum_matvec", _nfw_sum_matvec, randn(rng, 5), 5),
        ("sum_linsolve", _nfw_sum_linsolve, randn(rng, 5), 5),
        ("sum_matmat", _nfw_sum_matmat, randn(rng, 25), 8),
    ]
        nfwd_test_rule(rng, C, f, x)
    end
end

# ── misc_abstract_array integration ──────────────────────────────────────────
# View and getindex cases from
# test/integration_testing/misc_abstract_array/misc_abstract_array.jl.
# Wrapped as named functions because Int/UnitRange inputs are rejected by
# nfwd's input validation when passed directly to getindex/view.
#
# Excluded (API limitations, not missing rules):
#   getindex(arr, Int)     — Int64 input rejected
#   view(arr, Range, Int)  — UnitRange/Int64 inputs rejected
#   setindex! / push!      — mutating
#   Pointer operations     — not differentiable
#   map/broadcast wrappers — vector output (pullback needed, not gradient)
@testset "misc_abstract_array integration" begin
    rng = Xoshiro(3)
    @testset "$name" for (name, f, x, C) in [
        ("sum_view", _nfw_sum_view, randn(rng, 10), 5),
        ("sum_getindex", _nfw_sum_getindex, randn(rng, 10), 5),
    ]
        nfwd_test_rule(rng, C, f, x)
    end
end

# ── wrapped previously-excluded cases ─────────────────────────────────────────
# Functions that previously couldn't be tested directly because they either
# had kwarg-lambda callables, vector/adjoint outputs, or mutating interfaces.
# All are wrapped as singleton named functions that return a scalar.
@testset "wrapped integration" begin
    using LogExpFunctions
    rng = Xoshiro(4)
    @testset "$name" for (name, f, x, C) in [
        # kwarg lambdas wrapped as named functions
        ("lse_dims1", _nfw_lse_dims1, randn(rng, 5, 4), 5),
        ("lse_dims2", _nfw_lse_dims2, randn(rng, 5, 4), 5),
        # vector/adjoint outputs reduced to scalar via dot/sum
        ("softmax_dot", _nfw_softmax_dot, randn(rng, 6), 5),
        ("adjoint_vec", _nfw_sum_adjoint_v, randn(rng, 7), 5),
        ("adjoint_mat", _nfw_sum_adjoint_m, randn(rng, 12), 5),
        ("map_sin", _nfw_sum_map_sin, randn(rng, 8), 5),
        # hardcoded-index wrappers (avoid passing Int/UnitRange as nfwd inputs)
        ("getindex_4", _nfw_getindex_4, randn(rng, 9), 3),
        ("view_23_1", _nfw_view_23_1, randn(rng, 3, 4), 3),
        # mutating cases using copy/similar so element type follows x
        ("setindex_sum", _nfw_setindex_sum, randn(rng, 6), 5),
        ("mul_sum", _nfw_mul_sum, randn(rng, 5), 5),
        ("push_sum", _nfw_push_sum, randn(rng, 5), 5),
        ("lse_bang_sum", _nfw_lse_bang_sum, randn(rng, 20), 5),
    ]
        nfwd_test_rule(rng, C, f, x)
    end
end
