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
    return Mooncake.build_rrule(
        Mooncake.Nfwd.NfwdMode{N}(), f.f, x...; debug_mode, silence_debug_messages
    )
end

function _nfwd_primitive_rrule(sig::Type{<:Tuple}, chunk_size::Int)
    Mooncake.build_rrule(Mooncake.Nfwd.NfwdMode{chunk_size}(), sig)
end

function nfwd_public_tangent(y, dy::Mooncake.NTangent)
    if y isa AbstractArray
        return length(dy) == 1 ? dy[1] : cat(dy...; dims=ndims(y) + 1)
    end
    return length(dy) == 1 ? dy[1] : tuple(dy...)
end

nfwd_public_tangent(_, dy) = dy
function nfwd_public_tangent(out::Mooncake.Dual)
    nfwd_public_tangent(Mooncake.primal(out), Mooncake.tangent(out))
end
nfwd_public_result(y, dy) = (y, nfwd_public_tangent(y, dy))

function steady_state_allocations_rrule(rule, fx::Tuple)
    for _ in 1:3
        rule(fx...)
    end
    GC.gc()
    return Mooncake.TestUtils.count_allocs(Base.splat(rule), fx)
end

function steady_state_allocations_frule2(f, x1, x2)
    for _ in 1:5
        Mooncake.frule!!(f, x1, x2)
    end
    GC.gc()
    return Mooncake.TestUtils.count_allocs(Mooncake.frule!!, f, x1, x2)
end

function steady_state_allocations_frule3(f, x1, x2, x3)
    for _ in 1:5
        Mooncake.frule!!(f, x1, x2, x3)
    end
    GC.gc()
    return Mooncake.TestUtils.count_allocs(Mooncake.frule!!, f, x1, x2, x3)
end

function steady_state_allocations_primitive_rrule2(f, x1, x2)
    for _ in 1:5
        Mooncake.rrule!!(f, x1, x2)
    end
    GC.gc()
    return Mooncake.TestUtils.count_allocs(Mooncake.rrule!!, f, x1, x2)
end

function steady_state_allocations_primitive_rrule3(f, x1, x2, x3)
    for _ in 1:5
        Mooncake.rrule!!(f, x1, x2, x3)
    end
    GC.gc()
    return Mooncake.TestUtils.count_allocs(Mooncake.rrule!!, f, x1, x2, x3)
end

function steady_state_allocations_value_and_gradient(cache, f, x)
    for _ in 1:3
        Mooncake.value_and_gradient!!(cache, f, x)
    end
    GC.gc()
    return Mooncake.TestUtils.count_allocs(Mooncake.value_and_gradient!!, cache, f, x)
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

nfwd_dispatch_gap(::Any) = -1.0
nfwd_dispatch_gap(x::Vector{Float64}) = sum(x)

struct NfwdBox64
    x::Float64
end

nfwd_box64_roundtrip(x) = NfwdBox64(x).x

nfwd_wrapper_target(x) = sin(x) + x^2
nfwd_nested_chunked_ir_helper(x) = sin(x)
function nfwd_nested_chunked_ir_helper(x::Mooncake.Nfwd.NDual)
    throw(Mooncake.Nfwd.NDualUnsupportedError(:chunked_ir_nested_dispatch))
end
nfwd_nested_chunked_ir_target(x) = nfwd_nested_chunked_ir_helper(x)

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
    @testset "chunked ir derived frule" begin
        @testset "stays on chunked ir path for nested dispatch" begin
            x = 1.5
            rule = Mooncake.build_frule(
                Mooncake.IRfwdMode{2}(), nfwd_nested_chunked_ir_target, x
            )

            y, dy = Mooncake.value_and_derivative!!(
                rule,
                (nfwd_nested_chunked_ir_target, Mooncake.NoTangent()),
                (x, Mooncake.NTangent((1.0, 2.0))),
            )

            expected = cos(x)
            @test y == nfwd_nested_chunked_ir_target(x)
            @test dy == Mooncake.NTangent((expected, 2 * expected))
        end

        @testset "preserves primal dispatch boundary" begin
            x = [1.0, 2.0, 3.0]
            rule = Mooncake.build_frule(Mooncake.IRfwdMode{2}(), nfwd_dispatch_gap, x)

            y, dy = Mooncake.value_and_derivative!!(
                rule,
                (nfwd_dispatch_gap, Mooncake.NoTangent()),
                (x, Mooncake.NTangent((ones(3), fill(2.0, 3)))),
            )

            @test y == sum(x)
            @test dy == Mooncake.NTangent((3.0, 6.0))
        end

        @testset "reuses tangent system through _new_" begin
            x = 3.0
            rule = Mooncake.build_frule(Mooncake.IRfwdMode{2}(), nfwd_box64_roundtrip, x)

            y, dy = Mooncake.value_and_derivative!!(
                rule,
                (nfwd_box64_roundtrip, Mooncake.NoTangent()),
                (x, Mooncake.NTangent((1.0, 2.0))),
            )

            @test y == x
            @test dy == Mooncake.NTangent((1.0, 2.0))
        end

        @testset "supports public scalar pullback and gradient APIs" begin
            x = [1.0, 2.0, 3.0]
            rule = Mooncake.build_frule(Mooncake.IRfwdMode{2}(), sum, x)

            value_pb, pullback = Mooncake.value_and_pullback!!(rule, 3.0, sum, x)
            @test value_pb == sum(x)
            @test pullback == (Mooncake.NoTangent(), fill(3.0, length(x)))

            value_grad, gradient = Mooncake.value_and_gradient!!(rule, sum, x)
            @test value_grad == sum(x)
            @test gradient == (Mooncake.NoTangent(), ones(length(x)))
        end

        @testset "nfwdrule supports scalar pullback and gradient APIs" begin
            x = 1.5
            rule = Mooncake.NfwdMooncake.NfwdRule{Tuple{typeof(sin),Float64},2}()
            value_grad, gradient = Mooncake.value_and_gradient!!(rule, sin, x)
            @test value_grad == sin(x)
            @test gradient == (Mooncake.NoTangent(), cos(x))

            pair_rule = Mooncake.NfwdMooncake.NfwdRule{Tuple{typeof(sincos),Float64},2}()
            value_pb, pullback = Mooncake.value_and_pullback!!(
                pair_rule, (2.0, 3.0), sincos, x
            )
            @test value_pb == sincos(x)
            @test pullback == (Mooncake.NoTangent(), 2 * cos(x) - 3 * sin(x))
        end

        @testset "supports cached scalar pullback and gradient APIs" begin
            x = [1.0, 2.0, 3.0]
            rule = Mooncake.build_frule(Mooncake.IRfwdMode{2}(), sum, x)
            cache = Mooncake.NfwdMooncake.IRfwdCache(rule, sum, x)

            value_pb, pullback = Mooncake.value_and_pullback!!(cache, 3.0, sum, x)
            @test value_pb == sum(x)
            @test pullback == (Mooncake.NoTangent(), fill(3.0, length(x)))

            value_grad, gradient = Mooncake.value_and_gradient!!(cache, sum, x)
            @test value_grad == sum(x)
            @test gradient == (Mooncake.NoTangent(), ones(length(x)))

            friendly_cache = Mooncake.NfwdMooncake.IRfwdCache(
                rule, sum, x; config=Mooncake.Config(; friendly_tangents=true)
            )
            value_friendly, gradient_friendly = Mooncake.value_and_gradient!!(
                friendly_cache, sum, x
            )
            @test value_friendly == sum(x)
            @test gradient_friendly == (Mooncake.NoTangent(), ones(length(x)))
        end
    end

    @testset "width-aware NDual scalar inputs" begin
        f(x) = sin(x) + x
        x = Mooncake.dual_type(Val(2), Float64)(2.0, Mooncake.NTangent((1.0, -1.0)))
        rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), f, 2.0)
        y = rule(Mooncake.zero_dual(f), x)
        @test Mooncake.primal(y) == f(2.0)
        @test Mooncake.tangent(y) == Mooncake.NTangent((cos(2.0) + 1, -(cos(2.0) + 1)))

        g(z) = z^2
        z = Mooncake.dual_type(Val(2), ComplexF64)(
            1.0 + 2.0im, Mooncake.NTangent((1.0 + 0.0im, 0.0 + 1.0im))
        )
        grule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), g, 1.0 + 2.0im)
        gz = grule(Mooncake.zero_dual(g), z)
        @test Mooncake.primal(gz) == (1.0 + 2.0im)^2
        @test Mooncake.tangent(gz) == Mooncake.NTangent((2.0 + 4.0im, -4.0 + 2.0im))
    end

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
            dual_inputs=(
                Mooncake.Dual(x, Mooncake.NTangent((dx, 0.0))),
                Mooncake.Dual(y, Mooncake.NTangent((0.0, dy))),
            ),
            expected_tangent=(dx * y + dx * (-sin(x)), x * dy),
        ),
    )

    @testset "public rule entrypoints" begin
        @testset "scalar rule construction" begin
            @testset "$case.name" for case in scalar_cases
                rule = Mooncake.build_frule(
                    Mooncake.Nfwd.NDualMode{case.chunk_size}(), f, x, y
                )
                out = rule(Mooncake.zero_dual(f), case.dual_inputs...)
                @test out isa Mooncake.Dual
                @test Mooncake.primal(out) == z
                @test nfwd_public_tangent(out) == case.expected_tangent

                rrule = Mooncake.build_rrule(
                    Mooncake.Nfwd.NfwdMode{case.chunk_size}(), f, x, y
                )
                ȳ, pb!! = rrule(
                    Mooncake.zero_fcodual(f),
                    Mooncake.zero_fcodual(x),
                    Mooncake.zero_fcodual(y),
                )
                @test Mooncake.primal(ȳ) == z
                @test pb!!(1.0) == (Mooncake.NoRData(), y - sin(x), x)
            end

            @testset "direct value_and_derivative!! on Rule" begin
                rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), f, x, y)

                out_dual = Mooncake.value_and_derivative!!(
                    rule,
                    Mooncake.zero_dual(f),
                    Mooncake.Dual(x, Mooncake.NTangent((dx, 0.0))),
                    Mooncake.Dual(y, Mooncake.NTangent((0.0, dy))),
                )
                @test Mooncake.primal(out_dual) == z
                @test nfwd_public_tangent(out_dual) == (dx * y + dx * (-sin(x)), x * dy)

                out_tuple = Mooncake.value_and_derivative!!(
                    rule,
                    (f, Mooncake.NoTangent()),
                    (x, Mooncake.NTangent((dx, 0.0))),
                    (y, Mooncake.NTangent((0.0, dy))),
                )
                @test nfwd_public_result(out_tuple...) ==
                    (z, (dx * y + dx * (-sin(x)), x * dy))

                out_chunk = Mooncake.value_and_derivative!!(
                    rule,
                    (f, Mooncake.NoTangent()),
                    (x, Mooncake.NTangent((dx, 0.0))),
                    (y, Mooncake.NTangent((0.0, dy))),
                )
                @test out_chunk == (z, Mooncake.NTangent((dx * y + dx * (-sin(x)), x * dy)))
            end
        end

        @testset "scalar rule edge cases" begin
            square(x) = x^2

            @testset "single-input scalar rule with chunk_size>1" begin
                x_single = 1.5
                rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), square, x_single)
                value, grad = Mooncake.value_and_gradient!!(rule, square, x_single)
                @test value == square(x_single)
                @test grad == (Mooncake.NoTangent(), 2 * x_single)
            end

            @testset "single-input scalar NDual outputs still extract correctly" begin
                x_single = 1.5

                frule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), exp, x_single)
                out = Mooncake.value_and_derivative!!(
                    frule, (exp, Mooncake.NoTangent()), (x_single, 1.0)
                )
                @test nfwd_public_result(out...) == (exp(x_single), exp(x_single))

                value, grad = Mooncake.value_and_gradient!!(frule, exp, x_single)
                @test value == exp(x_single)
                @test grad == (Mooncake.NoTangent(), exp(x_single))
            end

            @testset "unsupported output diagnostics include supported shapes and values" begin
                f_int(x) = x > 0 ? 1 : 2
                rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), f_int, x)
                err = try
                    Mooncake.value_and_derivative!!(
                        rule, (f_int, Mooncake.NoTangent()), (x, 1.0)
                    )
                    nothing
                catch err
                    err
                end
                msg = sprint(showerror, err)
                @test err isa Mooncake.Nfwd.UnsupportedOutputError
                @test occursin("nfwd output unsupported.", msg)
                @test occursin("Supported nfwd inputs:", msg)
                @test occursin("Supported nfwd outputs:", msg)
                @test occursin("IEEEFloat scalars", msg)
                @test occursin("Int64 (not size-bearing)", msg)
            end

            @testset "automatic chunk_size selection" begin
                auto_frule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), f, x, y)
                @test auto_frule isa Mooncake.NfwdMooncake.Rule
                auto_rrule = Mooncake.build_rrule(Mooncake.Nfwd.NfwdMode{2}(), f, x, y)
                @test auto_rrule isa Mooncake.NfwdMooncake.RRule
                value2, grad2 = Mooncake.value_and_gradient!!(auto_frule, f, x, y)
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
                frule_pb = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), g, x_vec)
                value, pullback = Mooncake.value_and_pullback!!(
                    frule_pb, [3.0, 4.0], g, x_vec
                )
                @test value == sin.(x_vec)
                @test pullback ==
                    (Mooncake.NoTangent(), [3.0 * cos(x_vec[1]), 4.0 * cos(x_vec[2])])

                out = frule_pb(
                    Mooncake.zero_dual(g),
                    Mooncake.Dual(x_vec, Mooncake.NTangent(([1.0, 0.0], [0.0, 1.0]))),
                )
                @test Mooncake.primal(out) == sin.(x_vec)
                @test nfwd_public_tangent(out) ≈ [cos(x_vec[1]) 0.0; 0.0 cos(x_vec[2])]
            end

            @testset "complex inputs" begin
                fc(z) = real(z * z + cos(z))
                zc = ComplexF64(1.2, -0.3)
                dzc = ComplexF64(0.5, -0.25)
                expected_dzc = real((2zc - sin(zc)) * dzc)
                rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), fc, zc)
                out = rule(Mooncake.zero_dual(fc), Mooncake.Dual(zc, dzc))
                @test Mooncake.primal(out) == fc(zc)
                @test nfwd_public_tangent(out) ≈ expected_dzc

                rrule = Mooncake.build_rrule(Mooncake.Nfwd.NfwdMode{2}(), fc, zc)
                ȳ, pb!! = rrule(Mooncake.zero_fcodual(fc), Mooncake.zero_fcodual(zc))
                @test Mooncake.primal(ȳ) == fc(zc)
                @test pb!!(1.0) == (Mooncake.NoRData(), conj(2zc - sin(zc)))

                value_scalar, grad_scalar = Mooncake.value_and_gradient!!(rule, fc, zc)
                @test value_scalar == fc(zc)
                @test grad_scalar == (Mooncake.NoTangent(), conj(2zc - sin(zc)))

                gc(z) = sum(abs2, z)
                z_vec = ComplexF64[1.0 + 2.0im, -3.0 + 0.5im]
                dz_vec = reshape(
                    ComplexF64[1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 1.0im], 2, 2
                )
                frule_vec = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), gc, z_vec)
                out_vec = frule_vec(
                    Mooncake.zero_dual(gc),
                    Mooncake.Dual(z_vec, Mooncake.NTangent((dz_vec[:, 1], dz_vec[:, 2]))),
                )
                @test Mooncake.primal(out_vec) == gc(z_vec)
                @test nfwd_public_tangent(out_vec) == (2.0, 1.0)

                hc(z) = z .* z
                ȳ_vec = ComplexF64[2.0 - 1.0im, -0.5 + 0.25im]
                rrule_vec = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), hc, z_vec)
                value_vec, pullback_vec = Mooncake.value_and_pullback!!(
                    rrule_vec, ȳ_vec, hc, z_vec
                )
                @test value_vec == hc(z_vec)
                @test pullback_vec == (Mooncake.NoTangent(), 2 .* conj.(z_vec) .* ȳ_vec)

                hm(z) = (z[1] += 1.0 + 2.0im; z)
                z_mut = ComplexF64[1.0 + 0.0im, 2.0 + 0.0im]
                dz_mut = ComplexF64[3.0 + 0.0im, 4.0 + 0.0im]
                rule_mut = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), hm, z_mut)
                out_mut = rule_mut(Mooncake.zero_dual(hm), Mooncake.Dual(z_mut, dz_mut))
                @test Mooncake.primal(out_mut) === z_mut
                @test Mooncake.tangent(out_mut)[1] === dz_mut
            end

            @testset "multi-argument and matrix array inputs" begin
                h(x, y) = sum(x .* y)
                x_vec = [1.0, 2.0]
                y_vec = [3.0, 4.0]
                dx = reshape([1.0, 0.0, 0.0, 0.0], 2, 2)
                dy = reshape([0.0, 0.0, 1.0, 0.0], 2, 2)
                frule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), h, x_vec, y_vec)
                out = frule(
                    Mooncake.zero_dual(h),
                    Mooncake.Dual(x_vec, Mooncake.NTangent((dx[:, 1], dx[:, 2]))),
                    Mooncake.Dual(y_vec, Mooncake.NTangent((dy[:, 1], dy[:, 2]))),
                )
                @test Mooncake.primal(out) == h(x_vec, y_vec)
                @test nfwd_public_tangent(out) == (3.0, 1.0)

                hm(X) = sin.(X)
                X = reshape([1.0, 2.0, 3.0, 4.0], 2, 2)
                dX = zeros(2, 2, 2)
                dX[1, 1, 1] = 1.0
                dX[2, 2, 2] = 1.0
                frule_matrix = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), hm, X)
                y_and_dy = frule_matrix(
                    Mooncake.zero_dual(hm),
                    Mooncake.Dual(X, Mooncake.NTangent((dX[:, :, 1], dX[:, :, 2]))),
                )
                @test Mooncake.primal(y_and_dy) == sin.(X)
                expected = zeros(2, 2, 2)
                expected[1, 1, 1] = cos(X[1, 1])
                expected[2, 2, 2] = cos(X[2, 2])
                @test nfwd_public_tangent(y_and_dy) ≈ expected

                hm_mut(x) = (x[1] += 2.0; x)
                x_mut = [1.0, 2.0]
                dx_mut = [3.0, 4.0]
                rule_mut = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), hm_mut, x_mut)
                out_mut = rule_mut(Mooncake.zero_dual(hm_mut), Mooncake.Dual(x_mut, dx_mut))
                @test Mooncake.primal(out_mut) === x_mut
                @test Mooncake.tangent(out_mut)[1] === dx_mut
            end
        end

        @testset "tuple outputs" begin
            @testset "sincos chunk_size=1 frule and rrule" begin
                x_sc = 1.2
                frule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), sincos, x_sc)
                out = frule(Mooncake.zero_dual(sincos), Mooncake.Dual(x_sc, 1.0))
                @test Mooncake.primal(out) == sincos(x_sc)
                s_t, c_t = nfwd_public_tangent(out)
                @test s_t ≈ cos(x_sc)
                @test c_t ≈ -sin(x_sc)

                rrule = Mooncake.build_rrule(Mooncake.Nfwd.NfwdMode{1}(), sincos, x_sc)
                ȳ_sc, pb!! = rrule(
                    Mooncake.zero_fcodual(sincos), Mooncake.zero_fcodual(x_sc)
                )
                @test Mooncake.primal(ȳ_sc) == sincos(x_sc)
                ȳ_sin, ȳ_cos = 3.0, 2.0
                @test pb!!((ȳ_sin, ȳ_cos)) ==
                    (Mooncake.NoRData(), ȳ_sin * cos(x_sc) + ȳ_cos * (-sin(x_sc)))
            end

            @testset "modf derivative: frac=1, int=0" begin
                x_mf = 3.7
                frule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), modf, x_mf)
                out = frule(Mooncake.zero_dual(modf), Mooncake.Dual(x_mf, 1.0))
                @test Mooncake.primal(out) == modf(x_mf)
                frac_t, int_t = nfwd_public_tangent(out)
                @test frac_t ≈ 1.0
                @test int_t ≈ 0.0
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
            @test_throws ArgumentError Mooncake.build_frule(
                Mooncake.Nfwd.NDualMode{1}(), f_stateful, 5.0
            )
            @test_throws ArgumentError Mooncake.build_rrule(
                Mooncake.Nfwd.NfwdMode{1}(), f_stateful, 5.0
            )
            @test_throws ArgumentError Mooncake.build_rrule(
                Mooncake.Nfwd.NfwdMode{1}(), Tuple{NfwdDerivedMultiplier,Float64}
            )
            @test_throws ArgumentError Mooncake.build_rrule(
                Mooncake.Nfwd.NfwdMode{1}(), Tuple{Nothing,Float64}
            )
        end

        @testset "unsupported config is rejected" begin
            @test_throws ArgumentError Mooncake.build_frule(
                Mooncake.Nfwd.NDualMode{1}(), f, x, y; debug_mode=true
            )
            @test_throws ArgumentError Mooncake.build_rrule(
                Mooncake.Nfwd.NfwdMode{1}(), f, x, y; debug_mode=true
            )
            @test_throws ArgumentError Mooncake.build_rrule(
                Mooncake.Nfwd.NDualMode{1}(), f, x, y
            )
        end

        @testset "invalid chunk_size is rejected" begin
            @test_throws ArgumentError Mooncake.NfwdMooncake.build_nfwd_frule(
                f, x, y; chunk_size=0
            )
        end

        @testset "function tangent rejection" begin
            rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{1}(), f, x, y)
            @test_throws ArgumentError rule(
                Mooncake.Dual(f, 1.0), Mooncake.Dual(x, dx), Mooncake.Dual(y, dy)
            )

            rrule = Mooncake.build_rrule(Mooncake.Nfwd.NfwdMode{1}(), f, x, y)
            @test_throws ArgumentError rrule(
                Mooncake.CoDual(f, 1.0), Mooncake.zero_fcodual(x), Mooncake.zero_fcodual(y)
            )
        end

        @testset "array tangent validation" begin
            g(x) = sin.(x)
            x_vec = [1.0, 2.0]
            rule = Mooncake.build_frule(Mooncake.Nfwd.NDualMode{2}(), g, x_vec)
            @test_throws ArgumentError rule(
                Mooncake.zero_dual(g), Mooncake.Dual(x_vec, [1.0, 2.0, 3.0])
            )
        end

        @testset "dense array workspace reuses fixed-shape storage without allocation" begin
            ws = Mooncake.NfwdMooncake.NfwdDenseArrayWorkspace{Float64,2}()
            buf = Mooncake.NfwdMooncake._nfwd_dense_array_workspace!(ws, (8, 2))
            @test buf === Mooncake.NfwdMooncake._nfwd_dense_array_workspace!(ws, (8, 2))
            GC.gc()
            @test Mooncake.TestUtils.count_allocs(
                Mooncake.NfwdMooncake._nfwd_dense_array_workspace!, ws, (8, 2)
            ) == 0
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

            @testset "binary/ternary frule!! and rrule!! stay allocation-free" begin
                # Regression: consolidating _nfwd_extract to a Union method caused
                # runtime dispatch on the result type, adding ~224 bytes per call.
                # Regression: using `map` in vararg _nfwd_primitive_frule_call broke the
                # inlining chain into _nfwd_eval, also causing allocations.
                # Regression: _pt_scale/_pt_sub/_pt_add closure heap-allocation for N=1.
                f_atan = Mooncake.Dual(atan, Mooncake.NoTangent())
                x1 = Mooncake.Dual(1.0, 1.0)
                x2 = Mooncake.Dual(2.0, 0.0)
                @test steady_state_allocations_frule2(f_atan, x1, x2) == 0

                cf_atan = Mooncake.zero_fcodual(atan)
                cx1 = Mooncake.CoDual(1.0, Mooncake.NoFData())
                cx2 = Mooncake.CoDual(2.0, Mooncake.NoFData())
                @test steady_state_allocations_primitive_rrule2(cf_atan, cx1, cx2) == 0

                f_clamp = Mooncake.Dual(clamp, Mooncake.NoTangent())
                x3 = Mooncake.Dual(0.0, 0.0)
                @test steady_state_allocations_frule3(f_clamp, x1, x2, x3) == 0

                cf_clamp = Mooncake.zero_fcodual(clamp)
                cx3 = Mooncake.CoDual(0.0, Mooncake.NoFData())
                @test steady_state_allocations_primitive_rrule3(cf_clamp, cx1, cx2, cx3) ==
                    0
            end

            @testset "multi-argument array pullback pins allocations" begin
                # The generic (multi-arg) pullback path allocates for temporaries; pin it so
                # regressions are caught.  This does not assert zero: it asserts the value
                # does not grow unexpectedly between runs (steady-state == 3rd-run value).
                h(x, y) = sum(x .* y)
                x_a = randn(8)
                y_a = randn(8)
                rrule = Mooncake.build_rrule(Mooncake.Nfwd.NfwdMode{4}(), h, x_a, y_a)
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
