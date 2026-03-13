using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using Mooncake: lgetfield
using Mooncake.TestUtils:
    test_tangent_interface,
    test_tangent_splitting,
    test_rule,
    test_frule_interface,
    test_rrule_interface
using LinearAlgebra

# Access NDual and friends from the CUDA extension (they are not exported from Mooncake core).
const _MooncakeCUDAExt = Base.get_extension(Mooncake, :MooncakeCUDAExt)
const NDual = _MooncakeCUDAExt.NDual
const ndual_value = _MooncakeCUDAExt.ndual_value
const ndual_partial = _MooncakeCUDAExt.ndual_partial
const NDualUnsupportedError = _MooncakeCUDAExt.NDualUnsupportedError

@testset "cuda" begin
    cuda = CUDA.functional()
    if cuda
        # TODO: move test case definitions to `src/ext/MooncakeCUDAExt.jl`, in line
        # with other rules.
        #
        # Check we can operate on CuArrays of various element types.
        @testset for ET in (Float32, Float64, ComplexF32, ComplexF64)
            # Use `undef` to test against garbage memory (NaNs, Infs, subnormals).
            # `randn` generates well-behaved values and can miss these edge cases.
            p = CuArray{ET,2,CUDA.DeviceMemory}(undef, 8, 8)
            test_tangent_interface(StableRNG(123456), p; interface_only=false)
            test_tangent_splitting(StableRNG(123456), p)

            # Check we can instantiate a CuArray.
            test_rule(
                StableRNG(123456),
                CuArray{ET,1,CUDA.DeviceMemory},
                undef,
                256;
                interface_only=true,
                is_primitive=false,
            )
            test_rule(
                StableRNG(123456),
                CuArray{ET,2,CUDA.DeviceMemory},
                undef,
                (16, 32);
                interface_only=true,
                is_primitive=true,
            )
            dp = Mooncake.zero_codual(p)
            if ET <: Real
                @test Mooncake.arrayify(dp) == (p, Mooncake.zero_tangent(p))
            elseif ET <: Complex
                primal_p, tangent_p = Mooncake.arrayify(dp)
                @test (primal_p, tangent_p) isa
                    Tuple{CuArray{ET,2,CUDA.DeviceMemory},CuArray{ET,2,CUDA.DeviceMemory}}
                @test all(iszero, tangent_p)
            end
        end
        rng = StableRNG(123)
        _rand = (rng, size...) -> CuArray(randn(rng, size...))
        _rand_pos = (rng, size...) -> CuArray(abs.(randn(rng, size...)) .+ 1e-3)
        _bcast_sum_sin(x) = sum(sin.(x))
        _bcast_sum_pow7(x) = sum(x .^ 7)
        _bcast_sum_log(x) = sum(log.(x))
        _bcast_sum_exp(x) = sum(exp.(x))
        _bcast_sum_lit_mul(x) = sum(2.0 .* x)
        _bcast_sum_mul(x, y) = sum(x .* y)
        _bcast_sum_sin_pow2(x) = sum(sin.(x .^ 2))
        _sum_f_sin(x) = sum(sin, x)
        _sum_f_exp(x) = sum(exp, x)
        # complex sum(f, x) wrappers
        _sum_f_cx_abs2(x) = sum(abs2, x)
        _sum_f_cx_sin_re(x) = real(sum(sin, x))
        # complex broadcast wrappers
        _bcast_cx_abs2(x) = sum(abs2.(x))
        _bcast_cx_sin_re(x) = real(sum(sin.(x)))
        _bcast_cx_mul_re(x, y) = real(sum(x .* y))
        # Adjoint / Transpose broadcast wrappers
        _bcast_adj_lit_add(x) = sum(x' .+ 1.0)        # real adjoint
        _bcast_adj_cx_abs2(x) = sum(abs2.(x'))         # complex adjoint, non-holomorphic
        _bcast_tp_lit_add(x) = sum(transpose(x) .+ 1.0) # real transpose
        # Shape-broadcasting: vector broadcast against matrix — tests _unbroadcast
        _bcast_vec_mat_add(v, m) = sum(v .+ m)     # v:(n,) broadcast to (n,p)
        _bcast_vec_mat_mul(v, m) = sum(v .* m)     # v:(n,) broadcast to (n,p)
        # map wrappers — map(f, ::CuArray) dispatches to broadcast in CUDA.jl,
        # so these are covered transitively by the materialize rule.
        _map_sin(x) = sum(map(sin, x))
        _map_mul(x, y) = sum(map(*, x, y))
        _map_cx_abs2(x) = sum(map(abs2, x))
        _map_cx_sin_re(x) = real(sum(map(sin, x)))
        _cu_sum(x) = sum(cu(x))
        _array_sum(x) = sum(Array(x))     # GPU→CPU transfer
        _diagonal_sum(x) = sum(Diagonal(x)) # GPU Diagonal construction
        _sum_f_abs(x) = sum(abs, x)          # sum(f, x) with non-smooth f
        _sum_f_abs2(x) = sum(abs2, x)        # sum(f, x) real abs2
        # scalar variable in a broadcast — gradient w.r.t. both x (CuArray) and c (scalar)
        _bcast_scalar_mul(x, c) = sum(c .* x)
        _bcast_scalar_add(x, c) = sum(x .+ c)
        _bcast_cx_scalar_mul(x, c) = real(sum(c .* x))     # real scalar, complex array
        _bcast_cx_cx_scalar_mul(x, c) = real(sum(c .* x))  # complex scalar, complex array
        _host_rand = (rng, size...) -> randn(rng, size...)
        @testset "_new_ interface" begin
            # Test the `_new_` frule!!/rrule!! interfaces directly.
            # `test_rule` would create `randn_dual` inputs for `CuDataRef`, which would
            # require custom `randn_tangent_internal`/`zero_tangent_internal` methods.
            # We avoid that because those methods would mainly exist to satisfy the test helper.
            for ET in (Float64, ComplexF64)
                data = getfield(_rand(rng, ET, 64, 32), :data)
                test_frule_interface(
                    Mooncake.Dual(Mooncake._new_, Mooncake.NoTangent()),
                    Mooncake.Dual(CuArray{ET,2,CUDA.DeviceMemory}, Mooncake.NoTangent()),
                    Mooncake.Dual(data, copy(data)),
                    Mooncake.Dual(2048, Mooncake.NoTangent()),
                    Mooncake.Dual(0, Mooncake.NoTangent()),
                    Mooncake.Dual((64, 32), Mooncake.NoTangent());
                    frule=Mooncake.frule!!,
                )
                test_rrule_interface(
                    Mooncake.CoDual(Mooncake._new_, Mooncake.NoTangent()),
                    Mooncake.CoDual(CuArray{ET,2,CUDA.DeviceMemory}, Mooncake.NoTangent()),
                    Mooncake.CoDual(data, copy(data)),
                    Mooncake.CoDual(2048, Mooncake.NoTangent()),
                    Mooncake.CoDual(0, Mooncake.NoTangent()),
                    Mooncake.CoDual((64, 32), Mooncake.NoTangent());
                    rrule=Mooncake.rrule!!,
                )
            end
        end
        test_cases = Any[
            # sum
            (false, :none, false, sum, _rand(rng, 64, 32)),
            # similar
            (true, :none, false, similar, _rand(rng, 64, 32)),
            # adjoint
            (false, :none, false, adjoint, _rand(rng, 64, 32)),
            (false, :none, false, adjoint, _rand(rng, ComplexF64, 64, 32)),
            # transpose
            (false, :none, false, transpose, _rand(rng, 64, 32)),
            (false, :none, false, transpose, _rand(rng, ComplexF64, 64, 32)),
            # reshape — exercises the DataRef-based _new_ rule
            (false, :none, false, x -> reshape(x, 32, 64), _rand(rng, 64, 32)),
            (false, :none, false, x -> reshape(x, 32, 64), _rand(rng, ComplexF64, 64, 32)),
            # lgetfield
            # `data` is an opaque storage handle, so only test the AD interface for these.
            (true, :none, true, lgetfield, _rand(rng, 64, 32), Val(1)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(2)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(3)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(4)),
            (true, :none, true, lgetfield, _rand(rng, 64, 32), Val(:data)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:maxsize)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:offset)),
            (false, :none, true, lgetfield, _rand(rng, 64, 32), Val(:dims)),
            # CPU→GPU transfer (cu)
            (false, :none, false, _cu_sum, _host_rand(rng, 16)),
            # GPU→CPU transfer (Array)
            (false, :none, false, _array_sum, _rand(rng, 16)),
            # GPU Diagonal construction
            (false, :none, false, _diagonal_sum, _rand(rng, 16)),
            # sum(::CuComplexArray) — 1-arg widened rule, sum itself is the primitive
            (false, :none, true, sum, _rand(rng, ComplexF64, 16)),
            # sum(f, ::CuFloatArray)
            (false, :none, false, _sum_f_sin, _rand(rng, 16)),
            (false, :none, false, _sum_f_exp, _rand(rng, 16)),
            # GPU broadcasts (materialize rule, real CuArrays)
            (false, :none, false, _bcast_sum_sin, _rand(rng, 16)),
            (false, :none, false, _bcast_sum_pow7, _rand(rng, 16)),
            (false, :none, false, _bcast_sum_log, _rand_pos(rng, 16)),
            (false, :none, false, _bcast_sum_exp, _rand(rng, 16)),
            (false, :none, false, _bcast_sum_lit_mul, _rand(rng, 16)),
            (false, :none, false, _bcast_sum_mul, _rand(rng, 16), _rand(rng, 16)),
            (false, :none, false, _bcast_sum_sin_pow2, _rand(rng, 16)),
            # Float32 broadcast variants — same functions, different element type
            (false, :none, false, _bcast_sum_sin, _rand(rng, Float32, 16)),
            (false, :none, false, _bcast_sum_lit_mul, _rand(rng, Float32, 16)),
            (
                false,
                :none,
                false,
                _bcast_sum_mul,
                _rand(rng, Float32, 16),
                _rand(rng, Float32, 16),
            ),
            # 2D broadcast inputs — exercises _unbroadcast and reshape paths
            (false, :none, false, _bcast_sum_sin, _rand(rng, 8, 4)),
            (false, :none, false, _bcast_sum_exp, _rand(rng, 8, 4)),
            # sum(f, ::CuFloatArray) — Float32 variant
            (false, :none, false, _sum_f_sin, _rand(rng, Float32, 16)),
            # sum(f, ::CuComplexArray) — 2-wide Duals, f:ℂ→ℝ and f:ℂ→ℂ
            (false, :none, false, _sum_f_cx_abs2, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _sum_f_cx_sin_re, _rand(rng, ComplexF64, 16)),
            # sum(f, ::CuComplexArray) — ComplexF32 variant
            (false, :none, false, _sum_f_cx_abs2, _rand(rng, ComplexF32, 16)),
            # GPU broadcasts on complex CuArrays
            (false, :none, false, _bcast_cx_abs2, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _bcast_cx_sin_re, _rand(rng, ComplexF64, 16)),
            (
                false,
                :none,
                false,
                _bcast_cx_mul_re,
                _rand(rng, ComplexF64, 16),
                _rand(rng, ComplexF64, 16),
            ),
            # ComplexF32 broadcast variants
            (false, :none, false, _bcast_cx_abs2, _rand(rng, ComplexF32, 16)),
            (false, :none, false, _bcast_cx_sin_re, _rand(rng, ComplexF32, 16)),
            # GPU broadcasts through Adjoint/Transpose leaves
            (false, :none, false, _bcast_adj_lit_add, _rand(rng, 16)),
            (false, :none, false, _bcast_adj_cx_abs2, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _bcast_tp_lit_add, _rand(rng, 16)),
            # Shape-broadcasting: vector vs matrix — exercises _unbroadcast in pullback
            (false, :none, false, _bcast_vec_mat_add, _rand(rng, 8), _rand(rng, 8, 4)),
            (false, :none, false, _bcast_vec_mat_mul, _rand(rng, 8), _rand(rng, 8, 4)),
            # map(f, ::CuArray) — transitive via materialize rule (CUDA.jl dispatches to broadcast)
            (false, :none, false, _map_sin, _rand(rng, 16)),
            (false, :none, false, _map_mul, _rand(rng, 16), _rand(rng, 16)),
            (false, :none, false, _map_cx_abs2, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _map_cx_sin_re, _rand(rng, ComplexF64, 16)),
            # sum(f, x) with non-smooth f (abs) and real abs2
            (false, :none, false, _sum_f_abs, _rand(rng, 16)),
            (false, :none, false, _sum_f_abs2, _rand(rng, 16)),
            # scalar variable in a broadcast — gradient w.r.t. both the CuArray and the scalar
            (false, :none, false, _bcast_scalar_mul, _rand(rng, 16), randn(rng)),
            (false, :none, false, _bcast_scalar_add, _rand(rng, 16), randn(rng)),
            # Float32 scalar broadcast variants
            (
                false,
                :none,
                false,
                _bcast_scalar_mul,
                _rand(rng, Float32, 16),
                randn(rng, Float32),
            ),
            (
                false,
                :none,
                false,
                _bcast_scalar_add,
                _rand(rng, Float32, 16),
                randn(rng, Float32),
            ),
            (
                false,
                :none,
                false,
                _bcast_cx_scalar_mul,
                _rand(rng, ComplexF64, 16),
                randn(rng),
            ),
            (
                false,
                :none,
                false,
                _bcast_cx_cx_scalar_mul,
                _rand(rng, ComplexF64, 16),
                randn(rng, ComplexF64),
            ),
        ]
        @testset "$(typeof(fargs))" for (
            interface_only, perf_flag, is_primitive, fargs...
        ) in test_cases

            @info "$(typeof(fargs))"
            perf_flag = cuda ? :none : perf_flag
            test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only)
        end

    else
        println("Tests are skipped because no CUDA device was found.")
    end

    include("ndual.jl")
end
