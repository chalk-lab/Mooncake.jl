using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using CUDA.GPUArrays: unsafe_free!
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
        # mapreduce / reduce wrappers — CUDA uses opaque reduction kernels; explicit rules
        # intercept op=+ / op=Base.add_sum and redirect to the ForwardDiff.Dual machinery.
        # Note: in Julia 1.11, sum(f, x) dispatches through Base._sum → mapreduce(f, add_sum, x)
        # rather than being intercepted by our sum(f, x) primitive; both code paths are tested.
        # _sum_f_sin and _sum_f_abs2 are already defined above (from broadcast section).
        _mapreduce_sin(x) = mapreduce(sin, +, x)
        _mapreduce_exp(x) = mapreduce(exp, +, x)
        _mapreduce_cx_abs2(x) = mapreduce(abs2, +, x)
        _mapreduce_cx_sin_re(x) = real(mapreduce(sin, +, x))
        _reduce_plus(x) = reduce(+, x)
        _reduce_plus_cx(x) = reduce(+, x)
        _reduce_mul(x) = reduce(*, x)
        _reduce_mul_cx(x) = reduce(*, x)
        # norm / dot — CUBLAS routines with explicit rules
        _norm(x) = norm(x)
        _norm_cx(x) = norm(x)
        _dot(x, y) = dot(x, y)
        # prod / cumsum / cumprod / accumulate(+) — explicit rules
        _prod(x) = prod(x)
        _prod_cx(x) = real(prod(x))
        _cumsum_sum(x) = sum(cumsum(x))
        _cumsum_cx_sum(x) = real(sum(cumsum(x)))
        _cumprod_sum(x) = sum(cumprod(x))
        _cumprod_cx_sum(x) = real(sum(cumprod(x)))
        _accumulate_plus_sum(x) = sum(accumulate(+, x))
        _accumulate_plus_cx_sum(x) = real(sum(accumulate(+, x)))
        # vector indexing — gather/scatter-add
        _gather_sum(x, idx) = sum(x[idx])
        _gather_sum_cx(x, idx) = real(sum(x[idx]))
        _cu_sum(x) = sum(cu(x))
        _array_sum(x) = sum(Array(x))     # GPU→CPU transfer
        _diagonal_sum(x) = sum(Diagonal(x)) # GPU Diagonal construction
        _diagonal_field_bcast(x) = sum(exp.(Diagonal(x).diag))  # Diagonal + lgetfield + broadcast
        _sum_f_abs(x) = sum(abs, x)          # sum(f, x) with non-smooth f
        _sum_f_abs2(x) = sum(abs2, x)        # sum(f, x) real abs2
        _sum_adj_pow3(x) = real(sum(y -> y^3, x'))  # sum(f, Adjoint)
        # sum(A') and sum(transpose(A)) for complex arrays
        _sum_cx_adj(x) = real(sum(x'))          # sum(adjoint) of complex CuArray
        _sum_cx_tr(x) = real(sum(transpose(x))) # sum(transpose) of complex CuArray
        # scalar variable in a broadcast — gradient w.r.t. both x (CuArray) and c (scalar)
        _bcast_scalar_mul(x, c) = sum(c .* x)
        _bcast_scalar_add(x, c) = sum(x .+ c)
        _bcast_cx_scalar_mul(x, c) = real(sum(c .* x))     # real scalar, complex array
        _bcast_cx_cx_scalar_mul(x, c) = real(sum(c .* x))  # complex scalar, complex array
        # adjoint of a CuVector times a CuMatrix — dispatches through generic_matmatmul!
        # because CUBLAS.gemm! only accepts CuMatrix inputs; now covered by the explicit rule.
        _cu_slice_adj_mul(x, cy) = sum(cu(x[:, 1])' * cy)
        # copy(CuArray) → copyto! → unsafe_copyto! — exercises the unsafe_copyto! rule.
        _copy_sum(x) = sum(copy(x))
        _copy_sum_cx(x) = real(sum(copy(x)))
        # in-place broadcast (x .= f.(y)) — exercises materialize! frule!! / rrule!!.
        # _inplace_add_alias! tests the aliasing-safe path: dest appears in bc.args.
        # _inplace_cx_abs2! tests real-output-into-complex-dest: abs2(ℂ)→ℝ written into
        # a ComplexF64 array, exercising Float64→ComplexF64 promotion and 2-DOF partials.
        _inplace_sin!(x, y) = (x.=sin.(y); sum(x))
        _inplace_add_alias!(x, y) = (x.=x .+ y; sum(x))
        _inplace_cx_abs2!(x, y) = (x.=abs2.(y); real(sum(x)))
        # GPU→CPU transfer inside the function: Array(x::CuArray) path.
        _gpu_to_cpu(x) = sum(Array(x) .^ 2)
        # CPU→GPU transfer: copies a host Array into a GPU dest via unsafe_copyto!(GPU←CPU).
        # Exercises the mixed-device rrule (dest::CuArray, src::Array).
        # The gradient flows back from the GPU cotangent to the CPU src tangent.
        function _cpu_to_gpu_sum(x)
            dest = similar(x)
            copyto!(dest, Array(x))
            return sum(dest)
        end
        # CuPtr arithmetic — exercises the CuPtr{T} + Integer primitives.
        # _view_sum: view(x, range) triggers SubArray → unsafe_convert(CuPtr{T}, parent) +
        # offset, which is CuPtr{Float32} + Integer (differentiable T).
        _view_sum(x) = sum(view(x, 2:length(x)))
        _view_sum_cx(x) = real(sum(view(x, 2:length(x))))
        # _view_bool_gate_sum: Bool mask applied via a view; CuArray{Bool} is
        # non-differentiable (tangent_type(Bool)=NoTangent), so gradient flows
        # through x only.  Verifies that Bool CuArray views don't crash AD.
        # Uses eltype(x) conversion to work for any float precision.
        _view_bool_gate_sum(x) = sum(
            x .* eltype(x).(view(x .> zero(eltype(x)), 1:length(x)))
        )
        # Helpers for non-default memory types.
        _rand_unified =
            (rng, sz...) ->
                CuArray{Float32,length(sz),CUDA.UnifiedMemory}(randn(rng, Float32, sz...))
        _rand_host =
            (rng, sz...) ->
                CuArray{Float32,length(sz),CUDA.HostMemory}(randn(rng, Float32, sz...))
        # Dense-layer-style: W*x + b — exercises matmul (mightalias via copy in
        # the rrule) plus bias broadcast on GPU.
        _linear(W, x, b) = sum(W * x .+ b)
        _linear_cx(W, x, b) = real(sum(W * x .+ b))
        # These functions exercise operations not yet fully differentiable on GPU.
        # They are used in the "unsupported operations" testset below.
        _cu_cx_slice_adj_mul(x, cy) = real(sum(cu(x[:, 1])' * cy))
        _bcast_cx_mixed(x, y) = sum(abs2, x .^ 2 .+ y)
        _vcat_cu_sum(x, y) = sum(vcat(x, y))
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
            # mul! (matrix × matrix, Float64)
            (
                false,
                :stability,
                false,
                mul!,
                _rand(rng, 16, 32),
                _rand(rng, 16, 8),
                _rand(rng, 8, 32),
            ),
            # mul! (matrix × vector, Float64)
            (
                false,
                :stability,
                false,
                mul!,
                _rand(rng, 16),
                _rand(rng, 16, 8),
                _rand(rng, 8),
            ),
            # mul! (matrix × matrix, ComplexF64) — cuBLAS bug on Julia ≤ 1.10, skip.
            (if VERSION >= v"1.11"
                [(
                    false,
                    :stability,
                    false,
                    mul!,
                    _rand(rng, ComplexF64, 16, 32),
                    _rand(rng, ComplexF64, 16, 8),
                    _rand(rng, ComplexF64, 8, 32),
                )]
            else
                []
            end)...,
            # mul! (matrix × vector, Float32)
            (
                false,
                :stability,
                false,
                mul!,
                _rand(rng, Float32, 16),
                _rand(rng, Float32, 16, 8),
                _rand(rng, Float32, 8),
            ),
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
            # sum(f, x) — exercises mapreduce(f, add_sum, x) path (Julia 1.11 specific)
            (false, :none, false, _sum_f_sin, _rand(rng, 16)),
            (false, :none, false, _sum_f_abs2, _rand(rng, 16)),
            (false, :none, false, _sum_f_abs2, _rand(rng, ComplexF64, 16)),
            # mapreduce(f, +, x) — explicit rule, redirects to ForwardDiff.Dual machinery
            (false, :none, false, _mapreduce_sin, _rand(rng, 16)),
            (false, :none, false, _mapreduce_exp, _rand(rng, 16)),
            (false, :none, false, _mapreduce_cx_abs2, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _mapreduce_cx_sin_re, _rand(rng, ComplexF64, 16)),
            # reduce(+, x) — explicit rule, redirects to sum machinery
            (false, :none, false, _reduce_plus, _rand(rng, 16)),
            (false, :none, false, _reduce_plus, _rand(rng, Float32, 16)),
            (false, :none, false, _reduce_plus_cx, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _reduce_plus_cx, _rand(rng, ComplexF32, 16)),
            # reduce(*, x) — explicit rule, redirects to prod machinery
            (false, :none, false, _reduce_mul, _rand_pos(rng, 16)),
            (false, :none, false, _reduce_mul, _rand_pos(rng, Float32, 16)),
            (false, :none, false, _reduce_mul_cx, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _reduce_mul_cx, _rand(rng, ComplexF32, 16)),
            # norm — CUBLAS rule (real and complex)
            (false, :none, false, _norm, _rand(rng, 16)),
            (false, :none, false, _norm_cx, _rand(rng, ComplexF64, 16)),
            # dot — CUBLAS rule (real vectors)
            (false, :none, false, _dot, _rand(rng, 16), _rand(rng, 16)),
            # prod — explicit rule (real and complex)
            (false, :none, false, _prod, _rand_pos(rng, 16)),
            (false, :none, false, _prod_cx, _rand(rng, ComplexF64, 16)),
            # cumsum — explicit rule (real and complex)
            (false, :none, false, _cumsum_sum, _rand(rng, 16)),
            (false, :none, false, _cumsum_cx_sum, _rand(rng, ComplexF64, 16)),
            # cumprod — explicit rule (real and complex, nonzero inputs)
            (false, :none, false, _cumprod_sum, _rand_pos(rng, 16)),
            (false, :none, false, _cumprod_cx_sum, _rand(rng, ComplexF64, 16)),
            # accumulate(+) — explicit rule (real and complex)
            (false, :none, false, _accumulate_plus_sum, _rand(rng, 16)),
            (false, :none, false, _accumulate_plus_cx_sum, _rand(rng, ComplexF64, 16)),
            # vector indexing — gather forward, scatter-add pullback
            (
                false,
                :none,
                false,
                _gather_sum,
                _rand(rng, 16),
                CuArray(Int32[2, 5, 7, 3, 1, 8]),
            ),
            (
                false,
                :none,
                false,
                _gather_sum_cx,
                _rand(rng, ComplexF64, 16),
                CuArray(Int32[2, 5, 7, 3, 1, 8]),
            ),
            # Diagonal + lgetfield(:diag) + broadcast — exercises the full pipeline
            (false, :none, false, _diagonal_field_bcast, _rand_pos(rng, 16)),
            # sum(f, x) with non-smooth f (abs) and real abs2
            (false, :none, false, _sum_f_abs, _rand(rng, 16)),
            (false, :none, false, _sum_f_abs2, _rand(rng, 16)),
            # sum(f, Adjoint) — tests sum(f, x) dispatch when input is an Adjoint wrapper
            (false, :none, false, _sum_adj_pow3, _rand(rng, 16)),
            # sum(A') / sum(transpose(A)) for complex arrays
            (false, :none, false, _sum_cx_adj, _rand(rng, ComplexF64, 16)),
            (false, :none, false, _sum_cx_tr, _rand(rng, ComplexF64, 16)),
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
            # slicing CPU array then adjoint+matmul on GPU — goes through generic_matvecmul!
            # (CUBLAS gemv path); forward mode now works because CUBLAS.handle is a primitive.
            (
                false,
                :none,
                false,
                _cu_slice_adj_mul,
                _host_rand(rng, Float32, 3, 3),
                _rand(rng, Float32, 3, 3),
            ),
            # copy(CuArray) → copyto! → unsafe_copyto! — regression for UpsilonNode error.
            (false, :none, false, _copy_sum, _rand(rng, 16)),
            (false, :none, false, _copy_sum_cx, _rand(rng, ComplexF64, 16)),
            # UnifiedMemory and HostMemory CuArrays — same unsafe_copyto! rule, different M.
            (false, :none, false, _copy_sum, _rand_unified(rng, 16)),
            (false, :none, false, _copy_sum, _rand_host(rng, 16)),
            # Direct unsafe_copyto!(dest, doffs, src, soffs, n) tests (is_primitive=true).
            # Full-array copy: doffs=soffs=1, n=length(src).
            (false, :none, true, unsafe_copyto!, _rand(rng, 16), 1, _rand(rng, 16), 1, 16),
            # Sub-range copy: only elements 2..5 of dest are overwritten; rest unchanged.
            (false, :none, true, unsafe_copyto!, _rand(rng, 16), 2, _rand(rng, 16), 1, 4),
            # Complex full-array copy.
            (
                false,
                :none,
                true,
                unsafe_copyto!,
                _rand(rng, ComplexF64, 8),
                1,
                _rand(rng, ComplexF64, 8),
                1,
                8,
            ),
            # GPU→CPU transfer: Array(x::CuArray) path.
            (false, :none, false, _gpu_to_cpu, _rand(rng, 16)),
            # CPU→GPU transfer: copyto!(CuArray, Array) → unsafe_copyto!(GPU, CPU).
            (false, :none, false, _cpu_to_gpu_sum, _rand(rng, 16)),
            # CuPtr{T} + Integer — differentiable T (Float32): view(x, range) internally
            # calls unsafe_convert(CuPtr{Float32}, SubArray) = unsafe_convert(parent) + offset.
            (false, :none, false, _view_sum, _rand(rng, 16)),
            (false, :none, false, _view_sum_cx, _rand(rng, ComplexF64, 16)),
            # Bool-masked sum: CuArray{Bool} is non-differentiable; gradient flows through x.
            # Test both Float32 (original) and Float64 (regression for DataRef zero_tangent).
            (false, :none, false, _view_bool_gate_sum, _rand_pos(rng, 16)),
            (false, :none, false, _view_bool_gate_sum, _rand_pos(rng, Float64, 16)),
            # fill!(CuArray, val) — GPU fill! has internal try/catch → UpsilonNode.
            # Regression for Flux LSTM hidden-state reset (fill! with integer 0).
            # Also test float value to exercise gradient propagation through x.
            (false, :none, true, fill!, _rand(rng, 16), 0.0f0),
            (false, :none, true, fill!, _rand(rng, 4, 4), 0.0f0),
            # Complex CuArray: tests rdata_type(ComplexF64) + sum(da) on complex tangent.
            (false, :none, true, fill!, _rand(rng, ComplexF64, 8), 0.5 + 0.5im),
            # Lambda wrapper: not itself a primitive; is_primitive=false so test_rule does not
            # assert that the built rule is frule!!/rrule!!.
            (false, :none, false, (a) -> (fill!(a, Int32(0)); sum(a)), _rand(rng, 16)),
            # in-place broadcast — exercises materialize! frule!! / rrule!!.
            # Three cases: basic (sin), aliased dest (x .= x .+ y),
            # and real-output-into-complex-dest (abs2: ℂ→ℝ stored into ComplexF64 array).
            (false, :none, false, _inplace_sin!, _rand(rng, 16), _rand(rng, 16)),
            (false, :none, false, _inplace_add_alias!, _rand(rng, 16), _rand(rng, 16)),
            (
                false,
                :none,
                false,
                _inplace_cx_abs2!,
                _rand(rng, ComplexF64, 16),
                _rand(rng, ComplexF64, 16),
            ),
            # Dense-layer-style forward pass: W*x + b → relu → sum.
            # Exercises the 7-arg generic_matmatmul! rule + bias broadcast + mightalias.
            (
                false,
                :none,
                false,
                _linear,
                _rand(rng, 4, 4),
                _rand(rng, 4, 4),
                _rand(rng, 4),
            ),
            (
                false,
                :none,
                false,
                _linear_cx,
                _rand(rng, ComplexF64, 4, 4),
                _rand(rng, ComplexF64, 4, 4),
                _rand(rng, ComplexF64, 4),
            ),
        ]
        @testset "$(typeof(fargs))" for (
            interface_only, perf_flag, is_primitive, fargs...
        ) in test_cases

            @info "$(typeof(fargs))"
            perf_flag = cuda ? :none : perf_flag
            test_rule(StableRNG(123), fargs...; perf_flag, is_primitive, interface_only)
        end

        # Direct unit tests for CuPtr{T} + Integer frule!! / rrule!!.
        #
        # Background: there are two dispatch branches in the rule:
        #   • Differentiable T (e.g. Float32): fdata_type(CuPtr{Float32}) = CuPtr{Float32}.
        #     Both primal and tangent pointers are offset by n.
        #   • Non-differentiable T (e.g. Cvoid, Bool): fdata_type(CuPtr{Cvoid}) = NoFData.
        #     Only the primal is offset; the tangent stays NoTangent / NoFData.
        #
        # Why direct calls and not test_rule?
        #   CuPtr is not an array type, so test_rule cannot construct meaningful inputs.
        #   The functional path (_view_sum / _view_sum_cx) exercises the differentiable-T
        #   branch end-to-end via SubArray → unsafe_convert → CuPtr{Float32} + offset.
        #   However, that path never touches the non-differentiable-T branch: a
        #   CuArray{Bool} view has tangent_type(Bool)=NoTangent, so unsafe_convert is
        #   never called with a Bool fdata, and CuPtr{Bool}+Integer is never reached.
        #   These direct tests are therefore the only coverage for the NoFData branch.
        @testset "CuPtr{T} + Integer direct (Float32 and Cvoid)" begin
            # ── frule!! — differentiable T ────────────────────────────────────────────
            # Both primal and tangent pointers must advance by the same byte offset n.
            p32 = CuPtr{Float32}(UInt64(4096))
            dp32 = Mooncake.Dual(p32, CuPtr{Float32}(UInt64(4096)))  # tangent = same base addr
            dn = Mooncake.Dual(Int64(64), NoTangent())
            result = _MooncakeCUDAExt.frule!!(Mooncake.Dual(+, NoTangent()), dp32, dn)
            @test primal(result) == p32 + 64
            @test tangent(result) == CuPtr{Float32}(UInt64(4096)) + 64

            # ── frule!! — non-differentiable T (Cvoid) ───────────────────────────────
            # Only primal advances; tangent must remain NoTangent (not crash or wrong type).
            pv = CuPtr{Cvoid}(UInt64(4096))
            dpv = Mooncake.Dual(pv, NoTangent())
            result_v = _MooncakeCUDAExt.frule!!(Mooncake.Dual(+, NoTangent()), dpv, dn)
            @test primal(result_v) == pv + 64
            @test tangent(result_v) isa NoTangent

            # ── rrule!! — differentiable T ────────────────────────────────────────────
            # Output tangent (fdata) must be the offset tangent pointer.
            dp32_co = CoDual(p32, CuPtr{Float32}(UInt64(4096)))
            dn_co = CoDual(Int64(64), NoFData())
            out, pb = _MooncakeCUDAExt.rrule!!(CoDual(+, NoFData()), dp32_co, dn_co)
            @test primal(out) == p32 + 64
            @test tangent(out) == CuPtr{Float32}(UInt64(4096)) + 64

            # ── rrule!! — non-differentiable T (Cvoid) ───────────────────────────────
            # Output fdata must be NoFData (not crash, not a stray pointer).
            dpv_co = CoDual(pv, NoFData())
            out_v, pb_v = _MooncakeCUDAExt.rrule!!(CoDual(+, NoFData()), dpv_co, dn_co)
            @test primal(out_v) == pv + 64
            @test tangent(out_v) isa NoFData
        end

        # Direct unit tests for Core.finalizer, hasfieldcount, and copy(::CuDataRef).
        #
        # test_rule cannot be used for these because:
        #   - Core.finalizer has a side effect (GC registration) and returns nothing.
        #   - hasfieldcount takes a Type value; test_rule cannot construct array-like
        #     tangents for Type arguments.
        #   - copy(::CuDataRef) requires randn_tangent_internal for DataRef, which does
        #     not exist (DataRef is opaque — it has no numerical content to randomise).
        @testset "Core.finalizer frule!! / rrule!!" begin
            # Core.finalizer(f, x) registers f as a GC finalizer for x; returns nothing.
            # The rule simply calls the primal and returns Dual/CoDual(nothing, NoTangent).
            fin = _ -> nothing
            arr = _rand(rng, Float32, 4)
            tarr = zero_tangent(arr)

            # frule!!: output is Dual(nothing, NoTangent()).
            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(Core.finalizer, NoTangent()),
                Mooncake.Dual(fin, NoTangent()),
                Mooncake.Dual(arr, tarr),
            )
            @test primal(result) === nothing
            @test tangent(result) isa NoTangent

            # rrule!!: output fdata is NoFData; pullback returns NoRData for all inputs.
            out, pb = _MooncakeCUDAExt.rrule!!(
                CoDual(Core.finalizer, NoFData()), CoDual(fin, NoFData()), CoDual(arr, tarr)
            )
            @test primal(out) === nothing
            @test tangent(out) isa NoFData
            @test all(x -> x isa NoRData, pb(NoRData()))
        end

        @testset "hasfieldcount frule!! / rrule!!" begin
            # CUDA.hasfieldcount(T) returns Bool — no gradient path.
            # Verify the primal result is forwarded and tangent is always NoTangent/NoFData.
            for T in (ComplexF64, Float32, Any)
                expected = CUDA.hasfieldcount(T)

                result = _MooncakeCUDAExt.frule!!(
                    Mooncake.Dual(CUDA.hasfieldcount, NoTangent()),
                    Mooncake.Dual(T, NoTangent()),
                )
                @test primal(result) === expected
                @test tangent(result) isa NoTangent

                out, pb = _MooncakeCUDAExt.rrule!!(
                    CoDual(CUDA.hasfieldcount, NoFData()), CoDual(T, NoFData())
                )
                @test primal(out) === expected
                @test tangent(out) isa NoFData
                @test all(x -> x isa NoRData, pb(NoRData()))
            end
        end

        @testset "copy(::CuDataRef) frule!! / rrule!!" begin
            # copy(::DataRef) increments the refcount and returns a new handle to the
            # same GPU buffer.  frule!!: both primal and tangent DataRefs are copied.
            # rrule!!: same; pullback is NoPullback (no numerical gradient through DataRef).
            ref = getfield(_rand(rng, Float32, 16), :data)
            tref = copy(ref)

            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(copy, NoTangent()), Mooncake.Dual(ref, tref)
            )
            @test primal(result) isa typeof(ref)
            @test primal(result) !== ref    # must be a new handle, not the same object
            @test tangent(result) isa typeof(tref)
            @test tangent(result) !== tref  # tangent DataRef also copied

            out, pb = _MooncakeCUDAExt.rrule!!(CoDual(copy, NoFData()), CoDual(ref, tref))
            @test primal(out) isa typeof(ref)
            @test primal(out) !== ref
            @test tangent(out) isa typeof(tref)
            @test tangent(out) !== tref
            @test all(x -> x isa NoRData, pb(NoRData()))
        end

        @testset "unsafe_free! frule!! / rrule!!" begin
            # unsafe_free! releases GPU memory early; pure side-effect, no gradient.
            # frule!!: returns Dual(nothing, NoTangent()); both primal and tangent freed.
            # rrule!!: returns CoDual(nothing, NoFData()) — regression test for the bug
            #          where NoTangent() was incorrectly used in the fdata slot.
            arr = _rand(rng, Float32, 4)
            tarr = zero_tangent(arr)

            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(unsafe_free!, NoTangent()), Mooncake.Dual(arr, tarr)
            )
            @test primal(result) === nothing
            @test tangent(result) isa NoTangent

            arr2 = _rand(rng, Float32, 4)
            tarr2 = zero_tangent(arr2)
            out, pb = _MooncakeCUDAExt.rrule!!(
                CoDual(unsafe_free!, NoFData()), CoDual(arr2, tarr2)
            )
            @test primal(out) === nothing
            @test tangent(out) isa NoFData  # must be NoFData, not NoTangent
            @test all(x -> x isa NoRData, pb(NoRData()))
        end

        # Verify that unsupported GPU operations throw user-friendly ArgumentErrors rather
        # than silent wrong answers or opaque internal crashes.  Each case exercises an
        # explicit catch-all rule that blocks an unimplemented differentiation path.
        # If a case gains a proper rule in the future, move it back into test_cases above
        # and delete it from here.
        @testset "unsupported operations throw ArgumentError" begin
            # Mixed-precision GPU broadcast (Float32 array .+ ComplexF32 array) is not
            # supported.  The materialize frule/rrule detects mismatched GPU element types
            # and throws before any kernel launch.
            @testset "mixed-eltype GPU broadcast" begin
                f = _bcast_cx_mixed
                x = _rand(rng, Float32, 4)
                y = CuArray(randn(rng, ComplexF32, 4))
                @test_throws r"GPU broadcast over arrays with mixed element types" value_and_gradient!!(
                    prepare_gradient_cache(f, x, y), f, x, y
                )
            end

            # vcat/hcat/cat on CuArrays are not yet differentiable — explicit rules throw
            # rather than letting Mooncake trace into opaque CUDA memory kernels.
            @testset "vcat CuArray not differentiable" begin
                f = _vcat_cu_sum
                x = _rand(rng, Float32, 4)
                y = _rand(rng, Float32, 4)
                @test_throws r"vcat on CuArray is not yet differentiable" value_and_gradient!!(
                    prepare_gradient_cache(f, x, y), f, x, y
                )
            end

            # Scalar getindex/setindex! on CuArray — throw to prevent silent scalar GPU ops.
            @testset "scalar getindex CuArray not differentiable" begin
                f = x -> x[1]
                x = _rand(rng, Float32, 4)
                @test_throws r"scalar indexing of CuArray is not differentiable" value_and_gradient!!(
                    prepare_gradient_cache(f, x), f, x
                )
            end
            @testset "scalar setindex! CuArray not differentiable" begin
                f = x -> (x[1]=0.0f0; sum(x))
                x = _rand(rng, Float32, 4)
                @test_throws r"scalar indexing of CuArray is not differentiable" value_and_gradient!!(
                    prepare_gradient_cache(f, x), f, x
                )
            end

            # accumulate with unsupported op — catch-all rule throws ArgumentError.
            @testset "accumulate non-+ CuArray not differentiable" begin
                f = x -> sum(accumulate(*, x))
                x = _rand(rng, Float32, 4)
                @test_throws r"accumulate on CuArray only supports op=\+" value_and_gradient!!(
                    prepare_gradient_cache(f, x), f, x
                )
            end

            # Complex slice-adjoint-matvec: cu(x[:, 1])' * cy — cu() downcasts ComplexF64
            # to ComplexF32, producing a type mismatch with cy::CuMatrix{ComplexF64}.
            # The generic_matvecmul! frule/rrule detects the mismatch before any CUBLAS call.
            @testset "complex slice-adjoint-matvec type mismatch" begin
                f = _cu_cx_slice_adj_mul
                x = _host_rand(rng, ComplexF64, 3, 3)
                cy = _rand(rng, ComplexF64, 3, 3)
                @test_throws r"GPU gemv with mismatched element types" value_and_gradient!!(
                    prepare_gradient_cache(f, x, cy), f, x, cy
                )
            end
        end
    else
        println("Tests are skipped because no CUDA device was found.")
    end

    include("ndual.jl")
end
