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
        # GPU→CPU transfer inside the function: Array(x::CuArray) path.
        _gpu_to_cpu(x) = sum(Array(x) .^ 2)
        # Bool mask via broadcast — creates a CuArray{Bool} internally; verifies that
        # integer/bool CuArrays (tangent_type = NoTangent) don't crash AD.
        _bool_mask_sum(x) = sum(x .* (x .> zero(eltype(x))))
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
            # accumulate(+) — explicit rule
            (false, :none, false, _accumulate_plus_sum, _rand(rng, 16)),
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
            # GPU→CPU transfer: Array(x::CuArray) path.
            (false, :none, false, _gpu_to_cpu, _rand(rng, 16)),
            # Bool mask broadcast — CuArray{Bool} tangent_type = NoTangent must not crash.
            (false, :none, false, _bool_mask_sum, _rand_pos(rng, 16)),
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
