using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, CUDA, JET, Mooncake, StableRNGs, Test
using CUDA.CUDACore.GPUArrays: unsafe_free!
using CUDA.CUDACore: hasfieldcount
using Base: unsafe_convert
using Mooncake: lgetfield
using Mooncake.TestUtils:
    test_tangent_interface,
    test_tangent_splitting,
    test_rule,
    test_frule_interface,
    test_rrule_interface
using LinearAlgebra, Statistics

const _MooncakeCUDAExt = Base.get_extension(Mooncake, :MooncakeCUDAExt)

@testset "cuda" begin
    cuda = CUDA.functional()
    if cuda
        # TODO: move test case definitions to `src/ext/MooncakeCUDAExt.jl`, in line
        # with other rules.
        #
        # Check we can operate on CuArrays of various element types.

        @testset "_copy_output / _copy_to_output!! for CuArray{$ET}" for ET in (
            Float32, Float64, ComplexF32, ComplexF64
        )
            p = CuArray(randn(ET, 4, 4))
            p_copy = Mooncake._copy_output(p)
            @test p_copy == p
            @test p_copy !== p
            @test typeof(p_copy) == typeof(p)
            p2 = CuArray(randn(ET, 4, 4))
            Mooncake._copy_to_output!!(p_copy, p2)
            @test p_copy == p2
        end

        @testset for ET in (Float32, Float64, ComplexF32, ComplexF64)
            # Use `undef` to test against garbage memory (NaNs, Infs, subnormals).
            # `randn` generates well-behaved values and can miss these edge cases.
            p = CuArray{ET,2,CUDA.DeviceMemory}(undef, 8, 8)
            test_tangent_interface(StableRNG(123456), p; interface_only=false)
            test_tangent_splitting(StableRNG(123456), p)

            # Check we can instantiate a CuArray.
            # 1D: goes through DerivedRule (not a registered primitive).
            test_rule(
                StableRNG(123456),
                CuArray{ET,1,CUDA.DeviceMemory},
                undef,
                256;
                interface_only=true,
                is_primitive=false,
            )
            # 2D: marked is_primitive=true to test the primitive interface directly
            # (Mooncake has a _new_ primitive rule for CuArray construction).
            test_rule(
                StableRNG(123456),
                CuArray{ET,2,CUDA.DeviceMemory},
                undef,
                (16, 32);
                interface_only=true,
                is_primitive=true,
            )
            dp = Mooncake.zero_codual(p)
            primal_p, tangent_p = Mooncake.arrayify(dp)
            @test primal_p === p
            if ET <: Real
                @test tangent_p == Mooncake.zero_tangent(p)
            elseif ET <: Complex
                @test (primal_p, tangent_p) isa
                    Tuple{CuArray{ET,2,CUDA.DeviceMemory},CuArray{ET,2,CUDA.DeviceMemory}}
                @test all(iszero, tangent_p)
            end
        end
        rng = StableRNG(123)
        _rand = (rng, size...) -> CuArray(randn(rng, size...))
        _rand_pos = (rng, size...) -> CuArray(abs.(randn(rng, size...)) .+ 1.0e-3)
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
        # Non-contiguous SubArray leaf: rows 1:2 of a column-major matrix are strided, so the
        # view stays a SubArray (a contiguous view collapses to a plain CuArray).
        _bcast_noncontig_view(x) = sum(exp.(view(x, 1:2, :)))
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
        # Note: _sum_f_sin is defined above (line 79); _sum_f_abs2 is defined below (line 135).
        _mapreduce_sin(x) = mapreduce(sin, +, x)
        _mapreduce_exp(x) = mapreduce(exp, +, x)
        _mapreduce_cx_abs2(x) = mapreduce(abs2, +, x)
        _mapreduce_cx_sin_re(x) = real(mapreduce(sin, +, x))
        _reduce_plus(x) = reduce(+, x)
        # _reduce_plus_cx returns a complex scalar for complex input (no real() wrap), unlike
        # _prod_cx / _cumsum_cx_sum etc.  The separate alias keeps the testset name distinct.
        _reduce_plus_cx(x) = reduce(+, x)
        _reduce_mul(x) = reduce(*, x)
        _reduce_mul_cx(x) = reduce(*, x)
        # norm / dot — cuBLAS routines with explicit rules.
        # norm() always returns a real scalar regardless of element type, so _norm_cx has
        # the same body as _norm; the alias exists solely to label the complex-input testset.
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
        _bcast_sum_abs2(x) = sum(abs2.(x))  # regression for mixed-precision reduced pullback
        _bcast_cx_scalar_mul(x, c) = real(sum(c .* x))     # real scalar, complex array
        _bcast_cx_cx_scalar_mul(x, c) = real(sum(c .* x))  # complex scalar, complex array
        _bcast_nested_sin_add(x, y) = sum(sin.(x .+ y))
        _bcast_nested_float_cast_sin(x) = sum(sin.(Float64.(x)))
        _bcast_zero_dof_nested(x, c, b) = sum(x .+ c .* Float64.(b .> 0))
        _bcast_all_scalar_leaf(x, s) = sum(x .* (s .+ 1.0))
        _inplace_zero_dof_nested!(dest, x, c, b) =
            (dest.=x .+ c .* Float64.(b .> 0); sum(dest))
        # adjoint of a CuVector times a CuMatrix — dispatches through generic_matmatmul!
        # because cuBLAS.gemm! only accepts CuMatrix inputs; now covered by the explicit rule.
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
        _vcat_cu_sum(xs...) = sum(vcat(xs...))  # vararg: reused for 2-arg and N-arg tests
        _hcat_cu_sum(xs...) = sum(hcat(xs...))  # vararg: reused for 2-arg and N-arg tests
        _cat_cu_sum(d) = (xs...) -> sum(cat(xs...; dims=d))  # vararg: reused for 2-arg and N-arg tests
        _permutedims_sum(perm) = x -> sum(permutedims(x, perm))
        # Wrappers for Statistics.varm GPU rule tests.
        _varm_sum_d1(x, m) = sum(varm(x, m; dims=1, corrected=false))
        _varm_sum_d2(x, m) = sum(varm(x, m; dims=2, corrected=true))
        # no-dims path; reused for real, complex, and mixed real/complex scalar means.
        _varm_nodims_scalar(x, m) = varm(x, m; corrected=true)
        # Tuple dims: what GroupNorm/InstanceNorm/BatchNorm pass (ntuple(static, N-1)).
        _varm_sum_dtuple(x, m) = sum(varm(x, m; dims=(1, 2), corrected=false))
        # UnitRange dims: default LayerNorm(shape) passes `1:(N-M)` here, not a Tuple.
        _varm_sum_drange(x, m) = sum(varm(x, m; dims=1:2, corrected=false))
        # Array m with `dims=:`: the only test hitting the array-mean rule's Colon branch.
        _varm_sum_dcolon_arraymean(x, m) = sum(varm(x, m; dims=:, corrected=false))
        # Repeated dims (a careless ntuple could produce them): the denominator must count
        # dim 1 once, not twice.
        _varm_sum_ddup(x, m) = sum(varm(x, m; dims=(1, 1), corrected=false))
        # Empty dims: the rule's denominator must mirror _mean_denom's `init=1`.
        _varm_sum_dempty(x, m) = sum(varm(x, m; dims=(), corrected=false))
        # Bare 2-arg spelling: bypasses Core.kwcall, hitting the dedicated bare-call rule.
        _varm_bare_nodims_scalar(x, m) = varm(x, m)
        # Kwarg sets the real function rejects must throw identically under AD (the rules
        # call the real method for their primal).
        _varm_arraymean_missing_dims(x, m) = varm(x, m; corrected=false)
        _varm_scalarmean_stray_dims(x, m) = varm(x, m; dims=1)
        # Wrappers for Statistics.mean GPU rule tests.
        _mean_sum_d1(x) = sum(mean(x; dims=1))
        _mean_sum_d2(x) = sum(mean(x; dims=2))
        _mean_sum_dtuple(x) = sum(mean(x; dims=(1, 2)))
        _mean_sum_drange(x) = sum(mean(x; dims=1:2))
        _mean_sum_ddup(x) = sum(mean(x; dims=(1, 1)))
        _mean_sum_dempty(x) = sum(mean(x; dims=()))  # empty dims: see _varm_sum_dempty
        _mean_nodims(x) = mean(x; dims=:)
        # Bare spelling: decomposed (no rule); used to pin rule/decomposition agreement.
        _mean_bare(x) = mean(x)
        # Complex CuArray mean variants
        _mean_cx_nodims(x) = real(mean(x; dims=:))   # ComplexF32 → real part for scalar grad test
        _mean_cx_sum_d1(x) = real(sum(mean(x; dims=1)))
        # Wrappers for keyword sum GPU rule tests (#1273).
        _sum_kw_d1(x) = sum(sum(x; dims=1))
        _sum_kw_nodims(x) = sum(x; dims=:)
        _sum_kw_cx_d1(x) = real(sum(sum(x; dims=1)))
        _sum_kw_init_wide(x) = sum(sum(x; dims=1, init=0.0))
        _sum_kw_init_active(a, x) = sum(sum(x; dims=1, init=a))
        _sum_kw_badkw(x) = sum(x; bad=1.0)
        # Wrappers for the maximum/minimum GPU rules.
        _max_bare(x) = maximum(x)
        _max_nodims(x) = maximum(x; dims=:)
        _max_d1(x) = sum(maximum(x; dims=1))
        _min_d1(x) = sum(minimum(x; dims=1))
        _host_rand = (rng, size...) -> randn(rng, size...)
        @testset "_new_ interface" begin
            # Test the `_new_` frule!!/rrule!! interfaces directly.
            # `test_rule` would create `randn_dual` inputs for `CuDataRef`, which would
            # require custom `randn_tangent_internal`/`zero_tangent_internal` methods.
            # We avoid that because those methods would mainly exist to satisfy the test helper.
            #
            # NOTE: test_frule_interface and test_rrule_interface both take full tangents
            # (tangent_type) in the second Dual/CoDual slot, then extract fdata internally
            # via to_fwds before calling the rule.  Non-differentiable args therefore take
            # NoTangent() here — NOT NoFData(), even for the rrule interface test.
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
                :none,
                false,
                mul!,
                _rand(rng, 16, 32),
                _rand(rng, 16, 8),
                _rand(rng, 8, 32),
            ),
            # mul! (matrix × vector, Float64)
            (false, :none, false, mul!, _rand(rng, 16), _rand(rng, 16, 8), _rand(rng, 8)),
            # mul! (matrix × matrix, ComplexF64) — cuBLAS bug on Julia ≤ 1.10, skip.
            (if VERSION >= v"1.11"
                [(
                    false,
                    :none,
                    false,
                    mul!,
                    _rand(rng, ComplexF64, 16, 32),
                    _rand(rng, ComplexF64, 16, 8),
                    _rand(rng, ComplexF64, 8, 32),
                ),]
            else
                []
            end)...,
            # mul! (matrix × vector, Float32)
            (
                false,
                :none,
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
            (false, :none, false, _bcast_sum_abs2, _rand(rng, Float32, 16)),
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
            # Non-contiguous SubArray broadcast leaf (rows 1:2 of a 4x3 stay a SubArray)
            (false, :none, false, _bcast_noncontig_view, _rand(rng, 4, 3)),
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
            # norm — cuBLAS rule (real and complex)
            (false, :none, false, _norm, _rand(rng, 16)),
            (false, :none, false, _norm_cx, _rand(rng, ComplexF64, 16)),
            # dot — cuBLAS rule (real vectors)
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
            # sum(f, x) with non-smooth f (abs)
            (false, :none, false, _sum_f_abs, _rand(rng, 16)),
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
            # (cuBLAS gemv path); forward mode now works because cuBLAS.handle is a primitive.
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
            # Wrapped destinations fell through to the untraceable `cufunction` until the
            # bound became `CuMaybeWrappedArray`; summing and restoring go through the wrapper.
            (false, :none, true, fill!, _rand(rng, 4, 4)', 0.0f0),
            (false, :none, true, fill!, transpose(_rand(rng, 4, 4)), 0.0f0),
            (false, :none, true, fill!, view(_rand(rng, 4, 4), 1:2, :), 0.0f0),
            (false, :none, true, fill!, _rand(rng, ComplexF64, 4, 4)', 0.5 + 0.5im),
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
            # vcat on CuArrays
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                _rand(rng, Float32, 8),
                _rand(rng, Float32, 4),
            ),
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                _rand(rng, Float32, 8, 3),
                _rand(rng, Float32, 4, 3),
            ),
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                _rand(rng, Float64, 6),
                _rand(rng, Float64, 6),
            ),
            # hcat on CuArrays
            (
                false,
                :none,
                false,
                _hcat_cu_sum,
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 4, 2),
            ),
            (
                false,
                :none,
                false,
                _hcat_cu_sum,
                _rand(rng, Float64, 4, 3),
                _rand(rng, Float64, 4, 2),
            ),
            # cat on CuArrays (dims kwarg)
            (
                false,
                :none,
                false,
                _cat_cu_sum(1),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 2, 3),
            ),
            (
                false,
                :none,
                false,
                _cat_cu_sum(2),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 4, 2),
            ),
            # cat on CuArrays (dims kwarg as Val{N}, per _unwrap_cat_dim(::Val{N}))
            (
                false,
                :none,
                false,
                _cat_cu_sum(Val(1)),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 2, 3),
            ),
            # cat on CuArrays (Tuple dims kwarg: block-diagonal concatenation)
            (
                false,
                :none,
                false,
                _cat_cu_sum((1, 2)),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 2, 5),
            ),
            (
                false,
                :none,
                false,
                _cat_cu_sum((1, 2)),
                _rand(rng, Float64, 3, 2),
                _rand(rng, Float64, 5, 4),
            ),
            # UnitRange dims: a spelling Base accepts, as for `varm` above.
            (
                false,
                :none,
                false,
                _cat_cu_sum(1:2),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 2, 5),
            ),
            # Tuple dims, N-arg: exercises the running-offsets tuple in _cu_concat_pb!.
            (
                false,
                :none,
                false,
                _cat_cu_sum((1, 2)),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 2, 5),
                _rand(rng, Float32, 3, 2),
            ),
            # Complex CuArrays: CuMaybeWrappedArray covers them via CuFloatOrComplex.
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                _rand(rng, ComplexF32, 4, 3),
                _rand(rng, ComplexF32, 2, 3),
            ),
            (
                false,
                :none,
                false,
                _hcat_cu_sum,
                _rand(rng, ComplexF64, 4, 3),
                _rand(rng, ComplexF64, 4, 2),
            ),
            (
                false,
                :none,
                false,
                _cat_cu_sum((1, 2)),
                _rand(rng, ComplexF64, 3, 2),
                _rand(rng, ComplexF64, 5, 4),
            ),
            # Wrapped and mixed arguments: each is canonicalised independently via
            # `arrayify`, so any combination works.
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                adjoint(_rand(rng, Float32, 3, 4)),
                _rand(rng, Float32, 2, 3),
            ),
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                transpose(_rand(rng, Float32, 3, 4)),
                transpose(_rand(rng, Float32, 3, 4)),
            ),
            (
                false,
                :none,
                false,
                _vcat_cu_sum,
                view(_rand(rng, Float32, 8, 3), 1:4, :),
                _rand(rng, Float32, 2, 3),
            ),
            (
                false,
                :none,
                false,
                _hcat_cu_sum,
                transpose(_rand(rng, Float32, 3, 4)),
                view(_rand(rng, Float32, 4, 2), :, :),
            ),
            # N-arg: Vararg{CuMaybeWrappedArray} matches each argument independently
            # rather than requiring a uniform type.
            (
                false,
                :none,
                false,
                _cat_cu_sum(1),
                _rand(rng, Float32, 4, 3),
                adjoint(_rand(rng, Float32, 3, 2)),
                transpose(_rand(rng, Float32, 3, 5)),
            ),
            (
                false,
                :none,
                false,
                _hcat_cu_sum,
                adjoint(_rand(rng, Float32, 3, 4)),
                transpose(_rand(rng, Float32, 2, 4)),
                view(_rand(rng, Float32, 4, 6), :, 1:5),
            ),
            # permutedims on CuArrays
            (false, :none, false, _permutedims_sum((2, 1)), _rand(rng, Float32, 8, 4)),
            (false, :none, false, _permutedims_sum((2, 1)), _rand(rng, Float64, 8, 4)),
            (
                false,
                :none,
                false,
                _permutedims_sum((2, 1)),
                adjoint(_rand(rng, Float32, 3, 4)),
            ),
            (
                false,
                :none,
                false,
                _permutedims_sum((2, 1)),
                transpose(_rand(rng, Float32, 3, 4)),
            ),
            (
                false,
                :none,
                false,
                _permutedims_sum((2, 1)),
                view(_rand(rng, Float32, 8, 4), 1:4, :),
            ),
            (false, :none, false, _permutedims_sum((2, 1)), _rand(rng, ComplexF32, 8, 4)),
            (
                false,
                :none,
                false,
                _permutedims_sum((2, 1, 3)),
                _rand(rng, Float32, 4, 6, 3),
            ),
            # cat with dims beyond either 2-D input's own ndims (new trailing axis).
            (
                false,
                :none,
                false,
                _cat_cu_sum(3),
                _rand(rng, Float32, 4, 3),
                _rand(rng, Float32, 4, 3),
            ),
            # hcat of two bare CuVectors, not matrices.
            (
                false,
                :none,
                false,
                _hcat_cu_sum,
                _rand(rng, Float32, 5),
                _rand(rng, Float32, 5),
            ),
        ]
        @testset "$(typeof(fargs))" for (interface_only, _, is_primitive, fargs...) in
                                        test_cases

            argtypes = join(string.(typeof.(fargs[2:end])), ", ")
            @info "[GPU] testing $(fargs[1])($argtypes)"
            # CUDA.jl internal dispatch patterns produce spurious JET/AllocCheck hits
            # unrelated to our rules, so stability checks are not meaningful on GPU.
            test_rule(
                StableRNG(123), fargs...; perf_flag=:none, is_primitive, interface_only
            )
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
            dp32 = Mooncake.Dual(p32, CuPtr{Float32}(UInt64(4096)))  # Mooncake.tangent = same base addr
            dn = Mooncake.Dual(Int64(64), Mooncake.NoTangent())
            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(+, Mooncake.NoTangent()), dp32, dn
            )
            @test Mooncake.primal(result) == p32 + 64
            @test Mooncake.tangent(result) == CuPtr{Float32}(UInt64(4096)) + 64

            # ── frule!! — non-differentiable T (Cvoid) ───────────────────────────────
            # Only primal advances; tangent must remain NoTangent (not crash or wrong type).
            pv = CuPtr{Cvoid}(UInt64(4096))
            dpv = Mooncake.Dual(pv, Mooncake.NoTangent())
            result_v = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(+, Mooncake.NoTangent()), dpv, dn
            )
            @test Mooncake.primal(result_v) == pv + 64
            @test Mooncake.tangent(result_v) isa Mooncake.NoTangent

            # ── rrule!! — differentiable T ────────────────────────────────────────────
            # Output tangent (fdata) must be the offset tangent pointer.
            dp32_co = Mooncake.CoDual(p32, CuPtr{Float32}(UInt64(4096)))
            dn_co = Mooncake.CoDual(Int64(64), Mooncake.NoFData())
            out, pb = _MooncakeCUDAExt.rrule!!(
                Mooncake.CoDual(+, Mooncake.NoFData()), dp32_co, dn_co
            )
            @test Mooncake.primal(out) == p32 + 64
            @test Mooncake.tangent(out) == CuPtr{Float32}(UInt64(4096)) + 64

            # ── rrule!! — non-differentiable T (Cvoid) ───────────────────────────────
            # Output fdata must be NoFData (not crash, not a stray pointer).
            dpv_co = Mooncake.CoDual(pv, Mooncake.NoFData())
            out_v, pb_v = _MooncakeCUDAExt.rrule!!(
                Mooncake.CoDual(+, Mooncake.NoFData()), dpv_co, dn_co
            )
            @test Mooncake.primal(out_v) == pv + 64
            @test Mooncake.tangent(out_v) isa Mooncake.NoFData
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
            # The rule simply calls the primal and returns Dual(nothing, NoTangent()) /
            # CoDual(nothing, NoFData()).
            fin = _ -> nothing
            arr = _rand(rng, Float32, 4)
            tarr = Mooncake.zero_tangent(arr)

            # frule!!: output is Dual(nothing, NoTangent()).
            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(Core.finalizer, Mooncake.NoTangent()),
                Mooncake.Dual(fin, Mooncake.NoTangent()),
                Mooncake.Dual(arr, tarr),
            )
            @test Mooncake.primal(result) === nothing
            @test Mooncake.tangent(result) isa Mooncake.NoTangent

            # rrule!!: output fdata is NoFData; pullback returns NoRData for all inputs.
            out, pb = _MooncakeCUDAExt.rrule!!(
                Mooncake.CoDual(Core.finalizer, Mooncake.NoFData()),
                Mooncake.CoDual(fin, Mooncake.NoFData()),
                Mooncake.CoDual(arr, tarr),
            )
            @test Mooncake.primal(out) === nothing
            @test Mooncake.tangent(out) isa Mooncake.NoFData
            @test all(x -> x isa Mooncake.NoRData, pb(Mooncake.NoRData()))
        end

        @testset "hasfieldcount frule!! / rrule!!" begin
            # hasfieldcount(T) returns Bool — no gradient path.
            # Verify the primal result is forwarded and tangent is always NoTangent/NoFData.
            for T in (ComplexF64, Float32, Any)
                expected = hasfieldcount(T)

                result = _MooncakeCUDAExt.frule!!(
                    Mooncake.Dual(hasfieldcount, Mooncake.NoTangent()),
                    Mooncake.Dual(T, Mooncake.NoTangent()),
                )
                @test Mooncake.primal(result) === expected
                @test Mooncake.tangent(result) isa Mooncake.NoTangent

                out, pb = _MooncakeCUDAExt.rrule!!(
                    Mooncake.CoDual(hasfieldcount, Mooncake.NoFData()),
                    Mooncake.CoDual(T, Mooncake.NoFData()),
                )
                @test Mooncake.primal(out) === expected
                @test Mooncake.tangent(out) isa Mooncake.NoFData
                @test all(x -> x isa Mooncake.NoRData, pb(Mooncake.NoRData()))
            end
        end

        @testset "copy(::CuDataRef) frule!! / rrule!!" begin
            # copy(::DataRef) increments the refcount and returns a new handle to the
            # same GPU buffer.  frule!!: both primal and tangent DataRefs are copied.
            # rrule!!: same; pullback is NoPullback (no numerical gradient through DataRef).
            ref = getfield(_rand(rng, Float32, 16), :data)
            tref = copy(ref)

            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(copy, Mooncake.NoTangent()), Mooncake.Dual(ref, tref)
            )
            @test Mooncake.primal(result) isa typeof(ref)
            @test Mooncake.primal(result) !== ref    # must be a new handle, not the same object
            @test Mooncake.tangent(result) isa typeof(tref)
            @test Mooncake.tangent(result) !== tref  # Mooncake.tangent DataRef also copied

            out, pb = _MooncakeCUDAExt.rrule!!(
                Mooncake.CoDual(copy, Mooncake.NoFData()), Mooncake.CoDual(ref, tref)
            )
            @test Mooncake.primal(out) isa typeof(ref)
            @test Mooncake.primal(out) !== ref
            @test Mooncake.tangent(out) isa typeof(tref)
            @test Mooncake.tangent(out) !== tref
            @test all(x -> x isa Mooncake.NoRData, pb(Mooncake.NoRData()))
        end

        @testset "unsafe_free! frule!! / rrule!!" begin
            # unsafe_free! releases GPU memory early; pure side-effect, no gradient.
            # frule!!: returns Dual(nothing, NoTangent()); both primal and tangent freed.
            # rrule!!: returns CoDual(nothing, NoFData()) — regression test for the bug
            #          where NoTangent() was incorrectly used in the fdata slot.
            arr = _rand(rng, Float32, 4)
            tarr = Mooncake.zero_tangent(arr)

            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(unsafe_free!, Mooncake.NoTangent()), Mooncake.Dual(arr, tarr)
            )
            @test Mooncake.primal(result) === nothing
            @test Mooncake.tangent(result) isa Mooncake.NoTangent

            arr2 = _rand(rng, Float32, 4)
            tarr2 = Mooncake.zero_tangent(arr2)
            out, pb = _MooncakeCUDAExt.rrule!!(
                Mooncake.CoDual(unsafe_free!, Mooncake.NoFData()),
                Mooncake.CoDual(arr2, tarr2),
            )
            @test Mooncake.primal(out) === nothing
            @test Mooncake.tangent(out) isa Mooncake.NoFData  # must be Mooncake.NoFData, not Mooncake.NoTangent
            @test all(x -> x isa Mooncake.NoRData, pb(Mooncake.NoRData()))
        end

        # unsafe_convert dispatch — invariant type-parameter regression test.
        #
        # Issue: the original rules were declared as frule!!(x::Dual{CuArray{T},CuArray{T}})
        # and rrule!!(x::CoDual{CuArray{T},CuArray{T}}).  Julia's type parameters are
        # invariant, so a concrete CuArray{Float32,2,DeviceMemory} does NOT match the
        # UnionAll CuArray{Float32} as a type parameter, and dispatch silently misses.
        # Fix: use Dual{X,X} / CoDual{X,X} where X<:CuArray{T} to push subtyping into
        # the where-clause, allowing X to be unified with the fully-specified concrete type.
        @testset "unsafe_convert frule!! / rrule!! dispatch on concrete CuArray" begin
            arr = _rand(rng, Float32, 4, 4)  # CuArray{Float32,2,DeviceMemory} — 3 type params
            tarr = Mooncake.zero_tangent(arr)

            # frule!!: both primal and tangent pointers returned.
            result = _MooncakeCUDAExt.frule!!(
                Mooncake.Dual(unsafe_convert, Mooncake.NoTangent()),
                Mooncake.Dual(CuPtr{Float32}, Mooncake.NoTangent()),
                Mooncake.Dual(arr, tarr),
            )
            @test Mooncake.primal(result) isa CuPtr{Float32}
            @test Mooncake.tangent(result) isa CuPtr{Float32}

            # rrule!!: output is CoDual of primal and tangent pointers; pullback is NoPullback.
            arr2 = _rand(rng, Float32, 4, 4)
            tarr2 = Mooncake.zero_tangent(arr2)
            out, pb = _MooncakeCUDAExt.rrule!!(
                Mooncake.CoDual(unsafe_convert, Mooncake.NoFData()),
                Mooncake.CoDual(CuPtr{Float32}, Mooncake.NoFData()),
                Mooncake.CoDual(arr2, tarr2),
            )
            @test Mooncake.primal(out) isa CuPtr{Float32}
            @test Mooncake.tangent(out) isa CuPtr{Float32}
            @test all(x -> x isa Mooncake.NoRData, pb(Mooncake.NoRData()))
        end

        # _premat_nondiff_args: structural invariant test.
        #
        # Issue: Base.Broadcast.flatten composes nested Broadcasted nodes into a single
        # function object.  When an inner broadcast uses a non-differentiable function
        # such as Type{Float64} (e.g. `Float64.(bool_array)`), flatten embeds that type
        # into the composed function's closure.  Type{Float64} is not isbits, so passing
        # it to a GPU kernel fails with "non-bitstype argument" on Julia 1.10 (on Julia
        # 1.12 a separate all-NoTangent collapse in tangent_type happens to hide the bug).
        #
        # Fix: _premat_nondiff_args walks the primal Broadcasted tree before flatten and
        # replaces any sub-Broadcasted whose total Dual-slot count (_total_bcast_dof) is
        # zero with its already-materialized plain CuArray value.  After that replacement
        # flatten only sees plain arrays as leaves, and its composed function is isbits.
        @testset "_premat_nondiff_args makes flat_bc.f isbits" begin
            x = CUDA.rand(Float64, 4)
            bool_mask = x .> 0  # CuArray{Bool}

            # Construct `x .* Float64.(bool_mask)` as a nested Broadcasted tree.
            # The inner node captures Type{Float64} which is NOT isbits.
            inner = Base.Broadcast.broadcasted(Float64, bool_mask)
            outer = Base.Broadcast.broadcasted(*, x, inner)

            # After _premat_nondiff_args: inner node (dof==0) replaced by plain CuArray.
            fixed = _MooncakeCUDAExt._premat_nondiff_args(outer)
            @test !(fixed.args[2] isa Base.Broadcast.Broadcasted)
            flat_fixed = Base.Broadcast.flatten(fixed)
            @test isbitstype(typeof(flat_fixed.f))
        end

        @testset "nested GPU broadcast gradients keep tree alignment" begin
            x = CuArray(randn(rng, 4))
            y = CuArray(randn(rng, 4))
            rule = Mooncake.build_rrule(_bcast_nested_sin_add, x, y)
            val, grads = value_and_gradient!!(
                rule, _bcast_nested_sin_add, x, y; friendly_tangents=true
            )
            @test val ≈ sum(Array(sin.(x .+ y)))
            expected = Array(cos.(x .+ y))
            @test Array(grads[2]) ≈ expected
            @test Array(grads[3]) ≈ expected
        end

        @testset "differentiable nested float casts still propagate gradients" begin
            x = CuArray(randn(rng, Float32, 4))
            rule = Mooncake.build_rrule(_bcast_nested_float_cast_sin, x)
            val, grads = value_and_gradient!!(
                rule, _bcast_nested_float_cast_sin, x; friendly_tangents=true
            )
            expected_val = sum(sin.(Float64.(Array(x))))
            expected_grad = Float32.(cos.(Float64.(Array(x))))
            @test val ≈ expected_val
            @test Array(grads[2]) ≈ expected_grad
        end

        @testset "zero-DOF nested broadcast scalar gradients reconstruct on reverse pass" begin
            x = CuArray(randn(rng, 4))
            c = 2.5
            b = CuArray(Float64[-2.0, 1.0, -3.0, 4.0])
            mask = Float64.(Array(b) .> 0)
            rule = Mooncake.build_rrule(_bcast_zero_dof_nested, x, c, b)
            val, grads = value_and_gradient!!(
                rule, _bcast_zero_dof_nested, x, c, b; friendly_tangents=true
            )
            @test val ≈ sum(Array(x) .+ c .* mask)
            @test Array(grads[2]) ≈ ones(length(mask))
            @test grads[3] ≈ sum(mask)
        end

        @testset "all-scalar nested broadcast leaves keep scalar gradients" begin
            # Regression for the differentiable-scalar guard in _premat_nondiff_args:
            # (s .+ 1.0) is a nested Broadcasted with NoFData fdata (scalar gradients live
            # in rdata) but one differentiable DOF; collapsing it to a constant dropped s
            # from flat_pargs, crashing the reverse pass with UndefRefError.
            x = CuArray(randn(rng, 4))
            s = 0.75
            rule = Mooncake.build_rrule(_bcast_all_scalar_leaf, x, s)
            val, grads = value_and_gradient!!(rule, _bcast_all_scalar_leaf, x, s)
            @test val ≈ sum(Array(x) .* (s + 1.0))
            @test Array(grads[2]) ≈ fill(s + 1.0, 4)
            @test grads[3] ≈ sum(Array(x))
        end

        @testset "in-place zero-DOF nested broadcasts reconstruct scalar gradients" begin
            dest = CuArray(zeros(4))
            x = CuArray(randn(rng, 4))
            c = -1.25
            b = CuArray(Float64[-2.0, 1.0, -3.0, 4.0])
            mask = Float64.(Array(b) .> 0)
            rule = Mooncake.build_rrule(_inplace_zero_dof_nested!, dest, x, c, b)
            val, grads = value_and_gradient!!(
                rule, _inplace_zero_dof_nested!, dest, x, c, b; friendly_tangents=true
            )
            @test val ≈ sum(Array(x) .+ c .* mask)
            @test Array(grads[3]) ≈ ones(length(mask))
            @test grads[4] ≈ sum(mask)
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
                    Mooncake.build_rrule(f, x, y), f, x, y
                )
            end

            # Scalar getindex/setindex! on CuArray — throw to prevent silent scalar GPU ops.
            @testset "scalar getindex CuArray not differentiable" begin
                f = x -> x[1]
                x = _rand(rng, Float32, 4)
                @test_throws r"scalar indexing of CuArray is not differentiable" value_and_gradient!!(
                    Mooncake.build_rrule(f, x), f, x
                )
            end
            @testset "scalar setindex! CuArray not differentiable" begin
                f = x -> (x[1]=0.0f0; sum(x))
                x = _rand(rng, Float32, 4)
                @test_throws r"scalar indexing of CuArray is not differentiable" value_and_gradient!!(
                    Mooncake.build_rrule(f, x), f, x
                )
            end

            # accumulate with unsupported op — catch-all rule throws ArgumentError.
            @testset "accumulate non-+ CuArray not differentiable" begin
                f = x -> sum(accumulate(*, x))
                x = _rand(rng, Float32, 4)
                @test_throws r"accumulate on CuArray only supports op=\+" value_and_gradient!!(
                    Mooncake.build_rrule(f, x), f, x
                )
            end

            # Keyword and mapped-form reductions without real rules fall back to
            # friendly errors instead of dying inside cufunction (#1273).
            @testset "keyword/mapped reduction fallbacks throw ArgumentError" begin
                x = _rand(rng, Float32, 4)
                dx = Mooncake.zero_tangent(x)
                # One entry per @eval claim family; other functions in the same
                # loop share the generated code verbatim.
                for f in (
                    z -> sum(sum(abs2, z; dims=1)),     # kwcall sum(f, x; dims)
                    z -> maximum(abs, z),               # mapped, positional
                    z -> sum(maximum(abs, z; dims=1)),  # mapped, kwcall
                    z -> sum(sort(z; rev=true)),        # catch-all, kwcall
                    z -> sum(sort(z)),                  # catch-all, positional
                )
                    @test_throws r"not yet differentiable" value_and_gradient!!(
                        Mooncake.build_rrule(f, x), f, x
                    )
                    @test_throws r"not yet differentiable" Mooncake.value_and_derivative!!(
                        Mooncake.build_frule(f, x),
                        Mooncake.Dual(f, Mooncake.NoTangent()),
                        Mooncake.Dual(x, dx),
                    )
                end
            end

            # Complex slice-adjoint-matvec: cu(x[:, 1])' * cy — cu() downcasts ComplexF64
            # to ComplexF32, producing a type mismatch with cy::CuMatrix{ComplexF64}.
            # The generic_matvecmul! frule/rrule detects the mismatch before any cuBLAS call.
            @testset "complex slice-adjoint-matvec type mismatch" begin
                f = _cu_cx_slice_adj_mul
                x = _host_rand(rng, ComplexF64, 3, 3)
                cy = _rand(rng, ComplexF64, 3, 3)
                @test_throws r"GPU gemv with mismatched element types" value_and_gradient!!(
                    Mooncake.build_rrule(f, x, cy), f, x, cy
                )
            end

            @testset "mixed GPU/CPU cat guards" begin
                # One unified Vararg{Union{AbstractArray,Number}} guard per function
                # covers array/scalar mixing at any arity/order, so each check below
                # (reachability, is_primitive, N-arg) targets a distinct property
                # instead of re-testing the same compiled method.
                gpu1 = _rand(rng, Float32, 4)
                gpu2 = _rand(rng, Float32, 4, 3)
                tgpu1 = Mooncake.zero_tangent(gpu1)
                tgpu2 = Mooncake.zero_tangent(gpu2)
                cpu_vec = _host_rand(rng, Float32, 4)
                tcpu_vec = zero(cpu_vec)
                cpu_mat = _host_rand(rng, Float32, 4, 2)
                tcpu_mat = zero(cpu_mat)
                gpu3 = _rand(rng, Float32, 3)
                tgpu3 = Mooncake.zero_tangent(gpu3)
                s = 1.0f0
                wc = Base.get_world_counter()

                @test_throws r"mix of GPU" value_and_gradient!!(
                    Mooncake.build_rrule(_vcat_cu_sum, gpu1, cpu_vec),
                    _vcat_cu_sum,
                    gpu1,
                    cpu_vec,
                )
                @test_throws r"mix of GPU" value_and_gradient!!(
                    Mooncake.build_rrule(_hcat_cu_sum, gpu2, cpu_mat),
                    _hcat_cu_sum,
                    gpu2,
                    cpu_mat,
                )
                @test_throws r"mix of GPU" value_and_gradient!!(
                    Mooncake.build_rrule(_cat_cu_sum(1), gpu1, s), _cat_cu_sum(1), gpu1, s
                )

                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{typeof(vcat),typeof(gpu1),typeof(gpu3)},
                    wc,
                )
                @test !Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{typeof(vcat),typeof(cpu_vec),typeof(cpu_vec)},
                    wc,
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{typeof(vcat),typeof(gpu1),typeof(cpu_vec)},
                    wc,
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{typeof(hcat),typeof(gpu2),typeof(gpu2)},
                    wc,
                )
                @test !Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{typeof(hcat),typeof(cpu_mat),typeof(cpu_mat)},
                    wc,
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{typeof(hcat),typeof(gpu2),typeof(cpu_mat)},
                    wc,
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{
                        typeof(Core.kwcall),
                        typeof((dims=1,)),
                        typeof(cat),
                        typeof(gpu1),
                        typeof(gpu3),
                    },
                    wc,
                )
                @test !Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{
                        typeof(Core.kwcall),
                        typeof((dims=1,)),
                        typeof(cat),
                        typeof(cpu_vec),
                        typeof(cpu_vec),
                    },
                    wc,
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx,
                    Mooncake.Mode,
                    Tuple{
                        typeof(Core.kwcall),
                        typeof((dims=1,)),
                        typeof(cat),
                        typeof(gpu1),
                        typeof(cpu_vec),
                    },
                    wc,
                )

                @test_throws r"mix of GPU" _MooncakeCUDAExt.frule!!(
                    Mooncake.Dual(vcat, Mooncake.NoTangent()),
                    Mooncake.Dual(gpu1, tgpu1),
                    Mooncake.Dual(s, zero(s)),
                )
                @test_throws r"mix of GPU" _MooncakeCUDAExt.rrule!!(
                    Mooncake.CoDual(hcat, Mooncake.NoFData()),
                    Mooncake.CoDual(gpu2, tgpu2),
                    Mooncake.CoDual(cpu_mat, tcpu_mat),
                )
                @test_throws r"mix of GPU" _MooncakeCUDAExt.frule!!(
                    Mooncake.Dual(Core.kwcall, Mooncake.NoTangent()),
                    Mooncake.Dual((dims=1,), Mooncake.NoTangent()),
                    Mooncake.Dual(cat, Mooncake.NoTangent()),
                    Mooncake.Dual(gpu1, tgpu1),
                    Mooncake.Dual(cpu_vec, tcpu_vec),
                )

                # N-arg: CPU array sandwiched between two GPU arrays.
                @test_throws r"mix of GPU" _MooncakeCUDAExt.rrule!!(
                    Mooncake.CoDual(vcat, Mooncake.NoFData()),
                    Mooncake.CoDual(gpu1, tgpu1),
                    Mooncake.CoDual(cpu_vec, tcpu_vec),
                    Mooncake.CoDual(gpu3, tgpu3),
                )
            end

            @testset "regression: pure-CPU splatted vcat/hcat/cat with CUDA loaded" begin
                # Regression: `_is_primitive` used to be `@generated`, which crashed or
                # misclassified splatted calls (unknown arity -> `Vararg` type) as needing
                # the guard even with no GPU arrays involved.
                _splat_vcat_sum(xs) = sum(vcat(xs...))
                _splat_hcat_sum(xs) = sum(hcat(xs...))
                _splat_cat_sum(xs) = sum(cat(xs...; dims=1))
                xs = [_host_rand(rng, 3) for _ in 1:3]
                for f in (_splat_vcat_sum, _splat_hcat_sum, _splat_cat_sum)
                    val, (_, dxs) = value_and_gradient!!(Mooncake.build_rrule(f, xs), f, xs)
                    @test val ≈ f(xs)
                    @test length(dxs) == length(xs)
                end
            end

            @testset "Float16 support" begin
                # Analytic gradient (ones), not finite differences: unreliable at Float16.
                x16 = _rand(rng, Float16, 4)
                y16 = _rand(rng, Float16, 4)
                val, (_, dx, dy) = value_and_gradient!!(
                    Mooncake.build_rrule(_vcat_cu_sum, x16, y16), _vcat_cu_sum, x16, y16
                )
                @test val ≈ sum(vcat(x16, y16))
                @test all(==(one(Float16)), Array(dx))
                @test all(==(one(Float16)), Array(dy))

                # Float16 SubArrays are excluded from CuMaybeWrappedArray. A strided view
                # stays a genuine SubArray (unlike the contiguous 1-D view above, which
                # CUDA.jl collapses to a plain CuArray) and the N-arg mixed-device guard
                # does not count it as GPU either, so it errors with "mix of GPU" rather
                # than reaching the interpreter's untraceable `cufunction` try/finally.
                x16_mat = _rand(rng, Float16, 4, 3)
                y16_mat = _rand(rng, Float16, 2, 3)
                f_view(x, y) = sum(vcat(view(x, 1:2, :), y))
                @test_throws r"mix of GPU" value_and_gradient!!(
                    Mooncake.build_rrule(f_view, x16_mat, y16_mat), f_view, x16_mat, y16_mat
                )
            end

            @testset "_unwrap_cat_dim rejects unsupported dims types" begin
                # dims must be an Integer, Val{N}, or Tuple{Vararg{Integer}}.
                @test_throws ArgumentError _MooncakeCUDAExt._unwrap_cat_dim(1.0)
                @test_throws ArgumentError _MooncakeCUDAExt._unwrap_cat_dim((1, 2.0))
            end
        end

        @testset "Statistics.varm GPU rule" begin
            # varm(x, m; dims, corrected): used by LayerNorm / GroupNorm / InstanceNorm via
            # LuxLib.Impl.mean_var → var → varm. is_primitive=false: the wrapper is not one.
            @testset "dims=1, corrected=false (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 1, 3)
                test_rule(
                    StableRNG(1), _varm_sum_d1, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=2, corrected=true (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 4, 1)
                test_rule(
                    StableRNG(2), _varm_sum_d2, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=1, corrected=false (Float64)" begin
                x = _rand(rng, Float64, 4, 3)
                m = _rand(rng, Float64, 1, 3)
                test_rule(
                    StableRNG(3), _varm_sum_d1, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "no dims, scalar mean (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                m_scalar = randn(StableRNG(8), Float32)
                test_rule(
                    StableRNG(8),
                    _varm_nodims_scalar,
                    x,
                    m_scalar;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "dims=(1,2) tuple, corrected=false (Float32)" begin
                x = _rand(rng, Float32, 4, 3, 2)
                m = _rand(rng, Float32, 1, 1, 2)
                test_rule(
                    StableRNG(14),
                    _varm_sum_dtuple,
                    x,
                    m;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "dims=1:2 UnitRange, corrected=false (Float32)" begin
                x = _rand(rng, Float32, 4, 3, 2)
                m = _rand(rng, Float32, 1, 1, 2)
                test_rule(
                    StableRNG(16),
                    _varm_sum_drange,
                    x,
                    m;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "dims=:, array-shaped mean, corrected=false (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 1, 1)
                test_rule(
                    StableRNG(20),
                    _varm_sum_dcolon_arraymean,
                    x,
                    m;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "repeated dims=(1,1), corrected=false (Float32)" begin
                # Regression: denominator must count dim 1 once, not size(x,1)^2.
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 1, 3)
                test_rule(
                    StableRNG(21), _varm_sum_ddup, x, m; is_primitive=false, perf_flag=:none
                )
                @test _varm_sum_ddup(x, m) ≈ _varm_sum_d1(x, m)
            end
            @testset "bare 2-arg spelling, no keywords (Float32)" begin
                # Regression: `varm(x, m)` with no keyword syntax at all bypasses
                # Core.kwcall entirely; without the dedicated bare-call primitive it
                # falls through to Statistics' captured-mean mapreduce, which Mooncake
                # can't trace on GPU.
                x = _rand(rng, Float32, 4, 3)
                m_scalar = randn(StableRNG(22), Float32)
                test_rule(
                    StableRNG(22),
                    _varm_bare_nodims_scalar,
                    x,
                    m_scalar;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "bare 2-arg spelling, no keywords (ComplexF32)" begin
                x = _rand(rng, ComplexF32, 4, 3)
                m_cx = randn(StableRNG(23), ComplexF32)
                test_rule(
                    StableRNG(23),
                    _varm_bare_nodims_scalar,
                    x,
                    m_cx;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "mixed real/complex scalar mean" begin
                # Regression: a real-valued rdata/fdata slot must project a mismatched
                # complex intermediate onto its real part rather than throwing
                # InexactError on the implicit conversion.
                @testset "complex x, real m" begin
                    x = _rand(rng, ComplexF32, 4, 3)
                    m = randn(StableRNG(24), Float32)
                    test_rule(
                        StableRNG(24),
                        _varm_nodims_scalar,
                        x,
                        m;
                        is_primitive=false,
                        perf_flag=:none,
                    )
                end
                @testset "real x, complex m" begin
                    x = _rand(rng, Float32, 4, 3)
                    m = randn(StableRNG(25), ComplexF32)
                    test_rule(
                        StableRNG(25),
                        _varm_nodims_scalar,
                        x,
                        m;
                        is_primitive=false,
                        perf_flag=:none,
                    )
                end
            end
            # No complex/mixed array-mean tests: GPUArrays' accelerated varm needs Real
            # eltypes on both arguments, and anything else falls to a generic
            # scalar-indexing path that cannot run on GPU, so there is no ground truth.
            # The array-m rules are restricted to real eltypes to match, since a wider
            # signature would make AD succeed where the primal throws; the norm layers only
            # ever pass x and its own real mean.
            @testset "m broadcast against x: full-shape m, dims=1 (Float32)" begin
                # Regression: the primal broadcasts m against x, so m's gradient must
                # reduce over exactly the dims m is broadcast along (none, here) rather than
                # over `dims`, which used to column-sum it.
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(26), _varm_sum_d1, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "m broadcast against x: (1,1) m, dims=1 (Float32)" begin
                # Regression: m broadcast along a dim outside `dims` (dim 2 here) used
                # to throw DimensionMismatch in the pullback.
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 1, 1)
                test_rule(
                    StableRNG(27), _varm_sum_d1, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "x broadcast against m: (1,3) x, full m, dims=1 (Float32)" begin
                # Regression: x can be the broadcast-expanded operand too, so the pullback
                # must unbroadcast to BOTH shapes (dx used to throw DimensionMismatch here).
                x = _rand(rng, Float32, 1, 3)
                m = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(29), _varm_sum_d1, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "cross-precision arrays: Float32 x, Float64 m, dims=1" begin
                # Claimed by the array-m signature (no precision tie, unlike scalar-m
                # below): GPUArrays' method has no n==0 type bifurcation, so the
                # output type is concrete and mixed real precisions just work.
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float64, 1, 3)
                test_rule(
                    StableRNG(30), _varm_sum_d1, x, m; is_primitive=false, perf_flag=:none
                )
            end
            @testset "empty dims=() collection (Float32)" begin
                # Regression: the primal reduces over nothing and returns the full
                # array; the rule's denominator must mirror _mean_denom's `init=1`
                # instead of throwing on a prod over an empty collection.
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 1, 3)
                test_rule(
                    StableRNG(31),
                    _varm_sum_dempty,
                    x,
                    m;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
            @testset "kwarg sets the primal rejects throw under AD" begin
                x = _rand(rng, Float32, 4, 3)
                m = _rand(rng, Float32, 1, 3)
                rule = Mooncake.build_rrule(_varm_arraymean_missing_dims, x, m)
                @test_throws UndefKeywordError Mooncake.value_and_gradient!!(
                    rule, _varm_arraymean_missing_dims, x, m
                )
                m_scalar = randn(StableRNG(28), Float32)
                rule2 = Mooncake.build_rrule(_varm_scalarmean_stray_dims, x, m_scalar)
                @test_throws MethodError Mooncake.value_and_gradient!!(
                    rule2, _varm_scalarmean_stray_dims, x, m_scalar
                )
            end
            @testset "empty array, scalar mean, corrected=true (Float32)" begin
                # Regression: an empty input must give NaN, not 0/(0-1) = -0.0, matching the
                # guard in Statistics._varm(A, m, corrected, ::Colon) — scalar m is not
                # overridden by GPUArrays. The unguarded value is a constant 0 regardless of
                # m, so the gradient must be 0, not NaN.
                x = CuArray(Float32[])
                m = 0.0f0
                @test isnan(_varm_nodims_scalar(x, m))
                rule = Mooncake.build_rrule(_varm_nodims_scalar, x, m)
                out, (_, dx, dm) = Mooncake.value_and_gradient!!(
                    rule, _varm_nodims_scalar, x, m
                )
                @test isnan(out)
                @test isempty(dx)
                @test dm == 0.0f0
            end
            @testset "empty array, array mean, dims=: (Float32)" begin
                # Regression: GPUArrays' array-m `varm` has no empty-input guard and gives
                # 0, not NaN; the scalar-mean NaN guard above must not be applied here.
                x = CuArray(Float32[])
                m = CuArray(Float32[])
                @test _varm_sum_dcolon_arraymean(x, m) == 0.0f0
                rule = Mooncake.build_rrule(_varm_sum_dcolon_arraymean, x, m)
                out, _ = Mooncake.value_and_gradient!!(
                    rule, _varm_sum_dcolon_arraymean, x, m
                )
                @test out == 0.0f0
            end
            @testset "empty x, non-empty m, dims=:, corrected=false (Float32)" begin
                # Regression: coeff = 2/(0-0) = Inf times a pre-reduced sum_diff gave
                # Inf * 0 = NaN in dm; forming the elementwise gradient first reduces to 0.
                x = CuArray(Float32[])
                m = CuArray(Float32[0.0f0])
                rule = Mooncake.build_rrule(_varm_sum_dcolon_arraymean, x, m)
                out, (_, _, dm) = Mooncake.value_and_gradient!!(
                    rule, _varm_sum_dcolon_arraymean, x, m
                )
                @test out == 0.0f0
                @test Array(dm) == [0.0f0]
            end
            @testset "scalar m must match x's underlying precision" begin
                # Mixed precision makes Statistics' scalar-m varm infer
                # Union{Float32,Float64} (its n==0 branch types σ² off x alone, its main
                # branch promotes with m), which Mooncake's rule builder cannot handle
                # (zero(::Type{Union{...}})).
                x = _rand(rng, Float32, 4, 3)
                world = Base.get_world_counter()
                kwsig = @NamedTuple{corrected::Bool}
                mixed = Tuple{typeof(Core.kwcall),kwsig,typeof(varm),typeof(x),Float64}
                same = Tuple{typeof(Core.kwcall),kwsig,typeof(varm),typeof(x),Float32}
                @test !Mooncake.is_primitive(
                    Mooncake.MinimalCtx, Mooncake.ReverseMode, mixed, world
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx, Mooncake.ReverseMode, same, world
                )
                @test !Mooncake.is_primitive(
                    Mooncake.MinimalCtx, Mooncake.ForwardMode, mixed, world
                )
                @test Mooncake.is_primitive(
                    Mooncake.MinimalCtx, Mooncake.ForwardMode, same, world
                )
            end
            @testset "empty x forward tangent is the zero map" begin
                # frule!! counterpart of the empty test above: the primal is a constant
                # NaN at n==0, so the JVP must be 0, not the divide-after 0/0 = NaN.
                x = CuArray(Float32[])
                d = Mooncake.frule!!(
                    Mooncake.Dual(Core.kwcall, Mooncake.NoTangent()),
                    Mooncake.Dual((; corrected=true), Mooncake.NoTangent()),
                    Mooncake.Dual(varm, Mooncake.NoTangent()),
                    Mooncake.Dual(x, Mooncake.zero_tangent(x)),
                    Mooncake.Dual(0.0f0, 0.0f0),
                )
                @test isnan(Mooncake.primal(d))
                @test Mooncake.tangent(d) === 0.0f0
            end
            @testset "Float16 dims=1 avoids overflow on large magnitudes" begin
                # Regression: summing raw squares before dividing (rather than scaling
                # by 1/(n-corrected) before reducing, as GPUArrays does) overflows Inf
                # for representable Float16 inputs well before the true variance does.
                x = CuArray(fill(Float16(1000), 200, 1))
                m = CuArray(fill(Float16(999), 200, 1))
                out = only(Array(varm(x, m; dims=1, corrected=false)))
                @test isfinite(out)
                @test out == Float16(1)
            end
            @testset "Float16 large n keeps λ finite in gradients" begin
                # Regression: `one(T) / n` converts n to Float16 first, so at n > 65504 λ
                # became 1/Inf16 = 0 and every gradient collapsed to zero, while the primal
                # (inverting in Float64 first, as GPUArrays does) stayed healthy.
                x = CUDA.ones(Float16, 70000)
                m = CUDA.zeros(Float16, 1)
                rule = Mooncake.build_rrule(_varm_sum_d1, x, m)
                out, (_, gx, _) = Mooncake.value_and_gradient!!(rule, _varm_sum_d1, x, m)
                @test isfinite(out)
                @test Array(gx)[1] == 2 * Float16(inv(70000))
                # Pin the frule's λ arithmetic too (the rrule path above does not
                # exercise it): with unit x-tangents the JVP is 2λ·n ≈ 2, whereas a
                # `one(T)/n` λ would give exactly 0.
                d = Mooncake.frule!!(
                    Mooncake.Dual(Core.kwcall, Mooncake.NoTangent()),
                    Mooncake.Dual((; dims=1, corrected=false), Mooncake.NoTangent()),
                    Mooncake.Dual(varm, Mooncake.NoTangent()),
                    Mooncake.Dual(x, CUDA.ones(Float16, 70000)),
                    Mooncake.Dual(m, CUDA.zeros(Float16, 1)),
                )
                @test only(Array(Mooncake.tangent(d))) > Float16(1.5)
            end
            @testset "scalar m: Float16 huge n matches the generic primal" begin
                # Unlike the GPUArrays array-m method above, Statistics' scalar-m varm
                # divides by the Int denominator in Float16, so at n > 65504 it promotes
                # to Inf16 and both primal and CPU-traced gradient are exactly 0; the CUDA
                # rule must not substitute healthier arithmetic on one backend only.
                x = CuArray(vcat(ones(Float16, 1), zeros(Float16, 69999)))
                m = Float16(0)
                @test _varm_nodims_scalar(x, m) == Float16(0)
                rule = Mooncake.build_rrule(_varm_nodims_scalar, x, m)
                out, (_, gx, gm) = Mooncake.value_and_gradient!!(
                    rule, _varm_nodims_scalar, x, m
                )
                @test out == Float16(0)
                @test all(iszero, Array(gx))
                @test iszero(gm)
                # Pin the frule's divide-after arithmetic too: the residual-weighted sum
                # is finite (2 · 1 · 0.5 = 1) and the Inf16-promoted denominator zeroes the
                # quotient; a λ-prescaled form would give ≈1.4e-5 instead.
                d = Mooncake.frule!!(
                    Mooncake.Dual(Core.kwcall, Mooncake.NoTangent()),
                    Mooncake.Dual((; corrected=true), Mooncake.NoTangent()),
                    Mooncake.Dual(varm, Mooncake.NoTangent()),
                    Mooncake.Dual(x, CUDA.fill(Float16(0.5), 70000)),
                    Mooncake.Dual(m, Float16(0)),
                )
                @test Mooncake.tangent(d) === Float16(0)
            end
            @testset "scalar m: Float16 residual overflow keeps dm finite" begin
                # Regression: m's cotangent used a precomputed raw sum(diff), which
                # overflows Float16 to Inf here (70000 × 0.94 > 65504) while the
                # dσ²/denom coefficient is zeroed by its Inf16-promoted denominator:
                # dm = 0 * Inf = NaN. Scaling elementwise before reducing (as in the
                # array-m pullback and the CPU-traced generic method) gives the
                # correct 0.
                x = CuArray(fill(Float16(0.94), 70000))
                m = Float16(0)
                @test _varm_nodims_scalar(x, m) == Float16(0)
                rule = Mooncake.build_rrule(_varm_nodims_scalar, x, m)
                out, (_, gx, gm) = Mooncake.value_and_gradient!!(
                    rule, _varm_nodims_scalar, x, m
                )
                @test out == Float16(0)
                @test iszero(gm)
                @test all(iszero, Array(gx))
            end
        end

        @testset "Statistics.mean GPU rule" begin
            # GPUArrays._mean calls sum(Fix1(*,λ), x; dims), which Mooncake cannot trace;
            # the rule calls it natively for the value and hand-rolls the derivative.
            @testset "dims=1 (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(4), _mean_sum_d1, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=2 (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(5), _mean_sum_d2, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=1 (Float64)" begin
                x = _rand(rng, Float64, 4, 3)
                test_rule(
                    StableRNG(6), _mean_sum_d1, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=: scalar output (Float32)" begin
                x = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(9), _mean_nodims, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=(1,2) tuple (Float32)" begin
                x = _rand(rng, Float32, 4, 3, 2)
                test_rule(
                    StableRNG(15), _mean_sum_dtuple, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=1:2 UnitRange (Float32)" begin
                x = _rand(rng, Float32, 4, 3, 2)
                test_rule(
                    StableRNG(17), _mean_sum_drange, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "repeated dims=(1,1) (Float32)" begin
                # Regression: denominator must count dim 1 once, not size(x,1)^2.
                x = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(18), _mean_sum_ddup, x; is_primitive=false, perf_flag=:none
                )
                @test _mean_sum_ddup(x) ≈ _mean_sum_d1(x)
            end
            @testset "empty dims=() collection (Float32)" begin
                # Regression: same `init=1` denominator mirror as the varm test above.
                x = _rand(rng, Float32, 4, 3)
                test_rule(
                    StableRNG(32), _mean_sum_dempty, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "empty array, dims=: gives NaN (Float32)" begin
                # Regression: the primal is sum/length = 0/0 = NaN; the rule takes its
                # value from the real function, so it must agree (a hand-rolled
                # sum(λ .* x) sums zero terms and gives 0 instead).
                x = CuArray(Float32[])
                @test isnan(_mean_nodims(x))
                rule = Mooncake.build_rrule(_mean_nodims, x)
                out, (_, dx) = Mooncake.value_and_gradient!!(rule, _mean_nodims, x)
                @test isnan(out)
                @test isempty(dx)
                # Forward-mode counterpart: the primal is a constant NaN, so the JVP
                # is the zero map — not the 0/0 = NaN of the divide-after formula.
                d = Mooncake.frule!!(
                    Mooncake.Dual(Core.kwcall, Mooncake.NoTangent()),
                    Mooncake.Dual((; dims=:), Mooncake.NoTangent()),
                    Mooncake.Dual(mean, Mooncake.NoTangent()),
                    Mooncake.Dual(x, Mooncake.zero_tangent(x)),
                )
                @test isnan(Mooncake.primal(d))
                @test Mooncake.tangent(d) === 0.0f0
            end
            @testset "dims=: gradient matches bare mean (Float16, n > 65504)" begin
                # Regression: the Colon primal divides AFTER summing, so its true
                # (as-implemented) derivative is dμ / Float16(n) — 0 at this scale.
                # The λ-prescaled form gave 1.43e-5 instead, so mean(x) and
                # mean(x; dims=:) disagreed under AD.
                x = CUDA.ones(Float16, 70000)
                rule_b = Mooncake.build_rrule(_mean_bare, x)
                _, (_, g_bare) = Mooncake.value_and_gradient!!(rule_b, _mean_bare, x)
                rule_c = Mooncake.build_rrule(_mean_nodims, x)
                _, (_, g_colon) = Mooncake.value_and_gradient!!(rule_c, _mean_nodims, x)
                @test Array(g_colon) == Array(g_bare)
                @test all(iszero, Array(g_colon))
                # Pin the frule's divide-after arithmetic too: sum(0.5-tangents) is finite
                # (35000 < 65504) and / Float16(70000) = / Inf16 gives 0; λ-prescaled ≈ 0.5.
                d = Mooncake.frule!!(
                    Mooncake.Dual(Core.kwcall, Mooncake.NoTangent()),
                    Mooncake.Dual((; dims=:), Mooncake.NoTangent()),
                    Mooncake.Dual(mean, Mooncake.NoTangent()),
                    Mooncake.Dual(x, CUDA.fill(Float16(0.5), 70000)),
                )
                @test Mooncake.tangent(d) === Float16(0)
            end
        end

        @testset "Statistics.varm GPU rule (complex)" begin
            # Complex scalar m of matching precision: σ² = sum(abs2(x-m))/n is always real.
            @testset "no dims, scalar mean (ComplexF32)" begin
                x = _rand(rng, ComplexF32, 4, 3)
                m_cx = randn(StableRNG(10), ComplexF32)
                test_rule(
                    StableRNG(10),
                    _varm_nodims_scalar,
                    x,
                    m_cx;
                    is_primitive=false,
                    perf_flag=:none,
                )
            end
        end

        @testset "Statistics.mean GPU rule (complex)" begin
            @testset "dims=: scalar output (ComplexF32)" begin
                x = _rand(rng, ComplexF32, 4, 3)
                test_rule(
                    StableRNG(12), _mean_cx_nodims, x; is_primitive=false, perf_flag=:none
                )
            end
            @testset "dims=1 (ComplexF32)" begin
                x = _rand(rng, ComplexF32, 4, 3)
                test_rule(
                    StableRNG(13), _mean_cx_sum_d1, x; is_primitive=false, perf_flag=:none
                )
            end
        end

        @testset "keyword sum GPU rule (#1273)" begin
            # One case per rule code path: array/Colon output branch, real/complex
            # claim arm, and the output-typed tangent seed for a widening init.
            @testset "$name" for (seed, name, f, x) in [
                (40, "dims=1 (Float32)", _sum_kw_d1, _rand(rng, Float32, 4, 3)),
                (47, "dims=: scalar output", _sum_kw_nodims, _rand(rng, Float32, 4, 3)),
                (49, "dims=1 (ComplexF32)", _sum_kw_cx_d1, _rand(rng, ComplexF32, 4, 3)),
                (52, "init=0.0 widens", _sum_kw_init_wide, _rand(rng, Float32, 4, 3)),
            ]
                test_rule(StableRNG(seed), f, x; is_primitive=false, perf_flag=:none)
            end
            @testset "kwarg sets the primal rejects throw under AD" begin
                # The float-typed bad kwarg gives the frule a tangent NamedTuple
                # without an `init` field: reading `dkw.init` unguarded would
                # throw a field error instead of the primal's MethodError.
                x = _rand(rng, Float32, 4, 3)
                @test_throws MethodError Mooncake.value_and_derivative!!(
                    Mooncake.build_frule(_sum_kw_badkw, x),
                    Mooncake.Dual(_sum_kw_badkw, Mooncake.NoTangent()),
                    Mooncake.Dual(x, Mooncake.zero_tangent(x)),
                )
            end
            @testset "init is a non-differentiated constant" begin
                # Not a test_rule case: finite differences see the backend's init
                # folding (nonzero ∂/∂init) while the rule deliberately treats
                # init as a constant. debug_mode verifies the kw rdata type.
                a, x = 0.0f0, _rand(rng, Float32, 4, 3)
                rule = Mooncake.build_rrule(_sum_kw_init_active, a, x; debug_mode=true)
                _, (_, da, _) = Mooncake.value_and_gradient!!(
                    rule, _sum_kw_init_active, a, x
                )
                @test da == 0.0f0
                frule = Mooncake.build_frule(_sum_kw_init_active, a, x)
                @test_throws r"init.*constant" Mooncake.value_and_derivative!!(
                    frule,
                    Mooncake.Dual(_sum_kw_init_active, Mooncake.NoTangent()),
                    Mooncake.Dual(a, 1.0f0),
                    Mooncake.Dual(x, Mooncake.zero_tangent(x)),
                )
            end
        end

        @testset "maximum/minimum GPU rules" begin
            # One case per rule path. The vector cases are not eltype padding:
            # GPUArrays returns a linear Int index when ndims == 1 and a
            # CartesianIndex otherwise, and comparing the two kinds is silently
            # false, which would give an all-zero gradient.
            @testset "$name" for (seed, name, f, x) in [
                (60, "maximum(x) vector", _max_bare, _rand(rng, Float32, 6)),
                (61, "maximum(x) 3d", _max_bare, _rand(rng, Float32, 3, 3, 2)),
                (62, "maximum(x; dims=:)", _max_nodims, _rand(rng, Float32, 4, 3)),
                (63, "maximum(x; dims=1)", _max_d1, _rand(rng, Float32, 4, 3)),
                (64, "maximum(vec; dims=1)", _max_d1, _rand(rng, Float32, 6)),
                (65, "minimum(x; dims=1)", _min_d1, _rand(rng, Float32, 4, 3)),
            ]
                test_rule(StableRNG(seed), f, x; is_primitive=false, perf_flag=:none)
            end
            @testset "ties pick the lowest linear index" begin
                # Not reachable from test_rule's random inputs. findmax names the
                # first tied element, matching Base and ChainRules; the CPU
                # decomposition of mapreduce(identity, max, x) lands on the last.
                x = CuArray(Float32[5, 5, 3])
                _, g = value_and_gradient!!(
                    Mooncake.build_rrule(_max_bare, x), _max_bare, x
                )
                @test Array(g[2]) == Float32[1, 0, 0]
            end
        end

        # Forward-over-reverse (HVP): works for non-elementwise ops; NDual-based
        # elementwise rules error loudly (perturbation confusion).
        @testset "forward-over-reverse (HVP)" begin
            x = _rand(rng, Float32, 8)
            v = _rand(rng, Float32, 8)
            # sum(x) is linear ⇒ hvp = 0.
            _, g, h = value_and_hvp!!(prepare_hvp_cache(sum, x), sum, v, x)
            @test isapprox(Array(g), ones(Float32, 8); rtol=1.0f-4)
            @test isapprox(Array(h), zeros(Float32, 8); atol=1.0f-5)
            # dot(x, x) has Hessian 2I ⇒ hvp = 2v.
            dotsq = z -> dot(z, z)
            _, _, h = value_and_hvp!!(prepare_hvp_cache(dotsq, x), dotsq, v, x)
            @test isapprox(Array(h), 2 .* Array(v); rtol=1.0f-4)
            # Full Hessian: buffers are device-resident, no scalar indexing.
            hess_cache = Mooncake.prepare_hessian_cache(dotsq, x)
            _, _, H = Mooncake.value_gradient_and_hessian!!(hess_cache, dotsq, x)
            @test H isa CuMatrix{Float32}
            @test isapprox(Array(H), 2 * I(8); atol=1.0f-4)
            # NDual-based elementwise rules error loudly under forward-over-reverse.
            for f in (z -> sum(abs2, z), z -> sum(abs2.(z)))
                @test_throws r"not yet supported" value_and_hvp!!(
                    prepare_hvp_cache(f, x), f, v, x
                )
            end
        end
    else
        println("Tests are skipped because no CUDA device was found.")
    end
end
