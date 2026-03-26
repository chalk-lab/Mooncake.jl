# bench/nforward_benchmarks.jl
#
# Benchmark nforward frule / rrule against primal, standard Mooncake, and ForwardDiff.
# Uses the same functions as the inter-framework comparison in run_benchmarks.jl.
#
# Two tables are produced:
#   Full gradient (rrule) – primal, mc_rrule, nf_rrule c1/cN/cDOF,
#                           fd_grad c1/cN/c≤DOF (full gradient)        (× primal)
#   Forward mode (frule)  – primal, mc_frule, nf_frule c1/cN/cDOF,
#                           fd_grad c1/cN/c≤DOF /pass                  (× primal)
#
# Column semantics:
#   nf_frule cC  – one nforward pass computing C simultaneous JVPs
#   fd_grad  cC/pass – ForwardDiff.gradient with Chunk{C}, total time ÷ ⌈DOF/C⌉
#                      (cost of one ForwardDiff pass; comparable to nf_frule cC)
#
# Results are saved to bench/results/nforward_<commit>.txt.
#
# Run:
#   julia --project=bench bench/nforward_benchmarks.jl

using Pkg
Pkg.develop(; path=joinpath(@__DIR__, ".."))

using Chairmarks,
    DiffTests,
    ForwardDiff,
    LogExpFunctions,
    Mooncake,
    PrettyTables,
    Printf,
    StableRNGs,
    Statistics

using Mooncake:
    Dual, CoDual, NoFData, NoTangent, primal, tangent, fdata, zero_tangent, zero_rdata

# ── Named wrappers ────────────────────────────────────────────────────────────
# The inter-framework suite defines several of these as anonymous functions
# (closures), which nforward cannot use (stateless callables only).
# These named top-level functions are equivalent and nforward-compatible.

@inline function _nfb_g(x, a, ::Val{N}) where {N}
    return N > 0 ? _nfb_g(x * a, a, Val(N - 1)) : x
end

function _nfb_sum(x::AbstractArray{<:Real})
    y = zero(eltype(x))
    n = 0
    while n < length(x)
        n += 1
        y += x[n]
    end
    return y
end

_nfb_sum_sin(x::AbstractArray{<:Real}) = sum(sin, x)

function _nfb_sum_sin_loop(x::AbstractArray{<:Real})
    y = zero(eltype(x))
    n = 0
    while n < length(x)
        n += 1
        y += sin(x[n])
    end
    return y
end

function _nfb_naive_map_sin_cos_exp(x::AbstractArray{<:Real})
    y = similar(x)
    for n in eachindex(x)
        y[n] = sin(cos(exp(x[n])))
    end
    return sum(y)
end

_nfb_map_sin_cos_exp(x::AbstractArray{<:Real}) = sum(map(z -> sin(cos(exp(z))), x))
_nfb_broadcast_sin_cos_exp(x::AbstractArray{<:Real}) = sum(sin.(cos.(exp.(x))))
_nfb_large_single_block(x::AbstractVector{<:Real}) = _nfb_g(x[1], x[2], Val(400))

# ── DiffTests: num2num (scalar→scalar wrapped as 1-element vector) ─────────────
# Wrapping as x -> f(x[1]) gives DOF=1; ForwardDiff.gradient on a 1-element vec
# is equivalent to ForwardDiff.derivative, so FD comparison remains valid.
_nfb_num2num_1(x) = DiffTests.num2num_1(x[1])   # sin(x)^2 / cos(x)^2
_nfb_num2num_2(x) = DiffTests.num2num_2(x[1])   # 2x + sqrt(x³),  x > 0
_nfb_num2num_3(x) = DiffTests.num2num_3(x[1])   # 10.31^(2x) − x
_nfb_num2num_5(x) = DiffTests.num2num_5(x[1])   # sigmoid(x)

# ── DiffTests: mat2num (additional matrix→scalar functions) ───────────────────
# mat2num_2 uses reshape() internally so it accepts a flattened vector directly.
# Inputs must be small and positive to keep the log() argument above zero.
_nfb_mat2num_2(x) = DiffTests.mat2num_2(x)
# mat2num_4(M) = mean(sum(sin.(M) * M, dims=2)) requires a 2-D matrix; wrap so
# ForwardDiff can use the flattened vector (FD calls f(vec(x)) internally).
_nfb_mat2num_4(x) = DiffTests.mat2num_4(reshape(x, 4, 4))

# ── logexpfunctions: scalar functions applied element-wise (vec→scalar) ────────
# Mirrors test/integration_testing/logexpfunctions/ scalar cases.
_nfb_logsumexp(x) = logsumexp(x)
_nfb_sum_xlogx(x) = sum(xlogx.(x))            # x*log(x),  x > 0
_nfb_sum_logistic(x) = sum(logistic.(x))          # 1/(1+exp(−x))
_nfb_sum_log1pexp(x) = sum(log1pexp.(x))          # log(1+exp(x))
_nfb_sum_logcosh(x) = sum(logcosh.(x))           # log(cosh(x))
# logaddexp uses two halves of the input vector as the two arguments
_nfb_sum_logaddexp(x) = sum(logaddexp.(x[1:(end ÷ 2)], x[(end ÷ 2 + 1):end]))

# ── array: linear-algebra scalar wrappers ─────────────────────────────────────
# Mirrors test/integration_testing/array/array.jl matrix-operation test cases.
# Constants are fixed at load time; the gradient input is the vector x.
using LinearAlgebra
const _nfb_A5 = randn(StableRNG(1), 5, 5)                  # arbitrary 5×5 matrix
const _nfb_M5 = _nfb_A5 * _nfb_A5' + 5 * I                # positive-definite solve target
_nfb_sum_matvec(x) = sum(_nfb_A5 * x)                      # matrix–vector product
_nfb_sum_linsolve(x) = sum(_nfb_M5 \ x)                     # linear solve (A\b)
_nfb_sum_matmat(x) = sum(reshape(x, 5, 5) * reshape(x, 5, 5))  # matrix–matrix product

# ── misc_abstract_array: view / indexing scalar wrappers ─────────────────────
# Mirrors test/integration_testing/misc_abstract_array/misc_abstract_array.jl.
# (map and broadcast already covered in the Mooncake baseline section above.)
_nfb_sum_view(x) = sum(view(x, 1:5))           # SubArray gradient
_nfb_sum_getindex(x) = x[1] + x[3] + x[5] + x[7] + x[9]  # scattered reads

# ── DOF helpers ───────────────────────────────────────────────────────────────

_dof(x::AbstractArray{<:Real}) = length(x)
_dof(x::AbstractArray{<:Complex}) = 2 * length(x)
_dof(::Real) = 1
_dof(::Complex) = 2
_total_dof(x::Tuple) = sum(_dof, x)

# ── Dual / CoDual input construction ─────────────────────────────────────────

# nforward's tangent convention (from _nforward_seed_tangent / _nforward_scalar_lane):
#   chunk_size == 1  → tangent is a bare scalar / same-shape array (NOT a 1-tuple)
#   chunk_size == N  → tangent is an NTuple{N} for scalars, or (size(x)..., N) array
# Identity seeding: lane k carries a 1 at position k, 0 elsewhere.  This matches both
# ForwardDiff's unit-vector seeding and nforward's internal pullback seeding, ensuring
# the frule benchmark exercises the same tangent arithmetic as real gradient computation.
function _nf_dual(x::Float64, N::Int)
    N == 1 ? Dual(x, one(x)) : Dual(x, ntuple(k -> ifelse(k == 1, one(x), zero(x)), N))
end
function _nf_dual(x::AbstractArray{Float64}, N::Int)
    if N == 1
        dx = zeros(Float64, size(x))
        dx[1] = 1.0
        return Dual(x, dx)
    else
        dx = zeros(Float64, size(x)..., N)
        ci = CartesianIndices(x)
        for k in 1:min(N, length(x))
            dx[ci[k], k] = 1.0
        end
        return Dual(x, dx)
    end
end

_mc_fwd_dual(x) = _nf_dual(x, 1)   # forward-mode (frule) input: unit-vector seed, consistent with nf_frule c1
_rev_codual(x) = CoDual(x, fdata(zero_tangent(x))) # reverse-mode (rrule) input

# ── ForwardDiff helpers ───────────────────────────────────────────────────────
# All test cases have exactly one array argument, so GradientConfig applies directly.
# chunk_size is capped at 12 to avoid register pressure from very wide chunk types;
# the cap is noted in the column label.

const _FD_MAX_CHUNK = 12

function _fd_cfg(f, x::AbstractArray, chunk::Int)
    c = min(chunk, _FD_MAX_CHUNK)
    return ForwardDiff.GradientConfig(f, vec(x), ForwardDiff.Chunk{c}()), c
end

function _bench_fd_grad(f, x::AbstractArray, cfg)
    return ForwardDiff.gradient(f, vec(x), cfg)
end

# ── Benchmark helpers ─────────────────────────────────────────────────────────

function _bench_rrule(rule, f_cd, x_cds)
    out, pb = rule(f_cd, x_cds...)
    return pb(zero_rdata(primal(out)))
end

_bench_frule(rule, f_dual, x_duals) = rule(f_dual, x_duals...)

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry:  (label, f, x_tuple, cN)
# cN is the medium chunk size; cDOF (total DOF) is computed automatically.

function nfb_cases(rng)
    v1k = randn(rng, 1_000)
    m10 = randn(rng, 10, 10)
    # Small / medium vectors for DiffTests cases.
    # DiffTests requires length >= 5; use n=10 (DOF=10) for small cases, n=100 for larger.
    v10 = abs.(randn(rng, 10)) .+ 0.1  # positive entries for log/sqrt safety
    v100 = randn(rng, 100)
    m5 = randn(rng, 5, 5)

    # Extra input vectors for new sections.
    v1_pos = [1.5]                              # DOF=1 scalar wrap, positive
    v16_pos = abs.(randn(rng, 16)) .* 0.1 .+ 0.1  # DOF=16 small-positive for mat2num_2
    v100_pos = abs.(randn(rng, 100)) .+ 0.01    # positive for xlogx/logit
    v100_pair = randn(rng, 100)                  # even-length for logaddexp split
    v5 = randn(rng, 5)                         # DOF=5 for LA cases
    v10_misc = randn(rng, 10)                    # DOF=10 for misc_abstract_array

    return [
        # ── Mooncake baseline suite ──────────────────────────────────────────────
        ("sum_1000", sum, (v1k,), 8),
        ("_sum_1000", _nfb_sum, (v1k,), 8),
        ("sum_sin_1000", _nfb_sum_sin, (v1k,), 8),
        ("_sum_sin_1000", _nfb_sum_sin_loop, (v1k,), 8),
        ("naive_map_sin_cos_exp", _nfb_naive_map_sin_cos_exp, (m10,), 8),
        ("map_sin_cos_exp", _nfb_map_sin_cos_exp, (m10,), 8),
        ("broadcast_sin_cos_exp", _nfb_broadcast_sin_cos_exp, (m10,), 8),
        # DOF=2 so cN=cDOF; the cN and cDOF columns (and all three fd_grad columns) are identical.
        ("large_single_block", _nfb_large_single_block, ([0.9, 0.99],), 2),
        # ── DiffTests: vector→scalar (small DOF) ─────────────────────────────────
        ("rosenbrock_1", DiffTests.rosenbrock_1, (v100,), 8),
        ("rosenbrock_2", DiffTests.rosenbrock_2, (v10,), 8),
        ("rosenbrock_3", DiffTests.rosenbrock_3, (v10,), 8),
        ("rosenbrock_4", DiffTests.rosenbrock_4, (v10,), 8),
        ("ackley", DiffTests.ackley, (v100,), 8),
        ("self_weighted_logit", DiffTests.self_weighted_logit, (v10,), 8),
        ("vec2num_1", DiffTests.vec2num_1, (v10,), 8),
        ("vec2num_2", DiffTests.vec2num_2, (v10,), 8),
        ("vec2num_3", DiffTests.vec2num_3, (v10,), 8),
        ("vec2num_5", DiffTests.vec2num_5, (v10,), 8),
        ("vec2num_6", DiffTests.vec2num_6, (v10,), 8),
        ("vec2num_7", DiffTests.vec2num_7, (v10,), 8),
        # ── DiffTests: matrix→scalar ─────────────────────────────────────────────
        # mat2num_1 omitted: calls x*x which means matrix product for Matrix input but
        # vector product for vec(x) (used in the FD comparison), so benchmarks differ.
        ("mat2num_3", DiffTests.mat2num_3, (m5,), 8),
        # ── DiffTests: num2num (scalar→scalar, wrapped as 1-element vector) ──────
        # From DiffTests.NUMBER_TO_NUMBER_FUNCS; vec2num_4 omitted (returns Int64=1).
        # Wrapped as f(x[1]) so FD comparison uses gradient of a 1-element array.
        ("num2num_1", _nfb_num2num_1, (v1_pos,), 1),
        ("num2num_2", _nfb_num2num_2, (v1_pos,), 1),
        ("num2num_3", _nfb_num2num_3, (v1_pos,), 1),
        ("num2num_5", _nfb_num2num_5, (v1_pos,), 1),
        # ── DiffTests: additional matrix→scalar ──────────────────────────────────
        # mat2num_2 accepts a flattened vector (uses reshape internally).
        # mat2num_4 wraps reshape(x, 4, 4) to accept vector input for FD comparison.
        ("mat2num_2", _nfb_mat2num_2, (v16_pos,), 8),
        ("mat2num_4", _nfb_mat2num_4, (randn(rng, 16),), 8),
        # ── logexpfunctions integration test functions ────────────────────────────
        # Mirrors test/integration_testing/logexpfunctions/logexpfunctions.jl.
        # Array→scalar: logsumexp directly; scalar functions applied element-wise.
        # v100_pair has even length so logaddexp can split it into two halves.
        ("logsumexp", _nfb_logsumexp, (v100,), 8),
        ("sum_xlogx", _nfb_sum_xlogx, (v100_pos,), 8),
        ("sum_logistic", _nfb_sum_logistic, (v100,), 8),
        ("sum_log1pexp", _nfb_sum_log1pexp, (v100,), 8),
        ("sum_logcosh", _nfb_sum_logcosh, (v100,), 8),
        ("sum_logaddexp", _nfb_sum_logaddexp, (v100_pair,), 8),
        # ── array integration test functions ─────────────────────────────────────
        # Mirrors test/integration_testing/array/array.jl linear-algebra cases.
        # Constants _nfb_A5 / _nfb_M5 are fixed at load time; gradient is w.r.t. x.
        ("sum_matvec", _nfb_sum_matvec, (v5,), 5),
        ("sum_linsolve", _nfb_sum_linsolve, (v5,), 5),
        ("sum_matmat", _nfb_sum_matmat, (vec(m5),), 8),
        # ── misc_abstract_array integration test functions ────────────────────────
        # Mirrors test/integration_testing/misc_abstract_array/misc_abstract_array.jl.
        # map/broadcast already covered by naive_map_sin_cos_exp / broadcast_sin_cos_exp.
        ("sum_view", _nfb_sum_view, (v10_misc,), 5),
        ("sum_getindex", _nfb_sum_getindex, (v10_misc,), 5),
        # ── battery_tests: not applicable ────────────────────────────────────────
        # test/integration_testing/battery_tests/battery_tests.jl runs Mooncake.TestUtils.test_data
        # on plain values/types (booleans, ints, strings, arrays, custom structs) to verify
        # tangent-type correctness.  It does not differentiate any functions, so there are
        # no gradient benchmarks to add from that file.
    ]
end

# ── Main benchmark loop ───────────────────────────────────────────────────────

function run_nfb(; seconds=0.5)
    rng = StableRNG(42)
    cases = nfb_cases(rng)

    rrule_rows = []
    frule_rows = []
    summary_rows = []  # (label, cN, nf_mult, fd_mult) for the compact nf-vs-FD table

    for (label, f, x, cN) in cases
        dof = _total_dof(x)
        @info "Benchmarking: $label  (DOF=$dof)"

        # Pre-build all rules (compilation excluded from timing).
        mc_rrule = Mooncake.build_rrule(f, x...)
        mc_frule = Mooncake.build_frule(f, x...)
        nf_rrule1 = Mooncake.nforward_build_rrule(f, x...; chunk_size=1)
        nf_rruleN = Mooncake.nforward_build_rrule(f, x...; chunk_size=cN)
        nf_rruleD = Mooncake.nforward_build_rrule(f, x...; chunk_size=dof)
        nf_frule1 = Mooncake.nforward_build_frule(f, x...; chunk_size=1)
        nf_fruleN = Mooncake.nforward_build_frule(f, x...; chunk_size=cN)
        nf_fruleD = Mooncake.nforward_build_frule(f, x...; chunk_size=dof)

        # ForwardDiff configs.  Only single-array inputs are supported here.
        fd_cfg1, _ = _fd_cfg(f, only(x), 1)
        fd_cfgN, cN_fd = _fd_cfg(f, only(x), cN)
        fd_cfgD, cD_fd = _fd_cfg(f, only(x), dof)

        # Shared Mooncake inputs.
        f_cd = CoDual(f, NoFData())
        x_cds = map(_rev_codual, x)  # used for warm-up only; timed rrule calls use fresh copies
        f_md = Dual(f, NoTangent())
        x_mds = map(_mc_fwd_dual, x)
        x_ds1 = map(xi -> _nf_dual(xi, 1), x)
        x_dsN = map(xi -> _nf_dual(xi, cN), x)
        x_dsD = map(xi -> _nf_dual(xi, dof), x)

        # Warm up.
        for _ in 1:3
            f(x...)
            _bench_rrule(mc_rrule, f_cd, x_cds)
            _bench_frule(mc_frule, f_md, x_mds)
            _bench_rrule(nf_rrule1, f_cd, x_cds)
            _bench_rrule(nf_rruleN, f_cd, x_cds)
            _bench_rrule(nf_rruleD, f_cd, x_cds)
            _bench_frule(nf_frule1, f_md, x_ds1)
            _bench_frule(nf_fruleN, f_md, x_dsN)
            _bench_frule(nf_fruleD, f_md, x_dsD)
            _bench_fd_grad(f, only(x), fd_cfg1)
            _bench_fd_grad(f, only(x), fd_cfgN)
            _bench_fd_grad(f, only(x), fd_cfgD)
        end
        GC.gc(true)

        t_prim = median(@be(f($x...), seconds=seconds)).time
        # Fresh x_cds per evaluation: mc_rrule accumulates cotangents into fdata in-place,
        # so reusing x_cds across @be iterations would leave the fdata in a dirty state.
        # Using Chairmarks init (not timed) ensures each call starts with zero fdata.
        t_mc_rr = median(
            @be(
                map(_rev_codual, $x),
                x_cds -> _bench_rrule($mc_rrule, $f_cd, x_cds),
                seconds=seconds
            )
        ).time
        t_nf_rr1 = median(
            @be(
                map(_rev_codual, $x),
                x_cds -> _bench_rrule($nf_rrule1, $f_cd, x_cds),
                seconds=seconds
            )
        ).time
        t_nf_rrN = median(
            @be(
                map(_rev_codual, $x),
                x_cds -> _bench_rrule($nf_rruleN, $f_cd, x_cds),
                seconds=seconds
            )
        ).time
        t_nf_rrD = median(
            @be(
                map(_rev_codual, $x),
                x_cds -> _bench_rrule($nf_rruleD, $f_cd, x_cds),
                seconds=seconds
            )
        ).time
        t_mc_fr = median(@be(_bench_frule($mc_frule, $f_md, $x_mds), seconds=seconds)).time
        t_nf_fr1 =
            median(@be(_bench_frule($nf_frule1, $f_md, $x_ds1), seconds=seconds)).time
        t_nf_frN =
            median(@be(_bench_frule($nf_fruleN, $f_md, $x_dsN), seconds=seconds)).time
        t_nf_frD =
            median(@be(_bench_frule($nf_fruleD, $f_md, $x_dsD), seconds=seconds)).time
        t_fd_gr1 =
            median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfg1), seconds=seconds)).time
        t_fd_grN =
            median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfgN), seconds=seconds)).time
        t_fd_grD =
            median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfgD), seconds=seconds)).time

        # Normalise fd_grad to per-pass cost so each column is comparable to nf_frule cC.
        # fd_grad c1 runs dof passes; fd_grad cC runs ⌈dof/C⌉ passes.
        t_fd_gr1_pp = t_fd_gr1 / dof
        t_fd_grN_pp = t_fd_grN / cld(dof, cN_fd)
        t_fd_grD_pp = t_fd_grD / cld(dof, cD_fd)

        push!(
            rrule_rows,
            [
                label,
                string(dof),
                _fmt(t_prim),
                _ratio(t_mc_rr, t_prim),
                _ratio(t_nf_rr1, t_prim),
                "c$cN: " * _ratio(t_nf_rrN, t_prim),
                "c$dof: " * _ratio(t_nf_rrD, t_prim),
                _ratio(t_fd_gr1, t_prim),
                "c$cN_fd: " * _ratio(t_fd_grN, t_prim),
                "c$cD_fd: " * _ratio(t_fd_grD, t_prim),
            ],
        )
        push!(summary_rows, (label, cN, t_nf_rrN / t_prim, t_fd_grN / t_prim))
        push!(
            frule_rows,
            [
                label,
                string(dof),
                _fmt(t_prim),
                _ratio(t_mc_fr, t_prim),
                _ratio(t_nf_fr1, t_prim),
                "c$cN: " * _ratio(t_nf_frN, t_prim),
                "c$dof: " * _ratio(t_nf_frD, t_prim),
                _ratio(t_fd_gr1_pp, t_prim),
                "c$cN_fd: " * _ratio(t_fd_grN_pp, t_prim),
                "c$cD_fd: " * _ratio(t_fd_grD_pp, t_prim),
            ],
        )
    end

    rrule_header = [
        "Test",
        "DOF",
        "Primal",
        "mc_rrule",
        "nf_rrule c1",
        "nf_rrule cN",
        "nf_rrule cDOF",
        "fd_grad c1",
        "fd_grad cN",
        "fd_grad c≤DOF",
    ]
    frule_header = [
        "Test",
        "DOF",
        "Primal",
        "mc_frule",
        "nf_frule c1",
        "nf_frule cN",
        "nf_frule cDOF",
        "fd_grad c1/pass",
        "fd_grad cN/pass",
        "fd_grad c≤DOF/pass",
    ]

    rrule_mat = permutedims(reduce(hcat, rrule_rows))
    frule_mat = permutedims(reduce(hcat, frule_rows))
    rrule_aln = [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r]
    frule_aln = [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r]

    io = IOBuffer()

    commit = strip(read(`git rev-parse --short HEAD`, String))
    header_line = "nforward benchmark  commit=$(commit)  julia=$(VERSION)"
    for out in (stdout, io)
        println(out, "\n", header_line)
        println(out, "\n=== Full gradient (rrule) — time relative to primal ===")
        println(
            out,
            "    fd_grad  cC  = ForwardDiff.gradient with Chunk{C}, full gradient time (c≤DOF capped at $_FD_MAX_CHUNK)\n",
        )
        pretty_table(
            out,
            rrule_mat;
            column_labels=rrule_header,
            alignment=rrule_aln,
            display_size=(-1, -1),
        )
        println(out, "\n=== Forward-mode (frule) — time relative to primal ===")
        println(out, "    nf_frule cC  = one nforward pass, C simultaneous JVPs")
        println(
            out,
            "    fd_grad  cC/pass = ForwardDiff.gradient with Chunk{C}, total time ÷ ⌈DOF/C⌉ passes (c≤DOF capped at $_FD_MAX_CHUNK)\n",
        )
        pretty_table(
            out,
            frule_mat;
            column_labels=frule_header,
            alignment=frule_aln,
            display_size=(-1, -1),
        )
        _print_summary_table(out, summary_rows)
    end

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    outfile = joinpath(results_dir, "nforward_$(commit).txt")
    write(outfile, String(take!(io)))
    println("\nResults saved to $outfile")
end

function _fmt(t)
    if t < 1e-6
        "$(round(t*1e9; sigdigits=3)) ns"
    elseif t < 1e-3
        "$(round(t*1e6; sigdigits=3)) μs"
    elseif t < 1.0
        "$(round(t*1e3; sigdigits=3)) ms"
    else
        "$(round(t;     sigdigits=3)) s"
    end
end

_ratio(t, ref) = string(round(t / ref; sigdigits=3)) * "×"

# Compact multiplier: round to 3 sig figs, then drop trailing .0 for integers.
function _sfmt(v::Float64)
    r = round(v; sigdigits=3)
    r == round(r) && return @sprintf("%d×", round(Int, r))
    r >= 10 && return @sprintf("%.1f×", r)
    return @sprintf("%.2f×", r)
end

function _print_summary_table(io, rows)
    col1 = 32
    println(io)
    println(io, "=== nforward vs ForwardDiff (nf_rrule cN vs fd_grad cN) ===")
    println(io)
    println(
        io,
        @sprintf(
            "| %-*s | %-8s | %-7s | %-6s |",
            col1,
            "benchmark (c8 unless noted)",
            "nforward",
            "FD",
            "nf/fd"
        )
    )
    println(io, @sprintf("|-%s-|-%s-|-%s-|-%s-|", "-"^col1, "-"^8, "-"^7, "-"^6))
    for (label, cN, nf, fd) in rows
        bench_label = cN == 8 ? label : "$label (c$cN)"
        println(
            io,
            @sprintf(
                "| %-*s | %-8s | %-7s | %-6s |",
                col1,
                bench_label,
                _sfmt(nf),
                _sfmt(fd),
                _sfmt(nf / fd)
            )
        )
    end
end

run_nfb()
