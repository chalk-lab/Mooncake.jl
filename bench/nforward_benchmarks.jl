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

using Chairmarks, DiffTests, ForwardDiff, Mooncake, PrettyTables, Printf, StableRNGs, Statistics

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

# ── DOF helpers ───────────────────────────────────────────────────────────────

_dof(x::AbstractArray{<:Real})    = length(x)
_dof(x::AbstractArray{<:Complex}) = 2 * length(x)
_dof(::Real)                      = 1
_dof(::Complex)                   = 2
_total_dof(x::Tuple)              = sum(_dof, x)

# ── Dual / CoDual input construction ─────────────────────────────────────────

# nforward's tangent convention (from _nforward_seed_tangent / _nforward_scalar_lane):
#   chunk_size == 1  → tangent is a bare scalar / same-shape array (NOT a 1-tuple)
#   chunk_size == N  → tangent is an NTuple{N} for scalars, or (size(x)..., N) array
# Identity seeding: lane k carries a 1 at position k, 0 elsewhere.  This matches both
# ForwardDiff's unit-vector seeding and nforward's internal pullback seeding, ensuring
# the frule benchmark exercises the same tangent arithmetic as real gradient computation.
_nf_dual(x::Float64, N::Int) = N == 1 ? Dual(x, one(x)) :
    Dual(x, ntuple(k -> ifelse(k == 1, one(x), zero(x)), N))
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
_rev_codual(x)  = CoDual(x, fdata(zero_tangent(x))) # reverse-mode (rrule) input

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
    return [
        ("sum_1000",              sum,                        (v1k,),         8),
        ("_sum_1000",             _nfb_sum,                   (v1k,),         8),
        ("sum_sin_1000",          _nfb_sum_sin,               (v1k,),         8),
        ("_sum_sin_1000",         _nfb_sum_sin_loop,          (v1k,),         8),
        ("naive_map_sin_cos_exp", _nfb_naive_map_sin_cos_exp, (m10,),         8),
        ("map_sin_cos_exp",       _nfb_map_sin_cos_exp,       (m10,),         8),
        ("broadcast_sin_cos_exp", _nfb_broadcast_sin_cos_exp, (m10,),         8),
        # DOF=2 so cN=cDOF; the cN and cDOF columns (and all three fd_grad columns) are identical.
        ("large_single_block",    _nfb_large_single_block,    ([0.9, 0.99],), 2),
        ("rosenbrock_1",          DiffTests.rosenbrock_1,     (randn(rng, 100),), 8),
        ("ackley",                DiffTests.ackley,           (randn(rng, 100),), 8),
    ]
end

# ── Main benchmark loop ───────────────────────────────────────────────────────

function run_nfb(; seconds=0.5)
    rng = StableRNG(42)
    cases = nfb_cases(rng)

    rrule_rows   = []
    frule_rows   = []
    summary_rows = []  # (label, cN, nf_mult, fd_mult) for the compact nf-vs-FD table

    for (label, f, x, cN) in cases
        dof = _total_dof(x)
        @info "Benchmarking: $label  (DOF=$dof)"

        # Pre-build all rules (compilation excluded from timing).
        mc_rrule   = Mooncake.build_rrule(f, x...)
        mc_frule   = Mooncake.build_frule(f, x...)
        nf_rrule1  = Mooncake.nforward_build_rrule(f, x...; chunk_size=1)
        nf_rruleN  = Mooncake.nforward_build_rrule(f, x...; chunk_size=cN)
        nf_rruleD  = Mooncake.nforward_build_rrule(f, x...; chunk_size=dof)
        nf_frule1  = Mooncake.nforward_build_frule(f, x...; chunk_size=1)
        nf_fruleN  = Mooncake.nforward_build_frule(f, x...; chunk_size=cN)
        nf_fruleD  = Mooncake.nforward_build_frule(f, x...; chunk_size=dof)

        # ForwardDiff configs.  Only single-array inputs are supported here.
        fd_cfg1, _     = _fd_cfg(f, only(x), 1)
        fd_cfgN, cN_fd = _fd_cfg(f, only(x), cN)
        fd_cfgD, cD_fd = _fd_cfg(f, only(x), dof)

        # Shared Mooncake inputs.
        f_cd  = CoDual(f, NoFData())
        x_cds = map(_rev_codual, x)  # used for warm-up only; timed rrule calls use fresh copies
        f_md  = Dual(f, NoTangent())
        x_mds = map(_mc_fwd_dual, x)
        x_ds1 = map(xi -> _nf_dual(xi, 1),   x)
        x_dsN = map(xi -> _nf_dual(xi, cN),  x)
        x_dsD = map(xi -> _nf_dual(xi, dof), x)

        # Warm up.
        for _ in 1:3
            f(x...)
            _bench_rrule(mc_rrule,  f_cd, x_cds)
            _bench_frule(mc_frule,  f_md, x_mds)
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

        t_prim    = median(@be(f($x...), seconds=seconds)).time
        # Fresh x_cds per evaluation: mc_rrule accumulates cotangents into fdata in-place,
        # so reusing x_cds across @be iterations would leave the fdata in a dirty state.
        # Using Chairmarks init (not timed) ensures each call starts with zero fdata.
        t_mc_rr  = median(@be(map(_rev_codual, $x), x_cds -> _bench_rrule($mc_rrule,  $f_cd, x_cds), seconds=seconds)).time
        t_nf_rr1 = median(@be(map(_rev_codual, $x), x_cds -> _bench_rrule($nf_rrule1, $f_cd, x_cds), seconds=seconds)).time
        t_nf_rrN = median(@be(map(_rev_codual, $x), x_cds -> _bench_rrule($nf_rruleN, $f_cd, x_cds), seconds=seconds)).time
        t_nf_rrD = median(@be(map(_rev_codual, $x), x_cds -> _bench_rrule($nf_rruleD, $f_cd, x_cds), seconds=seconds)).time
        t_mc_fr   = median(@be(_bench_frule($mc_frule,  $f_md, $x_mds), seconds=seconds)).time
        t_nf_fr1  = median(@be(_bench_frule($nf_frule1, $f_md, $x_ds1), seconds=seconds)).time
        t_nf_frN  = median(@be(_bench_frule($nf_fruleN, $f_md, $x_dsN), seconds=seconds)).time
        t_nf_frD  = median(@be(_bench_frule($nf_fruleD, $f_md, $x_dsD), seconds=seconds)).time
        t_fd_gr1  = median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfg1), seconds=seconds)).time
        t_fd_grN  = median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfgN), seconds=seconds)).time
        t_fd_grD  = median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfgD), seconds=seconds)).time

        # Normalise fd_grad to per-pass cost so each column is comparable to nf_frule cC.
        # fd_grad c1 runs dof passes; fd_grad cC runs ⌈dof/C⌉ passes.
        t_fd_gr1_pp = t_fd_gr1 / dof
        t_fd_grN_pp = t_fd_grN / cld(dof, cN_fd)
        t_fd_grD_pp = t_fd_grD / cld(dof, cD_fd)

        push!(rrule_rows, [
            label, string(dof), _fmt(t_prim),
            _ratio(t_mc_rr,  t_prim),
            _ratio(t_nf_rr1, t_prim),
            "c$cN: "   * _ratio(t_nf_rrN, t_prim),
            "c$dof: "  * _ratio(t_nf_rrD, t_prim),
            _ratio(t_fd_gr1, t_prim),
            "c$cN_fd: " * _ratio(t_fd_grN, t_prim),
            "c$cD_fd: " * _ratio(t_fd_grD, t_prim),
        ])
        push!(summary_rows, (label, cN, t_nf_rrN / t_prim, t_fd_grN / t_prim))
        push!(frule_rows, [
            label, string(dof), _fmt(t_prim),
            _ratio(t_mc_fr,  t_prim),
            _ratio(t_nf_fr1, t_prim),
            "c$cN: "   * _ratio(t_nf_frN, t_prim),
            "c$dof: "  * _ratio(t_nf_frD, t_prim),
            _ratio(t_fd_gr1_pp, t_prim),
            "c$cN_fd: " * _ratio(t_fd_grN_pp, t_prim),
            "c$cD_fd: " * _ratio(t_fd_grD_pp, t_prim),
        ])
    end

    rrule_header = [
        "Test", "DOF", "Primal",
        "mc_rrule", "nf_rrule c1", "nf_rrule cN", "nf_rrule cDOF",
        "fd_grad c1", "fd_grad cN", "fd_grad c≤DOF",
    ]
    frule_header = [
        "Test", "DOF", "Primal",
        "mc_frule", "nf_frule c1", "nf_frule cN", "nf_frule cDOF",
        "fd_grad c1/pass", "fd_grad cN/pass", "fd_grad c≤DOF/pass",
    ]

    rrule_mat  = permutedims(reduce(hcat, rrule_rows))
    frule_mat  = permutedims(reduce(hcat, frule_rows))
    rrule_aln  = [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r]
    frule_aln  = [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r]

    io = IOBuffer()

    commit = strip(read(`git rev-parse --short HEAD`, String))
    header_line = "nforward benchmark  commit=$(commit)  julia=$(VERSION)"
    for out in (stdout, io)
        println(out, "\n", header_line)
        println(out, "\n=== Full gradient (rrule) — time relative to primal ===")
        println(out, "    fd_grad  cC  = ForwardDiff.gradient with Chunk{C}, full gradient time (c≤DOF capped at $_FD_MAX_CHUNK)\n")
        pretty_table(out, rrule_mat; column_labels=rrule_header, alignment=rrule_aln, display_size=(-1,-1))
        println(out, "\n=== Forward-mode (frule) — time relative to primal ===")
        println(out, "    nf_frule cC  = one nforward pass, C simultaneous JVPs")
        println(out, "    fd_grad  cC/pass = ForwardDiff.gradient with Chunk{C}, total time ÷ ⌈DOF/C⌉ passes (c≤DOF capped at $_FD_MAX_CHUNK)\n")
        pretty_table(out, frule_mat; column_labels=frule_header, alignment=frule_aln, display_size=(-1,-1))
        _print_summary_table(out, summary_rows)
    end

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    outfile = joinpath(results_dir, "nforward_$(commit).txt")
    write(outfile, String(take!(io)))
    println("\nResults saved to $outfile")
end

_fmt(t) = t < 1e-6 ? "$(round(t*1e9; sigdigits=3)) ns" :
          t < 1e-3 ? "$(round(t*1e6; sigdigits=3)) μs" :
          t < 1.0  ? "$(round(t*1e3; sigdigits=3)) ms" :
                     "$(round(t;     sigdigits=3)) s"

_ratio(t, ref) = string(round(t / ref; sigdigits=3)) * "×"

# Compact multiplier: round to 3 sig figs, then drop trailing .0 for integers.
function _sfmt(v::Float64)
    r = round(v; sigdigits=3)
    r == round(r) && return @sprintf("%d×", round(Int, r))
    r >= 10       && return @sprintf("%.1f×", r)
    return               @sprintf("%.2f×", r)
end

function _print_summary_table(io, rows)
    col1 = 32
    println(io)
    println(io, "=== nforward vs ForwardDiff (nf_rrule cN vs fd_grad cN) ===")
    println(io)
    println(io, @sprintf("| %-*s | %-8s | %-7s | %-6s |",
                         col1, "benchmark (c8 unless noted)", "nforward", "FD", "nf/fd"))
    println(io, @sprintf("|-%s-|-%s-|-%s-|-%s-|", "-"^col1, "-"^8, "-"^7, "-"^6))
    for (label, cN, nf, fd) in rows
        bench_label = cN == 8 ? label : "$label (c$cN)"
        println(io, @sprintf("| %-*s | %-8s | %-7s | %-6s |",
                             col1, bench_label, _sfmt(nf), _sfmt(fd), _sfmt(nf / fd)))
    end
end

run_nfb()
