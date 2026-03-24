# bench/nforward_benchmarks.jl
#
# Benchmark nforward frule / rrule against primal, standard Mooncake, and ForwardDiff.
# Uses the same functions as the inter-framework comparison in run_benchmarks.jl.
#
# Two tables are produced:
#   Full gradient (rrule) – primal, mc_rrule, nf_rrule c1/cN/cDOF   (× primal)
#   Forward mode (frule)  – primal, mc_frule, nf_frule c1/cN/cDOF,
#                           fd_grad c1/cN/cDOF                        (× primal)
#
# Column semantics:
#   nf_frule cC  – one nforward pass computing C simultaneous JVPs
#   fd_grad  cC  – ForwardDiff.gradient with Chunk{C} (full gradient,
#                  ceil(DOF/C) passes); at cDOF this is a single pass
#
# Results are saved to bench/results/nforward_<commit>.txt.
#
# Run:
#   julia --project=bench bench/nforward_benchmarks.jl

using Pkg
Pkg.develop(; path=joinpath(@__DIR__, ".."))

using Chairmarks, ForwardDiff, Mooncake, PrettyTables, StableRNGs, Statistics

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

_nf_dual(x::Float64, N::Int) = N == 1 ? Dual(x, one(x)) : Dual(x, ntuple(_ -> one(x), N))
function _nf_dual(x::AbstractArray{Float64}, N::Int)
    dx = N == 1 ? ones(Float64, size(x)) : ones(Float64, size(x)..., N)
    return Dual(x, dx)
end

_mc_dual(x) = Mooncake.zero_dual(x)
_codual(x)  = CoDual(x, fdata(zero_tangent(x)))

# ── ForwardDiff helpers ───────────────────────────────────────────────────────
# All test cases have exactly one array argument, so GradientConfig applies directly.
# chunk_size is capped at ForwardDiff's recommended maximum (12) to avoid register
# pressure from very wide chunk types; the cap is noted in the column label.

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
        ("large_single_block",    _nfb_large_single_block,    ([0.9, 0.99],), 2),
    ]
end

# ── Main benchmark loop ───────────────────────────────────────────────────────

function run_nfb(; seconds=0.5)
    rng = StableRNG(42)
    cases = nfb_cases(rng)

    rrule_rows = []
    frule_rows = []

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
        x_vec = vec(only(x))

        # Shared Mooncake inputs.
        f_cd  = CoDual(f, NoFData())
        x_cds = map(_codual, x)
        f_md  = Dual(f, NoTangent())
        x_mds = map(_mc_dual, x)
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
        t_mc_rr   = median(@be(_bench_rrule($mc_rrule,  $f_cd, $x_cds), seconds=seconds)).time
        t_nf_rr1  = median(@be(_bench_rrule($nf_rrule1, $f_cd, $x_cds), seconds=seconds)).time
        t_nf_rrN  = median(@be(_bench_rrule($nf_rruleN, $f_cd, $x_cds), seconds=seconds)).time
        t_nf_rrD  = median(@be(_bench_rrule($nf_rruleD, $f_cd, $x_cds), seconds=seconds)).time
        t_mc_fr   = median(@be(_bench_frule($mc_frule,  $f_md, $x_mds), seconds=seconds)).time
        t_nf_fr1  = median(@be(_bench_frule($nf_frule1, $f_md, $x_ds1), seconds=seconds)).time
        t_nf_frN  = median(@be(_bench_frule($nf_fruleN, $f_md, $x_dsN), seconds=seconds)).time
        t_nf_frD  = median(@be(_bench_frule($nf_fruleD, $f_md, $x_dsD), seconds=seconds)).time
        t_fd_gr1  = median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfg1), seconds=seconds)).time
        t_fd_grN  = median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfgN), seconds=seconds)).time
        t_fd_grD  = median(@be(_bench_fd_grad($f, $(only(x)), $fd_cfgD), seconds=seconds)).time

        push!(rrule_rows, [
            label, string(dof), _fmt(t_prim),
            _ratio(t_mc_rr,  t_prim),
            _ratio(t_nf_rr1, t_prim),
            "c$cN: "   * _ratio(t_nf_rrN, t_prim),
            "c$dof: "  * _ratio(t_nf_rrD, t_prim),
        ])
        push!(frule_rows, [
            label, string(dof), _fmt(t_prim),
            _ratio(t_mc_fr,  t_prim),
            _ratio(t_nf_fr1, t_prim),
            "c$cN: "   * _ratio(t_nf_frN, t_prim),
            "c$dof: "  * _ratio(t_nf_frD, t_prim),
            _ratio(t_fd_gr1, t_prim),
            "c$cN_fd: " * _ratio(t_fd_grN, t_prim),
            "c$cD_fd: " * _ratio(t_fd_grD, t_prim),
        ])
    end

    rrule_header = [
        "Test", "DOF", "Primal",
        "mc_rrule", "nf_rrule c1", "nf_rrule cN", "nf_rrule cDOF",
    ]
    frule_header = [
        "Test", "DOF", "Primal",
        "mc_frule", "nf_frule c1", "nf_frule cN", "nf_frule cDOF",
        "fd_grad c1", "fd_grad cN", "fd_grad cDOF",
    ]

    rrule_mat  = permutedims(reduce(hcat, rrule_rows))
    frule_mat  = permutedims(reduce(hcat, frule_rows))
    rrule_aln  = [:l, :r, :r, :r, :r, :r, :r]
    frule_aln  = [:l, :r, :r, :r, :r, :r, :r, :r, :r, :r]

    io = IOBuffer()

    commit = strip(read(`git rev-parse --short HEAD`, String))
    header_line = "nforward benchmark  commit=$(commit)  julia=$(VERSION)"
    for out in (stdout, io)
        println(out, "\n", header_line)
        println(out, "\n=== Full gradient (rrule) — time relative to primal ===\n")
        pretty_table(out, rrule_mat; column_labels=rrule_header, alignment=rrule_aln, display_size=(-1,-1))
        println(out, "\n=== Forward-mode (frule) — time relative to primal ===")
        println(out, "    nf_frule cC  = one nforward pass, C simultaneous JVPs")
        println(out, "    fd_grad  cC  = ForwardDiff.gradient with Chunk{C} (full gradient; capped at $_FD_MAX_CHUNK)\n")
        pretty_table(out, frule_mat; column_labels=frule_header, alignment=frule_aln, display_size=(-1,-1))
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

run_nfb()
