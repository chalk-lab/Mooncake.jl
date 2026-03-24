# bench/nforward_benchmarks.jl
#
# Benchmark nforward frule / rrule against primal and standard Mooncake rules,
# using the same functions as the inter-framework comparison in run_benchmarks.jl.
#
# Two tables are produced:
#   Full gradient  – primal, mc_rrule, nf_rrule_c1, nf_rrule_cN (× primal)
#   JVP (1 pass)   – primal, mc_frule, nf_frule_c1, nf_frule_cN (× primal)
#
# Run:
#   julia --project=bench bench/nforward_benchmarks.jl

using Pkg
Pkg.develop(; path=joinpath(@__DIR__, ".."))

using Chairmarks, Mooncake, PrettyTables, StableRNGs, Statistics

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

# ── Dual / CoDual input construction ─────────────────────────────────────────

# Build an nforward frule Dual input.
# chunk_size=1: tangent is same shape as x.
# chunk_size=N: tangent gets an extra trailing dimension of size N.
_nf_dual(x::Float64, N::Int) = N == 1 ? Dual(x, one(x)) : Dual(x, ntuple(_ -> one(x), N))
function _nf_dual(x::AbstractArray{Float64}, N::Int)
    dx = N == 1 ? ones(Float64, size(x)) : ones(Float64, size(x)..., N)
    return Dual(x, dx)
end

# Build a standard Mooncake frule Dual input (zero tangent, same shape).
_mc_dual(x) = Mooncake.zero_dual(x)

# Build a CoDual for rrule calls (zero fdata).
_codual(x) = CoDual(x, fdata(zero_tangent(x)))

# ── Benchmark helpers ─────────────────────────────────────────────────────────

function _bench_rrule(rule, f_cd, x_cds)
    out, pb = rule(f_cd, x_cds...)
    return pb(zero_rdata(primal(out)))
end

function _bench_frule(rule, f_dual, x_duals)
    return rule(f_dual, x_duals...)
end

# ── Test cases ────────────────────────────────────────────────────────────────
# Each entry:  (label, f, x_tuple, chunk_size_for_cN)

function nfb_cases(rng)
    v1k = randn(rng, 1_000)
    m10 = randn(rng, 10, 10)
    return [
        ("sum_1000",              sum,                      (v1k,),  8),
        ("_sum_1000",             _nfb_sum,                 (v1k,),  8),
        ("sum_sin_1000",          _nfb_sum_sin,             (v1k,),  8),
        ("_sum_sin_1000",         _nfb_sum_sin_loop,        (v1k,),  8),
        ("naive_map_sin_cos_exp", _nfb_naive_map_sin_cos_exp, (m10,), 8),
        ("map_sin_cos_exp",       _nfb_map_sin_cos_exp,     (m10,),  8),
        ("broadcast_sin_cos_exp", _nfb_broadcast_sin_cos_exp, (m10,), 8),
        ("large_single_block",    _nfb_large_single_block,  ([0.9, 0.99],), 2),
    ]
end

# ── Main benchmark loop ───────────────────────────────────────────────────────

function run_nfb(; seconds=0.5)
    rng = StableRNG(42)
    cases = nfb_cases(rng)

    rrule_rows = []
    frule_rows = []

    for (label, f, x, cN) in cases
        @info "Benchmarking: $label"

        # Pre-build all rules (compilation excluded from timing).
        mc_rrule  = Mooncake.build_rrule(f, x...)
        mc_frule  = Mooncake.build_frule(f, x...)
        nf_rrule1 = Mooncake.nforward_build_rrule(f, x...; chunk_size=1)
        nf_rruleN = Mooncake.nforward_build_rrule(f, x...; chunk_size=cN)
        nf_frule1 = Mooncake.nforward_build_frule(f, x...; chunk_size=1)
        nf_fruleN = Mooncake.nforward_build_frule(f, x...; chunk_size=cN)

        # Shared input views (rules must not mutate x itself).
        f_cd   = CoDual(f, NoFData())
        x_cds  = map(_codual, x)
        f_md   = Dual(f, NoTangent())
        x_mds  = map(_mc_dual, x)
        x_ds1  = map(xi -> _nf_dual(xi, 1),  x)
        x_dsN  = map(xi -> _nf_dual(xi, cN), x)

        # Warm up all paths.
        for _ in 1:3
            f(x...)
            _bench_rrule(mc_rrule, f_cd, x_cds)
            _bench_frule(mc_frule, f_md, x_mds)
            _bench_rrule(nf_rrule1, f_cd, x_cds)
            _bench_rrule(nf_rruleN, f_cd, x_cds)
            _bench_frule(nf_frule1, f_md, x_ds1)
            _bench_frule(nf_fruleN, f_md, x_dsN)
        end
        GC.gc(true)

        t_prim   = median(@be(f($x...), seconds=seconds)).time
        t_mc_rr  = median(@be(_bench_rrule($mc_rrule,  $f_cd, $x_cds),  seconds=seconds)).time
        t_nf_rr1 = median(@be(_bench_rrule($nf_rrule1, $f_cd, $x_cds),  seconds=seconds)).time
        t_nf_rrN = median(@be(_bench_rrule($nf_rruleN, $f_cd, $x_cds),  seconds=seconds)).time
        t_mc_fr  = median(@be(_bench_frule($mc_frule,  $f_md, $x_mds),  seconds=seconds)).time
        t_nf_fr1 = median(@be(_bench_frule($nf_frule1, $f_md, $x_ds1),  seconds=seconds)).time
        t_nf_frN = median(@be(_bench_frule($nf_fruleN, $f_md, $x_dsN),  seconds=seconds)).time

        push!(rrule_rows, (
            label         = label,
            primal        = _fmt(t_prim),
            mc_rrule      = _ratio(t_mc_rr, t_prim),
            nf_rrule_c1   = _ratio(t_nf_rr1, t_prim),
            nf_rrule_c8   = "c$cN: " * _ratio(t_nf_rrN, t_prim),
        ))
        push!(frule_rows, (
            label         = label,
            primal        = _fmt(t_prim),
            mc_frule      = _ratio(t_mc_fr, t_prim),
            nf_frule_c1   = _ratio(t_nf_fr1, t_prim),
            nf_frule_c8   = "c$cN: " * _ratio(t_nf_frN, t_prim),
        ))
    end

    println("\n=== Full gradient (rrule) — time relative to primal ===\n")
    _print_table(
        rrule_rows,
        ["Test", "Primal", "mc_rrule", "nf_rrule c1", "nf_rrule cN"],
    )

    println("\n=== JVP / single forward pass (frule) — time relative to primal ===\n")
    _print_table(
        frule_rows,
        ["Test", "Primal", "mc_frule", "nf_frule c1", "nf_frule cN"],
    )
end

function _print_table(rows, header)
    mat = hcat(
        [r.label      for r in rows],
        [r.primal     for r in rows],
        [r[3]         for r in rows],
        [r[4]         for r in rows],
        [r[5]         for r in rows],
    )
    pretty_table(mat; column_labels=header, alignment=[:l, :r, :r, :r, :r])
end

_fmt(t) = t < 1e-6 ? "$(round(t*1e9; sigdigits=3)) ns" :
          t < 1e-3 ? "$(round(t*1e6; sigdigits=3)) μs" :
          t < 1.0  ? "$(round(t*1e3; sigdigits=3)) ms" :
                     "$(round(t;     sigdigits=3)) s"

_ratio(t, ref) = string(round(t / ref; sigdigits=3)) * "×"

run_nfb()
