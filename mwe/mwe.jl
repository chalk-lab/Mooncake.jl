using Mooncake

# On main, threaded_map! is not defined — expected failure.
if !isdefined(Mooncake, :threaded_map!)
    println("FAIL: Mooncake.threaded_map! is not defined")
    exit(1)
end

n = 100_000  # large enough that threading overhead is amortised

x = randn(Float64, n)
out1 = zeros(Float64, n)
out2 = zeros(Float64, n)

# More compute-heavy function: tanh is transcendental with higher compute-to-memory ratio.
f_bench = tanh

# --- Correctness: primal ---
threaded_map!(f_bench, out1, x)
map!(f_bench, out2, x)
if !isapprox(out1, out2; rtol=1e-12)
    println("FAIL: threaded_map! primal result differs from map!")
    exit(1)
end

# --- Correctness: gradient ---
loss(v) = sum(threaded_map!(f_bench, zeros(n), v))
cache = prepare_gradient_cache(loss, x)
val, (_, grad) = value_and_gradient!!(cache, loss, x)

# tanh'(x) = 1 - tanh(x)^2 = sech(x)^2
expected_val  = sum(tanh.(x))
expected_grad = 1 .- tanh.(x).^2

if !isapprox(val, expected_val; rtol=1e-12)
    println("FAIL: value mismatch: got $val, expected $expected_val")
    exit(1)
end
if !isapprox(grad, expected_grad; rtol=1e-6)
    println("FAIL: gradient mismatch (max err=$(maximum(abs, grad .- expected_grad)))")
    exit(1)
end

# --- Primal timing (informational — not asserted) ---
# Note: map! can be SIMD-vectorised by the compiler, making the primal speedup of
# Threads.@threads over map! unpredictable for lightweight f.
for _ in 1:5
    threaded_map!(f_bench, out1, x)
    map!(f_bench, out2, x)
end
t_threaded = minimum([@elapsed(threaded_map!(f_bench, out1, x)) for _ in 1:20])
t_serial   = minimum([@elapsed(map!(f_bench, out2, x))          for _ in 1:20])
nthreads   = Threads.nthreads()
println("primal: nthreads=$nthreads  serial=$(round(t_serial*1e3; digits=2))ms  " *
        "threaded=$(round(t_threaded*1e3; digits=2))ms  " *
        "speedup=$(round(t_serial/t_threaded; digits=2))x  (informational)")

# --- AD speedup: threaded_map! vs map! differentiated by Mooncake ---
# Both functions have the same structure: sum(map!-variant(f, zeros(n), v)).
# threaded_map! uses a hand-coded parallel rrule!!; map! is traced by Mooncake's interpreter.
# Both benchmarks start from a prepared gradient cache (no compilation overhead).
function loss_threaded(v)
    return sum(threaded_map!(f_bench, zeros(length(v)), v))
end
function loss_serial_ad(v)
    return sum(map!(f_bench, zeros(length(v)), v))
end

cache_t = prepare_gradient_cache(loss_threaded, x)
cache_s = prepare_gradient_cache(loss_serial_ad, x)

# Warm up — run a few iterations to ensure any lazy initialisation is done.
for _ in 1:3
    value_and_gradient!!(cache_t, loss_threaded, x)
    value_and_gradient!!(cache_s, loss_serial_ad, x)
end

# Benchmark from prepared state: time only value_and_gradient!! (not prepare_gradient_cache).
t_ad_t = minimum([@elapsed(value_and_gradient!!(cache_t, loss_threaded, x)) for _ in 1:20])
t_ad_s = minimum([@elapsed(value_and_gradient!!(cache_s, loss_serial_ad, x)) for _ in 1:20])
ad_speedup = t_ad_s / t_ad_t
println("AD:     nthreads=$nthreads  serial=$(round(t_ad_s*1e3; digits=2))ms  " *
        "threaded=$(round(t_ad_t*1e3; digits=2))ms  " *
        "speedup=$(round(ad_speedup; digits=2))x")

if nthreads > 1 && ad_speedup < 1.3
    println("FAIL: expected AD speedup >= 1.3x with $nthreads threads, got $(round(ad_speedup; digits=2))x")
    exit(1)
end

println("PASS")
exit(0)
