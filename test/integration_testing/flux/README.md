# Flux gradient benchmarks

This README is generated automatically by `benchmarks.jl`; do not edit it directly.

Run the complete benchmark from the repository root:

```sh
julia --project=test/integration_testing/flux \
    test/integration_testing/flux/benchmarks.jl
```

Pass `cpu` or `gpu` to select one device. A following integer selects one model, for example `benchmarks.jl cpu 1`.

## Environment

```text
Julia 1.12.7, Flux 0.16.11, Mooncake 0.5.49 (bb27233ea), CUDA 6.3.1, cuDNN 6.3.1
aarch64-linux-gnu; CPU: Cortex-A725 + Cortex-X925; GPU: NVIDIA GB10
1 Julia thread(s); BLAS lbt, 1 thread(s)
2.0-second warm budget; 32x input workload
```

## Method

Each first-gradient measurement runs in a fresh Julia process after package and model loading. Mooncake's measurement includes `prepare_gradient_cache` and the first `value_and_gradient!!` call. Warm measurements reuse the prepared cache. Both backends differentiate the same model and input with respect to both arguments.

Inputs are enlarged by a factor of 32, and Flux layers run in test mode. `Mc / Zyg` is Mooncake time divided by Zygote time, so values below one favour Mooncake. GPU runs use `Flux.gpu` and `CUDA.cu`, which convert floating-point arrays to `Float32`.

## CPU results

```text
===============================================================================================================================
                                                                     first gradient                     warm gradient
                                                           ---------------------------------  ---------------------------------
Model                                                           Zygote   Mooncake   Mc / Zyg       Zygote   Mooncake   Mc / Zyg
-------------------------------------------------------------------------------------------------------------------------------
Dense(2 => 4)                                                 693.0 ms     12.5 s     17.99x       4.0 us   976.0 ns      0.24x
Chain(Dense(2 => 4, tanh), Dense(4 => 3))                       1.05 s     16.1 s     15.31x      8.42 us    3.46 us      0.41x
f64(Chain(Dense(2 => 4), Dense(4 => 2)))                      784.0 ms     13.1 s     16.67x      8.34 us    2.08 us      0.25x
Flux.Scale(4, abs2)                                           828.0 ms     15.6 s     18.89x      3.66 us    4.67 us      1.28x
Conv((3, 3), 2 => 3)                                            1.42 s     17.8 s     12.55x      22.4 us    20.7 us      0.92x
Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))         1.88 s     20.7 s     11.04x      69.4 us    64.1 us      0.92x
Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))                   2.93 s     20.3 s      6.93x     307.0 us   300.0 us      0.98x
Maxout(Dense(5 => 4, tanh), 3)                                  4.76 s     18.0 s      3.78x     138.0 us    11.4 us      0.08x
SkipConnection(Dense(2 => 2), vcat)                           883.0 ms     17.5 s     19.77x      5.68 us    5.84 us      1.03x
Bilinear((2, 2) => 3)                                           1.29 s     15.8 s     12.28x      14.9 us    3.06 us      0.21x
ConvTranspose((3, 3), 3 => 2)                                   1.43 s     18.0 s     12.61x      62.1 us    62.8 us      1.01x
LayerNorm(2)                                                    3.88 s     23.7 s      6.11x      38.8 us    55.0 us      1.42x
BatchNorm(2)                                                    1.45 s     22.4 s     15.45x      15.4 us    17.8 us      1.15x
MultiHeadAttention(16)                                          13.0 s     27.9 s      2.15x      2.22 ms    3.18 ms      1.44x
RNN(3 => 2)                                                     2.87 s     36.9 s     12.86x      1.62 ms   311.0 us      0.19x
LSTM(3 => 5)                                                    5.02 s     49.6 s      9.88x      4.11 ms    1.73 ms      0.42x
GRU(3 => 5)                                                     5.23 s     52.4 s     10.02x      39.8 ms    19.1 ms      0.48x
Chain(RNN(3 => 4), RNN(4 => 3))                                 3.08 s     39.8 s     12.93x      3.45 ms   633.0 us      0.18x
Chain(LSTM(3 => 5), LSTM(5 => 3))                               5.23 s     51.4 s      9.82x      7.64 ms    3.59 ms      0.47x
===============================================================================================================================
```

Across the 19 supported models, the geometric-mean Mooncake/Zygote ratio is 10.62 for first-gradient time and 0.518 for warm-gradient time.
Mooncake is 1.93 times faster on warm gradients by geometric mean.

## GPU results

```text
===============================================================================================================================
                                                                     first gradient                     warm gradient
                                                           ---------------------------------  ---------------------------------
Model                                                           Zygote   Mooncake   Mc / Zyg       Zygote   Mooncake   Mc / Zyg
-------------------------------------------------------------------------------------------------------------------------------
Dense(2 => 4)                                                   5.05 s     19.4 s      3.84x     106.0 us   115.0 us      1.09x
Chain(Dense(2 => 4, tanh), Dense(4 => 3))                       5.81 s     22.1 s      3.80x     174.0 us   231.0 us      1.33x
f64(Chain(Dense(2 => 4), Dense(4 => 2)))                        5.13 s     19.8 s      3.85x     170.0 us   188.0 us      1.11x
Flux.Scale(4, abs2)                                             4.73 s     18.7 s      3.96x      94.1 us   128.0 us      1.36x
Conv((3, 3), 2 => 3)                                            8.39 s     26.5 s      3.16x     115.0 us   150.0 us      1.31x
Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))         10.9 s     31.5 s      2.90x     197.0 us   299.0 us      1.52x
Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))                   10.9 s     30.4 s      2.80x     200.0 us   222.0 us      1.11x
Maxout(Dense(5 => 4, tanh), 3)                                  10.2 s     22.1 s      2.17x     458.0 us   447.0 us      0.98x
SkipConnection(Dense(2 => 2), vcat)                             6.19 s     21.6 s      3.48x     135.0 us   149.0 us      1.10x
Bilinear((2, 2) => 3)                                           6.11 s     22.1 s      3.61x     155.0 us   159.0 us      1.03x
ConvTranspose((3, 3), 3 => 2)                                   8.63 s     27.3 s      3.17x     152.0 us   181.0 us      1.19x
LayerNorm(2)                                                    11.7 s     29.7 s      2.53x     246.0 us   287.0 us      1.17x
BatchNorm(2)                                                    7.51 s     22.2 s      2.96x     100.0 us   117.0 us      1.17x
MultiHeadAttention(16)                                          22.8 s     50.6 s      2.22x     689.0 us   580.0 us      0.84x
RNN(3 => 2)                                                     8.85 s     40.8 s      4.61x      7.86 ms    10.9 ms      1.39x
LSTM(3 => 5)                                                    11.7 s     54.4 s      4.64x      18.1 ms    21.8 ms      1.21x
GRU(3 => 5)                                                     12.9 s     56.7 s      4.40x     127.0 ms   149.0 ms      1.17x
Chain(RNN(3 => 4), RNN(4 => 3))                                 9.03 s     44.5 s      4.92x      15.9 ms    21.9 ms      1.38x
Chain(LSTM(3 => 5), LSTM(5 => 3))                               12.2 s     56.7 s      4.64x      36.2 ms    43.7 ms      1.21x
===============================================================================================================================
```

Across the 19 supported models, the geometric-mean Mooncake/Zygote ratio is 3.46 for first-gradient time and 1.181 for warm-gradient time.
Mooncake is 1.18 times slower on warm gradients by geometric mean.
