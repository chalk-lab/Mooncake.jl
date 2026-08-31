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
Julia 1.12.7, Flux 0.16.11, Mooncake 0.5.51 (ed99f98b0-dirty), CUDA 6.3.1, cuDNN 6.3.1
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
Dense(2 => 4)                                                 693.0 ms     12.5 s     18.02x      3.98 us   960.0 ns      0.24x
Chain(Dense(2 => 4, tanh), Dense(4 => 3))                       1.05 s     16.2 s     15.52x      8.27 us    3.44 us      0.42x
f64(Chain(Dense(2 => 4), Dense(4 => 2)))                      783.0 ms     13.1 s     16.74x      8.26 us    1.76 us      0.21x
Flux.Scale(4, abs2)                                           828.0 ms     15.7 s     18.91x      3.71 us    4.69 us      1.26x
Conv((3, 3), 2 => 3)                                            1.42 s     17.7 s     12.50x      22.3 us    20.5 us      0.92x
Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))         1.87 s     20.4 s     10.89x      69.5 us    64.7 us      0.93x
Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))                   2.92 s     20.3 s      6.95x     308.0 us   300.0 us      0.98x
Maxout(Dense(5 => 4, tanh), 3)                                  4.76 s     18.0 s      3.79x     136.0 us    11.4 us      0.08x
SkipConnection(Dense(2 => 2), vcat)                           881.0 ms     17.5 s     19.82x      5.65 us    5.84 us      1.03x
Bilinear((2, 2) => 3)                                           1.28 s     15.5 s     12.07x      14.9 us    2.99 us      0.20x
ConvTranspose((3, 3), 3 => 2)                                   1.45 s     18.2 s     12.57x      62.6 us    62.8 us      1.00x
LayerNorm(2)                                                     3.9 s     23.8 s      6.11x      39.4 us    54.6 us      1.38x
BatchNorm(2)                                                    1.45 s     20.6 s     14.20x      15.3 us    10.1 us      0.66x
MultiHeadAttention(16)                                          13.0 s     28.0 s      2.16x      2.23 ms    3.06 ms      1.37x
RNN(3 => 2)                                                     2.89 s     37.0 s     12.80x      1.62 ms   313.0 us      0.19x
LSTM(3 => 5)                                                    5.02 s     49.7 s      9.88x      4.04 ms    1.75 ms      0.43x
GRU(3 => 5)                                                     5.24 s     52.4 s     10.00x      39.9 ms    18.1 ms      0.45x
Chain(RNN(3 => 4), RNN(4 => 3))                                 3.09 s     39.8 s     12.87x      3.46 ms   638.0 us      0.18x
Chain(LSTM(3 => 5), LSTM(5 => 3))                               5.23 s     51.4 s      9.82x      8.27 ms    3.56 ms      0.43x
===============================================================================================================================
```

Across the 19 supported models, the geometric-mean Mooncake/Zygote ratio is 10.57 for first-gradient time and 0.493 for warm-gradient time.
Mooncake is 2.03 times faster on warm gradients by geometric mean.

## GPU results

```text
===============================================================================================================================
                                                                     first gradient                     warm gradient
                                                           ---------------------------------  ---------------------------------
Model                                                           Zygote   Mooncake   Mc / Zyg       Zygote   Mooncake   Mc / Zyg
-------------------------------------------------------------------------------------------------------------------------------
Dense(2 => 4)                                                   5.21 s     18.2 s      3.49x     106.0 us   105.0 us      0.99x
Chain(Dense(2 => 4, tanh), Dense(4 => 3))                       5.64 s     19.3 s      3.42x     174.0 us   186.0 us      1.07x
f64(Chain(Dense(2 => 4), Dense(4 => 2)))                        5.16 s     18.4 s      3.55x     166.0 us   179.0 us      1.08x
Flux.Scale(4, abs2)                                             4.92 s     17.6 s      3.58x      94.8 us    98.3 us      1.04x
Conv((3, 3), 2 => 3)                                            8.25 s     25.1 s      3.04x     116.0 us   121.0 us      1.05x
Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))         10.8 s     28.3 s      2.63x     203.0 us   222.0 us      1.10x
Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))                   10.9 s     28.8 s      2.64x     197.0 us   195.0 us      0.99x
Maxout(Dense(5 => 4, tanh), 3)                                  10.3 s     20.7 s      2.02x     454.0 us   331.0 us      0.73x
SkipConnection(Dense(2 => 2), vcat)                             6.09 s     19.2 s      3.15x     130.0 us   132.0 us      1.02x
Bilinear((2, 2) => 3)                                           5.77 s     20.2 s      3.51x     153.0 us   137.0 us      0.90x
ConvTranspose((3, 3), 3 => 2)                                    8.5 s     25.9 s      3.04x     146.0 us   153.0 us      1.05x
LayerNorm(2)                                                    11.2 s     26.4 s      2.35x     242.0 us   215.0 us      0.89x
BatchNorm(2)                                                    7.17 s     21.0 s      2.93x      98.4 us   101.0 us      1.03x
MultiHeadAttention(16)                                          22.7 s     45.9 s      2.03x     685.0 us   521.0 us      0.76x
RNN(3 => 2)                                                     8.48 s     37.5 s      4.42x      7.68 ms    7.83 ms      1.02x
LSTM(3 => 5)                                                    11.4 s     50.8 s      4.46x      17.4 ms    15.3 ms      0.88x
GRU(3 => 5)                                                     12.1 s     53.2 s      4.38x     120.0 ms    93.7 ms      0.78x
Chain(RNN(3 => 4), RNN(4 => 3))                                 8.59 s     40.7 s      4.74x      15.5 ms    15.8 ms      1.02x
Chain(LSTM(3 => 5), LSTM(5 => 3))                               11.4 s     52.8 s      4.61x      35.8 ms    29.7 ms      0.83x
===============================================================================================================================
```

Across the 19 supported models, the geometric-mean Mooncake/Zygote ratio is 3.26 for first-gradient time and 0.952 for warm-gradient time.
Mooncake is 1.05 times faster on warm gradients by geometric mean.
