# Flux CPU gradient benchmarks

- Julia 1.12.7, Flux 0.16.11, Mooncake 0.5.49
- aarch64-linux-gnu
- CPU: Cortex-A725 + Cortex-X925 (20 logical threads)
- 1 Julia thread
- BLAS: lbt, 1 thread(s)
- a 2.0-second warm sampling budget per model and backend

Each first-gradient measurement ran in a fresh Julia process after packages and model values had loaded. Mooncake's first-gradient time includes `prepare_gradient_cache` and the first `value_and_gradient!!` call. Warm Mooncake measurements reuse the prepared cache. `Mc / Zyg` is Mooncake time divided by Zygote time, so values below one favour Mooncake.

```text
===============================================================================================================================
                                                                     first gradient                     warm gradient
                                                           ---------------------------------  ---------------------------------
Model                                                           Zygote   Mooncake   Mc / Zyg       Zygote   Mooncake   Mc / Zyg
-------------------------------------------------------------------------------------------------------------------------------
Dense(2 => 4)                                                 552.0 ms     18.0 s     32.56x      3.39 us   512.0 ns      0.15x
Chain(Dense(2 => 4, tanh), Dense(4 => 3))                     886.0 ms     19.3 s     21.80x      7.46 us   832.0 ns      0.11x
f64(Chain(Dense(2 => 4), Dense(4 => 2)))                      778.0 ms     13.3 s     17.11x      7.58 us    1.49 us      0.20x
Flux.Scale(4, abs2)                                           667.0 ms     15.7 s     23.60x      2.05 us   368.0 ns      0.18x
Conv((3, 3), 2 => 3)                                             1.3 s     17.8 s     13.70x      3.09 us    1.95 us      0.63x
Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))         1.86 s     20.6 s     11.06x      10.3 us    4.29 us      0.42x
Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))                   2.77 s     20.1 s      7.28x      17.8 us    12.2 us      0.68x
Maxout(Dense(5 => 4, tanh), 3)                                  4.75 s     18.1 s      3.81x     128.0 us    1.14 us      0.01x
SkipConnection(Dense(2 => 2), vcat)                           900.0 ms     17.7 s     19.63x      4.42 us   912.0 ns      0.21x
Bilinear((2, 2) => 3)                                           1.38 s     15.4 s     11.17x      12.7 us    1.22 us      0.10x
ConvTranspose((3, 3), 3 => 2)                                   1.29 s     17.9 s     13.85x      4.22 us    3.49 us      0.83x
LayerNorm(2)                                                    3.84 s     23.6 s      6.14x      27.5 us    2.34 us      0.08x
BatchNorm(2)                                                    1.43 s     22.0 s     15.40x      11.6 us    3.14 us      0.27x
MultiHeadAttention(16)                                          11.3 s     27.7 s      2.44x     211.0 us    97.5 us      0.46x
RNN(3 => 2)                                                     2.72 s     36.2 s     13.30x      51.4 us    22.8 us      0.44x
LSTM(3 => 5)                                                    4.98 s     48.4 s      9.72x     129.0 us    77.9 us      0.60x
GRU(3 => 5)                                                     5.19 s     51.7 s      9.95x     949.0 us   448.0 us      0.47x
Chain(RNN(3 => 4), RNN(4 => 3))                                 2.89 s     39.1 s     13.50x     107.0 us    46.5 us      0.44x
Chain(LSTM(3 => 5), LSTM(5 => 3))                               5.16 s     50.0 s      9.68x     263.0 us   166.0 us      0.63x
===============================================================================================================================
```

Across the 19 models, the geometric-mean Mooncake/Zygote ratio is 11.50 for first-gradient time and 0.256 for warm-gradient time. Equivalently, Mooncake is 3.91 times faster on warm gradients by geometric mean.
