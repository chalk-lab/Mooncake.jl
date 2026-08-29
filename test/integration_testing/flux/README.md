# Flux CPU gradient benchmarks

- Julia 1.12.7
- Flux 0.16.11
- Mooncake 0.5.49
- aarch64-linux-gnu
- 1 Julia thread
- a 2.0-second warm sampling budget per model and backend

Each first-gradient measurement ran in a fresh Julia process after packages and model values had
loaded. Mooncake's first-gradient time includes `prepare_gradient_cache` and the first
`value_and_gradient!!` call. Warm Mooncake measurements reuse the prepared cache. `Mc / Zyg` is
Mooncake time divided by Zygote time, so values below one favour Mooncake.

```text
================================================================================================================================
                                                                    first gradient                       warm gradient
                                                          ----------------------------------  ----------------------------------
Model                                                         Zygote    Mooncake    Mc / Zyg      Zygote    Mooncake    Mc / Zyg
--------------------------------------------------------------------------------------------------------------------------------
Dense(2 => 4)                                               555.0 ms      18.6 s      33.51x     3.36 us    528.0 ns       0.16x
Chain(Dense(2 => 4, tanh), Dense(4 => 3))                   898.0 ms      19.3 s      21.49x     7.57 us    832.0 ns       0.11x
f64(Chain(Dense(2 => 4), Dense(4 => 2)))                    779.0 ms      13.1 s      16.82x     7.65 us      1.5 us       0.20x
Flux.Scale(4, abs2)                                         667.0 ms      15.2 s      22.79x     2.03 us    352.0 ns       0.17x
Conv((3, 3), 2 => 3)                                          1.26 s      17.6 s      13.97x     3.12 us     1.92 us       0.62x
Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh))       1.87 s      20.5 s      10.96x     10.3 us     4.26 us       0.41x
Chain(Conv((4, 4), 2 => 2), MeanPool((5, 5)))                 2.75 s      20.1 s       7.31x     18.4 us     12.2 us       0.66x
Maxout(Dense(5 => 4, tanh), 3)                                4.79 s      17.9 s       3.74x    128.0 us     1.14 us       0.01x
SkipConnection(Dense(2 => 2), vcat)                         901.0 ms      17.6 s      19.53x     4.45 us    912.0 ns       0.20x
Bilinear((2, 2) => 3)                                          1.4 s      15.6 s      11.14x     12.8 us     1.25 us       0.10x
ConvTranspose((3, 3), 3 => 2)                                 1.27 s      17.9 s      14.09x     4.22 us     3.46 us       0.82x
LayerNorm(2)                                                  3.84 s      23.6 s       6.15x     27.4 us      2.3 us       0.08x
BatchNorm(2)                                                  1.44 s      21.9 s      15.21x     11.6 us     3.12 us       0.27x
MultiHeadAttention(16)                                        11.3 s      27.7 s       2.45x    214.0 us     98.4 us       0.46x
RNN(3 => 2)                                                   2.74 s      37.2 s      13.58x     51.0 us     22.7 us       0.45x
LSTM(3 => 5)                                                  5.01 s      48.4 s       9.66x    129.0 us     78.0 us       0.60x
GRU(3 => 5)                                                   5.15 s      51.0 s       9.90x    926.0 us    437.0 us       0.47x
Chain(RNN(3 => 4), RNN(4 => 3))                               2.91 s      38.8 s      13.33x    107.0 us     46.7 us       0.44x
Chain(LSTM(3 => 5), LSTM(5 => 3))                             5.18 s      50.1 s       9.67x    262.0 us    161.0 us       0.61x
================================================================================================================================
```

Across the 19 models, the geometric-mean Mooncake/Zygote ratio is 11.47 for first-gradient time and
0.255 for warm-gradient time. Equivalently, Mooncake is 3.93 times faster on warm gradients by
geometric mean.
