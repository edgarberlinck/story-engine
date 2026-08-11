# Video Generation Caveats

Notes on running Wan 2.2 I2V A14B and LTX-Video on this machine.

## Machine

- Apple M5 Pro MacBook Pro (Mac17,9) — 18-core CPU / 20-core GPU, Metal 4
- 64 GB unified memory (~1.3 TB free disk)
- torch 2.13.0 with MPS available, diffusers 0.39.0
- `generators/video_generator.py` targets both `wan22_i2v` and `ltx_video` via diffusers

## Verdict

Both models run. The constraint is throughput, not memory.

## LTX-Video (2B) — comfortable yes

- ~4–5 GB of weights at FP16; trivial vs. 64 GB
- 121 frames @ 704×480 is well within reach
- Official MLX support (`mlx-video`) → near-native Metal performance;
  typically seconds to a couple minutes per clip

## Wan 2.2 I2V A14B (14B) — runs, but only if quantized

| Variant | Weight size | Verdict on 64 GB |
|---|---|---|
| FP16 via diffusers/MPS | ~28 GB + text encoder + Wan-VAE + activations | Fits, but slow — MPS falls back to CPU for many ops in the Wan pipeline |
| MLX / GGUF Q4–Q5 | ~9–12 GB | Sweet spot — comfortable headroom, ~2–5 min per 81-frame @ 832×480 clip |

## Practical recommendations

1. **Skip torch-MPS for Wan 2.2.** Diffusers' Wan2.2 pipeline on Metal is
   partially unaccelerated; the 20-core GPU is left mostly idle. Use
   `mlx-wan` (or ComfyUI + GGUF) — MLX is built directly on Metal and is
   dramatically faster.
2. **LTX-Video via `mlx-video`** is the easy win; the original 2B runs great
   on M-series.
3. Thermal note: sustained 14B inference will throttle the M5 Pro over long
   runs, but a 40-step @ Q4 generation finishes before that becomes a real
   problem.
4. Memory is a non-issue for both — both models could run simultaneously.
   The practical ceiling is speed, not capacity.

## Bottom line

- LTX-Video: definitely.
- Wan 2.2 A14B: yes, with 4/5-bit quantization via MLX — diffusers-MPS is
  not a daily driver for it.