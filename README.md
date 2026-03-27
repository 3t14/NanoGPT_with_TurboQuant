# NanoGPT with TurboQuant

[nanoGPT](https://github.com/karpathy/nanoGPT) + [TurboQuant](https://arxiv.org/abs/2504.19874) KV cache compression for faster inference.

## What is TurboQuant?

TurboQuant (Google Research, ICLR 2026) compresses the Key-Value cache in transformer inference from 16-bit to ~3-bit per entry with near-zero quality loss:

- **Stage 1 (PolarQuant):** Random orthogonal rotation + Lloyd-Max optimal scalar quantization
- **Stage 2 (QJL):** 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased attention scores

Result: **~5x KV cache compression** at 3-bit, enabling longer sequences and larger batch sizes.

## Quick Start

```bash
# Prepare data
python data/shakespeare_char/prepare.py

# Train (standard)
python train.py config/train_shakespeare_char.py

# Train with TurboQuant evaluation
python train.py config/train_shakespeare_char_turboquant.py

# Sample with TurboQuant
python sample.py --out_dir=out-shakespeare-char --use_turboquant=True --turboquant_bits=3

# Compare standard vs TurboQuant quality
python sample.py --out_dir=out-shakespeare-char --compare_turboquant=True

# Run benchmarks
python benchmark_turboquant.py
```

## TurboQuant Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_turboquant` | `False` | Enable TurboQuant KV cache compression |
| `turboquant_bits` | `3` | Bits per coordinate (2=aggressive, 3=quality-neutral, 4=conservative) |

TurboQuant is applied **during inference only** — training uses standard full-precision attention to preserve gradient flow.

## How It Works

### Integration Points

1. **`turboquant.py`** — Self-contained TurboQuant implementation:
   - Lloyd-Max codebook solver (precomputed per dimension/bit-width)
   - Random rotation matrix generation (Haar-distributed via QR)
   - QJL projection for unbiased inner products
   - `TurboQuantKVCompressor` class for KV cache compression

2. **`model.py`** — Modified `CausalSelfAttention` with TurboQuant path:
   - During inference with TurboQuant enabled, keys and values are compressed
   - Attention scores computed directly from compressed keys (asymmetric estimator)
   - Values decompressed only after softmax weighting (MSE errors average out)

3. **`train.py`** / **`sample.py`** — TurboQuant configuration options

### Compression Details

For 3-bit quantization with head_dim=64:
- **Keys:** 2-bit MSE indices + 1-bit QJL signs + 16-bit norm per vector ≈ 3.25 bits/entry
- **Values:** 3-bit MSE indices + 16-bit norm per vector ≈ 3.25 bits/entry
- **vs FP16:** 16 bits/entry → **~4.9x compression**

## File Structure

```
├── model.py                    # GPT model with TurboQuant attention
├── turboquant.py               # TurboQuant implementation
├── train.py                    # Training script
├── sample.py                   # Sampling/inference script
├── benchmark_turboquant.py     # TurboQuant benchmarks
├── configurator.py             # Config override system
├── config/
│   ├── train_shakespeare_char.py           # Standard training config
│   └── train_shakespeare_char_turboquant.py # TurboQuant training config
└── data/
    └── shakespeare_char/
        └── prepare.py          # Data preparation script
```

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrey Karpathy
- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni (ICLR 2026)
- [PolarQuant](https://arxiv.org/abs/2502.02617) — KV cache quantization with polar transformation
- [turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — Reference PyTorch implementation
