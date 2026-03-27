"""
Benchmark TurboQuant KV cache compression.

Tests quantization quality, compression ratio, and speed across different
bit-widths and head dimensions.

Usage:
    python benchmark_turboquant.py
    python benchmark_turboquant.py --device=cuda
"""
import sys
import time
import torch
import math

from turboquant import TurboQuantKVCompressor, solve_lloyd_max, get_codebook


def benchmark_lloyd_max():
    """Test Lloyd-Max codebook quality."""
    print("=" * 60)
    print("Lloyd-Max Codebook Quality")
    print("=" * 60)

    for d in [64, 128, 256]:
        for bits in [2, 3, 4]:
            centroids, boundaries = get_codebook(d, bits)
            n_levels = 2 ** bits
            sigma = 1.0 / math.sqrt(d)
            print(f"  d={d:4d}, bits={bits}: {n_levels} levels, "
                  f"range=[{centroids[0]:.6f}, {centroids[-1]:.6f}], "
                  f"sigma={sigma:.6f}")


def benchmark_compression_quality(device='cpu'):
    """Measure MSE and inner product preservation."""
    print("\n" + "=" * 60)
    print("Compression Quality (MSE & Inner Product Fidelity)")
    print("=" * 60)

    B, H, T, D = 2, 8, 128, 64
    torch.manual_seed(42)

    for bits in [2, 3, 4]:
        compressor = TurboQuantKVCompressor(D, bits=bits, device=device)

        # Random KV tensors
        k = torch.randn(B, H, T, D, device=device)
        v = torch.randn(B, H, T, D, device=device)
        q = torch.randn(B, H, T, D, device=device)

        # Compress and decompress
        ck = compressor.compress_keys(k)
        cv = compressor.compress_values(v)
        v_hat = compressor.decompress_values(cv)

        # MSE for values
        mse_v = ((v - v_hat) ** 2).mean().item()

        # Inner product fidelity for keys
        # True attention scores
        scale = 1.0 / math.sqrt(D)
        true_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        tq_scores = compressor.attention_scores(q, ck)

        score_mse = ((true_scores - tq_scores) ** 2).mean().item()
        # Cosine similarity of flattened score matrices
        cos_sim = torch.nn.functional.cosine_similarity(
            true_scores.reshape(-1).unsqueeze(0),
            tq_scores.reshape(-1).unsqueeze(0)
        ).item()

        info = compressor.memory_savings_info(T, H)

        print(f"\n  {bits}-bit TurboQuant:")
        print(f"    Value MSE:              {mse_v:.6f}")
        print(f"    Attention score MSE:    {score_mse:.6f}")
        print(f"    Attention cosine sim:   {cos_sim:.6f}")
        print(f"    Compression ratio:      {info['compression_ratio']:.2f}x")
        print(f"    Effective bits/entry:   {info['bits_per_entry']:.2f}")


def benchmark_speed(device='cpu'):
    """Benchmark compression and attention computation speed."""
    print("\n" + "=" * 60)
    print(f"Speed Benchmark (device={device})")
    print("=" * 60)

    configs = [
        (2, 12, 512, 64, "GPT-2 small"),
        (2, 12, 1024, 64, "GPT-2 small (long)"),
        (1, 16, 512, 64, "GPT-2 medium"),
    ]

    for B, H, T, D, name in configs:
        print(f"\n  Config: {name} (B={B}, H={H}, T={T}, D={D})")

        k = torch.randn(B, H, T, D, device=device)
        v = torch.randn(B, H, T, D, device=device)
        q = torch.randn(B, H, T, D, device=device)

        for bits in [3, 4]:
            compressor = TurboQuantKVCompressor(D, bits=bits, device=device)

            # Warmup
            ck = compressor.compress_keys(k)
            cv = compressor.compress_values(v)
            _ = compressor.attention_scores(q, ck)
            _ = compressor.decompress_values(cv)

            if device == 'cuda':
                torch.cuda.synchronize()

            # Benchmark compression
            n_iter = 10
            t0 = time.time()
            for _ in range(n_iter):
                ck = compressor.compress_keys(k)
                cv = compressor.compress_values(v)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_compress = (time.time() - t0) / n_iter

            # Benchmark attention
            t0 = time.time()
            for _ in range(n_iter):
                scores = compressor.attention_scores(q, ck)
                v_hat = compressor.decompress_values(cv)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_attn = (time.time() - t0) / n_iter

            # Standard attention for comparison
            t0 = time.time()
            for _ in range(n_iter):
                scale = 1.0 / math.sqrt(D)
                s = torch.matmul(q, k.transpose(-2, -1)) * scale
                _ = torch.matmul(torch.softmax(s, dim=-1), v)
            if device == 'cuda':
                torch.cuda.synchronize()
            t_standard = (time.time() - t0) / n_iter

            info = compressor.memory_savings_info(T, H)
            print(f"    {bits}-bit: compress={t_compress * 1000:.1f}ms, "
                  f"tq_attn={t_attn * 1000:.1f}ms, "
                  f"std_attn={t_standard * 1000:.1f}ms, "
                  f"ratio={info['compression_ratio']:.1f}x")


if __name__ == '__main__':
    device = 'cpu'
    for arg in sys.argv[1:]:
        if arg.startswith('--device='):
            device = arg.split('=')[1]

    print("TurboQuant Benchmark")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    benchmark_lloyd_max()
    benchmark_compression_quality(device)
    benchmark_speed(device)

    print("\nDone!")
