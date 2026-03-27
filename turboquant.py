"""
TurboQuant: KV cache compression for nanoGPT.

Implements the two-stage vector quantization from:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)

Stage 1 (MSE): Random orthogonal rotation + per-coordinate Lloyd-Max quantization
Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

Key insight: After random rotation, each coordinate of a unit vector follows a known
distribution (approx N(0, 1/d) for large d). This allows precomputing an optimal
scalar quantizer (Lloyd-Max) without any calibration data.
"""

import torch
import math
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Lloyd-Max codebook solver (Gaussian approximation for d >= 32)
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve Lloyd-Max optimal quantizer for the coordinate distribution N(0, 1/d).

    Uses numerical integration (midpoint rule with fine grid) to avoid scipy dependency.

    Returns:
        centroids: (2^bits,) sorted optimal centroids
        boundaries: (2^bits - 1,) sorted boundaries
    """
    n_levels = 2 ** bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    lo, hi = -4.0 * sigma, 4.0 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    # Fine grid for numerical integration
    n_grid = 10000
    grid = [lo * 1.5 + (hi * 1.5 - lo * 1.5) * i / n_grid for i in range(n_grid + 1)]
    dx = (hi * 1.5 - lo * 1.5) / n_grid

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = 0.0
            den = 0.0
            for j in range(n_grid):
                x_mid = (grid[j] + grid[j + 1]) / 2.0
                if a <= x_mid <= b:
                    p = _gaussian_pdf(x_mid, sigma2) * dx
                    num += x_mid * p
                    den += p
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Codebook cache: precomputed per (d, bits) pair
# ---------------------------------------------------------------------------

_CODEBOOK_CACHE: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def get_codebook(d: int, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get or compute Lloyd-Max codebook for given dimension and bits."""
    key = (d, bits)
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = solve_lloyd_max(d, bits)
    return _CODEBOOK_CACHE[key]


# ---------------------------------------------------------------------------
# Random matrix generation
# ---------------------------------------------------------------------------

def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR decomposition."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q


def generate_qjl_matrix(d: int, seed: int = 43) -> torch.Tensor:
    """Generate random Gaussian projection matrix for QJL."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(d, d, generator=gen)


# ---------------------------------------------------------------------------
# TurboQuant KV Cache Compressor
# ---------------------------------------------------------------------------

class TurboQuantKVCompressor:
    """
    Compresses KV cache using TurboQuant two-stage quantization.

    For keys: TurboQuant-Prod (MSE + QJL) for unbiased attention scores.
    For values: TurboQuant-MSE only (reconstruction averaging reduces error).

    Usage in attention:
        compressor = TurboQuantKVCompressor(head_dim=64, bits=3)
        compressed_k = compressor.compress_keys(k)
        compressed_v = compressor.compress_values(v)
        attn_scores = compressor.attention_scores(q, compressed_k)
        v_decompressed = compressor.decompress_values(compressed_v)
    """

    def __init__(self, head_dim: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        # Precompute rotation matrices (separate for keys and values)
        self.Pi_k = generate_rotation_matrix(head_dim, seed=seed).to(device)
        self.Pi_v = generate_rotation_matrix(head_dim, seed=seed + 100).to(device)

        # QJL projection matrix for keys
        self.S = generate_qjl_matrix(head_dim, seed=seed + 200).to(device)

        # Lloyd-Max codebooks
        centroids_k, _ = get_codebook(head_dim, self.mse_bits)
        centroids_v, _ = get_codebook(head_dim, bits)
        self.centroids_k = centroids_k.to(device)
        self.centroids_v = centroids_v.to(device)

    def to(self, device: str) -> 'TurboQuantKVCompressor':
        """Move all tensors to device."""
        self.device = device
        self.Pi_k = self.Pi_k.to(device)
        self.Pi_v = self.Pi_v.to(device)
        self.S = self.S.to(device)
        self.centroids_k = self.centroids_k.to(device)
        self.centroids_v = self.centroids_v.to(device)
        return self

    @torch.no_grad()
    def compress_keys(self, keys: torch.Tensor) -> dict:
        """
        Compress keys using TurboQuant-Prod (MSE + QJL).

        Args:
            keys: (B, n_head, T, head_dim)

        Returns:
            dict with k_mse (reconstructed), qjl_signs, residual_norm
        """
        B, H, T, D = keys.shape
        flat = keys.reshape(-1, D).float()

        # Normalize per-vector
        norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_normed = flat / (norms + 1e-8)

        # Stage 1: Rotate and quantize
        rotated = flat_normed @ self.Pi_k.T
        diffs = rotated.unsqueeze(-1) - self.centroids_k  # (N, D, n_levels)
        indices = diffs.abs().argmin(dim=-1)  # (N, D)
        reconstructed_rot = self.centroids_k[indices]  # (N, D)
        k_mse = (reconstructed_rot @ self.Pi_k) * norms  # back to original space

        # Stage 2: QJL on residual
        residual = flat - k_mse
        r_norm = torch.norm(residual, dim=-1)  # (N,)
        projected = residual @ self.S.T  # (N, D)
        signs = (projected >= 0).to(torch.int8) * 2 - 1  # {-1, +1}

        return {
            "k_mse": k_mse.to(keys.dtype).reshape(B, H, T, D),
            "qjl_signs": signs.reshape(B, H, T, D),
            "residual_norm": r_norm.to(keys.dtype).reshape(B, H, T),
        }

    @torch.no_grad()
    def compress_values(self, values: torch.Tensor) -> dict:
        """
        Compress values using TurboQuant-MSE only.

        Args:
            values: (B, n_head, T, head_dim)

        Returns:
            dict with indices and norms for reconstruction
        """
        B, H, T, D = values.shape
        flat = values.reshape(-1, D).float()

        norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_normed = flat / (norms + 1e-8)

        rotated = flat_normed @ self.Pi_v.T
        diffs = rotated.unsqueeze(-1) - self.centroids_v
        indices = diffs.abs().argmin(dim=-1).to(torch.uint8)

        return {
            "indices": indices,
            "norms": norms.squeeze(-1).to(values.dtype),
            "shape": (B, H, T, D),
        }

    @torch.no_grad()
    def decompress_values(self, compressed: dict) -> torch.Tensor:
        """Decompress values from MSE quantized representation."""
        B, H, T, D = compressed["shape"]
        indices = compressed["indices"].long()
        reconstructed = self.centroids_v[indices] @ self.Pi_v  # (N, D)
        norms = compressed["norms"].float().unsqueeze(-1)  # (N, 1)
        return (reconstructed * norms).reshape(B, H, T, D)

    @torch.no_grad()
    def attention_scores(self, queries: torch.Tensor, compressed_keys: dict, scale: float = None) -> torch.Tensor:
        """
        Compute scaled attention scores directly from compressed keys.

        Uses asymmetric estimator:
            <q, k> ~ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, sign(S@r_k)>

        Args:
            queries: (B, n_head, T_q, head_dim) - NOT compressed
            compressed_keys: dict from compress_keys()
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            scores: (B, n_head, T_q, T_k) attention scores
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        k_mse = compressed_keys["k_mse"].float()
        signs = compressed_keys["qjl_signs"].float()
        r_norm = compressed_keys["residual_norm"].float()

        q = queries.float()

        # Term 1: Q @ K_mse^T
        term1 = torch.matmul(q, k_mse.transpose(-2, -1))

        # Term 2: QJL correction
        q_proj = torch.matmul(q, self.S.T)  # (B, H, T_q, D)
        qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))  # (B, H, T_q, T_k)

        m = self.S.shape[0]
        correction = math.sqrt(math.pi / 2) / m
        term2 = correction * qjl_ip * r_norm.unsqueeze(-2)  # broadcast r_norm over T_q

        return (term1 + term2) * scale

    def memory_savings_info(self, seq_len: int, n_heads: int) -> dict:
        """Estimate compression ratio."""
        fp16_bits_per_entry = 16
        total_entries = seq_len * n_heads * self.head_dim

        # Keys: mse_bits per coord + 1 bit QJL sign + 16 bits norm per vector
        key_bits = total_entries * self.mse_bits + total_entries * 1 + seq_len * n_heads * 16
        # Values: bits per coord + 16 bits norm per vector
        value_bits = total_entries * self.bits + seq_len * n_heads * 16

        fp16_total = total_entries * 2 * fp16_bits_per_entry  # keys + values
        compressed_total = key_bits + value_bits

        return {
            "fp16_bits": fp16_total,
            "compressed_bits": compressed_total,
            "compression_ratio": fp16_total / compressed_total if compressed_total > 0 else 0,
            "bits_per_entry": compressed_total / (total_entries * 2) if total_entries > 0 else 0,
        }
