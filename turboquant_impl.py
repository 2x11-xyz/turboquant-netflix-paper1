"""
Minimal TurboQuant implementation from the paper (arXiv:2504.19874).
Implements TurboQuant_IP: one-sided unbiased inner product estimator.

Only depends on torch. No external quantization libraries.

Reference equations:
- Definition 1 (QJL): Q_qjl(x) = sign(Sx), Q_qjl^{-1}(z) = coeff * S^T z
- Theorem 1 (MSE): Near-optimal scalar quantization after random rotation
- Theorem 2 (IP): E[<y, TQ(x)>] = <y, x> (unbiased one-sided inner product)
"""
import math
import torch
import numpy as np
from scipy.stats import norm as scipy_norm


def _random_orthogonal(d: int, generator: torch.Generator) -> torch.Tensor:
    """Sample a Haar-random orthogonal matrix via QR of Gaussian."""
    G = torch.randn(d, d, generator=generator)
    Q, R = torch.linalg.qr(G)
    # Fix sign ambiguity so Q is truly Haar-distributed
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    return Q


def _lloyd_max_codebook(d: int, bits: int) -> torch.Tensor:
    """
    Compute optimal Lloyd-Max centroids for the marginal distribution of
    a coordinate of a uniform point on S^{d-1}.

    The marginal distribution approaches N(0, 1/d) for large d.
    For exactness at small d, we use the beta-derived density.
    """
    n_centroids = 2 ** bits
    # For the unit sphere in R^d, each coordinate x_i has marginal:
    # f(x) ∝ (1 - x^2)^{(d-3)/2} for x in [-1, 1]
    # For d >= 3, approximate as N(0, 1/d) which is tight for d >= 20
    sigma = 1.0 / math.sqrt(d)

    # Lloyd-Max via iterative optimization on N(0, sigma^2)
    # Initialize with uniform quantile boundaries
    boundaries = scipy_norm.ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])

    for _ in range(200):  # Lloyd iterations
        # Compute centroids as conditional expectations
        centroids = np.zeros(n_centroids)
        for i in range(n_centroids):
            lo, hi = boundaries[i], boundaries[i + 1]
            # E[X | lo < X < hi] for X ~ N(0, sigma^2)
            p = scipy_norm.cdf(hi, scale=sigma) - scipy_norm.cdf(lo, scale=sigma)
            if p < 1e-15:
                centroids[i] = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else 0.0
            else:
                centroids[i] = sigma ** 2 * (
                    scipy_norm.pdf(lo, scale=sigma) - scipy_norm.pdf(hi, scale=sigma)
                ) / p

        # Update boundaries as midpoints of adjacent centroids
        new_boundaries = np.concatenate([
            [-np.inf],
            (centroids[:-1] + centroids[1:]) / 2,
            [np.inf],
        ])
        if np.allclose(boundaries[1:-1], new_boundaries[1:-1], atol=1e-12):
            break
        boundaries = new_boundaries

    return torch.tensor(centroids, dtype=torch.float32)


class TurboQuantIP:
    """
    TurboQuant one-sided inner product estimator.

    Quantizes item embeddings so that <user, dequant(quant(item))> is an
    unbiased estimator of <user, item>.

    Architecture:
    - Stage 1 (b-1 bits): MSE-optimal scalar quantization after random rotation
    - Stage 2 (1 bit): QJL sign sketch of the residual for unbiased IP correction

    Args:
        dim: Embedding dimension
        bits: Total bits per coordinate (>= 2; 1 for MSE + 1 for QJL)
        seed: Random seed for rotation matrix Π and sign matrix S
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        assert bits >= 2, "Need at least 2 bits (1 MSE + 1 QJL)"
        self.dim = dim
        self.total_bits = bits
        self.mse_bits = bits - 1  # reserve 1 bit for QJL

        gen = torch.Generator().manual_seed(seed)

        # Random orthogonal rotation (Theorem 1: makes coordinates ~iid)
        self.Pi = _random_orthogonal(dim, gen)  # (d, d)
        self.Pi_T = self.Pi.T

        # Lloyd-Max codebook for MSE quantization
        self.codebook = _lloyd_max_codebook(dim, self.mse_bits)  # (2^mse_bits,)
        self.n_centroids = len(self.codebook)

        # Boundaries for nearest-centroid assignment
        self._boundaries = torch.tensor(
            np.concatenate([
                [-np.inf],
                ((self.codebook[:-1] + self.codebook[1:]) / 2).numpy(),
                [np.inf],
            ]),
            dtype=torch.float32,
        )

        # Random Gaussian matrix for QJL (standard N(0,1) entries per paper Def. 1)
        self.S = torch.randn(dim, dim, generator=gen)  # (d, d), N(0, 1)

        # QJL dequantization coefficient (Lemma 4)
        # For S with N(0,1) entries: coeff = sqrt(pi/2) / d
        self.qjl_coeff = math.sqrt(math.pi / 2) / dim

    def quantize(self, x: torch.Tensor):
        """
        Quantize a batch of vectors.

        Args:
            x: (n, d) tensor of vectors to quantize

        Returns:
            mse_indices: (n, d) int tensor of codebook indices
            norms: (n, 1) tensor of original vector norms
            qjl_signs: (n, d) bool tensor of QJL sign bits
            residual_norms: (n, 1) tensor of residual norms
        """
        # Store and normalize
        norms = x.norm(dim=1, keepdim=True)  # (n, 1)
        x_unit = x / norms.clamp(min=1e-10)  # (n, d)

        # Rotate
        x_rot = x_unit @ self.Pi_T  # (n, d), multiply by Π^T

        # MSE quantize each coordinate
        # Find nearest centroid for each value
        mse_indices = torch.bucketize(x_rot, self._boundaries[1:-1])  # (n, d)
        x_mse = self.codebook[mse_indices]  # (n, d) reconstructed rotated coords

        # Residual in rotated space
        residual = x_rot - x_mse  # (n, d)
        residual_norms = residual.norm(dim=1, keepdim=True)  # (n, 1)
        residual_unit = residual / residual_norms.clamp(min=1e-10)

        # QJL: store signs of S @ residual_unit
        qjl_proj = residual_unit @ self.S.T  # (n, d)
        qjl_signs = (qjl_proj >= 0)  # (n, d) bool

        return mse_indices, norms, qjl_signs, residual_norms

    def dequantize(self, mse_indices, norms, qjl_signs, residual_norms):
        """
        Reconstruct vectors. The reconstructed vectors satisfy:
        E[<y, dequant(quant(x))>] = <y, x> for any y (Theorem 2).

        Returns:
            x_hat: (n, d) tensor of reconstructed vectors
        """
        # MSE reconstruction in rotated space
        x_mse = self.codebook[mse_indices]  # (n, d)

        # QJL reconstruction of residual
        signs = 2.0 * qjl_signs.float() - 1.0  # {0,1} -> {-1,+1}
        # Paper Def 1: Q_qjl^{-1}(z) = (sqrt(pi/2) / d) * S^T @ z
        # With N(0,1) entries in S, this is unbiased for unit-norm inputs
        r_hat = residual_norms * self.qjl_coeff * (signs @ self.S)  # (n, d)

        # Combine and un-rotate
        x_rot_hat = x_mse + r_hat  # (n, d) in rotated space
        x_hat = x_rot_hat @ self.Pi  # (n, d) back to original space

        # Rescale by original norms
        x_hat = x_hat * norms

        return x_hat

    def estimate_ip(self, queries, mse_indices, norms, qjl_signs, residual_norms):
        """
        Compute unbiased inner product estimates: E[result] = queries @ items.T

        This is equivalent to queries @ dequantize(...).T but makes the
        one-sided nature explicit.

        Args:
            queries: (m, d) tensor of query vectors (full precision)
            mse_indices, norms, qjl_signs, residual_norms: from quantize()

        Returns:
            ip_estimates: (m, n) tensor of estimated inner products
        """
        x_hat = self.dequantize(mse_indices, norms, qjl_signs, residual_norms)
        return queries @ x_hat.T
