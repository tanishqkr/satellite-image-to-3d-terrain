"""Compute RMSE, MAE, Pearson r between predicted and reference DSMs."""
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    rmse: float
    mae: float
    pearson_r: float
    delta_1: float  # % pixels within 1.25x of GT
    n_pixels: int


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> ValidationMetrics:
    """
    Compute accuracy metrics between predicted and reference DSMs.
    Both must be (H, W) float arrays with same shape.
    """
    assert pred.shape == ref.shape, f"Shape mismatch: {pred.shape} vs {ref.shape}"

    # Mask invalid pixels and nodata values (e.g., -9999)
    valid = (
        np.isfinite(pred)
        & np.isfinite(ref)
        & (ref > -9000.0)
        & (ref < 15000.0)
        & (pred > -9000.0)
        & (pred < 15000.0)
    )
    p = pred[valid]
    r = ref[valid]
    n = p.size

    if n == 0:
        return ValidationMetrics(rmse=0, mae=0, pearson_r=0, delta_1=0, n_pixels=0)

    # RMSE
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))

    # MAE
    mae = float(np.mean(np.abs(p - r)))

    # Pearson correlation
    if np.std(p) > 1e-8 and np.std(r) > 1e-8:
        pearson_r = float(np.corrcoef(p, r)[0, 1])
    else:
        pearson_r = 0.0

    # Delta accuracy (% within 1.25x)
    # If all values are strictly positive, compute standard depth ratio
    if np.all(p > 0.01) and np.all(r > 0.01):
        ratio = np.maximum(p / (r + 1e-8), r / (p + 1e-8))
        delta_1 = float(np.mean(ratio < 1.25) * 100)
    else:
        # For terrain with negative or sea-level elevations, shift baseline to ensure positive ratio
        shift = max(0.0, -float(min(p.min(), r.min()))) + 1.0
        p_shift = p + shift
        r_shift = r + shift
        ratio = np.maximum(p_shift / r_shift, r_shift / p_shift)
        delta_1 = float(np.mean(ratio < 1.25) * 100)

    return ValidationMetrics(
        rmse=rmse, mae=mae, pearson_r=pearson_r,
        delta_1=delta_1, n_pixels=n
    )
