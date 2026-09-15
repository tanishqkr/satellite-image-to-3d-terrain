"""
Single-Pass Evidential Deep Learning for Elevation Inversion (§1.4).

Formulation based on Amini et al., "Deep Evidential Regression" (NeurIPS 2020).
Replaces Monte Carlo Dropout (5-10 passes) with a single forward pass placing
a Normal-Inverse-Gamma (NIG) prior over the predicted depth distribution:
    y ~ N(gamma, beta / (alpha - 1))
    (mu, sigma^2) ~ NIG(gamma, nu, alpha, beta)

Parameters:
    gamma: Predicted depth mean
    nu:    Virtual evidence count (pseudo-observations for mean)
    alpha: Shape parameter of Inverse-Gamma (evidence for variance, alpha > 1)
    beta:  Scale parameter of Inverse-Gamma (beta > 0)

Decomposition:
    Aleatoric Uncertainty (data noise):         Var[y] = beta / (alpha - 1)
    Epistemic Uncertainty (model ignorance):   Var[mu] = beta / (nu * (alpha - 1))
"""
import io
import base64
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.logging_config import log


@dataclass
class EvidentialPrediction:
    depth: np.ndarray                    # (H, W) float32 in [0, 1]
    epistemic_uncertainty: np.ndarray    # (H, W) float32 standard deviation
    aleatoric_uncertainty: np.ndarray    # (H, W) float32 standard deviation
    total_uncertainty: np.ndarray        # (H, W) float32 combined standard deviation
    confidence: np.ndarray               # (H, W) float32 in [0, 1]
    nu: np.ndarray                       # (H, W) virtual evidence
    alpha: np.ndarray                    # (H, W) Inverse-Gamma shape
    beta: np.ndarray                     # (H, W) Inverse-Gamma scale


class EvidentialDepthHead(nn.Module):
    """
    4-parameter Normal-Inverse-Gamma (NIG) Evidential Decoder Head.
    Drop-in replacement / augmentation for DepthAnythingDepthEstimationHead.
    """
    def __init__(self, in_channels: int = 64, mid_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.activation1 = nn.ReLU()
        # 4 outputs: gamma (mean), nu (evidence count), alpha (shape > 1), beta (scale > 0)
        self.conv_evidential = nn.Conv2d(mid_channels, 4, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass producing (gamma, nu, alpha, beta).
        """
        x = self.conv1(hidden_states)
        x = self.conv2(x)
        x = self.activation1(x)
        raw = self.conv_evidential(x)

        # 1. gamma: predicted depth mean (ReLU or Softplus ensures non-negative depth)
        gamma = F.relu(raw[:, 0:1])

        # 2. nu > 0: evidence count for the mean
        nu = F.softplus(raw[:, 1:2]) + 1e-4

        # 3. alpha > 1: shape parameter for Inverse-Gamma distribution
        alpha = F.softplus(raw[:, 2:3]) + 1.0 + 1e-4

        # 4. beta > 0: scale parameter for Inverse-Gamma distribution
        beta = F.softplus(raw[:, 3:4]) + 1e-4

        return gamma, nu, alpha, beta

    def compute_uncertainty(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (epistemic_std, aleatoric_std, confidence) from NIG parameters.
        """
        results = compute_evidential_uncertainties(gamma, nu, alpha, beta)
        return results["epistemic_std"], results["aleatoric_std"], results["confidence"]

    @classmethod
    def from_standard_head(cls, standard_head: nn.Module) -> "EvidentialDepthHead":
        """
        Transfer pretrained weights from standard Depth Anything V2 head.
        Initializes gamma from original conv3 and sets neutral NIG priors on evidence channels.
        """
        head = cls(in_channels=64, mid_channels=32)
        with torch.no_grad():
            # Copy conv1 and conv2
            head.conv1.weight.copy_(standard_head.conv1.weight)
            head.conv1.bias.copy_(standard_head.conv1.bias)
            head.conv2.weight.copy_(standard_head.conv2.weight)
            head.conv2.bias.copy_(standard_head.conv2.bias)

            # Initialize evidential conv
            head.conv_evidential.weight.zero_()
            head.conv_evidential.bias.zero_()

            # Channel 0 (gamma): exact replica of original conv3 depth prediction
            head.conv_evidential.weight[0].copy_(standard_head.conv3.weight[0])
            head.conv_evidential.bias[0].copy_(standard_head.conv3.bias[0])

            # Channel 1 (nu): bias=0.5413 -> softplus(0.5413) ~ 1.0 (neutral evidence)
            head.conv_evidential.bias[1] = 0.5413

            # Channel 2 (alpha): bias=0.5413 -> softplus(0.5413) + 1.0 ~ 2.0 (finite variance prior)
            head.conv_evidential.bias[2] = 0.5413

            # Channel 3 (beta): bias=-2.0 -> softplus(-2.0) ~ 0.126 (conservative low initial variance)
            head.conv_evidential.bias[3] = -2.0

        return head


def compute_evidential_uncertainties(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute aleatoric and epistemic uncertainties from NIG parameters.
    """
    # Aleatoric variance: beta / (alpha - 1)
    aleatoric_var = beta / (alpha - 1.0)
    # Epistemic variance: beta / (nu * (alpha - 1))
    epistemic_var = beta / (nu * (alpha - 1.0))

    aleatoric_std = torch.sqrt(aleatoric_var)
    epistemic_std = torch.sqrt(epistemic_var)
    total_std = torch.sqrt(aleatoric_var + epistemic_var)

    # Confidence score in [0, 1]: 1 / (1 + epistemic_std)
    confidence = 1.0 / (1.0 + epistemic_std)

    return {
        "depth": gamma,
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
        "total_std": total_std,
        "confidence": confidence,
        "nu": nu,
        "alpha": alpha,
        "beta": beta,
    }


def colorize_risk_heatmap(
    uncertainty: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """
    Colorize an uncertainty / risk map using the unified Green -> Yellow -> Red convention.
    Green (#10B981): Low uncertainty / high operator confidence
    Yellow (#F59E0B): Moderate uncertainty
    Red (#EF4444): High uncertainty / tactical hazard

    Returns:
        RGB image array (H, W, 3) uint8.
    """
    u = np.asarray(uncertainty, dtype=np.float32)
    u_valid = u[np.isfinite(u)]

    if len(u_valid) > 0:
        lo = float(np.percentile(u_valid, 2.0)) if min_val is None else min_val
        hi = float(np.percentile(u_valid, 98.0)) if max_val is None else max_val
    else:
        lo, hi = 0.0, 1.0

    span = max(hi - lo, 1e-6)
    norm = np.clip((u - lo) / span, 0.0, 1.0)

    # Interpolate along Green (16, 185, 129) -> Yellow (245, 158, 11) -> Red (239, 68, 68)
    # Piecewise linear: [0, 0.5] green to yellow, [0.5, 1.0] yellow to red
    c_green = np.array([16, 185, 129], dtype=np.float32)
    c_yellow = np.array([245, 158, 11], dtype=np.float32)
    c_red = np.array([239, 68, 68], dtype=np.float32)

    H, W = norm.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    mask_low = norm < 0.5
    t_low = (norm[mask_low] / 0.5)[:, None]
    rgb[mask_low] = np.clip(c_green * (1.0 - t_low) + c_yellow * t_low, 0, 255).astype(np.uint8)

    mask_high = ~mask_low
    t_high = ((norm[mask_high] - 0.5) / 0.5)[:, None]
    rgb[mask_high] = np.clip(c_yellow * (1.0 - t_high) + c_red * t_high, 0, 255).astype(np.uint8)

    return rgb


def risk_heatmap_to_base64(rgb_array: np.ndarray) -> str:
    """Convert RGB array to base64 PNG string."""
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
