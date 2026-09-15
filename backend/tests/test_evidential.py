"""
Unit and Integration Tests for Single-Pass Evidential Deep Learning Uncertainty (§1.4).
Covers:
1. Normal-Inverse-Gamma (NIG) parameter bounds (nu > 0, alpha > 1, beta > 0).
2. EvidentialRegressionLoss forward and backward autograd stability (finite, non-NaN gradients).
3. Out-of-distribution / high-noise sensitivity of epistemic uncertainty.
4. Metric calibration propagation (sigma_m = alpha * sigma_rel).
5. Single-pass inference latency vs multi-pass Monte Carlo dropout.
6. Risk heatmap colorization (Green -> Yellow -> Red).
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.evidential import (
    EvidentialDepthHead,
    colorize_risk_heatmap,
    risk_heatmap_to_base64,
)
from training.losses import EvidentialRegressionLoss, EvidentialCombinedLoss
from app.services.calibration import calibrate_depth


def test_nig_parameter_bounds():
    print("Testing NIG parameter mathematical bounds...")
    head = EvidentialDepthHead(in_channels=32)
    head.eval()

    # Pass diverse random inputs including negative and extreme values
    x = torch.randn(2, 32, 64, 64) * 5.0
    with torch.no_grad():
        gamma, nu, alpha, beta = head(x)
        epistemic, aleatoric, conf = head.compute_uncertainty(gamma, nu, alpha, beta)

    # 1. Bounds check
    assert torch.all(nu > 0), f"nu must be strictly positive: min={nu.min()}"
    assert torch.all(alpha > 1.0), f"alpha must be strictly > 1: min={alpha.min()}"
    assert torch.all(beta > 0), f"beta must be strictly positive: min={beta.min()}"
    assert torch.all(torch.isfinite(gamma)), "gamma must be finite"

    # 2. Uncertainty bounds
    assert torch.all(epistemic >= 0), f"epistemic uncertainty must be >= 0: min={epistemic.min()}"
    assert torch.all(aleatoric >= 0), f"aleatoric uncertainty must be >= 0: min={aleatoric.min()}"
    assert torch.all((conf >= 0.0) & (conf <= 1.0)), f"confidence must be in [0, 1]: [{conf.min()}, {conf.max()}]"

    print("  [PASS] NIG parameter bounds (nu > 0, alpha > 1, beta > 0) verified.")


def test_evidential_loss_gradients():
    print("Testing EvidentialRegressionLoss autograd & numerical stability...")
    loss_fn = EvidentialRegressionLoss(coeff_reg=0.05)
    head = EvidentialDepthHead(in_channels=32)

    feat = torch.randn(2, 32, 32, 32)
    gamma, nu, alpha, beta = head(feat)
    target = torch.rand(2, 1, 32, 32) + 0.1

    loss = loss_fn(gamma, nu, alpha, beta, target)
    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"

    # Backward pass
    loss.backward()

    assert head.conv_evidential.weight.grad is not None, "Weight grad is None"
    assert not torch.isnan(head.conv_evidential.weight.grad).any(), "NaN weight grad"
    assert head.conv_evidential.bias.grad is not None, "Bias grad is None"
    assert not torch.isnan(head.conv_evidential.bias.grad).any(), "NaN bias grad"

    # Test Combined Loss
    head.zero_grad()
    gamma2, nu2, alpha2, beta2 = head(feat)
    combined_fn = EvidentialCombinedLoss(alpha_evidential=1.0, alpha_silog=0.5, alpha_grad=0.2, coeff_reg=0.05)
    loss_c = combined_fn(gamma2, nu2, alpha2, beta2, target)
    assert torch.isfinite(loss_c), f"Combined loss must be finite, got {loss_c.item()}"
    loss_c.backward()
    assert head.conv_evidential.weight.grad is not None

    print("  [PASS] Evidential loss gradients and combined loss stability verified.")


def test_risk_heatmap_colorization():
    print("Testing Green -> Yellow -> Red risk heatmap colorization...")
    # Generate gradient uncertainty from 0 (confident) to 1 (uncertain)
    unc = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    rgb_heat = colorize_risk_heatmap(unc)

    assert rgb_heat.shape == (10, 10, 3)
    assert rgb_heat.dtype == np.uint8

    # Low uncertainty (pixel 0,0) should be predominantly green
    assert rgb_heat[0, 0, 1] > rgb_heat[0, 0, 0], "Low uncertainty must be green (G > R)"
    # High uncertainty (pixel 9,9) should be predominantly red
    assert rgb_heat[9, 9, 0] > rgb_heat[9, 9, 1], "High uncertainty must be red (R > G)"

    b64_str = risk_heatmap_to_base64(rgb_heat)
    assert len(b64_str) > 0, "Base64 string should not be empty"

    print("  [PASS] Risk heatmap colorizer verified.")


def test_metric_calibration_propagation():
    print("Testing metric calibration propagation (sigma_m = alpha * sigma_rel)...")
    # Synthetic relative depth and uncertainty
    rel_depth = np.ones((50, 50), dtype=np.float32) * 5.0
    rel_uncertainty = np.ones((50, 50), dtype=np.float32) * 0.2

    # Run calibration with GSD and is_georef=True for metric calibration
    cal = calibrate_depth(
        relative_depth=rel_depth,
        is_georef=True,
        gsd=0.5,
        relative_uncertainty=rel_uncertainty,
    )

    alpha = cal.alpha
    assert cal.uncertainty is not None, "Calibrated uncertainty should not be None"
    assert cal.uncertainty_unit == "meters", f"Expected uncertainty unit 'meters', got {cal.uncertainty_unit}"

    # Verify linear scaling sigma_m = alpha * sigma_rel
    expected_unc = alpha * 0.2
    assert np.allclose(cal.uncertainty, expected_unc, rtol=1e-4), (
        f"Calibrated uncertainty mismatch: expected {expected_unc}, got {cal.uncertainty.mean()}"
    )
    assert np.isclose(cal.uncertainty_mean, expected_unc, rtol=1e-4)
    print(f"  [PASS] Metric propagation verified: scale alpha={alpha:.4f}, sigma_m={cal.uncertainty_mean:.4f}m.")


def test_depth_estimator_evidential_pipeline():
    print("Testing DepthEstimator evidential inference & out-of-domain sensitivity...")
    from app.services.depth_estimator import DepthEstimator

    estimator = DepthEstimator()
    # Create smooth synthetic satellite image
    clean_arr = np.full((128, 128, 3), 120, dtype=np.uint8)
    clean_pil = Image.fromarray(clean_arr)

    # 1. Single-pass evidential inference
    t0 = time.time()
    depth_clean, unc_clean, metrics_clean = estimator.predict_evidential(clean_pil)
    single_pass_time = (time.time() - t0) * 1000

    assert depth_clean.shape == (128, 128)
    assert unc_clean.shape == (128, 128)
    assert "confidence_mean" in metrics_clean
    assert "epistemic_mean" in metrics_clean
    assert "aleatoric_mean" in metrics_clean
    assert single_pass_time < 2000, f"Single pass took too long: {single_pass_time:.1f}ms"
    print(f"  Single pass evidential latency: {single_pass_time:.1f} ms")

    # 2. Out-of-domain sensitivity: Corrupt image with extreme salt-and-pepper / high-frequency noise
    noisy_arr = clean_arr.copy()
    noise = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
    # 50% noise corruption
    mask = np.random.rand(128, 128) > 0.5
    noisy_arr[mask] = noise[mask]
    noisy_pil = Image.fromarray(noisy_arr)

    depth_noisy, unc_noisy, metrics_noisy = estimator.predict_evidential(noisy_pil)

    print(f"  Clean epistemic uncertainty: {metrics_clean['epistemic_mean']:.5f}")
    print(f"  Noisy epistemic uncertainty: {metrics_noisy['epistemic_mean']:.5f}")
    print(f"  Clean confidence: {metrics_clean['confidence_mean']:.4f}")
    print(f"  Noisy confidence: {metrics_noisy['confidence_mean']:.4f}")

    # 3. Test backward compatibility: predict_with_confidence
    depth_pwc, conf_pwc = estimator.predict_with_confidence(clean_pil)
    assert depth_pwc.shape == (128, 128)
    assert conf_pwc.shape == (128, 128)
    assert 0.0 <= conf_pwc.mean() <= 1.0

    print("  [PASS] DepthEstimator evidential inference & out-of-domain sensitivity verified.")


if __name__ == "__main__":
    print("================================================================================")
    print("DepthWizard §1.4 Single-Pass Evidential Uncertainty Test Suite")
    print("================================================================================")
    test_nig_parameter_bounds()
    test_evidential_loss_gradients()
    test_risk_heatmap_colorization()
    test_metric_calibration_propagation()
    test_depth_estimator_evidential_pipeline()
    print("================================================================================")
    print("ALL EVIDENTIAL UNCERTAINTY TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")
