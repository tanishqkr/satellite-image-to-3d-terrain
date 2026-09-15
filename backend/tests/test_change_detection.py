"""
Unit and Integration Tests for Automated 3D BDA & Volumetric Differencing (§4.2).
Covers:
1. Sub-pixel FFT Phase Correlation co-registration accuracy under known lateral translations.
2. Cut/Fill volumetric integration against analytical crater and rubble models.
3. Structural footprint isolation and percentage collapse severity classification.
4. Evidential uncertainty propagation (§1.4) into volumetric error bounds.
5. FastAPI /api/bda/diff endpoint end-to-end integration.
"""
import sys
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from scipy import ndimage

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.change_detection import (
    co_register_dsms,
    compute_volumetric_diff,
    assess_structural_bda,
    colorize_diff_map,
)
from app.main import app
from app.config import EXPORTS_DIR


def test_fft_phase_correlation_coregistration():
    print("Testing FFT phase correlation co-registration accuracy...")
    H, W = 80, 80
    gsd = 1.0  # 1m/px

    # Create distinct topographical surface with gradient and structures
    x = np.linspace(-3, 3, W)
    y = np.linspace(-3, 3, H)
    xx, yy = np.meshgrid(x, y)
    dsm_pre = (20.0 + 5.0 * np.sin(xx) + 4.0 * np.cos(yy)).astype(np.float32)
    # Add a prominent 15m structure
    dsm_pre[30:50, 30:50] += 15.0

    # Inject known shift: row_shift = +3.0 px, col_shift = -2.0 px
    true_shift_r = 3.0
    true_shift_c = -2.0
    dsm_shifted = ndimage.shift(dsm_pre, shift=(true_shift_r, true_shift_c), order=1, mode="nearest")

    # Run co-registration
    aligned, shift_x_m, shift_y_m, conf = co_register_dsms(dsm_pre, dsm_shifted, gsd=gsd)

    print(f"  Injected shift: r={true_shift_r:.1f}m, c={true_shift_c:.1f}m")
    print(f"  Recovered shift: X={shift_x_m:.2f}m, Y={shift_y_m:.2f}m, confidence={conf:.3f}")

    # The recovered shift should negate or identify the injected offset
    assert abs(shift_y_m - (-true_shift_r)) < 0.35, f"Y shift error: expected {-true_shift_r}, got {shift_y_m}"
    assert abs(shift_x_m - (-true_shift_c)) < 0.35, f"X shift error: expected {-true_shift_c}, got {shift_x_m}"
    assert conf > 0.5, f"Confidence too low: {conf}"

    # Verify residual alignment error
    residual = np.nanmean(np.abs(aligned[10:-10, 10:-10] - dsm_pre[10:-10, 10:-10]))
    print(f"  Mean residual alignment error: {residual:.3f}m")
    assert residual < 0.5, f"Residual alignment error too high: {residual}m"

    print("  [PASS] FFT phase correlation sub-pixel co-registration verified.")


def test_volumetric_cut_fill_math():
    print("Testing cut/fill volumetric differencing math against ground truth...")
    H, W = 100, 100
    gsd = 1.0  # 1m/px -> 1m² cell area
    dsm_pre = np.full((H, W), 20.0, dtype=np.float32)
    dsm_post = dsm_pre.copy()

    # Dig a known cylindrical crater: radius = 10m, depth = 5m
    # Theoretical volume = pi * r^2 * depth = pi * 100 * 5 = ~1570.8 m³
    mid_r, mid_c = 30, 30
    crater_radius = 10.0
    crater_depth = 5.0

    # Build a known debris rubble mound: radius = 8m, height = 3m
    # Theoretical volume = pi * r^2 * height = pi * 64 * 3 = ~603.2 m³
    rubble_r, rubble_c = 70, 70
    rubble_radius = 8.0
    rubble_height = 3.0

    for r in range(H):
        for c in range(W):
            dist_crater = np.hypot(r - mid_r, c - mid_c)
            if dist_crater <= crater_radius:
                dsm_post[r, c] -= crater_depth

            dist_rubble = np.hypot(r - rubble_r, c - rubble_c)
            if dist_rubble <= rubble_radius:
                dsm_post[r, c] += rubble_height

    res = compute_volumetric_diff(
        dsm_pre=dsm_pre,
        dsm_post=dsm_post,
        gsd=gsd,
        threshold_m=0.2,
        auto_coregister=False,
    )

    expected_cut = np.pi * (crater_radius ** 2) * crater_depth
    expected_fill = np.pi * (rubble_radius ** 2) * rubble_height

    print(f"  Measured cut: {res.cut_volume_m3} m³ (expected ~{expected_cut:.1f} m³)")
    print(f"  Measured fill: {res.fill_volume_m3} m³ (expected ~{expected_fill:.1f} m³)")
    print(f"  Net volume: {res.net_volume_m3} m³ (expected ~{expected_fill - expected_cut:.1f} m³)")

    assert abs(res.cut_volume_m3 - expected_cut) / expected_cut < 0.05, "Cut volume error > 5%"
    assert abs(res.fill_volume_m3 - expected_fill) / expected_fill < 0.05, "Fill volume error > 5%"
    assert abs(res.max_cut_depth_m - crater_depth) < 0.1, "Max cut depth mismatch"
    assert abs(res.max_fill_height_m - rubble_height) < 0.1, "Max fill height mismatch"

    print("  [PASS] Volumetric cut/fill numerical integration verified.")


def test_structural_footprint_collapse_classification():
    print("Testing structural BDA footprint isolation and collapse rating...")
    H, W = 80, 80
    gsd = 1.0
    # Flat ground at 5m
    dsm_pre = np.full((H, W), 5.0, dtype=np.float32)

    # Building A: 20x20 structure at (15:35, 15:35), height 12m (7m above ground)
    dsm_pre[15:35, 15:35] = 12.0

    # Building B: 15x15 structure at (50:65, 50:65), height 10m (5m above ground)
    dsm_pre[50:65, 50:65] = 10.0

    dsm_post = dsm_pre.copy()
    # Strike 1: Building A is 90% destroyed (collapsed down to ground level 5m)
    dsm_post[15:35, 15:35] = 5.5

    # Building B: untouched (post == pre)

    structures = assess_structural_bda(
        dsm_pre=dsm_pre,
        dsm_post=dsm_post,
        gsd=gsd,
        building_threshold_m=2.0,
        threshold_m=0.25,
    )

    assert len(structures) >= 2, f"Expected at least 2 detected structures, got {len(structures)}"

    # Find Building A (at row ~25, col ~25)
    struct_a = next(s for s in structures if abs(s.center_row - 25.0) < 5.0)
    # Find Building B (at row ~57, col ~57)
    struct_b = next(s for s in structures if abs(s.center_row - 57.0) < 5.0)

    print(f"  Building A collapse: {struct_a.collapse_percentage}%, rating: {struct_a.damage_rating}")
    print(f"  Building B collapse: {struct_b.collapse_percentage}%, rating: {struct_b.damage_rating}")

    assert struct_a.collapse_percentage > 75.0, f"Expected Building A to be >75% collapsed, got {struct_a.collapse_percentage}%"
    assert struct_a.damage_rating == "DESTROYED"
    assert struct_b.collapse_percentage < 5.0, f"Expected Building B to be undamaged, got {struct_b.collapse_percentage}%"
    assert struct_b.damage_rating == "LIGHT"

    print("  [PASS] Structural footprint isolation and collapse rating verified.")


def test_evidential_uncertainty_propagation_in_bda():
    print("Testing evidential uncertainty propagation (§1.4) into BDA bounds...")
    H, W = 40, 40
    gsd = 1.0
    dsm_pre = np.full((H, W), 10.0, dtype=np.float32)
    dsm_post = dsm_pre.copy()
    dsm_post[15:25, 15:25] -= 4.0  # 10x10 crater (100 m² footprint)

    unc_pre = np.full((H, W), 0.3, dtype=np.float32)
    unc_post = np.full((H, W), 0.4, dtype=np.float32)

    res = compute_volumetric_diff(
        dsm_pre=dsm_pre,
        dsm_post=dsm_post,
        gsd=gsd,
        unc_pre=unc_pre,
        unc_post=unc_post,
        auto_coregister=False,
    )

    assert res.uncertainty_volume_m3 is not None, "Uncertainty volume should not be None"
    # Expected sigma_diff = sqrt(0.3^2 + 0.4^2) = 0.5m
    # 100 cells * 1m2 * 0.5m = 50.0 m³
    expected_unc = 100 * 0.5
    print(f"  Measured volume uncertainty: ±{res.uncertainty_volume_m3} m³ (expected ~{expected_unc:.1f} m³)")
    assert abs(res.uncertainty_volume_m3 - expected_unc) < 2.0, "Uncertainty volume mismatch"

    print("  [PASS] Evidential uncertainty propagation verified.")


def test_fastapi_bda_diff_endpoint():
    print("Testing FastAPI /api/bda/diff endpoint integration...")
    client = TestClient(app)

    req_pre = "test_bda_pre"
    req_post = "test_bda_post"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    H, W = 50, 50
    dsm_pre = np.full((H, W), 15.0, dtype=np.float32)
    dsm_pre[20:30, 20:30] += 10.0  # Pre-strike structure

    dsm_post = dsm_pre.copy()
    dsm_post[20:30, 20:30] -= 5.0  # Crater at building site

    np.save(EXPORTS_DIR / f"{req_pre}_dsm.npy", dsm_pre)
    np.save(EXPORTS_DIR / f"{req_post}_dsm.npy", dsm_post)

    meta_dict = {
        "is_georef": True,
        "crs": "EPSG:32643",
        "transform": [1.0, 0, 0, 0, -1.0, 50],
        "width": 50,
        "height": 50,
    }
    with open(EXPORTS_DIR / f"{req_pre}_meta.json", "w") as f:
        json.dump(meta_dict, f)
    with open(EXPORTS_DIR / f"{req_post}_meta.json", "w") as f:
        json.dump(meta_dict, f)

    payload = {
        "request_id_pre": req_pre,
        "request_id_post": req_post,
        "noise_threshold_m": 0.20,
        "auto_coregister": True,
    }

    resp = client.post("/api/bda/diff", json=payload)
    assert resp.status_code == 200, f"BDA diff endpoint failed: {resp.text}"
    bda_data = resp.json()

    assert bda_data["request_id_pre"] == req_pre
    assert bda_data["request_id_post"] == req_post
    assert bda_data["cut_volume_m3"] > 0
    assert bda_data["diff_map_b64"] is not None
    assert len(bda_data["diff_map_b64"]) > 100
    assert bda_data["beta_mode"] is True
    assert len(bda_data["structures"]) > 0

    print(f"  /api/bda/diff returned: cut_vol={bda_data['cut_volume_m3']} m³, structures={len(bda_data['structures'])}, beta={bda_data['beta_mode']}")
    print("  [PASS] FastAPI /api/bda/diff endpoint verified.")


if __name__ == "__main__":
    print("================================================================================")
    print("DepthWizard §4.2 Automated 3D BDA & Volumetric Differencing Test Suite")
    print("================================================================================")
    test_fft_phase_correlation_coregistration()
    test_volumetric_cut_fill_math()
    test_structural_footprint_collapse_classification()
    test_evidential_uncertainty_propagation_in_bda()
    test_fastapi_bda_diff_endpoint()
    print("================================================================================")
    print("ALL 3D BDA & VOLUMETRIC DIFFERENCING (§4.2) TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")
