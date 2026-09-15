"""
Unit and Integration Tests for Shadow-to-Scale Photometric Inversion (§1.1).
Covers:
1. Solar ephemeris calculation via pvlib.
2. Colorimetric/luminance cast shadow segmentation.
3. Closed-form shadow length projection and scale recovery across multiple structures.
4. Edge cases & graceful fallback (solar noon, night, overcast, featureless).
5. End-to-end integration into calibration and API /upload pipeline.
"""
import sys
import io
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.shadow_scale import (
    compute_solar_position,
    segment_shadows,
    estimate_scale_from_shadows,
)
from app.services.calibration import calibrate_depth


def test_solar_ephemeris():
    print("Testing solar ephemeris computation via pvlib...")
    # Summer solstice noon at Tropic of Cancer (lat ~23.44 N) -> solar elevation near 90 deg
    elev, azim = compute_solar_position("2026-06-21 12:00:00", latitude=23.4368, longitude=0.0)
    assert 85.0 <= elev <= 90.0, f"Expected near-zenith elevation, got {elev}"
    assert 0.0 <= azim <= 360.0

    # Winter night in London -> negative elevation
    elev_night, _ = compute_solar_position("2026-12-21 00:00:00", latitude=51.5074, longitude=-0.1278)
    assert elev_night < 0.0, f"Night elevation must be negative, got {elev_night}"
    print("  [PASS] Solar ephemeris deterministic calculation: PASSED")


def test_shadow_segmentation():
    print("Testing cast shadow segmentation...")
    H, W = 128, 128
    # Background: sunlit grassy terrain (R=120, G=160, B=80)
    rgb = np.full((H, W, 3), [120, 160, 80], dtype=np.uint8)
    # Cast shadow patch: dark and blue-shifted (R=30, G=40, B=55)
    rgb[40:70, 40:65] = [30, 40, 55]

    mask = segment_shadows(rgb, min_blob_size=16)
    assert mask.shape == (H, W)
    assert mask.dtype == bool

    # Shadow area should be detected
    shadow_px = int(mask.sum())
    expected_px = 30 * 25
    assert abs(shadow_px - expected_px) < 100, f"Expected ~{expected_px} px, got {shadow_px}"
    print(f"  [PASS] Shadow segmentation correctly isolated shadow patch ({shadow_px} px): PASSED")


def test_photometric_scale_recovery():
    print("Testing photometric scale recovery from cast shadows...")
    H, W = 160, 160
    gsd = 0.5  # 0.5 m/px
    solar_elev = 45.0  # tan(45) = 1.0
    solar_azim = 180.0  # Sun from South -> shadows cast to North (-Y)

    rgb = np.full((H, W, 3), [130, 150, 110], dtype=np.uint8)
    ndsm = np.zeros((H, W), dtype=np.float32)

    # Structure 1: Building 1 (15m high, relative relief = 0.30 -> target k = 50.0)
    # Roof at (70:90, 30:50)
    rgb[70:90, 30:50] = [200, 200, 200]
    ndsm[70:90, 30:50] = 0.30
    # Shadow directly North: length = 15m / 0.5 = 30px -> (40:70, 30:50)
    rgb[40:70, 30:50] = [30, 40, 55]

    # Structure 2: Building 2 (20m high, relative relief = 0.40 -> target k = 50.0)
    # Roof at (90:110, 90:110)
    rgb[90:110, 90:110] = [200, 200, 200]
    ndsm[90:110, 90:110] = 0.40
    # Shadow directly North: length = 20m / 0.5 = 40px -> (50:90, 90:110)
    rgb[50:90, 90:110] = [30, 40, 55]

    res = estimate_scale_from_shadows(
        rgb=rgb,
        ndsm_relative=ndsm,
        gsd=gsd,
        solar_elevation_deg=solar_elev,
        solar_azimuth_deg=solar_azim,
        min_structures=2,
    )

    assert res is not None, "Failed to estimate scale from shadows"
    recovered_k = res["scale_factor"]
    print(f"  Recovered scale factor: {recovered_k:.2f} (expected ~50.0)")
    assert np.isclose(recovered_k, 50.0, atol=3.0), f"Scale {recovered_k} differs from target 50.0"
    assert res["valid_structures"] >= 2
    print("  [PASS] Closed-form shadow-to-scale inversion accurate to target physics: PASSED")


def test_edge_cases_and_fallback():
    print("Testing edge cases & fallback...")
    H, W = 64, 64
    rgb = np.full((H, W, 3), 120, dtype=np.uint8)
    ndsm = np.zeros((H, W), dtype=np.float32)

    # 1. Solar noon (>80 deg) -> returns None
    res_noon = estimate_scale_from_shadows(rgb, ndsm, 0.5, 85.0, 180.0)
    assert res_noon is None, "Should reject solar noon"

    # 2. Night / low sun (<5 deg) -> returns None
    res_night = estimate_scale_from_shadows(rgb, ndsm, 0.5, 3.0, 180.0)
    assert res_night is None, "Should reject low sun"

    # 3. Calibration gracefully falls back when no shadows
    cal = calibrate_depth(
        relative_depth=np.full((H, W), 0.5, dtype=np.float32),
        is_georef=True,
        gsd=0.5,
        rgb=rgb,
        solar_elevation=45.0,
        solar_azimuth=180.0,
    )
    assert not cal.shadow_calibrated
    assert cal.mode in ("metric_prior", "metric_dtm_fusion")
    print("  [PASS] Graceful fallback on edge cases: PASSED")


def test_api_upload_with_shadow_parameters():
    print("Testing API /upload with solar parameters...")
    from fastapi.testclient import TestClient
    from app.main import app

    img = Image.new("RGB", (64, 64), color=(120, 140, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    with TestClient(app) as client:
        resp = client.post(
            "/api/upload?solar_elevation=45.0&solar_azimuth=180.0",
            files={"file": ("test_shadow.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "calibration" in data
        assert "shadow_calibrated" in data["calibration"]
        print("  [PASS] API /upload accepted solar parameters without error: PASSED")


if __name__ == "__main__":
    test_solar_ephemeris()
    test_shadow_segmentation()
    test_photometric_scale_recovery()
    test_edge_cases_and_fallback()
    test_api_upload_with_shadow_parameters()
    print("\nALL SHADOW-TO-SCALE (§1.1) TESTS PASSED!")
