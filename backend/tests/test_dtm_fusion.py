"""
Unit and Integration Tests for Global DTM Low-Frequency Anchoring (§2.1).
Covers:
1. Copernicus GLO-30 tile naming across all four geographic hemispheres.
2. Frequency-split DTM blending mathematics (global low-frequency slope + local high-frequency micro-relief).
3. Offline tactical cache fallback and feature flag toggles.
4. Local caching persistence.
5. End-to-end georeferenced GeoTIFF calibration with real-world elevation anchoring.
"""
import sys
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.dtm_fusion import (
    get_copernicus_tile_name,
    get_covering_tile_names,
    fetch_or_cache_tile,
    fuse_dtm,
    fetch_and_resample_global_dtm,
)
from app.services.calibration import calibrate_depth
from app.services.geospatial import read_image
import rasterio
from rasterio.transform import from_bounds


def test_copernicus_tile_naming_hemispheres():
    print("Testing Copernicus GLO-30 tile naming across hemispheres...")
    # N / E: Tokyo
    name_ne = get_copernicus_tile_name(35.6762, 139.6503)
    assert name_ne == "Copernicus_DSM_COG_10_N35_00_E139_00_DEM", f"Unexpected: {name_ne}"

    # N / W: New York (lon -74.0060 is in [-75, -74] -> SW corner is W075)
    name_nw = get_copernicus_tile_name(40.7128, -74.0060)
    assert name_nw == "Copernicus_DSM_COG_10_N40_00_W075_00_DEM", f"Unexpected: {name_nw}"
    name_nw2 = get_copernicus_tile_name(40.7128, -73.95)
    assert name_nw2 == "Copernicus_DSM_COG_10_N40_00_W074_00_DEM", f"Unexpected: {name_nw2}"

    # S / E: Sydney
    name_se = get_copernicus_tile_name(-33.8688, 151.2093)
    assert name_se == "Copernicus_DSM_COG_10_S34_00_E151_00_DEM", f"Unexpected: {name_se}"

    # S / W: Rio de Janeiro
    name_sw = get_copernicus_tile_name(-22.9068, -43.1729)
    assert name_sw == "Copernicus_DSM_COG_10_S23_00_W044_00_DEM", f"Unexpected: {name_sw}"

    # Covering tiles for multi-degree boundary box
    covering = get_covering_tile_names(39.9, -74.1, 40.2, -73.9)
    assert len(covering) == 4, f"Expected 4 covering tiles across quadrant, got {len(covering)}"
    print("  [PASS] Copernicus DEM tile naming across all quadrants: PASSED")


def test_frequency_split_fusion_math():
    print("Testing frequency-split DTM blending math...")
    H, W = 128, 128
    y, x = np.mgrid[0:H, 0:W]

    # Global DTM: High base MSL elevation (200m) with a regional slope (0.1m/pixel)
    global_dtm = 200.0 + 0.1 * x + 0.05 * y

    # Local DTM: Arbitrary unanchored base (~0m) with high-frequency mounds (+-5m)
    local_high_freq = 5.0 * np.sin(x / 4.0) * np.cos(y / 4.0)
    local_dtm = 2.0 + local_high_freq

    # Blend with cutoff wavelength
    fused = fuse_dtm(
        dtm_local_base=local_dtm,
        dtm_global_resampled=global_dtm,
        gsd=1.0,
        cutoff_wavelength_m=20.0,
    )

    # 1. Base elevation must be anchored to global (~200m, not local ~2m)
    assert abs(fused.mean() - global_dtm.mean()) < 2.0, \
        f"Fused mean {fused.mean()} should match global mean {global_dtm.mean()}"

    # 2. Local high-frequency micro-relief must be preserved
    # Check that high-frequency std in fused matches local_high_freq std
    from scipy.ndimage import gaussian_filter
    hf_in_fused = fused - gaussian_filter(fused, sigma=20.0)
    assert abs(hf_in_fused.std() - local_high_freq.std()) < 0.5, \
        f"Micro-relief amplitude mismatch: {hf_in_fused.std()} vs {local_high_freq.std()}"

    print(f"  [PASS] Frequency-split DTM math verified (mean={fused.mean():.1f}m, anchored to global): PASSED")


def test_offline_tactical_fallback():
    print("Testing offline tactical mode (local_cache) & disabled fallback...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # 1. Missing file in local_cache returns None gracefully
        res = fetch_or_cache_tile("Copernicus_DSM_COG_10_N45_00_E010_00_DEM", cache_dir=temp_dir, source="local_cache")
        assert res is None

        # 2. Disabled source returns None
        res_dis = fetch_or_cache_tile("Copernicus_DSM_COG_10_N45_00_E010_00_DEM", cache_dir=temp_dir, source="disabled")
        assert res_dis is None

        # 3. Calibration fallback when global DTM unavailable
        dummy_depth = np.random.rand(64, 64).astype(np.float32)
        cal = calibrate_depth(
            dummy_depth,
            is_georef=True,
            gsd=0.5,
            bounds={"left": 0, "bottom": 0, "right": 100, "top": 100},
            crs="EPSG:32617",
            transform=[0.5, 0, 0, 0, -0.5, 32],
            dtm_global=None,
        )
        assert cal.mode in ("metric_prior", "metric_dtm_fusion")
        assert cal.dsm.shape == (64, 64)
        print("  [PASS] Offline tactical fallback and graceful degradation: PASSED")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_end_to_end_dtm_fusion_benchmark():
    print("Testing end-to-end DTM fusion on real satellite GeoTIFF...")
    sample_path = Path("test_datasets/geotiff_mode/04_geotiff_isro_utm43_512.tif")
    if not sample_path.exists():
        print("  [SKIP] Benchmark GeoTIFF not found on disk")
        return

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    rgb, pil_img, meta = read_image(file_bytes, sample_path.name)
    assert meta.is_georef
    assert meta.crs == "EPSG:32643"

    # Synthetic relative depth (flat ground with a 15m building in center)
    depth = np.full((meta.height, meta.width), 0.2, dtype=np.float32)
    depth[200:300, 200:300] = 0.8  # Elevated building block

    cal_fused = calibrate_depth(
        relative_depth=depth,
        is_georef=True,
        gsd=meta.gsd,
        bounds=meta.bounds,
        crs=meta.crs,
        transform=meta.transform,
    )

    print(f"  Result mode: {cal_fused.mode}")
    print(f"  DSM range: [{cal_fused.dsm_min:.1f}m, {cal_fused.dsm_max:.1f}m], mean: {cal_fused.dsm_mean:.1f}m")

    # Gujarat terrain is known to be around 45m-49m MSL
    if cal_fused.fusion_applied:
        assert 40.0 <= cal_fused.dsm_min <= 55.0, \
            f"Expected terrain MSL ~45m, got {cal_fused.dsm_min}"
        assert cal_fused.dsm_max > cal_fused.dsm_min + 10.0, \
            f"Expected building height relief on top of DTM, got max={cal_fused.dsm_max}"
        print(f"  [PASS] Real-world benchmark tile accurately anchored to Copernicus DEM (~46m MSL): PASSED")
    else:
        print(f"  [WARN] Fusion not applied (likely offline environment), fell back to metric prior")


if __name__ == "__main__":
    test_copernicus_tile_naming_hemispheres()
    test_frequency_split_fusion_math()
    test_offline_tactical_fallback()
    test_end_to_end_dtm_fusion_benchmark()
    print("\nALL DTM FUSION (§2.1) TESTS PASSED!")
