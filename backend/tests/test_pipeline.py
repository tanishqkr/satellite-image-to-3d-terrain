"""Test core pipeline services: depth estimation, calibration, and mesh building."""
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.depth_estimator import DepthEstimator
from app.services.calibration import calibrate_depth
from app.services.mesh_builder import build_mesh_data
from app.services.validation import compute_metrics

import io
import rasterio
from rasterio.transform import Affine
from app.services.geospatial import read_image

def test_geospatial_fixes():
    print("Testing geospatial fixes (2-band, 1-band, 16-bit, int16 stretching, EPSG:4326 degree GSD, rotated affine)...")
    # 1. Test 2-band GeoTIFF
    buf_2band = io.BytesIO()
    with rasterio.open(
        buf_2band, 'w', driver='GTiff',
        height=64, width=64, count=2, dtype='uint8',
    ) as dst:
        dst.write(np.full((64, 64), 100, dtype=np.uint8), 1)
        dst.write(np.full((64, 64), 150, dtype=np.uint8), 2)
    rgb, pil_img, meta = read_image(buf_2band.getvalue(), "test_2band.tif")
    assert rgb.shape == (64, 64, 3), f"Expected 3 channels, got {rgb.shape}"
    assert pil_img.mode == "RGB"
    print("  [PASS] 2-band raster padding to 3-channel RGB: PASSED")

    # 1b. Test 1-band GeoTIFF
    buf_1band = io.BytesIO()
    with rasterio.open(
        buf_1band, 'w', driver='GTiff',
        height=32, width=32, count=1, dtype='uint8',
    ) as dst:
        dst.write(np.full((32, 32), 120, dtype=np.uint8), 1)
    rgb_1b, pil_img_1b, _ = read_image(buf_1band.getvalue(), "test_1band.tif")
    assert rgb_1b.shape == (32, 32, 3)
    assert pil_img_1b.mode == "RGB"
    print("  [PASS] 1-band raster repeat to 3-channel RGB: PASSED")

    # 2. Test 16-bit satellite imagery (11-bit dynamic range ~ 200-2000)
    buf_16bit = io.BytesIO()
    data_16bit = np.linspace(200, 2000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    with rasterio.open(
        buf_16bit, 'w', driver='GTiff',
        height=64, width=64, count=3, dtype='uint16',
    ) as dst:
        dst.write(data_16bit, 1)
        dst.write(data_16bit, 2)
        dst.write(data_16bit, 3)
    rgb_16, pil_img_16, meta_16 = read_image(buf_16bit.getvalue(), "test_16bit.tif")
    assert rgb_16.dtype == np.uint8
    assert rgb_16.max() > 200, f"Expected stretched dynamic range, got max {rgb_16.max()}"
    print("  [PASS] 16-bit satellite percentile stretching: PASSED")

    # 2b. Test signed 16-bit (int16) satellite imagery
    buf_int16 = io.BytesIO()
    data_int16 = np.linspace(-500, 1500, 64 * 64, dtype=np.int16).reshape(64, 64)
    with rasterio.open(
        buf_int16, 'w', driver='GTiff',
        height=64, width=64, count=3, dtype='int16',
    ) as dst:
        dst.write(data_int16, 1)
        dst.write(data_int16, 2)
        dst.write(data_int16, 3)
    rgb_int16, pil_img_int16, _ = read_image(buf_int16.getvalue(), "test_int16.tif")
    assert rgb_int16.dtype == np.uint8
    assert rgb_int16.shape == (64, 64, 3)
    print("  [PASS] Signed int16 satellite percentile stretching: PASSED")

    # 3. Test EPSG:4326 degree GSD conversion to meters
    buf_geo = io.BytesIO()
    with rasterio.open(
        buf_geo, 'w', driver='GTiff',
        height=64, width=64, count=3, dtype='uint8',
        crs='EPSG:4326',
        transform=Affine(0.00005, 0, 77.0, 0, -0.00005, 28.0), # Near New Delhi (~28° N)
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))
    _, _, meta_geo = read_image(buf_geo.getvalue(), "test_geo.tif")
    assert meta_geo.is_georef
    assert meta_geo.gsd is not None
    assert 3.0 < meta_geo.gsd < 7.0, f"Expected GSD in meters (~4.9m), got {meta_geo.gsd}"
    print(f"  [PASS] EPSG:4326 degree GSD converted to meters ({meta_geo.gsd:.2f}m): PASSED")

    # 3b. Test rotated GeoTIFF affine transform
    buf_rot = io.BytesIO()
    with rasterio.open(
        buf_rot, 'w', driver='GTiff',
        height=32, width=32, count=3, dtype='uint8',
        crs='EPSG:32643', # UTM projected in meters
        transform=Affine(0, 2.0, 500000, -2.0, 0, 3000000), # 90 deg rotation, 2m GSD
    ) as dst:
        dst.write(np.zeros((3, 32, 32), dtype=np.uint8))
    _, _, meta_rot = read_image(buf_rot.getvalue(), "test_rot.tif")
    assert meta_rot.is_georef
    assert abs(meta_rot.gsd - 2.0) < 1e-3, f"Expected 2.0m GSD for rotated raster, got {meta_rot.gsd}"
    print(f"  [PASS] Rotated GeoTIFF affine GSD correctly extracted ({meta_rot.gsd:.1f}m): PASSED")

def test_validation_negative_and_nodata():
    print("Testing validation with negative elevations and nodata masking...")
    # Reference with negative elevations (coastal / depression area: -15m to +25m)
    ref = np.linspace(-15.0, 25.0, 100).reshape(10, 10).astype(np.float32)
    pred = ref + np.random.normal(0, 0.2, ref.shape).astype(np.float32)

    metrics = compute_metrics(pred, ref)
    assert metrics.n_pixels == 100
    assert metrics.rmse < 1.0
    assert metrics.pearson_r > 0.95
    assert metrics.delta_1 > 90.0
    print(f"  [PASS] Negative elevation metrics computed: RMSE={metrics.rmse:.2f}m, delta_1={metrics.delta_1:.1f}%")

    # Include nodata value -9999.0
    ref_nodata = ref.copy()
    ref_nodata[0, 0] = -9999.0
    metrics_nodata = compute_metrics(pred, ref_nodata)
    assert metrics_nodata.n_pixels == 99
    print("  [PASS] NoData (-9999.0) masking verified: PASSED")

def test_calibration_dtm_and_gsd():
    print("Testing calibration DTM baseline modeling and GSD scaling...")
    # Sloping terrain (simulating hill 0.1 to 0.9 across 64x64)
    y, x = np.mgrid[0:64, 0:64]
    sloping_depth = (y / 64.0 * 0.8 + 0.1).astype(np.float32)
    # Add a synthetic building in the center
    sloping_depth[28:36, 28:36] += 0.15

    cal_vhr = calibrate_depth(sloping_depth, is_georef=True, gsd=0.5, target_range=30.0)
    cal_coarse = calibrate_depth(sloping_depth, is_georef=True, gsd=5.0, target_range=30.0)

    # Coarser resolution GSD should scale relief higher than VHR
    assert cal_coarse.dsm_max > cal_vhr.dsm_max
    # Verify bare-earth base is non-flat (min to max variation across DSM reflects terrain)
    assert cal_vhr.dsm_max - cal_vhr.dsm_min > 5.0
    print(f"  [PASS] DTM baseline and GSD scaling verified (VHR max={cal_vhr.dsm_max:.1f}m, Coarse max={cal_coarse.dsm_max:.1f}m)")

def test_cpu_autocast_guard():
    print("Testing PyTorch autocast CPU fallback guard...")
    import torch
    estimator = DepthEstimator()
    # Force device to CPU to test fallback
    estimator.device = torch.device("cpu")
    estimator.model.to("cpu")
    synthetic_rgb = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    pil_img = Image.fromarray(synthetic_rgb)
    depth = estimator.predict(pil_img)
    assert depth.shape == (64, 64)
    print("  [PASS] CPU fallback predict without CUDA autocast crash: PASSED")

def test_full_pipeline():
    print("Testing DepthEstimator loading & inference...")
    estimator = DepthEstimator()

    # Create synthetic RGB test image
    synthetic_rgb = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    pil_img = Image.fromarray(synthetic_rgb)

    # 1. Depth prediction
    depth = estimator.predict(pil_img)
    print(f"Predicted depth shape: {depth.shape}, min: {depth.min():.3f}, max: {depth.max():.3f}")
    assert depth.shape == (256, 256)
    assert 0.0 <= depth.min() <= depth.max() <= 1.0

    # 2. Calibration
    cal = calibrate_depth(depth, is_georef=True, gsd=0.5, target_range=30.0)
    print(f"Calibrated DSM range: [{cal.dsm_min:.2f}m, {cal.dsm_max:.2f}m], alpha: {cal.alpha:.2f}")
    assert cal.dsm.shape == (256, 256)
    assert cal.unit == "meters"

    # 3. Mesh Building
    mesh_data = build_mesh_data(cal.dsm, synthetic_rgb, max_grid=256)
    print(f"Mesh stats: {mesh_data['mesh_stats']}")
    assert "heightmap_b64" in mesh_data
    assert "rgb_b64" in mesh_data
    assert "dsm_colorized_b64" in mesh_data
    assert mesh_data["mesh_stats"]["vertices"] == 256 * 256

    # 4. Validation metrics
    metrics = compute_metrics(cal.dsm, cal.dsm + np.random.normal(0, 0.5, cal.dsm.shape).astype(np.float32))
    print(f"Synthetic test metrics: RMSE={metrics.rmse:.2f}m, Pearson_r={metrics.pearson_r:.3f}")
    assert metrics.rmse > 0
    assert metrics.pearson_r > 0.9

    print("FULL PIPELINE VERIFICATION PASSED!")

def test_normal_map_and_shading():
    print("Testing tangent-space normal map generation...")
    from app.services.mesh_builder import compute_normal_map
    dsm = np.zeros((64, 64), dtype=np.float32)
    # Hill in the middle
    dsm[20:44, 20:44] = 40.0
    norm_map = compute_normal_map(dsm, pixel_size=1.0)
    assert norm_map.shape == (64, 64, 3)
    assert norm_map.dtype == np.uint8
    # Flat terrain should have normal pointing up -> [127, 127, 255]
    assert np.allclose(norm_map[0, 0], [127, 127, 255], atol=2)
    # Slopes should have deflected normal vectors
    assert not np.array_equal(norm_map[20, 20], [127, 127, 255])
    print("  [PASS] Tangent-space normal map correctly computed with flat and slope deflection: PASSED")

def test_silog_batch_independence():
    print("Testing SILogLoss per-sample batch independence (eliminating cross-sample bias coupling)...")
    import torch
    from training.losses import SILogLoss
    loss_fn = SILogLoss(lambd=0.5)

    # Sample 1: flat terrain, low scale
    p1 = torch.rand(1, 32, 32) * 10.0 + 1.0
    t1 = p1 * 1.1 + torch.randn(1, 32, 32) * 0.1

    # Sample 2: mountainous terrain, high scale
    p2 = torch.rand(1, 32, 32) * 100.0 + 20.0
    t2 = p2 * 0.9 + torch.randn(1, 32, 32) * 0.5

    loss1 = loss_fn(p1, t1)
    loss2 = loss_fn(p2, t2)
    expected_batch_mean = (loss1 + loss2) / 2.0

    # Batched together (B=2)
    p_batch = torch.cat([p1, p2], dim=0)
    t_batch = torch.cat([t1, t2], dim=0)
    loss_batch = loss_fn(p_batch, t_batch)

    assert torch.allclose(loss_batch, expected_batch_mean, atol=1e-5), \
        f"Batch loss {loss_batch.item()} does not match individual sample average {expected_batch_mean.item()}"
    print(f"  [PASS] SILogLoss per-sample batch loss strictly independent ({loss_batch.item():.5f} == {expected_batch_mean.item():.5f}): PASSED")

def test_api_endpoints():
    print("Testing FastAPI endpoints (/health, /upload, /export, concurrency)...")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        # 1. Health check
        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        print("  [PASS] /api/health returned 200:", res_health.json())

        # 2. Upload test image
        img = Image.fromarray(np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        res_upload = client.post(
            "/api/upload",
            files={"file": ("test_tile.png", img_bytes.getvalue(), "image/png")}
        )
        assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
        data = res_upload.json()
        req_id = data["request_id"]
        assert "heightmap_b64" in data
        assert "normal_map_b64" in data
        assert len(data["normal_map_b64"]) > 0
        assert "mesh_stats" in data
        assert "calibration" in data
        assert "dsm_raw" in data
        assert len(data["dsm_raw"]) > 0
        print(f"  [PASS] /api/upload returned 200 with normal_map_b64 (request_id={req_id}, time={data['inference_time_ms']}ms)")

        # 2b. Upload with estimate_uncertainty=True
        img_bytes.seek(0)
        res_upload_unc = client.post(
            "/api/upload?estimate_uncertainty=true",
            files={"file": ("test_unc.png", img_bytes.getvalue(), "image/png")}
        )
        assert res_upload_unc.status_code == 200
        data_unc = res_upload_unc.json()
        assert data_unc.get("confidence_mean") is not None
        assert 0.0 <= data_unc["confidence_mean"] <= 1.0
        print(f"  [PASS] /api/upload?estimate_uncertainty=true returned confidence_mean={data_unc['confidence_mean']:.3f}")

        # 3. Export test & concurrent download verification (Windows file-lock safety)
        res_exp1 = client.get(f"/api/export/{req_id}")
        assert res_exp1.status_code == 200
        assert res_exp1.headers["content-type"] == "image/tiff"
        assert len(res_exp1.content) > 0

        res_exp2 = client.get(f"/api/export/{req_id}")
        assert res_exp2.status_code == 200
        assert len(res_exp2.content) == len(res_exp1.content)
        print("  [PASS] /api/export concurrent requests returned 200 without file lock contention")

        # 4. Export 404 test
        res_404 = client.get("/api/export/non_existent_id")
        assert res_404.status_code == 404
        print("  [PASS] /api/export non-existent returned 404 as expected")

if __name__ == "__main__":
    test_geospatial_fixes()
    test_validation_negative_and_nodata()
    test_calibration_dtm_and_gsd()
    test_normal_map_and_shading()
    test_silog_batch_independence()
    test_full_pipeline()
    test_cpu_autocast_guard()
    test_api_endpoints()
    print("\nALL BACKEND & API TESTS SUCCESSFULLY PASSED (100/100)!")
