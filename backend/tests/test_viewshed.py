"""
Unit and Integration Tests for Tactical Viewshed & Line-of-Sight (LoS) Analysis (§4.1).
Covers:
1. Radial 360° viewshed computation over synthetic terrain (flat vs blocked).
2. Observer mast / structure height sensitivity (raising mast expands coverage over obstacles).
3. Point-to-point Line of Sight (LoS) ray marching and elevation profile extraction.
4. Obstacle intersection coordinate detection for tactical dead ground.
5. FastAPI API endpoints (/api/viewshed and /api/los) end-to-end integration.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.visibility import compute_viewshed, ray_march_los, sample_dsm_bilinear
from app.main import app
from app.config import EXPORTS_DIR


def test_viewshed_flat_vs_barrier():
    print("Testing 360° viewshed on flat terrain vs terrain with barrier...")
    H, W = 50, 50
    # Flat terrain at 10m MSL
    dsm_flat = np.full((H, W), 10.0, dtype=np.float32)

    # Observer at center (25, 25) with 2m mast (total 12m MSL)
    obs_pos = (25.0, 12.0, 25.0)
    vs_flat = compute_viewshed(dsm_flat, observer_pos=obs_pos, gsd=1.0)
    assert np.all(vs_flat == 1), "Flat terrain should be 100% visible to observer"
    print("  [PASS] Flat terrain 100% viewshed visibility verified.")

    # Add a tall vertical wall barrier at row 15 (columns 10 to 40, height 30m MSL)
    dsm_barrier = dsm_flat.copy()
    dsm_barrier[15, 10:40] = 30.0

    vs_barrier = compute_viewshed(dsm_barrier, observer_pos=obs_pos, gsd=1.0)
    # Observer at row 25 cannot see ground behind row 15 (i.e. rows 0 to 14 behind the barrier)
    blocked_cells = vs_barrier[0:15, 15:35]
    assert np.any(blocked_cells == 0), "Dead ground behind 30m barrier must be occluded (viewshed == 0)"
    num_visible = np.sum(vs_barrier)
    assert num_visible < H * W, f"Viewshed should have occluded cells, got {num_visible}/{H * W}"
    print(f"  [PASS] Barrier obstruction verified: {H * W - num_visible} dead ground cells masked.")


def test_mast_height_coverage_scaling():
    print("Testing observer mast height coverage scaling...")
    H, W = 60, 60
    dsm = np.full((H, W), 10.0, dtype=np.float32)
    # Ring of 20m high hills surrounding the center at radius 15
    for r in range(H):
        for c in range(W):
            dist = np.hypot(r - 30, c - 30)
            if 12.0 <= dist <= 16.0:
                dsm[r, c] = 20.0

    # Prone sniper (0.5m AGL -> 10.5m MSL)
    vs_prone = compute_viewshed(dsm, observer_pos=(30.0, 10.5, 30.0), gsd=1.0)
    prone_visible = int(np.sum(vs_prone))

    # Tactical Radar Tower (25m AGL -> 35.0m MSL, towering above the 20m hills)
    vs_tower = compute_viewshed(dsm, observer_pos=(30.0, 35.0, 30.0), gsd=1.0)
    tower_visible = int(np.sum(vs_tower))

    assert tower_visible > prone_visible, (
        f"Radar tower ({tower_visible}) must see more terrain than prone sniper ({prone_visible})"
    )
    coverage_boost = ((tower_visible - prone_visible) / (H * W)) * 100
    print(f"  Prone sniper visible: {prone_visible} cells ({(prone_visible / (H * W)) * 100:.1f}%)")
    print(f"  Radar tower visible: {tower_visible} cells ({(tower_visible / (H * W)) * 100:.1f}%)")
    print(f"  [PASS] Mast height expansion verified (+{coverage_boost:.1f}% coverage boost).")


def test_point_to_point_los_clear_and_blocked():
    print("Testing point-to-point LoS ray marching & obstacle detection...")
    H, W = 100, 100
    dsm = np.full((H, W), 5.0, dtype=np.float32)
    # Add an obstacle building at col 50 (height 25m)
    dsm[:, 48:52] = 25.0

    # Test 1: Both points on same side of obstacle (Col 10 to Col 40)
    p_a = (10.0, 7.0, 50.0)  # 2m AGL
    p_b = (40.0, 7.0, 50.0)  # 2m AGL
    res_clear = ray_march_los(p_a, p_b, dsm, gsd=1.0)
    assert res_clear["visible"] is True, "Line of sight on open terrain must be clear"
    assert res_clear["blocking_point"] is None
    assert len(res_clear["elevation_profile"]) > 5

    # Test 2: Sightline passing through the building (Col 10 to Col 90)
    p_c = (90.0, 7.0, 50.0)
    res_blocked = ray_march_los(p_a, p_c, dsm, gsd=1.0)
    assert res_blocked["visible"] is False, "Sightline through 25m obstacle must be blocked"
    assert res_blocked["blocking_point"] is not None
    bx, by, bz = res_blocked["blocking_point"]
    assert 47.0 <= bx <= 53.0, f"Blocking point X coordinate expected near obstacle, got {bx}"
    assert by >= 7.0, f"Blocking obstacle height expected >= ray height (7m), got {by}"

    print(f"  [PASS] Point-to-point LoS verified: obstruction intercepted at X={bx:.1f}m, Y={by:.1f}m.")


def test_fastapi_viewshed_and_los_endpoints():
    print("Testing FastAPI /api/viewshed and /api/los endpoints...")
    client = TestClient(app)

    # 1. Setup cached DSM fixture for request_id 'test_geo_41'
    req_id = "test_geo_41"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    H, W = 64, 64
    dsm_data = np.full((H, W), 20.0, dtype=np.float32)
    # Ridge at center
    dsm_data[30:34, :] = 45.0
    np.save(EXPORTS_DIR / f"{req_id}_dsm.npy", dsm_data)

    meta_dict = {
        "is_georef": True,
        "crs": "EPSG:32643",
        "transform": [1.0, 0, 0, 0, -1.0, 64],
        "width": 64,
        "height": 64,
    }
    with open(EXPORTS_DIR / f"{req_id}_meta.json", "w") as f:
        json.dump(meta_dict, f)

    # 2. Test /api/viewshed endpoint
    vs_payload = {
        "request_id": req_id,
        "observer_col": 32.0,
        "observer_row": 10.0,
        "observer_agl": 2.5,
        "max_range_m": 100.0,
        "target_height_m": 0.0,
    }
    resp = client.post("/api/viewshed", json=vs_payload)
    assert resp.status_code == 200, f"Viewshed endpoint failed: {resp.text}"
    vs_data = resp.json()
    assert vs_data["request_id"] == req_id
    assert vs_data["visible_cells"] > 0
    assert 0.0 < vs_data["visible_fraction"] < 1.0
    assert vs_data["viewshed_b64"] is not None
    assert len(vs_data["viewshed_b64"]) > 100
    print(f"  /api/viewshed returned: visible_ratio={vs_data['visible_fraction']:.2%}, eye_msl={vs_data['observer_elevation_msl']}m")

    # 3. Test /api/los endpoint
    los_payload = {
        "request_id": req_id,
        "p0": [32.0, 22.0, 10.0],
        "p1": [32.0, 22.0, 50.0],
        "sample_step_m": 1.0,
    }
    resp_los = client.post("/api/los", json=los_payload)
    assert resp_los.status_code == 200, f"LoS endpoint failed: {resp_los.text}"
    los_data = resp_los.json()
    assert los_data["visible"] is False, "Sightline across 45m ridge should be blocked"
    assert los_data["blocking_point"] is not None
    assert len(los_data["elevation_profile"]) > 10
    print(f"  /api/los returned: visible={los_data['visible']}, dist={los_data['distance_m']}m, block={los_data['blocking_point']}")

    print("  [PASS] FastAPI /api/viewshed and /api/los endpoints verified.")


if __name__ == "__main__":
    print("================================================================================")
    print("DepthWizard §4.1 Viewshed & Line-of-Sight Test Suite")
    print("================================================================================")
    test_viewshed_flat_vs_barrier()
    test_mast_height_coverage_scaling()
    test_point_to_point_los_clear_and_blocked()
    test_fastapi_viewshed_and_los_endpoints()
    print("================================================================================")
    print("ALL VIEWSHED & LINE-OF-SIGHT (§4.1) TESTS PASSED SUCCESSFULLY!")
    print("================================================================================")
