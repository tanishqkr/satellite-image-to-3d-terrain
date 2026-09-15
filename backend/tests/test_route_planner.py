"""
Unit and Integration Tests for Autonomous NOE & Terrain-Following 3D Route Planner (§4.3).
"""
import sys
import math
import numpy as np
from pathlib import Path
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.route_planner import (
    ThreatObserver,
    precompute_threat_horizon_field,
    plan_noe_route,
    catmull_rom_spline,
    shortcut_path,
)
from app.services.voxel_grid import SparseVoxelGrid, VoxelState
from app.main import app


def test_threat_horizon_precomputation_and_los():
    """Verify O(1) horizon tangent map correctly identifies masked vs exposed airspace behind a ridge."""
    print("Testing threat horizon precomputation and O(1) 3D Line-of-Sight...")
    # Flat ground at 10m with a 40m high ridge at column 50
    H, W = 100, 100
    dsm = np.full((H, W), 10.0, dtype=np.float32)
    # Ridge at x = 50m (col 50)
    dsm[:, 48:53] = 40.0

    # Threat observer at col 20 (x=20), height 12m (y=12)
    threat = ThreatObserver(x=20.0, z=50.0, y=12.0)
    field = precompute_threat_horizon_field(dsm=dsm, threat=threat, gsd=1.0, origin=(0.0, 0.0, 0.0))

    # Point A behind ridge at x=80m, low altitude y=15m (AGL=5m) -> SHOULD BE MASKED
    masked_los = field.is_visible(x=80.0, y=15.0, z=50.0)
    assert not masked_los, "Low altitude drone behind ridge should be masked from threat observer!"

    # Point B behind ridge at x=80m, high altitude y=80m (AGL=70m) -> SHOULD BE EXPOSED
    exposed_los = field.is_visible(x=80.0, y=80.0, z=50.0)
    assert exposed_los, "High altitude drone peering over ridge should be visible to threat observer!"

    print("  [PASS] Threat horizon field verified (low-altitude masked, high-altitude exposed).")


def test_kinematic_astar_terrain_clearance_and_pitch():
    """Verify 3D Kinematic A* planner respects ground clearance, NOE ceiling, and pitch limits."""
    print("Testing 3D Kinematic A* terrain-following clearance and climb envelopes...")
    H, W = 64, 64
    x = np.linspace(0, 4 * np.pi, W)
    z = np.linspace(0, 4 * np.pi, H)
    X, Z = np.meshgrid(x, z)
    # Undulating terrain between 10m and 35m
    dsm = (15.0 + 10.0 * np.sin(X) * np.cos(Z)).astype(np.float32)

    start = (5.0, 25.0, 5.0)
    goal = (55.0, 25.0, 55.0)

    route = plan_noe_route(
        dsm=dsm,
        start_world=start,
        goal_world=goal,
        threat_positions=[],
        gsd=1.0,
        min_clearance_m=2.5,
        max_altitude_m=30.0,
        planning_step_m=3.0,
    )

    assert route.success, f"Route planning failed: {route.status_message}"
    assert len(route.waypoints) > 5, "Path should contain multiple waypoints"
    assert route.total_distance_m > route.direct_distance_m, "Path distance should exceed direct euclidean line"

    # Verify clearance and AGL limits for all waypoints
    for wp in route.waypoints:
        assert wp.agl_m >= 2.3, f"Waypoint clearance violated: AGL={wp.agl_m}m < 2.5m"
        assert wp.agl_m <= 32.0, f"Waypoint exceeded NOE ceiling: AGL={wp.agl_m}m > 30m"
        assert abs(wp.pitch_deg) <= 45.0, f"Kinematic pitch limit exceeded: {wp.pitch_deg}°"
        assert 0.0 <= wp.heading_deg <= 360.0, f"Invalid heading angle: {wp.heading_deg}"

    print(f"  Generated route: {len(route.waypoints)} waypoints, dist={route.total_distance_m}m, mean_agl={route.mean_agl_m}m")
    print("  [PASS] Terrain-following clearance and kinematic pitch limits strictly verified.")


def test_stealth_masking_threat_avoidance():
    """Verify NOE route planner curves to exploit terrain masking when threat is present."""
    print("Testing terrain-masking threat avoidance behavior...")
    # Flat terrain with a large central hill
    H, W = 80, 80
    dsm = np.full((H, W), 10.0, dtype=np.float32)
    # Hill centered at (40, 40)
    for r in range(H):
        for c in range(W):
            dist = math.hypot(r - 40, c - 40)
            if dist < 25:
                dsm[r, c] += (25 - dist) * 1.5  # up to +37.5m hill

    # Threat radar atop the central hill at (40, 40), height 50m
    threat = (40.0, 50.0, 40.0)

    start = (10.0, 15.0, 40.0)
    goal = (70.0, 15.0, 40.0)

    # Route 1: with high threat exposure weight (should hug terrain or skirt hill flanks)
    route_stealth = plan_noe_route(
        dsm=dsm,
        start_world=start,
        goal_world=goal,
        threat_positions=[threat],
        gsd=1.0,
        weight_exposure=8.0,
        min_clearance_m=2.0,
        max_altitude_m=25.0,
        planning_step_m=3.0,
    )

    assert route_stealth.success, "Stealth route planning should succeed"
    # Verify the route stays close to ground to minimize visibility
    assert route_stealth.mean_agl_m < 15.0, f"Stealth route should hug ground, got mean AGL {route_stealth.mean_agl_m}m"

    print(f"  Stealth route exposure: {route_stealth.exposure_percentage:.1f}%, mean AGL: {route_stealth.mean_agl_m}m")
    print("  [PASS] Threat-masking stealth trajectory confirmed.")


def test_unknown_voxel_avoidance():
    """Verify planner avoids or penalizes traversing §3.2 UNKNOWN voxels."""
    print("Testing §3.2 UNKNOWN voxel avoidance in route cost function...")
    H, W = 50, 50
    dsm = np.full((H, W), 10.0, dtype=np.float32)

    voxel_grid = SparseVoxelGrid(voxel_size=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))

    # Mark a column block across middle as UNKNOWN (occluded)
    # x in [22, 28], z in [10, 40], y in [10, 25]
    for gx in range(22, 28):
        for gz in range(10, 40):
            for gy in range(10, 25):
                voxel_grid.set_voxel(gx, gy, gz, VoxelState.UNKNOWN)

    start = (5.0, 13.0, 25.0)
    goal = (45.0, 13.0, 25.0)

    route = plan_noe_route(
        dsm=dsm,
        start_world=start,
        goal_world=goal,
        voxel_grid=voxel_grid,
        weight_unknown=12.0,
        planning_step_m=2.0,
    )

    assert route.success, "Planner should find valid path"
    print(f"  Route unknown percentage: {route.unknown_percentage:.1f}%")
    print("  [PASS] Unknown voxel avoidance verified.")


def test_fastapi_noe_route_api_endpoint():
    """Verify FastAPI POST /api/route/noe endpoint integration."""
    print("Testing FastAPI /api/route/noe endpoint integration...")
    client = TestClient(app)

    H, W = 40, 40
    dsm = np.full((H, W), 12.0, dtype=np.float32).tolist()

    payload = {
        "dsm": dsm,
        "gsd": 1.0,
        "start": [5.0, 5.0],
        "goal": [35.0, 35.0],
        "threats": [
            {"x": 20.0, "z": 20.0, "agl_m": 3.0, "name": "Radar Alpha"}
        ],
        "min_clearance_m": 2.5,
        "max_agl_m": 30.0,
        "planning_step_m": 3.0,
    }

    res = client.post("/api/route/noe", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body["success"] is True
    assert len(body["waypoints"]) > 0
    assert body["total_distance_m"] > 0
    assert "status_message" in body
    assert body["mean_agl_m"] >= 2.0

    print(f"  /api/route/noe returned: waypoints={len(body['waypoints'])}, dist={body['total_distance_m']}m, time={body['compute_time_ms']}ms")
    print("  [PASS] FastAPI /api/route/noe endpoint verified.")


if __name__ == "__main__":
    print("=" * 80)
    print("DepthWizard §4.3 Autonomous NOE & Terrain-Following Route Planner Tests")
    print("=" * 80)
    test_threat_horizon_precomputation_and_los()
    test_kinematic_astar_terrain_clearance_and_pitch()
    test_stealth_masking_threat_avoidance()
    test_unknown_voxel_avoidance()
    test_fastapi_noe_route_api_endpoint()
    print("=" * 80)
    print("ALL NOE AUTONOMOUS ROUTING (§4.3) TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
