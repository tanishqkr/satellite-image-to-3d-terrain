"""
Unit and Integration Tests for Visibility, Ray Marching, and Probabilistic Occupancy (§3.2, §4.1).

Covers:
1. Bayesian log-odds formulation, inverse sensor model, and clamping bounds.
2. Shared ray-marching point-to-point Line of Sight (LoS) and elevation profile generation.
3. Single-pass camera / sensor geometric occlusion pass populating unobserved regions.
4. Tri-state voxel grid integration (populating VoxelState.UNKNOWN).
5. Observer viewshed computation (shared foundation for §4.1).
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.visibility import (
    probability_to_log_odds,
    log_odds_to_probability,
    bayesian_log_odds_update,
    ray_march_los,
    compute_camera_occlusion_mask,
    compute_viewshed,
)
from app.services.voxel_grid import SparseVoxelGrid, VoxelState


def test_bayesian_log_odds():
    print("Testing Bayesian log-odds update math...")
    # 1. Neutral prior (p = 0.5) corresponds to log-odds = 0.0
    l_init = probability_to_log_odds(0.5)
    assert abs(l_init) < 1e-6, f"Expected 0.0 log-odds for p=0.5, got {l_init}"
    assert abs(log_odds_to_probability(0.0) - 0.5) < 1e-6

    # 2. Single positive hit with p_occ=0.70 increases log-odds
    l_1 = bayesian_log_odds_update(0.0, is_occupied=True, p_occ=0.70)
    expected_delta = np.log(0.70 / 0.30)
    assert abs(l_1 - expected_delta) < 1e-5, f"Expected {expected_delta}, got {l_1}"
    p_1 = log_odds_to_probability(l_1)
    assert 0.69 < p_1 < 0.71, f"Expected p ~ 0.70, got {p_1}"

    # 3. Multiple consecutive hits accumulate and clamp at l_max (+5.0)
    l_curr = 0.0
    for _ in range(10):
        l_curr = bayesian_log_odds_update(l_curr, is_occupied=True, l_max=5.0)
    assert abs(l_curr - 5.0) < 1e-6, f"Expected clamped to 5.0, got {l_curr}"
    p_high = log_odds_to_probability(l_curr)
    assert p_high > 0.99, f"Expected probability > 0.99, got {p_high}"

    # 4. Multiple consecutive misses accumulate downward and clamp at l_min (-5.0)
    for _ in range(25):
        l_curr = bayesian_log_odds_update(l_curr, is_occupied=False, l_min=-5.0)
    assert abs(l_curr - (-5.0)) < 1e-6, f"Expected clamped to -5.0, got {l_curr}"
    p_low = log_odds_to_probability(l_curr)
    assert p_low < 0.01, f"Expected probability < 0.01, got {p_low}"

    # 5. Vectorized array test
    priors = np.zeros((3, 3), dtype=np.float64)
    obs = np.array([[True, False, True], [False, True, False], [True, True, False]])
    updated = bayesian_log_odds_update(priors, obs)
    assert updated.shape == (3, 3)
    assert updated[0, 0] > 0.0
    assert updated[0, 1] < 0.0

    print("  [PASS] Bayesian log-odds conversion, accumulation, and clamping: PASSED")


def test_ray_march_los():
    print("Testing shared ray-marching point-to-point Line of Sight (LoS)...")
    H, W = 50, 50
    # Flat terrain at elevation 5m
    dsm = np.full((H, W), 5.0, dtype=np.float32)

    # 1. Unobstructed line of sight between two masts at 15m elevation
    p0 = (5.0, 15.0, 5.0)
    p1 = (45.0, 15.0, 45.0)
    res_clear = ray_march_los(p0, p1, dsm, gsd=1.0)
    assert res_clear["visible"] is True, "Line of sight across flat ground must be visible"
    assert res_clear["blocking_point"] is None
    assert len(res_clear["elevation_profile"]) > 10

    # 2. Introduce a 30m ridge at center (25, 25)
    dsm[22:28, 22:28] = 30.0
    res_blocked = ray_march_los(p0, p1, dsm, gsd=1.0)
    assert res_blocked["visible"] is False, "Line of sight must be obstructed by 30m ridge"
    assert res_blocked["blocking_point"] is not None
    bx, by, bz = res_blocked["blocking_point"]
    assert 20.0 <= bx <= 30.0 and 20.0 <= bz <= 30.0
    assert by > 15.0, f"Blocking elevation {by} must exceed ray altitude (15.0)"

    print("  [PASS] Line of sight clear and blocked conditions with elevation profile: PASSED")


def test_camera_occlusion_mask():
    print("Testing single-pass camera/sensor geometric occlusion pass (§3.2)...")
    H, W = 40, 40
    # Flat terrain at elevation 0m
    dsm = np.zeros((H, W), dtype=np.float32)
    # Tall 20m structure at center (row 15:20, col 15:20)
    dsm[15:20, 15:20] = 20.0

    # Drone camera positioned at (col=5, row=5) with elevation Y=35m
    # Looking down obliquely toward southeast
    camera_pos = (5.0, 35.0, 5.0)  # X, Y, Z in meters

    occlusion = compute_camera_occlusion_mask(
        dsm=dsm,
        camera_pos=camera_pos,
        gsd=1.0,
        origin=(0.0, 0.0, 0.0),
    )
    assert occlusion.shape == (H, W)

    # 1. The ground in front of the building (closer to camera) is visible
    assert not occlusion[8, 8], "Ground in front of structure must be visible"

    # 2. The building roof itself is visible
    assert not occlusion[16, 16], "Top of building must be visible to camera"

    # 3. Ground directly behind the building in the geometric shadow cast from camera is occluded
    # Ray from (5, 35, 5) passing over (19, 20, 19) strikes ground further down (e.g. 25, 25)
    shadow_cells_occluded = np.sum(occlusion[21:30, 21:30])
    assert shadow_cells_occluded > 0, f"Expected occluded shadow behind building, got {shadow_cells_occluded}"

    print(f"  [PASS] Geometric occlusion shadow detected {shadow_cells_occluded} occluded cells: PASSED")


def test_voxel_grid_tri_state_occlusion():
    print("Testing SparseVoxelGrid tri-state UNKNOWN population...")
    H, W = 32, 32
    dsm = np.full((H, W), 2.0, dtype=np.float32)
    dsm[10:16, 10:16] = 18.0  # Structure

    cam_pos = (2.0, 25.0, 2.0)
    grid = SparseVoxelGrid.from_dsm(
        dsm=dsm,
        gsd=1.0,
        voxel_size_y=1.0,
        camera_pos=cam_pos,
    )

    occ_coords = grid.get_occupied_coords()
    unk_coords = grid.get_unknown_coords()

    assert len(occ_coords) > 0, "Must have occupied voxels"
    assert len(unk_coords) > 0, f"Must have UNKNOWN voxels in camera shadow, got {len(unk_coords)}"

    # Check that roof of structure is OCCUPIED (height_index is 16, so voxels 0..15 are solid)
    roof_state = grid.get_voxel(12, 15, 12)
    assert roof_state == VoxelState.OCCUPIED, f"Expected OCCUPIED roof, got {roof_state}"

    # Also test post-processing apply_occlusion_mask
    fresh_grid = SparseVoxelGrid.from_dsm(dsm=dsm, gsd=1.0, voxel_size_y=1.0)
    assert len(fresh_grid.get_unknown_coords()) == 0, "Fresh grid without camera pos has 0 UNKNOWN voxels"

    mask = compute_camera_occlusion_mask(dsm, cam_pos, gsd=1.0)
    fresh_grid.apply_occlusion_mask(mask)
    assert len(fresh_grid.get_unknown_coords()) > 0, "Post-processed grid must have UNKNOWN voxels"

    print("  [PASS] SparseVoxelGrid tri-state UNKNOWN state integration verified: PASSED")


def test_viewshed_computation():
    print("Testing observer viewshed computation (§4.1 foundation)...")
    H, W = 40, 40
    # Flat terrain with elevation 0m
    dsm = np.zeros((H, W), dtype=np.float32)
    # Hill crest across the map at row 20
    dsm[20, :] = 15.0

    # Observer at south side (col=20, row=10), eye height = 2m
    obs_pos = (20.0, 2.0, 10.0)

    viewshed = compute_viewshed(
        dsm=dsm,
        observer_pos=obs_pos,
        gsd=1.0,
        origin=(0.0, 0.0, 0.0),
    )

    assert viewshed.shape == (H, W)
    # Observer side of the ridge (rows 10-19) is visible
    assert viewshed[15, 20] == 1, "Ground between observer and ridge must be visible"
    # The ridge crest itself is visible
    assert viewshed[20, 20] == 1, "Ridge crest must be visible"
    # Ground on the far side of the ridge (rows 22-30) is occluded by the 15m hill
    assert viewshed[25, 20] == 0, "Ground behind 15m ridge must be occluded"

    print("  [PASS] Observer viewshed ridge occlusion verified: PASSED")


if __name__ == "__main__":
    test_bayesian_log_odds()
    test_ray_march_los()
    test_camera_occlusion_mask()
    test_voxel_grid_tri_state_occlusion()
    test_viewshed_computation()
    print("\nALL VISIBILITY & OCCUPANCY GRID (§3.2, §4.1) TESTS PASSED!")
