"""
Shared Visibility, Ray Marching, and Probabilistic Occupancy Service (§3.2, §4.1).

Provides:
1. Standalone Bayesian log-odds occupancy formulation (OctoMap-compatible inverse sensor model).
2. Geometric line-of-sight (LoS) ray marching between 3D points.
3. Camera / sensor occlusion pass identifying unobserved (UNKNOWN) terrain and structure shadows.
4. Radial viewshed analysis for tactical observer siting (§4.1 foundation).
"""
from typing import Tuple, List, Dict, Optional, Union
import numpy as np

from app.config import (
    BAYESIAN_P_OCC,
    BAYESIAN_P_FREE,
    BAYESIAN_L_MIN,
    BAYESIAN_L_MAX,
)
from app.logging_config import log


# =============================================================================
# 1. BAYESIAN LOG-ODDS FORMULATION (OctoMap Inverse Sensor Model)
# =============================================================================

def probability_to_log_odds(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert probability in (0, 1) to log-odds representation:
    L(p) = ln(p / (1 - p)).
    Clamps p to [1e-6, 1 - 1e-6] to avoid division by zero or infinities.
    """
    p_clamped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clamped / (1.0 - p_clamped))


def log_odds_to_probability(l: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert log-odds value to probability in [0, 1]:
    p(L) = 1 / (1 + exp(-L)) = 1 - 1 / (1 + exp(L)).
    Numerically stable sigmoid preventing overflow.
    """
    l_arr = np.asarray(l, dtype=np.float64)
    pos_mask = l_arr >= 0
    p = np.empty_like(l_arr, dtype=np.float64)
    p[pos_mask] = 1.0 / (1.0 + np.exp(-l_arr[pos_mask]))
    exp_l = np.exp(l_arr[~pos_mask])
    p[~pos_mask] = exp_l / (1.0 + exp_l)

    if isinstance(l, (float, int, np.floating)):
        return float(p.item())
    return p


def bayesian_log_odds_update(
    prior_log_odds: Union[float, np.ndarray],
    is_occupied: Union[bool, np.ndarray],
    p_occ: float = BAYESIAN_P_OCC,
    p_free: float = BAYESIAN_P_FREE,
    l_min: float = BAYESIAN_L_MIN,
    l_max: float = BAYESIAN_L_MAX,
) -> Union[float, np.ndarray]:
    """
    OctoMap recursive log-odds update:
        L(m_i | z_1:t) = L(m_i | z_1:t-1) + L(m_i | z_t) - L_0

    Parameters:
        prior_log_odds: Previous log-odds value L(m_i | z_1:t-1) (0.0 = completely unknown / unobserved).
        is_occupied: True if current observation indicates occupied (hit), False if free (miss).
        p_occ: Probability of occupancy given a hit (default 0.70).
        p_free: Probability of occupancy given a miss (default 0.40).
        l_min: Lower clamping bound on log-odds (default -5.0).
        l_max: Upper clamping bound on log-odds (default +5.0).

    Returns:
        Updated log-odds clamped to [l_min, l_max].
    """
    l_occ = np.log(p_occ / (1.0 - p_occ))
    l_free = np.log(p_free / (1.0 - p_free))

    if isinstance(is_occupied, (bool, int, np.bool_)):
        delta = l_occ if is_occupied else l_free
        new_l = prior_log_odds + delta
        return float(np.clip(new_l, l_min, l_max))

    # Array path
    is_occ_arr = np.asarray(is_occupied, dtype=bool)
    delta_arr = np.where(is_occ_arr, l_occ, l_free)
    new_l = np.asarray(prior_log_odds, dtype=np.float64) + delta_arr
    return np.clip(new_l, l_min, l_max)


# =============================================================================
# 2. SHARED RAY-MARCHING LINE OF SIGHT (LoS) UTILITY
# =============================================================================

def sample_dsm_bilinear(
    dsm: np.ndarray,
    col_f: float,
    row_f: float,
    base_elevation: float = 0.0,
) -> float:
    """Sample DSM elevation with bilinear interpolation."""
    H, W = dsm.shape
    c0 = int(np.floor(col_f))
    r0 = int(np.floor(row_f))
    c1 = min(W - 1, c0 + 1)
    r1 = min(H - 1, r0 + 1)

    c0 = max(0, min(W - 1, c0))
    r0 = max(0, min(H - 1, r0))

    fc = col_f - c0
    fr = row_f - r0

    v00 = dsm[r0, c0]
    v10 = dsm[r0, c1]
    v01 = dsm[r1, c0]
    v11 = dsm[r1, c1]

    # Handle NaNs safely
    v00 = base_elevation if np.isnan(v00) else v00
    v10 = base_elevation if np.isnan(v10) else v10
    v01 = base_elevation if np.isnan(v01) else v01
    v11 = base_elevation if np.isnan(v11) else v11

    top = v00 * (1.0 - fc) + v10 * fc
    bottom = v01 * (1.0 - fc) + v11 * fc
    return float(top * (1.0 - fr) + bottom * fr)


def ray_march_los(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    dsm: np.ndarray,
    gsd: float = 1.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    sample_step_m: Optional[float] = None,
    tolerance_m: float = 0.05,
) -> Dict:
    """
    March a 3D ray between p0 and p1 through DSM terrain surface.

    Parameters:
        p0: (x0, y0, z0) starting world position (meters). y is vertical elevation.
        p1: (x1, y1, z1) ending world position (meters).
        dsm: 2D numpy array (H, W) elevation grid.
        gsd: Ground sample distance in meters.
        origin: (ox, oy, oz) world origin corresponding to dsm[0, 0].
        sample_step_m: Step length along ray in meters (defaults to 0.5 * gsd).
        tolerance_m: Clearance threshold to avoid endpoint self-intersection.

    Returns:
        Dict with:
            visible: bool (True if line of sight is unobstructed)
            distance_m: total 3D distance in meters
            blocking_point: (bx, by, bz) world coordinate of first obstruction, or None
            elevation_profile: List of samples along ray with distance, ray height, terrain height, and blocked flag
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    ox, oy, oz = origin

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    dist_3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))

    if dist_3d < 1e-4:
        return {
            "visible": True,
            "distance_m": 0.0,
            "blocking_point": None,
            "elevation_profile": [],
        }

    H, W = dsm.shape
    step = sample_step_m if sample_step_m is not None else max(0.2, 0.5 * gsd)
    num_steps = max(2, int(np.ceil(dist_3d / step)))

    profile: List[Dict] = []
    visible = True
    blocking_point = None

    for i in range(num_steps + 1):
        t = i / float(num_steps)
        curr_x = x0 + t * dx
        curr_y = y0 + t * dy
        curr_z = z0 + t * dz
        curr_dist = t * dist_3d

        col_f = (curr_x - ox) / gsd
        row_f = (curr_z - oz) / gsd

        in_bounds = 0.0 <= col_f <= (W - 1) and 0.0 <= row_f <= (H - 1)
        if in_bounds:
            terr_y = sample_dsm_bilinear(dsm, col_f, row_f)
        else:
            terr_y = float(np.nanmin(dsm))

        is_blocked = False
        if 0.02 < t < 0.98 and in_bounds:
            if terr_y > (curr_y + tolerance_m):
                is_blocked = True
                if visible:
                    visible = False
                    blocking_point = (round(curr_x, 2), round(terr_y, 2), round(curr_z, 2))

        profile.append({
            "distance_m": round(curr_dist, 2),
            "ray_elevation_m": round(curr_y, 2),
            "terrain_elevation_m": round(terr_y, 2),
            "blocked": is_blocked,
        })

    return {
        "visible": visible,
        "distance_m": round(dist_3d, 2),
        "blocking_point": blocking_point,
        "elevation_profile": profile,
    }


# =============================================================================
# 3. SENSOR / CAMERA OCCLUSION PASS (§3.2 Populating UNKNOWN Voxels)
# =============================================================================

def compute_camera_occlusion_mask(
    dsm: np.ndarray,
    camera_pos: Tuple[float, float, float],
    gsd: float = 1.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    tolerance_deg: float = 0.1,
) -> np.ndarray:
    """
    Compute 2D boolean occlusion mask (H, W) for single-pass imagery.
    Determines whether a ground/roof column is occluded from the camera position
    by an intervening taller structure.

    Parameters:
        dsm: 2D numpy array (H, W) elevation grid.
        camera_pos: (cam_x, cam_y, cam_z) world coordinate of camera sensor in meters.
                    cam_y is vertical elevation.
        gsd: Ground sample distance in meters.
        origin: (ox, oy, oz) world coordinates of dsm[0, 0].
        tolerance_deg: Angular tolerance buffer.

    Returns:
        occlusion_mask: np.ndarray (H, W) of dtype bool.
                        True = OCCLUDED (camera could not observe this column -> UNKNOWN)
                        False = VISIBLE (observed directly by sensor).
    """
    H, W = dsm.shape
    cam_x, cam_y, cam_z = camera_pos
    ox, oy, oz = origin

    cam_col = (cam_x - ox) / gsd
    cam_row = (cam_z - oz) / gsd

    occlusion_mask = np.zeros((H, W), dtype=bool)

    # Perimeter cells to cast radial rays to
    boundary_cells = []
    for c in range(W):
        boundary_cells.append((0, c))
        boundary_cells.append((H - 1, c))
    for r in range(1, H - 1):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, W - 1))

    tol_tan = float(np.tan(np.radians(tolerance_deg)))

    for target_r, target_c in boundary_cells:
        dr = target_r - cam_row
        dc = target_c - cam_col
        dist_grid = np.hypot(dr, dc)
        if dist_grid < 1e-4:
            continue

        num_steps = max(2, int(np.ceil(dist_grid)))
        step_r = dr / num_steps
        step_c = dc / num_steps

        # Track maximum elevation angle (slope) from camera downward
        # slope = (terrain_h - cam_y) / (dist_m).
        max_slope = -float("inf")

        for s in range(1, num_steps + 1):
            r_idx = int(round(cam_row + s * step_r))
            c_idx = int(round(cam_col + s * step_c))

            if 0 <= r_idx < H and 0 <= c_idx < W:
                h_val = float(dsm[r_idx, c_idx])
                if np.isnan(h_val):
                    continue

                dist_m = float(np.hypot(r_idx - cam_row, c_idx - cam_col) * gsd)
                if dist_m < 1e-3:
                    continue

                slope = (h_val - cam_y) / dist_m

                if slope < (max_slope - tol_tan):
                    occlusion_mask[r_idx, c_idx] = True
                else:
                    max_slope = max(max_slope, slope)

    num_occluded = int(np.sum(occlusion_mask))
    total_cells = H * W
    log.info(
        "Camera occlusion pass computed",
        occluded_cells=num_occluded,
        total_cells=total_cells,
        occlusion_ratio=f"{(num_occluded / total_cells) * 100:.1f}%",
    )
    return occlusion_mask


# =============================================================================
# 4. OBSERVER VIEWSHED UTILITY (Shared Foundation for §4.1)
# =============================================================================

def compute_viewshed(
    dsm: np.ndarray,
    observer_pos: Tuple[float, float, float],
    gsd: float = 1.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_range_m: Optional[float] = None,
    target_height_m: float = 0.0,
) -> np.ndarray:
    """
    Compute 2D boolean viewshed (H, W) from an observer location.

    Parameters:
        dsm: 2D numpy array (H, W) elevation grid.
        observer_pos: (obs_x, obs_y, obs_z) in world units. obs_y is eye/mast elevation.
        gsd: Ground sample distance in meters.
        origin: (ox, oy, oz) world coordinate of dsm[0, 0].
        max_range_m: Optional maximum radius of regard in meters.
        target_height_m: Elevation offset of target above ground (e.g. 1.8m for human standing).

    Returns:
        viewshed: np.ndarray (H, W) uint8 (1 = Visible to observer, 0 = Occluded / Hidden).
    """
    H, W = dsm.shape
    obs_x, obs_y, obs_z = observer_pos
    ox, oy, oz = origin

    obs_col = (obs_x - ox) / gsd
    obs_row = (obs_z - oz) / gsd

    viewshed = np.zeros((H, W), dtype=np.uint8)

    obs_r_int = int(round(obs_row))
    obs_c_int = int(round(obs_col))
    if 0 <= obs_r_int < H and 0 <= obs_c_int < W:
        viewshed[obs_r_int, obs_c_int] = 1

    boundary_cells = []
    for c in range(W):
        boundary_cells.append((0, c))
        boundary_cells.append((H - 1, c))
    for r in range(1, H - 1):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, W - 1))

    for target_r, target_c in boundary_cells:
        dr = target_r - obs_row
        dc = target_c - obs_col
        dist_grid = np.hypot(dr, dc)
        if dist_grid < 1e-4:
            continue

        num_steps = max(2, int(np.ceil(dist_grid)))
        step_r = dr / num_steps
        step_c = dc / num_steps

        max_tangent = -float("inf")

        for s in range(1, num_steps + 1):
            r_idx = int(round(obs_row + s * step_r))
            c_idx = int(round(obs_col + s * step_c))

            if 0 <= r_idx < H and 0 <= c_idx < W:
                dist_m = float(np.hypot(r_idx - obs_row, c_idx - obs_col) * gsd)
                if max_range_m is not None and dist_m > max_range_m:
                    break

                h_val = float(dsm[r_idx, c_idx])
                if np.isnan(h_val):
                    continue

                target_tangent = ((h_val + target_height_m) - obs_y) / dist_m
                terrain_tangent = (h_val - obs_y) / dist_m

                if target_tangent >= max_tangent:
                    viewshed[r_idx, c_idx] = 1

                if terrain_tangent > max_tangent:
                    max_tangent = terrain_tangent

    return viewshed
