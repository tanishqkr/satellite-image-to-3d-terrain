"""
Oblique & Off-Nadir Perspective Rectification (True-Orthorectification) (§2.2).
Provides two-pass rectification:
- Pass A (pre_rectify): Coarse de-obliquing using global DTM proxy surface before ML inference
  so the model receives near-nadir input geometry matching its training distribution.
- Pass B (post_rectify): Fine true-orthorectification using calibrated fine DSM with
  Z-buffer occlusion handling to prevent facade duplication and leaning-building artifacts.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import map_coordinates

from app.logging_config import log
from app.config import ORTHORECTIFY_ENABLED, OBLIQUE_THRESHOLD_DEG


@dataclass
class CameraPose:
    pitch_deg: float = 0.0      # Off-nadir angle in degrees (0 = nadir, 45 = 45 deg tilt)
    roll_deg: float = 0.0       # Camera roll around optical axis in degrees
    yaw_deg: float = 0.0        # Heading/azimuth in degrees (0 = North)
    altitude_m: float = 100.0   # Height above ground in meters
    fov_deg: float = 60.0       # Field of view in degrees
    is_oblique: bool = False

    def __post_init__(self):
        # Auto-flag obliqueness if off-nadir pitch exceeds threshold
        if abs(self.pitch_deg) > OBLIQUE_THRESHOLD_DEG or abs(self.roll_deg) > OBLIQUE_THRESHOLD_DEG:
            self.is_oblique = True


def get_camera_basis(
    pitch_deg: float,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
    altitude_m: float = 100.0,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute camera center C and orthonormal basis {u_cam (right), v_cam (down), w_cam (forward)}
    in standard computer vision coordinate convention from tactical drone orientation.
    """
    T = np.array(target, dtype=np.float64)
    pitch = np.radians(max(abs(pitch_deg), 1e-4))
    roll = np.radians(roll_deg)
    yaw = np.radians(yaw_deg)
    alt = max(float(altitude_m), 10.0)

    # Displace camera position horizontally in opposite direction of heading
    horiz_dist = alt * np.tan(pitch)
    C = np.array([
        T[0] - horiz_dist * np.sin(yaw),
        T[1] - horiz_dist * np.cos(yaw),
        T[2] + alt,
    ], dtype=np.float64)

    # Forward vector looking at ground target
    w_cam = (T - C) / np.linalg.norm(T - C)

    # World up vector
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u_base = np.cross(w_cam, up)
    norm_u = np.linalg.norm(u_base)
    if norm_u < 1e-7:
        u_base = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u_base = u_base / norm_u

    v_base = np.cross(w_cam, u_base)
    v_base = v_base / np.linalg.norm(v_base)

    # Apply camera roll around optical axis
    u_cam = np.cos(roll) * u_base + np.sin(roll) * v_base
    v_cam = -np.sin(roll) * u_base + np.cos(roll) * v_base

    return C, u_cam, v_cam, w_cam


def project_ground_to_camera(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    C: np.ndarray,
    u_cam: np.ndarray,
    v_cam: np.ndarray,
    w_cam: np.ndarray,
    f: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized projection of 3D ground coordinates to camera sensor plane (u, v)."""
    Delta_X = X - C[0]
    Delta_Y = Y - C[1]
    Delta_Z = Z - C[2]

    X_c = Delta_X * u_cam[0] + Delta_Y * u_cam[1] + Delta_Z * u_cam[2]
    Y_c = Delta_X * v_cam[0] + Delta_Y * v_cam[1] + Delta_Z * v_cam[2]
    Z_c = Delta_X * w_cam[0] + Delta_Y * w_cam[1] + Delta_Z * w_cam[2]

    valid = Z_c > 1.0
    safe_Z_c = np.where(valid, Z_c, 1.0)

    u = f * (X_c / safe_Z_c) + cx
    v = f * (Y_c / safe_Z_c) + cy
    dist = np.sqrt(X_c**2 + Y_c**2 + Z_c**2)

    return u, v, dist, valid


def pre_rectify(
    image: np.ndarray,
    coarse_dtm: Optional[np.ndarray],
    camera_pose: CameraPose,
    gsd: float = 0.5,
) -> np.ndarray:
    """
    Pass A: Pre-inference de-obliquing.
    Projects oblique input image onto a nadir coordinate grid using the coarse DTM proxy surface.
    Keeps depth estimation in-distribution before inference.
    """
    if not ORTHORECTIFY_ENABLED or not camera_pose.is_oblique:
        return image

    H, W = image.shape[:2]
    safe_gsd = max(float(gsd), 0.05) if gsd is not None else 0.5

    # Ground coordinates centered at (0, 0)
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    X = (x_idx - W / 2.0) * safe_gsd
    Y = (y_idx - H / 2.0) * safe_gsd

    if coarse_dtm is not None and coarse_dtm.shape == (H, W):
        Z = coarse_dtm.astype(np.float64)
    else:
        Z = np.zeros((H, W), dtype=np.float64)

    target_center = (0.0, 0.0, float(Z[H // 2, W // 2]))
    C, u_cam, v_cam, w_cam = get_camera_basis(
        pitch_deg=camera_pose.pitch_deg,
        roll_deg=camera_pose.roll_deg,
        yaw_deg=camera_pose.yaw_deg,
        altitude_m=camera_pose.altitude_m,
        target=target_center,
    )

    fov_rad = np.radians(camera_pose.fov_deg)
    f = (W / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0

    u, v, _, valid = project_ground_to_camera(X, Y, Z, C, u_cam, v_cam, w_cam, f, cx, cy)
    in_bounds = valid & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)

    rectified = np.zeros_like(image)
    coords = np.vstack([v.ravel(), u.ravel()])

    for c in range(3):
        sampled = map_coordinates(
            image[..., c].astype(np.float32),
            coords,
            order=1,
            mode='nearest',
        ).reshape(H, W)
        rectified[..., c] = np.where(in_bounds, np.clip(sampled, 0, 255), image[..., c]).astype(np.uint8)

    log.info(
        "Pre-rectification Pass A complete",
        pitch=camera_pose.pitch_deg,
        valid_coverage=f"{float(in_bounds.mean()) * 100:.1f}%",
    )
    return rectified


def post_rectify(
    image: np.ndarray,
    fine_dsm: np.ndarray,
    camera_pose: CameraPose,
    gsd: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pass B: Post-inference true-orthorectification.
    Back-projects nadir grid through camera pose using fine DSM with Z-buffer occlusion handling.
    Returns:
        (ortho_rgb, occlusion_mask)
    """
    if not ORTHORECTIFY_ENABLED or not camera_pose.is_oblique:
        return image, np.zeros(image.shape[:2], dtype=bool)

    H, W = image.shape[:2]
    safe_gsd = max(float(gsd), 0.05) if gsd is not None else 0.5

    y_idx, x_idx = np.mgrid[0:H, 0:W]
    X = (x_idx - W / 2.0) * safe_gsd
    Y = (y_idx - H / 2.0) * safe_gsd
    Z = fine_dsm.astype(np.float64)

    target_center = (0.0, 0.0, float(Z[H // 2, W // 2]))
    C, u_cam, v_cam, w_cam = get_camera_basis(
        pitch_deg=camera_pose.pitch_deg,
        roll_deg=camera_pose.roll_deg,
        yaw_deg=camera_pose.yaw_deg,
        altitude_m=camera_pose.altitude_m,
        target=target_center,
    )

    fov_rad = np.radians(camera_pose.fov_deg)
    f = (W / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = W / 2.0, H / 2.0

    u, v, dist, valid = project_ground_to_camera(X, Y, Z, C, u_cam, v_cam, w_cam, f, cx, cy)
    in_bounds = valid & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)

    # --- Z-buffer occlusion test ---
    u_px = np.clip(np.round(u[in_bounds]).astype(np.int32), 0, W - 1)
    v_px = np.clip(np.round(v[in_bounds]).astype(np.int32), 0, H - 1)
    d_vals = dist[in_bounds].astype(np.float32)

    z_buffer = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, (v_px, u_px), d_vals)

    # Occluded if distance exceeds minimum distance recorded in cell by threshold
    tol = np.maximum(0.02 * dist, 1.5)
    occluded = np.zeros((H, W), dtype=bool)

    u_all_px = np.clip(np.round(u).astype(np.int32), 0, W - 1)
    v_all_px = np.clip(np.round(v).astype(np.int32), 0, H - 1)
    occluded[in_bounds] = dist[in_bounds] > (z_buffer[v_all_px[in_bounds], u_all_px[in_bounds]] + tol[in_bounds])

    # Visible & in-bounds pixels get sampled
    visible = in_bounds & (~occluded)

    ortho_rgb = np.zeros_like(image)
    coords = np.vstack([v.ravel(), u.ravel()])

    for c in range(3):
        sampled = map_coordinates(
            image[..., c].astype(np.float32),
            coords,
            order=1,
            mode='nearest',
        ).reshape(H, W)
        # Visible points get direct true-ortho projection; occluded get smoothed fall-back
        ortho_rgb[..., c] = np.where(visible, np.clip(sampled, 0, 255), image[..., c]).astype(np.uint8)

    log.info(
        "Post-rectification Pass B complete",
        occluded_fraction=f"{float(occluded.mean()) * 100:.2f}%",
        visible_fraction=f"{float(visible.mean()) * 100:.2f}%",
    )
    return ortho_rgb, occluded
