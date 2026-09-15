"""
Shadow-to-Scale Photometric Inversion (§1.1).
Recovers physical metric scale (alpha) from cast shadows and solar ephemeris:
H = L_meters * tan(theta_sun) = L_px * GSD * tan(theta_sun)
k = H_true / H_relative
Consistently anchors building relief without ground truth LiDAR.
"""
from typing import Optional, Tuple, List, Union
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, binary_closing, label

from app.logging_config import log
from app.config import SHADOW_SCALE_ENABLED, SHADOW_MIN_STRUCTURES

try:
    import pvlib
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False


def compute_solar_position(
    timestamp: Union[str, datetime],
    latitude: float,
    longitude: float,
) -> Tuple[float, float]:
    """
    Compute solar elevation and azimuth using pvlib deterministic ephemeris.
    Returns:
        (solar_elevation_deg, solar_azimuth_deg)
    """
    if not HAS_PVLIB:
        raise RuntimeError("pvlib is required for solar ephemeris calculation")

    ts = pd.Timestamp(timestamp)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")

    sol = pvlib.solarposition.get_solarposition(ts, latitude=latitude, longitude=longitude)
    elev = float(sol["apparent_elevation"].iloc[0])
    azim = float(sol["azimuth"].iloc[0])
    return elev, azim


def segment_shadows(rgb: np.ndarray, min_blob_size: int = 16) -> np.ndarray:
    """
    Segment cast shadows in RGB image using colorimetry and luminance thresholding.
    Shadows exhibit lower luminance and higher blue-to-red ratios due to diffuse skylight illumination.
    Returns:
        boolean mask of shape (H, W) where True indicates cast shadow.
    """
    H, W = rgb.shape[:2]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Robust local thresholding against dynamic range
    p15 = np.percentile(lum, 15)
    p50 = np.percentile(lum, 50)
    lum_thresh = min(p15 * 1.35, p50 * 0.70)

    # Skylight Rayleigh scattering blue-shift
    r = rgb[..., 0].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    blue_ratio = b / (r + 10.0)

    candidate = (lum < lum_thresh) & (blue_ratio > 0.70)

    # Morphological cleanup
    struct = np.ones((3, 3), dtype=bool)
    cleaned = binary_opening(binary_closing(candidate, structure=struct), structure=struct)

    # Filter small noise blobs
    lbl, num_features = label(cleaned)
    if num_features == 0:
        return np.zeros((H, W), dtype=bool)

    counts = np.bincount(lbl.ravel())
    large_enough = counts >= min_blob_size
    large_enough[0] = False  # background

    return large_enough[lbl]


def estimate_scale_from_shadows(
    rgb: np.ndarray,
    ndsm_relative: np.ndarray,
    gsd: float,
    solar_elevation_deg: float,
    solar_azimuth_deg: float,
    min_structures: Optional[int] = None,
) -> Optional[dict]:
    """
    Estimate metric height scale factor k = H_true / H_relative across cast shadows.

    Parameters:
        rgb: (H, W, 3) uint8 image.
        ndsm_relative: (H, W) float32 relative structural heights (depth - dtm_base).
        gsd: Ground Sampling Distance (meters/pixel).
        solar_elevation_deg: Solar elevation angle above horizon (degrees).
        solar_azimuth_deg: Solar azimuth angle from North clockwise (degrees).
        min_structures: Minimum number of valid structures needed to accept estimate.

    Returns:
        dict with keys {"scale_factor": float, "valid_structures": int, "solar_elevation": float}
        or None if estimate could not be reliably determined.
    """
    if not SHADOW_SCALE_ENABLED:
        return None

    # Sun position bounds check: near-noon (>80 deg) has short shadows; low-sun (<5 deg) stretches into infinity
    if solar_elevation_deg < 5.0 or solar_elevation_deg > 80.0:
        log.warning(
            "Solar elevation out of reliable range for shadow-to-scale inversion",
            elevation=solar_elevation_deg,
        )
        return None

    safe_gsd = max(float(gsd), 0.05) if gsd is not None else 0.5
    req_structures = min_structures or SHADOW_MIN_STRUCTURES
    H, W = rgb.shape[:2]

    # Sun direction vector in image plane:
    # Azimuth 0 = North (-Y), 90 = East (+X), 180 = South (+Y), 270 = West (-X)
    phi_rad = np.radians(solar_azimuth_deg)
    d_sun = np.array([np.sin(phi_rad), -np.cos(phi_rad)], dtype=np.float64)
    theta_rad = np.radians(solar_elevation_deg)
    tan_theta = np.tan(theta_rad)

    shadow_mask = segment_shadows(rgb)
    lbl, num_features = label(shadow_mask)

    if num_features == 0:
        log.info("No cast shadows detected for photometric inversion")
        return None

    scale_candidates: List[float] = []

    for blob_id in range(1, num_features + 1):
        y_coords, x_coords = np.where(lbl == blob_id)
        if len(y_coords) < 16:
            continue

        pts = np.column_stack([x_coords, y_coords])
        proj = pts @ d_sun
        L_px = float(proj.max() - proj.min())

        # Discard too-short shadows (sub-pixel noise)
        if L_px < 4.0:
            continue

        # Physical true height from shadow length:
        H_true = L_px * safe_gsd * tan_theta

        # Base point of the shadow closest to the casting structure
        base_pt = pts[np.argmax(proj)]
        # Sample structure relief just in the direction of the sun
        struct_x = int(np.clip(round(base_pt[0] + 3.0 * d_sun[0]), 0, W - 1))
        struct_y = int(np.clip(round(base_pt[1] + 3.0 * d_sun[1]), 0, H - 1))

        h_rel = float(ndsm_relative[struct_y, struct_x])
        if h_rel < 0.05:
            # Check a 3x3 window around struct point for maximum structure relief
            y_min = max(0, struct_y - 2)
            y_max = min(H, struct_y + 3)
            x_min = max(0, struct_x - 2)
            x_max = min(W, struct_x + 3)
            h_rel = float(ndsm_relative[y_min:y_max, x_min:x_max].max())

        if h_rel >= 0.05:
            k = H_true / h_rel
            # Sanity bound check: typical building relief scales [5m to 250m]
            if 5.0 <= k <= 300.0:
                scale_candidates.append(k)

    if len(scale_candidates) < req_structures:
        log.info(
            "Insufficient valid shadow-structure pairs for robust scale recovery",
            candidates_found=len(scale_candidates),
            required=req_structures,
        )
        return None

    recovered_alpha = float(np.median(scale_candidates))
    log.info(
        "Shadow-to-scale inversion successful",
        recovered_alpha=f"{recovered_alpha:.2f}",
        structures_paired=len(scale_candidates),
        solar_elevation=f"{solar_elevation_deg:.1f}deg",
    )

    return {
        "scale_factor": recovered_alpha,
        "valid_structures": len(scale_candidates),
        "solar_elevation": solar_elevation_deg,
        "solar_azimuth": solar_azimuth_deg,
    }
