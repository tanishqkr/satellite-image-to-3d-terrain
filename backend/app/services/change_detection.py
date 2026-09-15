"""
Automated 3D Battle Damage Assessment (BDA) & Volumetric Differencing Service (§4.2).

Provides:
1. Sub-pixel FFT Phase Correlation Co-Registration between Pre- and Post-strike DSMs.
2. Cut/Fill Volumetric Differencing with Noise Floor Thresholding.
3. Evidential Uncertainty Propagation (§1.4) into Volumetric Bounds (± m³).
4. Automated Structural Footprint Isolation & Percentage Collapse Estimation.
5. Military BDA Damage Severity Classification (Light, Moderate, Severe, Destroyed).
6. Tactical Diff Heatmap Colorization (Red = Crater/Cut, Cyan = Rubble/Fill, Dark = Neutral).
"""
import io
import base64
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
from scipy import ndimage
from PIL import Image

from app.config import (
    BDA_ENABLED,
    BDA_NOISE_THRESHOLD_M,
    BDA_BETA_MODE,
    BDA_MIN_BUILDING_HEIGHT_M,
)
from app.logging_config import log


@dataclass
class StructureBda:
    structure_id: int
    center_col: float
    center_row: float
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    footprint_area_m2: float
    pre_strike_volume_m3: float
    volume_loss_m3: float
    collapse_percentage: float
    damage_rating: str  # 'LIGHT', 'MODERATE', 'SEVERE', 'DESTROYED'


@dataclass
class VolumetricDiffResult:
    cut_volume_m3: float
    fill_volume_m3: float
    net_volume_m3: float
    cut_area_m2: float
    fill_area_m2: float
    max_cut_depth_m: float
    max_fill_height_m: float
    noise_threshold_m: float
    registration_shift_x_m: float
    registration_shift_y_m: float
    registration_confidence: float
    diff_map: np.ndarray
    structures: List[StructureBda] = field(default_factory=list)
    uncertainty_volume_m3: Optional[float] = None
    diff_map_b64: Optional[str] = None
    beta_mode: bool = True


# =============================================================================
# 1. SUB-PIXEL PHASE CORRELATION CO-REGISTRATION
# =============================================================================

def co_register_dsms(
    dsm_pre: np.ndarray,
    dsm_post: np.ndarray,
    gsd: float = 1.0,
    max_shift_px: int = 15,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Sub-pixel FFT Phase Correlation co-registration between Pre- and Post-strike DSMs.
    Detects and corrects residual lateral translational drift before differencing.

    Parameters:
        dsm_pre: Reference baseline DSM (H, W).
        dsm_post: Target post-strike DSM (H, W).
        gsd: Ground sample distance in meters.
        max_shift_px: Maximum allowable shift to prevent gross misalignment.

    Returns:
        Tuple of:
            aligned_post: np.ndarray (H, W) shifted to match dsm_pre.
            shift_x_m: Lateral shift along X (columns) in meters.
            shift_y_m: Lateral shift along Y (rows) in meters.
            confidence: Peak correlation value in [0, 1].
    """
    H, W = dsm_pre.shape
    if dsm_post.shape != (H, W):
        raise ValueError(f"Shape mismatch: pre {dsm_pre.shape} vs post {dsm_post.shape}")

    # 1. Normalize images and remove DC bias
    pre_clean = np.nan_to_num(dsm_pre, nan=float(np.nanmean(dsm_pre)))
    post_clean = np.nan_to_num(dsm_post, nan=float(np.nanmean(dsm_post)))

    # Apply 2D Hanning window to reduce spectral leakage at borders
    win_r = np.hanning(H)
    win_c = np.hanning(W)
    window = np.outer(win_r, win_c)

    pre_win = (pre_clean - np.mean(pre_clean)) * window
    post_win = (post_clean - np.mean(post_clean)) * window

    # 2. FFT Cross-Power Spectrum
    F_pre = np.fft.fft2(pre_win)
    F_post = np.fft.fft2(post_win)

    eps = 1e-8
    cross_power = (F_pre * np.conj(F_post)) / (np.abs(F_pre * np.conj(F_post)) + eps)
    correlation = np.fft.ifftshift(np.real(np.fft.ifft2(cross_power)))

    # 3. Peak detection
    peak_r, peak_c = np.unravel_index(np.argmax(correlation), correlation.shape)
    mid_r = H // 2
    mid_c = W // 2

    raw_shift_r = peak_r - mid_r
    raw_shift_c = peak_c - mid_c

    # Sub-pixel quadratic refinement around peak
    sub_r = float(raw_shift_r)
    sub_c = float(raw_shift_c)
    if 1 <= peak_r < H - 1 and 1 <= peak_c < W - 1:
        # Parabolic peak interpolation
        dx = correlation[peak_r, peak_c + 1] - correlation[peak_r, peak_c - 1]
        denom_x = 2.0 * (2.0 * correlation[peak_r, peak_c] - correlation[peak_r, peak_c + 1] - correlation[peak_r, peak_c - 1])
        if abs(denom_x) > 1e-6:
            sub_c += dx / denom_x

        dy = correlation[peak_r + 1, peak_c] - correlation[peak_r - 1, peak_c]
        denom_y = 2.0 * (2.0 * correlation[peak_r, peak_c] - correlation[peak_r + 1, peak_c] - correlation[peak_r - 1, peak_c])
        if abs(denom_y) > 1e-6:
            sub_r += dy / denom_y

    confidence = float(np.clip(correlation[peak_r, peak_c], 0.0, 1.0))

    # Bound shift to max_shift_px
    if abs(sub_r) > max_shift_px or abs(sub_c) > max_shift_px or confidence < 0.05:
        log.warning(
            "Co-registration shift exceeded bounds or low confidence; using identity alignment",
            shift_r=round(sub_r, 2),
            shift_c=round(sub_c, 2),
            confidence=round(confidence, 3),
        )
        return dsm_post.copy(), 0.0, 0.0, confidence

    # 4. Resample post-strike DSM by detected offset
    # Note: to align post with pre, we shift post by (+sub_r, +sub_c)
    aligned_post = ndimage.shift(
        post_clean,
        shift=(sub_r, sub_c),
        order=1,
        mode="nearest",
    ).astype(np.float32)

    shift_x_m = float(sub_c * gsd)
    shift_y_m = float(sub_r * gsd)

    log.info(
        "Co-registration complete",
        shift_x_m=round(shift_x_m, 2),
        shift_y_m=round(shift_y_m, 2),
        confidence=round(confidence, 3),
    )
    return aligned_post, shift_x_m, shift_y_m, confidence


# =============================================================================
# 2. VOLUMETRIC DIFFERENCING & CUT/FILL CALCULATION
# =============================================================================

def compute_volumetric_diff(
    dsm_pre: np.ndarray,
    dsm_post: np.ndarray,
    gsd: float = 1.0,
    threshold_m: float = BDA_NOISE_THRESHOLD_M,
    unc_pre: Optional[np.ndarray] = None,
    unc_post: Optional[np.ndarray] = None,
    auto_coregister: bool = True,
) -> VolumetricDiffResult:
    """
    Compute cut and fill volumetric change between Pre- and Post-strike DSMs.

    Parameters:
        dsm_pre: Pre-event DSM in meters.
        dsm_post: Post-event DSM in meters.
        gsd: Ground sample distance in meters.
        threshold_m: Elevation noise floor below which changes are considered noise (meters).
        unc_pre: Optional evidential uncertainty map for pre DSM (§1.4).
        unc_post: Optional evidential uncertainty map for post DSM (§1.4).
        auto_coregister: If True, executes sub-pixel FFT phase correlation alignment first.

    Returns:
        VolumetricDiffResult with comprehensive cut/fill metrics and structural BDA.
    """
    if auto_coregister:
        aligned_post, shift_x, shift_y, conf = co_register_dsms(dsm_pre, dsm_post, gsd=gsd)
    else:
        aligned_post = dsm_post
        shift_x, shift_y, conf = 0.0, 0.0, 1.0

    cell_area = float(gsd * gsd)
    diff = aligned_post - dsm_pre

    # 1. Cut (Material removed / craters / structural collapse)
    cut_mask = diff < -threshold_m
    cut_cells = int(np.sum(cut_mask))
    cut_volume = float(np.sum(np.abs(diff[cut_mask])) * cell_area) if cut_cells > 0 else 0.0
    cut_area = float(cut_cells * cell_area)
    max_cut_depth = float(np.max(-diff[cut_mask])) if cut_cells > 0 else 0.0

    # 2. Fill (Material added / rubble mounds / debris berms)
    fill_mask = diff > threshold_m
    fill_cells = int(np.sum(fill_mask))
    fill_volume = float(np.sum(diff[fill_mask]) * cell_area) if fill_cells > 0 else 0.0
    fill_area = float(fill_cells * cell_area)
    max_fill_height = float(np.max(diff[fill_mask])) if fill_cells > 0 else 0.0

    net_volume = fill_volume - cut_volume

    # 3. Evidential Uncertainty Propagation (§1.4 synergy)
    unc_vol = None
    if unc_pre is not None and unc_post is not None:
        try:
            sigma_diff = np.sqrt(np.square(unc_pre) + np.square(unc_post))
            active_mask = cut_mask | fill_mask
            if np.any(active_mask):
                unc_vol = float(np.sum(sigma_diff[active_mask]) * cell_area)
        except Exception as e:
            log.warning("Failed to propagate evidential uncertainty into BDA", error=str(e))

    # 4. Structural BDA Analysis
    structures = assess_structural_bda(
        dsm_pre=dsm_pre,
        dsm_post=aligned_post,
        gsd=gsd,
        threshold_m=threshold_m,
    )

    # 5. Diff Map Colorization
    rgb_diff = colorize_diff_map(diff, max_delta=max(max_cut_depth, max_fill_height, 2.0))
    diff_b64 = diff_map_to_base64(rgb_diff)

    return VolumetricDiffResult(
        cut_volume_m3=round(cut_volume, 2),
        fill_volume_m3=round(fill_volume, 2),
        net_volume_m3=round(net_volume, 2),
        cut_area_m2=round(cut_area, 2),
        fill_area_m2=round(fill_area, 2),
        max_cut_depth_m=round(max_cut_depth, 2),
        max_fill_height_m=round(max_fill_height, 2),
        noise_threshold_m=threshold_m,
        registration_shift_x_m=round(shift_x, 2),
        registration_shift_y_m=round(shift_y, 2),
        registration_confidence=round(conf, 3),
        diff_map=diff,
        structures=structures,
        uncertainty_volume_m3=round(unc_vol, 2) if unc_vol is not None else None,
        diff_map_b64=diff_b64,
        beta_mode=BDA_BETA_MODE,
    )


# =============================================================================
# 3. STRUCTURAL FOOTPRINT BDA & PERCENTAGE COLLAPSE
# =============================================================================

def assess_structural_bda(
    dsm_pre: np.ndarray,
    dsm_post: np.ndarray,
    gsd: float = 1.0,
    building_threshold_m: float = BDA_MIN_BUILDING_HEIGHT_M,
    threshold_m: float = BDA_NOISE_THRESHOLD_M,
    min_footprint_px: int = 15,
) -> List[StructureBda]:
    """
    Isolate structures in pre-strike imagery and calculate percentage collapse per structure.

    Parameters:
        dsm_pre: Baseline DSM.
        dsm_post: Post-strike DSM (already co-registered).
        gsd: Ground sample distance in meters.
        building_threshold_m: Elevation above baseline DTM to qualify as building (meters).
        threshold_m: Noise floor for volume loss (meters).
        min_footprint_px: Minimum pixel count to filter tree/noise artifacts.

    Returns:
        List of StructureBda objects.
    """
    H, W = dsm_pre.shape
    cell_area = float(gsd * gsd)

    # 1. Bare-Earth DTM baseline approximation via morphological minimum
    kernel_px = int(np.clip(30.0 / max(gsd, 0.1), 5, 25))
    if kernel_px % 2 == 0:
        kernel_px += 1
    dtm_pre = ndimage.minimum_filter(dsm_pre, size=kernel_px)
    dtm_pre = ndimage.gaussian_filter(dtm_pre, sigma=max(kernel_px / 3.0, 1.0))

    # 2. Structural height above ground (nDSM)
    ndsm_pre = np.maximum(dsm_pre - dtm_pre, 0.0)
    building_mask = ndsm_pre >= building_threshold_m

    # Morphological closing to fill small internal roof holes
    struct_mask = ndimage.binary_closing(building_mask, structure=np.ones((3, 3)))

    # 3. Connected component labeling
    labeled, num_features = ndimage.label(struct_mask)
    if num_features == 0:
        return []

    diff = dsm_post - dsm_pre
    structures_bda: List[StructureBda] = []

    for struct_id in range(1, num_features + 1):
        mask_k = labeled == struct_id
        pixel_count = int(np.sum(mask_k))
        if pixel_count < min_footprint_px:
            continue

        footprint_m2 = float(pixel_count * cell_area)
        pre_volume = float(np.sum(ndsm_pre[mask_k]) * cell_area)

        # Volume loss specifically within this building's footprint
        loss_mask = mask_k & (diff < -threshold_m)
        loss_volume = float(np.sum(np.abs(diff[loss_mask])) * cell_area) if np.any(loss_mask) else 0.0

        collapse_pct = min(100.0, (loss_volume / max(pre_volume, 1e-4)) * 100.0)

        # Standard Military GEOINT BDA Rating
        if collapse_pct < 10.0:
            rating = "LIGHT"
        elif collapse_pct < 40.0:
            rating = "MODERATE"
        elif collapse_pct < 75.0:
            rating = "SEVERE"
        else:
            rating = "DESTROYED"

        # Coordinates
        rows_k, cols_k = np.where(mask_k)
        center_r = float(np.mean(rows_k))
        center_c = float(np.mean(cols_k))
        bbox = (int(np.min(rows_k)), int(np.min(cols_k)), int(np.max(rows_k)), int(np.max(cols_k)))

        structures_bda.append(
            StructureBda(
                structure_id=struct_id,
                center_col=round(center_c, 1),
                center_row=round(center_r, 1),
                bbox=bbox,
                footprint_area_m2=round(footprint_m2, 1),
                pre_strike_volume_m3=round(pre_volume, 1),
                volume_loss_m3=round(loss_volume, 1),
                collapse_percentage=round(collapse_pct, 1),
                damage_rating=rating,
            )
        )

    # Sort structures by volume loss descending (highest damage first)
    structures_bda.sort(key=lambda s: s.volume_loss_m3, reverse=True)
    return structures_bda


# =============================================================================
# 4. COLORIZED DIFF HEATMAP GENERATOR
# =============================================================================

def colorize_diff_map(
    diff: np.ndarray,
    max_delta: Optional[float] = None,
    threshold: float = BDA_NOISE_THRESHOLD_M,
) -> np.ndarray:
    """
    Colorize volumetric difference map using tactical GEOINT convention:
    - Red (#EF4444): Crater / Cut / Structural Collapse (Material Removed)
    - Cyan (#06B6D4): Rubble Mound / Debris / Fill (Material Added)
    - Dark Slate (#0F172A / Translucent): Neutral / Stable / Unchanged (|Delta| <= threshold)

    Returns:
        RGB uint8 array (H, W, 3).
    """
    H, W = diff.shape
    d = np.asarray(diff, dtype=np.float32)

    if max_delta is None or max_delta <= threshold:
        abs_valid = np.abs(d[np.isfinite(d)])
        max_delta = float(np.percentile(abs_valid, 98.0)) if len(abs_valid) > 0 else 5.0
        max_delta = max(max_delta, threshold * 2.0)

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    # Background neutral dark slate
    rgb[:] = [15, 23, 42]

    # 1. Negative Change (Cut / Crater / Collapse) -> Red
    cut_mask = d < -threshold
    if np.any(cut_mask):
        # Normalized magnitude in [0, 1]
        t_cut = np.clip((-d[cut_mask] - threshold) / (max_delta - threshold), 0.0, 1.0)[:, None]
        # Interpolate Dark Red (120, 20, 20) -> Bright Red (239, 68, 68)
        c_dark_red = np.array([120, 20, 20], dtype=np.float32)
        c_bright_red = np.array([239, 68, 68], dtype=np.float32)
        rgb[cut_mask] = np.clip(c_dark_red * (1.0 - t_cut) + c_bright_red * t_cut, 0, 255).astype(np.uint8)

    # 2. Positive Change (Fill / Rubble / Debris) -> Cyan
    fill_mask = d > threshold
    if np.any(fill_mask):
        t_fill = np.clip((d[fill_mask] - threshold) / (max_delta - threshold), 0.0, 1.0)[:, None]
        # Interpolate Dark Teal (10, 80, 100) -> Bright Cyan (6, 182, 212)
        c_dark_cyan = np.array([10, 80, 100], dtype=np.float32)
        c_bright_cyan = np.array([6, 182, 212], dtype=np.float32)
        rgb[fill_mask] = np.clip(c_dark_cyan * (1.0 - t_fill) + c_bright_cyan * t_fill, 0, 255).astype(np.uint8)

    return rgb


def diff_map_to_base64(rgb_array: np.ndarray) -> str:
    """Convert RGB array to base64 PNG string."""
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
