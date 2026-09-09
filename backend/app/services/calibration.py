"""
Two-Component Elevation Decomposition.
DSM(x,y) = DTM_base(x,y) + α · nDSM_pred(x,y)

For MVP: uses statistical prior (no real SRTM download).
"""
import numpy as np
from dataclasses import dataclass
from app.logging_config import log


from scipy.ndimage import minimum_filter, gaussian_filter


from typing import Optional, Union
from app.services.dtm_fusion import fetch_and_resample_global_dtm, fuse_dtm
from app.services.shadow_scale import estimate_scale_from_shadows, compute_solar_position
from app.config import (
    DTM_FUSION_ENABLED,
    DTM_SOURCE,
    DTM_CACHE_DIR,
    DTM_CUTOFF_WAVELENGTH_M,
    SHADOW_SCALE_ENABLED,
)

@dataclass
class CalibrationResult:
    dsm: np.ndarray           # (H, W) float32, calibrated elevation
    alpha: float               # scale factor
    dsm_min: float
    dsm_max: float
    dsm_mean: float
    mode: str                  # "metric_dtm_fusion", "metric_prior", or "relative"
    unit: str                  # "meters" or "relative"
    dtm_base: Optional[np.ndarray] = None  # fused or bare-earth DTM
    fusion_applied: bool = False
    shadow_calibrated: bool = False
    uncertainty: Optional[np.ndarray] = None        # (H, W) propagated uncertainty
    uncertainty_mean: Optional[float] = None
    uncertainty_max: Optional[float] = None
    uncertainty_unit: Optional[str] = None          # "meters" or "relative"


def calibrate_depth(
    relative_depth: np.ndarray,
    is_georef: bool,
    gsd: float = None,
    target_range: float = 50.0,  # default assumed max building height in meters
    bounds: Optional[dict] = None,
    crs: Optional[str] = None,
    transform: Optional[Union[list, object]] = None,
    dtm_global: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    solar_elevation: Optional[float] = None,
    solar_azimuth: Optional[float] = None,
    capture_time: Optional[str] = None,
    relative_uncertainty: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """
    Convert relative depth [0,1] to calibrated DSM.

    For georeferenced: scale to approximate metric range using GSD and statistical prior,
    anchored to Copernicus GLO-30 DTM and solar cast shadows if available.
    Propagates evidential uncertainty units through alpha into meters (§1.4).
    For non-georeferenced: keep as relative [0,1].
    """
    if is_georef and gsd is not None:
        return _metric_calibration(
            relative_depth,
            gsd=gsd,
            target_range=target_range,
            bounds=bounds,
            crs=crs,
            transform=transform,
            dtm_global=dtm_global,
            rgb=rgb,
            solar_elevation=solar_elevation,
            solar_azimuth=solar_azimuth,
            capture_time=capture_time,
            relative_uncertainty=relative_uncertainty,
        )
    else:
        return _relative_calibration(relative_depth, relative_uncertainty=relative_uncertainty)


def _metric_calibration(
    depth: np.ndarray,
    gsd: float,
    target_range: float,
    bounds: Optional[dict] = None,
    crs: Optional[str] = None,
    transform: Optional[Union[list, object]] = None,
    dtm_global: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    solar_elevation: Optional[float] = None,
    solar_azimuth: Optional[float] = None,
    capture_time: Optional[str] = None,
    relative_uncertainty: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """
    Two-Component Elevation Decomposition:
    DSM(x, y) = DTM_base(x, y) + α · nDSM(x, y)

    - DTM_base is modeled using spatial morphological filtering and anchored via
      frequency-split blending with Copernicus GLO-30 global DTM (§2.1).
    - GSD scales the effective relief range according to sensor resolution and ground footprint.
    - α maps normalized structural variations to metric elevation (meters), calibrated
      via solar photometric shadow inversion (§1.1) when available.
    """
    safe_gsd = max(float(gsd), 0.05) if gsd is not None else 1.0

    # 1. Incorporate GSD into effective height range
    # VHR imagery (~0.3-0.5m) resolves single buildings (target_range ~ 30-50m).
    # Medium resolution (~5-10m) captures wider spatial extents with larger topographic relief.
    gsd_scale = float(np.clip(np.sqrt(safe_gsd / 0.5), 0.7, 3.0))
    effective_range = target_range * gsd_scale

    import time
    t_dtm_start = time.perf_counter()
    # 2. Bare-Earth DTM baseline modeling via morphological minimum + gaussian smoothing
    # Kernel size corresponds to typical maximum structural footprint (~30-50m) in pixels
    min_dim = min(depth.shape)
    if min_dim >= 12:
        max_kernel = max(3, min_dim // 6)
        kernel_px = int(np.clip(40.0 / safe_gsd, 3, max_kernel))
        if kernel_px % 2 == 0:
            kernel_px += 1
        dtm_raw = minimum_filter(depth, size=kernel_px)
        dtm_base = gaussian_filter(dtm_raw, sigma=max(kernel_px / 3.0, 1.0))
    else:
        # Fallback for very small patches
        dtm_base = np.full_like(depth, np.percentile(depth, 5))
    t_dtm_ms = (time.perf_counter() - t_dtm_start) * 1000

    # 3. Structural height above ground (nDSM)
    ndsm = np.maximum(depth - dtm_base, 0.0)

    # 4. Scale factor α computed from statistical prior
    ndsm_span = float(np.percentile(ndsm, 99) - np.percentile(ndsm, 5))
    if ndsm_span < 1e-6:
        depth_span = float(np.percentile(depth, 99) - np.percentile(depth, 1))
        alpha = (effective_range / depth_span) if depth_span > 1e-6 else 1.0
    else:
        alpha = effective_range / ndsm_span

    # 4b. Shadow-to-Scale Photometric Inversion (§1.1)
    shadow_calibrated = False
    t_shadow_ms = 0.0
    if SHADOW_SCALE_ENABLED and rgb is not None:
        t_shadow_start = time.perf_counter()
        sol_elev = solar_elevation
        sol_azim = solar_azimuth
        if sol_elev is None and capture_time and bounds and crs:
            try:
                center_x = (bounds["left"] + bounds["right"]) / 2.0
                center_y = (bounds["bottom"] + bounds["top"]) / 2.0
                if "4326" in crs.upper():
                    c_lat, c_lon = center_y, center_x
                else:
                    from rasterio.warp import transform_bounds
                    wgs = transform_bounds(crs, "EPSG:4326", bounds["left"], bounds["bottom"], bounds["right"], bounds["top"])
                    c_lon = (wgs[0] + wgs[2]) / 2.0
                    c_lat = (wgs[1] + wgs[3]) / 2.0
                sol_elev, sol_azim = compute_solar_position(capture_time, c_lat, c_lon)
            except Exception as e:
                log.warning("Could not compute solar position from timestamp", error=str(e))

        if sol_elev is not None and sol_azim is not None:
            shadow_info = estimate_scale_from_shadows(
                rgb=rgb,
                ndsm_relative=ndsm,
                gsd=safe_gsd,
                solar_elevation_deg=sol_elev,
                solar_azimuth_deg=sol_azim,
            )
            if shadow_info is not None:
                alpha = shadow_info["scale_factor"]
                shadow_calibrated = True
                log.info("Shadow-to-scale calibrated alpha applied", alpha=f"{alpha:.2f}")
        t_shadow_ms = (time.perf_counter() - t_shadow_start) * 1000

    log.info(
        "Metric Calibration Timing Breakdown",
        dtm_filter_ms=round(t_dtm_ms, 2),
        shadow_ms=round(t_shadow_ms, 2) if SHADOW_SCALE_ENABLED and rgb is not None else None,
        kernel_px=kernel_px if min_dim >= 12 else None,
        alpha=round(alpha, 2),
    )

    # 5. Composite DSM: bare-earth terrain relief + structural elevation
    dtm_metric = (dtm_base - float(dtm_base.min())) * (alpha * 0.4)
    ndsm_metric = alpha * ndsm

    # 6. Global DTM Low-Frequency Anchoring (§2.1)
    if dtm_global is None and DTM_FUSION_ENABLED and bounds and crs and transform:
        try:
            dtm_global = fetch_and_resample_global_dtm(
                bounds=bounds,
                crs=crs,
                target_shape=depth.shape,
                target_transform=transform,
                cache_dir=DTM_CACHE_DIR,
                source=DTM_SOURCE,
            )
        except Exception as e:
            log.warning("Global DTM fetch failed, falling back to morphological prior", error=str(e))
            dtm_global = None

    if dtm_global is not None:
        dtm_fused = fuse_dtm(
            dtm_local_base=dtm_metric,
            dtm_global_resampled=dtm_global,
            gsd=safe_gsd,
            cutoff_wavelength_m=DTM_CUTOFF_WAVELENGTH_M,
        )
        dsm = dtm_fused + ndsm_metric
        mode = "metric_dtm_fusion_shadow" if shadow_calibrated else "metric_dtm_fusion"
        fusion_applied = True
        log.info(
            "Metric calibration (Global DTM anchored)",
            alpha=f"{alpha:.2f}",
            gsd=f"{safe_gsd:.2f}m",
            dtm_mean=f"{float(dtm_fused.mean()):.1f}m",
            dsm_range=f"[{dsm.min():.1f}, {dsm.max():.1f}]m",
        )
    else:
        dtm_fused = dtm_metric
        dsm = dtm_metric + ndsm_metric
        mode = "metric_shadow_scale" if shadow_calibrated else "metric_prior"
        fusion_applied = False
        log.info(
            "Metric calibration",
            alpha=f"{alpha:.2f}",
            gsd=f"{safe_gsd:.2f}m",
            dsm_range=f"[{dsm.min():.1f}, {dsm.max():.1f}]m",
        )

    unc_map = None
    unc_mean = None
    unc_max = None
    unc_unit = None
    if relative_uncertainty is not None:
        unc_map = (relative_uncertainty * float(alpha)).astype(np.float32)
        unc_mean = float(np.nanmean(unc_map))
        unc_max = float(np.nanmax(unc_map))
        unc_unit = "meters"

    return CalibrationResult(
        dsm=dsm.astype(np.float32),
        alpha=float(alpha),
        dsm_min=float(dsm.min()),
        dsm_max=float(dsm.max()),
        dsm_mean=float(dsm.mean()),
        mode=mode,
        unit="meters",
        dtm_base=dtm_fused.astype(np.float32),
        fusion_applied=fusion_applied,
        shadow_calibrated=shadow_calibrated,
        uncertainty=unc_map,
        uncertainty_mean=unc_mean,
        uncertainty_max=unc_max,
        uncertainty_unit=unc_unit,
    )


def _relative_calibration(
    depth: np.ndarray,
    relative_uncertainty: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """Keep as relative depth normalized to [0, 1]."""
    log.info("Relative calibration (non-georeferenced)")
    unc_map = None
    unc_mean = None
    unc_max = None
    unc_unit = None
    if relative_uncertainty is not None:
        unc_map = relative_uncertainty.astype(np.float32)
        unc_mean = float(np.nanmean(unc_map))
        unc_max = float(np.nanmax(unc_map))
        unc_unit = "relative"

    return CalibrationResult(
        dsm=depth.astype(np.float32),
        alpha=1.0,
        dsm_min=float(depth.min()),
        dsm_max=float(depth.max()),
        dsm_mean=float(depth.mean()),
        mode="relative",
        unit="relative",
        uncertainty=unc_map,
        uncertainty_mean=unc_mean,
        uncertainty_max=unc_max,
        uncertainty_unit=unc_unit,
    )
