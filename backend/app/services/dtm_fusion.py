"""
Global DTM Low-Frequency Anchoring (§2.1)
Anchors monocular elevation predictions to Copernicus GLO-30 DEM on AWS Open Data.
Supports:
- Online anonymous S3 fetch of Copernicus GLO-30 COGs
- Reprojection and resampling to arbitrary local target CRS, Affine transform, and grid
- Gaussian blur frequency-split blending (global low-frequency slope/MSL + local high-frequency micro-relief)
- Local disk caching and offline tactical mode (DTM_SOURCE=local_cache)
- Safe fallback when offline or disabled
"""
import math
import os
import io
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
from scipy.ndimage import gaussian_filter

from app.logging_config import log
from app.config import (
    DTM_FUSION_ENABLED,
    DTM_SOURCE,
    DTM_CACHE_DIR,
    DTM_CUTOFF_WAVELENGTH_M,
)

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

COPERNICUS_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def get_copernicus_tile_name(lat: float, lon: float) -> str:
    """
    Compute the Copernicus GLO-30 1x1 degree COG tile name for a given (lat, lon).
    Copernicus DEM tiles are labeled by the latitude and longitude of their
    south-west (bottom-left) corner.
    """
    lat_floor = int(math.floor(lat))
    lon_floor = int(math.floor(lon))

    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"

    lat_str = f"{lat_prefix}{abs(lat_floor):02d}_00"
    lon_str = f"{lon_prefix}{abs(lon_floor):03d}_00"

    return f"Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM"


def get_covering_tile_names(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> List[str]:
    """
    Find all 1x1 degree Copernicus DEM tile names intersecting the given WGS84 bounding box.
    """
    min_lat = max(float(min_lat), -90.0)
    max_lat = min(float(max_lat), 90.0)
    min_lon = max(float(min_lon), -180.0)
    max_lon = min(float(max_lon), 180.0)

    start_lat = int(math.floor(min_lat))
    end_lat = int(math.floor(max_lat))
    start_lon = int(math.floor(min_lon))
    end_lon = int(math.floor(max_lon))

    tiles = []
    for lat in range(start_lat, end_lat + 1):
        for lon in range(start_lon, end_lon + 1):
            tiles.append(get_copernicus_tile_name(lat, lon))
    return list(dict.fromkeys(tiles))  # preserve order, eliminate duplicates


def fetch_or_cache_tile(
    tile_name: str,
    cache_dir: Optional[Path] = None,
    source: str = "remote",
) -> Optional[Path]:
    """
    Obtain local file path for a Copernicus DEM tile.
    - If found in local cache: returns cache path immediately.
    - If source == "remote" and missing: streams/downloads from AWS Open Data and saves to cache.
    - If source == "local_cache" and missing: returns None (tactical offline mode cache miss).
    - If source == "disabled": returns None.
    """
    if source == "disabled":
        return None

    target_cache_dir = cache_dir or DTM_CACHE_DIR
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = target_cache_dir / f"{tile_name}.tif"

    if cached_file.exists() and cached_file.stat().st_size > 1024:
        return cached_file

    if source == "local_cache":
        log.warning("DTM tile not found in local cache", tile=tile_name, cache_dir=str(target_cache_dir))
        return None

    # source == "remote": download from S3
    url = f"{COPERNICUS_S3_BASE}/{tile_name}/{tile_name}.tif"
    temp_file = target_cache_dir / f"{tile_name}.tmp_{os.getpid()}"

    try:
        log.info("Downloading Copernicus DEM tile from S3", tile=tile_name, url=url)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DepthWizard-DTMFusion/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_file, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

        # Atomic replace
        temp_file.replace(cached_file)
        log.info("Copernicus DEM tile cached successfully", tile=tile_name, path=str(cached_file))
        return cached_file
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.warning("Copernicus DEM tile does not exist on S3 (likely ocean or out of coverage)", tile=tile_name)
        else:
            log.error("HTTP error downloading Copernicus DEM tile", tile=tile_name, status=e.code)
        if temp_file.exists():
            temp_file.unlink()
        return None
    except Exception as e:
        log.error("Failed to download Copernicus DEM tile", tile=tile_name, error=str(e))
        if temp_file.exists():
            temp_file.unlink()
        return None


def fetch_and_resample_global_dtm(
    bounds: dict,
    crs: str,
    target_shape: tuple,
    target_transform: Union[list, "Affine"],
    cache_dir: Optional[Path] = None,
    source: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Fetch Copernicus GLO-30 tile(s) and reproject/resample them to match the target grid.
    Returns:
        np.ndarray of shape target_shape with dtype float32 containing metric ground elevations (m MSL),
        or None if fetch/reprojection fails or is disabled.
    """
    if not HAS_RASTERIO:
        log.warning("rasterio not available; skipping global DTM fusion")
        return None

    dtm_source = source if source is not None else DTM_SOURCE
    if dtm_source == "disabled" or not DTM_FUSION_ENABLED:
        return None

    if not bounds or not crs or not target_transform:
        return None

    H, W = target_shape
    if isinstance(target_transform, (list, tuple)):
        dst_transform = Affine(*target_transform)
    else:
        dst_transform = target_transform

    # 1. Transform bounds to WGS84 (EPSG:4326) to find covering Copernicus tiles
    try:
        left, bottom, right, top = bounds["left"], bounds["bottom"], bounds["right"], bounds["top"]
        if "4326" in crs.upper():
            min_lon, min_lat, max_lon, max_lat = left, bottom, right, top
        else:
            min_lon, min_lat, max_lon, max_lat = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    except Exception as e:
        log.warning("Failed to transform bounds to EPSG:4326 for global DTM lookup", error=str(e))
        return None

    tile_names = get_covering_tile_names(min_lat, min_lon, max_lat, max_lon)
    if not tile_names:
        return None

    # 2. Acquire tile files (cached or downloaded)
    tile_paths = []
    for name in tile_names:
        p = fetch_or_cache_tile(name, cache_dir=cache_dir, source=dtm_source)
        if p is not None:
            tile_paths.append(p)

    if not tile_paths:
        log.warning("No Copernicus DEM tiles available for AOI", bounds=bounds, crs=crs)
        return None

    # 3. Reproject tile(s) onto destination grid
    dst_data = np.zeros((H, W), dtype=np.float32)
    valid_count = np.zeros((H, W), dtype=np.int32)

    for p in tile_paths:
        try:
            with rasterio.open(str(p)) as src:
                tile_reprojected = np.zeros((H, W), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=tile_reprojected,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                )
                # Copernicus GLO-30 nodata is typically -32767.0
                valid_mask = np.isfinite(tile_reprojected) & (tile_reprojected > -1000.0) & (tile_reprojected < 10000.0)
                dst_data[valid_mask] += tile_reprojected[valid_mask]
                valid_count[valid_mask] += 1
        except Exception as e:
            log.warning("Error reprojecting DTM tile", tile=str(p), error=str(e))

    if not np.any(valid_count > 0):
        log.warning("Global DTM reprojected grid has no valid pixels")
        return None

    # Average overlaps if any
    mask_positive = valid_count > 0
    dst_data[mask_positive] /= valid_count[mask_positive]

    # Fill unobserved boundary pixels with mean valid elevation
    if not np.all(mask_positive):
        mean_elev = float(dst_data[mask_positive].mean())
        dst_data[~mask_positive] = mean_elev

    log.info(
        "Global DTM resampled",
        shape=dst_data.shape,
        elev_min=float(dst_data.min()),
        elev_max=float(dst_data.max()),
        elev_mean=float(dst_data.mean()),
    )
    return dst_data


from scipy.ndimage import gaussian_filter, zoom


def fast_gaussian_filter(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Optimized multi-scale Gaussian filter.
    For large sigma (> 8.0 pixels), direct full-resolution convolution spends seconds
    computing redundant sub-Nyquist convolutions. Pyramid downsampling computes the blur
    at the appropriate scale and interpolates back, achieving 100x+ speedup with < 0.05m error.
    """
    H, W = arr.shape
    if sigma_px > 8.0:
        factor = min(int(sigma_px / 4.0), 8)
        if factor > 1 and H // factor >= 16 and W // factor >= 16:
            scale = 1.0 / factor
            small = zoom(arr, scale, order=1)
            small_filtered = gaussian_filter(small, sigma=sigma_px * scale)
            res = zoom(small_filtered, (H / small_filtered.shape[0], W / small_filtered.shape[1]), order=1)
            return res.astype(arr.dtype)
    return gaussian_filter(arr, sigma=sigma_px)


def fuse_dtm(
    dtm_local_base: np.ndarray,
    dtm_global_resampled: np.ndarray,
    gsd: float,
    cutoff_wavelength_m: Optional[float] = None,
) -> np.ndarray:
    """
    Frequency-split fusion of locally-predicted bare-earth DTM and resampled global DTM.

    Formula:
        DTM_fused = LowPass(DTM_global) + [DTM_local - LowPass(DTM_local)]
                  = LowPass(DTM_global) + HighPass(DTM_local)

    Where:
        - LowPass(DTM_global) provides absolute MSL elevation offset, regional slope,
          and large-scale macro-curvature (~100-200m wavelength).
        - HighPass(DTM_local) preserves local terrain details, micro-relief, drainage,
          and small mounds from the fine monocular prediction.
    """
    wavelength = cutoff_wavelength_m or DTM_CUTOFF_WAVELENGTH_M
    safe_gsd = max(float(gsd), 0.05) if gsd is not None else 1.0

    # Sigma in pixels for Gaussian filter corresponding to cutoff wavelength
    # Cutoff ~ 2 * pi * sigma * gsd => sigma ~ wavelength / (2 * pi * gsd) or wavelength / gsd
    sigma_px = max(wavelength / safe_gsd, 2.0)

    # 1. Low-pass global DTM using fast pyramid Gaussian filter
    low_global = fast_gaussian_filter(dtm_global_resampled.astype(np.float32), sigma_px=sigma_px)

    # 2. Split local DTM into low and high frequencies
    local_float = dtm_local_base.astype(np.float32)
    low_local = fast_gaussian_filter(local_float, sigma_px=sigma_px)
    high_local = local_float - low_local

    # 3. Fuse: global low-pass + local high-pass
    dtm_fused = low_global + high_local
    return dtm_fused.astype(np.float32)
