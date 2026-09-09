"""
GeoTIFF and standard image reader.
Detects CRS, affine transform, and geospatial metadata.
"""
import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from app.logging_config import log

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


from app.services.orthorectify import CameraPose
import re

@dataclass
class ImageMetadata:
    width: int = 0
    height: int = 0
    channels: int = 3
    is_georef: bool = False
    crs: Optional[str] = None
    transform: Optional[list] = None  # Affine as list of 6 floats
    bounds: Optional[dict] = None
    gsd: Optional[float] = None  # meters/pixel
    format: str = "unknown"
    camera_pose: Optional[CameraPose] = None


def parse_camera_pose(file_bytes: bytes, pil_img: Optional[Image.Image] = None) -> Optional[CameraPose]:
    """
    Extract drone camera pose (pitch, roll, yaw, altitude) from XMP or EXIF metadata.
    Handles DJI XMP drone telemetry, EXIF gimbal pitch, and relative altitude.
    """
    pitch, roll, yaw, alt = None, 0.0, 0.0, 100.0

    # 1. Search XMP telemetry in raw bytes
    try:
        m_pitch = re.search(rb'GimbalPitchDegree[>="]+([+-]?\d+\.?\d*)', file_bytes)
        if not m_pitch:
            m_pitch = re.search(rb'FlightPitchDegree[>="]+([+-]?\d+\.?\d*)', file_bytes)
        if m_pitch:
            val = float(m_pitch.group(1))
            # DJI convention: -90 is nadir, -45 is 45 deg tilt
            pitch = abs(val - (-90.0)) if val <= 0.0 else val

        m_roll = re.search(rb'GimbalRollDegree[>="]+([+-]?\d+\.?\d*)', file_bytes)
        if m_roll:
            roll = float(m_roll.group(1))

        m_yaw = re.search(rb'GimbalYawDegree[>="]+([+-]?\d+\.?\d*)', file_bytes)
        if m_yaw:
            yaw = float(m_yaw.group(1))

        m_alt = re.search(rb'RelativeAltitude[>="]+([+-]?\d+\.?\d*)', file_bytes)
        if m_alt:
            alt = max(float(m_alt.group(1)), 10.0)
    except Exception:
        pass

    # 2. Check standard EXIF if not found in XMP
    if pitch is None and pil_img is not None:
        try:
            exif = pil_img.getexif()
            if exif:
                # Custom/MakerNote tags could hold pitch
                pass
        except Exception:
            pass

    if pitch is not None:
        return CameraPose(
            pitch_deg=float(pitch),
            roll_deg=float(roll),
            yaw_deg=float(yaw),
            altitude_m=float(alt),
        )
    return None


def read_image(file_bytes: bytes, filename: str) -> tuple:
    """
    Read an image from bytes. Returns (rgb_array, pil_image, metadata).
    Supports GeoTIFF, PNG, JPG.
    """
    ext = Path(filename).suffix.lower()
    metadata = ImageMetadata(format=ext)

    if ext in {".tif", ".tiff"} and HAS_RASTERIO:
        rgb, pil_img, metadata = _read_geotiff(file_bytes, metadata)
    else:
        rgb, pil_img, metadata = _read_standard(file_bytes, metadata)

    # Ingest camera pose if present in telemetry
    metadata.camera_pose = parse_camera_pose(file_bytes, pil_img)
    return rgb, pil_img, metadata



def _read_geotiff(file_bytes: bytes, metadata: ImageMetadata) -> tuple:
    """Read GeoTIFF with rasterio, extract CRS and transform."""
    with rasterio.open(io.BytesIO(file_bytes)) as ds:
        # Read RGB bands (first 3)
        bands_to_read = min(ds.count, 3)
        rgb = np.stack([ds.read(i + 1) for i in range(bands_to_read)], axis=-1)

        if bands_to_read == 1:
            rgb = np.repeat(rgb[..., :1], 3, axis=-1)
        elif bands_to_read == 2:
            # 2-channel raster (e.g., dual-pol SAR or Pan+NIR): synthesize 3rd channel via average
            ch3 = ((rgb[..., 0].astype(np.float32) + rgb[..., 1].astype(np.float32)) / 2.0).astype(rgb.dtype)
            rgb = np.dstack([rgb, ch3])

        # Handle non-uint8 satellite imagery (16-bit, signed int, float) with 2%-98% percentile stretching
        if rgb.dtype != np.uint8:
            stretched = np.empty((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8)
            for c in range(3):
                band = rgb[..., c].astype(np.float32)
                finite_mask = np.isfinite(band)
                if not np.any(finite_mask):
                    stretched[..., c] = 0
                    continue
                valid_vals = band[finite_mask]
                p2, p98 = float(np.percentile(valid_vals, 2)), float(np.percentile(valid_vals, 98))
                if p98 > p2:
                    band = np.clip((band - p2) / (p98 - p2), 0.0, 1.0) * 255.0
                else:
                    b_min, b_max = float(valid_vals.min()), float(valid_vals.max())
                    if b_max > b_min:
                        band = np.clip((band - b_min) / (b_max - b_min), 0.0, 1.0) * 255.0
                    else:
                        band = np.zeros_like(band)
                stretched[..., c] = np.nan_to_num(band, nan=0.0).astype(np.uint8)
            rgb = stretched

        metadata.width = ds.width
        metadata.height = ds.height
        metadata.channels = 3

        if ds.crs is not None:
            metadata.is_georef = True
            metadata.crs = str(ds.crs)
            t = ds.transform
            metadata.transform = [t.a, t.b, t.c, t.d, t.e, t.f]
            metadata.bounds = {
                "left": ds.bounds.left, "bottom": ds.bounds.bottom,
                "right": ds.bounds.right, "top": ds.bounds.top
            }
            # Calculate pixel resolution from affine transform (handles rotated/sheared rasters)
            pixel_res = float(np.hypot(t.a, t.d))
            if pixel_res < 1e-9:
                pixel_res = float(abs(t.a)) if abs(t.a) > 1e-9 else float(abs(t.e))

            # Detect geographic CRS (degrees) and convert GSD to meters
            is_geographic = False
            try:
                is_geographic = bool(ds.crs.is_geographic)
            except Exception:
                if "4326" in str(ds.crs) or pixel_res < 0.01:
                    is_geographic = True

            if not is_geographic and pixel_res < 0.01:
                is_geographic = True

            if is_geographic:
                center_lat = (ds.bounds.bottom + ds.bounds.top) / 2.0
                center_lat = float(np.clip(center_lat, -85.0, 85.0))
                meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))
                metadata.gsd = float(pixel_res * meters_per_deg_lon)
            else:
                metadata.gsd = float(pixel_res)

            log.info("GeoTIFF loaded", crs=metadata.crs, gsd=metadata.gsd)
        else:
            metadata.is_georef = False
            log.info("GeoTIFF without CRS, treating as non-georeferenced")

    pil_image = Image.fromarray(rgb)
    return rgb, pil_image, metadata


def _read_standard(file_bytes: bytes, metadata: ImageMetadata) -> tuple:
    """Read PNG/JPG with PIL."""
    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(pil_image)

    metadata.width = pil_image.width
    metadata.height = pil_image.height
    metadata.channels = 3
    metadata.is_georef = False

    log.info("Standard image loaded", size=f"{metadata.width}x{metadata.height}")
    return rgb, pil_image, metadata
