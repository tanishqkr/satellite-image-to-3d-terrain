"""
Heightfield mesh generation with mathematically correct vertices, faces, UVs, normals.
Outputs mesh data as numpy arrays and a 16-bit PNG heightmap for GPU displacement.
"""
import numpy as np
from PIL import Image
import io
import base64
from app.logging_config import log


from typing import Optional


def build_mesh_data(
    dsm: np.ndarray,
    rgb: np.ndarray,
    max_grid: int = 512,
    vertical_scale: float = 1.0,
    pixel_size: float = 1.0,
    uncertainty: Optional[np.ndarray] = None,
) -> dict:
    """
    Build heightfield mesh data from DSM.

    Returns dict with:
      - heightmap_b64: base64 16-bit PNG for GPU displacement
      - rgb_b64: base64 JPEG of the RGB texture
      - mesh_stats: {width, height, vertices, triangles, elevation_range}
      - dsm_colorized_b64: base64 PNG of Turbo-colormapped DSM
    """
    import time
    t_start = time.perf_counter()
    H, W = dsm.shape

    # 1. Decimate if needed (for WebGL performance)
    t_decimate_start = time.perf_counter()
    if H > max_grid or W > max_grid:
        from scipy.ndimage import zoom
        scale_h = max_grid / H
        scale_w = max_grid / W
        scale = min(scale_h, scale_w)
        dsm_decimated = zoom(dsm, scale, order=1)
    else:
        dsm_decimated = dsm
    t_decimate_ms = (time.perf_counter() - t_decimate_start) * 1000

    dH, dW = dsm_decimated.shape

    # 2. Heightmap as 16-bit PNG (fast compress_level=1)
    t_hm_start = time.perf_counter()
    d_min, d_max = dsm_decimated.min(), dsm_decimated.max()
    if d_max - d_min > 1e-8:
        hm_normalized = (dsm_decimated - d_min) / (d_max - d_min)
    else:
        hm_normalized = np.zeros_like(dsm_decimated)

    hm_16bit = (hm_normalized * 65535).astype(np.uint16)
    hm_image = Image.fromarray(hm_16bit, mode='I;16')
    hm_buf = io.BytesIO()
    hm_image.save(hm_buf, format='PNG', compress_level=1)
    heightmap_b64 = base64.b64encode(hm_buf.getvalue()).decode()
    t_hm_ms = (time.perf_counter() - t_hm_start) * 1000

    # 3. RGB texture as JPEG
    t_rgb_start = time.perf_counter()
    rgb_image = Image.fromarray(rgb)
    rgb_buf = io.BytesIO()
    rgb_image.save(rgb_buf, format='JPEG', quality=90)
    rgb_b64 = base64.b64encode(rgb_buf.getvalue()).decode()
    t_rgb_ms = (time.perf_counter() - t_rgb_start) * 1000

    # 4. Colorized DSM (Turbo colormap, fast compress_level=1)
    t_turbo_start = time.perf_counter()
    dsm_colorized = apply_turbo_colormap(hm_normalized)
    dsm_color_image = Image.fromarray(dsm_colorized)
    dsm_buf = io.BytesIO()
    dsm_color_image.save(dsm_buf, format='PNG', compress_level=1)
    dsm_colorized_b64 = base64.b64encode(dsm_buf.getvalue()).decode()
    t_turbo_ms = (time.perf_counter() - t_turbo_start) * 1000

    # 5. Tangent-space normal map for WebGL relief shading (vectorized + fast compress_level=1)
    t_norm_start = time.perf_counter()
    normal_map = compute_normal_map(dsm_decimated, pixel_size=pixel_size)
    norm_image = Image.fromarray(normal_map)
    norm_buf = io.BytesIO()
    norm_image.save(norm_buf, format='PNG', compress_level=1)
    normal_map_b64 = base64.b64encode(norm_buf.getvalue()).decode()
    t_norm_ms = (time.perf_counter() - t_norm_start) * 1000

    # 6. Risk / Evidential Uncertainty heatmap (§1.4)
    t_risk_start = time.perf_counter()
    risk_map_b64 = None
    if uncertainty is not None:
        if H > max_grid or W > max_grid:
            from scipy.ndimage import zoom
            scale_h = max_grid / H
            scale_w = max_grid / W
            scale = min(scale_h, scale_w)
            unc_decimated = zoom(uncertainty, scale, order=1)
        else:
            unc_decimated = uncertainty

        from app.services.evidential import colorize_risk_heatmap, risk_heatmap_to_base64
        risk_rgb = colorize_risk_heatmap(unc_decimated)
        risk_map_b64 = risk_heatmap_to_base64(risk_rgb)
    t_risk_ms = (time.perf_counter() - t_risk_start) * 1000

    # 7. Raw DSM Serialization: Float32 binary buffer (Base64) + compact preview list
    t_dsm_raw_start = time.perf_counter()
    dsm_float32 = dsm_decimated.astype(np.float32)
    dsm_raw_b64 = base64.b64encode(dsm_float32.tobytes()).decode()

    # For JSON list backward compatibility, downsample to 128x128 max if grid is large
    if dH > 128 or dW > 128:
        from scipy.ndimage import zoom
        scale_raw = 128.0 / max(dH, dW)
        dsm_raw_preview = zoom(dsm_decimated, scale_raw, order=1)
    else:
        dsm_raw_preview = dsm_decimated
    dsm_raw_data = np.round(dsm_raw_preview, 1).tolist()
    t_dsm_raw_ms = (time.perf_counter() - t_dsm_raw_start) * 1000

    # 8. Mesh stats
    num_vertices = dH * dW
    num_triangles = 2 * (dH - 1) * (dW - 1)

    t_total_ms = (time.perf_counter() - t_start) * 1000

    timings = {
        "decimation_ms": round(t_decimate_ms, 2),
        "heightmap_16bit_ms": round(t_hm_ms, 2),
        "rgb_texture_ms": round(t_rgb_ms, 2),
        "turbo_colormap_ms": round(t_turbo_ms, 2),
        "normal_map_ms": round(t_norm_ms, 2),
        "risk_heatmap_ms": round(t_risk_ms, 2),
        "dsm_raw_tolist_ms": round(t_dsm_raw_ms, 2),
        "total_mesh_builder_ms": round(t_total_ms, 2),
    }

    stats = {
        "width": dW,
        "height": dH,
        "original_width": W,
        "original_height": H,
        "vertices": num_vertices,
        "triangles": num_triangles,
        "elevation_min": float(d_min),
        "elevation_max": float(d_max),
        "elevation_range": float(d_max - d_min),
        "pixel_size": pixel_size,
    }

    log.info(
        "3D Mesh Builder Timing Breakdown",
        **timings,
        vertices=num_vertices,
        triangles=num_triangles,
        grid_shape=f"{dH}x{dW}",
    )

    return {
        "heightmap_b64": heightmap_b64,
        "rgb_b64": rgb_b64,
        "normal_map_b64": normal_map_b64,
        "dsm_colorized_b64": dsm_colorized_b64,
        "risk_map_b64": risk_map_b64,
        "mesh_stats": stats,
        "dsm_raw": dsm_raw_data,  # Backward-compatible compact grid
        "dsm_raw_b64": dsm_raw_b64,  # High-fidelity Float32 binary buffer
        "timings_ms": timings,
    }


def apply_turbo_colormap(values: np.ndarray) -> np.ndarray:
    """
    Apply Turbo colormap to normalized [0,1] values.
    Returns (H, W, 3) uint8 RGB array.
    """
    H, W = values.shape
    flat = values.flatten()

    # Generate turbo-like colormap: blue → cyan → green → yellow → red
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0.0, t * 4, 1.0
        elif t < 0.5:
            r, g, b = 0.0, 1.0, 1.0 - (t - 0.25) * 4
        elif t < 0.75:
            r, g, b = (t - 0.5) * 4, 1.0, 0.0
        else:
            r, g, b = 1.0, 1.0 - (t - 0.75) * 4, 0.0
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]

    indices = (flat * 255).astype(np.uint8)
    colorized = lut[indices].reshape(H, W, 3)
    return colorized


def compute_normal_map(dsm: np.ndarray, pixel_size: float = 1.0, vertical_scale: float = 1.0) -> np.ndarray:
    """
    Compute tangent-space normal map from DSM elevation grid with vectorized memory optimization.
    Returns (H, W, 3) uint8 RGB array where normal (0, 0, 1) -> (128, 128, 255).
    """
    safe_pixel = max(float(pixel_size), 0.01)
    gy, gx = np.gradient(dsm, safe_pixel, safe_pixel)
    nx = -gx * vertical_scale
    ny = gy * vertical_scale

    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    nx *= inv_norm
    ny *= inv_norm
    nz = inv_norm

    H, W = dsm.shape
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[..., 0] = np.clip(nx * 127.5 + 127.5, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(ny * 127.5 + 127.5, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(nz * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return out
