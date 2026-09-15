"""
Comprehensive Profiler for 3D Mesh Generation Pipeline in DepthWizard.
Measures latency (in milliseconds), memory footprint, and data transfer sizes.
"""
import sys
import os
import time
import json
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure backend root is in sys.path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.depth_estimator import DepthEstimator
from app.services.calibration import calibrate_depth
from app.services.mesh_builder import build_mesh_data, compute_normal_map, apply_turbo_colormap
from app.services.geospatial import read_image
from app.api.routes import _process_pipeline


def profile_image(image_path: Path, runs: int = 3):
    print(f"\n{'='*80}")
    print(f"PROFILING: {image_path.name} ({image_path.stat().st_size / 1024:.1f} KB)")
    print(f"{'='*80}")

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    estimator = DepthEstimator()

    # Warmup pass
    print("Running warmup pass...")
    _ = _process_pipeline(
        file_bytes=file_bytes,
        filename=image_path.name,
        request_id="warmup",
        estimator=estimator,
    )
    print("Warmup complete.\n")

    # Detailed measurement runs
    pipeline_records = []
    mesh_records = []

    for r in range(runs):
        req_id = f"bench_{r+1}"
        t_all_start = time.perf_counter()

        # Step 1: Read image
        t0 = time.perf_counter()
        rgb, pil_img, meta = read_image(file_bytes, image_path.name)
        t_ingest = (time.perf_counter() - t0) * 1000

        # Step 2: Depth Estimator
        t0 = time.perf_counter()
        depth = estimator.predict(pil_img)
        t_depth = (time.perf_counter() - t0) * 1000

        # Step 3: Calibration
        t0 = time.perf_counter()
        cal = calibrate_depth(
            depth,
            is_georef=meta.is_georef,
            gsd=meta.gsd,
            bounds=meta.bounds,
            crs=meta.crs,
            transform=meta.transform,
            rgb=rgb,
        )
        t_calib = (time.perf_counter() - t0) * 1000

        # Step 4: Mesh Builder with deep inspection
        t0 = time.perf_counter()
        mesh_result = build_mesh_data(
            dsm=cal.dsm,
            rgb=rgb,
            pixel_size=meta.gsd or 1.0,
            uncertainty=cal.uncertainty,
        )
        t_mesh = (time.perf_counter() - t0) * 1000

        # Step 5: Cache and GeoTIFF export
        t0 = time.perf_counter()
        export_dir = backend_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        np.save(export_dir / f"{req_id}_dsm.npy", cal.dsm)
        np.save(export_dir / f"{req_id}_rgb.npy", rgb)
        t_cache = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_all_start) * 1000

        pipeline_records.append({
            "ingest_ms": t_ingest,
            "depth_ms": t_depth,
            "calibration_ms": t_calib,
            "mesh_ms": t_mesh,
            "cache_ms": t_cache,
            "total_ms": t_total,
        })
        mesh_records.append(mesh_result["timings_ms"])

    # Average pipeline stats
    avg_pipe = {k: np.mean([r[k] for r in pipeline_records]) for k in pipeline_records[0]}
    avg_mesh = {k: np.mean([r[k] for r in mesh_records]) for k in mesh_records[0]}

    print("\n--- 1. FULL PIPELINE LATENCY BREAKDOWN (Averaged over runs) ---")
    for k, v in avg_pipe.items():
        pct = (v / avg_pipe['total_ms']) * 100
        print(f"  • {k:<20}: {v:8.2f} ms ({pct:5.1f}%)")

    print("\n--- 2. 3D MESH BUILDER SUB-STEP BREAKDOWN ---")
    for k, v in avg_mesh.items():
        if k != "total_mesh_builder_ms":
            pct = (v / avg_mesh['total_mesh_builder_ms']) * 100
            print(f"  • {k:<25}: {v:8.2f} ms ({pct:5.1f}%)")
        else:
            print(f"  --> {k:<21}: {v:8.2f} ms (100.0%)")

    # Deep Micro-benchmarking inside Mesh Builder components
    print("\n--- 3. MICRO-BENCHMARK: SUB-STEP DEEP DIVE & PAYLOAD SIZES ---")
    H, W = cal.dsm.shape
    dsm_512 = mesh_result["mesh_stats"]["width"], mesh_result["mesh_stats"]["height"]
    print(f"  Input Resolution : {W}x{H}")
    print(f"  Mesh Grid Size   : {dsm_512[0]}x{dsm_512[1]} ({mesh_result['mesh_stats']['vertices']} vertices)")

    # Data payload sizes
    hm_bytes = len(mesh_result["heightmap_b64"])
    rgb_bytes = len(mesh_result["rgb_b64"])
    turbo_bytes = len(mesh_result["dsm_colorized_b64"])
    norm_bytes = len(mesh_result["normal_map_b64"]) if mesh_result["normal_map_b64"] else 0
    dsm_raw_json = len(json.dumps(mesh_result["dsm_raw"]))
    dsm_raw_b64_bytes = len(mesh_result.get("dsm_raw_b64", ""))

    total_payload = hm_bytes + rgb_bytes + turbo_bytes + norm_bytes + dsm_raw_json + dsm_raw_b64_bytes

    print("\n  PAYLOAD SIZES (Over the wire to client):")
    print(f"  • Heightmap 16-bit PNG (Base64) : {hm_bytes / 1024:8.1f} KB ({hm_bytes / total_payload * 100:4.1f}%)")
    print(f"  • RGB Texture JPEG (Base64)     : {rgb_bytes / 1024:8.1f} KB ({rgb_bytes / total_payload * 100:4.1f}%)")
    print(f"  • Turbo Colorized DSM (Base64)  : {turbo_bytes / 1024:8.1f} KB ({turbo_bytes / total_payload * 100:4.1f}%)")
    print(f"  • Tangent Normal Map (Base64)   : {norm_bytes / 1024:8.1f} KB ({norm_bytes / total_payload * 100:4.1f}%)")
    print(f"  • dsm_raw Compact JSON Array    : {dsm_raw_json / 1024:8.1f} KB ({dsm_raw_json / total_payload * 100:4.1f}%)")
    print(f"  • dsm_raw_b64 Float32 Buffer    : {dsm_raw_b64_bytes / 1024:8.1f} KB ({dsm_raw_b64_bytes / total_payload * 100:4.1f}%)")
    print(f"  -------------------------------------------------------------")
    print(f"  • TOTAL RESPONSE PAYLOAD        : {total_payload / (1024*1024):8.2f} MB")

    # Test PNG compression level impact
    print("\n--- 4. OPTIMIZATION EXPERIMENT: PNG COMPRESSION LEVEL BENCHMARK ---")
    hm_normalized = (cal.dsm - cal.dsm.min()) / (cal.dsm.max() - cal.dsm.min() + 1e-8)
    if H > 512 or W > 512:
        from scipy.ndimage import zoom
        hm_normalized = zoom(hm_normalized, 512 / max(H, W), order=1)
    hm_16 = (hm_normalized * 65535).astype(np.uint16)
    img_16 = Image.fromarray(hm_16, mode='I;16')

    for compress_lvl in [0, 1, 3, 6, 9]:
        t_c0 = time.perf_counter()
        buf = io.BytesIO()
        img_16.save(buf, format='PNG', compress_level=compress_lvl)
        b64 = base64.b64encode(buf.getvalue()).decode()
        t_c = (time.perf_counter() - t_c0) * 1000
        print(f"  compress_level={compress_lvl}: {t_c:6.2f} ms | size={len(b64)/1024:6.1f} KB")

    # Test dsm_raw alternatives
    print("\n--- 5. OPTIMIZATION EXPERIMENT: dsm_raw SERIALIZATION ---")
    t0 = time.perf_counter()
    _ = np.round(cal.dsm[:512, :512], 2).tolist()
    t_round_tolist = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = cal.dsm[:512, :512].tolist()
    t_raw_tolist = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    # Float32 base64 binary buffer alternative
    raw_bytes = cal.dsm[:512, :512].astype(np.float32).tobytes()
    raw_b64 = base64.b64encode(raw_bytes).decode()
    t_binary_b64 = (time.perf_counter() - t0) * 1000

    print(f"  np.round().tolist() (current) : {t_round_tolist:6.2f} ms | JSON size: {dsm_raw_json/1024:.1f} KB")
    print(f"  .tolist() without np.round    : {t_raw_tolist:6.2f} ms")
    print(f"  Float32 Base64 Binary Buffer  : {t_binary_b64:6.2f} ms | size: {len(raw_b64)/1024:.1f} KB ({(1 - len(raw_b64)/dsm_raw_json)*100:.1f}% reduction!)")


if __name__ == "__main__":
    test_files = [
        Path("frontend/public/sample_satellite_urban.png"),
        Path("frontend/public/sample_satellite_georef.tif"),
    ]
    for p in test_files:
        if p.exists():
            profile_image(p, runs=3)
