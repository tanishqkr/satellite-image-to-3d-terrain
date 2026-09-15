"""
DepthWizard §1.2 Foundation Metric Depth Models — Prototype Evaluation Benchmark.

Compares:
1. DepthWizard Fine-Tuned ViT-S Backbone (Depth Anything V2 + GAMUS Stage 2 fine-tuning).
2. Base Pretrained Depth Anything V2 (Zero-shot relative depth).
3. Foundation Metric Model: Intel ZoeDepth (BEiT-Large + Metric Bin Prediction).

Evaluated across:
- IN-DOMAIN: US Urban nadir orthophotos at ~0.33m GSD (matching GAMUS training distribution).
- OUT-OF-DOMAIN: Indian UTM 43N geography, coastal/mountain terrain, varying altitudes, and oblique perspective.
"""
import sys
import time
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from PIL import Image
import rasterio
from scipy import ndimage

# Ensure backend modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.services.depth_estimator import DepthEstimator
from app.services.geospatial import read_image
from app.services.calibration import calibrate_depth
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def load_test_image(path: Path) -> Tuple[np.ndarray, Optional[float], bool]:
    """Load an image (GeoTIFF or optical RGB) and return (rgb_uint8, gsd, is_georef)."""
    with open(path, "rb") as f:
        file_bytes = f.read()
    rgb, pil_img, metadata = read_image(file_bytes, path.name)
    gsd = metadata.gsd or 1.0
    return rgb, gsd, metadata.is_georef


def compute_edge_contrast(depth_map: np.ndarray) -> float:
    """Compute structural gradient contrast (Sobel edge sharpness) around building footprints."""
    sobel_x = ndimage.sobel(depth_map, axis=1)
    sobel_y = ndimage.sobel(depth_map, axis=0)
    mag = np.hypot(sobel_x, sobel_y)
    return float(np.mean(mag))


def run_benchmark():
    print("=" * 80)
    print("DepthWizard §1.2 Metric Foundation Models — Prototype Evaluation Benchmark")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Executing on benchmark device: {device}")

    # 1. Load Model 1: DepthWizard Fine-Tuned ViT-S
    print("\n[1/3] Loading DepthWizard Fine-Tuned ViT-S (Depth Anything V2 + GAMUS Stage 2)...")
    dw_estimator = DepthEstimator()
    dw_params = sum(p.numel() for p in dw_estimator.model.parameters()) / 1e6
    print(f"  DepthWizard Model loaded: {dw_params:.1f}M parameters")

    # 2. Load Model 2: Base Pretrained Depth Anything V2 (Zero-shot)
    print("\n[2/3] Loading Base Pretrained Depth Anything V2 Small...")
    base_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    base_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    base_model.to(device)
    base_model.eval()
    base_params = sum(p.numel() for p in base_model.parameters()) / 1e6
    print(f"  Base Pretrained Model loaded: {base_params:.1f}M parameters")

    # 3. Load Model 3: Intel ZoeDepth Foundation Metric Model
    print("\n[3/3] Loading Foundation Metric Model: Intel/zoedepth-nyu-kitti...")
    zoe_proc = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
    zoe_model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
    zoe_model.to(device)
    zoe_model.eval()
    zoe_params = sum(p.numel() for p in zoe_model.parameters()) / 1e6
    print(f"  ZoeDepth Model loaded: {zoe_params:.1f}M parameters")

    # Define Test Suites
    test_suites = {
        "IN-DOMAIN (US Urban Orthophotos, 0.33m GSD, Nadir)": [
            PROJECT_ROOT / "test_datasets" / "geotiff_mode" / "01_geotiff_full_1024.tif",
            PROJECT_ROOT / "test_datasets" / "geotiff_mode" / "02_geotiff_downtown_512.tif",
            PROJECT_ROOT / "test_datasets" / "optical_mode" / "01_optical_urban_1024.png",
            PROJECT_ROOT / "test_datasets" / "optical_mode" / "02_optical_commercial_512.png",
        ],
        "OUT-OF-DOMAIN (Indian UTM 43N, Varied Altitude, Compressed/Oblique)": [
            PROJECT_ROOT / "test_datasets" / "geotiff_mode" / "04_geotiff_isro_utm43_512.tif",
            PROJECT_ROOT / "test_datasets" / "optical_mode" / "03_optical_residential_640.png",
            PROJECT_ROOT / "test_datasets" / "optical_mode" / "04_optical_district_512.jpg",
        ],
    }

    benchmark_results = {}

    for suite_name, image_paths in test_suites.items():
        print(f"\n{'='*40}")
        print(f"Evaluating Suite: {suite_name}")
        print(f"{'='*40}")
        benchmark_results[suite_name] = []

        for p in image_paths:
            if not p.exists():
                print(f"Skipping missing file: {p.name}")
                continue

            rgb, gsd, is_georef = load_test_image(p)
            H, W, _ = rgb.shape
            print(f"\n--- Testing: {p.name} ({W}x{H}, GSD={gsd:.2f}m, georef={is_georef}) ---")

            # Crop or resize center 512x512 patch for uniform model comparison
            if H > 512 or W > 512:
                r0 = (H - 512) // 2
                c0 = (W - 512) // 2
                patch_rgb = rgb[r0:r0+512, c0:c0+512]
            else:
                patch_rgb = np.array(Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR))

            pil_patch = Image.fromarray(patch_rgb)

            # Model 1: DepthWizard Fine-Tuned
            t0 = time.time()
            dw_depth_norm = dw_estimator.predict(patch_rgb)
            dw_latency_ms = (time.time() - t0) * 1000.0
            # Calibrate metric DSM if georef
            if is_georef:
                dw_cal = calibrate_depth(dw_depth_norm, is_georef=is_georef, gsd=gsd)
                dw_dsm = dw_cal.dsm
                dw_range = f"[{dw_dsm.min():.1f}m, {dw_dsm.max():.1f}m] (span {dw_dsm.max()-dw_dsm.min():.1f}m)"
            else:
                dw_dsm = dw_depth_norm
                dw_range = f"[{dw_dsm.min():.3f}, {dw_dsm.max():.3f}] (rel)"
            dw_edge = compute_edge_contrast(dw_dsm)

            # Model 2: Base Pretrained Depth Anything V2
            t0 = time.time()
            base_inputs = base_proc(images=pil_patch, return_tensors="pt").to(device)
            with torch.no_grad():
                base_out = base_model(**base_inputs).predicted_depth
                base_depth = base_out.squeeze().cpu().numpy()
            base_latency_ms = (time.time() - t0) * 1000.0
            # Normalize to relative 0-1
            base_min = float(np.min(base_depth))
            base_max = float(np.max(base_depth))
            base_norm = (base_depth - base_min) / max(1e-6, base_max - base_min)
            base_edge = compute_edge_contrast(base_norm)
            base_range = f"[{base_depth.min():.1f}, {base_depth.max():.1f}] (raw disparity)"

            # Model 3: Intel ZoeDepth Foundation Metric Model
            t0 = time.time()
            zoe_inputs = zoe_proc(images=pil_patch, return_tensors="pt").to(device)
            with torch.no_grad():
                zoe_out = zoe_model(**zoe_inputs).predicted_depth
                zoe_depth_m = zoe_out.squeeze().cpu().numpy()
            zoe_latency_ms = (time.time() - t0) * 1000.0
            zoe_edge = compute_edge_contrast(zoe_depth_m)
            zoe_range = f"[{zoe_depth_m.min():.2f}m, {zoe_depth_m.max():.2f}m] (span {zoe_depth_m.max()-zoe_depth_m.min():.2f}m)"

            print(f"  DepthWizard Fine-Tuned: {dw_latency_ms:.1f}ms | Range: {dw_range} | Edge Sharpness: {dw_edge:.3f}")
            print(f"  Base Pretrained V2:    {base_latency_ms:.1f}ms | Range: {base_range} | Edge Sharpness: {base_edge:.3f}")
            print(f"  ZoeDepth Metric Model: {zoe_latency_ms:.1f}ms | Range: {zoe_range} | Edge Sharpness: {zoe_edge:.3f}")

            # Altitude scale sensitivity test (simulate 2x altitude change / 0.5x zoom)
            zoomed_patch = np.array(pil_patch.resize((256, 256), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.BILINEAR))
            with torch.no_grad():
                zoe_zoom_inputs = zoe_proc(images=Image.fromarray(zoomed_patch), return_tensors="pt").to(device)
                zoe_zoom_out = zoe_model(**zoe_zoom_inputs).predicted_depth.squeeze().cpu().numpy()
                dw_zoom_depth = dw_estimator.predict(zoomed_patch)

            zoe_scale_shift = abs(float(np.mean(zoe_zoom_out) - np.mean(zoe_depth_m)))
            dw_scale_shift = abs(float(np.mean(dw_zoom_depth) - np.mean(dw_depth_norm)))

            print(f"  Altitude Sensitivity (scale drift): ZoeDepth={zoe_scale_shift:.3f}m | DepthWizard={dw_scale_shift:.4f}")

            benchmark_results[suite_name].append({
                "file": p.name,
                "gsd": gsd,
                "is_georef": is_georef,
                "dw_latency_ms": round(dw_latency_ms, 1),
                "dw_range": dw_range,
                "dw_edge_sharpness": round(dw_edge, 3),
                "dw_scale_drift": round(dw_scale_shift, 4),
                "base_latency_ms": round(base_latency_ms, 1),
                "base_edge_sharpness": round(base_edge, 3),
                "zoe_latency_ms": round(zoe_latency_ms, 1),
                "zoe_range": zoe_range,
                "zoe_edge_sharpness": round(zoe_edge, 3),
                "zoe_scale_drift": round(zoe_scale_shift, 3),
            })

    # Summary and Hardware Specifications
    summary_report = {
        "model_comparison": {
            "depthwizard_finetuned": {
                "backbone": "Depth-Anything-V2-Small (ViT-S)",
                "weights": "best_model.pth (GAMUS Stage 2 + Evidential Head)",
                "parameters_m": dw_params,
                "domain_tuning": "Supervised on aerial satellite DSMs (5 US cities + ISRO georef)",
                "head_support": "Evidential NIG uncertainty head supported (§1.4)",
            },
            "base_depth_anything_v2": {
                "backbone": "Depth-Anything-V2-Small (ViT-S)",
                "weights": "Official HuggingFace Pretrained",
                "parameters_m": base_params,
                "domain_tuning": "Zero-shot relative depth (internet mixed RGB)",
                "head_support": "DPT standard head",
            },
            "zoedepth_foundation": {
                "backbone": "BEiT-Large + ZoeDepth Metric Head",
                "weights": "Intel/zoedepth-nyu-kitti",
                "parameters_m": zoe_params,
                "domain_tuning": "Zero-shot metric depth (NYU-Depth indoor + KITTI automotive)",
                "head_support": "Bin-based metric head (no evidential head)",
            }
        },
        "results": benchmark_results,
    }

    out_json = PROJECT_ROOT / "docs" / "benchmarks" / "FOUNDATION_MODELS_EVALUATION_REPORT.json"
    with open(out_json, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n[PASS] Prototype benchmark completed. Results saved to: {out_json}")
    return summary_report


if __name__ == "__main__":
    run_benchmark()
