import uuid
import time
import io
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.config import MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS, EXPORTS_DIR
from app.api.schemas import (
    InferenceResponse,
    HealthResponse,
    ViewshedRequest,
    ViewshedResponse,
    LineOfSightRequest,
    LineOfSightResponse,
    BdaDiffRequest,
    BdaDiffResponse,
    StructureBdaSchema,
    NoeRouteRequest,
    NoeRouteResponse,
    Waypoint3DSchema,
)
from app.logging_config import log
import torch

router = APIRouter()

# These get set by main.py on startup
depth_estimator = None


def set_estimator(estimator):
    global depth_estimator
    depth_estimator = estimator


@router.get("/health", response_model=HealthResponse)
async def health():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu = "CPU"
        vram = 0
    return HealthResponse(status="ok", gpu=gpu, vram_gb=round(vram, 2))


from starlette.concurrency import run_in_threadpool


def _process_pipeline(
    file_bytes: bytes,
    filename: str,
    request_id: str,
    estimator,
    estimate_uncertainty: bool = False,
    manual_pose: Optional[object] = None,
    solar_elevation: Optional[float] = None,
    solar_azimuth: Optional[float] = None,
    capture_time: Optional[str] = None,
):
    """Synchronous pipeline executed in threadpool to prevent event-loop blocking."""
    from app.services.geospatial import read_image
    from app.services.calibration import calibrate_depth
    from app.services.mesh_builder import build_mesh_data
    from app.services.orthorectify import pre_rectify, post_rectify, CameraPose
    from app.services.dtm_fusion import fetch_and_resample_global_dtm
    from PIL import Image
    import json
    import rasterio
    from rasterio.transform import Affine

    import time
    t_pipeline_start = time.perf_counter()

    # 1. Ingest image
    t_ingest_start = time.perf_counter()
    rgb, pil_image, metadata = read_image(file_bytes, filename)
    if manual_pose is not None:
        metadata.camera_pose = manual_pose
    t_ingest_ms = (time.perf_counter() - t_ingest_start) * 1000

    # Obliqueness check (§2.2)
    is_oblique = (
        metadata.camera_pose is not None and
        getattr(metadata.camera_pose, "is_oblique", False)
    )

    inference_pil = pil_image
    t_pre_rectify_ms = 0.0
    if is_oblique:
        t_pre_start = time.perf_counter()
        # Pass A: Pre-inference de-obliquing using coarse global DTM (or flat proxy)
        coarse_dtm = None
        if metadata.is_georef and metadata.bounds and metadata.crs and metadata.transform:
            try:
                coarse_dtm = fetch_and_resample_global_dtm(
                    metadata.bounds, metadata.crs, rgb.shape[:2], metadata.transform
                )
            except Exception:
                coarse_dtm = None

        rgb_deoblique = pre_rectify(
            image=rgb,
            coarse_dtm=coarse_dtm,
            camera_pose=metadata.camera_pose,
            gsd=metadata.gsd or 0.5,
        )
        inference_pil = Image.fromarray(rgb_deoblique)
        t_pre_rectify_ms = (time.perf_counter() - t_pre_start) * 1000

    # 2. Depth prediction (standard or Single-Pass Evidential uncertainty §1.4)
    t_depth_start = time.perf_counter()
    conf_mean = None
    uncertainty_raw = None
    unc_stats = None
    if estimate_uncertainty:
        relative_depth, uncertainty_raw, metrics = estimator.predict_evidential(inference_pil)
        conf_mean = float(metrics["confidence_mean"])
        unc_stats = {
            "confidence_mean": round(float(metrics["confidence_mean"]), 3),
            "epistemic_mean": round(float(metrics["epistemic_mean"]), 4),
            "aleatoric_mean": round(float(metrics["aleatoric_mean"]), 4),
        }
    else:
        relative_depth = estimator.predict(inference_pil)
    t_depth_ms = (time.perf_counter() - t_depth_start) * 1000

    # 3. Calibration (propagates uncertainty through alpha into meters)
    t_calib_start = time.perf_counter()
    cal_result = calibrate_depth(
        relative_depth,
        is_georef=metadata.is_georef,
        gsd=metadata.gsd,
        bounds=metadata.bounds,
        crs=metadata.crs,
        transform=metadata.transform,
        rgb=rgb,
        solar_elevation=solar_elevation,
        solar_azimuth=solar_azimuth,
        capture_time=capture_time,
        relative_uncertainty=uncertainty_raw,
    )

    if cal_result.uncertainty is not None and unc_stats is not None:
        unc_stats["uncertainty_mean_m"] = round(float(cal_result.uncertainty_mean), 2)
        unc_stats["uncertainty_max_m"] = round(float(cal_result.uncertainty_max), 2)
        unc_stats["unit"] = cal_result.uncertainty_unit
    t_calib_ms = (time.perf_counter() - t_calib_start) * 1000

    # Pass B: Post-inference true-orthorectification using fine DSM with z-buffering
    mesh_rgb = rgb
    rectification_info = None
    t_post_rectify_ms = 0.0
    if is_oblique:
        t_post_start = time.perf_counter()
        ortho_rgb, occluded = post_rectify(
            image=rgb,
            fine_dsm=cal_result.dsm,
            camera_pose=metadata.camera_pose,
            gsd=metadata.gsd or 0.5,
        )
        mesh_rgb = ortho_rgb
        rectification_info = {
            "applied": True,
            "pitch_deg": float(metadata.camera_pose.pitch_deg),
            "roll_deg": float(metadata.camera_pose.roll_deg),
            "occluded_fraction": float(occluded.mean()),
            "occluded_pixel_count": int(occluded.sum()),
        }
        t_post_rectify_ms = (time.perf_counter() - t_post_start) * 1000

    # 4. Mesh generation
    t_mesh_start = time.perf_counter()
    mesh_data = build_mesh_data(
        dsm=cal_result.dsm,
        rgb=mesh_rgb,
        pixel_size=metadata.gsd or 1.0,
        uncertainty=cal_result.uncertainty,
    )
    t_mesh_ms = (time.perf_counter() - t_mesh_start) * 1000

    # 5. Persist cache
    t_cache_start = time.perf_counter()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EXPORTS_DIR / f"{request_id}_dsm.npy", cal_result.dsm)
    np.save(EXPORTS_DIR / f"{request_id}_rgb.npy", mesh_rgb)
    if cal_result.uncertainty is not None:
        np.save(EXPORTS_DIR / f"{request_id}_risk.npy", cal_result.uncertainty)

    meta_dict = {
        "is_georef": metadata.is_georef,
        "crs": metadata.crs,
        "transform": metadata.transform,
        "width": metadata.width,
        "height": metadata.height,
        "rectification": rectification_info,
    }
    with open(EXPORTS_DIR / f"{request_id}_meta.json", "w") as f:
        json.dump(meta_dict, f)

    # 6. Pre-generate GeoTIFF once safely using atomic replace to avoid Windows file-lock contention
    output_path = EXPORTS_DIR / f"{request_id}_dsm.tif"
    if not output_path.exists():
        import os
        if metadata.transform:
            transform = Affine(*metadata.transform)
            crs = metadata.crs
        else:
            transform = Affine(1.0, 0, 0, 0, -1.0, cal_result.dsm.shape[0])
            crs = None

        temp_path = EXPORTS_DIR / f"{request_id}_{uuid.uuid4().hex[:6]}_temp.tif"
        with rasterio.open(
            str(temp_path), 'w', driver='GTiff',
            height=cal_result.dsm.shape[0], width=cal_result.dsm.shape[1],
            count=1, dtype='float32',
            crs=crs, transform=transform,
        ) as dst:
            dst.write(cal_result.dsm, 1)

        try:
            os.replace(temp_path, output_path)
        except OSError:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
    t_cache_save_ms = (time.perf_counter() - t_cache_start) * 1000
    t_pipeline_total_ms = (time.perf_counter() - t_pipeline_start) * 1000

    log.info(
        "End-to-End Pipeline Stage Timing Breakdown",
        request_id=request_id,
        image_shape=f"{metadata.width}x{metadata.height}",
        ingest_ms=round(t_ingest_ms, 2),
        pre_rectify_ms=round(t_pre_rectify_ms, 2) if is_oblique else None,
        depth_inference_ms=round(t_depth_ms, 2),
        calibration_ms=round(t_calib_ms, 2),
        post_rectify_ms=round(t_post_rectify_ms, 2) if is_oblique else None,
        mesh_builder_ms=round(t_mesh_ms, 2),
        cache_geotiff_save_ms=round(t_cache_save_ms, 2),
        total_pipeline_ms=round(t_pipeline_total_ms, 2),
    )

    return metadata, cal_result, mesh_data, conf_mean, rectification_info, unc_stats


@router.post("/upload", response_model=InferenceResponse)
async def upload_image(
    file: UploadFile = File(...),
    estimate_uncertainty: bool = False,
    camera_pitch: Optional[float] = None,
    camera_roll: Optional[float] = None,
    camera_yaw: Optional[float] = None,
    camera_altitude: Optional[float] = None,
    camera_fov: Optional[float] = None,
    solar_elevation: Optional[float] = None,
    solar_azimuth: Optional[float] = None,
    capture_time: Optional[str] = None,
):
    from app.services.orthorectify import CameraPose
    request_id = str(uuid.uuid4())[:8]
    t_start = time.time()

    log.info("Upload received", request_id=request_id, filename=file.filename)

    # 1. Validate file
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Accepted: {ALLOWED_EXTENSIONS}")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max: {MAX_UPLOAD_SIZE_MB}MB")

    if depth_estimator is None:
        raise HTTPException(503, "Depth estimator service is not ready")

    manual_pose = None
    if camera_pitch is not None:
        manual_pose = CameraPose(
            pitch_deg=camera_pitch,
            roll_deg=camera_roll or 0.0,
            yaw_deg=camera_yaw or 0.0,
            altitude_m=camera_altitude or 100.0,
            fov_deg=camera_fov or 60.0,
        )

    # 2. Run synchronous ML & Geospatial pipeline in worker threadpool (non-blocking)
    try:
        metadata, cal_result, mesh_data, conf_mean, rect_info, unc_stats = await run_in_threadpool(
            _process_pipeline,
            file_bytes,
            file.filename,
            request_id,
            depth_estimator,
            estimate_uncertainty,
            manual_pose,
            solar_elevation,
            solar_azimuth,
            capture_time,
        )
    except Exception as e:
        log.error("Pipeline failed", request_id=request_id, error=str(e))
        raise HTTPException(422, f"Pipeline failed: {str(e)}")

    inference_time = (time.time() - t_start) * 1000


    log.info("Inference complete", request_id=request_id,
             time_ms=f"{inference_time:.0f}",
             dsm_range=f"[{cal_result.dsm_min:.1f}, {cal_result.dsm_max:.1f}]")

    return InferenceResponse(
        request_id=request_id,
        heightmap_b64=mesh_data["heightmap_b64"],
        rgb_b64=mesh_data["rgb_b64"],
        normal_map_b64=mesh_data.get("normal_map_b64"),
        dsm_colorized_b64=mesh_data["dsm_colorized_b64"],
        mesh_stats=mesh_data["mesh_stats"],
        calibration={
            "alpha": cal_result.alpha,
            "min": cal_result.dsm_min,
            "max": cal_result.dsm_max,
            "mean": cal_result.dsm_mean,
            "mode": cal_result.mode,
            "unit": cal_result.unit,
            "fusion_applied": cal_result.fusion_applied,
            "shadow_calibrated": cal_result.shadow_calibrated,
        },
        dsm_raw=mesh_data["dsm_raw"],
        dsm_raw_b64=mesh_data.get("dsm_raw_b64"),
        is_georef=metadata.is_georef,
        crs=metadata.crs,
        confidence_mean=conf_mean,
        uncertainty_map_b64=mesh_data.get("risk_map_b64"),
        uncertainty_stats=unc_stats,
        rectification=rect_info,
        inference_time_ms=round(inference_time, 1),
    )


@router.get("/export/{request_id}")
async def export_dsm(request_id: str):
    """Download computed DSM as GeoTIFF without Windows file lock race."""
    output_path = EXPORTS_DIR / f"{request_id}_dsm.tif"
    dsm_path = EXPORTS_DIR / f"{request_id}_dsm.npy"
    meta_path = EXPORTS_DIR / f"{request_id}_meta.json"

    # If GeoTIFF does not already exist, create it once safely
    if not output_path.exists():
        if not dsm_path.exists() or not meta_path.exists():
            raise HTTPException(404, "DSM not found. Run inference first.")

        def _generate_tif():
            import json
            import os
            import rasterio
            from rasterio.transform import Affine

            dsm = np.load(dsm_path)
            with open(meta_path) as f:
                meta = json.load(f)

            if meta.get("transform"):
                transform = Affine(*meta["transform"])
                crs = meta["crs"]
            else:
                transform = Affine(1.0, 0, 0, 0, -1.0, dsm.shape[0])
                crs = None

            temp_path = EXPORTS_DIR / f"{request_id}_{uuid.uuid4().hex[:6]}_temp.tif"
            with rasterio.open(
                str(temp_path), 'w', driver='GTiff',
                height=dsm.shape[0], width=dsm.shape[1],
                count=1, dtype='float32',
                crs=crs, transform=transform,
            ) as dst:
                dst.write(dsm, 1)

            try:
                os.replace(temp_path, output_path)
            except OSError:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        await run_in_threadpool(_generate_tif)

    return FileResponse(
        str(output_path),
        media_type="image/tiff",
        filename=f"depthwizard_dsm_{request_id}.tif"
    )


@router.post("/viewshed", response_model=ViewshedResponse)
async def compute_viewshed_endpoint(req: ViewshedRequest):
    """Compute 2D observer viewshed over cached DSM (§4.1)."""
    dsm_path = EXPORTS_DIR / f"{req.request_id}_dsm.npy"
    meta_path = EXPORTS_DIR / f"{req.request_id}_meta.json"
    if not dsm_path.exists():
        raise HTTPException(404, f"DSM for request_id '{req.request_id}' not found. Run inference first.")

    def _calc_viewshed():
        import io
        import base64
        from PIL import Image
        from app.services.visibility import compute_viewshed, sample_dsm_bilinear

        dsm = np.load(dsm_path)
        H, W = dsm.shape

        gsd = 1.0
        if meta_path.exists():
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                    if meta.get("transform"):
                        gsd = abs(float(meta["transform"][0]))
            except Exception:
                pass

        ground_elev = sample_dsm_bilinear(dsm, req.observer_col, req.observer_row)
        obs_world_x = req.observer_col * gsd
        obs_world_y = ground_elev + req.observer_agl
        obs_world_z = req.observer_row * gsd

        viewshed_grid = compute_viewshed(
            dsm=dsm,
            observer_pos=(obs_world_x, obs_world_y, obs_world_z),
            gsd=gsd,
            origin=(0.0, 0.0, 0.0),
            max_range_m=req.max_range_m,
            target_height_m=req.target_height_m,
        )

        visible_count = int(np.sum(viewshed_grid))
        total_count = H * W
        vis_ratio = float(visible_count / total_count) if total_count > 0 else 0.0

        # Colorize overlay: RGBA where visible is emerald green (#10B981), occluded is crimson red (#EF4444)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[viewshed_grid == 1] = [16, 185, 129, 140]
        rgba[viewshed_grid == 0] = [239, 68, 68, 110]

        pil_img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode()

        return ViewshedResponse(
            request_id=req.request_id,
            observer_elevation_msl=round(ground_elev + req.observer_agl, 2),
            observer_agl=round(req.observer_agl, 2),
            visible_cells=visible_count,
            total_cells=total_count,
            visible_fraction=round(vis_ratio, 4),
            viewshed_b64=b64_str,
        )

    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_calc_viewshed)


@router.post("/los", response_model=LineOfSightResponse)
async def compute_los_endpoint(req: LineOfSightRequest):
    """Compute point-to-point Line of Sight (LoS) and elevation profile (§4.1)."""
    dsm_path = EXPORTS_DIR / f"{req.request_id}_dsm.npy"
    meta_path = EXPORTS_DIR / f"{req.request_id}_meta.json"
    if not dsm_path.exists():
        raise HTTPException(404, f"DSM for request_id '{req.request_id}' not found. Run inference first.")

    def _calc_los():
        from app.services.visibility import ray_march_los

        dsm = np.load(dsm_path)
        gsd = 1.0
        if meta_path.exists():
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                    if meta.get("transform"):
                        gsd = abs(float(meta["transform"][0]))
            except Exception:
                pass

        p0 = tuple(float(x) for x in req.p0)
        p1 = tuple(float(x) for x in req.p1)

        result = ray_march_los(
            p0=p0,
            p1=p1,
            dsm=dsm,
            gsd=gsd,
            sample_step_m=req.sample_step_m,
            tolerance_m=req.tolerance_m,
        )

        return LineOfSightResponse(
            request_id=req.request_id,
            visible=result["visible"],
            distance_m=result["distance_m"],
            blocking_point=list(result["blocking_point"]) if result["blocking_point"] else None,
            elevation_profile=result["elevation_profile"],
        )

    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_calc_los)


@router.post("/bda/diff", response_model=BdaDiffResponse)
async def compute_bda_diff_endpoint(req: BdaDiffRequest):
    """Compute automated volumetric diff and structural BDA between pre- and post-event DSMs (§4.2)."""
    pre_dsm_path = EXPORTS_DIR / f"{req.request_id_pre}_dsm.npy"
    post_dsm_path = EXPORTS_DIR / f"{req.request_id_post}_dsm.npy"
    pre_meta_path = EXPORTS_DIR / f"{req.request_id_pre}_meta.json"
    post_meta_path = EXPORTS_DIR / f"{req.request_id_post}_meta.json"

    if not pre_dsm_path.exists():
        raise HTTPException(404, f"Pre-event DSM '{req.request_id_pre}' not found.")
    if not post_dsm_path.exists():
        raise HTTPException(404, f"Post-event DSM '{req.request_id_post}' not found.")

    def _calc_bda():
        from app.services.change_detection import compute_volumetric_diff

        dsm_pre = np.load(pre_dsm_path)
        dsm_post = np.load(post_dsm_path)

        # Load GSD from metadata if present
        gsd = 1.0
        if pre_meta_path.exists():
            try:
                import json
                with open(pre_meta_path) as f:
                    meta = json.load(f)
                    if meta.get("transform"):
                        gsd = abs(float(meta["transform"][0]))
            except Exception:
                pass

        # Load evidential uncertainty if present (§1.4 synergy)
        unc_pre = None
        unc_post = None
        pre_risk_path = EXPORTS_DIR / f"{req.request_id_pre}_risk.npy"
        post_risk_path = EXPORTS_DIR / f"{req.request_id_post}_risk.npy"
        if pre_risk_path.exists():
            try:
                unc_pre = np.load(pre_risk_path)
            except Exception:
                pass
        if post_risk_path.exists():
            try:
                unc_post = np.load(post_risk_path)
            except Exception:
                pass

        res = compute_volumetric_diff(
            dsm_pre=dsm_pre,
            dsm_post=dsm_post,
            gsd=gsd,
            threshold_m=req.noise_threshold_m,
            unc_pre=unc_pre,
            unc_post=unc_post,
            auto_coregister=req.auto_coregister,
        )

        structures_out = [
            StructureBdaSchema(
                structure_id=s.structure_id,
                center_col=s.center_col,
                center_row=s.center_row,
                bbox=list(s.bbox),
                footprint_area_m2=s.footprint_area_m2,
                pre_strike_volume_m3=s.pre_strike_volume_m3,
                volume_loss_m3=s.volume_loss_m3,
                collapse_percentage=s.collapse_percentage,
                damage_rating=s.damage_rating,
            )
            for s in res.structures
        ]

        return BdaDiffResponse(
            request_id_pre=req.request_id_pre,
            request_id_post=req.request_id_post,
            cut_volume_m3=res.cut_volume_m3,
            fill_volume_m3=res.fill_volume_m3,
            net_volume_m3=res.net_volume_m3,
            cut_area_m2=res.cut_area_m2,
            fill_area_m2=res.fill_area_m2,
            max_cut_depth_m=res.max_cut_depth_m,
            max_fill_height_m=res.max_fill_height_m,
            noise_threshold_m=res.noise_threshold_m,
            registration_shift_x_m=res.registration_shift_x_m,
            registration_shift_y_m=res.registration_shift_y_m,
            registration_confidence=res.registration_confidence,
            uncertainty_volume_m3=res.uncertainty_volume_m3,
            diff_map_b64=res.diff_map_b64,
            structures=structures_out,
            beta_mode=res.beta_mode,
        )

    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_calc_bda)


@router.post("/route/noe", response_model=NoeRouteResponse)
async def plan_noe_route_endpoint(req: NoeRouteRequest):
    """
    Compute Autonomous Nap-of-the-Earth (NOE) & Terrain-Following 3D Route Corridor (§4.3).
    Combines §3.1 voxel grid collision bounds, §3.2 UNKNOWN occlusion avoidance,
    and §4.1 threat horizon exposure minimization.
    """
    from app.config import NOE_ROUTING_ENABLED
    if not NOE_ROUTING_ENABLED:
        raise HTTPException(403, "NOE routing subsystem is disabled via configuration.")

    def _calc_route():
        from app.services.route_planner import plan_noe_route, sample_dsm_bilinear

        # 1. Resolve DSM array and GSD
        dsm = None
        gsd = req.gsd or 1.0

        if req.request_id:
            dsm_path = EXPORTS_DIR / f"{req.request_id}_dsm.npy"
            if dsm_path.exists():
                dsm = np.load(dsm_path)
            elif req.dsm is not None:
                dsm = np.asarray(req.dsm, dtype=np.float32)
            else:
                raise HTTPException(404, f"Dataset for request_id '{req.request_id}' not found.")

            meta_path = EXPORTS_DIR / f"{req.request_id}_meta.json"
            if meta_path.exists() and req.gsd is None:
                try:
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                        if meta.get("transform"):
                            gsd = abs(float(meta["transform"][0]))
                except Exception:
                    pass
        elif req.dsm is not None:
            dsm = np.asarray(req.dsm, dtype=np.float32)
        else:
            raise HTTPException(400, "Either request_id or dsm array must be provided.")

        H, W = dsm.shape

        # 2. Parse start and goal coordinates
        if len(req.start) == 2:
            sx, sz = float(req.start[0]), float(req.start[1])
            col_f = sx / gsd
            row_f = sz / gsd
            sy = sample_dsm_bilinear(dsm, col_f, row_f) + req.min_clearance_m + 1.0
            start_pt = (sx, sy, sz)
        elif len(req.start) >= 3:
            start_pt = (float(req.start[0]), float(req.start[1]), float(req.start[2]))
        else:
            raise HTTPException(400, "Start coordinate must be [x, y, z] or [x, z]")

        if len(req.goal) == 2:
            gx, gz = float(req.goal[0]), float(req.goal[1])
            col_f = gx / gsd
            row_f = gz / gsd
            gy = sample_dsm_bilinear(dsm, col_f, row_f) + req.min_clearance_m + 1.0
            goal_pt = (gx, gy, gz)
        elif len(req.goal) >= 3:
            goal_pt = (float(req.goal[0]), float(req.goal[1]), float(req.goal[2]))
        else:
            raise HTTPException(400, "Goal coordinate must be [x, y, z] or [x, z]")

        # 3. Parse threat locations
        threats_tuples = []
        for t in req.threats:
            ty = t.y
            if ty is None:
                col_f = t.x / gsd
                row_f = t.z / gsd
                ty = sample_dsm_bilinear(dsm, col_f, row_f) + t.agl_m
            threats_tuples.append((float(t.x), float(ty), float(t.z)))

        # 4. Camera pose for §3.2 unknown space
        camera_pos = tuple(float(c) for c in req.camera_pos) if req.camera_pos else None

        # 5. Execute 3D Kinematic A* Planner
        result = plan_noe_route(
            dsm=dsm,
            start_world=start_pt,
            goal_world=goal_pt,
            threat_positions=threats_tuples,
            gsd=gsd,
            origin=(0.0, 0.0, 0.0),
            min_clearance_m=req.min_clearance_m,
            max_altitude_m=req.max_agl_m,
            weight_exposure=req.weight_exposure,
            weight_unknown=req.weight_unknown,
            weight_altitude=req.weight_altitude,
            weight_distance=req.weight_distance,
            planning_step_m=req.planning_step_m,
            camera_pos=camera_pos,
        )

        waypoints_out = [
            Waypoint3DSchema(
                x=wp.x,
                y=wp.y,
                z=wp.z,
                agl_m=wp.agl_m,
                exposed=wp.exposed,
                heading_deg=wp.heading_deg,
                pitch_deg=wp.pitch_deg,
                speed_mps=wp.speed_mps,
            )
            for wp in result.waypoints
        ]

        return NoeRouteResponse(
            request_id=req.request_id,
            success=len(waypoints_out) > 0,
            waypoints=waypoints_out,
            total_distance_m=result.total_distance_m,
            direct_distance_m=result.direct_distance_m,
            detour_ratio=result.detour_ratio,
            mean_agl_m=result.mean_agl_m,
            min_agl_m=result.min_agl_m,
            max_agl_m=result.max_agl_m,
            exposure_percentage=result.exposure_percentage,
            unknown_percentage=result.unknown_percentage,
            compute_time_ms=result.compute_time_ms,
            status_message=result.status_message,
        )

    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(_calc_route)

