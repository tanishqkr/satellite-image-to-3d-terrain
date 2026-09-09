from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    gpu: str
    vram_gb: float


class InferenceResponse(BaseModel):
    request_id: str
    heightmap_b64: str
    rgb_b64: str
    normal_map_b64: Optional[str] = None
    dsm_colorized_b64: str
    mesh_stats: dict
    calibration: dict
    dsm_raw: list  # Backward-compatible compact grid
    dsm_raw_b64: Optional[str] = None  # Full-fidelity Float32 binary buffer
    is_georef: bool
    crs: Optional[str] = None
    confidence_mean: Optional[float] = None
    uncertainty_map_b64: Optional[str] = None
    uncertainty_stats: Optional[dict] = None
    rectification: Optional[dict] = None
    inference_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class ViewshedRequest(BaseModel):
    request_id: str
    observer_col: float
    observer_row: float
    observer_agl: float = 2.0
    max_range_m: Optional[float] = None
    target_height_m: float = 0.0


class ViewshedResponse(BaseModel):
    request_id: str
    observer_elevation_msl: float
    observer_agl: float
    visible_cells: int
    total_cells: int
    visible_fraction: float
    viewshed_b64: Optional[str] = None


class LineOfSightRequest(BaseModel):
    request_id: str
    p0: list  # [x0, y0, z0]
    p1: list  # [x1, y1, z1]
    sample_step_m: Optional[float] = None
    tolerance_m: float = 0.05


class LineOfSightResponse(BaseModel):
    request_id: str
    visible: bool
    distance_m: float
    blocking_point: Optional[list] = None
    elevation_profile: list


class StructureBdaSchema(BaseModel):
    structure_id: int
    center_col: float
    center_row: float
    bbox: list
    footprint_area_m2: float
    pre_strike_volume_m3: float
    volume_loss_m3: float
    collapse_percentage: float
    damage_rating: str


class BdaDiffRequest(BaseModel):
    request_id_pre: str
    request_id_post: str
    noise_threshold_m: float = 0.20
    auto_coregister: bool = True


class BdaDiffResponse(BaseModel):
    request_id_pre: str
    request_id_post: str
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
    uncertainty_volume_m3: Optional[float] = None
    diff_map_b64: Optional[str] = None
    structures: list[StructureBdaSchema] = []
    beta_mode: bool = True


class ThreatLocation(BaseModel):
    x: float
    z: float
    y: Optional[float] = None
    agl_m: float = 2.0
    name: Optional[str] = "Threat Observer"
    range_m: Optional[float] = None


class Waypoint3DSchema(BaseModel):
    x: float
    y: float
    z: float
    agl_m: float
    exposed: bool
    heading_deg: float
    pitch_deg: float
    speed_mps: float


class NoeRouteRequest(BaseModel):
    request_id: Optional[str] = None
    dsm: Optional[list] = None
    gsd: Optional[float] = None
    start: list  # [x, y, z] or [x, z]
    goal: list   # [x, y, z] or [x, z]
    threats: list[ThreatLocation] = []
    min_clearance_m: float = 3.0
    max_agl_m: float = 35.0
    weight_exposure: float = 4.0
    weight_unknown: float = 6.0
    weight_altitude: float = 1.5
    weight_distance: float = 1.0
    planning_step_m: float = 2.5
    camera_pos: Optional[list] = None


class NoeRouteResponse(BaseModel):
    request_id: Optional[str] = None
    success: bool
    waypoints: list[Waypoint3DSchema] = []
    total_distance_m: float
    direct_distance_m: float
    detour_ratio: float
    mean_agl_m: float
    min_agl_m: float
    max_agl_m: float
    exposure_percentage: float
    unknown_percentage: float
    compute_time_ms: float
    status_message: str

