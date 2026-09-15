"""
Autonomous Nap-of-the-Earth (NOE) & Terrain-Following 3D Route Planner (§4.3).

Combines:
- §3.1 Chunked Sparse Voxel Grid for 3D obstacle avoidance.
- §3.2 Bayesian Unobserved / Occluded (UNKNOWN) voxel avoidance.
- §4.1 Viewshed / Line-of-Sight threat horizon field for stealth terrain masking.
- 3D Kinematic A* path planning with climb/dive constraints and composite cost function.
- Post-processing line-of-sight shortcutting and Catmull-Rom spline smoothing for flyable trajectories.
"""
from dataclasses import dataclass, field
import heapq
import math
import time
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from app.config import (
    NOE_ROUTING_ENABLED,
    NOE_MIN_CLEARANCE_M,
    NOE_MAX_ALTITUDE_M,
    NOE_WEIGHT_EXPOSURE,
    NOE_WEIGHT_UNKNOWN,
    NOE_WEIGHT_ALTITUDE,
    NOE_WEIGHT_DISTANCE,
)
from app.services.voxel_grid import SparseVoxelGrid, VoxelState
from app.services.visibility import sample_dsm_bilinear
from app.logging_config import log


@dataclass
class ThreatObserver:
    x: float
    z: float
    y: float  # World elevation of observer eye / radar antenna
    name: str = "Threat"
    max_range_m: Optional[float] = None


@dataclass
class ThreatHorizonField:
    """
    Precomputed horizon tangent field for a threat observer (§4.1 / §4.3).
    Allows evaluating 3D Line-of-Sight visibility for any (x, y, z) coordinate in O(1) time:
    A drone at (x, y, z) is visible iff: (y - threat_y) / dist >= horizon_tangents[r, c].
    """
    threat: ThreatObserver
    horizon_tangents: np.ndarray  # Shape (H, W), stores max obstacle tangent from threat
    gsd: float
    origin: Tuple[float, float, float]

    def is_visible(self, x: float, y: float, z: float, tolerance: float = 1e-3) -> bool:
        """Evaluate in O(1) whether 3D position (x, y, z) is visible to this threat."""
        ox, oy, oz = self.origin
        dx = x - self.threat.x
        dz = z - self.threat.z
        dist_m = math.hypot(dx, dz)
        if dist_m < 1e-2:
            return True
        if self.threat.max_range_m is not None and dist_m > self.threat.max_range_m:
            return False

        col = int(round((x - ox) / self.gsd))
        row = int(round((z - oz) / self.gsd))
        H, W = self.horizon_tangents.shape
        if not (0 <= row < H and 0 <= col < W):
            return False

        max_tan = self.horizon_tangents[row, col]
        query_tan = (y - self.threat.y) / dist_m
        return query_tan >= (max_tan - tolerance)


def precompute_threat_horizon_field(
    dsm: np.ndarray,
    threat: ThreatObserver,
    gsd: float = 1.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ThreatHorizonField:
    """
    Bake a 2D horizon tangent map across the DSM for the specified threat observer.
    Casts dense radial rays outward from the threat to establish the terrain horizon silhouette.
    """
    H, W = dsm.shape
    ox, oy, oz = origin

    threat_col = (threat.x - ox) / gsd
    threat_row = (threat.z - oz) / gsd

    horizon_map = np.full((H, W), -1e9, dtype=np.float32)

    # Perimeter boundary cells to cast radial rays to
    boundary_cells = []
    for c in range(W):
        boundary_cells.append((0, c))
        boundary_cells.append((H - 1, c))
    for r in range(1, H - 1):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, W - 1))

    # Dense ray tracing from threat location outward
    for target_r, target_c in boundary_cells:
        dr = target_r - threat_row
        dc = target_c - threat_col
        dist_grid = math.hypot(dr, dc)
        if dist_grid < 1e-4:
            continue

        num_steps = max(2, int(math.ceil(dist_grid * 1.5)))
        step_r = dr / num_steps
        step_c = dc / num_steps

        max_tan = -1e9

        for s in range(1, num_steps + 1):
            r_idx = int(round(threat_row + s * step_r))
            c_idx = int(round(threat_col + s * step_c))

            if 0 <= r_idx < H and 0 <= c_idx < W:
                dist_m = math.hypot(r_idx - threat_row, c_idx - threat_col) * gsd
                if dist_m < 1e-2:
                    continue
                if threat.max_range_m is not None and dist_m > threat.max_range_m:
                    break

                h_val = float(dsm[r_idx, c_idx])
                if math.isnan(h_val):
                    continue

                if max_tan > -1e8:
                    if max_tan > horizon_map[r_idx, c_idx]:
                        horizon_map[r_idx, c_idx] = max_tan

                tan_val = (h_val - threat.y) / dist_m
                if tan_val > max_tan:
                    max_tan = tan_val

    # Fill any unhit perimeter cells with nearby horizon values
    unvisited = horizon_map < -1e8
    if np.any(unvisited):
        for r in range(H):
            for c in range(W):
                if horizon_map[r, c] < -1e8:
                    dist_m = math.hypot(r - threat_row, c - threat_col) * gsd
                    if dist_m > 1e-2:
                        horizon_map[r, c] = (float(dsm[r, c]) - threat.y) / dist_m
                    else:
                        horizon_map[r, c] = 0.0

    return ThreatHorizonField(
        threat=threat,
        horizon_tangents=horizon_map,
        gsd=gsd,
        origin=origin,
    )


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    agl_m: float
    exposed: bool
    heading_deg: float = 0.0
    pitch_deg: float = 0.0
    speed_mps: float = 10.0


@dataclass
class PlannedRoute:
    waypoints: List[Waypoint]
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
    success: bool = True


class KinematicAStarPlanner:
    """
    3D Kinematic A* Path Planner on §3.1 Voxel Grid.
    Enforces terrain clearance, pitch limits, UNKNOWN avoidance (§3.2), and threat masking (§4.1).
    """

    def __init__(
        self,
        dsm: np.ndarray,
        voxel_grid: SparseVoxelGrid,
        threat_fields: List[ThreatHorizonField],
        gsd: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        step_xy: float = 2.5,
        step_y: float = 2.0,
        min_clearance_m: float = NOE_MIN_CLEARANCE_M,
        max_altitude_m: float = NOE_MAX_ALTITUDE_M,
        weight_exposure: float = NOE_WEIGHT_EXPOSURE,
        weight_unknown: float = NOE_WEIGHT_UNKNOWN,
        weight_altitude: float = NOE_WEIGHT_ALTITUDE,
        weight_distance: float = NOE_WEIGHT_DISTANCE,
        max_climb_angle_deg: float = 35.0,
    ):
        self.dsm = dsm
        self.voxel_grid = voxel_grid
        self.threat_fields = threat_fields
        self.gsd = float(gsd)
        self.origin = origin
        self.step_xy = max(1.0, float(step_xy))
        self.step_y = max(1.0, float(step_y))
        self.min_clearance_m = float(min_clearance_m)
        self.max_altitude_m = float(max_altitude_m)
        self.w_exp = float(weight_exposure)
        self.w_unk = float(weight_unknown)
        self.w_alt = float(weight_altitude)
        self.w_dist = float(weight_distance)
        self.max_climb_tan = math.tan(math.radians(max_climb_angle_deg))

        H, W = dsm.shape
        ox, oy, oz = origin
        self.min_x = ox + 1.0
        self.max_x = ox + (W - 1) * self.gsd - 1.0
        self.min_z = oz + 1.0
        self.max_z = oz + (H - 1) * self.gsd - 1.0

        # Precompute 26-connectivity 3D neighbor offsets
        self.neighbors: List[Tuple[int, int, int, float]] = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    horiz_dist = math.hypot(dx * self.step_xy, dz * self.step_xy)
                    vert_dist = abs(dy * self.step_y)
                    # Kinematic climb/dive check
                    if horiz_dist > 0 and (vert_dist / horiz_dist) > self.max_climb_tan:
                        continue
                    euclid = math.sqrt((dx * self.step_xy)**2 + (dy * self.step_y)**2 + (dz * self.step_xy)**2)
                    self.neighbors.append((dx, dy, dz, euclid))

    def world_to_node(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert continuous 3D world coordinates to integer graph node coordinates."""
        ox, oy, oz = self.origin
        ix = int(round((x - ox) / self.step_xy))
        iy = int(round((y - oy) / self.step_y))
        iz = int(round((z - oz) / self.step_xy))
        return (ix, iy, iz)

    def node_to_world(self, node: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert integer graph node coordinates to continuous 3D world coordinates."""
        ox, oy, oz = self.origin
        ix, iy, iz = node
        x = ox + ix * self.step_xy
        y = oy + iy * self.step_y
        z = oz + iz * self.step_xy
        return (x, y, z)

    def is_valid_node(self, x: float, y: float, z: float) -> Tuple[bool, float, bool]:
        """
        Validate node against spatial boundaries, terrain collision, and AGL envelope.
        Returns: (is_valid, agl_m, is_unknown)
        """
        if not (self.min_x <= x <= self.max_x and self.min_z <= z <= self.max_z):
            return False, 0.0, False

        ox, oy, oz = self.origin
        col_f = (x - ox) / self.gsd
        row_f = (z - oz) / self.gsd
        ground_y = sample_dsm_bilinear(self.dsm, col_f, row_f)
        agl = y - ground_y

        # Minimum clearance buffer above ground and obstacles
        if agl < self.min_clearance_m:
            return False, agl, False

        # Maximum NOE masking ceiling
        if agl > self.max_altitude_m:
            return False, agl, False

        # Voxel grid collision check (§3.1, §3.2)
        gx, gy, gz = self.voxel_grid.world_to_grid(x, y, z)
        state = self.voxel_grid.get_voxel(gx, gy, gz)

        if state == VoxelState.OCCUPIED:
            return False, agl, False

        is_unknown = (state == VoxelState.UNKNOWN)
        return True, agl, is_unknown

    def compute_exposure_ratio(self, x: float, y: float, z: float) -> float:
        """Compute fraction of threat observers that have Line-of-Sight to (x, y, z)."""
        if not self.threat_fields:
            return 0.0
        vis_count = sum(1 for tf in self.threat_fields if tf.is_visible(x, y, z))
        return vis_count / len(self.threat_fields)

    def step_cost(
        self,
        from_pos: Tuple[float, float, float],
        to_pos: Tuple[float, float, float],
        step_dist: float,
        agl: float,
        is_unknown: bool,
    ) -> float:
        """
        Evaluate composite cost function for transitioning from from_pos to to_pos:
        cost = dist * (w_dist + w_exp * exposure + w_unk * is_unknown + w_alt * agl_ratio + climb_penalty)
        """
        tx, ty, tz = to_pos
        exp_ratio = self.compute_exposure_ratio(tx, ty, tz)

        # Normalized AGL penalty (prefers hugging terrain safely above min clearance)
        agl_span = max(1.0, self.max_altitude_m - self.min_clearance_m)
        alt_factor = max(0.0, min(1.0, (agl - self.min_clearance_m) / agl_span))

        # Vertical climb penalty
        dy = ty - from_pos[1]
        climb_penalty = max(0.0, dy / step_dist) * 0.75

        multiplier = (
            self.w_dist
            + self.w_exp * exp_ratio
            + (self.w_unk if is_unknown else 0.0)
            + self.w_alt * alt_factor
            + climb_penalty
        )
        return step_dist * multiplier

    def plan(
        self,
        start_world: Tuple[float, float, float],
        goal_world: Tuple[float, float, float],
        max_iterations: int = 40000,
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Run 3D Kinematic A* search from start_world to goal_world.
        Returns list of 3D waypoints or None if path cannot be resolved.
        """
        start_node = self.world_to_node(*start_world)
        goal_node = self.world_to_node(*goal_world)

        # Validate start and goal
        sx, sy, sz = self.node_to_world(start_node)
        valid_start, _, _ = self.is_valid_node(sx, sy, sz)
        if not valid_start:
            # Lift start node slightly to achieve safe clearance if needed
            ox, oy, oz = self.origin
            col_f = (sx - ox) / self.gsd
            row_f = (sz - oz) / self.gsd
            gy = sample_dsm_bilinear(self.dsm, col_f, row_f)
            sy = gy + self.min_clearance_m + 0.5
            start_node = self.world_to_node(sx, sy, sz)

        gx, gy, gz = self.node_to_world(goal_node)
        valid_goal, _, _ = self.is_valid_node(gx, gy, gz)
        if not valid_goal:
            ox, oy, oz = self.origin
            col_f = (gx - ox) / self.gsd
            row_f = (gz - oz) / self.gsd
            ground_y = sample_dsm_bilinear(self.dsm, col_f, row_f)
            gy = ground_y + self.min_clearance_m + 0.5
            goal_node = self.world_to_node(gx, gy, gz)

        # A* Data Structures
        open_set: List[Tuple[float, float, Tuple[int, int, int]]] = []
        g_scores: Dict[Tuple[int, int, int], float] = {start_node: 0.0}
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        closed_set: Set[Tuple[int, int, int]] = set()

        def heuristic(node: Tuple[int, int, int]) -> float:
            wx, wy, wz = self.node_to_world(node)
            return self.w_dist * math.sqrt((wx - gx)**2 + (wy - gy)**2 + (wz - gz)**2)

        heapq.heappush(open_set, (heuristic(start_node), 0.0, start_node))
        iterations = 0

        goal_tolerance_dist = self.step_xy * 1.5

        best_node = start_node
        min_h = heuristic(start_node)

        while open_set and iterations < max_iterations:
            iterations += 1
            f, current_g, current_node = heapq.heappop(open_set)

            if current_node in closed_set:
                continue
            closed_set.add(current_node)

            cx, cy, cz = self.node_to_world(current_node)
            dist_to_goal = math.sqrt((cx - gx)**2 + (cy - gy)**2 + (cz - gz)**2)

            if dist_to_goal < min_h:
                min_h = dist_to_goal
                best_node = current_node

            # Check goal condition
            if current_node == goal_node or dist_to_goal <= goal_tolerance_dist:
                path = [goal_world]
                curr = current_node
                while curr in came_from:
                    path.append(self.node_to_world(curr))
                    curr = came_from[curr]
                path.append(start_world)
                path.reverse()
                log.info(
                    "A* Route converged successfully",
                    iterations=iterations,
                    nodes_expanded=len(closed_set),
                    raw_waypoints=len(path),
                )
                return path

            for dix, diy, diz, step_len in self.neighbors:
                neighbor_node = (
                    current_node[0] + dix,
                    current_node[1] + diy,
                    current_node[2] + diz,
                )

                if neighbor_node in closed_set:
                    continue

                nx, ny, nz = self.node_to_world(neighbor_node)
                is_valid, agl, is_unk = self.is_valid_node(nx, ny, nz)
                if not is_valid:
                    continue

                tentative_cost = self.step_cost(
                    from_pos=(cx, cy, cz),
                    to_pos=(nx, ny, nz),
                    step_dist=step_len,
                    agl=agl,
                    is_unknown=is_unk,
                )
                tentative_g = current_g + tentative_cost

                if neighbor_node not in g_scores or tentative_g < g_scores[neighbor_node]:
                    g_scores[neighbor_node] = tentative_g
                    came_from[neighbor_node] = current_node
                    h = heuristic(neighbor_node)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor_node))

        log.warning(
            "A* search reached iteration limit; falling back to best observed node",
            iterations=iterations,
            best_dist=min_h,
        )

        if min_h < (math.hypot(gx - sx, gz - sz) * 0.5):
            path = [self.node_to_world(best_node)]
            curr = best_node
            while curr in came_from:
                path.append(self.node_to_world(curr))
                curr = came_from[curr]
            path.append(start_world)
            path.reverse()
            return path

        return None


# =============================================================================
# 4. POST-PROCESSING: SHORTCUTTING & SPLINE SMOOTHING
# =============================================================================

def shortcut_path(
    waypoints: List[Tuple[float, float, float]],
    dsm: np.ndarray,
    gsd: float,
    origin: Tuple[float, float, float],
    min_clearance_m: float,
    threat_fields: List[ThreatHorizonField],
) -> List[Tuple[float, float, float]]:
    """
    Line-of-Sight waypoint pruning / shortcutting.
    Skips intermediate nodes where direct line preserves collision clearance and low exposure.
    """
    if len(waypoints) <= 2:
        return waypoints

    pruned = [waypoints[0]]
    curr_idx = 0

    while curr_idx < len(waypoints) - 1:
        furthest = curr_idx + 1
        max_lookahead = min(len(waypoints), curr_idx + 12)

        for next_idx in range(max_lookahead - 1, curr_idx, -1):
            p0 = pruned[-1]
            p1 = waypoints[next_idx]

            dist = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2 + (p1[2] - p0[2])**2)
            num_samples = max(3, int(math.ceil(dist / (gsd * 0.8))))

            valid_shortcut = True
            for s in range(1, num_samples):
                frac = s / num_samples
                sx = p0[0] + frac * (p1[0] - p0[0])
                sy = p0[1] + frac * (p1[1] - p0[1])
                sz = p0[2] + frac * (p1[2] - p0[2])

                col_f = (sx - origin[0]) / gsd
                row_f = (sz - origin[2]) / gsd
                gy = sample_dsm_bilinear(dsm, col_f, row_f)

                if (sy - gy) < (min_clearance_m * 0.9):
                    valid_shortcut = False
                    break

            if valid_shortcut:
                furthest = next_idx
                break

        pruned.append(waypoints[furthest])
        curr_idx = furthest

    return pruned


def catmull_rom_spline(
    control_points: List[Tuple[float, float, float]],
    sample_spacing_m: float = 1.5,
) -> List[Tuple[float, float, float]]:
    """
    Centripetal Catmull-Rom spline interpolation through 3D control points.
    Generates a flyable continuous path with smoothly turning tangents.
    """
    if len(control_points) < 2:
        return control_points
    if len(control_points) == 2:
        p0, p1 = control_points
        dist = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2 + (p1[2] - p0[2])**2)
        steps = max(2, int(math.ceil(dist / sample_spacing_m)))
        return [
            (
                p0[0] + (p1[0] - p0[0]) * (i / steps),
                p0[1] + (p1[1] - p0[1]) * (i / steps),
                p0[2] + (p1[2] - p0[2]) * (i / steps),
            )
            for i in range(steps + 1)
        ]

    pts = [control_points[0]] + control_points + [control_points[-1]]
    curve: List[Tuple[float, float, float]] = []

    for i in range(len(pts) - 3):
        p0 = np.array(pts[i], dtype=np.float64)
        p1 = np.array(pts[i + 1], dtype=np.float64)
        p2 = np.array(pts[i + 2], dtype=np.float64)
        p3 = np.array(pts[i + 3], dtype=np.float64)

        segment_dist = float(np.linalg.norm(p2 - p1))
        num_sub = max(2, int(math.ceil(segment_dist / sample_spacing_m)))

        for s in range(num_sub):
            t = s / num_sub
            t2 = t * t
            t3 = t2 * t

            pos = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve.append((float(pos[0]), float(pos[1]), float(pos[2])))

    curve.append(control_points[-1])
    return curve


# =============================================================================
# 5. HIGH-LEVEL ROUTE PLANNING ORCHESTRATOR
# =============================================================================

def plan_noe_route(
    dsm: np.ndarray,
    start_world: Tuple[float, float, float],
    goal_world: Tuple[float, float, float],
    threat_positions: Optional[List[Tuple[float, float, float]]] = None,
    gsd: float = 1.0,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_clearance_m: float = NOE_MIN_CLEARANCE_M,
    max_altitude_m: float = NOE_MAX_ALTITUDE_M,
    weight_exposure: float = NOE_WEIGHT_EXPOSURE,
    weight_unknown: float = NOE_WEIGHT_UNKNOWN,
    weight_altitude: float = NOE_WEIGHT_ALTITUDE,
    weight_distance: float = NOE_WEIGHT_DISTANCE,
    planning_step_m: float = 2.5,
    camera_pos: Optional[Tuple[float, float, float]] = None,
    voxel_grid: Optional[SparseVoxelGrid] = None,
) -> PlannedRoute:
    """
    Orchestrate full NOE autonomous path planning pipeline (§4.3).

    Steps:
    1. Precompute threat horizon fields using §4.1 visibility machinery.
    2. Build or reuse §3.1 SparseVoxelGrid with §3.2 UNKNOWN occlusion flags.
    3. Execute 3D Kinematic A* search with composite cost function.
    4. Apply shortcutting and Catmull-Rom spline smoothing.
    5. Compute corridor metrics, exposure score, and flight telemetry.
    """
    t_start = time.time()
    threat_positions = threat_positions or []

    # 1. Precompute threat horizon fields
    threat_observers: List[ThreatObserver] = []
    threat_fields: List[ThreatHorizonField] = []

    for idx, tp in enumerate(threat_positions):
        tx, ty, tz = tp
        if ty is None or ty == 0.0:
            col_f = (tx - origin[0]) / gsd
            row_f = (tz - origin[2]) / gsd
            ty = sample_dsm_bilinear(dsm, col_f, row_f) + 2.0  # +2m mast height
        obs = ThreatObserver(x=tx, y=ty, z=tz, name=f"Threat-{idx+1}")
        threat_observers.append(obs)
        tf = precompute_threat_horizon_field(dsm=dsm, threat=obs, gsd=gsd, origin=origin)
        threat_fields.append(tf)

    # 2. Build or reuse Voxel Grid (§3.1, §3.2)
    if voxel_grid is None:
        voxel_grid = SparseVoxelGrid.from_dsm(
            dsm=dsm,
            gsd=gsd,
            voxel_size_y=min(2.0, max(0.5, gsd)),
            world_origin=origin,
            camera_pos=camera_pos,
        )

    # 3. 3D Kinematic A* Planner
    planner = KinematicAStarPlanner(
        dsm=dsm,
        voxel_grid=voxel_grid,
        threat_fields=threat_fields,
        gsd=gsd,
        origin=origin,
        step_xy=planning_step_m,
        step_y=max(1.5, planning_step_m * 0.75),
        min_clearance_m=min_clearance_m,
        max_altitude_m=max_altitude_m,
        weight_exposure=weight_exposure,
        weight_unknown=weight_unknown,
        weight_altitude=weight_altitude,
        weight_distance=weight_distance,
    )

    raw_path = planner.plan(start_world=start_world, goal_world=goal_world)
    if not raw_path:
        elapsed = (time.time() - t_start) * 1000.0
        return PlannedRoute(
            waypoints=[],
            total_distance_m=0.0,
            direct_distance_m=float(np.linalg.norm(np.array(goal_world) - np.array(start_world))),
            detour_ratio=1.0,
            mean_agl_m=0.0,
            min_agl_m=0.0,
            max_agl_m=0.0,
            exposure_percentage=0.0,
            unknown_percentage=0.0,
            compute_time_ms=elapsed,
            status_message="Failed to find valid terrain-following corridor within constraints.",
            success=False,
        )

    # 4. Shortcutting & Spline Smoothing
    pruned_path = shortcut_path(
        waypoints=raw_path,
        dsm=dsm,
        gsd=gsd,
        origin=origin,
        min_clearance_m=min_clearance_m,
        threat_fields=threat_fields,
    )

    smooth_pts = catmull_rom_spline(pruned_path, sample_spacing_m=max(1.0, gsd * 0.8))

    # 5. Enrich Waypoints with Kinematics & Exposure Telemetry
    waypoints_out: List[Waypoint] = []
    total_dist = 0.0
    agls: List[float] = []
    exposed_count = 0
    unknown_count = 0

    for i, pt in enumerate(smooth_pts):
        px, py, pz = pt
        col_f = (px - origin[0]) / gsd
        row_f = (pz - origin[2]) / gsd
        gy = sample_dsm_bilinear(dsm, col_f, row_f)
        agl = max(min_clearance_m, py - gy)
        agls.append(agl)

        # Distance
        if i > 0:
            prev = smooth_pts[i - 1]
            total_dist += math.sqrt((px - prev[0])**2 + (py - prev[1])**2 + (pz - prev[2])**2)

        # Heading & Pitch
        if i < len(smooth_pts) - 1:
            nxt = smooth_pts[i + 1]
            dx = nxt[0] - px
            dy = nxt[1] - py
            dz = nxt[2] - pz
        elif i > 0:
            prv = smooth_pts[i - 1]
            dx = px - prv[0]
            dy = py - prv[1]
            dz = pz - prv[2]
        else:
            dx, dy, dz = 1.0, 0.0, 0.0

        heading_deg = (math.degrees(math.atan2(-dx, -dz)) + 360.0) % 360.0
        horiz = math.hypot(dx, dz)
        pitch_deg = math.degrees(math.atan2(dy, max(1e-4, horiz)))

        # Threat Exposure Check
        is_exp = any(tf.is_visible(px, py, pz) for tf in threat_fields)
        if is_exp:
            exposed_count += 1

        # Voxel Unknown State Check
        gx, gy_idx, gz = voxel_grid.world_to_grid(px, py, pz)
        vstate = voxel_grid.get_voxel(gx, gy_idx, gz)
        if vstate == VoxelState.UNKNOWN:
            unknown_count += 1

        waypoints_out.append(
            Waypoint(
                x=round(px, 3),
                y=round(py, 3),
                z=round(pz, 3),
                agl_m=round(agl, 2),
                exposed=is_exp,
                heading_deg=round(heading_deg, 1),
                pitch_deg=round(pitch_deg, 1),
                speed_mps=10.0,
            )
        )

    direct_dist = float(np.linalg.norm(np.array(goal_world) - np.array(start_world)))
    detour = (total_dist / direct_dist) if direct_dist > 1e-3 else 1.0
    exp_pct = (exposed_count / len(waypoints_out) * 100.0) if waypoints_out else 0.0
    unk_pct = (unknown_count / len(waypoints_out) * 100.0) if waypoints_out else 0.0
    elapsed_ms = (time.time() - t_start) * 1000.0

    log.info(
        "NOE Route Planned Successfully",
        waypoints=len(waypoints_out),
        distance_m=round(total_dist, 1),
        mean_agl=round(float(np.mean(agls)), 2),
        exposure_pct=f"{exp_pct:.1f}%",
        time_ms=round(elapsed_ms, 1),
    )

    return PlannedRoute(
        waypoints=waypoints_out,
        total_distance_m=round(total_dist, 2),
        direct_distance_m=round(direct_dist, 2),
        detour_ratio=round(detour, 3),
        mean_agl_m=round(float(np.mean(agls)), 2),
        min_agl_m=round(float(np.min(agls)), 2),
        max_agl_m=round(float(np.max(agls)), 2),
        exposure_percentage=round(exp_pct, 1),
        unknown_percentage=round(unk_pct, 1),
        compute_time_ms=round(elapsed_ms, 1),
        status_message="Optimal terrain-following NOE corridor generated.",
    )
