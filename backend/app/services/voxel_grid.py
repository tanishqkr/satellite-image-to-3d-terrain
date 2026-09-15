"""
Sparse Voxel Data Structure (§3.1 Phase 1).
Chunked 3D sparse grid (Minecraft-style 16x16x16 chunks) replacing 2.5D heightfields.
Features:
- Tri-state cell schema (FREE=0, OCCUPIED=1, UNKNOWN=2) designed jointly for §3.2.
- Sparse memory allocation: empty/sky chunks consume 0 bytes.
- Fast vectorized DSM column pooling (solid ground up to observed surface, empty above).
- Multi-resolution LOD generation for large-area rendering and routing acceleration (§4.3).
- Spatial indexing, collision bounds, and world-to-grid coordinate transforms.
"""
from enum import IntEnum
from typing import Tuple, Dict, Optional, List
import numpy as np

from app.logging_config import log


class VoxelState(IntEnum):
    FREE = 0       # Traversed or open air
    OCCUPIED = 1   # Solid observed terrain or building structure
    UNKNOWN = 2    # Occluded or unobserved volume (§3.2)


class SparseVoxelGrid:
    """
    Sparse 3D voxel grid stored as a hashmap of 16x16x16 uint8 chunks.
    Indexed by chunk coordinate (cx, cy, cz).
    """
    CHUNK_SIZE = 16

    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Parameters:
            voxel_size: (dx, dy, dz) physical dimensions of one voxel in meters.
            origin: (ox, oy, oz) world coordinates of grid cell (0, 0, 0).
        """
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.origin = tuple(float(o) for o in origin)
        # Map: (cx, cy, cz) -> np.ndarray of shape (16, 16, 16) uint8
        self.chunks: Dict[Tuple[int, int, int], np.ndarray] = {}

    def grid_to_chunk_coord(self, gx: int, gy: int, gz: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Convert global voxel indices (gx, gy, gz) to (chunk_coord, local_coord)."""
        S = self.CHUNK_SIZE
        cx, lx = divmod(gx, S)
        cy, ly = divmod(gy, S)
        cz, lz = divmod(gz, S)
        return (cx, cy, cz), (lx, ly, lz)

    def world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates (meters) to integer grid indices."""
        dx, dy, dz = self.voxel_size
        ox, oy, oz = self.origin
        gx = int(np.floor((x - ox) / dx))
        gy = int(np.floor((y - oy) / dy))
        gz = int(np.floor((z - oz) / dz))
        return gx, gy, gz

    def grid_to_world(self, gx: int, gy: int, gz: int) -> Tuple[float, float, float]:
        """Convert integer grid indices to world center coordinates (meters)."""
        dx, dy, dz = self.voxel_size
        ox, oy, oz = self.origin
        wx = ox + (gx + 0.5) * dx
        wy = oy + (gy + 0.5) * dy
        wz = oz + (gz + 0.5) * dz
        return wx, wy, wz

    def get_voxel(self, gx: int, gy: int, gz: int) -> int:
        """Get state of voxel at (gx, gy, gz). Returns VoxelState.FREE if chunk unallocated."""
        c_coord, (lx, ly, lz) = self.grid_to_chunk_coord(gx, gy, gz)
        chunk = self.chunks.get(c_coord)
        if chunk is None:
            return VoxelState.FREE
        return int(chunk[lx, ly, lz])

    def set_voxel(self, gx: int, gy: int, gz: int, state: int):
        """Set state of voxel at (gx, gy, gz), allocating chunk on demand."""
        c_coord, (lx, ly, lz) = self.grid_to_chunk_coord(gx, gy, gz)
        chunk = self.chunks.get(c_coord)
        if chunk is None:
            if state == VoxelState.FREE:
                return  # Do not allocate chunk for empty space
            chunk = np.zeros((self.CHUNK_SIZE, self.CHUNK_SIZE, self.CHUNK_SIZE), dtype=np.uint8)
            self.chunks[c_coord] = chunk
        chunk[lx, ly, lz] = state

    @classmethod
    def from_dsm(
        cls,
        dsm: np.ndarray,
        gsd: float = 1.0,
        voxel_size_y: float = 1.0,
        base_elevation: Optional[float] = None,
        world_origin: Optional[Tuple[float, float, float]] = None,
        camera_pos: Optional[Tuple[float, float, float]] = None,
        occlusion_mask: Optional[np.ndarray] = None,
    ) -> "SparseVoxelGrid":
        """
        Construct sparse voxel grid from a 2D DSM (H, W).
        Populates solid columns up to observed height:
        voxels for 0 <= y <= height_idx are OCCUPIED (or UNKNOWN if occluded from camera),
        voxels above are FREE.
        """
        from app.config import OCCUPANCY_UNKNOWN_ENABLED

        H, W = dsm.shape
        safe_gsd = max(float(gsd), 0.1) if gsd is not None else 1.0
        safe_v_res = max(float(voxel_size_y), 0.1)

        dsm_min = float(np.nanmin(dsm)) if base_elevation is None else base_elevation
        dsm_max = float(np.nanmax(dsm))

        origin = world_origin or (0.0, dsm_min, 0.0)
        grid = cls(voxel_size=(safe_gsd, safe_v_res, safe_gsd), origin=origin)

        # Compute occlusion mask if camera position provided and enabled
        if occlusion_mask is None and camera_pos is not None and OCCUPANCY_UNKNOWN_ENABLED:
            from app.services.visibility import compute_camera_occlusion_mask
            occlusion_mask = compute_camera_occlusion_mask(
                dsm=dsm,
                camera_pos=camera_pos,
                gsd=safe_gsd,
                origin=origin,
            )

        # Height index in voxels for each (row, col)
        # row -> Z, col -> X; ensure minimum 1-voxel thickness for ground surface
        height_indices = np.maximum(1, np.ceil((dsm - dsm_min) / safe_v_res).astype(np.int32))

        S = cls.CHUNK_SIZE
        num_chunk_x = int(np.ceil(W / S))
        num_chunk_z = int(np.ceil(H / S))
        max_h_idx = int(height_indices.max())
        num_chunk_y = int(np.ceil((max_h_idx + 1) / S))

        # Populate chunks block-by-block for optimal memory and cache locality
        for cz in range(num_chunk_z):
            z0 = cz * S
            z1 = min(H, z0 + S)
            len_z = z1 - z0

            for cx in range(num_chunk_x):
                x0 = cx * S
                x1 = min(W, x0 + S)
                len_x = x1 - x0

                patch_h = height_indices[z0:z1, x0:x1]
                max_patch_h = int(patch_h.max())
                if max_patch_h < 0:
                    continue

                patch_occ = None
                if occlusion_mask is not None:
                    patch_occ = occlusion_mask[z0:z1, x0:x1]

                max_cy = max_patch_h // S

                for cy in range(max_cy + 1):
                    y0 = cy * S
                    y1 = min(max_patch_h + 1, y0 + S)
                    if y1 <= y0:
                        continue

                    chunk = np.zeros((S, S, S), dtype=np.uint8)
                    # For each local (lz, lx), fill vertical range [0, min(patch_h - y0, 16)]
                    for lz in range(len_z):
                        for lx in range(len_x):
                            h_top = patch_h[lz, lx] - y0
                            if h_top > 0:
                                fill_state = (
                                    VoxelState.UNKNOWN
                                    if patch_occ is not None and patch_occ[lz, lx]
                                    else VoxelState.OCCUPIED
                                )
                                chunk[lx, :min(h_top, S), lz] = fill_state

                    if np.any(chunk > 0):
                        grid.chunks[(cx, cy, cz)] = chunk

        log.info(
            "Sparse voxel grid built from DSM",
            allocated_chunks=len(grid.chunks),
            grid_extents=f"{W}x{max_h_idx+1}x{H}",
            voxel_size=grid.voxel_size,
            has_occlusion=occlusion_mask is not None,
        )
        return grid

    def apply_occlusion_mask(self, occlusion_mask: np.ndarray):
        """
        Post-processing pass to update existing grid columns marked as occluded
        to VoxelState.UNKNOWN (§3.2).
        """
        H, W = occlusion_mask.shape
        S = self.CHUNK_SIZE
        for (cx, cy, cz), chunk in self.chunks.items():
            z0 = cz * S
            z1 = min(H, z0 + S)
            x0 = cx * S
            x1 = min(W, x0 + S)

            for lz in range(z1 - z0):
                gz = z0 + lz
                for lx in range(x1 - x0):
                    gx = x0 + lx
                    if occlusion_mask[gz, gx]:
                        # Where voxel is currently OCCUPIED, convert to UNKNOWN
                        occ_mask = (chunk[lx, :, lz] == VoxelState.OCCUPIED)
                        chunk[lx, occ_mask, lz] = VoxelState.UNKNOWN

    def get_lod(self, level: int = 1) -> "SparseVoxelGrid":
        """
        Generate coarser multi-resolution Level-of-Detail (LOD).
        Downsamples by 2^level using maximum pooling.
        """
        if level <= 0:
            return self

        factor = 2 ** level
        lod_voxel_size = (
            self.voxel_size[0] * factor,
            self.voxel_size[1] * factor,
            self.voxel_size[2] * factor,
        )
        lod_grid = SparseVoxelGrid(voxel_size=lod_voxel_size, origin=self.origin)

        # Downsample existing allocated chunks
        for (cx, cy, cz), chunk in self.chunks.items():
            # In-chunk downsample: (16, 16, 16) -> (16/factor, 16/factor, 16/factor)
            if factor <= self.CHUNK_SIZE:
                new_shape = (
                    self.CHUNK_SIZE // factor,
                    self.CHUNK_SIZE // factor,
                    self.CHUNK_SIZE // factor,
                )
                # Reshape to (new_x, factor, new_y, factor, new_z, factor) and max-reduce
                view = chunk.reshape(
                    new_shape[0], factor,
                    new_shape[1], factor,
                    new_shape[2], factor,
                )
                coarse = view.max(axis=(1, 3, 5))
                # Set into lod_grid
                base_gx = cx * (self.CHUNK_SIZE // factor)
                base_gy = cy * (self.CHUNK_SIZE // factor)
                base_gz = cz * (self.CHUNK_SIZE // factor)

                for lx in range(new_shape[0]):
                    for ly in range(new_shape[1]):
                        for lz in range(new_shape[2]):
                            val = coarse[lx, ly, lz]
                            if val > 0:
                                lod_grid.set_voxel(base_gx + lx, base_gy + ly, base_gz + lz, val)

        return lod_grid

    def get_occupied_coords(self, as_world: bool = False) -> np.ndarray:
        """
        Return (N, 3) numpy array of all occupied voxel coordinates.
        Useful for downstream 3D collision, viewshed, and autonomous routing (§4.3).
        """
        all_coords: List[np.ndarray] = []
        S = self.CHUNK_SIZE

        for (cx, cy, cz), chunk in self.chunks.items():
            lx, ly, lz = np.where(chunk == VoxelState.OCCUPIED)
            if len(lx) == 0:
                continue
            gx = cx * S + lx
            gy = cy * S + ly
            gz = cz * S + lz
            all_coords.append(np.column_stack([gx, gy, gz]))

        if not all_coords:
            return np.empty((0, 3), dtype=np.float32 if as_world else np.int32)

        grid_coords = np.vstack(all_coords)
        if not as_world:
            return grid_coords

        dx, dy, dz = self.voxel_size
        ox, oy, oz = self.origin
        world_coords = np.empty_like(grid_coords, dtype=np.float32)
        world_coords[:, 0] = ox + (grid_coords[:, 0] + 0.5) * dx
        world_coords[:, 1] = oy + (grid_coords[:, 1] + 0.5) * dy
        world_coords[:, 2] = oz + (grid_coords[:, 2] + 0.5) * dz
        return world_coords

    def get_unknown_coords(self, as_world: bool = False) -> np.ndarray:
        """
        Return (N, 3) numpy array of all UNKNOWN (occluded/unobserved) voxel coordinates (§3.2).
        """
        all_coords: List[np.ndarray] = []
        S = self.CHUNK_SIZE

        for (cx, cy, cz), chunk in self.chunks.items():
            lx, ly, lz = np.where(chunk == VoxelState.UNKNOWN)
            if len(lx) == 0:
                continue
            gx = cx * S + lx
            gy = cy * S + ly
            gz = cz * S + lz
            all_coords.append(np.column_stack([gx, gy, gz]))

        if not all_coords:
            return np.empty((0, 3), dtype=np.float32 if as_world else np.int32)

        grid_coords = np.vstack(all_coords)
        if not as_world:
            return grid_coords

        dx, dy, dz = self.voxel_size
        ox, oy, oz = self.origin
        world_coords = np.empty_like(grid_coords, dtype=np.float32)
        world_coords[:, 0] = ox + (grid_coords[:, 0] + 0.5) * dx
        world_coords[:, 1] = oy + (grid_coords[:, 1] + 0.5) * dy
        world_coords[:, 2] = oz + (grid_coords[:, 2] + 0.5) * dz
        return world_coords

    def get_memory_stats(self, dense_bounding_shape: Optional[Tuple[int, int, int]] = None) -> dict:
        """Calculate sparse memory usage and savings vs full dense 3D grid."""
        allocated_chunks = len(self.chunks)
        voxel_bytes = allocated_chunks * (self.CHUNK_SIZE ** 3)  # 1 byte per voxel in chunk

        if dense_bounding_shape is not None:
            nx, ny, nz = dense_bounding_shape
            dense_voxels = nx * ny * nz
        elif allocated_chunks > 0:
            coords = np.array(list(self.chunks.keys()))
            min_c = coords.min(axis=0)
            max_c = coords.max(axis=0)
            dense_voxels = int(np.prod((max_c - min_c + 1) * self.CHUNK_SIZE))
        else:
            dense_voxels = 0

        savings = 0.0
        if dense_voxels > 0:
            savings = max(0.0, (1.0 - (voxel_bytes / dense_voxels)) * 100.0)

        return {
            "chunk_size": self.CHUNK_SIZE,
            "allocated_chunks": allocated_chunks,
            "allocated_voxels": voxel_bytes,
            "allocated_memory_kb": round(voxel_bytes / 1024, 2),
            "dense_equivalent_voxels": dense_voxels,
            "sparse_savings_percent": round(savings, 1),
        }
