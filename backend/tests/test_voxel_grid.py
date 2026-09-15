"""
Unit and Integration Tests for Sparse Voxel Data Structure (§3.1 Phase 1).
Covers:
1. Sparse chunked memory allocation (16x16x16 chunks) with zero-cost empty sky.
2. Tri-state cell schema (FREE=0, OCCUPIED=1, UNKNOWN=2).
3. Column-pooling population from 2D DSM elevation surfaces.
4. Multi-resolution Level-of-Detail (LOD) downsampling.
5. Memory savings metrics vs dense 3D bounding volumes.
6. 3D coordinate queries for downstream autonomous routing (§4.3) and viewshed (§4.1).
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.voxel_grid import SparseVoxelGrid, VoxelState


def test_sparse_chunk_allocation():
    print("Testing sparse chunk allocation and indexing...")
    grid = SparseVoxelGrid(voxel_size=(1.0, 1.0, 1.0))
    assert len(grid.chunks) == 0, "Initial grid must have zero allocated chunks"

    # Querying unallocated voxel returns FREE without allocating
    val = grid.get_voxel(100, 200, 300)
    assert val == VoxelState.FREE
    assert len(grid.chunks) == 0, "Read of empty voxel must not allocate a chunk"

    # Setting FREE on unallocated space does not allocate
    grid.set_voxel(50, 50, 50, VoxelState.FREE)
    assert len(grid.chunks) == 0

    # Setting OCCUPIED allocates exactly 1 chunk (16x16x16)
    grid.set_voxel(18, 5, 25, VoxelState.OCCUPIED)
    assert len(grid.chunks) == 1
    # Global (18, 5, 25) -> chunk (1, 0, 1)
    assert (1, 0, 1) in grid.chunks
    assert grid.get_voxel(18, 5, 25) == VoxelState.OCCUPIED

    # Another voxel in the same chunk does not allocate a new chunk
    grid.set_voxel(19, 6, 26, VoxelState.OCCUPIED)
    assert len(grid.chunks) == 1

    # Setting UNKNOWN (§3.2 tri-state schema)
    grid.set_voxel(18, 10, 25, VoxelState.UNKNOWN)
    assert grid.get_voxel(18, 10, 25) == VoxelState.UNKNOWN
    print("  [PASS] Sparse chunk allocation and indexing verified: PASSED")


def test_from_dsm_population():
    print("Testing column-pooling from DSM elevation surface...")
    H, W = 48, 48
    dsm = np.full((H, W), 5.0, dtype=np.float32)  # 5m base terrain
    dsm[20:28, 20:28] = 25.0  # 25m tall structure

    grid = SparseVoxelGrid.from_dsm(
        dsm=dsm,
        gsd=1.0,
        voxel_size_y=1.0,
        base_elevation=0.0,
    )
    assert len(grid.chunks) > 0

    # Low ground at (5, 5): solid from y=0 to y=4, empty above
    for y in range(5):
        assert grid.get_voxel(5, y, 5) == VoxelState.OCCUPIED, f"Expected occupied at y={y}"
    assert grid.get_voxel(5, 6, 5) == VoxelState.FREE, "Expected free above ground at y=6"

    # High building at (24, 24): solid up to y=24, empty above
    for y in range(25):
        assert grid.get_voxel(24, y, 24) == VoxelState.OCCUPIED, f"Expected occupied inside structure at y={y}"
    assert grid.get_voxel(24, 26, 24) == VoxelState.FREE, "Expected free above building roof at y=26"

    print(f"  [PASS] Solid columns populated accurately up to observed heights: PASSED")


def test_multiresolution_lod():
    print("Testing multi-resolution LOD generation...")
    H, W = 32, 32
    dsm = np.full((H, W), 10.0, dtype=np.float32)
    grid = SparseVoxelGrid.from_dsm(dsm, gsd=1.0, voxel_size_y=1.0, base_elevation=0.0)

    lod1 = grid.get_lod(level=1)
    assert lod1.voxel_size == (2.0, 2.0, 2.0)

    # Downsampled LOD coordinates
    # Original occupied at (0, 0, 0) maps to LOD (0, 0, 0)
    assert lod1.get_voxel(0, 0, 0) == VoxelState.OCCUPIED
    # Original occupied at (4, 4, 4) maps to LOD (2, 2, 2)
    assert lod1.get_voxel(2, 2, 2) == VoxelState.OCCUPIED
    # Sky above
    assert lod1.get_voxel(2, 10, 2) == VoxelState.FREE
    print("  [PASS] LOD 1 downsampling (2:1 factor) verified: PASSED")


def test_memory_savings_and_coordinate_export():
    print("Testing memory savings & occupied coordinate retrieval...")
    H, W = 64, 64
    # Local hill in center
    y, x = np.mgrid[0:H, 0:W]
    dsm = 10.0 + 30.0 * np.exp(-((x - 32)**2 + (y - 32)**2) / 150.0)

    grid = SparseVoxelGrid.from_dsm(dsm, gsd=1.0, voxel_size_y=1.0, base_elevation=0.0)
    stats = grid.get_memory_stats(dense_bounding_shape=(64, 45, 64))

    print(f"  Allocated chunks: {stats['allocated_chunks']}, Memory: {stats['allocated_memory_kb']} KB")
    print(f"  Dense equivalent: {stats['dense_equivalent_voxels']} voxels, Savings: {stats['sparse_savings_percent']}%")
    assert stats["sparse_savings_percent"] >= 40.0, "Expected at least 40% memory savings vs bounding box"

    # Test coordinate export for downstream routing (§4.3)
    coords = grid.get_occupied_coords(as_world=False)
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert len(coords) > 1000

    world_coords = grid.get_occupied_coords(as_world=True)
    assert world_coords.dtype == np.float32
    print("  [PASS] Memory statistics and 3D coordinate export verified: PASSED")


if __name__ == "__main__":
    test_sparse_chunk_allocation()
    test_from_dsm_population()
    test_multiresolution_lod()
    test_memory_savings_and_coordinate_export()
    print("\nALL SPARSE VOXEL DATA STRUCTURE (§3.1) TESTS PASSED!")
