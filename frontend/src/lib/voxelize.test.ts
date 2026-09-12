import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  voxelize,
  generateTurboPalette,
  sampleTurboRgb,
  rgbToHex,
  SparseChunkedGrid,
  VoxelState,
  buildSparseVoxelGrid,
} from './voxelize.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('sampleTurboRgb and rgbToHex generate valid color progression', () => {
  const blue = sampleTurboRgb(0.0);
  assert.deepEqual(blue, [0, 0, 1]);
  assert.equal(rgbToHex(blue), '#0000ff');

  const red = sampleTurboRgb(1.0);
  assert.deepEqual(red, [1, 0, 0]);
  assert.equal(rgbToHex(red), '#ff0000');

  const green = sampleTurboRgb(0.5);
  assert.deepEqual(green, [0, 1, 0]);
  assert.equal(rgbToHex(green), '#00ff00');
});

test('generateTurboPalette produces exact band count with unique stops', () => {
  for (const count of [5, 7, 8, 12]) {
    const palette = generateTurboPalette(count);
    assert.equal(palette.length, count);
    palette.forEach((hex) => {
      assert.match(hex, /^#[0-9a-f]{6}$/i);
    });
  }
});

test('voxelize pools synthetic ramp height grid correctly', () => {
  const H = 100;
  const W = 100;
  const syntheticGrid: number[][] = [];
  for (let r = 0; r < H; r++) {
    const row: number[] = [];
    for (let c = 0; c < W; c++) {
      row.push(r);
    }
    syntheticGrid.push(row);
  }

  const targetRes = 10;
  const bandCount = 8;
  const result = voxelize(syntheticGrid, {
    targetResolution: targetRes,
    bandCount,
    worldWidth: 10,
    worldDepth: 10,
  });

  assert.equal(result.rows, targetRes);
  assert.equal(result.cols, targetRes);
  assert.equal(result.blocks.length, targetRes * targetRes);

  const firstBlock = result.blocks[0];
  assert.equal(firstBlock.gridX, 0);
  assert.equal(firstBlock.gridZ, 0);
  assert.equal(firstBlock.rawHeight, 4.5);
  assert.equal(firstBlock.bandIndex, 0);

  const lastBlock = result.blocks[result.blocks.length - 1];
  assert.equal(lastBlock.gridX, targetRes - 1);
  assert.equal(lastBlock.gridZ, targetRes - 1);
  assert.equal(lastBlock.rawHeight, 94.5);
  assert.equal(lastBlock.bandIndex, bandCount - 1);

  assert.equal(result.elevationMin, 4.5);
  assert.equal(result.elevationMax, 94.5);
  assert.equal(result.elevationRange, 90);

  assert.equal(firstBlock.posX, -4.5);
  assert.equal(firstBlock.posZ, -4.5);
  assert.equal(lastBlock.posX, 4.5);
  assert.equal(lastBlock.posZ, 4.5);

  const expectedCellWidth = (10 / 10) * 0.90;
  assert.equal(result.cellWidth, expectedCellWidth);
  assert.equal(result.cellDepth, expectedCellWidth);
});

test('voxelize handles non-square aspect ratio properly', () => {
  const syntheticGrid: number[][] = Array.from({ length: 60 }, () =>
    new Array(120).fill(10)
  );

  const result = voxelize(syntheticGrid, { targetResolution: 64 });
  assert.equal(result.cols, 64);
  assert.equal(result.rows, 32);
  assert.equal(result.blocks.length, 64 * 32);
});

test('voxelize handles flat elevation plane without zero-division', () => {
  const flatGrid: number[][] = Array.from({ length: 30 }, () =>
    new Array(30).fill(42.0)
  );

  const result = voxelize(flatGrid, { targetResolution: 10, bandCount: 8 });
  assert.equal(result.elevationRange, 1e-6);
  assert.equal(result.blocks.length, 100);
  result.blocks.forEach((b) => {
    assert.equal(b.rawHeight, 42.0);
    assert.equal(b.normalizedHeight, 0);
    assert.equal(b.bandIndex, 0);
  });
});

test('voxelize handles empty or corrupt grid gracefully', () => {
  const emptyResult = voxelize([]);
  assert.equal(emptyResult.blocks.length, 0);
  assert.equal(emptyResult.rows, 0);
  assert.equal(emptyResult.cols, 0);
});

test('voxelize validates against real satellite DSM dataset', () => {
  const samplePath = path.join(__dirname, 'sample_real_dsm.json');
  assert.ok(fs.existsSync(samplePath), 'sample_real_dsm.json fixture must exist');

  const rawData = JSON.parse(fs.readFileSync(samplePath, 'utf8'));
  const dsmRaw: number[][] = rawData.dsm_raw;

  assert.equal(dsmRaw.length, 128);
  assert.equal(dsmRaw[0].length, 128);

  const bandCount = 8;
  const targetRes = 48;
  const result = voxelize(dsmRaw, {
    targetResolution: targetRes,
    bandCount,
    gapRatio: 0.90,
    worldWidth: 128 * 0.1,
    worldDepth: 128 * 0.1,
  });

  // Verify grid size
  assert.equal(result.rows, targetRes);
  assert.equal(result.cols, targetRes);
  assert.equal(result.blocks.length, targetRes * targetRes);

  // Verify bands are populated across real topography
  const bandCounts = new Array(bandCount).fill(0);
  result.blocks.forEach((b) => {
    assert.ok(Number.isFinite(b.posX), 'posX must be finite');
    assert.ok(Number.isFinite(b.posZ), 'posZ must be finite');
    assert.ok(Number.isFinite(b.rawHeight), 'rawHeight must be finite');
    assert.ok(b.normalizedHeight >= 0 && b.normalizedHeight <= 1.0, 'normH must be in [0, 1]');
    assert.ok(b.bandIndex >= 0 && b.bandIndex < bandCount, 'bandIndex in range');
    assert.match(b.colorHex, /^#[0-9a-f]{6}$/i, 'valid hex color');
    bandCounts[b.bandIndex]++;
  });

  // Both ground (band 0) and structural elevations (higher bands) should be present
  assert.ok(bandCounts[0] > 0, 'Ground level blocks present');
  const upperBands = bandCounts.slice(1).reduce((a, b) => a + b, 0);
  assert.ok(upperBands > 0, 'Elevated structure blocks present');

  // Verify real physical bounds
  assert.ok(result.elevationMin >= 0.0);
  assert.ok(result.elevationMax > 20.0); // Real satellite tile has structures over 20-45m
});

test('SparseChunkedGrid manages 16x16x16 chunks, tri-state voxels, and LOD', () => {
  const grid = new SparseChunkedGrid(1.0, 1.0, 1.0, 32, 32);
  assert.equal(grid.getVoxel(0, 0, 0), VoxelState.FREE);
  assert.equal(grid.chunks.size, 0);

  // Set occupied voxel at (5, 10, 5) -> chunk (0, 0, 0)
  grid.setVoxel(5, 10, 5, VoxelState.OCCUPIED);
  assert.equal(grid.getVoxel(5, 10, 5), VoxelState.OCCUPIED);
  assert.equal(grid.chunks.size, 1);

  // Set unknown voxel at (20, 5, 20) -> chunk (1, 0, 1)
  grid.setVoxel(20, 5, 20, VoxelState.UNKNOWN);
  assert.equal(grid.getVoxel(20, 5, 20), VoxelState.UNKNOWN);
  assert.equal(grid.chunks.size, 2);

  // LOD test
  const lod1 = grid.getLOD(1);
  assert.equal(lod1.getVoxel(2, 5, 2), VoxelState.OCCUPIED);
  assert.equal(lod1.getVoxel(10, 2, 10), VoxelState.UNKNOWN);

  // Memory stats
  const stats = grid.getMemoryStats();
  assert.equal(stats.allocatedChunks, 2);
  assert.ok(stats.sparseSavingsPercent > 0);
});

test('buildSparseVoxelGrid populates solid column voxels from DSM', () => {
  const dsm = [
    [10, 10, 10],
    [10, 25, 10],
    [10, 10, 10],
  ];
  const sparse = buildSparseVoxelGrid(dsm, { targetResolution: 3, bandCount: 5 });
  assert.ok(sparse.chunks.size > 0);

  // At ground level y=0, both border and center are occupied
  assert.equal(sparse.getVoxel(0, 0, 0), VoxelState.OCCUPIED);
  assert.equal(sparse.getVoxel(1, 0, 1), VoxelState.OCCUPIED);
});

test('voxelize and buildSparseVoxelGrid handle tri-state occlusion mask (§3.2)', () => {
  const dsm = [
    [5, 5, 5, 5],
    [5, 20, 20, 5],
    [5, 5, 5, 5],
    [5, 5, 5, 5],
  ];

  // Occlusion mask: bottom-right quadrant is occluded
  const mask = [
    [false, false, false, false],
    [false, false, false, false],
    [false, false, true, true],
    [false, false, true, true],
  ];

  const grid = voxelize(dsm, {
    targetResolution: 4,
    occlusionMask: mask,
  });

  const unkBlocks = grid.blocks.filter((b) => b.state === VoxelState.UNKNOWN);
  const occBlocks = grid.blocks.filter((b) => b.state === VoxelState.OCCUPIED);

  assert.ok(unkBlocks.length > 0, 'Must contain UNKNOWN blocks in occluded quadrant');
  assert.ok(occBlocks.length > 0, 'Must contain OCCUPIED blocks in visible quadrant');

  // Verify purple color formatting for UNKNOWN blocks
  for (const b of unkBlocks) {
    assert.equal(b.colorHex, '#8B5CF6');
  }

  // Verify buildSparseVoxelGrid sets UNKNOWN voxels
  const sparse = buildSparseVoxelGrid(dsm, {
    targetResolution: 4,
    occlusionMask: mask,
  });
  assert.ok(sparse.getUnknownCount() > 0, 'Sparse grid must contain UNKNOWN voxels');
});


