import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getTerrainElevationAt,
  computeAgl,
  computeMsl,
  resolveTerrainCollision,
  castSensor,
} from './dsmSampling.ts';
import type { MeshElevationStats } from './dsmSampling.ts';

test('getTerrainElevationAt returns flat ground elevation correctly', () => {
  const meshStats: MeshElevationStats = {
    width: 100, // worldW = 10
    height: 100, // worldD = 10
    elevation_min: 50,
    elevation_max: 150,
    elevation_range: 100,
  };

  // 10x10 uniform elevation of 70m
  const dsmRaw = Array.from({ length: 10 }, () => Array(10).fill(70));

  // At verticalScale = 1.0, worldY = (70 - 50) * 1.0 * 0.1 = 2.0
  const yCenter = getTerrainElevationAt(0, 0, dsmRaw, meshStats, 1.0);
  assert.ok(Math.abs(yCenter - 2.0) < 1e-4, `Expected 2.0, got ${yCenter}`);

  const yCorner = getTerrainElevationAt(-5, -5, dsmRaw, meshStats, 1.0);
  assert.ok(Math.abs(yCorner - 2.0) < 1e-4, `Expected 2.0, got ${yCorner}`);
});

test('getTerrainElevationAt interpolates ramp elevation bilinearly', () => {
  const meshStats: MeshElevationStats = {
    width: 200, // worldW = 20 (-10 to +10)
    height: 200, // worldD = 20 (-10 to +10)
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // 2x2 grid: bottom-left = 0, bottom-right = 100, top-left = 0, top-right = 100
  // row 0: [0, 100]
  // row 1: [0, 100]
  const dsmRaw = [
    [0, 100],
    [0, 100],
  ];

  // At center X = 0 (halfway across), elevation should be 50
  // worldY = (50 - 0) * 1.0 * 0.1 = 5.0
  const yMid = getTerrainElevationAt(0, 0, dsmRaw, meshStats, 1.0);
  assert.ok(Math.abs(yMid - 5.0) < 1e-3, `Expected 5.0 at midpoint, got ${yMid}`);

  // At left edge X = -10, elevation should be 0
  const yLeft = getTerrainElevationAt(-10, 0, dsmRaw, meshStats, 1.0);
  assert.ok(Math.abs(yLeft - 0.0) < 1e-3, `Expected 0.0 at left, got ${yLeft}`);

  // At right edge X = +10, elevation should be 10.0 (100m * 0.1)
  const yRight = getTerrainElevationAt(10, 0, dsmRaw, meshStats, 1.0);
  assert.ok(Math.abs(yRight - 10.0) < 1e-3, `Expected 10.0 at right, got ${yRight}`);
});

test('getTerrainElevationAt respects flood waterLevel when submerged', () => {
  const meshStats: MeshElevationStats = {
    width: 100,
    height: 100,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // Terrain at 10m elevation
  const dsmRaw = Array.from({ length: 4 }, () => Array(4).fill(10));

  // With water level at 25m:
  // Water worldY = (25 - 0) * 0.1 = 2.5
  // Terrain worldY = (10 - 0) * 0.1 = 1.0
  // Submerged elevation should be clamped up to water surface (2.5)
  const ySubmerged = getTerrainElevationAt(0, 0, dsmRaw, meshStats, 1.0, 25);
  assert.ok(Math.abs(ySubmerged - 2.5) < 1e-4, `Expected water surface 2.5, got ${ySubmerged}`);
});

test('computeAgl and computeMsl convert world altitude to metric telemetry', () => {
  const meshStats: MeshElevationStats = {
    width: 100,
    height: 100,
    elevation_min: 200,
    elevation_max: 300,
    elevation_range: 100,
  };

  // Ground is at 2.0 world units (which corresponds to 220m MSL)
  // Drone is at 3.5 world units (which corresponds to 235m MSL)
  const groundWorldY = 2.0;
  const droneWorldY = 3.5;

  const agl = computeAgl(droneWorldY, groundWorldY, meshStats, 1.0);
  const msl = computeMsl(droneWorldY, meshStats, 1.0);

  // AGL = (3.5 - 2.0) / 0.1 = 15.0m
  assert.ok(Math.abs(agl - 15.0) < 1e-3, `Expected AGL 15.0m, got ${agl}`);

  // MSL = 200 + 3.5 / 0.1 = 235.0m
  assert.ok(Math.abs(msl - 235.0) < 1e-3, `Expected MSL 235.0m, got ${msl}`);
});

test('resolveTerrainCollision enforces floor clearance and zeroes falling velocity', () => {
  const meshStats: MeshElevationStats = {
    width: 100,
    height: 100,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // Flat ground at 0 elevation (worldY = 0)
  const dsmRaw = Array.from({ length: 4 }, () => Array(4).fill(0));

  const prevPos = { x: 0, y: 2.0, z: 0 };
  // Drone tries to plunge below ground level to y = -0.5 with downward velocity
  const candidatePos = { x: 0, y: -0.5, z: 0 };
  const velocity = { x: 0, y: -5.0, z: 0 };

  const result = resolveTerrainCollision(
    prevPos,
    candidatePos,
    velocity,
    dsmRaw,
    meshStats,
    1.0,
    undefined,
    1.2 // minClearance
  );

  assert.equal(result.collidedFloor, true);
  assert.ok(result.position.y >= 1.2, `Expected position.y >= 1.2, got ${result.position.y}`);
  assert.equal(result.velocity.y, 0, 'Negative vertical velocity must be zeroed on floor contact');
});

test('resolveTerrainCollision prevents passing through vertical building walls', () => {
  const meshStats: MeshElevationStats = {
    width: 100, // worldW = 10 (-5 to +5)
    height: 100,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // Left half (cols 0, 1) is 0m (worldY = 0)
  // Right half (cols 2, 3) is a 50m tall building (worldY = 5.0)
  const dsmRaw = [
    [0, 0, 50, 50],
    [0, 0, 50, 50],
    [0, 0, 50, 50],
    [0, 0, 50, 50],
  ];

  // Drone is at altitude y = 1.5, flying rightwards into the building wall
  const prevPos = { x: -1.0, y: 1.5, z: 0 };
  const candidatePos = { x: 2.0, y: 1.5, z: 0 };
  const velocity = { x: 10.0, y: 0, z: 0 };

  const result = resolveTerrainCollision(
    prevPos,
    candidatePos,
    velocity,
    dsmRaw,
    meshStats,
    1.0,
    undefined,
    1.2
  );

  assert.equal(result.collidedWall, true);
  // Drone horizontal X position must be arrested at prevPos.x
  assert.equal(result.position.x, prevPos.x, 'Drone position.x must be preserved at prevPos on wall impact');
  // Horizontal velocity must be damped
  assert.ok(result.velocity.x < 1.0, 'Horizontal velocity towards wall must be stopped/damped');
});

test('castSensor measures downward AGL distance accurately', () => {
  const meshStats: MeshElevationStats = {
    width: 100,
    height: 100,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // Terrain at 2.0m world height
  const dsmRaw = Array.from({ length: 4 }, () => Array(4).fill(20)); // worldY = 2.0

  // Drone origin at (0, 10, 0), looking straight down
  const origin = { x: 0, y: 10, z: 0 };
  const dirWorld = { x: 0, y: -1, z: 0 };

  const dist = castSensor(origin, dirWorld, dsmRaw, meshStats, 1.0);
  // Expected distance = 10 - 2 = 8.0m
  assert.ok(Math.abs(dist - 8.0) < 1e-3, `Expected downward distance 8.0, got ${dist}`);
});

test('castSensor detects obstacle wall at accurate distance', () => {
  const meshStats: MeshElevationStats = {
    width: 200, // worldW = 20 (-10 to +10)
    height: 200,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // 10x10 grid:
  // Cols 0-4 are ground level (0m)
  // Cols 5-9 are a tall building (50m, worldY = 5.0)
  // Boundary between col 4 and 5 is at X = 0
  const dsmRaw = Array.from({ length: 10 }, () => [
    0, 0, 0, 0, 0, 50, 50, 50, 50, 50,
  ]);

  // Drone is at X = -5 (cols 2-3 area, flat ground at 0m), altitude Y = 2.0m
  // Sensor points right (+X) towards the building at X >= 0
  const origin = { x: -5, y: 2.0, z: 0 };
  const dirWorld = { x: 1, y: 0, z: 0 };

  const dist = castSensor(origin, dirWorld, dsmRaw, meshStats, 1.0, undefined, 50, 0.5);
  // The building wall is approximately 5m to the right
  assert.ok(dist >= 4.0 && dist <= 6.0, `Expected wall detection around 5.0m, got ${dist}`);
});

test('castSensor returns maxRange when no obstacle intersects ray', () => {
  const meshStats: MeshElevationStats = {
    width: 100,
    height: 100,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // Completely flat low ground (worldY = 0)
  const dsmRaw = Array.from({ length: 4 }, () => Array(4).fill(0));

  // Drone is at Y = 10m, ray points horizontally (+X)
  const origin = { x: 0, y: 10, z: 0 };
  const dirWorld = { x: 1, y: 0, z: 0 };

  const dist = castSensor(origin, dirWorld, dsmRaw, meshStats, 1.0, undefined, 50, 0.5);
  assert.equal(dist, 50, `Expected maxRange 50, got ${dist}`);
});

test('getTerrainElevationAt with voxel mode quantizes continuous ramp into discrete stepped blocks (§3)', () => {
  const meshStats: MeshElevationStats = {
    width: 200, // worldW = 20 (-10 to +10)
    height: 200,
    elevation_min: 0,
    elevation_max: 100,
    elevation_range: 100,
  };

  // Continuous ramp from 0 to 100 across 20 columns
  const dsmRaw = Array.from({ length: 20 }, () =>
    Array.from({ length: 20 }, (_, c) => (c / 19) * 100)
  );

  // In smooth mode, sampling two adjacent points gives continuous differing elevations
  const ySmooth1 = getTerrainElevationAt(1.0, 0, dsmRaw, meshStats, 1.0, undefined, { renderMode: 'smooth' });
  const ySmooth2 = getTerrainElevationAt(1.5, 0, dsmRaw, meshStats, 1.0, undefined, { renderMode: 'smooth' });
  assert.notEqual(ySmooth1, ySmooth2, 'Smooth mode elevations should vary continuously');

  // In voxel mode with coarse resolution (e.g. 8 blocks of width 2.5m, block 4 is [0, 2.5]),
  // two points in the same voxel block have identical stepped elevation
  const yVoxel1 = getTerrainElevationAt(1.0, 0, dsmRaw, meshStats, 1.0, undefined, {
    renderMode: 'voxel',
    voxelResolution: 8,
  });
  const yVoxel2 = getTerrainElevationAt(1.5, 0, dsmRaw, meshStats, 1.0, undefined, {
    renderMode: 'voxel',
    voxelResolution: 8,
  });
  assert.equal(yVoxel1, yVoxel2, 'Voxel mode within same cell must return identical quantized block height');
});

test('resolveTerrainCollision in voxel mode prevents clipping into stepped voxel blocks (§3)', () => {
  const meshStats: MeshElevationStats = {
    width: 100,
    height: 100,
    elevation_min: 0,
    elevation_max: 50,
    elevation_range: 50,
  };

  // 10x10 grid with tall voxel block (30m elevation, worldY = 3.0)
  const dsmRaw = Array.from({ length: 10 }, () => Array(10).fill(30));

  const prevPos = { x: 0, y: 5.0, z: 0 };
  const candidatePos = { x: 0, y: 1.0, z: 0 }; // trying to sink through block
  const velocity = { x: 0, y: -4.0, z: 0 };

  const collision = resolveTerrainCollision(
    prevPos,
    candidatePos,
    velocity,
    dsmRaw,
    meshStats,
    1.0,
    undefined,
    1.2,
    { renderMode: 'voxel', voxelResolution: 10 }
  );

  // Ground is at 3.0 worldY, min clearance is 1.2 -> drone must clamp to at least 4.2
  assert.ok(collision.position.y >= 4.2, `Expected position.y >= 4.2, got ${collision.position.y}`);
  assert.equal(collision.velocity.y, 0, 'Vertical velocity into voxel block must be zeroed');
  assert.equal(collision.collidedFloor, true);
});


