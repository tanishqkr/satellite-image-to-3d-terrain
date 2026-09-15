import test from 'node:test';
import assert from 'node:assert/strict';

import {
  computeViewshed,
  computeViewshedMetrics,
  checkLineOfSight,
  viewshedToCanvasDataUrl,
} from './rayMarch.ts';

test('computeViewshedMetrics calculates visible ratio, dead ground, and area correctly', () => {
  const grid: boolean[][] = [
    [true, true, false, false],
    [true, true, true, false],
    [false, false, false, false],
    [true, true, true, true],
  ];

  // 16 total cells, 9 visible cells (ratio = 9/16 = 0.5625)
  const metrics = computeViewshedMetrics(grid, 2.0); // 2m GSD -> cell area = 4m2

  assert.equal(metrics.totalCells, 16);
  assert.equal(metrics.visibleCells, 9);
  assert.equal(metrics.visibleRatio, 0.5625);
  assert.equal(metrics.deadGroundRatio, 0.4375);
  assert.equal(metrics.totalAreaKm2, (16 * 4) / 1_000_000);
  assert.equal(metrics.visibleAreaKm2, (9 * 4) / 1_000_000);
});

test('computeViewshed identifies direct line-of-sight and barrier shadows', () => {
  // 10x10 DSM: flat plane at 10m with a 40m high barrier at row 4
  const dsm: number[][] = Array.from({ length: 10 }, (_, r) =>
    Array.from({ length: 10 }, () => (r === 4 ? 40 : 10))
  );

  const meshStats = {
    width: 10,
    height: 10,
    elevation_min: 10,
    elevation_max: 40,
    elevation_range: 30,
  };

  // Observer at row 7, col 5 (eye elevation 12m)
  const observerPos: [number, number, number] = [0.0, 0.2, 0.2]; // World units
  const viewshed = computeViewshed(observerPos, dsm, meshStats, 1.0);

  assert.equal(viewshed.length, 10);
  assert.equal(viewshed[0].length, 10);

  // Ground at row 0 (behind row 4 barrier) should be obstructed
  const deadGroundBehindBarrier = viewshed[0][5];
  assert.equal(deadGroundBehindBarrier, false, 'Cells behind 40m barrier must be dead ground');

  // Ground in front of barrier should have visible cells
  const inFront = viewshed[6][5];
  assert.equal(inFront, true, 'Cells in front of barrier should be visible');
});

test('checkLineOfSight detects unobstructed and blocked sightlines with elevation profiles', () => {
  const dsm: number[][] = [
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 50, 10, 10], // Obstacle peak at (2, 2)
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
  ];

  const meshStats = {
    width: 5,
    height: 5,
    elevation_min: 10,
    elevation_max: 50,
    elevation_range: 40,
  };

  // Test 1: Unobstructed horizontal sightline at row 0
  const losClear = checkLineOfSight(
    [-0.2, 0.1, -0.2],
    [0.2, 0.1, -0.2],
    dsm,
    meshStats,
    1.0
  );
  assert.equal(losClear.visible, true);
  assert.equal(losClear.blockingPoint, null);
  assert.ok(losClear.profile.length > 2);

  // Test 2: Obstructed diagonal sightline through (2, 2)
  const losBlocked = checkLineOfSight(
    [-0.2, 0.1, -0.2],
    [0.2, 0.1, 0.2],
    dsm,
    meshStats,
    1.0
  );
  assert.equal(losBlocked.visible, false);
  assert.notEqual(losBlocked.blockingPoint, null);
});

test('viewshedToCanvasDataUrl handles server-side / headless environments gracefully', () => {
  const grid = [[true, false], [false, true]];
  // In node headless environment (typeof document === 'undefined'), returns null safely
  const result = viewshedToCanvasDataUrl(grid);
  assert.equal(result, null);
});
