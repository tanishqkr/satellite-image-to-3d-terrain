import test from 'node:test';
import assert from 'node:assert/strict';
import {
  probToLogOdds,
  logOddsToProb,
  bayesianLogOddsUpdate,
  sampleDsmBilinear,
  checkLineOfSight,
  computeCameraOcclusion,
  computeViewshed,
} from './rayMarch.ts';

test('probToLogOdds and logOddsToProb invertible conversion and clamping', () => {
  // 1. p = 0.5 -> log-odds = 0
  const l0 = probToLogOdds(0.5);
  assert.ok(Math.abs(l0) < 1e-6, `Expected ~0, got ${l0}`);
  assert.ok(Math.abs(logOddsToProb(0) - 0.5) < 1e-6);

  // 2. High confidence p = 0.95
  const lHigh = probToLogOdds(0.95);
  assert.ok(lHigh > 2.0);
  const pRecovered = logOddsToProb(lHigh);
  assert.ok(Math.abs(pRecovered - 0.95) < 1e-4);

  // 3. Extremes clamp without inf or NaN
  const lMax = probToLogOdds(1.0);
  assert.ok(Number.isFinite(lMax));
  const lMin = probToLogOdds(0.0);
  assert.ok(Number.isFinite(lMin));
});

test('bayesianLogOddsUpdate accumulates hits/misses and bounds strictly', () => {
  let l = 0.0;
  // Single hit with pOcc=0.70
  l = bayesianLogOddsUpdate(l, true, 0.70);
  assert.ok(l > 0.8, `Expected > 0.8, got ${l}`);
  assert.ok(logOddsToProb(l) > 0.69);

  // Accumulate 10 hits -> clamped to 5.0
  for (let i = 0; i < 10; i++) {
    l = bayesianLogOddsUpdate(l, true);
  }
  assert.equal(l, 5.0);
  assert.ok(logOddsToProb(l) > 0.99);

  // Accumulate 30 misses -> clamped to -5.0
  for (let i = 0; i < 30; i++) {
    l = bayesianLogOddsUpdate(l, false);
  }
  assert.equal(l, -5.0);
  assert.ok(logOddsToProb(l) < 0.01);
});

test('sampleDsmBilinear interpolates fractional coordinates accurately', () => {
  const dsm = [
    [10, 20],
    [30, 40],
  ];
  const centerVal = sampleDsmBilinear(dsm, 0.5, 0.5);
  assert.equal(centerVal, 25);

  const edgeVal = sampleDsmBilinear(dsm, 0.0, 0.5);
  assert.equal(edgeVal, 20);
});

test('checkLineOfSight detects unobstructed and terrain-obstructed paths', () => {
  const rows = 30;
  const cols = 30;
  const dsm: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(5.0));

  const meshStats = {
    width: 300,
    height: 300,
    elevation_min: 0,
    elevation_max: 30,
    elevation_range: 30,
  };

  // 1. Clear LoS across flat terrain
  const resClear = checkLineOfSight(
    [-10, 5, -10],
    [10, 5, 10],
    dsm,
    meshStats,
    1.0
  );
  assert.equal(resClear.visible, true);
  assert.equal(resClear.blockingPoint, null);
  assert.ok(resClear.profile.length > 5);

  // 2. High ridge at center blocking path
  for (let r = 12; r <= 18; r++) {
    for (let c = 12; c <= 18; c++) {
      dsm[r][c] = 25.0; // 25m structure
    }
  }

  const resBlocked = checkLineOfSight(
    [-10, 1.5, -10],
    [10, 1.5, 10],
    dsm,
    meshStats,
    1.0
  );
  assert.equal(resBlocked.visible, false);
  assert.ok(resBlocked.blockingPoint !== null);
  assert.ok(resBlocked.profile.some((p) => p.blocked));
});

test('computeCameraOcclusion marks geometric shadows behind tall obstacles', () => {
  const rows = 32;
  const cols = 32;
  const dsm: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0.0));

  // Tall 20m structure at center
  for (let r = 12; r <= 16; r++) {
    for (let c = 12; c <= 16; c++) {
      dsm[r][c] = 20.0;
    }
  }

  const meshStats = {
    width: 320,
    height: 320,
    elevation_min: 0,
    elevation_max: 20,
    elevation_range: 20,
  };

  // Oblique drone camera in northwest corner at low altitude Y=6.0m
  const cameraPos: [number, number, number] = [-12, 6.0, -12];

  const occlusion = computeCameraOcclusion(dsm, cameraPos, meshStats, 1.0);
  assert.equal(occlusion.length, rows);
  assert.equal(occlusion[0].length, cols);

  // Surface in front of building is visible
  assert.equal(occlusion[6][6], false);
  // Roof is visible
  assert.equal(occlusion[14][14], false);

  // Ground directly behind building in the shadow cast from camera is occluded
  let occludedBehindCount = 0;
  for (let r = 16; r < 24; r++) {
    for (let c = 16; c < 24; c++) {
      if (occlusion[r][c]) occludedBehindCount++;
    }
  }
  assert.ok(occludedBehindCount > 0, `Expected occluded cells in shadow, got ${occludedBehindCount}`);
});

test('computeViewshed identifies line-of-sight from observer and ridge blockages', () => {
  const rows = 30;
  const cols = 30;
  const dsm: number[][] = Array.from({ length: rows }, () => new Array(cols).fill(0.0));

  // Ridge across row 15
  for (let c = 0; c < cols; c++) {
    dsm[15][c] = 12.0;
  }

  const meshStats = {
    width: 300,
    height: 300,
    elevation_min: 0,
    elevation_max: 12,
    elevation_range: 12,
  };

  // Observer on south side looking north
  const observerPos: [number, number, number] = [0, 1.8, -8];

  const viewshed = computeViewshed(observerPos, dsm, meshStats, 1.0);

  // Ground near observer is visible
  assert.equal(viewshed[7][15], true);
  // Ridge crest is visible
  assert.equal(viewshed[15][15], true);
  // Ground on far side behind 12m ridge is occluded
  assert.equal(viewshed[22][15], false);
});
