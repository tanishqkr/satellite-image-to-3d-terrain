/**
 * Shared Ray Marching, Visibility, and Probabilistic Occupancy (§3.2, §4.1).
 *
 * Provides:
 * 1. OctoMap-compatible Bayesian log-odds calculations and clamping.
 * 2. Point-to-point Line of Sight (LoS) testing with elevation profiles.
 * 3. Camera occlusion pass identifying unobserved volumes (populating UNKNOWN voxels).
 * 4. Radial horizon-sweep viewshed calculation for observer siting (§4.1 foundation).
 *
 * Zero rendering dependencies — purely mathematical and independently testable.
 */

export interface ElevationProfileSample {
  distance: number;
  rayElevation: number;
  terrainElevation: number;
  blocked: boolean;
}

export interface LineOfSightResult {
  visible: boolean;
  distance: number;
  blockingPoint: [number, number, number] | null;
  profile: ElevationProfileSample[];
}

export interface MeshElevationStats {
  width: number;
  height: number;
  elevation_min: number;
  elevation_max: number;
  elevation_range: number;
}

// =============================================================================
// 1. BAYESIAN LOG-ODDS FORMULATION (OctoMap Inverse Sensor Model)
// =============================================================================

export function probToLogOdds(p: number): number {
  const pClamped = Math.max(1e-6, Math.min(1.0 - 1e-6, p));
  return Math.log(pClamped / (1.0 - pClamped));
}

export function logOddsToProb(l: number): number {
  if (l >= 0) {
    return 1.0 / (1.0 + Math.exp(-l));
  }
  const expL = Math.exp(l);
  return expL / (1.0 + expL);
}

export function bayesianLogOddsUpdate(
  priorLogOdds: number,
  isOccupied: boolean,
  pOcc: number = 0.70,
  pFree: number = 0.40,
  lMin: number = -5.0,
  lMax: number = 5.0
): number {
  const lOcc = Math.log(pOcc / (1.0 - pOcc));
  const lFree = Math.log(pFree / (1.0 - pFree));
  const delta = isOccupied ? lOcc : lFree;
  return Math.max(lMin, Math.min(lMax, priorLogOdds + delta));
}

// =============================================================================
// 2. SAMPLING & LINE OF SIGHT (LoS)
// =============================================================================

export function sampleDsmBilinear(
  dsmRaw: number[][],
  colF: number,
  rowF: number,
  baseElevation: number = 0.0
): number {
  const rows = dsmRaw.length;
  const cols = dsmRaw[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return baseElevation;

  const c0 = Math.max(0, Math.min(cols - 1, Math.floor(colF)));
  const r0 = Math.max(0, Math.min(rows - 1, Math.floor(rowF)));
  const c1 = Math.max(0, Math.min(cols - 1, c0 + 1));
  const r1 = Math.max(0, Math.min(rows - 1, r0 + 1));

  const fc = colF - c0;
  const fr = rowF - r0;

  const v00 = Number.isFinite(dsmRaw[r0]?.[c0]) ? (dsmRaw[r0][c0] as number) : baseElevation;
  const v10 = Number.isFinite(dsmRaw[r0]?.[c1]) ? (dsmRaw[r0][c1] as number) : baseElevation;
  const v01 = Number.isFinite(dsmRaw[r1]?.[c0]) ? (dsmRaw[r1][c0] as number) : baseElevation;
  const v11 = Number.isFinite(dsmRaw[r1]?.[c1]) ? (dsmRaw[r1][c1] as number) : baseElevation;

  const top = v00 * (1.0 - fc) + v10 * fc;
  const bottom = v01 * (1.0 - fc) + v11 * fc;
  return top * (1.0 - fr) + bottom * fr;
}

export function checkLineOfSight(
  p0: [number, number, number],
  p1: [number, number, number],
  dsmRaw: number[][],
  meshStats: MeshElevationStats,
  verticalScale: number = 1.0,
  sampleStep: number = 0.5,
  tolerance: number = 0.05
): LineOfSightResult {
  const [x0, y0, z0] = p0;
  const [x1, y1, z1] = p1;

  const dx = x1 - x0;
  const dy = y1 - y0;
  const dz = z1 - z0;
  const dist3D = Math.hypot(dx, dy, dz);

  if (dist3D < 1e-4) {
    return {
      visible: true,
      distance: 0.0,
      blockingPoint: null,
      profile: [],
    };
  }

  const rows = dsmRaw.length;
  const cols = dsmRaw[0]?.length ?? 0;
  const worldW = Math.max(1e-4, (meshStats?.width ?? 100) * 0.1);
  const worldD = Math.max(1e-4, (meshStats?.height ?? 100) * 0.1);
  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;

  const numSteps = Math.max(2, Math.ceil(dist3D / sampleStep));
  const profile: ElevationProfileSample[] = [];
  let visible = true;
  let blockingPoint: [number, number, number] | null = null;

  for (let i = 0; i <= numSteps; i++) {
    const t = i / numSteps;
    const currX = x0 + t * dx;
    const currY = y0 + t * dy;
    const currZ = z0 + t * dz;
    const currDist = t * dist3D;

    // Convert world X, Z to grid normalized uv
    const u = currX / worldW + 0.5;
    const v = currZ / worldD + 0.5;
    const inBounds = u >= 0 && u <= 1 && v >= 0 && v <= 1;

    let terrWorldY = 0;
    if (inBounds) {
      const colF = u * (cols - 1);
      const rowF = v * (rows - 1);
      const rawElev = sampleDsmBilinear(dsmRaw, colF, rowF, meshStats.elevation_min);

      terrWorldY = isRelative
        ? ((rawElev - meshStats.elevation_min) / Math.max(1e-4, meshStats.elevation_range)) * 18.0 * verticalScale
        : Math.max(0.12, (rawElev - meshStats.elevation_min) * verticalScale * 0.1);
    }

    let isBlocked = false;
    if (t > 0.02 && t < 0.98 && inBounds) {
      if (terrWorldY > currY + tolerance) {
        isBlocked = true;
        if (visible) {
          visible = false;
          blockingPoint = [
            Math.round(currX * 100) / 100,
            Math.round(terrWorldY * 100) / 100,
            Math.round(currZ * 100) / 100,
          ];
        }
      }
    }

    profile.push({
      distance: Math.round(currDist * 100) / 100,
      rayElevation: Math.round(currY * 100) / 100,
      terrainElevation: Math.round(terrWorldY * 100) / 100,
      blocked: isBlocked,
    });
  }

  return {
    visible,
    distance: Math.round(dist3D * 100) / 100,
    blockingPoint,
    profile,
  };
}

// =============================================================================
// 3. SENSOR / CAMERA OCCLUSION PASS (§3.2 Populating UNKNOWN Voxels)
// =============================================================================

export function computeCameraOcclusion(
  dsmRaw: number[][],
  cameraPos: [number, number, number],
  meshStats: MeshElevationStats,
  verticalScale: number = 1.0,
  toleranceDeg: number = 0.1
): boolean[][] {
  const rows = dsmRaw.length;
  const cols = dsmRaw[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return [];

  const [camX, camY, camZ] = cameraPos;
  const worldW = Math.max(1e-4, (meshStats?.width ?? 100) * 0.1);
  const worldD = Math.max(1e-4, (meshStats?.height ?? 100) * 0.1);
  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;

  const camCol = (camX / worldW + 0.5) * (cols - 1);
  const camRow = (camZ / worldD + 0.5) * (rows - 1);

  const occlusion: boolean[][] = Array.from({ length: rows }, () =>
    new Array(cols).fill(false)
  );

  const boundaryCells: [number, number][] = [];
  for (let c = 0; c < cols; c++) {
    boundaryCells.push([0, c]);
    boundaryCells.push([rows - 1, c]);
  }
  for (let r = 1; r < rows - 1; r++) {
    boundaryCells.push([r, 0]);
    boundaryCells.push([r, cols - 1]);
  }

  const tolTan = Math.tan((toleranceDeg * Math.PI) / 180.0);

  for (let b = 0; b < boundaryCells.length; b++) {
    const [targetR, targetC] = boundaryCells[b];
    const dr = targetR - camRow;
    const dc = targetC - camCol;
    const distGrid = Math.hypot(dr, dc);
    if (distGrid < 1e-4) continue;

    const numSteps = Math.max(2, Math.ceil(distGrid));
    const stepR = dr / numSteps;
    const stepC = dc / numSteps;

    let maxSlope = -Infinity;

    for (let s = 1; s <= numSteps; s++) {
      const rIdx = Math.round(camRow + s * stepR);
      const cIdx = Math.round(camCol + s * stepC);

      if (rIdx >= 0 && rIdx < rows && cIdx >= 0 && cIdx < cols) {
        const rawH = dsmRaw[rIdx][cIdx];
        if (!Number.isFinite(rawH)) continue;

        const terrWorldY = isRelative
          ? ((rawH - meshStats.elevation_min) / Math.max(1e-4, meshStats.elevation_range)) * 18.0 * verticalScale
          : Math.max(0.12, (rawH - meshStats.elevation_min) * verticalScale * 0.1);

        const worldX = (cIdx / Math.max(1, cols - 1) - 0.5) * worldW;
        const worldZ = (rIdx / Math.max(1, rows - 1) - 0.5) * worldD;
        const distM = Math.hypot(worldX - camX, worldZ - camZ);

        if (distM < 1e-3) continue;

        const slope = (terrWorldY - camY) / distM;

        if (slope < maxSlope - tolTan) {
          occlusion[rIdx][cIdx] = true;
        } else {
          maxSlope = Math.max(maxSlope, slope);
        }
      }
    }
  }

  return occlusion;
}

// =============================================================================
// 4. OBSERVER VIEWSHED UTILITY (Shared Foundation for §4.1)
// =============================================================================

export function computeViewshed(
  observerPos: [number, number, number],
  dsmRaw: number[][],
  meshStats: MeshElevationStats,
  verticalScale: number = 1.0,
  maxRangeM?: number,
  targetHeightM: number = 0.0
): boolean[][] {
  const rows = dsmRaw.length;
  const cols = dsmRaw[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return [];

  const [obsX, obsY, obsZ] = observerPos;
  const worldW = Math.max(1e-4, (meshStats?.width ?? 100) * 0.1);
  const worldD = Math.max(1e-4, (meshStats?.height ?? 100) * 0.1);
  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;

  const obsCol = (obsX / worldW + 0.5) * (cols - 1);
  const obsRow = (obsZ / worldD + 0.5) * (rows - 1);

  const viewshed: boolean[][] = Array.from({ length: rows }, () =>
    new Array(cols).fill(false)
  );

  const obsRInt = Math.round(obsRow);
  const obsCInt = Math.round(obsCol);
  if (obsRInt >= 0 && obsRInt < rows && obsCInt >= 0 && obsCInt < cols) {
    viewshed[obsRInt][obsCInt] = true;
  }

  const boundaryCells: [number, number][] = [];
  for (let c = 0; c < cols; c++) {
    boundaryCells.push([0, c]);
    boundaryCells.push([rows - 1, c]);
  }
  for (let r = 1; r < rows - 1; r++) {
    boundaryCells.push([r, 0]);
    boundaryCells.push([r, cols - 1]);
  }

  for (let b = 0; b < boundaryCells.length; b++) {
    const [targetR, targetC] = boundaryCells[b];
    const dr = targetR - obsRow;
    const dc = targetC - obsCol;
    const distGrid = Math.hypot(dr, dc);
    if (distGrid < 1e-4) continue;

    const numSteps = Math.max(2, Math.ceil(distGrid));
    const stepR = dr / numSteps;
    const stepC = dc / numSteps;

    let maxTangent = -Infinity;

    for (let s = 1; s <= numSteps; s++) {
      const rIdx = Math.round(obsRow + s * stepR);
      const cIdx = Math.round(obsCol + s * stepC);

      if (rIdx >= 0 && rIdx < rows && cIdx >= 0 && cIdx < cols) {
        const rawH = dsmRaw[rIdx][cIdx];
        if (!Number.isFinite(rawH)) continue;

        const terrWorldY = isRelative
          ? ((rawH - meshStats.elevation_min) / Math.max(1e-4, meshStats.elevation_range)) * 18.0 * verticalScale
          : Math.max(0.12, (rawH - meshStats.elevation_min) * verticalScale * 0.1);

        const worldX = (cIdx / Math.max(1, cols - 1) - 0.5) * worldW;
        const worldZ = (rIdx / Math.max(1, rows - 1) - 0.5) * worldD;
        const distM = Math.hypot(worldX - obsX, worldZ - obsZ);

        if (maxRangeM !== undefined && distM > maxRangeM) break;

        const targetTangent = (terrWorldY + targetHeightM - obsY) / distM;
        const terrainTangent = (terrWorldY - obsY) / distM;

        if (targetTangent >= maxTangent) {
          viewshed[rIdx][cIdx] = true;
        }

        if (terrainTangent > maxTangent) {
          maxTangent = terrainTangent;
        }
      }
    }
  }

  return viewshed;
}

export interface ViewshedMetrics {
  visibleCells: number;
  totalCells: number;
  visibleRatio: number;
  deadGroundRatio: number;
  visibleAreaKm2: number;
  totalAreaKm2: number;
}

export function computeViewshedMetrics(
  viewshed: boolean[][],
  pixelSizeM: number = 1.0
): ViewshedMetrics {
  const rows = viewshed.length;
  const cols = viewshed[0]?.length ?? 0;
  const totalCells = rows * cols;
  if (totalCells === 0) {
    return {
      visibleCells: 0,
      totalCells: 0,
      visibleRatio: 0,
      deadGroundRatio: 0,
      visibleAreaKm2: 0,
      totalAreaKm2: 0,
    };
  }

  let visibleCells = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (viewshed[r][c]) visibleCells++;
    }
  }

  const cellAreaM2 = Math.max(0.01, pixelSizeM) * Math.max(0.01, pixelSizeM);
  const totalAreaKm2 = (totalCells * cellAreaM2) / 1_000_000;
  const visibleAreaKm2 = (visibleCells * cellAreaM2) / 1_000_000;
  const visibleRatio = visibleCells / totalCells;

  return {
    visibleCells,
    totalCells,
    visibleRatio,
    deadGroundRatio: 1.0 - visibleRatio,
    visibleAreaKm2,
    totalAreaKm2,
  };
}

export function viewshedToCanvasDataUrl(
  viewshed: boolean[][],
  visibleColor: [number, number, number, number] = [16, 185, 129, 140],
  occludedColor: [number, number, number, number] = [239, 68, 68, 100]
): string | null {
  const rows = viewshed.length;
  const cols = viewshed[0]?.length ?? 0;
  if (rows === 0 || cols === 0 || typeof document === 'undefined') return null;

  const canvas = document.createElement('canvas');
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const imgData = ctx.createImageData(cols, rows);
  const data = imgData.data;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = (r * cols + c) * 4;
      const isVis = viewshed[r][c];
      const color = isVis ? visibleColor : occludedColor;
      data[idx] = color[0];
      data[idx + 1] = color[1];
      data[idx + 2] = color[2];
      data[idx + 3] = color[3];
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}
