/**
 * Pure pooling and palette-quantization for Voxel/Block DSM rendering.
 *
 * Transforms a high-resolution 2D height grid (dsm_raw) into a downsampled
 * grid of discrete box instances with stepped Turbo palette colors.
 * Zero rendering dependencies — purely mathematical and independently testable.
 */

export interface VoxelizeOptions {
  /** Target block resolution along the primary axis (default: 64, range: 24–96) */
  targetResolution?: number;
  /** Number of discrete elevation bands / colors (default: 8, range: 5–12) */
  bandCount?: number;
  /** Optional minimum elevation override (default: min value in dsmRaw) */
  elevationMin?: number;
  /** Optional maximum elevation override (default: max value in dsmRaw) */
  elevationMax?: number;
  /** Gap ratio for X/Z block footprint to create the Minecraft style (default: 0.90, range: 0.85–0.95) */
  gapRatio?: number;
  /** World width of the terrain plane (default: dsmRaw[0].length * 0.1) */
  worldWidth?: number;
  /** World depth of the terrain plane (default: dsmRaw.length * 0.1) */
  worldDepth?: number;
  /** Optional 2D boolean occlusion mask from sensor ray marching (§3.2) */
  occlusionMask?: boolean[][];
}

export interface VoxelBlock {
  /** Column index in the downsampled grid (0 <= gridX < cols) */
  gridX: number;
  /** Row index in the downsampled grid (0 <= gridZ < rows) */
  gridZ: number;
  /** World-space X center coordinate */
  posX: number;
  /** World-space Z center coordinate */
  posZ: number;
  /** Averaged raw elevation in physical or relative units */
  rawHeight: number;
  /** Normalized elevation in [0, 1] relative to elevation range */
  normalizedHeight: number;
  /** Quantized band index (0 <= bandIndex < bandCount) */
  bandIndex: number;
  /** Hex color string (e.g. "#00bfff") */
  colorHex: string;
  /** Normalized RGB components in [0, 1] for direct use with Three.Color */
  colorRgb: [number, number, number];
  /** Tri-state occupancy condition (§3.1, §3.2) */
  state: VoxelState;
}

export const VoxelState = {
  FREE: 0,
  OCCUPIED: 1,
  UNKNOWN: 2,
} as const;

export type VoxelState = (typeof VoxelState)[keyof typeof VoxelState];

export interface VoxelGrid {
  /** All voxel block descriptors */
  blocks: VoxelBlock[];
  /** Downsampled grid rows count */
  rows: number;
  /** Downsampled grid columns count */
  cols: number;
  /** Cell physical footprint width */
  cellWidth: number;
  /** Cell physical footprint depth */
  cellDepth: number;
  /** Minimum elevation across the pooled grid */
  elevationMin: number;
  /** Maximum elevation across the pooled grid */
  elevationMax: number;
  /** Elevation range (elevationMax - elevationMin) */
  elevationRange: number;
  /** Discrete Turbo palette hex strings */
  palette: string[];
  /** Underlying sparse chunked 3D volume (§3.1) */
  sparseGrid?: SparseChunkedGrid;
}

/**
 * 3D Sparse Chunked Voxel Grid (§3.1 Phase 1).
 * Stores voxels sparsely in 16x16x16 typed-array chunks.
 * Empty sky or underground space consumes zero memory.
 */
export class SparseChunkedGrid {
  static readonly CHUNK_SIZE = 16;
  readonly chunks = new Map<string, Uint8Array>();

  cellWidth: number;
  cellDepth: number;
  cellHeight: number;
  worldWidth: number;
  worldDepth: number;

  cols: number = 0;
  rows: number = 0;
  heightLevels: number = 0;
  elevationMin: number = 0;
  elevationMax: number = 0;
  elevationRange: number = 0;

  constructor(
    cellWidth: number = 1.0,
    cellDepth: number = 1.0,
    cellHeight: number = 1.0,
    worldWidth: number = 10.0,
    worldDepth: number = 10.0
  ) {
    this.cellWidth = cellWidth;
    this.cellDepth = cellDepth;
    this.cellHeight = cellHeight;
    this.worldWidth = worldWidth;
    this.worldDepth = worldDepth;
  }

  static key(cx: number, cy: number, cz: number): string {
    return `${cx},${cy},${cz}`;
  }

  getVoxel(gx: number, gy: number, gz: number): VoxelState {
    const S = SparseChunkedGrid.CHUNK_SIZE;
    const cx = Math.floor(gx / S);
    const cy = Math.floor(gy / S);
    const cz = Math.floor(gz / S);
    const chunk = this.chunks.get(SparseChunkedGrid.key(cx, cy, cz));
    if (!chunk) return VoxelState.FREE;
    const lx = ((gx % S) + S) % S;
    const ly = ((gy % S) + S) % S;
    const lz = ((gz % S) + S) % S;
    return chunk[lx + S * (ly + S * lz)] as VoxelState;
  }

  setVoxel(gx: number, gy: number, gz: number, state: VoxelState): void {
    const S = SparseChunkedGrid.CHUNK_SIZE;
    const cx = Math.floor(gx / S);
    const cy = Math.floor(gy / S);
    const cz = Math.floor(gz / S);
    const k = SparseChunkedGrid.key(cx, cy, cz);
    let chunk = this.chunks.get(k);
    if (!chunk) {
      if (state === VoxelState.FREE) return;
      chunk = new Uint8Array(S * S * S);
      this.chunks.set(k, chunk);
    }
    const lx = ((gx % S) + S) % S;
    const ly = ((gy % S) + S) % S;
    const lz = ((gz % S) + S) % S;
    chunk[lx + S * (ly + S * lz)] = state;
  }

  /**
   * Generates a coarser Multi-Resolution Level-of-Detail (LOD).
   * Level 1 = 2:1 downsampling, Level 2 = 4:1 downsampling.
   */
  getLOD(level: number = 1): SparseChunkedGrid {
    if (level <= 0) return this;
    const factor = Math.pow(2, level);
    const lod = new SparseChunkedGrid(
      this.cellWidth * factor,
      this.cellDepth * factor,
      this.cellHeight * factor,
      this.worldWidth,
      this.worldDepth
    );
    lod.elevationMin = this.elevationMin;
    lod.elevationMax = this.elevationMax;
    lod.elevationRange = this.elevationRange;

    const S = SparseChunkedGrid.CHUNK_SIZE;
    for (const [key, chunk] of this.chunks.entries()) {
      const parts = key.split(',').map(Number);
      const cx = parts[0];
      const cy = parts[1];
      const cz = parts[2];
      for (let lz = 0; lz < S; lz++) {
        for (let ly = 0; ly < S; ly++) {
          for (let lx = 0; lx < S; lx++) {
            const val = chunk[lx + S * (ly + S * lz)];
            if (val !== VoxelState.FREE) {
              const gx = Math.floor((cx * S + lx) / factor);
              const gy = Math.floor((cy * S + ly) / factor);
              const gz = Math.floor((cz * S + lz) / factor);
              lod.setVoxel(gx, gy, gz, val as VoxelState);
            }
          }
        }
      }
    }
    return lod;
  }

  /**
   * Memory statistics and savings ratio vs dense 3D bounding box.
   */
  getMemoryStats(): {
    allocatedChunks: number;
    allocatedVoxels: number;
    allocatedMemoryKB: number;
    denseEquivalentVoxels: number;
    sparseSavingsPercent: number;
  } {
    const S = SparseChunkedGrid.CHUNK_SIZE;
    const allocatedChunks = this.chunks.size;
    const allocatedVoxels = allocatedChunks * (S * S * S);

    let denseEquivalentVoxels = 0;
    if (allocatedChunks > 0) {
      let minX = Infinity, maxX = -Infinity;
      let minY = Infinity, maxY = -Infinity;
      let minZ = Infinity, maxZ = -Infinity;
      for (const k of this.chunks.keys()) {
        const [cx, cy, cz] = k.split(',').map(Number);
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        if (cz < minZ) minZ = cz;
        if (cz > maxZ) maxZ = cz;
      }
      denseEquivalentVoxels = (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1) * (S * S * S);
    }

    const savings = denseEquivalentVoxels > 0
      ? Math.max(0, (1 - allocatedVoxels / denseEquivalentVoxels) * 100)
      : 0;

    return {
      allocatedChunks,
      allocatedVoxels,
      allocatedMemoryKB: Math.round(allocatedVoxels / 1024 * 100) / 100,
      denseEquivalentVoxels,
      sparseSavingsPercent: Math.round(savings * 10) / 10,
    };
  }

  /**
   * Post-processing pass to tag occluded columns as UNKNOWN (§3.2).
   */
  applyOcclusionMask(occlusionMask: boolean[][]): void {
    const maskRows = occlusionMask.length;
    const maskCols = occlusionMask[0]?.length ?? 0;
    if (maskRows === 0 || maskCols === 0) return;

    for (let r = 0; r < this.rows; r++) {
      const srcR = Math.floor((r * maskRows) / this.rows);
      for (let c = 0; c < this.cols; c++) {
        const srcC = Math.floor((c * maskCols) / this.cols);
        if (occlusionMask[srcR]?.[srcC]) {
          const S = SparseChunkedGrid.CHUNK_SIZE;
          const cx = Math.floor(c / S);
          const cz = Math.floor(r / S);
          const lx = ((c % S) + S) % S;
          const lz = ((r % S) + S) % S;

          for (let cy = 0; cy < 16; cy++) {
            const chunk = this.chunks.get(SparseChunkedGrid.key(cx, cy, cz));
            if (!chunk) continue;
            for (let ly = 0; ly < S; ly++) {
              const idx = lx + S * (ly + S * lz);
              if (chunk[idx] === VoxelState.OCCUPIED) {
                chunk[idx] = VoxelState.UNKNOWN;
              }
            }
          }
        }
      }
    }
  }

  getUnknownCount(): number {
    let count = 0;
    const S = SparseChunkedGrid.CHUNK_SIZE;
    for (const chunk of this.chunks.values()) {
      for (let i = 0; i < S * S * S; i++) {
        if (chunk[i] === VoxelState.UNKNOWN) count++;
      }
    }
    return count;
  }

  getOccupiedCount(): number {
    let count = 0;
    const S = SparseChunkedGrid.CHUNK_SIZE;
    for (const chunk of this.chunks.values()) {
      for (let i = 0; i < S * S * S; i++) {
        if (chunk[i] === VoxelState.OCCUPIED) count++;
      }
    }
    return count;
  }
}

/**
 * Samples a continuous Turbo-like colormap at normalized position t in [0, 1].
 * Matches the blue -> cyan -> green -> yellow -> red progression in mesh_builder.py.
 */
export function sampleTurboRgb(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  let r = 0;
  let g = 0;
  let b = 0;

  if (clamped < 0.25) {
    r = 0.0;
    g = clamped * 4.0;
    b = 1.0;
  } else if (clamped < 0.5) {
    r = 0.0;
    g = 1.0;
    b = 1.0 - (clamped - 0.25) * 4.0;
  } else if (clamped < 0.75) {
    r = (clamped - 0.5) * 4.0;
    g = 1.0;
    b = 0.0;
  } else {
    r = 1.0;
    g = 1.0 - (clamped - 0.75) * 4.0;
    b = 0.0;
  }

  return [r, g, b];
}

/**
 * Converts normalized [r, g, b] in [0, 1] to a 6-character hex string (#rrggbb).
 */
export function rgbToHex(rgb: [number, number, number]): string {
  const toHex = (c: number) => {
    const val = Math.max(0, Math.min(255, Math.round(c * 255)));
    return val.toString(16).padStart(2, '0');
  };
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;
}

/**
 * Generates an array of discrete Turbo palette colors for a specified band count.
 * Each band's color is sampled at the midpoint of its normalized bracket.
 */
export function generateTurboPalette(bandCount: number): string[] {
  const count = Math.max(1, Math.floor(bandCount));
  const palette: string[] = [];
  for (let i = 0; i < count; i++) {
    const t = (i + 0.5) / count;
    palette.push(rgbToHex(sampleTurboRgb(t)));
  }
  return palette;
}

/**
 * Pools a 2D height grid (dsm_raw) into a coarse block grid and quantizes
 * each block into discrete elevation bands with corresponding Turbo palette colors.
 *
 * @param dsmRaw - Row-major 2D array of elevations [row][col]
 * @param options - Configuration for target resolution, bands, footprint scaling, etc.
 * @returns VoxelGrid containing array of block descriptors, grid dimensions, and palette.
 */
export function voxelize(
  dsmRaw: number[][],
  options: VoxelizeOptions = {}
): VoxelGrid {
  if (!dsmRaw || dsmRaw.length === 0 || !dsmRaw[0] || dsmRaw[0].length === 0) {
    return {
      blocks: [],
      rows: 0,
      cols: 0,
      cellWidth: 0,
      cellDepth: 0,
      elevationMin: 0,
      elevationMax: 0,
      elevationRange: 0,
      palette: [],
    };
  }

  const srcRows = dsmRaw.length;
  const srcCols = dsmRaw[0].length;

  const targetRes = Math.max(4, Math.min(256, options.targetResolution ?? 64));
  const bandCount = Math.max(2, Math.min(32, options.bandCount ?? 8));
  const gapRatio = Math.max(0.5, Math.min(1.0, options.gapRatio ?? 0.90));

  // Preserve aspect ratio when determining target rows and columns
  let cols: number;
  let rows: number;
  if (srcCols >= srcRows) {
    cols = targetRes;
    rows = Math.max(1, Math.round(targetRes * (srcRows / srcCols)));
  } else {
    rows = targetRes;
    cols = Math.max(1, Math.round(targetRes * (srcCols / srcRows)));
  }

  // World dimensions matching Three.js PlaneGeometry(w * 0.1, h * 0.1)
  const worldW = options.worldWidth ?? srcCols * 0.1;
  const worldD = options.worldDepth ?? srcRows * 0.1;

  const cellWidth = (worldW / cols) * gapRatio;
  const cellDepth = (worldD / rows) * gapRatio;

  const palette = generateTurboPalette(bandCount);

  // First pass: pool elevations using area-averaging
  const pooledGrid: number[][] = [];
  let foundMin = Infinity;
  let foundMax = -Infinity;

  for (let r = 0; r < rows; r++) {
    const rowVals: number[] = [];
    const srcR0 = Math.floor((r * srcRows) / rows);
    const srcR1 = Math.max(srcR0 + 1, Math.floor(((r + 1) * srcRows) / rows));

    for (let c = 0; c < cols; c++) {
      const srcC0 = Math.floor((c * srcCols) / cols);
      const srcC1 = Math.max(srcC0 + 1, Math.floor(((c + 1) * srcCols) / cols));

      let sum = 0;
      let count = 0;

      for (let y = srcR0; y < srcR1; y++) {
        const srcRow = dsmRaw[y];
        if (!srcRow) continue;
        for (let x = srcC0; x < srcC1; x++) {
          const val = srcRow[x];
          if (val !== undefined && !Number.isNaN(val) && Number.isFinite(val)) {
            sum += val;
            count++;
          }
        }
      }

      const avg = count > 0 ? sum / count : 0;
      rowVals.push(avg);
      if (avg < foundMin) foundMin = avg;
      if (avg > foundMax) foundMax = avg;
    }
    pooledGrid.push(rowVals);
  }

  if (foundMin === Infinity) foundMin = 0;
  if (foundMax === -Infinity) foundMax = 1;

  // Use provided min/max overrides if specified
  const elevMin = options.elevationMin ?? foundMin;
  const elevMax = options.elevationMax ?? foundMax;
  const elevRange = Math.max(1e-6, elevMax - elevMin);

  // Second pass: construct VoxelBlocks with world coordinates and quantized colors
  const blocks: VoxelBlock[] = [];
  const halfWorldW = worldW / 2;
  const halfWorldD = worldD / 2;
  const stepX = worldW / cols;
  const stepZ = worldD / rows;

  for (let r = 0; r < rows; r++) {
    const posZ = -halfWorldD + (r + 0.5) * stepZ;
    const srcR0 = Math.floor((r * srcRows) / rows);
    const srcR1 = Math.max(srcR0 + 1, Math.floor(((r + 1) * srcRows) / rows));

    for (let c = 0; c < cols; c++) {
      const srcC0 = Math.floor((c * srcCols) / cols);
      const srcC1 = Math.max(srcC0 + 1, Math.floor(((c + 1) * srcCols) / cols));

      let isOccluded = false;
      if (options.occlusionMask && options.occlusionMask.length > 0) {
        let occCount = 0;
        let totalCount = 0;
        for (let y = srcR0; y < srcR1; y++) {
          const rowMask = options.occlusionMask[y];
          if (!rowMask) continue;
          for (let x = srcC0; x < srcC1; x++) {
            if (rowMask[x]) occCount++;
            totalCount++;
          }
        }
        if (totalCount > 0 && occCount / totalCount >= 0.5) {
          isOccluded = true;
        }
      }

      const rawH = pooledGrid[r][c];
      const normH = Math.max(0, Math.min(1, (rawH - elevMin) / elevRange));
      const bandIndex = Math.min(bandCount - 1, Math.floor(normH * bandCount));
      const posX = -halfWorldW + (c + 0.5) * stepX;

      const state = isOccluded ? VoxelState.UNKNOWN : VoxelState.OCCUPIED;
      // Coordinated palette: Unknown / unobserved voxels render with purple #8B5CF6 (§3.2)
      const colorHex = isOccluded ? '#8B5CF6' : (palette[bandIndex] ?? palette[0]);
      const colorRgb: [number, number, number] = isOccluded
        ? [0.545, 0.361, 0.965]
        : sampleTurboRgb((bandIndex + 0.5) / bandCount);

      blocks.push({
        gridX: c,
        gridZ: r,
        posX,
        posZ,
        rawHeight: rawH,
        normalizedHeight: normH,
        bandIndex,
        colorHex,
        colorRgb,
        state,
      });
    }
  }

  return {
    blocks,
    rows,
    cols,
    cellWidth,
    cellDepth,
    elevationMin: elevMin,
    elevationMax: elevMax,
    elevationRange: elevRange,
    palette,
  };
}

/**
 * Builds a true 3D SparseChunkedGrid from 2D DSM elevation data (§3.1 Phase 1).
 * Populates 16x16x16 chunked volumes with solid ground columns up to observed heights.
 */
export function buildSparseVoxelGrid(
  dsmRaw: number[][],
  options: VoxelizeOptions = {}
): SparseChunkedGrid {
  const vGrid = voxelize(dsmRaw, options);
  const cellHeight = Math.max(0.1, (vGrid.elevationRange / (options.bandCount ?? 8)) || 1.0);
  const worldW = options.worldWidth ?? (dsmRaw[0]?.length ?? 100) * 0.1;
  const worldD = options.worldDepth ?? (dsmRaw.length ?? 100) * 0.1;

  const sparse = new SparseChunkedGrid(
    vGrid.cellWidth,
    vGrid.cellDepth,
    cellHeight,
    worldW,
    worldD
  );
  sparse.cols = vGrid.cols;
  sparse.rows = vGrid.rows;
  sparse.elevationMin = vGrid.elevationMin;
  sparse.elevationMax = vGrid.elevationMax;
  sparse.elevationRange = vGrid.elevationRange;

  for (const block of vGrid.blocks) {
    const heightIdx = Math.max(1, Math.round((block.rawHeight - vGrid.elevationMin) / cellHeight));
    for (let y = 0; y < heightIdx; y++) {
      sparse.setVoxel(block.gridX, y, block.gridZ, block.state);
    }
  }

  return sparse;
}

/**
 * Computes quantized world-space surface height for a voxel block (§3),
 * matching VoxelMesh instance height in VoxelTerrain.tsx exactly.
 */
export function quantizeVoxelHeightWorld(
  rawH: number,
  elevationMin: number,
  elevationRange: number,
  verticalScale: number = 1.0,
  isRelative: boolean = false
): number {
  const normH = Math.max(0, Math.min(1, (rawH - elevationMin) / Math.max(1e-6, elevationRange)));
  return isRelative
    ? Math.max(0.12, normH * 18.0 * verticalScale)
    : Math.max(0.12, (rawH - elevationMin) * verticalScale * 0.1);
}

