/**
 * Matrix Grid Mosaic Compositor Utility
 * Computes non-uniform grid dimension packing, cell layout coordinates,
 * and boundary seam lines for multi-tile aerial/satellite maps.
 */

export interface MatrixTileInput {
  col: number;
  row: number;
  width: number;
  height: number;
}

export interface MatrixLayoutResult {
  cols: number;
  rows: number;
  colWidths: number[];
  rowHeights: number[];
  totalWidth: number;
  totalHeight: number;
  cellRects: Record<string, { x: number; y: number; width: number; height: number }>;
  colSeams: number[];
  rowSeams: number[];
}

export function computeMatrixLayout(
  cols: number,
  rows: number,
  tiles: MatrixTileInput[],
  defaultDim = 512
): MatrixLayoutResult {
  const safeCols = Math.max(1, cols);
  const safeRows = Math.max(1, rows);

  const colWidths = new Array<number>(safeCols).fill(0);
  const rowHeights = new Array<number>(safeRows).fill(0);

  for (const tile of tiles) {
    if (tile.col >= 0 && tile.col < safeCols && tile.width > 0) {
      colWidths[tile.col] = Math.max(colWidths[tile.col], tile.width);
    }
    if (tile.row >= 0 && tile.row < safeRows && tile.height > 0) {
      rowHeights[tile.row] = Math.max(rowHeights[tile.row], tile.height);
    }
  }

  // If any column or row had no tiles, fill with defaultDim
  for (let c = 0; c < safeCols; c++) {
    if (colWidths[c] <= 0) colWidths[c] = defaultDim;
  }
  for (let r = 0; r < safeRows; r++) {
    if (rowHeights[r] <= 0) rowHeights[r] = defaultDim;
  }

  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const totalHeight = rowHeights.reduce((a, b) => a + b, 0);

  const cellRects: Record<string, { x: number; y: number; width: number; height: number }> = {};
  let curY = 0;
  for (let r = 0; r < safeRows; r++) {
    let curX = 0;
    for (let c = 0; c < safeCols; c++) {
      cellRects[`${r}_${c}`] = {
        x: curX,
        y: curY,
        width: colWidths[c],
        height: rowHeights[r],
      };
      curX += colWidths[c];
    }
    curY += rowHeights[r];
  }

  const colSeams: number[] = [];
  let seamX = 0;
  for (let c = 0; c < safeCols - 1; c++) {
    seamX += colWidths[c];
    colSeams.push(seamX);
  }

  const rowSeams: number[] = [];
  let seamY = 0;
  for (let r = 0; r < safeRows - 1; r++) {
    seamY += rowHeights[r];
    rowSeams.push(seamY);
  }

  return {
    cols: safeCols,
    rows: safeRows,
    colWidths,
    rowHeights,
    totalWidth,
    totalHeight,
    cellRects,
    colSeams,
    rowSeams,
  };
}
