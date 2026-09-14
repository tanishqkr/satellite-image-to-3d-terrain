import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeMatrixLayout } from './matrixCompositor.ts';

test('computeMatrixLayout handles uniform 2x2 grid correctly', () => {
  const tiles = [
    { col: 0, row: 0, width: 512, height: 512 },
    { col: 1, row: 0, width: 512, height: 512 },
    { col: 0, row: 1, width: 512, height: 512 },
    { col: 1, row: 1, width: 512, height: 512 },
  ];

  const layout = computeMatrixLayout(2, 2, tiles);
  assert.equal(layout.totalWidth, 1024);
  assert.equal(layout.totalHeight, 1024);
  assert.deepEqual(layout.colWidths, [512, 512]);
  assert.deepEqual(layout.rowHeights, [512, 512]);
  assert.deepEqual(layout.colSeams, [512]);
  assert.deepEqual(layout.rowSeams, [512]);

  assert.deepEqual(layout.cellRects['0_0'], { x: 0, y: 0, width: 512, height: 512 });
  assert.deepEqual(layout.cellRects['0_1'], { x: 512, y: 0, width: 512, height: 512 });
  assert.deepEqual(layout.cellRects['1_0'], { x: 0, y: 512, width: 512, height: 512 });
  assert.deepEqual(layout.cellRects['1_1'], { x: 512, y: 512, width: 512, height: 512 });
});

test('computeMatrixLayout accommodates different image sizes per row and column', () => {
  // Col 0 max is 820, Col 1 max is 1000
  // Row 0 max is 620, Row 1 max is 700
  const tiles = [
    { col: 0, row: 0, width: 800, height: 600 },
    { col: 1, row: 0, width: 1000, height: 620 },
    { col: 0, row: 1, width: 820, height: 700 },
    { col: 1, row: 1, width: 980, height: 680 },
  ];

  const layout = computeMatrixLayout(2, 2, tiles);
  assert.equal(layout.colWidths[0], 820);
  assert.equal(layout.colWidths[1], 1000);
  assert.equal(layout.rowHeights[0], 620);
  assert.equal(layout.rowHeights[1], 700);

  assert.equal(layout.totalWidth, 1820);
  assert.equal(layout.totalHeight, 1320);
  assert.deepEqual(layout.colSeams, [820]);
  assert.deepEqual(layout.rowSeams, [620]);
});

test('computeMatrixLayout handles non-square aspect ratios (e.g. 3x1 wide panorama)', () => {
  const tiles = [
    { col: 0, row: 0, width: 600, height: 400 },
    { col: 1, row: 0, width: 700, height: 400 },
    { col: 2, row: 0, width: 650, height: 400 },
  ];

  const layout = computeMatrixLayout(3, 1, tiles);
  assert.equal(layout.cols, 3);
  assert.equal(layout.rows, 1);
  assert.equal(layout.totalWidth, 1950);
  assert.equal(layout.totalHeight, 400);
  assert.deepEqual(layout.colSeams, [600, 1300]);
  assert.deepEqual(layout.rowSeams, []);
});

test('computeMatrixLayout supports partial / empty cells gracefully with fallback dimensions', () => {
  // L-shape: cell [1, 1] is empty
  const tiles = [
    { col: 0, row: 0, width: 600, height: 500 },
    { col: 1, row: 0, width: 800, height: 500 },
    { col: 0, row: 1, width: 600, height: 600 },
  ];

  const layout = computeMatrixLayout(2, 2, tiles, 512);
  assert.equal(layout.colWidths[0], 600);
  assert.equal(layout.colWidths[1], 800);
  assert.equal(layout.rowHeights[0], 500);
  assert.equal(layout.rowHeights[1], 600);

  assert.equal(layout.totalWidth, 1400);
  assert.equal(layout.totalHeight, 1100);
  assert.ok(layout.cellRects['1_1']);
  assert.equal(layout.cellRects['1_1'].x, 600);
  assert.equal(layout.cellRects['1_1'].y, 500);
});
