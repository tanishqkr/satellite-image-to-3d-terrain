import React, { useState, useRef, useCallback } from 'react';
import { Grid3X3, X, Upload, Sparkles, Image as ImageIcon, Trash2, Check, RefreshCw } from 'lucide-react';
import { computeMatrixLayout, MatrixTileInput } from '../lib/matrixCompositor';

interface MatrixModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSynthesize: (compositeFile: File) => void;
}

interface MatrixCell {
  file?: File;
  previewUrl?: string;
  width?: number;
  height?: number;
  name?: string;
}

export default function MatrixModal({ isOpen, onClose, onSynthesize }: MatrixModalProps) {
  const [cols, setCols] = useState<number>(2);
  const [rows, setRows] = useState<number>(2);
  const [cells, setCells] = useState<Record<string, MatrixCell>>({});
  const [showBoundaries, setShowBoundaries] = useState<boolean>(true);
  const [isSynthesizing, setIsSynthesizing] = useState<boolean>(false);
  const [sampleLoading, setSampleLoading] = useState<boolean>(false);
  const [dragOverCell, setDragOverCell] = useState<string | null>(null);

  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  if (!isOpen) return null;

  const getCellKey = (r: number, c: number) => `${r}_${c}`;

  const handleCellFile = (r: number, c: number, file: File) => {
    const key = getCellKey(r, c);
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setCells((prev) => ({
        ...prev,
        [key]: {
          file,
          previewUrl: url,
          width: img.naturalWidth,
          height: img.naturalHeight,
          name: file.name,
        },
      }));
    };
    img.src = url;
  };

  const handleCellDrop = (e: React.DragEvent, r: number, c: number) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOverCell(null);
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith('image/') || file.name.endsWith('.tif') || file.name.endsWith('.tiff'))) {
      handleCellFile(r, c, file);
    }
  };

  const handleClearCell = (r: number, c: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const key = getCellKey(r, c);
    setCells((prev) => {
      const next = { ...prev };
      if (next[key]?.previewUrl) {
        URL.revokeObjectURL(next[key].previewUrl!);
      }
      delete next[key];
      return next;
    });
  };

  // Load a 2x2 matrix sliced from existing public satellite asset for 1-click verification
  const loadSampleMatrix = async () => {
    try {
      setSampleLoading(true);
      setCols(2);
      setRows(2);

      const res = await fetch('/sample_satellite_urban.png');
      if (!res.ok) throw new Error('Sample satellite asset not found');
      const blob = await res.blob();
      const img = new Image();
      const blobUrl = URL.createObjectURL(blob);

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = reject;
        img.src = blobUrl;
      });

      const halfW = Math.floor(img.naturalWidth / 2);
      const halfH = Math.floor(img.naturalHeight / 2);
      const newCells: Record<string, MatrixCell> = {};

      const quadrantNames = [
        ['NW Sector (Urban Core)', 'NE Sector (Industrial Park)'],
        ['SW Sector (Waterfront)', 'SE Sector (Logistics Hub)'],
      ];

      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const offscreen = document.createElement('canvas');
          offscreen.width = halfW;
          offscreen.height = halfH;
          const ctx = offscreen.getContext('2d');
          if (!ctx) continue;

          ctx.drawImage(img, c * halfW, r * halfH, halfW, halfH, 0, 0, halfW, halfH);

          const quadBlob = await new Promise<Blob | null>((resBlob) =>
            offscreen.toBlob(resBlob, 'image/png')
          );

          if (quadBlob) {
            const quadFile = new File([quadBlob], `quadrant_${r}_${c}.png`, { type: 'image/png' });
            const quadUrl = URL.createObjectURL(quadBlob);
            newCells[getCellKey(r, c)] = {
              file: quadFile,
              previewUrl: quadUrl,
              width: halfW,
              height: halfH,
              name: quadrantNames[r][c],
            };
          }
        }
      }

      setCells(newCells);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error('Failed to load sample matrix:', err);
    } finally {
      setSampleLoading(false);
    }
  };

  const synthesizeMosaic = async () => {
    // Collect all valid cells
    const activeCells = Object.entries(cells).filter(([_, cell]) => cell && cell.file && cell.previewUrl);
    if (activeCells.length === 0) {
      alert('Please upload at least one image tile into the matrix grid.');
      return;
    }

    setIsSynthesizing(true);

    try {
      // 1. Calculate column widths and row heights using computeMatrixLayout
      const tileInputs: MatrixTileInput[] = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cell = cells[getCellKey(r, c)];
          if (cell?.width && cell?.height) {
            tileInputs.push({ col: c, row: r, width: cell.width, height: cell.height });
          }
        }
      }

      const layout = computeMatrixLayout(cols, rows, tileInputs);
      const { colWidths, rowHeights, totalWidth, totalHeight } = layout;

      // Clamp max composite resolution for high-speed client-side processing
      const maxDim = 2048;
      const scale = Math.min(1.0, maxDim / Math.max(totalWidth, totalHeight));
      const targetW = Math.round(totalWidth * scale);
      const targetH = Math.round(totalHeight * scale);

      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not initialize 2D canvas context');

      // Neutral dark background for unpopulated/irregular cells
      ctx.fillStyle = '#10141a';
      ctx.fillRect(0, 0, targetW, targetH);

      // Load all cell images as HTMLImageElements
      const loadedImages: Record<string, HTMLImageElement> = {};
      await Promise.all(
        activeCells.map(async ([key, cell]) => {
          const img = new Image();
          await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = reject;
            img.src = cell.previewUrl!;
          });
          loadedImages[key] = img;
        })
      );

      // 2. Draw each tile into its allocated grid bounds
      let currY = 0;
      for (let r = 0; r < rows; r++) {
        const cellH = Math.round(rowHeights[r] * scale);
        let currX = 0;
        for (let c = 0; c < cols; c++) {
          const cellW = Math.round(colWidths[c] * scale);
          const key = getCellKey(r, c);
          const img = loadedImages[key];

          if (img) {
            // Draw tile scaled to fill allocated cell area
            ctx.drawImage(img, currX, currY, cellW, cellH);
          } else {
            // Unpopulated cell: subtle checkered or textured filler
            ctx.fillStyle = '#141a22';
            ctx.fillRect(currX, currY, cellW, cellH);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
            ctx.lineWidth = 1;
            ctx.strokeRect(currX, currY, cellW, cellH);
          }
          currX += cellW;
        }
        currY += cellH;
      }

      // 3. Draw subtle tile seam lines if requested
      if (showBoundaries) {
        // Vertical column dividers
        let seamX = 0;
        for (let c = 0; c < cols - 1; c++) {
          seamX += Math.round(colWidths[c] * scale);

          // Dark shadow edge
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
          ctx.lineWidth = 2.5;
          ctx.moveTo(seamX - 0.5, 0);
          ctx.lineTo(seamX - 0.5, targetH);
          ctx.stroke();

          // Subtle bright hairline guide
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.40)';
          ctx.lineWidth = 1.0;
          ctx.moveTo(seamX, 0);
          ctx.lineTo(seamX, targetH);
          ctx.stroke();
        }

        // Horizontal row dividers
        let seamY = 0;
        for (let r = 0; r < rows - 1; r++) {
          seamY += Math.round(rowHeights[r] * scale);

          // Dark shadow edge
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
          ctx.lineWidth = 2.5;
          ctx.moveTo(0, seamY - 0.5);
          ctx.lineTo(targetW, seamY - 0.5);
          ctx.stroke();

          // Subtle bright hairline guide
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.40)';
          ctx.lineWidth = 1.0;
          ctx.moveTo(0, seamY);
          ctx.lineTo(targetW, seamY);
          ctx.stroke();
        }
      }

      // 4. Export composite canvas as PNG File
      const compositeBlob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/png', 0.95)
      );

      if (!compositeBlob) throw new Error('Failed to generate composite image blob');

      const compositeFile = new File(
        [compositeBlob],
        `matrix_mosaic_${cols}x${rows}_${targetW}x${targetH}.png`,
        { type: 'image/png' }
      );

      onSynthesize(compositeFile);
      onClose();
    } catch (err: any) {
      console.error('Matrix synthesis error:', err);
      alert(`Matrix synthesis failed: ${err.message || err}`);
    } finally {
      setIsSynthesizing(false);
    }
  };

  const totalPopulated = Object.keys(cells).filter((k) => cells[k]?.file).length;
  const totalSlots = cols * rows;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
      <div className="bg-[#0B0E14] border border-white/20 w-full max-w-5xl max-h-[92vh] flex flex-col shadow-2xl overflow-hidden text-neutral-200">
        {/* Header */}
        <div className="px-6 py-5 border-b border-white/10 flex items-center justify-between bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <div className="p-2 border border-cyan-500/40 bg-cyan-500/10 text-cyan-400">
              <Grid3X3 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-black tracking-tight text-white uppercase font-mono flex items-center gap-2">
                Matrix Grid Mosaic Compositor
                <span className="text-[10px] px-2 py-0.5 border border-cyan-500/40 bg-cyan-500/10 text-cyan-300 font-mono">
                  MULTI-TILE DSM
                </span>
              </h2>
              <p className="text-xs text-neutral-400 font-mono">
                Assemble multi-quadrant aerial/satellite tiles into a continuous 3D elevation model with subtle seam guides.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-neutral-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Configuration Bar */}
        <div className="px-6 py-4 border-b border-white/10 bg-[#07090D] flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-center gap-6">
            {/* Columns (X) */}
            <div className="flex items-center gap-2">
              <span className="text-neutral-400 uppercase">COLUMNS (X):</span>
              <div className="flex items-center border border-white/20 bg-white/5">
                {[1, 2, 3, 4].map((c) => (
                  <button
                    key={c}
                    onClick={() => setCols(c)}
                    className={`px-3 py-1 font-bold transition-colors ${
                      cols === c ? 'bg-cyan-500 text-black' : 'text-neutral-300 hover:bg-white/10'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>

            {/* Rows (Y) */}
            <div className="flex items-center gap-2">
              <span className="text-neutral-400 uppercase">ROWS (Y):</span>
              <div className="flex items-center border border-white/20 bg-white/5">
                {[1, 2, 3, 4].map((r) => (
                  <button
                    key={r}
                    onClick={() => setRows(r)}
                    className={`px-3 py-1 font-bold transition-colors ${
                      rows === r ? 'bg-cyan-500 text-black' : 'text-neutral-300 hover:bg-white/10'
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            {/* Subtle Seam Boundaries */}
            <label className="flex items-center gap-2 cursor-pointer select-none text-neutral-300 hover:text-white">
              <input
                type="checkbox"
                checked={showBoundaries}
                onChange={(e) => setShowBoundaries(e.target.checked)}
                className="accent-cyan-400 cursor-pointer w-4 h-4"
              />
              <span>SHOW SUBTLE TILE BOUNDARIES</span>
            </label>
          </div>

          {/* Preset Buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={loadSampleMatrix}
              disabled={sampleLoading}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 border border-white/20 bg-white/5 hover:bg-white/10 hover:border-white/40 text-neutral-200 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 text-cyan-400 ${sampleLoading ? 'animate-spin' : ''}`} />
              <span>{sampleLoading ? 'SLICING SAMPLE...' : 'LOAD SAMPLE 2×2 MATRIX'}</span>
            </button>
          </div>
        </div>

        {/* Matrix Grid Canvas Viewport */}
        <div className="flex-1 p-6 overflow-y-auto bg-[#07080B]">
          <div
            className="grid gap-3 max-w-4xl mx-auto min-h-[360px]"
            style={{
              gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
              gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
            }}
          >
            {Array.from({ length: rows }).map((_, r) =>
              Array.from({ length: cols }).map((_, c) => {
                const key = getCellKey(r, c);
                const cell = cells[key];
                const isOver = dragOverCell === key;

                return (
                  <div
                    key={key}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOverCell(key);
                    }}
                    onDragLeave={() => setDragOverCell(null)}
                    onDrop={(e) => handleCellDrop(e, r, c)}
                    onClick={() => fileInputRefs.current[key]?.click()}
                    className={`
                      relative group border aspect-square flex flex-col items-center justify-center p-3 text-center cursor-pointer transition-all overflow-hidden
                      ${
                        cell?.previewUrl
                          ? 'border-white/40 bg-black/40'
                          : isOver
                          ? 'border-cyan-400 bg-cyan-500/10'
                          : 'border-white/15 hover:border-white/40 bg-white/[0.02]'
                      }
                    `}
                  >
                    <input
                      ref={(el) => (fileInputRefs.current[key] = el)}
                      type="file"
                      accept=".png,.jpg,.jpeg,.tif,.tiff"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleCellFile(r, c, file);
                        e.target.value = '';
                      }}
                      className="hidden"
                    />

                    {/* Coordinate Tag */}
                    <div className="absolute top-2 left-2 z-10 px-2 py-0.5 border border-white/20 bg-black/70 text-[10px] font-mono text-neutral-400">
                      [X:{c}, Y:{r}]
                    </div>

                    {cell?.previewUrl ? (
                      <>
                        <img
                          src={cell.previewUrl}
                          alt={`Cell ${r}_${c}`}
                          className="absolute inset-0 w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30 pointer-events-none" />

                        {/* Remove Button */}
                        <button
                          onClick={(e) => handleClearCell(r, c, e)}
                          title="Remove image from cell"
                          className="absolute top-2 right-2 z-10 p-1 bg-black/70 border border-white/30 text-white hover:bg-red-500/80 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>

                        {/* Cell Metadata Banner */}
                        <div className="absolute bottom-2 left-2 right-2 z-10 text-left font-mono">
                          <div className="text-[11px] text-white font-bold truncate">
                            {cell.name || `Tile [${c}, ${r}]`}
                          </div>
                          <div className="text-[9px] text-cyan-300">
                            {cell.width} × {cell.height} px
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="space-y-2 text-neutral-500 group-hover:text-neutral-300 transition-colors">
                        <Upload className="w-6 h-6 mx-auto stroke-1 text-neutral-400 group-hover:text-cyan-400 transition-colors" />
                        <div className="text-[11px] font-mono uppercase tracking-wider font-semibold">
                          Tile [{c}, {r}]
                        </div>
                        <div className="text-[10px] font-mono text-neutral-500">
                          Drop image or click
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-white/10 bg-[#07090D] flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
          <div className="flex items-center gap-3 text-neutral-400">
            <span>GRID: <strong className="text-white">{cols} × {rows}</strong></span>
            <span>·</span>
            <span>TILES POPULATED: <strong className="text-cyan-400">{totalPopulated}</strong> / {totalSlots}</span>
            {showBoundaries && (
              <>
                <span>·</span>
                <span className="text-emerald-400">SEAM GUIDES ACTIVE</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-5 py-2.5 border border-white/20 hover:border-white/40 text-neutral-300 hover:text-white transition-colors"
            >
              CANCEL
            </button>
            <button
              onClick={synthesizeMosaic}
              disabled={isSynthesizing || totalPopulated === 0}
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-black font-bold tracking-wider uppercase transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] disabled:opacity-40 disabled:pointer-events-none"
            >
              <Sparkles className={`w-4 h-4 ${isSynthesizing ? 'animate-spin' : ''}`} />
              <span>{isSynthesizing ? 'SYNTHESIZING...' : 'SYNTHESIZE MATRIX TERRAIN'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
