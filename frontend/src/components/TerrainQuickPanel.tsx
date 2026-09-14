import React, { useEffect } from 'react';
import { X, Boxes, Mountain, Sliders, Layers, Sun, Moon } from 'lucide-react';
import { TerrainSettings } from '../hooks/useTerrainSettings';
import { generateTurboPalette } from '../lib/voxelize';

export interface TerrainQuickPanelProps {
  isOpen: boolean;
  onClose: () => void;
  settings: TerrainSettings;
}

/**
 * TerrainQuickPanel (§5):
 * Dedicated floating overlay panel accessible mid-flight (and in spectator studio)
 * via header button or [Key P]. Factors terrain & voxel settings to allow tweaking
 * without leaving an active flight session.
 */
export default function TerrainQuickPanel({
  isOpen,
  onClose,
  settings,
}: TerrainQuickPanelProps) {
  const {
    verticalScale,
    setVerticalScale,
    showContours,
    setShowContours,
    contourInterval,
    setContourInterval,
    renderMode,
    setRenderMode,
    voxelBands,
    setVoxelBands,
    voxelResolution,
    setVoxelResolution,
    environmentTheme,
    setEnvironmentTheme,
  } = settings;

  // ESC key to close if open
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' || e.key.toLowerCase() === 'p') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const palette = generateTurboPalette(voxelBands);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm pointer-events-auto select-none font-mono">
      <div className="w-full max-w-md bg-[#0A0D14] border border-white/20 shadow-2xl p-6 text-white space-y-6 relative max-h-[90vh] overflow-y-auto">
        {/* Panel Header */}
        <div className="flex items-start justify-between border-b border-white/15 pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-[10px] text-[#38BDF8] font-bold tracking-widest uppercase">
              <Sliders className="w-3.5 h-3.5" />
              <span>TERRAIN & VOXEL QUICK PANEL [P]</span>
            </div>
            <h3 className="text-base font-black uppercase tracking-tight text-white">
              Surface Geometry Tuning
            </h3>
            <p className="text-[11px] text-neutral-400 font-normal">
              Adjust mesh coarseness and vertical relief mid-flight.
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 hover:bg-white/10 text-neutral-400 hover:text-white transition-colors border border-white/10"
            title="Close Quick Panel [ESC / P]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* 1. 3D Render Mode Toggle */}
        <div className="space-y-2.5">
          <div className="flex justify-between items-center text-xs">
            <span className="text-neutral-400 uppercase tracking-wider text-[11px] font-bold">
              3D Surface Mode
            </span>
            <span className="text-[10px] text-[#38BDF8] uppercase tracking-wider font-bold">
              {renderMode === 'voxel' ? 'BLOCK EXTRUSION' : 'PHOTOREAL DISPLACEMENT'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setRenderMode('voxel')}
              className={`py-2 px-3 flex items-center justify-center gap-2 text-xs font-bold border transition-all ${
                renderMode === 'voxel'
                  ? 'bg-white text-black border-white shadow-lg'
                  : 'bg-transparent text-neutral-400 border-white/15 hover:border-white/40 hover:text-white'
              }`}
            >
              <Boxes className="w-3.5 h-3.5" />
              VOXEL / BLOCKS
            </button>

            <button
              type="button"
              onClick={() => setRenderMode('smooth')}
              className={`py-2 px-3 flex items-center justify-center gap-2 text-xs font-bold border transition-all ${
                renderMode === 'smooth'
                  ? 'bg-white text-black border-white shadow-lg'
                  : 'bg-transparent text-neutral-400 border-white/15 hover:border-white/40 hover:text-white'
              }`}
            >
              <Mountain className="w-3.5 h-3.5" />
              SMOOTH / MESH
            </button>
          </div>
        </div>

        {/* 2. Voxel Detail Tuning (Only in Voxel mode) */}
        {renderMode === 'voxel' && (
          <div className="p-3.5 bg-white/[0.03] border border-white/15 space-y-4">
            <div className="text-[10px] font-bold uppercase tracking-widest text-[#38BDF8] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 bg-[#38BDF8]" />
              Voxel Block Tuning
            </div>

            {/* Discrete Band Count Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="text-neutral-400 text-[11px]">Elevation Bands</span>
                <span className="text-white font-black">{voxelBands} BANDS</span>
              </div>
              <input
                type="range"
                min={5}
                max={12}
                step={1}
                value={voxelBands}
                onChange={(e) => setVoxelBands(parseInt(e.target.value, 10))}
                className="w-full accent-[#38BDF8] cursor-pointer"
              />
              <div className="flex justify-between text-[9px] text-neutral-500">
                <span>5 (DISCRETE)</span>
                <span>8 (BALANCED)</span>
                <span>12 (DETAILED)</span>
              </div>

              {/* Realtime Turbo Palette Swatches */}
              <div className="pt-1">
                <div className="flex h-2 w-full gap-0.5 overflow-hidden border border-white/10">
                  {palette.map((hex, i) => (
                    <div
                      key={i}
                      className="flex-1 h-full"
                      style={{ backgroundColor: hex }}
                      title={`Band ${i + 1}: ${hex}`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Block Resolution Slider */}
            <div className="space-y-1.5 pt-2 border-t border-white/10">
              <div className="flex justify-between items-center text-xs">
                <span className="text-neutral-400 text-[11px]">Block Resolution</span>
                <span className="text-white font-black">{voxelResolution}×{voxelResolution}</span>
              </div>
              <input
                type="range"
                min={24}
                max={128}
                step={4}
                value={voxelResolution}
                onChange={(e) => setVoxelResolution(parseInt(e.target.value, 10))}
                className="w-full accent-[#38BDF8] cursor-pointer"
              />
              <div className="flex justify-between text-[9px] text-neutral-500">
                <span>24 (COARSE)</span>
                <span>64 (STANDARD)</span>
                <span>128 (ULTRA)</span>
              </div>
            </div>
          </div>
        )}

        {/* 3. Vertical Exaggeration Slider (Common to both modes) */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs">
            <span className="text-neutral-400 text-[11px] uppercase tracking-wider font-bold">
              Vertical Scale
            </span>
            <span className="text-white font-black text-sm">{verticalScale.toFixed(1)}×</span>
          </div>
          <input
            type="range"
            min={0.2}
            max={4.0}
            step={0.1}
            value={verticalScale}
            onChange={(e) => setVerticalScale(parseFloat(e.target.value))}
            className="w-full accent-white cursor-pointer"
          />
          <div className="flex justify-between text-[9px] text-neutral-500">
            <span>0.2× (FLAT)</span>
            <span>1.0× (TRUE 1:1)</span>
            <span>4.0× (EXAGGERATED)</span>
          </div>
        </div>

        {/* 4. Contour Lines Toggle & Interval */}
        <div className="space-y-2 pt-2 border-t border-white/15">
          <div className="flex justify-between items-center">
            <span className="text-neutral-400 text-[11px] uppercase tracking-wider font-bold flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-[#38BDF8]" />
              Topographic Contours
            </span>
            <button
              type="button"
              onClick={() => setShowContours(!showContours)}
              className={`px-2.5 py-1 text-[10px] font-bold border transition-colors ${
                showContours
                  ? 'bg-emerald-500 text-black border-emerald-400'
                  : 'bg-black/60 text-neutral-400 border-white/20 hover:text-white'
              }`}
            >
              {showContours ? 'ENABLED' : 'DISABLED'}
            </button>
          </div>

          {showContours && (
            <div className="space-y-1 pt-1">
              <div className="flex justify-between items-center text-[10px]">
                <span className="text-neutral-400">Contour Interval:</span>
                <span className="text-white font-bold">{contourInterval}m</span>
              </div>
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={contourInterval}
                onChange={(e) => setContourInterval(parseInt(e.target.value, 10))}
                className="w-full accent-emerald-400 cursor-pointer"
              />
            </div>
          )}
        </div>

        {/* 5. Environment Background Theme (Dark Void vs Light Studio) */}
        <div className="space-y-1.5 pt-2 border-t border-white/15">
          <div className="flex justify-between items-center text-xs">
            <span className="text-neutral-400 text-[11px] uppercase tracking-wider font-bold flex items-center gap-1.5">
              {environmentTheme === 'light' ? (
                <Sun className="w-3.5 h-3.5 text-amber-400" />
              ) : (
                <Moon className="w-3.5 h-3.5 text-[#38BDF8]" />
              )}
              Environment Mode
            </span>
          </div>
          <div className="grid grid-cols-2 gap-1.5 pt-1">
            <button
              type="button"
              onClick={() => setEnvironmentTheme('dark')}
              className={`py-1.5 px-2 border text-[10px] font-bold flex items-center justify-center gap-1.5 transition-all ${
                environmentTheme === 'dark'
                  ? 'bg-white/20 text-white border-white shadow-sm'
                  : 'bg-black/60 text-neutral-400 border-white/10 hover:border-white/30 hover:text-white'
              }`}
            >
              <Moon className="w-3 h-3 text-[#38BDF8]" />
              DARK VOID
            </button>
            <button
              type="button"
              onClick={() => setEnvironmentTheme('light')}
              className={`py-1.5 px-2 border text-[10px] font-bold flex items-center justify-center gap-1.5 transition-all ${
                environmentTheme === 'light'
                  ? 'bg-amber-400 text-black border-amber-400 font-black shadow-sm'
                  : 'bg-black/60 text-neutral-400 border-white/10 hover:border-white/30 hover:text-white'
              }`}
            >
              <Sun className="w-3 h-3 text-amber-400" />
              LIGHT STUDIO
            </button>
          </div>
        </div>

        {/* Footer info & close */}
        <div className="pt-2 border-t border-white/10 flex items-center justify-between text-[10px] text-neutral-500">
          <span>CHANGES APPLY LIVE</span>
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1 bg-white text-black font-bold hover:bg-neutral-200 transition-colors"
          >
            DONE
          </button>
        </div>
      </div>
    </div>
  );
}
