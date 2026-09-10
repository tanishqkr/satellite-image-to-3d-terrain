import React from 'react';
import { Mountain, Boxes, Compass, Layers } from 'lucide-react';
import ContourOverlay from './ContourOverlay';
import { generateTurboPalette } from '../lib/voxelize.ts';

interface MeshStats {
  elevation_min: number;
  elevation_max: number;
  original_width: number;
  original_height: number;
  pixel_size?: number;
}

interface Calibration {
  min: number;
  max: number;
  unit: string;
}

interface SettingsPanelProps {
  verticalScale: number;
  onVerticalScaleChange: (val: number) => void;
  showContours: boolean;
  onContoursToggle: () => void;
  contourInterval: number;
  onContourIntervalChange: (val: number) => void;
  renderMode: 'voxel' | 'smooth';
  onRenderModeChange: (mode: 'voxel' | 'smooth') => void;
  voxelBands: number;
  onVoxelBandsChange: (val: number) => void;
  voxelResolution: number;
  onVoxelResolutionChange: (val: number) => void;
  meshStats?: MeshStats;
  calibration?: Calibration | null;
  isGeoref?: boolean;
  crs?: string;
  droneSpawnPreset?: 'center' | 'north' | 'south' | 'west' | 'east';
  onDroneSpawnPresetChange?: (preset: 'center' | 'north' | 'south' | 'west' | 'east') => void;
}

export default function SettingsPanel({
  verticalScale,
  onVerticalScaleChange,
  showContours,
  onContoursToggle,
  contourInterval,
  onContourIntervalChange,
  renderMode,
  onRenderModeChange,
  voxelBands,
  onVoxelBandsChange,
  voxelResolution,
  onVoxelResolutionChange,
  meshStats,
  calibration,
  isGeoref = false,
  crs,
  droneSpawnPreset = 'center',
  onDroneSpawnPresetChange,
}: SettingsPanelProps) {
  const minElev = calibration ? calibration.min : (meshStats?.elevation_min ?? 0);
  const maxElev = calibration ? calibration.max : (meshStats?.elevation_max ?? 1);
  const spanElev = Math.max(0, maxElev - minElev);
  const unit = calibration ? calibration.unit : 'relative';
  const gsd = meshStats?.pixel_size ? `${meshStats.pixel_size.toFixed(2)}m` : 'SYNTHETIC';

  const palette = generateTurboPalette(voxelBands);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2 border-b border-white/15 pb-4">
        <div className="text-[10px] font-mono font-bold tracking-widest text-neutral-500 uppercase">
          01 / SURFACE CONTROLS
        </div>
        <h3 className="text-lg font-black text-white uppercase tracking-tight">
          Relief & Geometry
        </h3>
        <p className="text-xs text-neutral-400 leading-relaxed font-normal">
          Toggle between quantized voxel extrusion and continuous photoreal displacement, configure elevation bands, and inspect geometry.
        </p>
      </div>

      <div className="space-y-7">
        {/* Render Mode Toggle (§3.7, §4) */}
        <div className="space-y-2.5">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-neutral-400 uppercase tracking-wider text-[11px]">
              3D Render Mode
            </span>
            <span className="text-[10px] font-mono text-[#38BDF8] uppercase tracking-wider font-bold">
              {renderMode === 'voxel' ? 'BLOCK EXTRUSION' : 'PHOTOREAL DISPLACEMENT'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => onRenderModeChange('voxel')}
              className={`py-2.5 px-3 flex items-center justify-center gap-2 text-xs font-mono font-bold border transition-all ${
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
              onClick={() => onRenderModeChange('smooth')}
              className={`py-2.5 px-3 flex items-center justify-center gap-2 text-xs font-mono font-bold border transition-all ${
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

        {/* Voxel-Specific Configuration (§3.2, §3.8) */}
        {renderMode === 'voxel' && (
          <div className="p-4 bg-white/[0.02] border border-white/15 space-y-5">
            <div className="text-[10px] font-mono font-bold uppercase tracking-widest text-[#38BDF8] flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 bg-[#38BDF8]" />
              Voxel Elevation Tuning
            </div>

            {/* Discrete Band Count Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-neutral-400 uppercase tracking-wider text-[11px]">
                  Elevation Bands
                </span>
                <span className="text-white font-black text-sm">
                  {voxelBands} BANDS
                </span>
              </div>

              <input
                type="range"
                min={5}
                max={12}
                step={1}
                value={voxelBands}
                onChange={(e) => onVoxelBandsChange(parseInt(e.target.value, 10))}
                className="im-slider"
              />

              <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
                <span>5 (DISCRETE)</span>
                <span>8 (BALANCED)</span>
                <span>12 (DETAILED)</span>
              </div>

              {/* Realtime Turbo Palette Swatches */}
              <div className="pt-1.5">
                <div className="flex h-2.5 w-full gap-0.5 overflow-hidden border border-white/10">
                  {palette.map((hex, i) => (
                    <div
                      key={i}
                      className="flex-1 h-full transition-colors"
                      style={{ backgroundColor: hex }}
                      title={`Band ${i + 1}: ${hex}`}
                    />
                  ))}
                </div>
                <div className="flex justify-between text-[9px] text-neutral-500 font-mono mt-1">
                  <span>LOWEST</span>
                  <span>TURBO SPECTRUM</span>
                  <span>HIGHEST</span>
                </div>
              </div>
            </div>

            {/* Block Resolution Slider (§3.8) */}
            <div className="space-y-2 pt-2 border-t border-white/10">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-neutral-400 uppercase tracking-wider text-[11px]">
                  Block Resolution
                </span>
                <span className="text-white font-black text-sm">
                  {voxelResolution}×{voxelResolution}
                </span>
              </div>

              <input
                type="range"
                min={24}
                max={128}
                step={4}
                value={voxelResolution}
                onChange={(e) => onVoxelResolutionChange(parseInt(e.target.value, 10))}
                className="im-slider"
              />

              <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
                <span>24 (COARSE)</span>
                <span>64 (STANDARD)</span>
                <span>128 (ULTRA)</span>
              </div>
            </div>
          </div>
        )}

        {/* Vertical Exaggeration (Common to both modes) */}
        <div className="space-y-3.5">
          <div className="flex justify-between items-end text-xs font-mono">
            <span className="text-neutral-400 uppercase tracking-wider text-[11px]">
              Vertical Scale
            </span>
            <span className="text-white font-black text-lg tracking-tight">
              {verticalScale.toFixed(1)}×
            </span>
          </div>

          <input
            type="range"
            min={0.1}
            max={5}
            step={0.1}
            value={verticalScale}
            onChange={(e) => onVerticalScaleChange(parseFloat(e.target.value))}
            className="im-slider"
          />

          <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
            <span>0.1× SUBTLE</span>
            <span>5.0× STEEP</span>
          </div>

          {/* Scale Presets */}
          <div className="grid grid-cols-4 gap-2 pt-1">
            {[
              { label: '0.5×', val: 0.5 },
              { label: '1.0×', val: 1.0 },
              { label: '1.8×', val: 1.8 },
              { label: '3.0×', val: 3.0 },
            ].map(({ label, val }) => {
              const isSelected = Math.abs(verticalScale - val) < 0.05;
              return (
                <button
                  key={label}
                  type="button"
                  onClick={() => onVerticalScaleChange(val)}
                  className={`py-2 text-xs font-mono font-bold border transition-all ${
                    isSelected
                      ? 'bg-white text-black border-white'
                      : 'bg-transparent text-neutral-400 border-white/15 hover:border-white/40 hover:text-white'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Contour Lines Section */}
        <div className="pt-5 border-t border-white/15">
          {renderMode === 'smooth' ? (
            <ContourOverlay
              active={showContours}
              onToggle={onContoursToggle}
              interval={contourInterval}
              onIntervalChange={onContourIntervalChange}
              isGeoref={isGeoref}
            />
          ) : (
            <div className="p-3 bg-white/[0.02] border border-white/10 space-y-1.5 text-xs font-mono">
              <div className="flex items-center gap-2 text-neutral-300 font-bold text-[11px]">
                <Layers className="w-3.5 h-3.5 text-[#38BDF8]" />
                NATURAL VOXEL CONTOURS
              </div>
              <p className="text-[10px] text-neutral-400 leading-relaxed">
                In voxel mode, contour boundaries are organically formed by the color-band steps between adjacent blocks.
              </p>
            </div>
          )}
        </div>

        {/* FPV Drone Launch Ingress Location (§3 & §6) */}
        <div className="pt-5 border-t border-white/15 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-neutral-500">
              FPV Drone Launch Point (§6)
            </span>
            <span className="text-[10px] font-mono font-bold text-emerald-400 uppercase">
              {droneSpawnPreset.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-5 gap-1.5 text-xs font-mono">
            {[
              { id: 'center' as const, label: 'CENTER' },
              { id: 'north' as const, label: 'NORTH' },
              { id: 'south' as const, label: 'SOUTH' },
              { id: 'west' as const, label: 'WEST' },
              { id: 'east' as const, label: 'EAST' },
            ].map(({ id, label }) => {
              const isSelected = droneSpawnPreset === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => onDroneSpawnPresetChange?.(id)}
                  className={`py-1.5 px-1 text-center font-bold text-[10px] border transition-all ${
                    isSelected
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-sm'
                      : 'bg-white/[0.02] text-neutral-400 border-white/10 hover:border-white/30 hover:text-white'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>
          <p className="text-[10px] text-neutral-400 font-mono leading-relaxed">
            Sets initial reconnaissance spawn coordinates. Drone automatically clears terrain by +10m on spawn.
          </p>
        </div>

        {/* Elevation & Mesh Telemetry Card */}
        <div className="pt-5 border-t border-white/15 space-y-3">
          <div className="text-[10px] font-mono font-bold uppercase tracking-widest text-neutral-500">
            Elevation & Spatial Telemetry
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Base Elevation</span>
              <span className="text-white font-bold text-sm mt-1 block">{minElev.toFixed(1)} {unit}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Peak Elevation</span>
              <span className="text-[#38BDF8] font-bold text-sm mt-1 block">{maxElev.toFixed(1)} {unit}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Relief Span</span>
              <span className="text-white font-bold text-sm mt-1 block">{spanElev.toFixed(1)} {unit}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Pixel GSD</span>
              <span className="text-[#34D399] font-bold text-sm mt-1 block">{gsd}</span>
            </div>
          </div>
        </div>

        {/* Coordinate Reference Box */}
        <div className="p-3.5 bg-white/[0.02] border border-white/10 text-xs font-mono space-y-2">
          <div className="flex items-center justify-between gap-3">
            <span className="text-neutral-400 shrink-0 uppercase tracking-wider text-[10px]">Projection</span>
            <span className="text-white font-bold text-right truncate text-[11px]">{isGeoref ? `${crs || 'EPSG:32617'} (WGS84 UTM)` : 'RELATIVE SYNTHETIC GRID'}</span>
          </div>
          <div className="flex items-center justify-between gap-3 pt-2 border-t border-white/10">
            <span className="text-neutral-400 shrink-0 uppercase tracking-wider text-[10px]">Datum</span>
            <span className="text-white font-bold text-right truncate text-[11px]">{isGeoref ? 'ELLIPSOIDAL HEIGHT (M)' : 'NORMALIZED RANGE [0, 1]'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
