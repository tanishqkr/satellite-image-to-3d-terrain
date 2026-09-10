import React, { useState } from 'react';
import { Eye, Layers, Maximize2 } from 'lucide-react';

interface DualViewerProps {
  rgbB64: string;
  dsmColorizedB64: string;
  elevationMin: number;
  elevationMax: number;
  unit: string;
}

export default function DualViewer({
  rgbB64,
  dsmColorizedB64,
  elevationMin,
  elevationMax,
  unit,
}: DualViewerProps) {
  const [viewMode, setViewMode] = useState<'split' | 'rgb' | 'dsm'>('split');

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2 border-b border-white/15 pb-4">
        <div className="text-[10px] font-mono font-bold tracking-widest text-neutral-500 uppercase">
          04 / 2D ORTHO INSPECTION
        </div>
        <h3 className="text-lg font-black text-white uppercase tracking-tight">
          Spectral & Depth Models
        </h3>
        <p className="text-xs text-neutral-400 leading-relaxed font-normal">
          Inspect truecolor radiometric satellite tiles against high-precision colorized Digital Surface Models.
        </p>
      </div>

      <div className="space-y-6">
        {/* View Mode Segmented Controls - Sharp Architectural Grid */}
        <div className="grid grid-cols-3 border border-white/15 bg-[#050608]">
          {[
            { id: 'split' as const, label: 'SIDE-BY-SIDE' },
            { id: 'rgb' as const, label: 'OPTICAL RGB' },
            { id: 'dsm' as const, label: 'TURBO DSM' },
          ].map(({ id, label }) => {
            const isSelected = viewMode === id;
            return (
              <button
                key={id}
                type="button"
                onClick={() => setViewMode(id)}
                className={`py-2 text-xs font-mono font-bold transition-all border-r border-white/10 last:border-r-0 ${
                  isSelected
                    ? 'bg-white text-black'
                    : 'bg-transparent text-neutral-400 hover:text-white hover:bg-white/[0.03]'
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>

        {/* Viewport Renderings */}
        {viewMode === 'split' ? (
          <div className="grid grid-cols-2 gap-3">
            {/* RGB */}
            <div className="border border-white/15 bg-black overflow-hidden">
              <div className="bg-white/[0.04] text-neutral-300 text-[10px] font-mono font-bold px-3 py-2 border-b border-white/10 uppercase tracking-wider text-center">
                Optical RGB
              </div>
              <div className="w-full aspect-square flex items-center justify-center p-2 bg-black">
                <img
                  src={`data:image/jpeg;base64,${rgbB64}`}
                  alt="Optical RGB"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            {/* DSM */}
            <div className="border border-white/15 bg-black overflow-hidden">
              <div className="bg-white/[0.04] text-[#38BDF8] text-[10px] font-mono font-bold px-3 py-2 border-b border-white/10 uppercase tracking-wider text-center">
                Turbo DSM
              </div>
              <div className="w-full aspect-square flex items-center justify-center p-2 bg-black">
                <img
                  src={`data:image/png;base64,${dsmColorizedB64}`}
                  alt="Turbo DSM"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          </div>
        ) : viewMode === 'rgb' ? (
          <div className="border border-white/15 bg-black overflow-hidden">
            <div className="bg-white/[0.04] text-neutral-300 text-[10px] font-mono font-bold px-3 py-2 border-b border-white/10 uppercase flex justify-between items-center">
              <span>Full-Frame Optical RGB Source</span>
              <span className="text-neutral-500 font-mono text-[10px]">TRUECOLOR 3-BAND</span>
            </div>
            <div className="w-full aspect-square flex items-center justify-center p-2 bg-black">
              <img
                src={`data:image/jpeg;base64,${rgbB64}`}
                alt="Optical RGB Full"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        ) : (
          <div className="border border-white/15 bg-black overflow-hidden">
            <div className="bg-white/[0.04] text-neutral-300 text-[10px] font-mono font-bold px-3 py-2 border-b border-white/10 uppercase flex justify-between items-center">
              <span>Full-Frame Turbo DSM Model</span>
              <span className="text-[#38BDF8] font-mono text-[10px]">CALIBRATED RELIEF</span>
            </div>
            <div className="w-full aspect-square flex items-center justify-center p-2 bg-black">
              <img
                src={`data:image/png;base64,${dsmColorizedB64}`}
                alt="Turbo DSM Full"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        )}

        {/* Elevation Continuous Span Telemetry */}
        <div className="bg-white/[0.02] border border-white/15 p-4 space-y-3">
          <div className="flex justify-between items-center text-xs font-mono">
            <div>
              <span className="text-[10px] text-neutral-500 block uppercase tracking-widest">Base Elev</span>
              <span className="text-white font-bold text-sm mt-0.5">{elevationMin.toFixed(1)} {unit.toUpperCase()}</span>
            </div>
            <span className="text-[10px] uppercase tracking-widest text-neutral-500 font-bold">Relief Scale</span>
            <div className="text-right">
              <span className="text-[10px] text-neutral-500 block uppercase tracking-widest">Peak Elev</span>
              <span className="text-[#38BDF8] font-bold text-sm mt-0.5">{elevationMax.toFixed(1)} {unit.toUpperCase()}</span>
            </div>
          </div>

          {/* Color ramp bar */}
          <div
            className="h-2 w-full border border-white/20 overflow-hidden"
            style={{
              background: 'linear-gradient(to right, #30123b, #4675ed, #1bcfd4, #a4fc3c, #fe9b2d, #f36315, #7a0402)',
            }}
          />
        </div>

        {/* Radiometric Metadata */}
        <div className="p-4 bg-white/[0.02] border border-white/15 text-xs font-mono space-y-2.5">
          <div className="text-[10px] text-neutral-500 uppercase tracking-widest font-bold">
            Radiometric & Sensor Attributes
          </div>
          <div className="flex justify-between text-neutral-400 pt-2 border-t border-white/10">
            <span className="uppercase text-[10px] tracking-wider">Color Map:</span>
            <span className="text-white font-bold text-[11px]">Google Turbo (Perceptually Uniform)</span>
          </div>
          <div className="flex justify-between text-neutral-400 pt-2 border-t border-white/10">
            <span className="uppercase text-[10px] tracking-wider">Resampling:</span>
            <span className="text-white font-bold text-[11px]">Bicubic Spline Interpolation</span>
          </div>
        </div>
      </div>
    </div>
  );
}
