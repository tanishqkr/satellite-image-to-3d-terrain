import React, { useMemo } from 'react';
import { Waves, AlertTriangle } from 'lucide-react';

interface FloodSimulatorProps {
  waterLevel: number;
  onWaterLevelChange: (level: number) => void;
  elevationMin: number;
  elevationMax: number;
  unit: string;
  dsmRaw: number[][];
}

export default function FloodSimulator({
  waterLevel, onWaterLevelChange, elevationMin, elevationMax, unit, dsmRaw
}: FloodSimulatorProps) {
  // Precompute 512-bin cumulative distribution function (CDF) for O(1) slider evaluation
  const { cdf, totalPixels } = useMemo(() => {
    if (!dsmRaw || dsmRaw.length === 0) return { cdf: null, totalPixels: 0 };
    const numBins = 512;
    const counts = new Uint32Array(numBins);
    let total = 0;
    const range = elevationMax - elevationMin;
    const scale = range > 1e-6 ? (numBins - 1) / range : 0;

    for (let i = 0; i < dsmRaw.length; i++) {
      const row = dsmRaw[i];
      for (let j = 0; j < row.length; j++) {
        const val = row[j];
        const bin = Math.max(0, Math.min(numBins - 1, Math.floor((val - elevationMin) * scale)));
        counts[bin]++;
        total++;
      }
    }

    if (total === 0) return { cdf: null, totalPixels: 0 };

    const cdfArray = new Float32Array(numBins);
    let accum = 0;
    for (let b = 0; b < numBins; b++) {
      accum += counts[b];
      cdfArray[b] = (accum / total) * 100;
    }
    return { cdf: cdfArray, totalPixels: total };
  }, [dsmRaw, elevationMin, elevationMax]);

  // O(1) lookup of inundated percentage on slider tick with continuous piecewise linear interpolation
  const floodedPercent = useMemo(() => {
    if (!cdf || totalPixels === 0) return 0;
    if (waterLevel <= elevationMin) return 0;
    if (waterLevel >= elevationMax) return 100;
    const numBins = cdf.length;
    const range = elevationMax - elevationMin;
    if (range < 1e-6) return 0;
    const norm = Math.max(0, Math.min(1, (waterLevel - elevationMin) / range));
    const floatBin = norm * (numBins - 1);
    const i0 = Math.floor(floatBin);
    const frac = floatBin - i0;
    const c0 = i0 > 0 ? cdf[i0 - 1] : 0;
    const c1 = cdf[i0];
    return Math.min(100, Math.max(0, c0 + (c1 - c0) * frac));
  }, [cdf, waterLevel, elevationMin, elevationMax, totalPixels]);

  const riskLevel = floodedPercent > 40 ? 'CRITICAL' : floodedPercent > 15 ? 'MODERATE' : 'NOMINAL';
  const riskColor = floodedPercent > 40 ? 'text-rose-400 border-rose-500/50 bg-rose-500/10' : floodedPercent > 15 ? 'text-amber-400 border-amber-500/50 bg-amber-500/10' : 'text-emerald-400 border-emerald-500/50 bg-emerald-500/10';

  const setDelta = (delta: number) => {
    const next = Math.max(elevationMin, Math.min(elevationMax, waterLevel + delta));
    onWaterLevelChange(next);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2 border-b border-white/15 pb-4">
        <div className="text-[10px] font-mono font-bold tracking-widest text-neutral-500 uppercase">
          02 / HYDROLOGIC SIMULATION
        </div>
        <h3 className="text-lg font-black text-white uppercase tracking-tight">
          Inundation & Surge
        </h3>
        <p className="text-xs text-neutral-400 leading-relaxed font-normal">
          Simulate sea level rise, flash flooding, and dynamic low-lying terrain submergence in real time.
        </p>
      </div>

      <div className="space-y-7">
        {/* Surface Inundation Metric Display - Monumental Typography */}
        <div className="p-5 bg-white/[0.02] border border-white/15 space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <span className="text-[10px] font-mono uppercase tracking-widest text-neutral-500 block">
                Surface Submerged
              </span>
              <div className="text-5xl sm:text-6xl font-black text-white tracking-tighter font-mono mt-1">
                {floodedPercent.toFixed(1)}%
              </div>
            </div>

            <div className="text-right">
              <span className="text-[10px] font-mono uppercase tracking-widest text-neutral-500 block">
                Threat Status
              </span>
              <span className={`inline-block mt-2 px-2.5 py-1 text-[10px] font-mono font-bold border tracking-wider ${riskColor}`}>
                {riskLevel} RISK
              </span>
            </div>
          </div>

          {/* Hairline Linear Progress Bar */}
          <div className="h-1.5 w-full bg-white/10 overflow-hidden">
            <div
              className={`h-full transition-all duration-150 ${
                floodedPercent > 40 ? 'bg-rose-500' : floodedPercent > 15 ? 'bg-amber-400' : 'bg-[#38BDF8]'
              }`}
              style={{ width: `${floodedPercent}%` }}
            />
          </div>

          <div className="flex justify-between items-center text-xs font-mono pt-1 text-neutral-400 border-t border-white/10">
            <span className="text-[10px] uppercase tracking-wider">Waterline Altitude:</span>
            <span className="text-[#38BDF8] font-bold text-sm">{waterLevel.toFixed(1)} {unit.toUpperCase()}</span>
          </div>
        </div>

        {/* Water Level Slider & Step Adjustments */}
        <div className="space-y-3.5">
          <div className="flex justify-between items-end text-xs font-mono">
            <span className="text-neutral-400 uppercase tracking-wider text-[11px]">
              Waterline Level
            </span>
            <span className="text-white font-black text-lg tracking-tight">
              {waterLevel.toFixed(1)} {unit.toUpperCase()}
            </span>
          </div>

          <input
            type="range"
            min={elevationMin}
            max={elevationMax}
            step={(elevationMax - elevationMin) / 100 || 0.1}
            value={waterLevel}
            onChange={(e) => onWaterLevelChange(parseFloat(e.target.value))}
            className="im-slider"
          />

          <div className="flex justify-between text-[10px] font-mono text-neutral-500">
            <span>BASE: {elevationMin.toFixed(1)} {unit.toUpperCase()}</span>
            <span>PEAK: {elevationMax.toFixed(1)} {unit.toUpperCase()}</span>
          </div>

          {/* Step Adjustments - Sharp Monochromatic Buttons */}
          <div className="grid grid-cols-3 gap-2 pt-1">
            <button
              type="button"
              onClick={() => setDelta(-1.0)}
              className="py-2 text-xs font-mono font-bold border border-white/15 bg-transparent text-neutral-300 hover:border-white/40 hover:text-white transition-colors"
            >
              -1.0 {unit === 'meters' ? 'M' : ''}
            </button>
            <button
              type="button"
              onClick={() => onWaterLevelChange(elevationMin)}
              className="py-2 text-xs font-mono font-bold border border-white/15 bg-transparent text-neutral-300 hover:border-white/40 hover:text-white transition-colors"
            >
              BASE LEVEL
            </button>
            <button
              type="button"
              onClick={() => setDelta(1.0)}
              className="py-2 text-xs font-mono font-bold border border-white/15 bg-transparent text-neutral-300 hover:border-white/40 hover:text-white transition-colors"
            >
              +1.0 {unit === 'meters' ? 'M' : ''}
            </button>
          </div>
        </div>

        {/* Simulation Scenarios */}
        <div className="pt-5 border-t border-white/15 space-y-3">
          <div className="text-[10px] font-mono font-bold uppercase tracking-widest text-neutral-500">
            Disaster Scenario Presets
          </div>

          <div className="grid grid-cols-2 gap-2">
            {[
              { name: 'BASELINE SEA', add: 0 },
              { name: '+2M HIGH TIDE', add: 2 },
              { name: '+10M STORM SURGE', add: 10 },
              { name: '+25M FLASH FLOOD', add: 25 },
            ].map(({ name, add }) => {
              const target = Math.min(elevationMax, elevationMin + add);
              const isActive = Math.abs(waterLevel - target) < 0.2;
              return (
                <button
                  key={name}
                  type="button"
                  onClick={() => onWaterLevelChange(target)}
                  className={`p-3 text-left border transition-all ${
                    isActive
                      ? 'bg-white text-black border-white'
                      : 'border-white/15 bg-transparent hover:border-white/40 text-neutral-300 hover:text-white'
                  }`}
                >
                  <div className={`font-mono text-xs font-black tracking-wider ${isActive ? 'text-black' : 'text-white'}`}>
                    {name}
                  </div>
                  <div className={`text-[10px] font-mono mt-1 ${isActive ? 'text-neutral-700' : 'text-neutral-500'}`}>
                    ALT: {target.toFixed(1)} {unit.toUpperCase()}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Severity Alert */}
        {floodedPercent > 30 && (
          <div className="p-4 bg-amber-500/10 border border-amber-500/40 text-amber-300 text-xs font-mono flex items-start gap-3">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block tracking-wider uppercase">INUNDATION WARNING</span>
              <p className="text-[11px] text-amber-200/90 leading-relaxed font-sans">
                Over 30% of the surveyed terrain is submerged. Ground transit and municipal infrastructure are critically impacted.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
