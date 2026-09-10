import React from 'react';
import { Map } from 'lucide-react';

interface ContourOverlayProps {
  active: boolean;
  onToggle: () => void;
  interval: number;
  onIntervalChange: (val: number) => void;
  isGeoref?: boolean;
}

export default function ContourOverlay({
  active,
  onToggle,
  interval,
  onIntervalChange,
  isGeoref = false,
}: ContourOverlayProps) {
  const minInterval = isGeoref ? 0.5 : 0.02;
  const maxInterval = isGeoref ? 20.0 : 0.5;
  const stepInterval = isGeoref ? 0.5 : 0.02;
  const unitLabel = isGeoref ? 'm' : 'rel';

  return (
    <div className="space-y-3">
      <button
        onClick={onToggle}
        className={`w-full text-xs py-3 px-3.5 flex items-center justify-between border transition-all cursor-pointer ${
          active
            ? 'bg-white text-black border-white font-bold'
            : 'bg-white/[0.02] text-neutral-300 border-white/15 hover:border-white/40 hover:text-white'
        }`}
      >
        <span className="flex items-center gap-2 font-mono uppercase text-xs tracking-wider">
          <Map className="w-3.5 h-3.5" /> TOPOGRAPHIC CONTOURS
        </span>
        <span
          className={`font-mono text-[10px] font-bold px-2 py-0.5 border ${
            active ? 'bg-black text-white border-black' : 'bg-transparent text-neutral-500 border-white/15'
          }`}
        >
          {active ? 'ACTIVE' : 'OFF'}
        </span>
      </button>

      {active && (
        <div className="bg-white/[0.02] border border-white/15 p-4 space-y-3.5">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-neutral-400 uppercase tracking-wider text-[10px]">Contour Interval</span>
            <span className="text-white font-black text-sm">
              {interval.toFixed(isGeoref ? 1 : 2)} {unitLabel.toUpperCase()}
            </span>
          </div>

          <input
            type="range"
            min={minInterval}
            max={maxInterval}
            step={stepInterval}
            value={interval}
            onChange={(e) => onIntervalChange(parseFloat(e.target.value))}
            className="im-slider"
          />

          <div className="flex justify-between text-[10px] font-mono text-neutral-500">
            <span>{minInterval.toFixed(isGeoref ? 1 : 2)} {unitLabel.toUpperCase()}</span>
            <span>{maxInterval.toFixed(isGeoref ? 1 : 2)} {unitLabel.toUpperCase()}</span>
          </div>

          {/* Quick Step Presets */}
          {isGeoref && (
            <div className="grid grid-cols-4 gap-2 pt-1">
              {[2, 5, 10, 20].map((step) => {
                const isSelected = Math.abs(interval - step) < 0.1;
                return (
                  <button
                    key={step}
                    type="button"
                    onClick={() => onIntervalChange(step)}
                    className={`py-1.5 text-xs font-mono font-bold border transition-colors ${
                      isSelected
                        ? 'bg-white text-black border-white'
                        : 'bg-transparent text-neutral-400 border-white/15 hover:border-white/40 hover:text-white'
                    }`}
                  >
                    {step}M
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
