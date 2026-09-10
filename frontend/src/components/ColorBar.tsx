import React from 'react';

interface ColorBarProps {
  min: number;
  max: number;
  unit: string;
}

export default function ColorBar({ min, max, unit }: ColorBarProps) {
  return (
    <div className="p-3 flex flex-col items-center gap-2 backdrop-blur-md bg-black/90 border border-white/15">
      <span className="text-[10px] font-mono font-black tracking-widest text-neutral-400 uppercase">
        TURBO DSM
      </span>
      <div className="flex items-center gap-2.5 my-0.5">
        <div className="flex flex-col justify-between h-28 text-[10px] font-mono text-right py-0.5 select-none">
          <span className="text-white font-bold">{max.toFixed(1)}</span>
          <span className="text-neutral-400 font-medium">{((max + min) / 2).toFixed(1)}</span>
          <span className="text-neutral-500 font-medium">{min.toFixed(1)}</span>
        </div>
        <div
          className="w-2 h-28 border border-white/20 overflow-hidden"
          style={{
            background: 'linear-gradient(to bottom, #7a0402, #f36315, #fe9b2d, #a4fc3c, #1bcfd4, #4675ed, #30123b)',
          }}
        />
      </div>
      <span className="text-[9px] font-mono text-neutral-400 uppercase tracking-widest px-2 py-0.5 border border-white/15">
        {unit.toUpperCase()}
      </span>
    </div>
  );
}
