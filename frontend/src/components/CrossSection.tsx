import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Scissors, X } from 'lucide-react';

interface CrossSectionProps {
  dsmRaw: number[][];
  unit: string;
  active: boolean;
  onToggle: () => void;
  pixelSize?: number;
}

interface ProfilePoint {
  distance: number;
  elevation: number;
}

export default function CrossSection({ dsmRaw, unit, active, onToggle, pixelSize }: CrossSectionProps) {
  // Horizontal cross-section through center of the elevation map
  const isMetric = unit === 'meters' && pixelSize !== undefined && pixelSize > 0;
  const distScale = isMetric ? pixelSize! : 1.0;
  const distUnit = isMetric ? 'm' : 'px';

  const profileData = useMemo<ProfilePoint[]>(() => {
    if (!dsmRaw || dsmRaw.length === 0) return [];

    const h = dsmRaw.length;
    const w = dsmRaw[0]?.length || 0;
    if (w === 0) return [];

    const midRow = Math.floor(h / 2);
    const K = Math.min(w, 256);
    const points: ProfilePoint[] = [];

    for (let k = 0; k < K; k++) {
      const col = Math.floor((k / (K - 1)) * (w - 1));
      const elevation = dsmRaw[midRow]?.[col] ?? 0;
      const distance = (k / (K - 1)) * (w * distScale);
      points.push({ distance: Math.round(distance * 10) / 10, elevation: Math.round(elevation * 100) / 100 });
    }

    return points;
  }, [dsmRaw, distScale]);

  const { minElev, maxElev, maxDist } = useMemo(() => {
    if (profileData.length === 0) return { minElev: 0, maxElev: 0, maxDist: 0 };
    let min = Infinity;
    let max = -Infinity;
    for (const p of profileData) {
      if (p.elevation < min) min = p.elevation;
      if (p.elevation > max) max = p.elevation;
    }
    const dist = profileData[profileData.length - 1]?.distance || 0;
    return { minElev: min, maxElev: max, maxDist: dist };
  }, [profileData]);

  if (profileData.length === 0) {
    return (
      <div className="p-6 text-center text-xs text-neutral-500 font-mono">
        No DSM elevation data available for transect.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2 border-b border-white/15 pb-4">
        <div className="text-[10px] font-mono font-bold tracking-widest text-neutral-500 uppercase">
          03 / ELEVATION TRANSECT
        </div>
        <h3 className="text-lg font-black text-white uppercase tracking-tight">
          Topographical Profile
        </h3>
        <p className="text-xs text-neutral-400 leading-relaxed font-normal">
          High-resolution centerline cross-section sampled west to east across the synthesized DSM surface.
        </p>
      </div>

      <div className="space-y-6">
        {/* Metric Summary Bar - Sharp 3-column table */}
        <div className="grid grid-cols-3 border border-white/15 bg-white/[0.02] text-center">
          <div className="p-3.5">
            <div className="text-[10px] font-mono text-neutral-500 uppercase tracking-widest">Base Elev</div>
            <div className="text-base font-black font-mono text-white mt-1">
              {minElev.toFixed(1)} {unit === 'meters' ? 'M' : ''}
            </div>
          </div>
          <div className="p-3.5 border-x border-white/15">
            <div className="text-[10px] font-mono text-neutral-500 uppercase tracking-widest">Peak Elev</div>
            <div className="text-base font-black font-mono text-[#38BDF8] mt-1">
              {maxElev.toFixed(1)} {unit === 'meters' ? 'M' : ''}
            </div>
          </div>
          <div className="p-3.5">
            <div className="text-[10px] font-mono text-neutral-500 uppercase tracking-widest">Span Length</div>
            <div className="text-base font-black font-mono text-white mt-1">
              {maxDist.toFixed(1)} {distUnit.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Generous Chart Viewport */}
        <div className="space-y-2.5">
          <div className="flex justify-between items-center text-[10px] font-mono text-neutral-400 px-0.5">
            <span className="uppercase tracking-widest">WEST-TO-EAST CENTERLINE</span>
            <span className="im-tag">{profileData.length} PROBES</span>
          </div>

          <div className="h-72 w-full bg-[#050608] border border-white/15 p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={profileData} margin={{ top: 15, right: 15, left: -10, bottom: 8 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="distance"
                  tick={{ fontSize: 9, fill: '#71717A', fontFamily: 'JetBrains Mono, monospace' }}
                  unit={distUnit.toUpperCase()}
                  stroke="rgba(255,255,255,0.15)"
                  interval={45}
                  tickFormatter={(val) => Math.round(val).toString()}
                />
                <YAxis
                  dataKey="elevation"
                  tick={{ fontSize: 9, fill: '#71717A', fontFamily: 'JetBrains Mono, monospace' }}
                  unit={unit === 'meters' ? 'm' : ''}
                  stroke="rgba(255,255,255,0.15)"
                  domain={['auto', 'auto']}
                  width={35}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#050608',
                    border: '1px solid rgba(255,255,255,0.2)',
                    borderRadius: '0px',
                    fontSize: '11px',
                    fontFamily: 'JetBrains Mono, monospace',
                    color: '#FFFFFF',
                    padding: '8px 12px',
                  }}
                  formatter={(val: any) => [`${val} ${unit}`, 'Elevation']}
                  labelFormatter={(dist: any) => `Distance: ${dist} ${distUnit}`}
                />
                <Line
                  type="monotone"
                  dataKey="elevation"
                  stroke="#38BDF8"
                  strokeWidth={1.75}
                  dot={false}
                  activeDot={{ r: 4, fill: '#FFFFFF', stroke: '#38BDF8', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Slope & Morphological Assessment */}
        <div className="p-4 bg-white/[0.02] border border-white/15 text-xs font-mono space-y-3">
          <div className="text-neutral-400 font-bold uppercase tracking-widest text-[10px] flex items-center justify-between">
            <span>Geomorphological Gradient</span>
            <span className="text-neutral-500 font-normal">Center Slice</span>
          </div>
          <div className="grid grid-cols-2 gap-2 pt-1 text-neutral-400">
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-500 block uppercase tracking-wider">Slope Gradient</span>
              <span className="text-white font-bold text-base block mt-0.5">
                {maxDist > 0 ? (((maxElev - minElev) / maxDist) * 100).toFixed(1) : 0}%
              </span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-500 block uppercase tracking-wider">Sampling Resolution</span>
              <span className="text-[#38BDF8] font-bold text-base block mt-0.5">
                {maxDist > 0 && profileData.length > 1 ? (maxDist / (profileData.length - 1)).toFixed(2) : 1.0} {distUnit.toUpperCase()}
              </span>
            </div>
          </div>
          <div className="pt-1 text-[11px] text-neutral-400 font-sans leading-relaxed">
            Continuous elevation slice bisecting the scene center to reveal structural verticality and building roofline crests.
          </div>
        </div>
      </div>
    </div>
  );
}
