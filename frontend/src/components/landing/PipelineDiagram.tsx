import React from 'react';
import { ArrowRight, ArrowDown } from 'lucide-react';

interface PipelineStep {
  id: string;
  step: string;
  title: string;
  architecture: string;
  specs: [string, string];
  status: 'ARMED' | 'ONLINE' | 'ACTIVE' | 'VERIFIED';
  statusColor: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: '01',
    step: 'STAGE 01',
    title: 'INPUT IMAGE',
    architecture: 'OPTICAL SATELLITE TILE',
    specs: ['3-Band RGB / 1-Band Pan', 'Sub-meter GSD · Max 100MB'],
    status: 'ARMED',
    statusColor: 'bg-[#34D399]',
  },
  {
    id: '02',
    step: 'STAGE 02',
    title: 'DEPTH INFERENCE',
    architecture: 'DINOv2 ViT-S + LOG1P',
    specs: ['Scale-Invariant Loss (0.1500)', '~45ms Latency · RTX 4060'],
    status: 'ONLINE',
    statusColor: 'bg-[#38BDF8]',
  },
  {
    id: '03',
    step: 'STAGE 03',
    title: 'VOXELIZATION',
    architecture: 'ADAPTIVE DISCRETE POOLING',
    specs: ['Discrete Turbo Elevation Bands', 'Up to 128×128 Columns'],
    status: 'ACTIVE',
    statusColor: 'bg-[#FBBF24]',
  },
  {
    id: '04',
    step: 'STAGE 04',
    title: 'MESH SYNTHESIS',
    architecture: 'RADIAL GPU MATERIALIZATION',
    specs: ['Center-Out Waveform Reveal', '60 FPS Instanced Cubes'],
    status: 'ONLINE',
    statusColor: 'bg-[#38BDF8]',
  },
  {
    id: '05',
    step: 'STAGE 05',
    title: 'DSM OUTPUT',
    architecture: 'CALIBRATED METRIC RASTER',
    specs: ['32-Bit GeoTIFF Export', 'FPV Reconnaissance Flight'],
    status: 'VERIFIED',
    statusColor: 'bg-[#34D399]',
  },
];

export default function PipelineDiagram() {
  return (
    <div className="space-y-6 pt-4">
      {/* Section Sub-Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b border-white/15 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-neutral-500 font-bold tracking-widest uppercase">
              01 // ENGINEERING ARCHITECTURE
            </span>
          </div>
          <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight uppercase mt-0.5">
            Data Synthesis Pipeline
          </h3>
        </div>
        <div className="font-mono text-[10px] text-neutral-400 tracking-wider">
          SYNCHRONOUS PIPELINE BUS · ZERO CPU PER-FRAME OVERHEAD
        </div>
      </div>

      {/* Engineering Process Flow */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3 relative">
        {PIPELINE_STEPS.map((step, idx) => (
          <div key={step.id} className="relative flex flex-col">
            {/* Process Step Card */}
            <div className="im-panel p-4 flex-1 flex flex-col justify-between space-y-3 relative group hover:border-cyan-500/50 hover:bg-white/[0.02] transition-all">
              {/* Corner Reticles */}
              <span className="absolute top-1.5 left-1.5 font-mono text-[9px] text-white/20 select-none">+</span>
              <span className="absolute top-1.5 right-1.5 font-mono text-[9px] text-white/20 select-none">+</span>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-black text-neutral-500 tracking-wider">
                    {step.step}
                  </span>
                  <span className="inline-flex items-center gap-1.5 font-mono text-[9px] text-neutral-400">
                    <span className={`w-1.5 h-1.5 rounded-full ${step.statusColor} animate-pulse`} />
                    {step.status}
                  </span>
                </div>

                <div className="space-y-0.5">
                  <h4 className="text-sm font-black text-white tracking-tight uppercase font-mono">
                    {step.title}
                  </h4>
                  <p className="text-[10px] font-mono text-[#38BDF8] tracking-tight uppercase">
                    {step.architecture}
                  </p>
                </div>
              </div>

              <div className="pt-2 border-t border-white/10 space-y-1 font-mono text-[9px] text-neutral-400">
                <div>• {step.specs[0]}</div>
                <div>• {step.specs[1]}</div>
              </div>
            </div>

            {/* Connecting Chevron on Desktop */}
            {idx < PIPELINE_STEPS.length - 1 && (
              <div className="hidden md:flex absolute -right-2 top-1/2 -translate-y-1/2 z-20 pointer-events-none text-neutral-600">
                <ArrowRight className="w-3.5 h-3.5 text-white/40" />
              </div>
            )}

            {/* Connecting Chevron on Mobile */}
            {idx < PIPELINE_STEPS.length - 1 && (
              <div className="flex md:hidden justify-center py-1 text-neutral-600">
                <ArrowDown className="w-3.5 h-3.5 text-white/40" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
