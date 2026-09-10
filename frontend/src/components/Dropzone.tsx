import React, { useCallback, useState, useRef } from 'react';
import { Upload, ArrowUpRight, Sparkles, Grid3X3, Layers, Compass, FileCheck } from 'lucide-react';
import MatrixModal from './MatrixModal';
import LiveTerrainPreview from './landing/LiveTerrainPreview';
import PipelineDiagram from './landing/PipelineDiagram';

interface DropzoneProps {
  onFileSelected: (file: File, estimateUncertainty?: boolean) => void;
  loading: boolean;
  progress: number;
}

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];

export default function Dropzone({ onFileSelected, loading, progress }: DropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [estimateUncertainty, setEstimateUncertainty] = useState(false);
  const [isMatrixOpen, setIsMatrixOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported format. Accepted: GeoTIFF (.tif, .tiff), PNG, JPG`);
      return false;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum 100 MB allowed.');
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      onFileSelected(file, estimateUncertainty);
    }
  }, [onFileSelected, estimateUncertainty]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && validateFile(file)) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      onFileSelected(file, estimateUncertainty);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const loadSample = async (url: string, filename: string, mime: string) => {
    try {
      setLoadingSample(filename);
      setError(null);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Preset asset not found: ${filename} (HTTP ${res.status})`);
      }
      const blob = await res.blob();
      const file = new File([blob], filename, { type: mime });
      onFileSelected(file, estimateUncertainty);
    } catch (err: any) {
      setError(`Failed to load sample: ${err.message || err}`);
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-12">
      {/* Top Status & Context Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-[10px] font-mono text-neutral-500 border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#10B981] animate-pulse" />
          <span className="text-white font-bold tracking-wider uppercase">
            MISSION STATUS // STANDBY FOR INGESTION
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <span>SENSOR: OPTICAL HIGH-RES</span>
          <span>DATUM: WGS-84 / UTM</span>
          <span className="hidden sm:inline">ISRO SAC SIH26175</span>
        </div>
      </div>

      {/* Hero Viewport: Left Title & Instrument Dropzone, Right Live 3D Workstation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Title, Metadata, and Instrument Dropzone */}
        <div className="lg:col-span-7 space-y-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 border border-cyan-500/30 bg-cyan-500/5 text-[#38BDF8] font-mono text-[10px] tracking-widest uppercase">
              <span className="w-1.5 h-1.5 bg-[#38BDF8] animate-pulse" />
              <span>AUTONOMOUS RECONNAISSANCE SUITE · V2.4</span>
            </div>

            <h1 className="text-5xl sm:text-6xl md:text-7xl font-black tracking-tighter text-white uppercase leading-[0.86]">
              SATELLITE <br />
              <span className="stroke-text">DEPTH</span> <br />
              SYNTHESIS
            </h1>

            <p className="text-xs sm:text-sm text-neutral-400 font-normal leading-relaxed max-w-xl">
              High-precision metric Digital Surface Models (DSM) and interactive 60 FPS 3D terrain meshes synthesized from single-view optical satellite imagery.
            </p>

            <div className="flex flex-wrap items-center gap-2 pt-1 font-mono text-[10px] text-neutral-400">
              <span className="im-tag">CRS: EPSG:32617</span>
              <span className="im-tag">GSD: 0.33M</span>
              <span className="im-tag">DINOv2 ViT-S</span>
              <span className="im-tag">60 FPS WEBGL</span>
            </div>
          </div>

          {/* Instrument Drop Target */}
          <div
            className={`
              relative p-8 sm:p-10 text-center cursor-pointer transition-all border
              ${isDragOver
                ? 'border-cyan-400 bg-cyan-500/[0.06] shadow-[0_0_30px_rgba(6,182,212,0.2)]'
                : 'border-white/20 hover:border-white/50 bg-[#07080B]'
              }
              ${loading ? 'pointer-events-none' : ''}
            `}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={loading ? undefined : handleClick}
          >
            {/* Perimeter Technical Indicators & Corner Brackets */}
            <span className="absolute top-2 left-2 font-mono text-[9px] text-white/40 select-none">
              ┌ [INPUT: OPTICAL SENSOR]
            </span>
            <span className="absolute top-2 right-2 font-mono text-[9px] text-white/40 select-none">
              [EPSG:32617 / WGS-84] ┐
            </span>
            <span className="absolute bottom-2 left-2 font-mono text-[9px] text-white/40 select-none">
              └ [RESOLUTION: SUB-METER GSD]
            </span>
            <span className="absolute bottom-2 right-2 font-mono text-[9px] text-white/40 select-none">
              [BANDS: PAN / RGB · MAX 100MB] ┘
            </span>

            {/* Subtle Crosshairs */}
            <span className="absolute top-6 left-6 font-mono text-[11px] text-white/15 select-none">+</span>
            <span className="absolute top-6 right-6 font-mono text-[11px] text-white/15 select-none">+</span>
            <span className="absolute bottom-6 left-6 font-mono text-[11px] text-white/15 select-none">+</span>
            <span className="absolute bottom-6 right-6 font-mono text-[11px] text-white/15 select-none">+</span>

            <input
              ref={fileInputRef}
              type="file"
              accept={ALLOWED_EXTENSIONS.join(',')}
              onChange={handleInputChange}
              className="hidden"
            />

            {loading ? (
              <div className="space-y-6 py-6">
                <div className="w-14 h-14 mx-auto border border-cyan-500/40 bg-cyan-500/10 flex items-center justify-center">
                  <Sparkles className="w-7 h-7 text-cyan-400 animate-pulse" />
                </div>

                <div className="space-y-1.5">
                  <h3 className="text-xl font-black tracking-tight text-white uppercase font-mono">
                    Synthesizing Metric Terrain
                  </h3>
                  <p className="text-[11px] text-neutral-400 max-w-md mx-auto font-mono">
                    DINOv2 ViT-S Backbone · Scale-Invariant Log1p · Two-Component DTM Decomposition
                  </p>
                </div>

                <div className="w-full max-w-md mx-auto space-y-2 pt-2">
                  <div className="h-1.5 w-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full bg-cyan-400 transition-all duration-300 shadow-[0_0_12px_rgba(56,189,248,0.8)]"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] font-mono text-neutral-400">
                    <span>INFERENCE PROGRESS</span>
                    <span className="text-cyan-400 font-bold">{progress}%</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-5 py-4">
                <div className="w-14 h-14 mx-auto border border-white/20 bg-white/[0.02] flex items-center justify-center transition-transform duration-200 group-hover:scale-105">
                  <Upload className="w-6 h-6 text-white" />
                </div>

                <div className="space-y-1">
                  <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight uppercase">
                    {isDragOver ? 'Release Satellite Image to Process' : 'Drop Satellite Imagery'}
                  </h3>
                  <p className="text-xs text-neutral-400 max-w-md mx-auto">
                    Drag and drop raw satellite GeoTIFF or high-resolution optical imagery, or select a file to ingest.
                  </p>
                </div>

                <div className="pt-1">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleClick();
                    }}
                    className="im-btn-primary"
                  >
                    <span>SELECT FILE</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="flex items-center justify-center gap-2 text-[10px] text-neutral-500 font-mono uppercase tracking-wider">
                  <span>GeoTIFF (.tif)</span>
                  <span>·</span>
                  <span>PNG</span>
                  <span>·</span>
                  <span>JPEG</span>
                  <span>·</span>
                  <span>Max 100 MB</span>
                </div>

                {/* Instrument Settings & Mode Toggles */}
                <div className="pt-2 flex flex-wrap items-center justify-center gap-3">
                  <label
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-2 px-3.5 py-1.5 border border-white/15 bg-white/[0.02] hover:border-white/40 cursor-pointer transition-colors text-[10px] font-mono uppercase text-neutral-300"
                  >
                    <input
                      type="checkbox"
                      checked={estimateUncertainty}
                      onChange={(e) => setEstimateUncertainty(e.target.checked)}
                      className="accent-cyan-400 w-3.5 h-3.5 cursor-pointer"
                    />
                    <span>Estimate Uncertainty (Single-Pass Evidential)</span>
                  </label>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setIsMatrixOpen(true);
                    }}
                    className="inline-flex items-center gap-2 px-3.5 py-1.5 border border-cyan-500/40 bg-cyan-500/10 hover:bg-cyan-500/20 hover:border-cyan-400 text-cyan-200 text-[10px] font-mono font-bold uppercase tracking-wider transition-all cursor-pointer"
                  >
                    <Grid3X3 className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Matrix (Multi-Tile Grid)</span>
                  </button>
                </div>

                {error && (
                  <div className="p-3 border border-rose-500 bg-rose-500/10 text-rose-300 text-[11px] font-mono max-w-md mx-auto flex items-center justify-center gap-2">
                    <span>⚠️</span>
                    <span>{error}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Compact Live 3D Scientific Visualization Workstation Panel */}
        <div className="lg:col-span-5 flex flex-col">
          <LiveTerrainPreview
            onLoadPreset={() =>
              loadSample('/sample_satellite_georef.tif', 'real_satellite_georef.tif', 'image/tiff')
            }
          />
        </div>
      </div>

      {/* Engineering Architecture: 5-Stage Data Synthesis Pipeline */}
      <PipelineDiagram />

      {/* Verification Testbeds: Selectable Scientific Datasets */}
      <div className="space-y-6 pt-4">
        <div className="border-b border-white/15 pb-3 flex flex-col sm:flex-row sm:items-end justify-between gap-2">
          <div>
            <h2 className="text-[10px] font-mono font-bold tracking-widest text-neutral-500 uppercase">
              02 // BENCHMARKS
            </h2>
            <h3 className="text-xl sm:text-2xl font-black text-white tracking-tight uppercase mt-0.5">
              Verification Testbeds
            </h3>
          </div>
          <span className="text-[10px] font-mono text-neutral-400">
            REAL SATELLITE TILES · CALIBRATED ELEVATION GROUND TRUTH
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Dataset 01: Multispectral Satellite GeoTIFF */}
          <div
            onClick={() => loadSample('/sample_satellite_georef.tif', 'real_satellite_georef.tif', 'image/tiff')}
            className="im-panel p-6 sm:p-7 cursor-pointer im-panel-hover flex flex-col justify-between space-y-6 relative overflow-hidden group"
          >
            {/* Reticle Marks */}
            <span className="absolute top-2 left-2 font-mono text-[9px] text-white/30">+</span>
            <span className="absolute top-2 right-2 font-mono text-[9px] text-white/30">+</span>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-5xl font-black font-mono text-white/10 select-none tracking-tighter">
                  01
                </span>
                <div className="flex items-center gap-2">
                  <span className="im-tag im-tag-success">EPSG:32617</span>
                  <span className="im-tag">32-BIT FLOAT</span>
                </div>
              </div>

              <div className="space-y-1">
                <h4 className="text-lg sm:text-xl font-black text-white uppercase tracking-tight font-mono">
                  Multispectral Satellite GeoTIFF
                </h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Real urban relief tile with georeferenced coordinates and metric elevation prior calibration.
                </p>
              </div>

              {/* Scientific Telemetry Spec Table */}
              <div className="grid grid-cols-2 gap-2 p-3 bg-white/[0.02] border border-white/10 font-mono text-[10px]">
                <div>
                  <span className="text-neutral-500">GSD: </span>
                  <span className="text-white font-bold">0.33m / px</span>
                </div>
                <div>
                  <span className="text-neutral-500">DIMENSIONS: </span>
                  <span className="text-white font-bold">1024 × 1024</span>
                </div>
                <div>
                  <span className="text-neutral-500">DATUM: </span>
                  <span className="text-cyan-400 font-bold">UTM ZONE 17N</span>
                </div>
                <div>
                  <span className="text-neutral-500">CALIBRATION: </span>
                  <span className="text-emerald-400 font-bold">METRIC ABSOLUTE</span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-white/15">
              <span className="im-meta">SAMPLE_SATELLITE_GEOREF.TIF</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSample('/sample_satellite_georef.tif', 'real_satellite_georef.tif', 'image/tiff');
                }}
                disabled={loading || !!loadingSample}
                className="im-btn-primary"
              >
                {loadingSample === 'real_satellite_georef.tif' ? (
                  <>
                    <div className="w-3 h-3 border-2 border-current border-t-transparent animate-spin rounded-full" />
                    <span>LOADING...</span>
                  </>
                ) : (
                  <>
                    <span>LOAD DATASET</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Dataset 02: High-Resolution Urban Optical */}
          <div
            onClick={() => loadSample('/sample_satellite_urban.png', 'real_satellite_urban.png', 'image/png')}
            className="im-panel p-6 sm:p-7 cursor-pointer im-panel-hover flex flex-col justify-between space-y-6 relative overflow-hidden group"
          >
            {/* Reticle Marks */}
            <span className="absolute top-2 left-2 font-mono text-[9px] text-white/30">+</span>
            <span className="absolute top-2 right-2 font-mono text-[9px] text-white/30">+</span>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-5xl font-black font-mono text-white/10 select-none tracking-tighter">
                  02
                </span>
                <div className="flex items-center gap-2">
                  <span className="im-tag">OPTICAL RGB</span>
                  <span className="im-tag">STANDARD RGB</span>
                </div>
              </div>

              <div className="space-y-1">
                <h4 className="text-lg sm:text-xl font-black text-white uppercase tracking-tight font-mono">
                  High-Resolution Urban Optical
                </h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Dense architectural district with high building contrast and normalized relative DSM synthesis.
                </p>
              </div>

              {/* Scientific Telemetry Spec Table */}
              <div className="grid grid-cols-2 gap-2 p-3 bg-white/[0.02] border border-white/10 font-mono text-[10px]">
                <div>
                  <span className="text-neutral-500">SENSOR: </span>
                  <span className="text-white font-bold">3-BAND RGB</span>
                </div>
                <div>
                  <span className="text-neutral-500">DIMENSIONS: </span>
                  <span className="text-white font-bold">1024 × 1024</span>
                </div>
                <div>
                  <span className="text-neutral-500">COLOR SPACE: </span>
                  <span className="text-cyan-400 font-bold">sRGB 8-BIT</span>
                </div>
                <div>
                  <span className="text-neutral-500">CALIBRATION: </span>
                  <span className="text-emerald-400 font-bold">RELATIVE DSM</span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-white/15">
              <span className="im-meta">SAMPLE_SATELLITE_URBAN.PNG</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSample('/sample_satellite_urban.png', 'real_satellite_urban.png', 'image/png');
                }}
                disabled={loading || !!loadingSample}
                className="im-btn-primary"
              >
                {loadingSample === 'real_satellite_urban.png' ? (
                  <>
                    <div className="w-3 h-3 border-2 border-current border-t-transparent animate-spin rounded-full" />
                    <span>LOADING...</span>
                  </>
                ) : (
                  <>
                    <span>LOAD DATASET</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Aerospace System Telemetry Footer */}
      <div className="pt-8 pb-4 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4 text-[10px] font-mono text-neutral-500 uppercase tracking-widest">
        <div>
          ISRO SPACE APPLICATIONS CENTRE · SIH26175
        </div>
        <div className="text-center">
          DINOv2 ViT-S + GAMUS DECOMPOSITION // BEST LOSS 0.1500
        </div>
        <div>
          60 FPS WEBGL 3D WORKSTATION ENGINE
        </div>
      </div>

      {/* Matrix Grid Mosaic Modal */}
      <MatrixModal
        isOpen={isMatrixOpen}
        onClose={() => setIsMatrixOpen(false)}
        onSynthesize={(compositeFile) => {
          window.scrollTo({ top: 0, behavior: 'smooth' });
          onFileSelected(compositeFile, estimateUncertainty);
        }}
      />
    </div>
  );
}
