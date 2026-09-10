import React, { useState } from 'react';
import { useInference } from './hooks/useInference';
import Dropzone from './components/Dropzone';
import DualViewer from './components/DualViewer';
import TerrainCanvas from './components/TerrainCanvas';
import VoxelTerrain from './components/VoxelTerrain';
import Minimap from './components/Minimap';
import { useFlightMode } from './hooks/useFlightMode';
import DroneCanvas from './components/drone/DroneCanvas';
import { Radio } from 'lucide-react';
import FloodSimulator from './components/FloodSimulator';
import CrossSection from './components/CrossSection';
import ViewshedAnalyzer from './components/ViewshedAnalyzer';
import BdaAnalyzer from './components/BdaAnalyzer';
import SettingsPanel from './components/SettingsPanel';
import TerrainQuickPanel from './components/TerrainQuickPanel';
import ColorBar from './components/ColorBar';
import { useTerrainSettings } from './hooks/useTerrainSettings';
import {
  RotateCcw,
  Clock,
  Ruler,
  Globe,
  Sparkles,
  Mountain,
  Waves,
  Scissors,
  Eye,
  Bomb,
  Download,
  Check,
  Sliders,
  Sun,
  Moon,
} from 'lucide-react';

export default function App() {
  const { upload, data, loading, error, progress, reset } = useInference();
  const { isFpv, enterFpv, toggleFlightMode, exitFpv } = useFlightMode('studio');

  // Unified Terrain Settings (§5)
  const terrainSettings = useTerrainSettings({
    initialVerticalScale: 0.5,
    initialWaterLevel: 0,
    initialShowContours: false,
    initialContourInterval: 5,
    initialRenderMode: 'voxel',
    initialVoxelBands: 8,
    initialVoxelResolution: 64,
    initialEnvironmentTheme: 'dark',
  });

  const {
    verticalScale,
    setVerticalScale,
    waterLevel,
    setWaterLevel,
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
    toggleEnvironmentTheme,
  } = terrainSettings;

  const [isQuickPanelOpen, setIsQuickPanelOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'surface' | 'flood' | 'profile' | 'viewshed' | 'bda' | 'inspect'>('surface');
  const [viewshedOverlayUrl, setViewshedOverlayUrl] = useState<string | null>(null);
  const [bdaOverlayUrl, setBdaOverlayUrl] = useState<string | null>(null);
  const [observerPos, setObserverPos] = useState<[number, number, number] | null>(null);
  const [targetPos, setTargetPos] = useState<[number, number, number] | null>(null);
  const [sightlineClear, setSightlineClear] = useState<boolean | null>(null);
  const [strikePos, setStrikePos] = useState<[number, number, number] | null>(null);
  const [strikeSimulation, setStrikeSimulation] = useState<{
    active: boolean;
    col: number;
    row: number;
    radiusM: number;
    depthM: number;
    ejectaM: number;
    modifiedDsm: number[][] | null;
    timestamp: number;
  } | null>(null);
  const [externalClickPoint, setExternalClickPoint] = useState<{ col: number; row: number; isShift: boolean } | null>(null);
  const [noeRoute, setNoeRoute] = useState<any>(null);

  // Dynamic DSM accounting for lively strike crater excavations
  const currentDsm = (strikeSimulation?.active && strikeSimulation?.modifiedDsm)
    ? strikeSimulation.modifiedDsm
    : data?.dsm_raw;
  const [droneSpawnPreset, setDroneSpawnPreset] = useState<'center' | 'north' | 'south' | 'west' | 'east'>('center');
  const [exporting, setExporting] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  // Compute 3D spawn coordinates based on selected reconnaissance ingress preset (§3 & §6)
  const droneSpawnPoint = React.useMemo<[number, number, number] | undefined>(() => {
    if (!data?.mesh_stats) return undefined;
    const w = data.mesh_stats.width * 0.1;
    const d = data.mesh_stats.height * 0.1;
    switch (droneSpawnPreset) {
      case 'north':
        return [0, 0, -d * 0.38];
      case 'south':
        return [0, 0, d * 0.38];
      case 'west':
        return [-w * 0.38, 0, 0];
      case 'east':
        return [w * 0.38, 0, 0];
      case 'center':
      default:
        return [0, 0, 0];
    }
  }, [data?.mesh_stats, droneSpawnPreset]);

  // Synchronize water level and contour interval with newly loaded terrain elevation
  React.useEffect(() => {
    if (data?.calibration) {
      setWaterLevel(data.calibration.min);
      setContourInterval(data.is_georef ? 5.0 : 0.1);
    }
  }, [data]);

  // Global shortcut [Key L] for Environment Theme Toggle (Light / Dark)
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.key.toLowerCase() !== 'l'
      ) {
        return;
      }
      toggleEnvironmentTheme();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toggleEnvironmentTheme]);

  const handleFileSelected = (file: File, estimateUncertainty: boolean = false) => {
    upload(file, estimateUncertainty);
  };

  const handleExport = async () => {
    if (!data?.request_id) return;
    try {
      setExporting(true);
      const response = await fetch(`/api/export/${data.request_id}`);
      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errJson.detail || `Export failed with status ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `depthwizard_dsm_${data.request_id}.tif`;
      a.click();
      URL.revokeObjectURL(url);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 3000);
    } catch (err: any) {
      console.error('Export error:', err);
      alert(err.message || 'Export failed. Check server status.');
    } finally {
      setExporting(false);
    }
  };

  // No data yet — show Dropzone
  if (!data) {
    return (
      <div className="min-h-screen flex flex-col im-container tech-grid-bg">
        {/* Top Header */}
        <header className="border-b border-white/15 px-6 lg:px-10 py-4 flex items-center justify-between bg-[#050608]/90 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-3.5">
            <div className="w-8 h-8 border border-white/25 bg-white/5 flex items-center justify-center text-white font-mono font-bold text-xs tracking-wider">
              DW
            </div>
            <div>
              <h1 className="text-xs font-black text-white tracking-widest uppercase">
                DEPTHWIZARD
              </h1>
              <p className="text-[10px] text-neutral-400 font-mono">
                SIH26175 · ISRO SPACE APPLICATIONS CENTRE
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            
            <span className="im-tag hidden sm:inline-flex">
              GAMUS STAGE 2 // LOSS 0.1500
            </span>
          </div>
        </header>

        {/* Dropzone Hero */}
        <main className="flex-1 w-full">
          <Dropzone onFileSelected={handleFileSelected} loading={loading} progress={progress} />
        </main>

        {error && (
          <div className="max-w-6xl mx-auto w-full px-6 mb-8">
            <div className="p-4 border border-rose-500/50 bg-rose-500/10 text-rose-300 font-mono text-xs flex items-center gap-2">
              <span>⚠️ ERROR:</span>
              <span>{error}</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Data loaded — show interactive 3D studio
  const cal = data.calibration;
  const stats = data.mesh_stats;

  return (
    <div className="h-screen flex flex-col im-container overflow-hidden">
      {/* Top Telemetry Header */}
      <header className="border-b border-white/15 px-6 py-3 flex items-center justify-between z-20 bg-[#050608]/95 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 border border-white/25 bg-white/5 flex items-center justify-center text-white font-mono font-bold text-xs tracking-wider">
              DW
            </div>
            <div>
              <h1 className="text-xs font-black text-white tracking-widest uppercase">
                DEPTHWIZARD
              </h1>
              <p className="text-[10px] text-neutral-400 font-mono">
                ISRO SAC · SIH26175
              </p>
            </div>
          </div>

          <div className="h-5 w-px bg-white/15 hidden md:block" />

          {/* Metric Telemetry Badges */}
          <div className="hidden sm:flex items-center gap-2">
            <span className="im-tag">
              <Clock className="w-3 h-3 text-[#38BDF8]" />
              {data.inference_time_ms.toFixed(0)}MS
            </span>
            <span className="im-tag">
              <Ruler className="w-3 h-3 text-[#34D399]" />
              {stats.original_width}×{stats.original_height}
            </span>
            <span className={`im-tag ${data.is_georef ? 'im-tag-success' : ''}`}>
              <Globe className="w-3 h-3 text-neutral-400" />
              {data.is_georef ? `${data.crs || 'EPSG:32617'} (METRIC)` : 'RELATIVE DSM'}
            </span>
            {data.confidence_mean != null && !isNaN(data.confidence_mean) && (
              <span className="im-tag">
                <Sparkles className="w-3 h-3 text-[#38BDF8]" />
                CONF {(data.confidence_mean * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExport}
            disabled={exporting}
            className="im-btn-primary"
            title="Export 32-Bit GeoTIFF raster"
          >
            {exporting ? (
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 border-2 border-current border-t-transparent animate-spin rounded-full" />
                EXPORTING...
              </span>
            ) : downloaded ? (
              <span className="flex items-center gap-2">
                <Check className="w-3.5 h-3.5" />
                DOWNLOADED GEOTIFF
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <Download className="w-3.5 h-3.5" />
                EXPORT GEOTIFF
              </span>
            )}
          </button>

          {/* Environment Light/Dark Mode Toggle */}
          <button
            type="button"
            onClick={toggleEnvironmentTheme}
            className={`py-2 px-3 text-xs font-mono font-bold border transition-all flex items-center gap-1.5 ${
              environmentTheme === 'light'
                ? 'bg-amber-400 text-black border-amber-400 shadow-lg shadow-amber-400/20'
                : 'bg-transparent text-neutral-300 border-white/20 hover:border-white/50 hover:text-white hover:bg-white/5'
            }`}
            title="Toggle Environment Background: Dark Void vs Light Studio [Key L]"
          >
            {environmentTheme === 'light' ? (
              <Sun className="w-3.5 h-3.5 text-black" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-[#38BDF8]" />
            )}
            {environmentTheme === 'light' ? 'LIGHT MODE' : 'DARK MODE'}
          </button>

          {/* Terrain & Voxel Detail Quick Panel Trigger (§5) */}
          <button
            type="button"
            onClick={() => setIsQuickPanelOpen((prev) => !prev)}
            className={`py-2 px-3 text-xs font-mono font-bold border transition-all flex items-center gap-1.5 ${
              isQuickPanelOpen
                ? 'bg-[#38BDF8] text-black border-[#38BDF8] shadow-lg shadow-[#38BDF8]/20'
                : 'bg-transparent text-neutral-300 border-white/20 hover:border-white/50 hover:text-white hover:bg-white/5'
            }`}
            title="Toggle Terrain & Voxel Detail Quick Panel [Key P]"
          >
            <Sliders className="w-3.5 h-3.5 text-[#38BDF8]" />
            TERRAIN [P]
          </button>

          <button
            type="button"
            onClick={() => {
              if (document.pointerLockElement) {
                document.exitPointerLock();
              }
              toggleFlightMode();
            }}
            className={`py-2 px-3 text-xs font-mono font-bold border transition-all flex items-center gap-1.5 ${
              isFpv
                ? 'bg-[#10B981] text-black border-[#10B981] shadow-lg'
                : 'bg-transparent text-white border-white/20 hover:border-white/50 hover:bg-white/5'
            }`}
            title="Toggle First-Person View Drone Flight Mode"
          >
            <Radio className="w-3.5 h-3.5 text-[#38BDF8]" />
            {isFpv ? 'EXIT FPV' : 'DRONE FPV'}
          </button>

          <button
            onClick={reset}
            className="im-btn-secondary"
          >
            <RotateCcw className="w-3 h-3" />
            NEW IMAGE
          </button>
        </div>
      </header>

      {/* Main Studio Viewport */}
      <div className="flex-1 flex overflow-hidden relative">
        {isFpv ? (
          <DroneCanvas
            heightmapB64={data.heightmap_b64}
            rgbB64={data.rgb_b64}
            normalMapB64={data.normal_map_b64}
            meshStats={data.mesh_stats}
            verticalScale={verticalScale}
            waterLevel={waterLevel}
            dsmRaw={data.dsm_raw}
            spawnPoint={droneSpawnPoint}
            renderMode={renderMode}
            voxelResolution={voxelResolution}
            voxelBands={voxelBands}
            environmentTheme={environmentTheme}
            noeWaypoints={noeRoute?.waypoints}
            onOpenQuickPanel={() => setIsQuickPanelOpen((prev) => !prev)}
            onEnvironmentThemeToggle={toggleEnvironmentTheme}
            onExitFpv={() => {
              if (document.pointerLockElement) {
                document.exitPointerLock();
              }
              exitFpv();
            }}
          />
        ) : (
          <>
            {/* 3D WebGL Main Viewport */}
            <div className="flex-1 relative h-full">
          {renderMode === 'voxel' ? (
            <VoxelTerrain
              dsmRaw={currentDsm ?? []}
              meshStats={data.mesh_stats}
              verticalScale={verticalScale}
              waterLevel={waterLevel}
              targetResolution={voxelResolution}
              bandCount={voxelBands}
              environmentTheme={environmentTheme}
              observerPos={activeTab === 'viewshed' ? observerPos : null}
              targetPos={activeTab === 'viewshed' ? targetPos : null}
              sightlineClear={activeTab === 'viewshed' ? sightlineClear : null}
              strikePos={activeTab === 'bda' ? strikePos : null}
              strikeSimulation={activeTab === 'bda' ? strikeSimulation : null}
              noeWaypoints={noeRoute?.waypoints}
              onTerrainClick={setExternalClickPoint}
            />
          ) : (
            <TerrainCanvas
              heightmapB64={data.heightmap_b64}
              rgbB64={data.rgb_b64}
              normalMapB64={data.normal_map_b64}
              meshStats={data.mesh_stats}
              verticalScale={verticalScale}
              waterLevel={waterLevel}
              showContours={showContours}
              contourInterval={contourInterval}
              dsmRaw={currentDsm ?? []}
              environmentTheme={environmentTheme}
              viewshedOverlayUrl={activeTab === 'viewshed' ? viewshedOverlayUrl : (activeTab === 'bda' ? bdaOverlayUrl : null)}
              observerPos={activeTab === 'viewshed' ? observerPos : null}
              targetPos={activeTab === 'viewshed' ? targetPos : null}
              sightlineClear={activeTab === 'viewshed' ? sightlineClear : null}
              strikePos={activeTab === 'bda' ? strikePos : null}
              strikeSimulation={activeTab === 'bda' ? strikeSimulation : null}
              noeWaypoints={noeRoute?.waypoints}
              onTerrainClick={setExternalClickPoint}
            />
          )}

          {/* Smooth Mesh Minimap Overlay (§3.6) */}
          {renderMode === 'voxel' && (
            <div className="absolute top-4 right-4 z-20">
              <Minimap
                heightmapB64={data.heightmap_b64}
                rgbB64={data.rgb_b64}
                normalMapB64={data.normal_map_b64}
                meshStats={data.mesh_stats}
                verticalScale={verticalScale}
                waterLevel={waterLevel}
                showContours={false}
                dsmRaw={data.dsm_raw}
                environmentTheme={environmentTheme}
              />
            </div>
          )}

          {/* Floating Elevation Bar */}
          <div className="absolute top-4 left-4 z-10">
            <ColorBar
              min={cal ? cal.min : stats.elevation_min}
              max={cal ? cal.max : stats.elevation_max}
              unit={cal ? cal.unit : 'relative'}
            />
          </div>

          {/* Interactive Tactical HUD Prompt for 3D clicks */}
          {(activeTab === 'viewshed' || activeTab === 'bda') && (
            <div className="absolute top-4 right-4 z-20 pointer-events-none px-3 py-1.5 bg-black/85 backdrop-blur-md border border-cyan-500/40 text-[10px] font-mono text-cyan-300 shadow-xl flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span>
                {activeTab === 'viewshed'
                  ? 'CLICK 3D TERRAIN TO MOVE OBSERVER (SHIFT+CLICK FOR TARGET)'
                  : 'CLICK 3D TERRAIN TO SET BOMB STRIKE GROUND ZERO'}
              </span>
            </div>
          )}
        </div>

        {/* Right Sidebar: Analytical Controls */}
        <aside className="w-[380px] sm:w-[420px] lg:w-[450px] xl:w-[470px] shrink-0 border-l border-white/15 bg-[#07080B] flex flex-col z-10">
          {/* Architectural Segmented Tab Navigation */}
          <div className="grid grid-cols-6 border-b border-white/15 bg-[#050608]">
            {[
              { id: 'surface' as const, num: '01', label: 'SURFACE', icon: Mountain },
              { id: 'flood' as const, num: '02', label: 'FLOOD', icon: Waves },
              { id: 'profile' as const, num: '03', label: 'PROFILE', icon: Scissors },
              { id: 'viewshed' as const, num: '04', label: 'VIEWSHED', icon: Radio },
              { id: 'bda' as const, num: '05', label: 'BDA', icon: Bomb },
              { id: 'inspect' as const, num: '06', label: 'INSPECT', icon: Eye },
            ].map(({ id, num, label, icon: Icon }) => {
              const isActive = activeTab === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setActiveTab(id)}
                  className={`
                    py-3 px-1 flex flex-col items-center justify-center gap-1 transition-all border-r border-white/10 last:border-r-0 relative
                    ${isActive ? 'bg-white/[0.04] text-white' : 'text-neutral-400 hover:text-white hover:bg-white/[0.02]'}
                  `}
                >
                  <div className="flex items-center gap-0.5">
                    <span className="text-[8px] font-mono font-bold text-neutral-400">
                      {num}
                    </span>
                    <span className="text-[9px] font-mono font-black tracking-wider truncate">
                      {label}
                    </span>
                  </div>
                  {isActive && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#38BDF8]" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Active Tab Viewport */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeTab === 'surface' && (
              <SettingsPanel
                verticalScale={verticalScale}
                onVerticalScaleChange={setVerticalScale}
                showContours={showContours}
                onContoursToggle={() => setShowContours(!showContours)}
                contourInterval={contourInterval}
                onContourIntervalChange={setContourInterval}
                renderMode={renderMode}
                onRenderModeChange={setRenderMode}
                voxelBands={voxelBands}
                onVoxelBandsChange={setVoxelBands}
                voxelResolution={voxelResolution}
                onVoxelResolutionChange={setVoxelResolution}
                meshStats={data.mesh_stats}
                calibration={data.calibration}
                isGeoref={data.is_georef}
                crs={data.crs}
                droneSpawnPreset={droneSpawnPreset}
                onDroneSpawnPresetChange={setDroneSpawnPreset}
              />
            )}

            {activeTab === 'flood' && (
              <FloodSimulator
                waterLevel={waterLevel}
                onWaterLevelChange={setWaterLevel}
                elevationMin={cal ? cal.min : stats.elevation_min}
                elevationMax={cal ? cal.max : stats.elevation_max}
                unit={cal ? cal.unit : 'relative'}
                dsmRaw={data.dsm_raw}
              />
            )}

            {activeTab === 'profile' && (
              <CrossSection
                dsmRaw={data.dsm_raw}
                unit={cal ? cal.unit : 'relative'}
                active={true}
                onToggle={() => {}}
                pixelSize={data.mesh_stats.pixel_size}
              />
            )}

            {activeTab === 'viewshed' && (
              <ViewshedAnalyzer
                dsmRaw={data.dsm_raw}
                meshStats={data.mesh_stats}
                unit={cal ? cal.unit : 'relative'}
                requestId={data.request_id}
                noeRoute={noeRoute}
                onNoeRoutePlanned={setNoeRoute}
                onLaunchDroneFlight={enterFpv}
                onOverlayGenerated={setViewshedOverlayUrl}
                onObserverChanged={setObserverPos}
                onTargetChanged={setTargetPos}
                onSightlineStatusChanged={setSightlineClear}
                externalClickPoint={externalClickPoint}
              />
            )}

            {activeTab === 'bda' && (
              <BdaAnalyzer
                dsmRaw={data.dsm_raw}
                meshStats={data.mesh_stats}
                unit={cal ? cal.unit : 'relative'}
                uncertaintyStats={data.uncertainty_stats}
                onOverlayGenerated={setBdaOverlayUrl}
                onStrikeChanged={setStrikePos}
                onSimulateStrike={setStrikeSimulation}
                onResetStrike={() => setStrikeSimulation(null)}
                externalClickPoint={externalClickPoint}
              />
            )}

            {activeTab === 'inspect' && (
              <DualViewer
                rgbB64={data.rgb_b64}
                dsmColorizedB64={data.dsm_colorized_b64}
                elevationMin={cal ? cal.min : stats.elevation_min}
                elevationMax={cal ? cal.max : stats.elevation_max}
                unit={cal ? cal.unit : 'relative'}
              />
            )}
          </div>
        </aside>
          </>
        )}
      </div>

      {/* Floating Terrain & Voxel Detail Quick Panel Overlay (§5) */}
      <TerrainQuickPanel
        isOpen={isQuickPanelOpen}
        onClose={() => setIsQuickPanelOpen(false)}
        settings={terrainSettings}
      />
    </div>
  );
}
