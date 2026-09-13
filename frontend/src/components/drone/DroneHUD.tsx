import React, { useMemo } from 'react';
import {
  LogOut,
  Compass,
  Gauge,
  ArrowUp,
  ArrowDown,
  Activity,
  AlertTriangle,
  Radar,
  Radio,
  MapPin,
  Video,
  CircleDot,
  Eye,
  Layers,
  Sun,
  Moon,
} from 'lucide-react';

export interface DroneHUDProps {
  onExitFpv: () => void;
  speed?: number;
  verticalSpeed?: number;
  altitudeMsl?: number;
  altitudeAgl?: number;
  heading?: number;
  pitch?: number;
  roll?: number;
  gridX?: number;
  gridZ?: number;
  sensorLeft?: number;
  sensorRight?: number;
  sensorBottom?: number;
  sensorLeftScore?: number;
  sensorRightScore?: number;
  sensorBottomScore?: number;
  autopilotMode?: 'manual' | 'orbit' | 'transect' | 'noe';
  cameraGimbal?: boolean;
  cameraMode?: 'fpv' | 'tpp';
  droneScale?: number;
  environmentTheme?: 'dark' | 'light';
  onAutopilotModeChange?: (mode: 'manual' | 'orbit' | 'transect' | 'noe') => void;
  onCameraGimbalToggle?: () => void;
  onCameraModeToggle?: () => void;
  onDroneScaleChange?: (scale: number) => void;
  onQuickPanelToggle?: () => void;
  onEnvironmentThemeToggle?: () => void;
}

/**
 * DroneHUD (Phases 5 & 7 Polish):
 * Tactical 2D Heads-Up Display (HUD) / On-Screen Display (OSD) overlay for FPV Drone Mode.
 * Features:
 * - Dynamic artificial horizon with multi-degree pitch ladder bars and roll angle arc
 * - Authentic rolling compass tape with cardinal/degree ticks and centered indicator
 * - Dual airspeed gauges (m/s & km/h) with visual thrust bar
 * - MSL altitude and radar AGL clearance with Vertical Speed Indicator (VSI)
 * - Real-time survey grid coordinates
 * - 3-Sensor proximity LIDAR cluster with color-coded warning pulses
 * - Active proximity obstacle alarm banner
 * - Phase 7 Autopilot Mode Selector (Manual / Orbit / Transect) with instant override
 * - Horizon-Stabilized Gimbal vs FPV Nose Camera toggle
 */
export default function DroneHUD({
  onExitFpv,
  speed = 0,
  verticalSpeed = 0,
  altitudeMsl = 0,
  altitudeAgl = 0,
  heading = 0,
  pitch = 0,
  roll = 0,
  gridX = 0,
  gridZ = 0,
  sensorLeft = 50,
  sensorRight = 50,
  sensorBottom = 50,
  sensorLeftScore,
  sensorRightScore,
  sensorBottomScore,
  autopilotMode = 'manual',
  cameraGimbal = false,
  cameraMode = 'fpv',
  droneScale = 0.5,
  environmentTheme = 'dark',
  onAutopilotModeChange,
  onCameraGimbalToggle,
  onCameraModeToggle,
  onDroneScaleChange,
  onQuickPanelToggle,
  onEnvironmentThemeToggle,
}: DroneHUDProps) {
  const speedKmh = (speed * 3.6).toFixed(1);
  const speedMs = speed.toFixed(1);

  // Scaled scores out of 10 for proximity display (dynamically scaled or fallback)
  const leftScore = sensorLeftScore !== undefined ? sensorLeftScore : Math.min(10.0, Math.max(0, (sensorLeft / 20.0) * 10.0));
  const rightScore = sensorRightScore !== undefined ? sensorRightScore : Math.min(10.0, Math.max(0, (sensorRight / 20.0) * 10.0));
  const bottomScore = sensorBottomScore !== undefined ? sensorBottomScore : Math.min(10.0, Math.max(0, (sensorBottom / 20.0) * 10.0));

  // Floating-point deadband clamping to eliminate micro-jitter near zero (§1.1)
  const displayPitch = Math.abs(pitch) < 0.05 ? 0 : pitch;
  const displayRoll = Math.abs(roll) < 0.05 ? 0 : roll;

  // Compass heading direction label
  const getHeadingLabel = (deg: number) => {
    const d = ((deg % 360) + 360) % 360;
    if (d >= 337.5 || d < 22.5) return 'N';
    if (d >= 22.5 && d < 67.5) return 'NE';
    if (d >= 67.5 && d < 112.5) return 'E';
    if (d >= 112.5 && d < 157.5) return 'SE';
    if (d >= 157.5 && d < 202.5) return 'S';
    if (d >= 202.5 && d < 247.5) return 'SW';
    if (d >= 247.5 && d < 292.5) return 'W';
    return 'NW';
  };

  // Rolling compass tape generation: continuous ticks across visible window
  const visibleTicks = useMemo(() => {
    const ticks: { deg: number; label?: string; isMajor: boolean; offsetPx: number }[] = [];
    const minDeg = Math.floor((heading - 45) / 5) * 5;
    const maxDeg = Math.ceil((heading + 45) / 5) * 5;

    for (let d = minDeg; d <= maxDeg; d += 5) {
      const normalized = ((d % 360) + 360) % 360;
      const offsetPx = (d - heading) * 3.2;
      let label: string | undefined = undefined;

      if (normalized === 0) label = 'N';
      else if (normalized === 45) label = 'NE';
      else if (normalized === 90) label = 'E';
      else if (normalized === 135) label = 'SE';
      else if (normalized === 180) label = 'S';
      else if (normalized === 225) label = 'SW';
      else if (normalized === 270) label = 'W';
      else if (normalized === 315) label = 'NW';
      else if (normalized % 15 === 0) {
        label = normalized.toString().padStart(3, '0');
      }

      ticks.push({
        deg: normalized,
        label,
        isMajor: d % 15 === 0,
        offsetPx,
      });
    }
    return ticks;
  }, [heading]);

  // Proximity Alert: reduced by 80-90% to avoid constant alarms during normal flight
  const hasProximityAlert = sensorLeft < 0.6 || sensorRight < 0.6 || sensorBottom < 0.4;

  return (
    <div className="absolute inset-0 pointer-events-none z-30 select-none overflow-hidden font-mono text-white">
      {/* 1. Top Avionics & Flight Control Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-auto">
        {/* Left: Exit Button & Autopilot Mode Selector (§4.3 & §7) */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onExitFpv}
            className="px-3 py-1.5 bg-black/85 border border-white/20 hover:border-emerald-400 hover:text-emerald-400 text-white text-xs font-bold tracking-wider flex items-center gap-1.5 transition-all shadow-xl"
            title="Return to 3D Spectator Studio [ESC]"
          >
            <LogOut className="w-3.5 h-3.5 text-emerald-400" />
            EXIT [ESC]
          </button>

          {/* Autopilot Mode Selector */}
          <div className="flex items-center bg-black/85 border border-white/15 p-0.5 text-[10px] font-bold">
            <button
              type="button"
              onClick={() => onAutopilotModeChange?.('manual')}
              className={`px-2.5 py-1 transition-all ${
                autopilotMode === 'manual'
                  ? 'bg-emerald-500 text-black shadow-sm font-black'
                  : 'text-neutral-400 hover:text-white'
              }`}
              title="Manual Flight [Key 1]"
            >
              [1] MANUAL
            </button>
            <button
              type="button"
              onClick={() => onAutopilotModeChange?.('orbit')}
              className={`px-2.5 py-1 transition-all ${
                autopilotMode === 'orbit'
                  ? 'bg-[#38BDF8] text-black shadow-sm font-black'
                  : 'text-neutral-400 hover:text-white'
              }`}
              title="Orbital POI Recon Autopilot [Key 2]"
            >
              [2] ORBIT
            </button>
            <button
              type="button"
              onClick={() => onAutopilotModeChange?.('transect')}
              className={`px-2.5 py-1 transition-all ${
                autopilotMode === 'transect'
                  ? 'bg-amber-400 text-black shadow-sm font-black'
                  : 'text-neutral-400 hover:text-white'
              }`}
              title="Terrain-Hugging Transect Cruise [Key 3]"
            >
              [3] TRANSECT
            </button>
            <button
              type="button"
              onClick={() => onAutopilotModeChange?.('noe')}
              className={`px-2.5 py-1 transition-all ${
                autopilotMode === 'noe'
                  ? 'bg-[#8B5CF6] text-white shadow-sm font-black'
                  : 'text-neutral-400 hover:text-white'
              }`}
              title="3D Nap-of-the-Earth Autonomous Routing Corridor [Key 4]"
            >
              [4] NOE
            </button>
          </div>
        </div>

        {/* Center: Authentic Rolling Compass Tape */}
        <div className="flex flex-col items-center">
          <div className="flex items-center gap-1 text-[11px] font-bold text-[#38BDF8] bg-black/80 px-2 py-0.5 border border-white/10 mb-0.5 shadow-md">
            <Compass className="w-3 h-3" />
            <span>{heading.toString().padStart(3, '0')}°</span>
            <span className="text-neutral-300">({getHeadingLabel(heading)})</span>
          </div>

          <div className="relative w-64 h-8 bg-black/85 border border-white/20 overflow-hidden flex items-center justify-center shadow-lg">
            {/* Compass Center Cursor Needle */}
            <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-[1.5px] bg-[#38BDF8] z-10 shadow-[0_0_6px_#38bdf8]" />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-0 h-0 border-x-4 border-x-transparent border-t-[5px] border-t-[#38BDF8] z-10" />

            {/* Moving ticks container */}
            <div className="absolute inset-0 flex items-center justify-center">
              {visibleTicks.map((tick, idx) => (
                <div
                  key={idx}
                  className="absolute flex flex-col items-center"
                  style={{ transform: `translateX(${tick.offsetPx}px)` }}
                >
                  <div
                    className={`${
                      tick.isMajor
                        ? 'h-3 w-[1.5px] bg-white'
                        : 'h-1.5 w-[1px] bg-white/40'
                    }`}
                  />
                  {tick.label && (
                    <span
                      className={`text-[8px] font-bold tracking-tighter mt-0.5 ${
                        tick.label === 'N'
                          ? 'text-red-400 font-black'
                          : tick.label.length === 2
                          ? 'text-[#38BDF8]'
                          : 'text-neutral-300'
                      }`}
                    >
                      {tick.label}
                    </span>
                  )}
                </div>
              ))}
            </div>

            {/* Edge Fade Gradients */}
            <div className="absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-black via-black/80 to-transparent z-10 pointer-events-none" />
            <div className="absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-black via-black/80 to-transparent z-10 pointer-events-none" />
          </div>
        </div>

        {/* Right: Camera Mode Toggle, TPP Scale Slider, Quick Panel & Avionics Status (§2, §4, §5, §7) */}
        <div className="flex items-center gap-2">
          {/* Perspective Toggle Button (FPV / TPP) (§2) */}
          <button
            type="button"
            onClick={onCameraModeToggle}
            className={`px-2.5 py-1.5 border text-[10px] font-bold flex items-center gap-1.5 transition-all shadow-md ${
              cameraMode === 'tpp'
                ? 'bg-[#38BDF8]/25 text-[#38BDF8] border-[#38BDF8]/70 shadow-[#38BDF8]/20'
                : 'bg-black/80 text-neutral-300 border-white/20 hover:text-white'
            }`}
            title="Toggle Perspective: FPV Cockpit vs TPP Chase [Key T]"
          >
            <Eye className="w-3 h-3 text-[#38BDF8]" />
            <span>{cameraMode === 'tpp' ? 'CAM: TPP CHASE [T]' : 'CAM: FPV NOSE [T]'}</span>
          </button>

          {/* TPP Drone Size Slider (§4) */}
          {cameraMode === 'tpp' && (
            <div className="flex items-center gap-2 px-2.5 py-1 bg-black/85 border border-white/20 text-[10px] shadow-lg">
              <span className="text-neutral-400 font-bold">SIZE:</span>
              <input
                type="range"
                min={0.5}
                max={3.0}
                step={0.1}
                value={droneScale}
                onChange={(e) => onDroneScaleChange?.(parseFloat(e.target.value))}
                className="w-16 accent-[#38BDF8] cursor-pointer"
                title="Adjust visible drone model scale (0.5x - 3.0x)"
              />
              <span className="text-[#38BDF8] font-black min-w-[28px]">{droneScale.toFixed(1)}x</span>
            </div>
          )}

          {/* Stabilized Gimbal Toggle Button (FPV mode only) */}
          {cameraMode === 'fpv' && (
            <button
              type="button"
              onClick={onCameraGimbalToggle}
              className={`px-2.5 py-1.5 border text-[10px] font-bold flex items-center gap-1.5 transition-all shadow-md ${
                cameraGimbal
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/60 shadow-emerald-500/20'
                  : 'bg-black/80 text-neutral-400 border-white/20 hover:text-white'
              }`}
              title="Toggle Horizon-Stabilized Gimbal [Key G]"
            >
              <Video className="w-3 h-3" />
              <span>GIMBAL: {cameraGimbal ? 'STAB' : 'RIGID'} [G]</span>
            </button>
          )}

          {/* Quick Terrain & Voxel Detail Panel Button (§5) */}
          <button
            type="button"
            onClick={onQuickPanelToggle}
            className="px-2.5 py-1.5 bg-black/85 border border-white/20 hover:border-[#38BDF8] hover:text-[#38BDF8] text-neutral-300 text-[10px] font-bold flex items-center gap-1.5 transition-all shadow-md"
            title="Toggle Terrain & Voxel Detail Quick Panel [Key P]"
          >
            <Layers className="w-3 h-3 text-[#38BDF8]" />
            <span>TERRAIN [P]</span>
          </button>

          {/* Environment Theme Toggle (Dark Void vs Light Studio) */}
          <button
            type="button"
            onClick={onEnvironmentThemeToggle}
            className={`px-2.5 py-1.5 border text-[10px] font-bold flex items-center gap-1.5 transition-all shadow-md ${
              environmentTheme === 'light'
                ? 'bg-amber-400 text-black border-amber-400 font-black shadow-amber-400/20'
                : 'bg-black/85 text-neutral-300 border-white/20 hover:text-white hover:border-[#38BDF8]'
            }`}
            title="Toggle Environment Background: Dark Void vs Light Studio [Key L]"
          >
            {environmentTheme === 'light' ? (
              <Sun className="w-3 h-3 text-black" />
            ) : (
              <Moon className="w-3 h-3 text-[#38BDF8]" />
            )}
            <span>{environmentTheme === 'light' ? 'LIGHT [L]' : 'DARK [L]'}</span>
          </button>

          <div className="px-3 py-1 bg-black/80 border border-white/15 text-[10px] text-neutral-300 flex items-center gap-2 shadow-lg">
            <Radio className="w-3 h-3 text-[#38BDF8]" />
            <span>LINK: <strong className="text-emerald-400">99%</strong></span>
            <span className="text-neutral-500">|</span>
            <span>BATT: <strong className="text-emerald-400">16.2V</strong></span>
          </div>

          <div className="px-3 py-1 bg-black/80 border border-white/15 text-[10px] text-neutral-300 flex items-center gap-1.5 shadow-lg">
            <Activity className="w-3 h-3 text-[#38BDF8]" />
            <span>ARMED</span>
          </div>
        </div>
      </div>

      {/* Autopilot Status Notification Banner */}
      {autopilotMode !== 'manual' && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-black/90 border border-[#38BDF8] text-[#38BDF8] text-xs font-bold tracking-wider uppercase flex items-center gap-2.5 shadow-2xl backdrop-blur-md animate-pulse pointer-events-none">
          <CircleDot className="w-3.5 h-3.5 text-[#38BDF8] animate-spin" />
          <span>
            {autopilotMode === 'orbit'
              ? 'AUTOPILOT ENGAGED · ORBITAL POINT-OF-INTEREST RECON'
              : autopilotMode === 'transect'
              ? 'AUTOPILOT ENGAGED · TERRAIN-HUGGING TRANSECT (5.5M AGL)'
              : 'AUTOPILOT ENGAGED · 3D NAP-OF-THE-EARTH TERRAIN MASKING'}
          </span>
          <span className="text-[10px] text-neutral-400 font-normal">
            (TOUCH WASD TO OVERRIDE)
          </span>
        </div>
      )}

      {/* 2. Artificial Horizon & Center Pitch Ladder (§1.1: direct 60 FPS transform without transition jitter) */}
      <div
        className="absolute inset-0 flex items-center justify-center pointer-events-none"
        style={{
          transform: `rotate(${-displayRoll}deg) translateY(${displayPitch * 4}px)`,
        }}
      >
        <div className="relative w-80 h-80 flex items-center justify-center pointer-events-none">
          {/* Horizon Line (0 deg pitch) with center gap and bracketed ticks matching ladder (§1.1) */}
          <div className="absolute w-44 flex items-center justify-between">
            <div className="flex items-center">
              <div className="w-2 h-1.5 border-l-2 border-t-2 border-emerald-400/90" />
              <div className="w-14 h-[1.5px] bg-emerald-400/90 shadow-[0_0_8px_rgba(52,211,153,0.7)]" />
            </div>
            <div className="flex items-center">
              <div className="w-14 h-[1.5px] bg-emerald-400/90 shadow-[0_0_8px_rgba(52,211,153,0.7)]" />
              <div className="w-2 h-1.5 border-r-2 border-t-2 border-emerald-400/90" />
            </div>
          </div>

          {/* +20 deg pitch rung */}
          <div className="absolute top-6 w-32 flex justify-between text-[9px] text-emerald-400 font-bold">
            <div className="flex items-start gap-1">
              <span>+20</span>
              <div className="w-8 border-t-2 border-l-2 border-emerald-400/60 h-2" />
            </div>
            <div className="flex items-start gap-1">
              <div className="w-8 border-t-2 border-r-2 border-emerald-400/60 h-2" />
              <span>+20</span>
            </div>
          </div>

          {/* +10 deg pitch rung */}
          <div className="absolute top-20 w-24 flex justify-between text-[9px] text-emerald-400 font-bold">
            <div className="flex items-start gap-1">
              <span>+10</span>
              <div className="w-6 border-t border-l border-emerald-400/50 h-1.5" />
            </div>
            <div className="flex items-start gap-1">
              <div className="w-6 border-t border-r border-emerald-400/50 h-1.5" />
              <span>+10</span>
            </div>
          </div>

          {/* -10 deg pitch rung (dashed) */}
          <div className="absolute bottom-20 w-24 flex justify-between text-[9px] text-emerald-400/80 font-bold">
            <div className="flex items-end gap-1">
              <span>-10</span>
              <div className="w-6 border-b border-l border-dashed border-emerald-400/50 h-1.5" />
            </div>
            <div className="flex items-end gap-1">
              <div className="w-6 border-b border-r border-dashed border-emerald-400/50 h-1.5" />
              <span>-10</span>
            </div>
          </div>

          {/* -20 deg pitch rung (dashed) */}
          <div className="absolute bottom-6 w-32 flex justify-between text-[9px] text-emerald-400/80 font-bold">
            <div className="flex items-end gap-1">
              <span>-20</span>
              <div className="w-8 border-b-2 border-l-2 border-dashed border-emerald-400/60 h-2" />
            </div>
            <div className="flex items-end gap-1">
              <div className="w-8 border-b-2 border-r-2 border-dashed border-emerald-400/60 h-2" />
              <span>-20</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Static Center Boresight & Crosshair */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="relative flex items-center justify-center">
          {/* Outer Crosshair Wings */}
          <div className="absolute -left-12 w-6 h-[1.5px] bg-emerald-400/70" />
          <div className="absolute -right-12 w-6 h-[1.5px] bg-emerald-400/70" />
          <div className="absolute -top-12 h-6 w-[1.5px] bg-emerald-400/70" />

          {/* Center Flight Pip */}
          <div className="w-6 h-6 border border-emerald-400/60 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full shadow-[0_0_6px_#34d399]" />
          </div>
        </div>
      </div>

      {/* 4. Left Flight Telemetry: Airspeed & Attitude */}
      <div className="absolute left-6 top-1/2 -translate-y-1/2 p-3 bg-black/85 border border-white/15 space-y-2.5 text-xs shadow-2xl backdrop-blur-md">
        <div className="flex items-center gap-1.5 text-neutral-400 text-[10px] uppercase font-bold tracking-wider">
          <Gauge className="w-3.5 h-3.5 text-[#38BDF8]" />
          AIRSPEED
        </div>

        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-black text-white tracking-tight">{speedMs}</span>
          <span className="text-[10px] text-neutral-400 font-bold">M/S</span>
        </div>

        <div className="text-[10px] text-neutral-400">
          GROUND SPD: <strong className="text-neutral-200">{speedKmh} KM/H</strong>
        </div>

        {/* Speed Bar Gauge */}
        <div className="w-32 h-1.5 bg-neutral-800 rounded-full overflow-hidden border border-white/10">
          <div
            className="h-full bg-gradient-to-r from-emerald-500 to-[#38BDF8] transition-all duration-75"
            style={{ width: `${Math.min(100, (speed / 25) * 100)}%` }}
          />
        </div>

        <div className="pt-2 border-t border-white/10 text-[10px] space-y-1">
          <div className="flex justify-between gap-4">
            <span className="text-neutral-500">PITCH:</span>
            <span className="text-neutral-200 font-bold">{pitch >= 0 ? `+${pitch.toFixed(1)}°` : `${pitch.toFixed(1)}°`}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-neutral-500">BANK:</span>
            <span className="text-neutral-200 font-bold">{roll >= 0 ? `+${roll.toFixed(1)}°` : `${roll.toFixed(1)}°`}</span>
          </div>
        </div>
      </div>

      {/* 5. Right Flight Telemetry: Altitude & VSI */}
      <div className="absolute right-6 top-1/2 -translate-y-1/2 p-3 bg-black/85 border border-white/15 space-y-2.5 text-xs shadow-2xl backdrop-blur-md text-right">
        <div className="flex items-center justify-end gap-1.5 text-neutral-400 text-[10px] uppercase font-bold tracking-wider">
          <ArrowUp className="w-3.5 h-3.5 text-[#34D399]" />
          ALTITUDE (MSL)
        </div>

        <div className="flex items-baseline justify-end gap-1.5">
          <span className="text-2xl font-black text-white tracking-tight">{altitudeMsl.toFixed(1)}</span>
          <span className="text-[10px] text-neutral-400 font-bold">M</span>
        </div>

        <div className="text-[10px] text-neutral-400">
          RADAR AGL: <strong className="text-emerald-400">{altitudeAgl.toFixed(1)} M</strong>
        </div>

        {/* Vertical Speed Indicator (VSI) */}
        <div className="flex items-center justify-end gap-1.5 text-[10px]">
          <span className="text-neutral-500">VSI:</span>
          <span className={`font-bold flex items-center gap-0.5 ${
            verticalSpeed > 0.3
              ? 'text-emerald-400'
              : verticalSpeed < -0.3
              ? 'text-amber-400'
              : 'text-neutral-400'
          }`}>
            {verticalSpeed > 0.3 ? (
              <ArrowUp className="w-3 h-3" />
            ) : verticalSpeed < -0.3 ? (
              <ArrowDown className="w-3 h-3" />
            ) : null}
            {verticalSpeed >= 0 ? `+${verticalSpeed.toFixed(1)}` : verticalSpeed.toFixed(1)} M/S
          </span>
        </div>

        <div className="pt-2 border-t border-white/10 text-[10px] text-neutral-500">
          DATUM: WGS84 / EGM96
        </div>
      </div>

      {/* 6. Bottom-Left: Survey Grid Position */}
      <div className="absolute bottom-4 left-6 flex items-center gap-3 p-2.5 bg-black/85 border border-white/15 text-[10px] text-neutral-300 backdrop-blur-md shadow-2xl">
        <div className="flex items-center gap-1.5 text-[#38BDF8] font-bold">
          <MapPin className="w-3.5 h-3.5" />
          <span>GRID:</span>
        </div>
        <div>
          X: <strong className="text-white">{gridX >= 0 ? `+${gridX.toFixed(1)}` : gridX.toFixed(1)}m</strong>
        </div>
        <div>
          Z: <strong className="text-white">{gridZ >= 0 ? `+${gridZ.toFixed(1)}` : gridZ.toFixed(1)}m</strong>
        </div>
        <div className="text-neutral-500">|</div>
        <div className="text-neutral-400">
          RTK FIX <strong className="text-emerald-400">±2CM</strong>
        </div>
      </div>

      {/* 7. Bottom-Center: Flight Controls Cheat Sheet */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 p-2 bg-black/85 border border-white/15 text-[10px] text-neutral-300 flex items-center gap-2.5 backdrop-blur-md shadow-2xl">
        <div className="text-emerald-400 font-bold">CONTROLS:</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">W / S</kbd> Thrust</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">A / D</kbd> Yaw & Bank</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">SPACE / Q</kbd> Up</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">E</kbd> Down</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">1/2/3</kbd> Autopilot</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">T</kbd> TPP/FPV</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">P</kbd> Terrain</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">G</kbd> Gimbal</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">L</kbd> Theme</div>
      </div>

      {/* 8. Bottom-Right: 3-Sensor Proximity LIDAR Array Cluster (§5) */}
      <div className="absolute bottom-4 right-6 flex items-center gap-1.5 p-2 bg-black/85 border border-white/15 backdrop-blur-md text-[11px] shadow-2xl">
        <div className="flex items-center gap-1 text-[9px] text-neutral-400 font-bold tracking-wider px-0.5">
          <Radar className="w-3 h-3 text-[#38BDF8]" />
          LIDAR:
        </div>

        {/* Left Sensor */}
        <div
          className={`px-1.5 py-0.5 border font-bold flex items-center gap-1 transition-colors ${
            sensorLeft < 0.6
              ? 'bg-red-500/25 border-red-500 text-red-400 animate-pulse'
              : sensorLeft <= 2.0
              ? 'bg-amber-500/15 border-amber-500/60 text-amber-300'
              : 'bg-black/60 border-emerald-500/40 text-emerald-400'
          }`}
        >
          <span className="text-[8px] text-neutral-400">L:</span>
          <span>{leftScore.toFixed(1)}/10</span>
          <span className="text-[8px] opacity-70">({sensorLeft >= 50 ? '>50m' : `${sensorLeft.toFixed(1)}m`})</span>
        </div>

        {/* Down / AGL Sensor */}
        <div
          className={`px-1.5 py-0.5 border font-bold flex items-center gap-1 transition-colors ${
            sensorBottom < 0.4
              ? 'bg-red-500/25 border-red-500 text-red-400 animate-pulse'
              : sensorBottom <= 2.0
              ? 'bg-amber-500/15 border-amber-500/60 text-amber-300'
              : 'bg-black/60 border-emerald-500/40 text-emerald-400'
          }`}
        >
          <span className="text-[8px] text-neutral-400">AGL:</span>
          <span>{bottomScore.toFixed(1)}/10</span>
          <span className="text-[8px] opacity-70">({sensorBottom >= 50 ? '>50m' : `${sensorBottom.toFixed(1)}m`})</span>
        </div>

        {/* Right Sensor */}
        <div
          className={`px-1.5 py-0.5 border font-bold flex items-center gap-1 transition-colors ${
            sensorRight < 0.6
              ? 'bg-red-500/25 border-red-500 text-red-400 animate-pulse'
              : sensorRight <= 2.0
              ? 'bg-amber-500/15 border-amber-500/60 text-amber-300'
              : 'bg-black/60 border-emerald-500/40 text-emerald-400'
          }`}
        >
          <span className="text-[8px] text-neutral-400">R:</span>
          <span>{rightScore.toFixed(1)}/10</span>
          <span className="text-[8px] opacity-70">({sensorRight >= 50 ? '>50m' : `${sensorRight.toFixed(1)}m`})</span>
        </div>
      </div>

      {/* 9. Proximity Obstacle Warning Alert Banner */}
      {hasProximityAlert && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-red-500/25 border border-red-500 text-red-400 font-bold text-xs tracking-widest uppercase flex items-center gap-2 shadow-2xl animate-pulse">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <span>PROXIMITY WARNING - TERRAIN OBSTACLE CLOSE</span>
        </div>
      )}
    </div>
  );
}
