import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import DroneTerrainMesh from './DroneTerrainMesh.tsx';
import { VoxelMesh } from '../VoxelTerrain.tsx';
import DroneFlightController, { FlightTelemetry } from './DroneFlightController.tsx';
import DroneHUD from './DroneHUD.tsx';
import { SensorReadings } from './ProximitySensors.tsx';
import * as THREE from 'three';
import { getTerrainElevationAt } from './dsmSampling.ts';

export interface DroneCanvasProps {
  heightmapB64: string;
  rgbB64: string;
  normalMapB64?: string;
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
    pixel_size?: number;
  };
  verticalScale: number;
  waterLevel: number;
  dsmRaw: number[][];
  spawnPoint?: [number, number, number];
  renderMode?: 'voxel' | 'smooth';
  voxelResolution?: number;
  voxelBands?: number;
  environmentTheme?: 'dark' | 'light';
  noeWaypoints?: Array<{ x: number; y: number; z: number; agl_m: number; exposed: boolean; speed_mps?: number }>;
  onOpenQuickPanel?: () => void;
  onEnvironmentThemeToggle?: () => void;
  onExitFpv: () => void;
}

function NoeCorridorMesh({
  waypoints,
  meshStats,
  dsmRaw,
  verticalScale,
  waterLevel,
}: {
  waypoints?: Array<{ x: number; y: number; z: number; agl_m: number; exposed: boolean }>;
  meshStats: { width: number; height: number; elevation_min: number; elevation_max: number; elevation_range: number; pixel_size?: number };
  dsmRaw: number[][];
  verticalScale: number;
  waterLevel: number;
}) {
  if (!waypoints || waypoints.length < 2) return null;

  const points = React.useMemo(() => {
    const pxSize = meshStats.pixel_size || 1.0;
    return waypoints.map((wp) => {
      const col = wp.x / pxSize;
      const row = wp.z / pxSize;
      const threeX = (col - meshStats.width / 2) * 0.1;
      const threeZ = (row - meshStats.height / 2) * 0.1;
      const groundY = getTerrainElevationAt(threeX, threeZ, dsmRaw, meshStats, verticalScale, waterLevel);
      const threeY = groundY + wp.agl_m * 0.25;
      return new THREE.Vector3(threeX, threeY, threeZ);
    });
  }, [waypoints, meshStats, dsmRaw, verticalScale, waterLevel]);

  const lineGeometry = React.useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  return (
    <primitive object={new THREE.Line(lineGeometry, new THREE.LineBasicMaterial({ color: 0x8b5cf6, linewidth: 3 }))} />
  );
}

/**
 * DroneCanvas: Dedicated full-screen 3D FPV / TPP Drone flight viewport.
 * Supports dual-mesh flight over photoreal smooth terrain or quantized voxel blocks (§3),
 * with live perspective toggle (FPV / TPP), drone scaling (§4), and quick panel access (§5).
 */
export default function DroneCanvas({
  heightmapB64,
  rgbB64,
  normalMapB64,
  meshStats,
  verticalScale,
  waterLevel,
  dsmRaw,
  spawnPoint,
  renderMode = 'smooth',
  voxelResolution = 64,
  voxelBands = 8,
  environmentTheme = 'dark',
  noeWaypoints,
  onOpenQuickPanel,
  onEnvironmentThemeToggle,
  onExitFpv,
}: DroneCanvasProps) {
  const [autopilotMode, setAutopilotMode] = useState<'manual' | 'orbit' | 'transect' | 'noe'>('manual');
  const [cameraGimbal, setCameraGimbal] = useState<boolean>(false);
  const [cameraMode, setCameraMode] = useState<'fpv' | 'tpp'>('fpv');
  const [droneScale, setDroneScale] = useState<number>(0.5);

  const [telemetry, setTelemetry] = useState<FlightTelemetry>({
    speed: 0,
    verticalSpeed: 0,
    altitudeMsl: 0,
    altitudeAgl: 0,
    heading: 0,
    pitch: 0,
    roll: 0,
    gridX: 0,
    gridZ: 0,
    sensors: {
      left: 50,
      right: 50,
      bottom: 25,
      leftScore10: 10,
      rightScore10: 10,
      bottomScore10: 10,
      maxSensorRange: 50,
    },
    autopilotMode: 'manual',
    cameraGimbal: false,
    cameraMode: 'fpv',
  });

  // Global Esc key listener (§6)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onExitFpv();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onExitFpv]);

  const isLight = environmentTheme === 'light';
  // Light mode: Pure clean white environment with crisp black grid lines
  // Dark mode: Untouched dark void (#050608) with slate grid lines (#1e293b)
  const bgColor = isLight ? '#ffffff' : '#050608';
  const gridPrimary = isLight ? '#000000' : '#1e293b';
  const gridSecondary = isLight ? '#1f2937' : '#1e293b';

  // Procedural NOE reconnaissance waypoints fallback so NOE autopilot always flies immediately
  const effectiveNoeWaypoints = React.useMemo(() => {
    if (noeWaypoints && noeWaypoints.length > 1) {
      return noeWaypoints;
    }
    if (!dsmRaw || dsmRaw.length === 0 || !dsmRaw[0]) return undefined;

    const rows = dsmRaw.length;
    const cols = dsmRaw[0].length;
    const pxSize = meshStats.pixel_size || 1.0;
    const wps = [];
    const numPoints = 16;
    for (let i = 0; i < numPoints; i++) {
      const angle = (i / numPoints) * Math.PI * 2;
      // Figure-8 terrain reconnaissance circuit hugging valleys
      const normX = 0.5 + 0.35 * Math.cos(angle);
      const normZ = 0.5 + 0.32 * Math.sin(2 * angle);
      const c = Math.max(2, Math.min(cols - 3, Math.floor(normX * cols)));
      const r = Math.max(2, Math.min(rows - 3, Math.floor(normZ * rows)));
      const elev = dsmRaw[r]?.[c] ?? 0;
      wps.push({
        x: c * pxSize,
        y: elev + 5.0,
        z: r * pxSize,
        agl_m: 5.0,
        speed_mps: 12.0,
        exposed: false,
      });
    }
    return wps;
  }, [noeWaypoints, dsmRaw, meshStats]);

  return (
    <div className={`w-full h-full relative select-none ${isLight ? 'bg-white' : 'bg-[#050608]'}`}>
      <Canvas
        camera={{ fov: 75, near: 0.1, far: 2000 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: bgColor }}
      >
        <color attach="background" args={[bgColor]} />
        <fog attach="fog" args={[bgColor, 80, isLight ? 500 : 400]} />
        <ambientLight intensity={isLight ? 0.95 : 0.45} />
        <directionalLight position={[60, 100, 60]} intensity={isLight ? 1.3 : 1.3} castShadow />
        <directionalLight position={[-40, 50, -40]} intensity={isLight ? 0.5 : 0.3} />

        {/* Dual-Mesh Ground Support: Voxel Blocks vs Photoreal Continuous Displacement (§3) */}
        {renderMode === 'voxel' ? (
          <VoxelMesh
            dsmRaw={dsmRaw}
            meshStats={meshStats}
            verticalScale={verticalScale}
            waterLevel={waterLevel}
            targetResolution={voxelResolution}
            bandCount={voxelBands}
            playEntranceAnimation={false}
          />
        ) : (
          <DroneTerrainMesh
            heightmapB64={heightmapB64}
            rgbB64={rgbB64}
            normalMapB64={normalMapB64}
            meshStats={meshStats}
            verticalScale={verticalScale}
            waterLevel={waterLevel}
          />
        )}

        {/* 3D Nap-of-the-Earth (NOE) Flight Corridor Mesh (§4.3) */}
        <NoeCorridorMesh
          waypoints={autopilotMode === 'noe' ? effectiveNoeWaypoints : undefined}
          meshStats={meshStats}
          dsmRaw={dsmRaw}
          verticalScale={verticalScale}
          waterLevel={waterLevel}
        />

        {/* Procedural drone flight rig, autopilot, and nose/gimbal/TPP camera (§1-§7) */}
        <DroneFlightController
          spawnPoint={spawnPoint}
          dsmRaw={dsmRaw}
          meshStats={meshStats}
          verticalScale={verticalScale}
          waterLevel={waterLevel}
          autopilotMode={autopilotMode}
          noeWaypoints={effectiveNoeWaypoints}
          cameraGimbal={cameraGimbal}
          cameraMode={cameraMode}
          droneScale={droneScale}
          samplingOptions={{
            renderMode,
            voxelResolution,
            voxelBands,
          }}
          onAutopilotModeChange={setAutopilotMode}
          onCameraGimbalToggle={() => setCameraGimbal((prev) => !prev)}
          onCameraModeChange={setCameraMode}
          onQuickPanelToggle={onOpenQuickPanel}
          onEnvironmentThemeToggle={onEnvironmentThemeToggle}
          onTelemetryUpdate={setTelemetry}
        />

        <gridHelper args={[200, 50, gridPrimary, gridSecondary]} position={[0, -0.05, 0]} />
      </Canvas>

      {/* 2D FPV / TPP Heads-Up Display Overlay */}
      <DroneHUD
        onExitFpv={onExitFpv}
        speed={telemetry.speed}
        verticalSpeed={telemetry.verticalSpeed}
        altitudeMsl={telemetry.altitudeMsl}
        altitudeAgl={telemetry.altitudeAgl}
        heading={telemetry.heading}
        pitch={telemetry.pitch}
        roll={telemetry.roll}
        gridX={telemetry.gridX}
        gridZ={telemetry.gridZ}
        sensorLeft={telemetry.sensors.left}
        sensorRight={telemetry.sensors.right}
        sensorBottom={telemetry.sensors.bottom}
        sensorLeftScore={telemetry.sensors.leftScore10}
        sensorRightScore={telemetry.sensors.rightScore10}
        sensorBottomScore={telemetry.sensors.bottomScore10}
        autopilotMode={telemetry.autopilotMode ?? autopilotMode}
        cameraGimbal={telemetry.cameraGimbal ?? cameraGimbal}
        cameraMode={cameraMode}
        droneScale={droneScale}
        environmentTheme={environmentTheme}
        onAutopilotModeChange={setAutopilotMode}
        onCameraGimbalToggle={() => setCameraGimbal((prev) => !prev)}
        onCameraModeToggle={() => setCameraMode((prev) => (prev === 'fpv' ? 'tpp' : 'fpv'))}
        onDroneScaleChange={setDroneScale}
        onQuickPanelToggle={onOpenQuickPanel}
        onEnvironmentThemeToggle={onEnvironmentThemeToggle}
      />
    </div>
  );
}
