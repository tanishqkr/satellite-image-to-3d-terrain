import React, { useRef, useLayoutEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import DroneModel from './DroneModel.tsx';
import ProximitySensors, { SensorReadings } from './ProximitySensors.tsx';
import { useDroneInput } from './useDroneInput.ts';
import {
  getTerrainElevationAt,
  computeAgl,
  computeMsl,
  resolveTerrainCollision,
  TerrainSamplingOptions,
} from './dsmSampling.ts';

export interface FlightTelemetry {
  speed: number;
  verticalSpeed: number;
  altitudeMsl: number;
  altitudeAgl: number;
  heading: number;
  pitch: number;
  roll: number;
  gridX: number;
  gridZ: number;
  sensors: SensorReadings;
  autopilotMode: 'manual' | 'orbit' | 'transect' | 'noe';
  cameraGimbal: boolean;
  cameraMode?: 'fpv' | 'tpp';
}

export interface DroneFlightControllerProps {
  spawnPoint?: [number, number, number];
  dsmRaw: number[][];
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
    pixel_size?: number;
  };
  verticalScale?: number;
  waterLevel?: number;
  autopilotMode?: 'manual' | 'orbit' | 'transect' | 'noe';
  noeWaypoints?: Array<{ x: number; y: number; z: number; agl_m: number; exposed: boolean; speed_mps?: number }>;
  cameraGimbal?: boolean;
  cameraMode?: 'fpv' | 'tpp';
  droneScale?: number;
  samplingOptions?: TerrainSamplingOptions;
  onAutopilotModeChange?: (mode: 'manual' | 'orbit' | 'transect' | 'noe') => void;
  onCameraGimbalToggle?: () => void;
  onCameraModeChange?: (mode: 'fpv' | 'tpp') => void;
  onQuickPanelToggle?: () => void;
  onEnvironmentThemeToggle?: () => void;
  onTelemetryUpdate?: (telemetry: FlightTelemetry) => void;
}

/**
 * DroneFlightController (Phase 2 - 7 + TPP & Dual-Mesh):
 * - Newtonian flight physics with velocity integration, drag, auto-bank roll, thrust-pitch coupling
 * - Phase 3 AGL clearance & hard-floor/rooftop collision resolution
 * - Phase 4 Proximity sensor soft-bump velocity zeroing
 * - Phase 7 Autopilot Orbital POI recon, Terrain-Hugging Transect cruise, and Horizon-Stabilized Gimbal
 * - §2 TPP Smooth Chase Camera rig with lookAt follow and terrain clearance
 * - §3 Dual-Mesh flight over smooth or voxel-quantized terrain
 * - §4 Live Drone Scale factor
 */
export default function DroneFlightController({
  spawnPoint,
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel,
  autopilotMode: controlledMode,
  cameraGimbal: controlledGimbal,
  cameraMode = 'fpv',
  droneScale = 0.5,
  noeWaypoints,
  samplingOptions,
  onAutopilotModeChange,
  onCameraGimbalToggle,
  onCameraModeChange,
  onQuickPanelToggle,
  onEnvironmentThemeToggle,
  onTelemetryUpdate,
}: DroneFlightControllerProps) {
  const droneGroupRef = useRef<THREE.Group>(null);
  const { camera } = useThree();

  // Internal autopilot state with external control sync
  const [internalMode, setInternalMode] = useState<'manual' | 'orbit' | 'transect' | 'noe'>('manual');
  const [internalGimbal, setInternalGimbal] = useState<boolean>(false);
  const noeWaypointIdx = useRef<number>(0);

  const activeMode = controlledMode ?? internalMode;
  const isGimbal = controlledGimbal ?? internalGimbal;

  const handleModeSelect = (m: 'manual' | 'orbit' | 'transect' | 'noe') => {
    setInternalMode(m);
    onAutopilotModeChange?.(m);

    if (m === 'noe' && noeWaypoints && noeWaypoints.length > 0 && droneGroupRef.current) {
      const pos = droneGroupRef.current.position;
      const pxSize = meshStats.pixel_size || 1.0;
      let closestIdx = 0;
      let minD = Infinity;
      noeWaypoints.forEach((wp, idx) => {
        const targetX = (wp.x / pxSize - meshStats.width / 2) * 0.1;
        const targetZ = (wp.z / pxSize - meshStats.height / 2) * 0.1;
        const d = Math.hypot(pos.x - targetX, pos.z - targetZ);
        if (d < minD) {
          minD = d;
          closestIdx = idx;
        }
      });
      noeWaypointIdx.current = (closestIdx + 1) % noeWaypoints.length;
    }
  };

  const handleGimbalToggle = () => {
    setInternalGimbal((prev) => !prev);
    onCameraGimbalToggle?.();
  };

  const handleCameraModeToggle = () => {
    const next = cameraMode === 'fpv' ? 'tpp' : 'fpv';
    onCameraModeChange?.(next);
  };

  // Keyboard input listener with autopilot & utility shortcut dispatch (1/2/3/G/T/P/L)
  const inputRef = useDroneInput(true, {
    onModeSelect: handleModeSelect,
    onGimbalToggle: handleGimbalToggle,
    onCameraModeToggle: handleCameraModeToggle,
    onQuickPanelToggle,
    onThemeToggle: onEnvironmentThemeToggle,
  });

  // TPP chase camera smooth position tracking ref (§2)
  const chaseCamPos = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const prevCameraMode = useRef<'fpv' | 'tpp'>('fpv');

  // Physics state vectors
  const velocity = useRef(new THREE.Vector3(0, 0, 0));
  const yaw = useRef(0);
  const pitch = useRef(0);
  const roll = useRef(0);
  const propRotation = useRef(0);

  // Autopilot trajectory state (§4.3)
  const orbitAngle = useRef<number>(0);
  const transectDir = useRef<1 | -1>(1);

  // Prop spin animation state for DroneModel
  const [propAngle, setPropAngle] = useState(0);

  // Proximity sensor telemetry state for soft-bump response
  const sensorReadingsRef = useRef<SensorReadings>({
    left: 50,
    right: 50,
    bottom: 50,
    leftScore10: 10,
    rightScore10: 10,
    bottomScore10: 10,
    maxSensorRange: 50,
  });

  // World dimensions
  const worldW = Math.max(1, (meshStats?.width ?? 512) * 0.1);
  const worldD = Math.max(1, (meshStats?.height ?? 512) * 0.1);

  // Compute initial safe spawn altitude above terrain surface
  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;
  const maxElevWorld = isRelative
    ? 18.0 * verticalScale
    : (meshStats.elevation_max - meshStats.elevation_min) * verticalScale * 0.1;

  const limitX = worldW * 0.48;
  const limitZ = worldD * 0.48;
  const initialX = spawnPoint ? THREE.MathUtils.clamp(spawnPoint[0], -limitX, limitX) : 0;
  const initialZ = spawnPoint ? THREE.MathUtils.clamp(spawnPoint[2], -limitZ, limitZ) : 0;
  const spawnGroundY = getTerrainElevationAt(
    initialX,
    initialZ,
    dsmRaw,
    meshStats,
    verticalScale,
    waterLevel,
    samplingOptions
  );
  const initialY = (spawnPoint && spawnPoint[1] > 0)
    ? spawnPoint[1]
    : Math.max(16, spawnGroundY + 10.0, maxElevWorld + 8.0);

  // Initialize drone transform on mount
  useLayoutEffect(() => {
    if (!droneGroupRef.current) return;
    droneGroupRef.current.position.set(initialX, initialY, initialZ);
    droneGroupRef.current.rotation.set(0, 0, 0, 'YXZ');
    velocity.current.set(0, 0, 0);
    yaw.current = 0;
    pitch.current = 0;
    roll.current = 0;

    const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
    camera.position.copy(droneGroupRef.current.position).add(noseOffset);
    camera.quaternion.copy(droneGroupRef.current.quaternion);
    camera.updateMatrixWorld();
  }, [initialX, initialY, initialZ, camera]);

  // Main 60 FPS flight physics loop
  useFrame((_, delta) => {
    if (!droneGroupRef.current) return;

    const dt = Math.min(delta, 0.1);
    const input = inputRef.current;

    // Pilot safety override: any manual thrust input immediately disengages autopilot
    const hasManualThrust = input.pitchForward || input.pitchBackward || input.throttleUp || input.throttleDown || input.yawLeft || input.yawRight;
    if (hasManualThrust && activeMode !== 'manual') {
      handleModeSelect('manual');
    }

    const currentYaw = yaw.current;
    const forwardVec = new THREE.Vector3(-Math.sin(currentYaw), 0, -Math.cos(currentYaw));
    const rightVec = new THREE.Vector3(Math.cos(currentYaw), 0, -Math.sin(currentYaw));

    let currentSpeed = 0;
    let groundWorldY = 0;

    if (activeMode === 'orbit') {
      // 1. Orbital Point-of-Interest (POI) Recon Autopilot Mode (§4.3)
      // Circles the terrain center at fixed radius and safe cruise altitude
      const orbitRadius = Math.min(worldW, worldD) * 0.36;
      const orbitSpeed = 0.26; // rad/s
      orbitAngle.current += orbitSpeed * dt;

      const targetX = orbitRadius * Math.cos(orbitAngle.current);
      const targetZ = orbitRadius * Math.sin(orbitAngle.current);
      groundWorldY = getTerrainElevationAt(targetX, targetZ, dsmRaw, meshStats, verticalScale, waterLevel, samplingOptions);
      const targetY = Math.max(maxElevWorld + 6.0, groundWorldY + 6.0);

      droneGroupRef.current.position.lerp(new THREE.Vector3(targetX, targetY, targetZ), 5 * dt);

      // Tangent heading with smooth banking into orbit
      const targetYaw = -orbitAngle.current - Math.PI / 2;
      yaw.current = THREE.MathUtils.lerp(yaw.current, targetYaw, 4 * dt);
      pitch.current = THREE.MathUtils.lerp(pitch.current, -0.04, 4 * dt);
      roll.current = THREE.MathUtils.lerp(roll.current, -0.16, 5 * dt);

      currentSpeed = orbitRadius * orbitSpeed;
      velocity.current.set(-orbitRadius * Math.sin(orbitAngle.current) * orbitSpeed, 0, orbitRadius * Math.cos(orbitAngle.current) * orbitSpeed);
    } else if (activeMode === 'transect') {
      // 2. Terrain-Hugging Transect Tour Autopilot Mode (§4.3)
      // Cruises West-to-East across tile centerline, hugging terrain at +5.5m AGL
      const transectSpeed = 8.5; // m/s
      const limitX = worldW * 0.40;

      let nextX = droneGroupRef.current.position.x + transectDir.current * transectSpeed * dt;
      if (nextX > limitX) {
        transectDir.current = -1;
        nextX = limitX;
      } else if (nextX < -limitX) {
        transectDir.current = 1;
        nextX = -limitX;
      }

      droneGroupRef.current.position.x = nextX;
      droneGroupRef.current.position.z = THREE.MathUtils.lerp(droneGroupRef.current.position.z, 0, 2 * dt);

      groundWorldY = getTerrainElevationAt(
        droneGroupRef.current.position.x,
        droneGroupRef.current.position.z,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel,
        samplingOptions
      );
      const targetY = groundWorldY + 5.5;
      droneGroupRef.current.position.y = THREE.MathUtils.lerp(droneGroupRef.current.position.y, targetY, 4 * dt);

      const targetYaw = transectDir.current > 0 ? -Math.PI / 2 : Math.PI / 2;
      yaw.current = THREE.MathUtils.lerp(yaw.current, targetYaw, 4 * dt);
      pitch.current = THREE.MathUtils.lerp(pitch.current, -0.05, 4 * dt);
      roll.current = THREE.MathUtils.lerp(roll.current, 0, 4 * dt);

      currentSpeed = transectSpeed;
      velocity.current.set(transectDir.current * transectSpeed, (targetY - droneGroupRef.current.position.y) * 2, 0);
    } else if (activeMode === 'noe' && noeWaypoints && noeWaypoints.length > 0) {
      // 3. Autonomous Nap-of-the-Earth (NOE) Terrain-Following Mode (§4.3)
      const wpList = noeWaypoints;
      const targetWp = wpList[Math.min(wpList.length - 1, noeWaypointIdx.current)];

      // Map metric backend coordinates (origin at corner) to Three.js coordinates (centered at origin)
      const pxSize = meshStats.pixel_size || 1.0;
      const col = targetWp.x / pxSize;
      const row = targetWp.z / pxSize;
      const targetThreeX = (col - meshStats.width / 2) * 0.1;
      const targetThreeZ = (row - meshStats.height / 2) * 0.1;

      groundWorldY = getTerrainElevationAt(
        targetThreeX,
        targetThreeZ,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel,
        samplingOptions
      );

      // Safe terrain-following altitude above ground
      const targetThreeY = groundWorldY + targetWp.agl_m * 0.25;

      const targetPos = new THREE.Vector3(targetThreeX, targetThreeY, targetThreeZ);
      const toTarget = targetPos.clone().sub(droneGroupRef.current.position);
      const distToTarget = toTarget.length();

      // Advance waypoint index when within acceptance radius
      if (distToTarget < 3.2) {
        noeWaypointIdx.current = (noeWaypointIdx.current + 1) % wpList.length;
      }

      const cruiseSpeed = targetWp.speed_mps || 10.0;
      const dir = toTarget.clone().normalize();
      const targetVel = dir.multiplyScalar(cruiseSpeed);

      velocity.current.lerp(targetVel, Math.min(1.0, 5.0 * dt));

      droneGroupRef.current.position.add(velocity.current.clone().multiplyScalar(dt));

      // Heading and pitch alignment with corridor tangents
      const targetYaw = Math.atan2(-velocity.current.x, -velocity.current.z);
      yaw.current = THREE.MathUtils.lerp(yaw.current, targetYaw, 6 * dt);

      const horizSpeed = Math.hypot(velocity.current.x, velocity.current.z);
      const targetPitch = Math.atan2(velocity.current.y, Math.max(0.1, horizSpeed));
      pitch.current = THREE.MathUtils.lerp(pitch.current, -targetPitch * 0.4, 6 * dt);
      roll.current = THREE.MathUtils.lerp(roll.current, -dir.x * 0.25, 5 * dt);

      currentSpeed = velocity.current.length();
    } else {
      // 3. Manual Flight Mode (Phases 2 - 4)
      const turboMult = input.turbo ? 2.6 : 1.0;
      const baseThrust = 45.0 * turboMult;
      const thrustVec = new THREE.Vector3(0, 0, 0);

      // Forward / Backward thrust (W / S)
      if (input.pitchForward) {
        thrustVec.add(forwardVec.clone().multiplyScalar(baseThrust));
      }
      if (input.pitchBackward) {
        thrustVec.sub(forwardVec.clone().multiplyScalar(baseThrust * 0.65));
      }

      // Vertical Ascent / Descent throttle (Space / Q / E)
      let vertThrust = 0;
      if (input.throttleUp) vertThrust += 38.0 * turboMult;
      if (input.throttleDown) vertThrust -= 28.0 * turboMult;
      thrustVec.y += vertThrust;

      // Yaw rotation (A / D)
      let yawRate = 0;
      const yawSpeed = input.turbo ? 2.4 : 1.7;
      if (input.yawLeft) yawRate += yawSpeed;
      if (input.yawRight) yawRate -= yawSpeed;
      yaw.current += yawRate * dt;

      if (yaw.current > Math.PI * 2) yaw.current -= Math.PI * 2;
      if (yaw.current < 0) yaw.current += Math.PI * 2;

      // Newtonian velocity integration
      const dragCoeff = 2.8;
      const accel = thrustVec.clone().sub(velocity.current.clone().multiplyScalar(dragCoeff));
      velocity.current.add(accel.multiplyScalar(dt));

      // Soft-bump velocity zeroing from proximity sensors (§5 - reduced by 80-90% to eliminate false alarms)
      const sensors = sensorReadingsRef.current;
      if (sensors.right < 0.6) {
        const latSpeed = velocity.current.dot(rightVec);
        if (latSpeed > 0) velocity.current.sub(rightVec.clone().multiplyScalar(latSpeed));
      }
      if (sensors.left < 0.6) {
        const latSpeed = velocity.current.dot(rightVec);
        if (latSpeed < 0) velocity.current.sub(rightVec.clone().multiplyScalar(latSpeed));
      }
      if (sensors.bottom < 0.4 && velocity.current.y < 0) {
        velocity.current.y = 0;
      }

      // Terrain Clearance & Collision Resolution (§5)
      const prevPos = droneGroupRef.current.position.clone();
      const candidatePos = prevPos.clone().add(velocity.current.clone().multiplyScalar(dt));

      const collision = resolveTerrainCollision(
        prevPos,
        candidatePos,
        velocity.current,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel,
        1.2,
        samplingOptions
      );

      // Enforce rectangular bounds limits to keep flight over composite terrain
      if (Math.abs(collision.position.x) > limitX) {
        collision.position.x = Math.sign(collision.position.x) * limitX;
        velocity.current.x = 0;
      }
      if (Math.abs(collision.position.z) > limitZ) {
        collision.position.z = Math.sign(collision.position.z) * limitZ;
        velocity.current.z = 0;
      }

      droneGroupRef.current.position.set(
        collision.position.x,
        collision.position.y,
        collision.position.z
      );
      velocity.current.set(
        collision.velocity.x,
        collision.velocity.y,
        collision.velocity.z
      );
      groundWorldY = collision.groundWorldY;

      // Dynamic Pitch and Auto-bank Roll coupling
      const forwardSpeed = velocity.current.dot(forwardVec);
      const lateralSpeed = velocity.current.dot(rightVec);
      const targetPitch = Math.max(-0.26, Math.min(0.18, -forwardSpeed * 0.012));
      pitch.current = THREE.MathUtils.lerp(pitch.current, targetPitch, 8 * dt);

      const targetRoll = Math.max(-0.45, Math.min(0.45, -lateralSpeed * 0.04 - yawRate * 0.12));
      roll.current = THREE.MathUtils.lerp(roll.current, targetRoll, 10 * dt);

      currentSpeed = velocity.current.length();
    }

    // Apply drone body orientation in YXZ Euler order
    droneGroupRef.current.rotation.set(pitch.current, yaw.current, roll.current, 'YXZ');

    // Camera synchronization: Selectable FPV Nose / Gimbal vs TPP Chase View (§2)
    if (cameraMode === 'tpp') {
      // Third-Person Perspective (TPP) Chase Camera (§2)
      // Follows behind (+Z) and above (+Y) the drone's heading frame
      const chaseOffsetLocal = new THREE.Vector3(0, 2.2, 5.2);
      const yawQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw.current);
      const targetCamPos = droneGroupRef.current.position
        .clone()
        .add(chaseOffsetLocal.clone().applyQuaternion(yawQuat));

      // Terrain anti-clipping floor protection
      const camTerrainY = getTerrainElevationAt(
        targetCamPos.x,
        targetCamPos.z,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel,
        samplingOptions
      );
      targetCamPos.y = Math.max(targetCamPos.y, camTerrainY + 0.9);

      if (prevCameraMode.current !== 'tpp') {
        chaseCamPos.current.copy(targetCamPos);
      } else {
        chaseCamPos.current.lerp(targetCamPos, Math.min(1.0, 8.0 * dt));
      }
      camera.position.copy(chaseCamPos.current);

      const lookTarget = droneGroupRef.current.position.clone().add(new THREE.Vector3(0, 0.35, 0));
      camera.lookAt(lookTarget);
    } else {
      // First-Person Perspective (FPV) Nose Camera
      const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
      noseOffset.applyQuaternion(droneGroupRef.current.quaternion);
      camera.position.copy(droneGroupRef.current.position).add(noseOffset);

      if (isGimbal) {
        // 2-axis horizon-stabilized camera: follow heading, keep roll/pitch level
        const gimbalEuler = new THREE.Euler(0, yaw.current, 0, 'YXZ');
        camera.quaternion.setFromEuler(gimbalEuler);
      } else {
        // Rigid nose-locked FPV camera
        camera.quaternion.copy(droneGroupRef.current.quaternion);
      }
    }
    prevCameraMode.current = cameraMode;

    // Propeller spin speed tied to throttle magnitude
    const spinRate = (hasManualThrust || activeMode !== 'manual' ? (input.turbo ? 75 : 50) : 18) + currentSpeed * 0.7;
    propRotation.current += spinRate * dt;
    setPropAngle(propRotation.current);

    // Emit live telemetry to HUD
    if (onTelemetryUpdate) {
      let headingDeg = THREE.MathUtils.radToDeg(-yaw.current);
      if (headingDeg < 0) headingDeg += 360;

      const aglMeters = computeAgl(
        droneGroupRef.current.position.y,
        groundWorldY,
        meshStats,
        verticalScale
      );
      const mslMeters = computeMsl(
        droneGroupRef.current.position.y,
        meshStats,
        verticalScale
      );

      onTelemetryUpdate({
        speed: currentSpeed,
        verticalSpeed: velocity.current.y,
        altitudeMsl: mslMeters,
        altitudeAgl: aglMeters,
        heading: Math.round(headingDeg),
        pitch: THREE.MathUtils.radToDeg(pitch.current),
        roll: THREE.MathUtils.radToDeg(roll.current),
        gridX: droneGroupRef.current.position.x,
        gridZ: droneGroupRef.current.position.z,
        sensors: {
          left: sensorReadingsRef.current.left,
          right: sensorReadingsRef.current.right,
          bottom: aglMeters,
          leftScore10: sensorReadingsRef.current.leftScore10,
          rightScore10: sensorReadingsRef.current.rightScore10,
          bottomScore10:
            sensorReadingsRef.current.maxSensorRange > 0
              ? Math.max(0, Math.min(10.0, (aglMeters / sensorReadingsRef.current.maxSensorRange) * 10.0))
              : sensorReadingsRef.current.bottomScore10,
          maxSensorRange: sensorReadingsRef.current.maxSensorRange,
        },
        autopilotMode: activeMode,
        cameraGimbal: isGimbal,
        cameraMode,
      });
    }
  });

  return (
    <group ref={droneGroupRef}>
      {/* Visual drone model with live scaling in TPP chase view (§1.1, §2 & §4) */}
      {cameraMode === 'tpp' && (
        <group scale={droneScale}>
          <DroneModel
            throttle={velocity.current.length()}
            propRotation={propAngle}
          />
        </group>
      )}
      {/* Proximity sensors: raycast runs continuously, 3D laser visual lines only rendered in TPP (§1.2, §2 & §3) */}
      <ProximitySensors
        droneRef={droneGroupRef}
        dsmRaw={dsmRaw}
        meshStats={meshStats}
        verticalScale={verticalScale}
        waterLevel={waterLevel}
        showVisuals={cameraMode === 'tpp'}
        droneScale={droneScale}
        samplingOptions={samplingOptions}
        onReadingsUpdate={(r) => {
          sensorReadingsRef.current = r;
        }}
      />
    </group>
  );
}
