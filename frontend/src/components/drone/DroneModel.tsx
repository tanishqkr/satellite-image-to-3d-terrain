import React, { forwardRef, useMemo } from 'react';
import * as THREE from 'three';

export interface DroneModelProps {
  throttle?: number;
  propRotation?: number;
}

/**
 * Procedural Drone Mesh Rig (§1):
 * High-performance procedural quadcopter built from native Three.js primitives.
 * Includes central fuselage, 4 diagonal carbon arms, anodized brushless motors,
 * two-blade propellers, navigation LEDs, sensor mount nodes, and nose FPV camera.
 */
export const DroneModel = forwardRef<THREE.Group, DroneModelProps>(function DroneModel(
  { throttle = 0, propRotation = 0 },
  ref
) {
  // Motor arm offsets from center fuselage: [x, y, z]
  const armPositions = useMemo<[number, number, number][]>(() => [
    [-0.65, 0.02, -0.65], // Front-Left (Port)
    [0.65, 0.02, -0.65],  // Front-Right (Starboard)
    [-0.65, 0.02, 0.65],  // Rear-Left
    [0.65, 0.02, 0.65],   // Rear-Right
  ], []);

  // Arm rotations around Y axis: 45 deg, -45 deg, 135 deg, -135 deg
  const armRotations = useMemo<number[]>(() => [
    Math.PI / 4,
    -Math.PI / 4,
    (3 * Math.PI) / 4,
    -(3 * Math.PI) / 4,
  ], []);

  return (
    <group ref={ref}>
      {/* 1. Central Carbon Fuselage */}
      <mesh castShadow receiveShadow position={[0, 0, 0]}>
        <boxGeometry args={[0.42, 0.12, 0.65]} />
        <meshStandardMaterial color="#1a1e26" roughness={0.35} metalness={0.8} />
      </mesh>

      {/* Aerodynamic Top Canopy / Avionics Dome */}
      <mesh position={[0, 0.09, -0.04]}>
        <boxGeometry args={[0.28, 0.08, 0.42]} />
        <meshStandardMaterial color="#0f172a" roughness={0.2} metalness={0.9} />
      </mesh>

      {/* Flight Controller Status Indicator LED */}
      <mesh position={[0, 0.14, -0.05]}>
        <boxGeometry args={[0.06, 0.02, 0.06]} />
        <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={1.5} />
      </mesh>

      {/* 2. Front Nose FPV Camera Housing */}
      <group position={[0, 0.02, -0.34]}>
        {/* Camera Gimbal Bell */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.08, 16]} />
          <meshStandardMaterial color="#0f172a" metalness={0.9} roughness={0.2} />
        </mesh>
        {/* Lens Element */}
        <mesh position={[0, 0, -0.045]}>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial color="#38bdf8" roughness={0.1} metalness={0.9} />
        </mesh>
      </group>

      {/* 3. Four Diagonal Carbon Fiber Arms */}
      {armPositions.map((pos, idx) => (
        <group key={`arm-${idx}`}>
          {/* Carbon Fiber Tube */}
          <mesh
            position={[pos[0] * 0.5, pos[1], pos[2] * 0.5]}
            rotation={[0, armRotations[idx], 0]}
          >
            <boxGeometry args={[0.06, 0.04, 0.85]} />
            <meshStandardMaterial color="#111827" roughness={0.6} metalness={0.5} />
          </mesh>

          {/* Anodized Brushless Motor Bell */}
          <mesh position={pos}>
            <cylinderGeometry args={[0.09, 0.09, 0.10, 16]} />
            <meshStandardMaterial color="#0284c7" metalness={0.85} roughness={0.25} />
          </mesh>

          {/* Motor Shaft Pin */}
          <mesh position={[pos[0], pos[1] + 0.07, pos[2]]}>
            <cylinderGeometry args={[0.02, 0.02, 0.04, 8]} />
            <meshStandardMaterial color="#cbd5e1" metalness={1.0} roughness={0.1} />
          </mesh>

          {/* Propeller Assembly (Static in Phase 1) */}
          <group
            position={[pos[0], pos[1] + 0.08, pos[2]]}
            rotation={[0, propRotation * (idx % 2 === 0 ? 1 : -1) + (idx * Math.PI) / 2, 0]}
          >
            {/* Propeller Blade A */}
            <mesh position={[0.22, 0, 0]}>
              <boxGeometry args={[0.44, 0.012, 0.05]} />
              <meshStandardMaterial color="#e2e8f0" transparent opacity={0.85} roughness={0.4} />
            </mesh>
            {/* Propeller Blade B */}
            <mesh position={[-0.22, 0, 0]}>
              <boxGeometry args={[0.44, 0.012, 0.05]} />
              <meshStandardMaterial color="#e2e8f0" transparent opacity={0.85} roughness={0.4} />
            </mesh>
            {/* Spinner Cone */}
            <mesh position={[0, 0.02, 0]}>
              <coneGeometry args={[0.035, 0.04, 12]} />
              <meshStandardMaterial color="#0f172a" metalness={0.9} />
            </mesh>
          </group>
        </group>
      ))}

      {/* 4. FAA / Aviation Navigation Strobe LEDs */}
      {/* Front-Left: Red (Port) */}
      <mesh position={[-0.65, -0.02, -0.65]}>
        <sphereGeometry args={[0.035, 8, 8]} />
        <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={2.5} />
      </mesh>
      {/* Front-Right: Green (Starboard) */}
      <mesh position={[0.65, -0.02, -0.65]}>
        <sphereGeometry args={[0.035, 8, 8]} />
        <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={2.5} />
      </mesh>
      {/* Rear-Left: Amber */}
      <mesh position={[-0.65, -0.02, 0.65]}>
        <sphereGeometry args={[0.035, 8, 8]} />
        <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={2.0} />
      </mesh>
      {/* Rear-Right: White Strobe */}
      <mesh position={[0.65, -0.02, 0.65]}>
        <sphereGeometry args={[0.035, 8, 8]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={2.5} />
      </mesh>

      {/* 5. Proximity Sensor Emitter Nodes */}
      {/* Left Sensor Pad */}
      <mesh position={[-0.22, 0, 0]}>
        <boxGeometry args={[0.03, 0.04, 0.08]} />
        <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={1.0} />
      </mesh>
      {/* Right Sensor Pad */}
      <mesh position={[0.22, 0, 0]}>
        <boxGeometry args={[0.03, 0.04, 0.08]} />
        <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={1.0} />
      </mesh>
      {/* Ventral Bottom Altimeter */}
      <mesh position={[0, -0.07, 0]}>
        <cylinderGeometry args={[0.035, 0.035, 0.03, 12]} />
        <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={1.0} />
      </mesh>
    </group>
  );
});

export default DroneModel;
