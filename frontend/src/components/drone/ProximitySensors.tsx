import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { castSensor, MeshElevationStats, TerrainSamplingOptions } from './dsmSampling.ts';

export interface SensorReadings {
  left: number;
  right: number;
  bottom: number;
  leftScore10: number;
  rightScore10: number;
  bottomScore10: number;
  maxSensorRange: number;
}

export interface ProximitySensorsProps {
  droneRef: React.RefObject<THREE.Group | null>;
  dsmRaw: number[][];
  meshStats: MeshElevationStats;
  verticalScale?: number;
  waterLevel?: number;
  showVisuals?: boolean;
  droneScale?: number;
  samplingOptions?: TerrainSamplingOptions;
  onReadingsUpdate?: (readings: SensorReadings) => void;
}

/**
 * Proximity Alert Thresholds (Reduced by 80-90% to eliminate false alarms):
 * - Red Alert: Obstacle dangerously close (< 0.6m or < 5% of maxRange)
 * - Yellow Warning: Obstacle within caution zone (< 2.0m or < 18% of maxRange)
 * - Green: Clear (> 2.0m)
 */
function getSensorColor(distance: number, maxRange: number): THREE.Color {
  if (distance < Math.max(0.6, maxRange * 0.05)) return new THREE.Color('#ef4444');
  if (distance < Math.max(2.0, maxRange * 0.18)) return new THREE.Color('#fbbf24');
  return new THREE.Color('#10b981');
}

function getSensorColorHex(distance: number, maxRange: number): string {
  if (distance < Math.max(0.6, maxRange * 0.05)) return '#ef4444';
  if (distance < Math.max(2.0, maxRange * 0.18)) return '#fbbf24';
  return '#10b981';
}

/**
 * 3-Sensor Proximity Array (§5):
 * Casts client-side ray-marches Left (-X), Right (+X), and Bottom (-Y) into dsm_raw.
 * Renders in-scene color-coded laser lines locked to the drone body with live distance
 * numbers scaled out of 10 according to the input image / terrain size.
 */
export default function ProximitySensors({
  droneRef,
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel,
  showVisuals = true,
  droneScale = 0.5,
  samplingOptions,
  onReadingsUpdate,
}: ProximitySensorsProps) {
  const { camera } = useThree();

  // Laser line geometries with dynamic 2-vertex buffer attributes
  const leftGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);

  const rightGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);

  const bottomGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);

  // Three.js Line objects
  const leftLine = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#10b981', transparent: true, opacity: 0.85 });
    return new THREE.Line(leftGeo, mat);
  }, [leftGeo]);

  const rightLine = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#10b981', transparent: true, opacity: 0.85 });
    return new THREE.Line(rightGeo, mat);
  }, [rightGeo]);

  const bottomLine = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#10b981', transparent: true, opacity: 0.85 });
    return new THREE.Line(bottomGeo, mat);
  }, [bottomGeo]);

  useEffect(() => {
    return () => {
      leftGeo.dispose();
      rightGeo.dispose();
      bottomGeo.dispose();
      (leftLine.material as THREE.Material).dispose();
      (rightLine.material as THREE.Material).dispose();
      (bottomLine.material as THREE.Material).dispose();
    };
  }, [leftGeo, rightGeo, bottomGeo, leftLine, rightLine, bottomLine]);

  // Impact dot mesh refs
  const leftDotRef = useRef<THREE.Mesh>(null);
  const rightDotRef = useRef<THREE.Mesh>(null);
  const bottomDotRef = useRef<THREE.Mesh>(null);

  const leftDotMatRef = useRef<THREE.MeshBasicMaterial>(null);
  const rightDotMatRef = useRef<THREE.MeshBasicMaterial>(null);
  const bottomDotMatRef = useRef<THREE.MeshBasicMaterial>(null);

  // 3D Floating label groups and direct DOM refs for 60 FPS zero-rerender updates
  const leftLabelGroupRef = useRef<THREE.Group>(null);
  const rightLabelGroupRef = useRef<THREE.Group>(null);
  const bottomLabelGroupRef = useRef<THREE.Group>(null);

  const leftBoxRef = useRef<HTMLDivElement>(null);
  const rightBoxRef = useRef<HTMLDivElement>(null);
  const bottomBoxRef = useRef<HTMLDivElement>(null);

  const leftScoreRef = useRef<HTMLSpanElement>(null);
  const rightScoreRef = useRef<HTMLSpanElement>(null);
  const bottomScoreRef = useRef<HTMLSpanElement>(null);

  const leftDistRef = useRef<HTMLSpanElement>(null);
  const rightDistRef = useRef<HTMLSpanElement>(null);
  const bottomDistRef = useRef<HTMLSpanElement>(null);

  // Dynamic max sensor range scaled to the image/terrain dimensions:
  const worldW = Math.max(1, (meshStats?.width ?? 512) * 0.1);
  const worldD = Math.max(1, (meshStats?.height ?? 512) * 0.1);
  const mapSpan = Math.min(worldW, worldD);
  const maxSensorRange = Math.max(15.0, mapSpan * 0.40);

  // Guard against near-plane WebGL projection inversion
  const isBehindCameraNearPlane = (pt: THREE.Vector3) => {
    const pCam = pt.clone().applyMatrix4(camera.matrixWorldInverse);
    return pCam.z >= -(camera.near + 0.05);
  };

  useFrame(() => {
    if (!droneRef.current) return;

    const drone = droneRef.current;
    drone.updateMatrixWorld();

    const scale = droneScale ?? 1.0;

    // 1. Precise local anchor points locked to the physical emitter nodes on DroneModel:
    // Left emitter node: [-0.22, 0, 0]
    // Right emitter node: [0.22, 0, 0]
    // Bottom ventral altimeter: [0, -0.07, 0]
    const leftAnchorLocal = new THREE.Vector3(-0.22 * scale, 0, 0);
    const rightAnchorLocal = new THREE.Vector3(0.22 * scale, 0, 0);
    const bottomAnchorLocal = new THREE.Vector3(0, -0.07 * scale, 0);

    // World origins and directions for ray-marching
    const leftOriginWorld = leftAnchorLocal.clone().applyMatrix4(drone.matrixWorld);
    const rightOriginWorld = rightAnchorLocal.clone().applyMatrix4(drone.matrixWorld);
    const bottomOriginWorld = bottomAnchorLocal.clone().applyMatrix4(drone.matrixWorld);

    const leftDirWorld = new THREE.Vector3(-1, 0, 0).applyQuaternion(drone.quaternion).normalize();
    const rightDirWorld = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.quaternion).normalize();
    const bottomDirWorld = new THREE.Vector3(0, -1, 0).applyQuaternion(drone.quaternion).normalize();

    // Perform sub-millisecond CPU ray-marching against dsm_raw
    const leftDist = castSensor(
      leftOriginWorld,
      leftDirWorld,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      maxSensorRange,
      0.4,
      samplingOptions
    );

    const rightDist = castSensor(
      rightOriginWorld,
      rightDirWorld,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      maxSensorRange,
      0.4,
      samplingOptions
    );

    const bottomDist = castSensor(
      bottomOriginWorld,
      bottomDirWorld,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      maxSensorRange,
      0.4,
      samplingOptions
    );

    // 2. Scaled score out of 10 (dynamically scaled according to input image / map size)
    const leftScore10 = Math.max(0, Math.min(10.0, (leftDist / maxSensorRange) * 10.0));
    const rightScore10 = Math.max(0, Math.min(10.0, (rightDist / maxSensorRange) * 10.0));
    const bottomScore10 = Math.max(0, Math.min(10.0, (bottomDist / maxSensorRange) * 10.0));

    // World hit points
    const leftHitWorld = leftOriginWorld.clone().add(leftDirWorld.clone().multiplyScalar(leftDist));
    const rightHitWorld = rightOriginWorld.clone().add(rightDirWorld.clone().multiplyScalar(rightDist));
    const bottomHitWorld = bottomOriginWorld.clone().add(bottomDirWorld.clone().multiplyScalar(bottomDist));

    // 3. Convert hit points to LOCAL drone coordinates:
    // This rigidly pins the line start to the drone emitter and avoids double-transform waving!
    const leftHitLocal = drone.worldToLocal(leftHitWorld.clone());
    const rightHitLocal = drone.worldToLocal(rightHitWorld.clone());
    const bottomHitLocal = drone.worldToLocal(bottomHitWorld.clone());

    if (showVisuals) {
      // --- Update Left Line & Dot ---
      const leftPos = leftGeo.attributes.position as THREE.BufferAttribute;
      const leftArr = leftPos.array as Float32Array;
      leftArr[0] = leftAnchorLocal.x;
      leftArr[1] = leftAnchorLocal.y;
      leftArr[2] = leftAnchorLocal.z;
      leftArr[3] = leftHitLocal.x;
      leftArr[4] = leftHitLocal.y;
      leftArr[5] = leftHitLocal.z;
      leftPos.needsUpdate = true;

      const leftColor = getSensorColor(leftDist, maxSensorRange);
      (leftLine.material as THREE.LineBasicMaterial).color.copy(leftColor);
      leftLine.visible = !isBehindCameraNearPlane(leftOriginWorld) && !isBehindCameraNearPlane(leftHitWorld);

      if (leftDotRef.current && leftDotMatRef.current) {
        leftDotRef.current.position.copy(leftHitLocal);
        leftDotRef.current.visible = leftDist < maxSensorRange && !isBehindCameraNearPlane(leftHitWorld);
        leftDotMatRef.current.color.copy(leftColor);
      }

      // --- Update Right Line & Dot ---
      const rightPos = rightGeo.attributes.position as THREE.BufferAttribute;
      const rightArr = rightPos.array as Float32Array;
      rightArr[0] = rightAnchorLocal.x;
      rightArr[1] = rightAnchorLocal.y;
      rightArr[2] = rightAnchorLocal.z;
      rightArr[3] = rightHitLocal.x;
      rightArr[4] = rightHitLocal.y;
      rightArr[5] = rightHitLocal.z;
      rightPos.needsUpdate = true;

      const rightColor = getSensorColor(rightDist, maxSensorRange);
      (rightLine.material as THREE.LineBasicMaterial).color.copy(rightColor);
      rightLine.visible = !isBehindCameraNearPlane(rightOriginWorld) && !isBehindCameraNearPlane(rightHitWorld);

      if (rightDotRef.current && rightDotMatRef.current) {
        rightDotRef.current.position.copy(rightHitLocal);
        rightDotRef.current.visible = rightDist < maxSensorRange && !isBehindCameraNearPlane(rightHitWorld);
        rightDotMatRef.current.color.copy(rightColor);
      }

      // --- Update Bottom Line & Dot ---
      const bottomPos = bottomGeo.attributes.position as THREE.BufferAttribute;
      const bottomArr = bottomPos.array as Float32Array;
      bottomArr[0] = bottomAnchorLocal.x;
      bottomArr[1] = bottomAnchorLocal.y;
      bottomArr[2] = bottomAnchorLocal.z;
      bottomArr[3] = bottomHitLocal.x;
      bottomArr[4] = bottomHitLocal.y;
      bottomArr[5] = bottomHitLocal.z;
      bottomPos.needsUpdate = true;

      const bottomColor = getSensorColor(bottomDist, maxSensorRange);
      (bottomLine.material as THREE.LineBasicMaterial).color.copy(bottomColor);
      bottomLine.visible = !isBehindCameraNearPlane(bottomOriginWorld) && !isBehindCameraNearPlane(bottomHitWorld);

      if (bottomDotRef.current && bottomDotMatRef.current) {
        bottomDotRef.current.position.copy(bottomHitLocal);
        bottomDotRef.current.visible = bottomDist < maxSensorRange && !isBehindCameraNearPlane(bottomHitWorld);
        bottomDotMatRef.current.color.copy(bottomColor);
      }

      // 4. Update 3D Floating Labels along each of the 3 lines
      // Position each label along its laser line (clamped close to drone so it remains readable)
      const leftLabelPos = leftAnchorLocal.clone().lerp(leftHitLocal, Math.min(0.65, 2.0 / Math.max(0.1, leftDist)));
      leftLabelPos.y += 0.12;
      if (leftLabelGroupRef.current) {
        leftLabelGroupRef.current.position.copy(leftLabelPos);
        leftLabelGroupRef.current.visible = leftLine.visible;
      }

      const rightLabelPos = rightAnchorLocal.clone().lerp(rightHitLocal, Math.min(0.65, 2.0 / Math.max(0.1, rightDist)));
      rightLabelPos.y += 0.12;
      if (rightLabelGroupRef.current) {
        rightLabelGroupRef.current.position.copy(rightLabelPos);
        rightLabelGroupRef.current.visible = rightLine.visible;
      }

      const bottomLabelPos = bottomAnchorLocal.clone().lerp(bottomHitLocal, Math.min(0.65, 2.0 / Math.max(0.1, bottomDist)));
      bottomLabelPos.x += 0.12;
      if (bottomLabelGroupRef.current) {
        bottomLabelGroupRef.current.position.copy(bottomLabelPos);
        bottomLabelGroupRef.current.visible = bottomLine.visible;
      }

      // 5. Update DOM text & border colors directly (pure 60 FPS, 0 React re-renders)
      const leftHex = getSensorColorHex(leftDist, maxSensorRange);
      if (leftScoreRef.current) leftScoreRef.current.textContent = leftScore10.toFixed(1);
      if (leftDistRef.current) leftDistRef.current.textContent = `(${leftDist.toFixed(1)}m)`;
      if (leftBoxRef.current) {
        leftBoxRef.current.style.borderColor = leftHex;
        leftBoxRef.current.style.color = leftHex;
        leftBoxRef.current.style.boxShadow = `0 0 6px ${leftHex}55`;
      }

      const rightHex = getSensorColorHex(rightDist, maxSensorRange);
      if (rightScoreRef.current) rightScoreRef.current.textContent = rightScore10.toFixed(1);
      if (rightDistRef.current) rightDistRef.current.textContent = `(${rightDist.toFixed(1)}m)`;
      if (rightBoxRef.current) {
        rightBoxRef.current.style.borderColor = rightHex;
        rightBoxRef.current.style.color = rightHex;
        rightBoxRef.current.style.boxShadow = `0 0 6px ${rightHex}55`;
      }

      const bottomHex = getSensorColorHex(bottomDist, maxSensorRange);
      if (bottomScoreRef.current) bottomScoreRef.current.textContent = bottomScore10.toFixed(1);
      if (bottomDistRef.current) bottomDistRef.current.textContent = `(${bottomDist.toFixed(1)}m)`;
      if (bottomBoxRef.current) {
        bottomBoxRef.current.style.borderColor = bottomHex;
        bottomBoxRef.current.style.color = bottomHex;
        bottomBoxRef.current.style.boxShadow = `0 0 6px ${bottomHex}55`;
      }
    } else {
      leftLine.visible = false;
      rightLine.visible = false;
      bottomLine.visible = false;
      if (leftDotRef.current) leftDotRef.current.visible = false;
      if (rightDotRef.current) rightDotRef.current.visible = false;
      if (bottomDotRef.current) bottomDotRef.current.visible = false;
      if (leftLabelGroupRef.current) leftLabelGroupRef.current.visible = false;
      if (rightLabelGroupRef.current) rightLabelGroupRef.current.visible = false;
      if (bottomLabelGroupRef.current) bottomLabelGroupRef.current.visible = false;
    }

    // Report readings and scaled /10 scores to telemetry callback
    if (onReadingsUpdate) {
      onReadingsUpdate({
        left: leftDist,
        right: rightDist,
        bottom: bottomDist,
        leftScore10,
        rightScore10,
        bottomScore10,
        maxSensorRange,
      });
    }
  });

  return (
    <group visible={showVisuals}>
      {/* Left Sensor Laser Line */}
      <primitive object={leftLine} />
      <mesh ref={leftDotRef} visible={false}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial ref={leftDotMatRef} color="#10b981" />
      </mesh>
      <group ref={leftLabelGroupRef}>
        <Html center distanceFactor={8.4}>
          <div
            ref={leftBoxRef}
            className="px-1.5 py-0.5 font-mono text-[9px] font-bold border rounded shadow-md backdrop-blur-sm pointer-events-none select-none flex items-center gap-0.5"
            style={{
              backgroundColor: 'rgba(7, 10, 15, 0.90)',
              borderColor: '#10b981',
              color: '#10b981',
              boxShadow: '0 0 6px rgba(16, 185, 129, 0.35)',
              whiteSpace: 'nowrap',
            }}
          >
            <span style={{ opacity: 0.7 }}>L:</span>
            <span ref={leftScoreRef}>10.0</span>
            <span style={{ fontSize: '7.5px', opacity: 0.6 }}>/10</span>
            <span ref={leftDistRef} style={{ fontSize: '7.5px', opacity: 0.6, marginLeft: '1px' }}>(20.0m)</span>
          </div>
        </Html>
      </group>

      {/* Right Sensor Laser Line */}
      <primitive object={rightLine} />
      <mesh ref={rightDotRef} visible={false}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial ref={rightDotMatRef} color="#10b981" />
      </mesh>
      <group ref={rightLabelGroupRef}>
        <Html center distanceFactor={8.4}>
          <div
            ref={rightBoxRef}
            className="px-1.5 py-0.5 font-mono text-[9px] font-bold border rounded shadow-md backdrop-blur-sm pointer-events-none select-none flex items-center gap-0.5"
            style={{
              backgroundColor: 'rgba(7, 10, 15, 0.90)',
              borderColor: '#10b981',
              color: '#10b981',
              boxShadow: '0 0 6px rgba(16, 185, 129, 0.35)',
              whiteSpace: 'nowrap',
            }}
          >
            <span style={{ opacity: 0.7 }}>R:</span>
            <span ref={rightScoreRef}>10.0</span>
            <span style={{ fontSize: '7.5px', opacity: 0.6 }}>/10</span>
            <span ref={rightDistRef} style={{ fontSize: '7.5px', opacity: 0.6, marginLeft: '1px' }}>(20.0m)</span>
          </div>
        </Html>
      </group>

      {/* Bottom Sensor Laser Line */}
      <primitive object={bottomLine} />
      <mesh ref={bottomDotRef} visible={false}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial ref={bottomDotMatRef} color="#10b981" />
      </mesh>
      <group ref={bottomLabelGroupRef}>
        <Html center distanceFactor={8.4}>
          <div
            ref={bottomBoxRef}
            className="px-1.5 py-0.5 font-mono text-[9px] font-bold border rounded shadow-md backdrop-blur-sm pointer-events-none select-none flex items-center gap-0.5"
            style={{
              backgroundColor: 'rgba(7, 10, 15, 0.90)',
              borderColor: '#10b981',
              color: '#10b981',
              boxShadow: '0 0 6px rgba(16, 185, 129, 0.35)',
              whiteSpace: 'nowrap',
            }}
          >
            <span style={{ opacity: 0.7 }}>AGL:</span>
            <span ref={bottomScoreRef}>10.0</span>
            <span style={{ fontSize: '7.5px', opacity: 0.6 }}>/10</span>
            <span ref={bottomDistRef} style={{ fontSize: '7.5px', opacity: 0.6, marginLeft: '1px' }}>(20.0m)</span>
          </div>
        </Html>
      </group>
    </group>
  );
}
