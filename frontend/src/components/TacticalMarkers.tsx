import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { getTerrainElevationAt } from './drone/dsmSampling.ts';

export interface TacticalMarkersProps {
  observerPos?: [number, number, number] | null; // [col, row, agl_m]
  targetPos?: [number, number, number] | null;   // [col, row, agl_m]
  sightlineClear?: boolean | null;
  strikePos?: [number, number, number] | null;   // [col, row, radius_m]
  strikeSimulation?: {
    active: boolean;
    col: number;
    row: number;
    radiusM: number;
    depthM: number;
    ejectaM: number;
    timestamp: number;
  } | null;
  noeWaypoints?: Array<{ x: number; y: number; z: number; agl_m: number; exposed?: boolean }> | null;
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
    pixel_size: number;
  };
  dsmRaw: number[][];
  verticalScale: number;
  waterLevel?: number;
  renderMode?: 'voxel' | 'smooth';
  voxelResolution?: number;
}

interface DebrisParticle {
  vx: number;
  vy: number;
  vz: number;
  rotSpeed: number;
  size: number;
  color: string;
}

export default function TacticalMarkers({
  observerPos,
  targetPos,
  sightlineClear,
  strikePos,
  strikeSimulation,
  noeWaypoints,
  meshStats,
  dsmRaw,
  verticalScale,
  waterLevel,
  renderMode = 'smooth',
  voxelResolution = 64,
}: TacticalMarkersProps) {
  const pulseRef = useRef<THREE.Mesh>(null);
  const targetPulseRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const shockwaveRef = useRef<THREE.Mesh>(null);
  const fireballRef = useRef<THREE.Mesh>(null);
  const debrisGroupRef = useRef<THREE.Group>(null);

  // Pre-generate pseudo-random velocities for 28 lively blast debris particles
  const debrisParticles = useMemo<DebrisParticle[]>(() => {
    const list: DebrisParticle[] = [];
    const colors = ['#1e293b', '#334155', '#475569', '#ea580c', '#c2410c', '#78716c', '#0f172a'];
    for (let i = 0; i < 28; i++) {
      const angle = (i / 28) * Math.PI * 2 + (Math.sin(i * 3.7) * 0.4);
      const speedH = 3.5 + Math.abs(Math.sin(i * 1.9)) * 6.5;
      const vy = 5.0 + Math.abs(Math.cos(i * 2.3)) * 8.5;
      const vx = Math.cos(angle) * speedH;
      const vz = Math.sin(angle) * speedH;
      const size = 0.12 + Math.abs(Math.sin(i * 4.1)) * 0.22;
      const color = colors[i % colors.length];
      list.push({ vx, vy, vz, rotSpeed: 4.0 + Math.sin(i) * 5.0, size, color });
    }
    return list;
  }, []);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (pulseRef.current) {
      const s = 1.0 + 0.22 * Math.sin(t * 4);
      pulseRef.current.scale.set(s, s, s);
    }
    if (targetPulseRef.current) {
      const s = 1.0 + 0.25 * Math.sin(t * 5);
      targetPulseRef.current.scale.set(s, s, s);
    }
    if (ringRef.current) {
      const s = 1.0 + 0.08 * Math.sin(t * 3);
      ringRef.current.scale.set(s, s, s);
    }

    // Animate lively detonation blast if simulation is active
    if (strikeSimulation?.active && strikeSimulation.timestamp) {
      const elapsed = (Date.now() - strikeSimulation.timestamp) / 1000;
      const animDuration = 3.5;

      // 1. Expanding and dissipating fireball
      if (fireballRef.current) {
        if (elapsed < 1.2) {
          fireballRef.current.visible = true;
          const progress = elapsed / 1.2;
          const maxScale = Math.max(1.2, (strikeSimulation.radiusM || 12) * 0.08);
          const currentScale = maxScale * Math.sin(progress * Math.PI * 0.5);
          fireballRef.current.scale.set(currentScale, currentScale * 1.3, currentScale);
          const mat = fireballRef.current.material as THREE.MeshBasicMaterial;
          if (mat) {
            mat.opacity = Math.max(0, 1.0 - progress * progress);
          }
        } else {
          fireballRef.current.visible = false;
        }
      }

      // 2. Expanding shockwave ring
      if (shockwaveRef.current) {
        if (elapsed < 1.8) {
          shockwaveRef.current.visible = true;
          const progress = elapsed / 1.8;
          const maxRadius = Math.max(2.0, (strikeSimulation.radiusM || 12) * 0.25);
          const r = maxRadius * Math.pow(progress, 0.6);
          shockwaveRef.current.scale.set(r, r, 1);
          const mat = shockwaveRef.current.material as THREE.MeshBasicMaterial;
          if (mat) {
            mat.opacity = Math.max(0, 0.9 * (1.0 - progress));
          }
        } else {
          shockwaveRef.current.visible = false;
        }
      }

      // 3. Lively flying debris voxel / rubble particles with ballistic arcs
      if (debrisGroupRef.current) {
        if (elapsed < animDuration) {
          debrisGroupRef.current.visible = true;
          const children = debrisGroupRef.current.children;
          debrisParticles.forEach((p, idx) => {
            const mesh = children[idx] as THREE.Mesh;
            if (mesh) {
              const dt = Math.min(elapsed, 2.2);
              const x = p.vx * dt * 0.35;
              const z = p.vz * dt * 0.35;
              const y = Math.max(0.05, (p.vy * dt - 0.5 * 12.0 * dt * dt) * 0.3);
              mesh.position.set(x, y, z);
              mesh.rotation.x = elapsed * p.rotSpeed;
              mesh.rotation.y = elapsed * p.rotSpeed * 0.8;
              const mat = mesh.material as THREE.MeshStandardMaterial;
              if (mat && elapsed > 2.0) {
                mat.opacity = Math.max(0.0, 1.0 - (elapsed - 2.0) / (animDuration - 2.0));
              }
            }
          });
        } else {
          debrisGroupRef.current.visible = false;
        }
      }
    } else {
      // INACTIVE / RESET: Ensure all bomb blast particles and VFX are completely hidden
      if (fireballRef.current) fireballRef.current.visible = false;
      if (shockwaveRef.current) shockwaveRef.current.visible = false;
      if (debrisGroupRef.current) debrisGroupRef.current.visible = false;
    }
  });

  // Instantly clean up and reset blast VFX when strike simulation is reset or inactive
  useEffect(() => {
    if (!strikeSimulation?.active) {
      if (fireballRef.current) {
        fireballRef.current.visible = false;
        fireballRef.current.scale.set(0, 0, 0);
      }
      if (shockwaveRef.current) {
        shockwaveRef.current.visible = false;
        shockwaveRef.current.scale.set(0, 0, 1);
      }
      if (debrisGroupRef.current) {
        debrisGroupRef.current.visible = false;
        debrisGroupRef.current.children.forEach((child) => {
          child.position.set(0, 0, 0);
          child.rotation.set(0, 0, 0);
          const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial;
          if (mat) {
            mat.opacity = 1.0;
          }
        });
      }
    }
  }, [strikeSimulation?.active]);

  // 1. Observer Mast & Beacon 3D coordinates
  const obsData = useMemo(() => {
    if (!observerPos || !dsmRaw || dsmRaw.length === 0) return null;
    const [c, r, agl] = observerPos;
    const worldX = (c - meshStats.width / 2) * 0.1;
    const worldZ = (r - meshStats.height / 2) * 0.1;
    const groundY = getTerrainElevationAt(
      worldX,
      worldZ,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      { renderMode, voxelResolution }
    );
    const mastH = Math.max(0.6, agl * 0.25);
    return {
      worldX,
      worldZ,
      groundY,
      mastH,
      eyeY: groundY + mastH,
    };
  }, [observerPos, dsmRaw, meshStats, verticalScale, waterLevel, renderMode, voxelResolution]);

  // 2. Point-to-Point Target 3D coordinates
  const targetData = useMemo(() => {
    if (!targetPos || !dsmRaw || dsmRaw.length === 0) return null;
    const [c, r, agl] = targetPos;
    const worldX = (c - meshStats.width / 2) * 0.1;
    const worldZ = (r - meshStats.height / 2) * 0.1;
    const groundY = getTerrainElevationAt(
      worldX,
      worldZ,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      { renderMode, voxelResolution }
    );
    const mastH = Math.max(0.5, agl * 0.25);
    return {
      worldX,
      worldZ,
      groundY,
      mastH,
      eyeY: groundY + mastH,
    };
  }, [targetPos, dsmRaw, meshStats, verticalScale, waterLevel, renderMode, voxelResolution]);

  // 3. Line of Sight 3D Laser Ray geometry
  const losLineGeometry = useMemo(() => {
    if (!obsData || !targetData) return null;
    const points = [
      new THREE.Vector3(obsData.worldX, obsData.eyeY, obsData.worldZ),
      new THREE.Vector3(targetData.worldX, targetData.eyeY, targetData.worldZ),
    ];
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [obsData, targetData]);

  // 4. BDA Bomb Strike Crater & Blast Ring 3D coordinates
  const strikeData = useMemo(() => {
    const activePos = strikeSimulation?.active
      ? [strikeSimulation.col, strikeSimulation.row, strikeSimulation.radiusM]
      : strikePos;

    if (!activePos || !dsmRaw || dsmRaw.length === 0) return null;
    const [c, r, radiusM] = activePos;
    const worldX = (c - meshStats.width / 2) * 0.1;
    const worldZ = (r - meshStats.height / 2) * 0.1;
    const groundY = getTerrainElevationAt(
      worldX,
      worldZ,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      { renderMode, voxelResolution }
    );
    const ringRadius = Math.max(0.5, radiusM * 0.1);
    return {
      worldX,
      worldZ,
      groundY: groundY + 0.05,
      ringRadius,
    };
  }, [strikePos, strikeSimulation, dsmRaw, meshStats, verticalScale, waterLevel, renderMode, voxelResolution]);

  // 5. NOE Flight Corridor 3D Waypoint Path & Geometry (§4.3)
  const noeRouteData = useMemo(() => {
    if (!noeWaypoints || noeWaypoints.length < 2 || !dsmRaw || dsmRaw.length === 0) {
      return null;
    }
    const cols = meshStats.width;
    const rows = meshStats.height;
    const pxSize = meshStats.pixel_size || 1.0;

    const points3D: THREE.Vector3[] = [];
    const keyMarkers: Array<{ pos: [number, number, number]; label?: string; isEnd?: boolean }> = [];

    noeWaypoints.forEach((wp, idx) => {
      const c = Math.max(0, Math.min(cols - 1, wp.x / pxSize));
      const r = Math.max(0, Math.min(rows - 1, wp.z / pxSize));
      const worldX = (c - cols / 2) * 0.1;
      const worldZ = (r - rows / 2) * 0.1;
      const groundElev = getTerrainElevationAt(
        worldX,
        worldZ,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel,
        { renderMode, voxelResolution }
      );
      const flyY = groundElev + Math.max(0.6, (wp.agl_m || 3.0) * 0.15);
      points3D.push(new THREE.Vector3(worldX, flyY, worldZ));

      if (idx === 0) {
        keyMarkers.push({ pos: [worldX, flyY, worldZ], label: 'INGRESS', isEnd: true });
      } else if (idx === noeWaypoints.length - 1) {
        keyMarkers.push({ pos: [worldX, flyY, worldZ], label: 'EGRESS', isEnd: true });
      } else if (idx % 6 === 0) {
        keyMarkers.push({ pos: [worldX, flyY, worldZ], label: `WP${idx}` });
      }
    });

    const pathGeometry = new THREE.BufferGeometry().setFromPoints(points3D);
    return {
      pathGeometry,
      keyMarkers,
    };
  }, [noeWaypoints, dsmRaw, meshStats, verticalScale, waterLevel, renderMode, voxelResolution]);

  return (
    <group name="tactical-markers">
      {/* 1. Observer Mast & Beacon */}
      {obsData && (
        <group position={[obsData.worldX, 0, obsData.worldZ]}>
          {/* Mast Pole */}
          <mesh position={[0, obsData.groundY + obsData.mastH / 2, 0]}>
            <cylinderGeometry args={[0.08, 0.12, obsData.mastH, 12]} />
            <meshStandardMaterial color="#0284c7" metalness={0.8} roughness={0.2} />
          </mesh>

          {/* Glowing Observer Beacon */}
          <mesh ref={pulseRef} position={[0, obsData.eyeY, 0]}>
            <sphereGeometry args={[0.3, 16, 16]} />
            <meshBasicMaterial color="#38bdf8" />
          </mesh>

          {/* Pulsing Light Halo */}
          <pointLight position={[0, obsData.eyeY, 0]} color="#38bdf8" intensity={2.5} distance={15} />

          {/* Observer Radar Horizon Ring */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, obsData.eyeY, 0]}>
            <ringGeometry args={[0.5, 0.65, 32]} />
            <meshBasicMaterial color="#38bdf8" transparent opacity={0.6} side={THREE.DoubleSide} />
          </mesh>
        </group>
      )}

      {/* 2. Target Marker */}
      {targetData && (
        <group position={[targetData.worldX, 0, targetData.worldZ]}>
          <mesh position={[0, targetData.groundY + targetData.mastH / 2, 0]}>
            <cylinderGeometry args={[0.06, 0.1, targetData.mastH, 12]} />
            <meshStandardMaterial color="#d97706" metalness={0.7} roughness={0.3} />
          </mesh>
          <mesh ref={targetPulseRef} position={[0, targetData.eyeY, 0]}>
            <octahedronGeometry args={[0.38, 0]} />
            <meshBasicMaterial color="#ef4444" />
          </mesh>
          <pointLight position={[0, targetData.eyeY, 0]} color="#ef4444" intensity={2.5} distance={12} />
        </group>
      )}

      {/* 3. 3D Line-of-Sight Laser Ray */}
      {losLineGeometry && (
        <line>
          <primitive object={losLineGeometry} />
          <lineBasicMaterial
            color={sightlineClear ? '#10b981' : '#ef4444'}
            linewidth={3}
          />
        </line>
      )}

      {/* 4. 3D NOE Flight Corridor Waypoint Ribbon & Beacons (§4.3) */}
      {noeRouteData && (
        <group name="noe-corridor-ribbon">
          {/* Main Flight Path Glowing Ribbon */}
          <line>
            <primitive object={noeRouteData.pathGeometry} />
            <lineBasicMaterial color="#a855f7" linewidth={4} />
          </line>

          {/* Waypoint Nodes */}
          {noeRouteData.keyMarkers.map((wp, i) => (
            <group key={i} position={wp.pos}>
              <mesh>
                <octahedronGeometry args={[wp.isEnd ? 0.35 : 0.18, 0]} />
                <meshBasicMaterial color={wp.label === 'INGRESS' ? '#10b981' : wp.label === 'EGRESS' ? '#ec4899' : '#8b5cf6'} />
              </mesh>
              {wp.isEnd && (
                <pointLight
                  color={wp.label === 'INGRESS' ? '#10b981' : '#ec4899'}
                  intensity={1.8}
                  distance={8}
                />
              )}
            </group>
          ))}
        </group>
      )}

      {/* 5. BDA Bomb Strike Reticle, Blast Ring & Lively Detonation Simulation */}
      {strikeData && (
        <group position={[strikeData.worldX, strikeData.groundY, strikeData.worldZ]}>
          {/* Ground Zero Glowing Reticle */}
          <mesh position={[0, 0.05, 0]}>
            <octahedronGeometry args={[0.4, 0]} />
            <meshBasicMaterial color="#ef4444" />
          </mesh>
          <pointLight position={[0, 0.5, 0]} color="#ef4444" intensity={3.5} distance={20} />

          {/* Blast Crater Radius Ring */}
          <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
            <ringGeometry args={[strikeData.ringRadius * 0.92, strikeData.ringRadius, 48]} />
            <meshBasicMaterial color="#ef4444" transparent opacity={0.75} side={THREE.DoubleSide} />
          </mesh>

          {/* Secondary Ejecta Boundary Ring */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
            <ringGeometry args={[strikeData.ringRadius * 1.45, strikeData.ringRadius * 1.50, 48]} />
            <meshBasicMaterial color="#f97316" transparent opacity={0.45} side={THREE.DoubleSide} />
          </mesh>

          {/* Excavated Charred Crater Dish Mesh */}
          {strikeSimulation?.active && (
            <group position={[0, -0.05, 0]}>
              {/* Sunken crater basin */}
              <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <circleGeometry args={[strikeData.ringRadius * 0.92, 32]} />
                <meshStandardMaterial color="#09090b" roughness={0.95} metalness={0.1} />
              </mesh>
              {/* Raised rubble berm ring */}
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.08, 0]}>
                <ringGeometry args={[strikeData.ringRadius * 0.92, strikeData.ringRadius * 1.45, 32]} />
                <meshStandardMaterial color="#44403c" roughness={0.9} />
              </mesh>
            </group>
          )}

          {/* Lively Detonation Blast VFX (Fireball, Shockwave & Rubble Debris) */}
          {strikeSimulation?.active && (
            <>
              {/* Expanding Fireball Blast */}
              <mesh ref={fireballRef} visible={false} position={[0, 0.4, 0]}>
                <sphereGeometry args={[1, 24, 24]} />
                <meshBasicMaterial color="#ffedd5" transparent opacity={1.0} />
              </mesh>

              {/* Expanding Shockwave Ring */}
              <mesh ref={shockwaveRef} visible={false} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
                <ringGeometry args={[0.92, 1.0, 48]} />
                <meshBasicMaterial color="#fb923c" transparent opacity={0.9} side={THREE.DoubleSide} />
              </mesh>

              {/* Flying Debris Voxels / Rubble Particles */}
              <group ref={debrisGroupRef} visible={false}>
                {debrisParticles.map((p, idx) => (
                  <mesh key={idx}>
                    <boxGeometry args={[p.size, p.size, p.size]} />
                    <meshStandardMaterial color={p.color} roughness={0.8} />
                  </mesh>
                ))}
              </group>
            </>
          )}
        </group>
      )}
    </group>
  );
}
