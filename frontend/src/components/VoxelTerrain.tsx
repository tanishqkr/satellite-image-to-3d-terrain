import React, { useRef, useLayoutEffect, useMemo, useEffect, useState } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Sparkles, Eye } from 'lucide-react';
import { voxelize, VoxelGrid } from '../lib/voxelize.ts';
import { computeCameraOcclusion } from '../lib/rayMarch.ts';
import TacticalMarkers from './TacticalMarkers.tsx';

export interface VoxelTerrainProps {
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
  targetResolution?: number;
  bandCount?: number;
  gapRatio?: number;
  environmentTheme?: 'dark' | 'light';
  playEntranceAnimation?: boolean;
  replayTrigger?: number;
  cameraPose?: {
    pitch?: number;
    roll?: number;
    yaw?: number;
    altitude?: number;
  };
  showOcclusion?: boolean;
  onToggleOcclusion?: (show: boolean) => void;
  // Tactical interactive 3D markers & click callbacks
  observerPos?: [number, number, number] | null;
  targetPos?: [number, number, number] | null;
  sightlineClear?: boolean | null;
  strikePos?: [number, number, number] | null;
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
  onTerrainClick?: (pt: { col: number; row: number; isShift: boolean }) => void;
}

/**
 * Inner VoxelMesh component rendering the InstancedMesh and synchronized water plane.
 * Can be mounted inside any existing React Three Fiber <Canvas>.
 */
export function VoxelMesh({
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel = 0,
  targetResolution = 64,
  bandCount = 8,
  gapRatio = 0.90,
  playEntranceAnimation = true,
  replayTrigger = 0,
  showOcclusion = false,
  cameraPose,
  onTerrainClick,
}: VoxelTerrainProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const waterMeshRef = useRef<THREE.Mesh>(null);
  const waterMatRef = useRef<THREE.MeshStandardMaterial>(null);

  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;

  // In metric mode: 1 meter elevation corresponds to 0.1 Three.js world units.
  // In relative mode: normalized elevation [0, 1] scaled to 18.0 world units to match TerrainCanvas.
  const verticalFactor = isRelative ? 18.0 * verticalScale : verticalScale * 0.1;

  // World dimensions and radial bounds for center-out reveal wave
  const worldW = (meshStats?.width ?? 100) * 0.1;
  const worldD = (meshStats?.height ?? 100) * 0.1;
  const maxRadius = Math.hypot(worldW / 2, worldD / 2) * 1.05;
  const waveWidth = Math.max(6.0, maxRadius * 0.25);

  // Shader uniforms stored in stable ref across re-renders
  const uniformsRef = useRef({
    uProgress: { value: playEntranceAnimation ? 0.0 : 1.0 },
    uMaxRadius: { value: maxRadius },
    uWaveWidth: { value: waveWidth },
    uCenter: { value: new THREE.Vector2(0, 0) },
  });

  // Keep spatial parameters updated if meshStats dimensions change
  useEffect(() => {
    uniformsRef.current.uMaxRadius.value = maxRadius;
    uniformsRef.current.uWaveWidth.value = waveWidth;
  }, [maxRadius, waveWidth]);

  // Animation timing state
  const animTimeRef = useRef(playEntranceAnimation ? 0.0 : 999.0);
  const animDuration = 2.2; // Smooth 2.2s cinematic reveal
  const isAnimatingRef = useRef(playEntranceAnimation);
  const lastDsmRef = useRef<number[][] | null>(dsmRaw);
  const lastReplayTriggerRef = useRef<number>(replayTrigger);

  // Trigger animation ONLY on dataset switch or explicit replay request
  // Slider tweaks (verticalScale, targetResolution, waterLevel) do NOT re-trigger
  useEffect(() => {
    if (!playEntranceAnimation) {
      isAnimatingRef.current = false;
      uniformsRef.current.uProgress.value = 1.0;
      return;
    }

    const isNewDataset = !lastDsmRef.current ||
      lastDsmRef.current.length !== dsmRaw?.length ||
      lastDsmRef.current[0]?.length !== dsmRaw?.[0]?.length;
    const replayRequested = replayTrigger !== lastReplayTriggerRef.current;

    lastDsmRef.current = dsmRaw;
    lastReplayTriggerRef.current = replayTrigger;

    if (isNewDataset || replayRequested) {
      animTimeRef.current = 0.0;
      uniformsRef.current.uProgress.value = 0.0;
      isAnimatingRef.current = true;
    }
  }, [dsmRaw, replayTrigger, playEntranceAnimation]);

  // Frame loop driving smooth cubic ease-in-out radial wave
  useFrame((_, delta) => {
    if (!isAnimatingRef.current) return;

    const dt = Math.min(delta, 0.05);
    animTimeRef.current += dt;
    const t = Math.min(1.0, animTimeRef.current / animDuration);

    // Smooth cubic ease-in-out curve
    const eased = t < 0.5
      ? 4.0 * t * t * t
      : 1.0 - Math.pow(-2.0 * t + 2.0, 3.0) / 2.0;

    uniformsRef.current.uProgress.value = eased;

    if (waterMeshRef.current && waterMatRef.current) {
      waterMeshRef.current.scale.set(eased, eased, 1.0);
      waterMatRef.current.opacity = 0.55 * Math.min(1.0, eased * 1.5);
    }

    if (t >= 1.0) {
      uniformsRef.current.uProgress.value = 1.0;
      isAnimatingRef.current = false;
      if (waterMeshRef.current && waterMatRef.current) {
        waterMeshRef.current.scale.set(1.0, 1.0, 1.0);
        waterMatRef.current.opacity = 0.55;
      }
    }
  });

  // Compute geometric camera occlusion mask if showOcclusion is enabled (§3.2)
  const occlusionMask = useMemo(() => {
    if (!showOcclusion || !dsmRaw || dsmRaw.length === 0) return undefined;
    const pitch = cameraPose?.pitch ?? -45;
    const alt = cameraPose?.altitude ?? (worldW * 0.7);
    const radPitch = Math.abs(pitch * (Math.PI / 180));
    const distXZ = alt / Math.tan(Math.max(0.1, radPitch));
    const camX = -worldW * 0.4;
    const camZ = -worldD * 0.4 - distXZ * 0.2;
    const camY = Math.max(alt * 0.1, 5.0);

    return computeCameraOcclusion(dsmRaw, [camX, camY, camZ], meshStats, verticalScale);
  }, [showOcclusion, dsmRaw, cameraPose, meshStats, verticalScale, worldW, worldD]);

  // 1. Compute downsampled voxel grid via pure pooling and quantization
  const voxelGrid: VoxelGrid = useMemo(() => {
    return voxelize(dsmRaw, {
      targetResolution,
      bandCount,
      gapRatio,
      elevationMin: meshStats.elevation_min,
      elevationMax: meshStats.elevation_max,
      worldWidth: worldW,
      worldDepth: worldD,
      occlusionMask,
    });
  }, [
    dsmRaw,
    targetResolution,
    bandCount,
    gapRatio,
    meshStats.elevation_min,
    meshStats.elevation_max,
    worldW,
    worldD,
    occlusionMask,
  ]);

  const blockCount = voxelGrid.blocks.length;

  // Custom Shader Materials injected with radial reveal wave and shadow clipping
  const { voxelMaterial, depthMaterial } = useMemo(() => {
    const uniforms = uniformsRef.current;

    const mat = new THREE.MeshStandardMaterial({
      roughness: 0.45,
      metalness: 0.08,
      side: THREE.FrontSide,
    });
    mat.customProgramCacheKey = () => 'VoxelEntranceMaterial_v1';
    mat.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);

      shader.vertexShader = shader.vertexShader.replace(
        '#include <common>',
        `#include <common>
uniform float uProgress;
uniform float uMaxRadius;
uniform float uWaveWidth;
uniform vec2 uCenter;
varying float vReveal;
varying float vWaveFactor;`
      );

      shader.vertexShader = shader.vertexShader.replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
#ifdef USE_INSTANCING
if (uProgress < 1.0) {
  // instanceMatrix[3].xz is column 3 of the 4x4 instanceMatrix, representing (posX, posY, posZ)
  vec2 colPos = vec2(instanceMatrix[3][0], instanceMatrix[3][2]);
  float dist = length(colPos - uCenter);
  float currentRadius = uProgress * (uMaxRadius + uWaveWidth);

  if (dist > currentRadius) {
    vReveal = 0.0;
    vWaveFactor = 0.0;
    // Collapse to base ground plane (y = -0.5 in unit box geometry maps to y = 0 at base)
    transformed = vec3(0.0, -0.5, 0.0);
  } else {
    float waveOffset = currentRadius - dist;
    float revealRatio = clamp(waveOffset / uWaveWidth, 0.0, 1.0);
    // Smooth individual voxel emergence
    float scaleY = smoothstep(0.0, 1.0, revealRatio);
    float scaleXZ = 0.3 + 0.7 * scaleY;

    // Anchor base to y = 0 while height scales up smoothly
    transformed.y = (transformed.y + 0.5) * scaleY - 0.5;
    transformed.x *= scaleXZ;
    transformed.z *= scaleXZ;

    vReveal = scaleY;
    // Subtle energetic wavefront crest
    vWaveFactor = sin(scaleY * 3.14159265);
  }
} else {
  vReveal = 1.0;
  vWaveFactor = 0.0;
}
#else
vReveal = 1.0;
vWaveFactor = 0.0;
#endif`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <common>',
        `#include <common>
uniform float uProgress;
varying float vReveal;
varying float vWaveFactor;`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#include <color_fragment>
if (uProgress < 1.0) {
  if (vReveal < 0.005) {
    discard;
  }
  if (vWaveFactor > 0.01) {
    // Holographic cyan wavefront highlight
    diffuseColor.rgb += vec3(0.18, 0.65, 1.0) * (vWaveFactor * 0.40);
  }
}`
      );
    };

    const depth = new THREE.MeshDepthMaterial({
      depthPacking: THREE.RGBADepthPacking,
    });
    depth.customProgramCacheKey = () => 'VoxelEntranceDepthMaterial_v1';
    depth.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, uniforms);

      shader.vertexShader = shader.vertexShader.replace(
        '#include <common>',
        `#include <common>
uniform float uProgress;
uniform float uMaxRadius;
uniform float uWaveWidth;
uniform vec2 uCenter;
varying float vReveal;`
      );

      shader.vertexShader = shader.vertexShader.replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
#ifdef USE_INSTANCING
if (uProgress < 1.0) {
  vec2 colPos = vec2(instanceMatrix[3][0], instanceMatrix[3][2]);
  float dist = length(colPos - uCenter);
  float currentRadius = uProgress * (uMaxRadius + uWaveWidth);

  if (dist > currentRadius) {
    vReveal = 0.0;
    transformed = vec3(0.0, -0.5, 0.0);
  } else {
    float waveOffset = currentRadius - dist;
    float revealRatio = clamp(waveOffset / uWaveWidth, 0.0, 1.0);
    float scaleY = smoothstep(0.0, 1.0, revealRatio);
    float scaleXZ = 0.3 + 0.7 * scaleY;

    transformed.y = (transformed.y + 0.5) * scaleY - 0.5;
    transformed.x *= scaleXZ;
    transformed.z *= scaleXZ;

    vReveal = scaleY;
  }
} else {
  vReveal = 1.0;
}
#else
vReveal = 1.0;
#endif`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <common>',
        `#include <common>
uniform float uProgress;
varying float vReveal;`
      );

      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <clipping_planes_fragment>',
        `#include <clipping_planes_fragment>
if (uProgress < 1.0 && vReveal < 0.005) {
  discard;
}`
      );
    };

    return { voxelMaterial: mat, depthMaterial: depth };
  }, []);

  // Dispose materials on unmount
  useEffect(() => {
    return () => {
      voxelMaterial.dispose();
      depthMaterial.dispose();
    };
  }, [voxelMaterial, depthMaterial]);

  // Bind customDepthMaterial for synchronized shadow map rendering
  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.customDepthMaterial = depthMaterial;
    }
  }, [depthMaterial]);

  // 2. Populate InstancedMesh transform matrices and per-instance colors
  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || blockCount === 0) return;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < blockCount; i++) {
      const block = voxelGrid.blocks[i];
      const normH = block.normalizedHeight;
      const rawH = block.rawHeight;

      // Compute physical block height in Three.js world units
      const hWorld = isRelative
        ? Math.max(0.12, normH * 18.0 * verticalScale)
        : Math.max(0.12, (rawH - meshStats.elevation_min) * verticalScale * 0.1);

      // Vertical anchoring (§3.4): instance position.y is height / 2 so the base rests at y = 0
      dummy.position.set(block.posX, hWorld / 2, block.posZ);
      dummy.scale.set(voxelGrid.cellWidth, hWorld, voxelGrid.cellDepth);
      dummy.updateMatrix();

      mesh.setMatrixAt(i, dummy.matrix);

      // Apply discrete Turbo band color
      color.set(block.colorHex);
      mesh.setColorAt(i, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  }, [
    voxelGrid,
    blockCount,
    isRelative,
    verticalScale,
    meshStats.elevation_min,
  ]);

  const handlePointerDown = (e: any) => {
    if (!onTerrainClick) return;
    e.stopPropagation();
    const pt = e.point;
    const col = Math.max(0, Math.min(meshStats.width - 1, Math.round(pt.x / 0.1 + meshStats.width / 2)));
    const row = Math.max(0, Math.min(meshStats.height - 1, Math.round(pt.z / 0.1 + meshStats.height / 2)));
    onTerrainClick({ col, row, isShift: !!(e.nativeEvent?.shiftKey || e.nativeEvent?.altKey) });
  };

  if (blockCount === 0) return null;

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, blockCount]}
        castShadow
        receiveShadow
        material={voxelMaterial}
        onPointerDown={handlePointerDown}
      >
        <boxGeometry args={[1, 1, 1]} />
      </instancedMesh>

      {/* Synchronized Water plane for flood simulation */}
      {waterLevel > meshStats.elevation_min && (
        <mesh
          ref={waterMeshRef}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[
            0,
            (waterLevel - meshStats.elevation_min) * verticalFactor + 0.05,
            0,
          ]}
        >
          <planeGeometry args={[meshStats.width * 0.12, meshStats.height * 0.12]} />
          <meshStandardMaterial
            ref={waterMatRef}
            color="#0077be"
            transparent
            opacity={0.55}
            depthWrite={false}
            side={THREE.DoubleSide}
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      )}
    </>
  );
}

function VoxelCameraController() {
  const { camera, gl } = useThree();
  const keys = useRef<Set<string>>(new Set());
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const isLocked = useRef(false);
  const isDragging = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });

  useEffect(() => {
    camera.position.set(0, 30, 40);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    euler.current.setFromQuaternion(camera.quaternion, 'YXZ');
    euler.current.z = 0;

    const onKeyDown = (e: KeyboardEvent) => keys.current.add(e.key.toLowerCase());
    const onKeyUp = (e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase());

    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 0) {
        isDragging.current = true;
        previousMousePosition.current = { x: e.clientX, y: e.clientY };
      }
    };

    const onMouseUp = () => {
      isDragging.current = false;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (isLocked.current) {
        euler.current.y -= e.movementX * 0.002;
        euler.current.x -= e.movementY * 0.002;
        euler.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, euler.current.x));
        euler.current.z = 0;
        camera.quaternion.setFromEuler(euler.current);
      } else if (isDragging.current) {
        const deltaX = e.clientX - previousMousePosition.current.x;
        const deltaY = e.clientY - previousMousePosition.current.y;
        previousMousePosition.current = { x: e.clientX, y: e.clientY };

        euler.current.y -= deltaX * 0.003;
        euler.current.x -= deltaY * 0.003;
        euler.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, euler.current.x));
        euler.current.z = 0;
        camera.quaternion.setFromEuler(euler.current);
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const dir = new THREE.Vector3();
      camera.getWorldDirection(dir);
      const zoomStep = Math.max(1.0, Math.min(10.0, camera.position.length() * 0.08));
      camera.position.addScaledVector(dir, (e.deltaY > 0 ? -1 : 1) * zoomStep);
      if (camera.position.y < 3.0) camera.position.y = 3.0;
    };

    const onDblClick = () => {
      gl.domElement.requestPointerLock();
    };

    const onPointerLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
      if (isLocked.current) {
        euler.current.setFromQuaternion(camera.quaternion, 'YXZ');
        euler.current.z = 0;
      }
    };

    const onBlur = () => {
      keys.current.clear();
      isDragging.current = false;
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('mouseup', onMouseUp);
    gl.domElement.addEventListener('mousedown', onMouseDown);
    gl.domElement.addEventListener('mousemove', onMouseMove);
    gl.domElement.addEventListener('wheel', onWheel, { passive: false });
    gl.domElement.addEventListener('dblclick', onDblClick);
    document.addEventListener('pointerlockchange', onPointerLockChange);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('mouseup', onMouseUp);
      gl.domElement.removeEventListener('mousedown', onMouseDown);
      gl.domElement.removeEventListener('mousemove', onMouseMove);
      gl.domElement.removeEventListener('wheel', onWheel);
      gl.domElement.removeEventListener('dblclick', onDblClick);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      window.removeEventListener('blur', onBlur);
    };
  }, [camera, gl]);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1);
    const speed = keys.current.has('shift') ? 60 : 20;

    const yaw = euler.current.y;
    const forward = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const move = new THREE.Vector3();

    if (keys.current.has('w') || keys.current.has('arrowup')) move.add(forward);
    if (keys.current.has('s') || keys.current.has('arrowdown')) move.sub(forward);
    if (keys.current.has('d') || keys.current.has('arrowright')) move.add(right);
    if (keys.current.has('a') || keys.current.has('arrowleft')) move.sub(right);

    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(speed * dt);
      camera.position.add(move);
    }

    let upDown = 0;
    if (keys.current.has('q') || keys.current.has(' ')) upDown += 1;
    if (keys.current.has('e')) upDown -= 1;

    if (upDown !== 0) {
      camera.position.y += upDown * speed * dt;
    }

    if (camera.position.y < 3.0) camera.position.y = 3.0;
  });

  return null;
}

/**
 * Full VoxelTerrain canvas with lighting, camera flight controls, and telemetry HUD.
 */
export default function VoxelTerrain(props: VoxelTerrainProps) {
  const [replayCount, setReplayCount] = useState(0);
  const [showOcclusion, setShowOcclusion] = useState(props.showOcclusion ?? false);
  const isLight = props.environmentTheme === 'light';
  // Light mode: Pure clean white environment with crisp black grid lines
  // Dark mode: Untouched dark void (#050608) with slate grid lines (#1e293b)
  const bgColor = isLight ? '#ffffff' : '#050608';
  const gridPrimary = isLight ? '#000000' : '#1e293b';
  const gridSecondary = isLight ? '#1f2937' : '#1e293b';

  const handleToggleOcclusion = () => {
    const next = !showOcclusion;
    setShowOcclusion(next);
    props.onToggleOcclusion?.(next);
  };

  return (
    <div className={`w-full h-full relative select-none ${isLight ? 'bg-white' : 'bg-[#050608]'}`}>
      <Canvas
        camera={{ fov: 60, near: 0.1, far: 2000 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: bgColor }}
      >
        <color attach="background" args={[bgColor]} />
        <fog attach="fog" args={[bgColor, 60, isLight ? 450 : 250]} />
        <ambientLight intensity={isLight ? 0.95 : 0.5} />
        <directionalLight position={[50, 80, 50]} intensity={isLight ? 1.3 : 1.3} castShadow />
        <directionalLight position={[-30, 40, -30]} intensity={isLight ? 0.5 : 0.35} />

        <VoxelMesh
          {...props}
          showOcclusion={showOcclusion}
          replayTrigger={props.replayTrigger ?? replayCount}
        />

        {/* Tactical 3D Markers (Viewshed Observer Mast, LoS Laser Ray, BDA Crater Ring, NOE Ribbon) */}
        <TacticalMarkers
          observerPos={props.observerPos}
          targetPos={props.targetPos}
          sightlineClear={props.sightlineClear}
          strikePos={props.strikePos}
          strikeSimulation={props.strikeSimulation}
          noeWaypoints={props.noeWaypoints}
          meshStats={{ ...props.meshStats, pixel_size: props.meshStats.pixel_size ?? 1.0 }}
          dsmRaw={props.dsmRaw}
          verticalScale={props.verticalScale ?? 1.0}
          waterLevel={props.waterLevel ?? 0}
          renderMode="voxel"
          voxelResolution={props.targetResolution ?? 64}
        />

        <VoxelCameraController />

        <gridHelper args={[200, 50, gridPrimary, gridSecondary]} position={[0, -0.05, 0]} />
      </Canvas>

      {/* Flight Telemetry HUD & Animation Controls */}
      <div className={`absolute bottom-4 left-4 p-3 backdrop-blur-md border text-xs font-mono space-y-2 shadow-xl ${
        isLight ? 'bg-white/90 border-black/15 text-neutral-800' : 'bg-black/90 border-white/15 text-neutral-300'
      }`}>
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-[#10B981]" />
            <span className={`text-[10px] font-mono font-bold tracking-widest uppercase ${isLight ? 'text-neutral-900' : 'text-white'}`}>
              Voxel 3D Navigation
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={handleToggleOcclusion}
              className={`flex items-center gap-1 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider font-semibold border transition-all duration-150 cursor-pointer ${
                showOcclusion
                  ? 'border-purple-500 bg-purple-500/20 text-purple-300 active:scale-95'
                  : isLight
                  ? 'border-neutral-300 bg-neutral-100 hover:bg-neutral-200 text-neutral-800 active:scale-95'
                  : 'border-white/20 bg-white/10 hover:bg-white/20 text-neutral-200 active:scale-95'
              }`}
              title="Toggle tri-state unobserved/occluded voxel display (§3.2)"
            >
              <Eye className={`w-2.5 h-2.5 ${showOcclusion ? 'text-purple-400' : 'text-neutral-400'}`} />
              <span>Tri-State: {showOcclusion ? 'ON' : 'OFF'}</span>
            </button>
            <button
              onClick={() => setReplayCount((c) => c + 1)}
              className={`flex items-center gap-1 px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider font-semibold border transition-all duration-150 cursor-pointer ${
                isLight
                  ? 'border-neutral-300 bg-neutral-100 hover:bg-neutral-200 text-neutral-800 active:scale-95'
                  : 'border-white/20 bg-white/10 hover:bg-white/20 text-neutral-200 active:scale-95'
              }`}
              title="Replay cinematic radial reveal animation"
            >
              <Sparkles className="w-2.5 h-2.5 text-cyan-400" />
              <span>Replay</span>
            </button>
          </div>
        </div>

        {showOcclusion && (
          <div className="flex items-center gap-2 py-1 px-1.5 bg-purple-500/10 border border-purple-500/30 text-[9px] text-purple-300 font-mono">
            <span className="w-2 h-2 rounded-xs bg-[#8B5CF6] inline-block" />
            <span>Unobserved / Camera Occluded (§3.2)</span>
          </div>
        )}
        <div className={`text-[10px] space-y-1.5 font-mono ${isLight ? 'text-neutral-700' : 'text-neutral-300'}`}>
          <div className="flex items-center gap-2">
            <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>DRAG</kbd>
            <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>ROTATE</span>
            <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ml-1 ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>SCROLL</kbd>
            <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>ZOOM</span>
          </div>
          <div className="flex items-center gap-2">
            <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>WASD</kbd>
            <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>FLY</span>
            <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ml-1 ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>Q / E</kbd>
            <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>ALTITUDE</span>
          </div>
        </div>
      </div>
    </div>
  );
}
