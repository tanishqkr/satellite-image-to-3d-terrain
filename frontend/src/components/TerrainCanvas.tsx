import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import TacticalMarkers from './TacticalMarkers.tsx';

export interface TerrainCanvasProps {
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
  showContours: boolean;
  contourInterval?: number;
  dsmRaw: number[][];
  interactive?: boolean;
  environmentTheme?: 'dark' | 'light';
  viewshedOverlayUrl?: string | null;
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

function Terrain({
  heightmapB64,
  rgbB64,
  normalMapB64,
  meshStats,
  verticalScale,
  waterLevel,
  showContours,
  contourInterval = 5,
  viewshedOverlayUrl,
  onTerrainClick,
}: TerrainCanvasProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [heightTex, setHeightTex] = useState<THREE.Texture | null>(null);
  const [colorTex, setColorTex] = useState<THREE.Texture | null>(null);
  const [normalTex, setNormalTex] = useState<THREE.Texture | null>(null);
  const [viewshedTex, setViewshedTex] = useState<THREE.Texture | null>(null);

  const loader = useMemo(() => new THREE.TextureLoader(), []);

  // Safe 1x1 transparent fallback texture so WebGL never binds a null sampler2D
  const fallbackTex = useMemo(() => {
    const tex = new THREE.DataTexture(new Uint8Array([0, 0, 0, 0]), 1, 1, THREE.RGBAFormat);
    tex.needsUpdate = true;
    return tex;
  }, []);

  useEffect(() => {
    // Load RGB texture
    const rgbUrl = `data:image/jpeg;base64,${rgbB64}`;
    loader.load(rgbUrl, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      setColorTex(tex);
    });

    // Load heightmap texture
    const hmUrl = `data:image/png;base64,${heightmapB64}`;
    loader.load(hmUrl, (tex) => {
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      setHeightTex(tex);
    });

    // Load normal map for WebGL dynamic lighting & relief shading
    if (normalMapB64) {
      const normUrl = `data:image/png;base64,${normalMapB64}`;
      loader.load(normUrl, (tex) => {
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        setNormalTex(tex);
      });
    } else {
      setNormalTex(null);
    }
  }, [heightmapB64, rgbB64, normalMapB64, loader]);

  useEffect(() => {
    if (viewshedOverlayUrl) {
      loader.load(viewshedOverlayUrl, (tex) => {
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        setViewshedTex(tex);
      });
    } else {
      setViewshedTex(null);
    }
  }, [viewshedOverlayUrl, loader]);

  // Clean up GPU texture memory to prevent leaks
  useEffect(() => {
    return () => {
      if (colorTex) colorTex.dispose();
      if (heightTex) heightTex.dispose();
      if (normalTex) normalTex.dispose();
      if (viewshedTex) viewshedTex.dispose();
      fallbackTex.dispose();
    };
  }, [colorTex, heightTex, normalTex, viewshedTex, fallbackTex]);

  const geometry = useMemo(() => {
    const w = meshStats.width;
    const h = meshStats.height;
    const segments = Math.min(w, 512);
    return new THREE.PlaneGeometry(w * 0.1, h * 0.1, segments, segments);
  }, [meshStats]);

  useEffect(() => {
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  const isRelative = meshStats.elevation_range <= 2.0;

  // In metric mode: 1 meter elevation corresponds to 0.1 Three.js world units.
  // In relative mode: elevation span is [0, 1]. To produce rich, visible 3D urban relief on a 51.2-wide plane,
  // we scale the normalized [0, 1] relative depth to 18.0 world units (comparable to 180m metric relief).
  const displacementScale = isRelative
    ? 18.0 * verticalScale
    : meshStats.elevation_range * verticalScale * 0.1;

  const verticalFactor = isRelative
    ? 18.0 * verticalScale
    : verticalScale * 0.1;

  // WebGL shader uniforms for dynamic 3D contour lines and viewshed/tactical overlay
  const uniformsRef = useRef({
    uShowContours: { value: showContours ? 1.0 : 0.0 },
    uContourInterval: { value: contourInterval },
    uElevationMin: { value: meshStats.elevation_min },
    uVerticalScale: { value: verticalScale },
    uDisplacementScale: { value: displacementScale },
    uIsRelative: { value: isRelative ? 1.0 : 0.0 },
    uShowViewshed: { value: viewshedTex ? 1.0 : 0.0 },
    uViewshedTex: { value: viewshedTex || fallbackTex },
  });

  useEffect(() => {
    uniformsRef.current.uShowContours.value = showContours ? 1.0 : 0.0;
    uniformsRef.current.uContourInterval.value = contourInterval;
    uniformsRef.current.uElevationMin.value = meshStats.elevation_min;
    uniformsRef.current.uVerticalScale.value = verticalScale;
    uniformsRef.current.uDisplacementScale.value = displacementScale;
    uniformsRef.current.uIsRelative.value = isRelative ? 1.0 : 0.0;
    uniformsRef.current.uShowViewshed.value = viewshedTex ? 1.0 : 0.0;
    uniformsRef.current.uViewshedTex.value = viewshedTex || fallbackTex;
  }, [showContours, contourInterval, meshStats.elevation_min, verticalScale, displacementScale, isRelative, viewshedTex, fallbackTex]);

  const onBeforeCompile = useMemo(() => {
    return (shader: THREE.WebGLProgramParametersWithUniforms) => {
      shader.uniforms.uShowContours = uniformsRef.current.uShowContours;
      shader.uniforms.uContourInterval = uniformsRef.current.uContourInterval;
      shader.uniforms.uElevationMin = uniformsRef.current.uElevationMin;
      shader.uniforms.uVerticalScale = uniformsRef.current.uVerticalScale;
      shader.uniforms.uDisplacementScale = uniformsRef.current.uDisplacementScale;
      shader.uniforms.uIsRelative = uniformsRef.current.uIsRelative;
      shader.uniforms.uShowViewshed = uniformsRef.current.uShowViewshed;
      shader.uniforms.uViewshedTex = uniformsRef.current.uViewshedTex;

      shader.vertexShader = `
        varying vec3 vTerrainWorldPos;
        varying vec2 vTerrainUv;
        ${shader.vertexShader}
      `.replace(
        '#include <worldpos_vertex>',
        `
        #include <worldpos_vertex>
        vTerrainWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        vTerrainUv = uv;
        `
      );

      shader.fragmentShader = `
        varying vec3 vTerrainWorldPos;
        varying vec2 vTerrainUv;
        uniform float uShowContours;
        uniform float uContourInterval;
        uniform float uElevationMin;
        uniform float uVerticalScale;
        uniform float uDisplacementScale;
        uniform float uIsRelative;
        uniform float uShowViewshed;
        uniform sampler2D uViewshedTex;
        ${shader.fragmentShader}
      `.replace(
        '#include <dithering_fragment>',
        `
        #include <dithering_fragment>
        if (uShowContours > 0.5 && uContourInterval > 0.0001) {
          float vScale = uIsRelative > 0.5 ? max(uDisplacementScale, 0.0001) : max(uVerticalScale * 0.1, 0.0001);
          float elev = uElevationMin + (vTerrainWorldPos.y / vScale);
          float cInt = max(uContourInterval, 0.0001);
          float numIntervals = elev / cInt;
          float dist = abs(numIntervals - floor(numIntervals + 0.5)) * cInt;
          float dElev = max(fwidth(elev), 0.001);
          float line = 1.0 - smoothstep(0.0, dElev * 1.5, dist);

          float indexInt = cInt * 5.0;
          float numIndex = elev / indexInt;
          float indexDist = abs(numIndex - floor(numIndex + 0.5)) * indexInt;
          float indexLine = 1.0 - smoothstep(0.0, dElev * 2.2, indexDist);

          vec3 contourColor = mix(vec3(0.05, 0.1, 0.2), vec3(1.0, 1.0, 1.0), indexLine * 0.5);
          float alpha = max(line * 0.75, indexLine * 0.95);
          gl_FragColor.rgb = mix(gl_FragColor.rgb, contourColor, alpha);
        }

        if (uShowViewshed > 0.5) {
          vec4 vColor = texture2D(uViewshedTex, vTerrainUv);
          if (vColor.a > 0.02) {
            gl_FragColor.rgb = mix(gl_FragColor.rgb, vColor.rgb, vColor.a * 0.70);
          }
        }
        `
      );
    };
  }, []);

  const handlePointerDown = (e: any) => {
    if (!onTerrainClick) return;
    e.stopPropagation();
    const pt = e.point;
    const col = Math.max(0, Math.min(meshStats.width - 1, Math.round(pt.x / 0.1 + meshStats.width / 2)));
    const row = Math.max(0, Math.min(meshStats.height - 1, Math.round(pt.z / 0.1 + meshStats.height / 2)));
    onTerrainClick({ col, row, isShift: !!(e.nativeEvent?.shiftKey || e.nativeEvent?.altKey) });
  };

  if (!heightTex || !colorTex) return null;

  return (
    <>
      <mesh
        ref={meshRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        onPointerDown={handlePointerDown}
      >
        <primitive object={geometry} />
        <meshStandardMaterial
          map={colorTex}
          displacementMap={heightTex}
          displacementScale={displacementScale}
          displacementBias={0}
          normalMap={normalTex || undefined}
          normalScale={isRelative ? new THREE.Vector2(2.0, 2.0) : new THREE.Vector2(1.2, 1.2)}
          side={THREE.DoubleSide}
          roughness={0.7}
          metalness={0.1}
          onBeforeCompile={onBeforeCompile}
          customProgramCacheKey={() => 'terrain_contour_mat_v6'}
        />
      </mesh>

      {/* Synchronized Water plane for flood simulation */}
      {waterLevel > meshStats.elevation_min && (
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, (waterLevel - meshStats.elevation_min) * verticalFactor + 0.05, 0]}
        >
          <planeGeometry args={[meshStats.width * 0.12, meshStats.height * 0.12]} />
          <meshStandardMaterial
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

function CameraController() {
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
    // Fix first-click camera angle snap: extract current Euler orientation from quaternion
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
        // Re-sync Euler orientation when pointer lock engages to prevent any angle snap
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

    // Separate horizontal flight from vertical ascension
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
    if (keys.current.has('e') || keys.current.has(' ')) upDown += 1;
    if (keys.current.has('q') || keys.current.has('c')) upDown -= 1;

    if (upDown !== 0) {
      camera.position.y += upDown * speed * dt;
    }

    // Clamp minimum height to prevent subterranean clipping
    if (camera.position.y < 3.0) camera.position.y = 3.0;
  });

  return null;
}

function MinimapCameraController() {
  const { camera } = useThree();
  useEffect(() => {
    camera.position.set(0, 45, 45);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
  }, [camera]);
  return null;
}

export default function TerrainCanvas({
  interactive = true,
  environmentTheme = 'dark',
  ...props
}: TerrainCanvasProps) {
  const isLight = environmentTheme === 'light';
  // Light mode: Pure clean white environment with crisp black grid lines
  // Dark mode: Untouched dark void (#050608) with slate grid lines (#1e293b)
  const bgColor = isLight ? '#ffffff' : '#050608';
  const gridPrimary = isLight ? '#000000' : '#1e293b';
  const gridSecondary = isLight ? '#1f2937' : '#1e293b';

  return (
    <div className={`w-full h-full relative select-none ${isLight ? 'bg-white' : 'bg-[#050608]'}`}>
      <Canvas
        camera={{ fov: 60, near: 0.1, far: 2000 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: bgColor }}
      >
        <color attach="background" args={[bgColor]} />
        <fog attach="fog" args={[bgColor, isLight ? 80 : 60, isLight ? 450 : 250]} />
        <ambientLight intensity={isLight ? 0.95 : 0.4} />
        <directionalLight position={[50, 80, 50]} intensity={isLight ? 1.3 : 1.2} castShadow />
        <directionalLight position={[-30, 40, -30]} intensity={isLight ? 0.5 : 0.3} />

        <Terrain {...props} />

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
          verticalScale={props.verticalScale}
          waterLevel={props.waterLevel}
          renderMode="smooth"
        />

        {interactive ? <CameraController /> : <MinimapCameraController />}

        <gridHelper args={[200, 50, gridPrimary, gridSecondary]} position={[0, -0.1, 0]} />
      </Canvas>

      {/* Architectural Flight Telemetry HUD */}
      {interactive && (
        <div className={`absolute bottom-4 left-4 p-3 pointer-events-none backdrop-blur-md border text-xs font-mono space-y-2 shadow-xl ${
          isLight ? 'bg-white/90 border-black/15 text-neutral-800' : 'bg-black/90 border-white/15 text-neutral-300'
        }`}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-[#38BDF8]" />
            <span className={`text-[10px] font-mono font-bold tracking-widest uppercase ${isLight ? 'text-neutral-900' : 'text-white'}`}>
              3D Navigation Controls
            </span>
          </div>
          <div className={`text-[10px] space-y-1.5 font-mono ${isLight ? 'text-neutral-700' : 'text-neutral-300'}`}>
            <div className="flex items-center gap-2">
              <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>DRAG</kbd>
              <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>ROTATE VIEW</span>
              <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ml-1 ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>SCROLL</kbd>
              <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>ZOOM</span>
            </div>
            <div className="flex items-center gap-2">
              <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>WASD</kbd>
              <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>FLY</span>
              <kbd className={`px-1.5 py-0.5 border font-bold text-[9px] ml-1 ${isLight ? 'border-neutral-300 bg-neutral-100 text-neutral-900' : 'border-white/20 bg-white/5 text-white'}`}>Q / E</kbd>
              <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>ALTITUDE</span>
              <kbd className="px-1.5 py-0.5 border border-[#38BDF8]/40 bg-[#38BDF8]/10 text-[#38BDF8] font-bold text-[9px] ml-1">DBL-CLICK</kbd>
              <span className={isLight ? 'text-neutral-500' : 'text-neutral-400'}>LOCK MOUSE</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
