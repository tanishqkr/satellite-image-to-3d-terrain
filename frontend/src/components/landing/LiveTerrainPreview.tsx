import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Compass, ArrowUpRight } from 'lucide-react';

interface LiveTerrainPreviewProps {
  onLoadPreset?: () => void;
}

function PreviewMesh() {
  const groupRef = useRef<THREE.Group>(null);
  const isDragging = useRef(false);
  const prevPos = useRef({ x: 0, y: 0 });
  const rotationEuler = useRef(new THREE.Euler(0.45, -0.55, 0, 'YXZ'));

  // Generate procedural relief elevation geometry (36x36 grid)
  const { geometry, wireGeometry } = useMemo(() => {
    const size = 26;
    const segs = 36;
    const geom = new THREE.PlaneGeometry(size, size, segs, segs);
    // Rotate to lie in X-Z ground plane
    geom.rotateX(-Math.PI / 2);

    const pos = geom.attributes.position;
    const colors: number[] = [];
    const color = new THREE.Color();

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const dist = Math.hypot(x, z);

      // Topographic hill with canyon cut and stepped terraces
      const r1 = Math.sin(x * 0.32) * Math.cos(z * 0.32) * 3.4;
      const r2 = Math.sin(x * 0.65 + z * 0.45) * 1.6;
      const r3 = Math.cos(dist * 0.28) * 1.8;
      const canyon = Math.abs(Math.sin(x * 0.2 + z * 0.15)) * 0.8;
      const falloff = Math.max(0, 1.0 - Math.min(1.0, (dist / 14) ** 2));
      const y = Math.max(0.1, (r1 + r2 + r3 - canyon + 2.8) * falloff);

      pos.setY(i, y);

      // Discrete Turbo elevation color progression
      const normH = Math.max(0, Math.min(1, y / 6.5));
      if (normH < 0.25) {
        color.setRGB(0.08, 0.25 + normH * 2.5, 0.95);
      } else if (normH < 0.5) {
        color.setRGB(0.08, 0.95, 0.95 - (normH - 0.25) * 3.2);
      } else if (normH < 0.75) {
        color.setRGB(0.1 + (normH - 0.5) * 3.6, 0.95, 0.05);
      } else {
        color.setRGB(0.98, 0.95 - (normH - 0.75) * 3.2, 0.08);
      }

      colors.push(color.r, color.g, color.b);
    }

    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    const wireGeom = new THREE.WireframeGeometry(geom);
    return { geometry: geom, wireGeometry: wireGeom };
  }, []);

  useFrame((_, delta) => {
    if (!groupRef.current || isDragging.current) return;
    // Gentle continuous aerospace orbit
    rotationEuler.current.y += delta * 0.18;
    groupRef.current.rotation.copy(rotationEuler.current);
  });

  return (
    <group
      ref={groupRef}
      rotation={[rotationEuler.current.x, rotationEuler.current.y, 0]}
      onPointerDown={(e) => {
        isDragging.current = true;
        prevPos.current = { x: e.clientX, y: e.clientY };
        (e.target as HTMLElement)?.setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!isDragging.current || !groupRef.current) return;
        const dx = e.clientX - prevPos.current.x;
        const dy = e.clientY - prevPos.current.y;
        prevPos.current = { x: e.clientX, y: e.clientY };

        rotationEuler.current.y += dx * 0.01;
        rotationEuler.current.x = Math.max(-0.4, Math.min(0.8, rotationEuler.current.x + dy * 0.01));
        groupRef.current.rotation.copy(rotationEuler.current);
      }}
      onPointerUp={(e) => {
        isDragging.current = false;
        (e.target as HTMLElement)?.releasePointerCapture?.(e.pointerId);
      }}
      onPointerCancel={() => {
        isDragging.current = false;
      }}
    >
      {/* Primary Solid Topographic Surface */}
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          roughness={0.42}
          metalness={0.08}
          side={THREE.DoubleSide}
          flatShading
        />
      </mesh>

      {/* Wireframe Elevation Grid Overlay */}
      <lineSegments geometry={wireGeometry}>
        <lineBasicMaterial
          color="#38bdf8"
          transparent
          opacity={0.22}
        />
      </lineSegments>
    </group>
  );
}

export default function LiveTerrainPreview({ onLoadPreset }: LiveTerrainPreviewProps) {
  const [fps] = useState(60);

  return (
    <div className="relative w-full h-[400px] sm:h-[440px] lg:h-[480px] bg-[#07080b] border border-white/20 select-none overflow-hidden group">
      {/* Corner Brackets / Reticle Marks */}
      <span className="absolute top-2 left-2 font-mono text-[11px] text-white/40 z-10 select-none">+</span>
      <span className="absolute top-2 right-2 font-mono text-[11px] text-white/40 z-10 select-none">+</span>
      <span className="absolute bottom-2 left-2 font-mono text-[11px] text-white/40 z-10 select-none">+</span>
      <span className="absolute bottom-2 right-2 font-mono text-[11px] text-white/40 z-10 select-none">+</span>

      {/* Top Telemetry Header Bar */}
      <div className="absolute top-0 inset-x-0 p-3 flex items-center justify-between border-b border-white/10 bg-[#050608]/85 backdrop-blur-md z-10 pointer-events-none">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse" />
          <span className="font-mono text-[10px] font-bold tracking-widest text-white uppercase">
            LIVE RECONSTRUCTION
          </span>
          <span className="text-[10px] font-mono text-neutral-500 hidden sm:inline">
            // RT-PREVIEW
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span className="im-tag">FPS {fps}</span>
          <span className="im-tag hidden sm:inline-flex">RES: 36×36</span>
        </div>
      </div>

      {/* WebGL 3D Canvas */}
      <div className="w-full h-full cursor-grab active:cursor-grabbing">
        <Canvas
          camera={{ position: [0, 20, 24], fov: 42 }}
          gl={{ antialias: true, alpha: true }}
          style={{ background: '#07080b' }}
        >
          <ambientLight intensity={0.65} />
          <directionalLight position={[25, 40, 25]} intensity={1.2} />
          <directionalLight position={[-20, 20, -20]} intensity={0.4} />

          <PreviewMesh />

          <gridHelper
            args={[36, 18, '#1e293b', '#0f172a']}
            position={[0, -0.05, 0]}
          />
        </Canvas>
      </div>

      {/* Left Scientific Telemetry Overlay */}
      <div className="absolute top-14 left-3 space-y-1 font-mono text-[9px] text-neutral-400 pointer-events-none z-10 bg-black/50 backdrop-blur-sm p-2 border border-white/10 hidden sm:block">
        <div className="text-white font-bold text-[10px] border-b border-white/10 pb-1 mb-1">
          TELEM // RECON-01
        </div>
        <div>LAT: 23°07&apos;12&quot;N</div>
        <div>LON: 72°30&apos;44&quot;E</div>
        <div>ELEV: +342.5M MSL</div>
        <div>GSD: 0.33M / PX</div>
        <div className="text-cyan-400">DATUM: WGS-84</div>
      </div>

      {/* Bottom Status & Direct Action Bar */}
      <div className="absolute bottom-0 inset-x-0 p-3 flex items-center justify-between border-t border-white/10 bg-[#050608]/85 backdrop-blur-md z-10">
        <div className="flex items-center gap-2 text-[10px] font-mono text-neutral-400">
          <Compass className="w-3 h-3 text-[#38BDF8]" />
          <span>DRAG TO ORBIT · SCROLL TO ZOOM</span>
        </div>

        {onLoadPreset && (
          <button
            type="button"
            onClick={onLoadPreset}
            className="flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-mono font-bold uppercase tracking-wider text-white bg-white/10 hover:bg-white hover:text-black border border-white/30 transition-all cursor-pointer shadow-lg active:scale-95"
            title="Load calibrated multispectral satellite GeoTIFF dataset"
          >
            <span>LOAD TESTBED</span>
            <ArrowUpRight className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
}
