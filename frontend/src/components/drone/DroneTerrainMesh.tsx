import React, { useMemo, useEffect, useState } from 'react';
import * as THREE from 'three';

export interface DroneTerrainMeshProps {
  heightmapB64: string;
  rgbB64: string;
  normalMapB64?: string;
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
  };
  verticalScale: number;
  waterLevel: number;
}

/**
 * Renders the 3D terrain surface under the drone.
 * Decoupled from spectator cameras to preserve isolation.
 */
export default function DroneTerrainMesh({
  heightmapB64,
  rgbB64,
  normalMapB64,
  meshStats,
  verticalScale,
  waterLevel,
}: DroneTerrainMeshProps) {
  const [heightTex, setHeightTex] = useState<THREE.Texture | null>(null);
  const [colorTex, setColorTex] = useState<THREE.Texture | null>(null);
  const [normalTex, setNormalTex] = useState<THREE.Texture | null>(null);

  const loader = useMemo(() => new THREE.TextureLoader(), []);

  useEffect(() => {
    const rgbUrl = `data:image/jpeg;base64,${rgbB64}`;
    loader.load(rgbUrl, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      setColorTex(tex);
    });

    const hmUrl = `data:image/png;base64,${heightmapB64}`;
    loader.load(hmUrl, (tex) => {
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      setHeightTex(tex);
    });

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
    return () => {
      if (colorTex) colorTex.dispose();
      if (heightTex) heightTex.dispose();
      if (normalTex) normalTex.dispose();
    };
  }, [colorTex, heightTex, normalTex]);

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
  const displacementScale = isRelative
    ? 18.0 * verticalScale
    : meshStats.elevation_range * verticalScale * 0.1;

  const verticalFactor = isRelative
    ? 18.0 * verticalScale
    : verticalScale * 0.1;

  if (!heightTex || !colorTex) return null;

  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
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
        />
      </mesh>

      {/* Synchronized flood plane */}
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
