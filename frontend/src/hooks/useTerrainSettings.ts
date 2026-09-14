import { useState } from 'react';

export interface TerrainSettings {
  verticalScale: number;
  setVerticalScale: React.Dispatch<React.SetStateAction<number>>;
  waterLevel: number;
  setWaterLevel: React.Dispatch<React.SetStateAction<number>>;
  showContours: boolean;
  setShowContours: React.Dispatch<React.SetStateAction<boolean>>;
  contourInterval: number;
  setContourInterval: React.Dispatch<React.SetStateAction<number>>;
  renderMode: 'voxel' | 'smooth';
  setRenderMode: React.Dispatch<React.SetStateAction<'voxel' | 'smooth'>>;
  voxelBands: number;
  setVoxelBands: React.Dispatch<React.SetStateAction<number>>;
  voxelResolution: number;
  setVoxelResolution: React.Dispatch<React.SetStateAction<number>>;
  environmentTheme: 'dark' | 'light';
  setEnvironmentTheme: React.Dispatch<React.SetStateAction<'dark' | 'light'>>;
  toggleEnvironmentTheme: () => void;
}

export interface TerrainSettingsInitialValues {
  initialVerticalScale?: number;
  initialWaterLevel?: number;
  initialShowContours?: boolean;
  initialContourInterval?: number;
  initialRenderMode?: 'voxel' | 'smooth';
  initialVoxelBands?: number;
  initialVoxelResolution?: number;
  initialEnvironmentTheme?: 'dark' | 'light';
}

/**
 * useTerrainSettings (§5):
 * Unified single source of truth for terrain, voxel, and environment configuration.
 * Shared between the 01 SURFACE sidebar tab, floating TerrainQuickPanel, and FPV Drone mode.
 */
export function useTerrainSettings(initialValues?: TerrainSettingsInitialValues): TerrainSettings {
  const [verticalScale, setVerticalScale] = useState<number>(initialValues?.initialVerticalScale ?? 0.5);
  const [waterLevel, setWaterLevel] = useState<number>(initialValues?.initialWaterLevel ?? 0);
  const [showContours, setShowContours] = useState<boolean>(initialValues?.initialShowContours ?? false);
  const [contourInterval, setContourInterval] = useState<number>(initialValues?.initialContourInterval ?? 5);
  const [renderMode, setRenderMode] = useState<'voxel' | 'smooth'>(initialValues?.initialRenderMode ?? 'voxel');
  const [voxelBands, setVoxelBands] = useState<number>(initialValues?.initialVoxelBands ?? 8);
  const [voxelResolution, setVoxelResolution] = useState<number>(initialValues?.initialVoxelResolution ?? 64);
  const [environmentTheme, setEnvironmentTheme] = useState<'dark' | 'light'>(
    initialValues?.initialEnvironmentTheme ?? 'dark'
  );

  const toggleEnvironmentTheme = () => {
    setEnvironmentTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  return {
    verticalScale,
    setVerticalScale,
    waterLevel,
    setWaterLevel,
    showContours,
    setShowContours,
    contourInterval,
    setContourInterval,
    renderMode,
    setRenderMode,
    voxelBands,
    setVoxelBands,
    voxelResolution,
    setVoxelResolution,
    environmentTheme,
    setEnvironmentTheme,
    toggleEnvironmentTheme,
  };
}
