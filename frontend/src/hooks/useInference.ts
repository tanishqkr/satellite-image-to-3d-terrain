import { useState, useCallback } from 'react';

export interface MeshStats {
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  vertices: number;
  triangles: number;
  elevation_min: number;
  elevation_max: number;
  elevation_range: number;
  pixel_size: number;
}

export interface CalibrationData {
  alpha: number;
  min: number;
  max: number;
  mean: number;
  mode: string;
  unit: string;
}

export interface InferenceResult {
  request_id: string;
  heightmap_b64: string;
  rgb_b64: string;
  normal_map_b64?: string;
  dsm_colorized_b64: string;
  mesh_stats: MeshStats;
  calibration: CalibrationData;
  dsm_raw: number[][];
  dsm_raw_b64?: string;
  is_georef: boolean;
  crs?: string;
  confidence_mean?: number;
  uncertainty_map_b64?: string;
  uncertainty_stats?: {
    confidence_mean?: number;
    epistemic_mean?: number;
    aleatoric_mean?: number;
    uncertainty_mean_m?: number;
    uncertainty_max_m?: number;
    unit?: string;
  };
  rectification?: {
    applied: boolean;
    pitch_deg: number;
    roll_deg: number;
    occluded_fraction: number;
    occluded_pixel_count: number;
  };
  inference_time_ms: number;
}

export function useInference() {
  const [data, setData] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const upload = useCallback(async (file: File, estimateUncertainty: boolean = false) => {
    setLoading(true);
    setError(null);
    setProgress(10);

    try {
      const formData = new FormData();
      formData.append('file', file);

      setProgress(30);

      const url = estimateUncertainty ? '/api/upload?estimate_uncertainty=true' : '/api/upload';
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      setProgress(80);

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `Server error: ${response.status}` }));
        throw new Error(err.detail || `Server error: ${response.status}`);
      }

      const result: InferenceResult = await response.json();

      // Unpack high-performance Float32 binary buffer if provided (§mesh optimization)
      if (result.dsm_raw_b64) {
        try {
          const binary = atob(result.dsm_raw_b64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          const floats = new Float32Array(bytes.buffer);
          const rows = result.mesh_stats?.height || 512;
          const cols = result.mesh_stats?.width || 512;
          const grid: number[][] = new Array(rows);
          for (let r = 0; r < rows; r++) {
            grid[r] = Array.from(floats.subarray(r * cols, (r + 1) * cols));
          }
          result.dsm_raw = grid;
        } catch (err) {
          console.warn('Failed to decode dsm_raw_b64 binary buffer, using fallback dsm_raw', err);
        }
      }

      setData(result);
      setProgress(100);
    } catch (e: any) {
      setError(e.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setProgress(0);
  }, []);

  return { upload, data, loading, error, progress, reset };
}
