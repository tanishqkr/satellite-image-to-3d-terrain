/**
 * Turbo colormap: maps a value in [0, 1] to an RGB tuple [r, g, b] each in [0, 255].
 */
export function turboColormap(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  let r: number, g: number, b: number;

  if (t < 0.25) {
    r = 0;
    g = t * 4;
    b = 1;
  } else if (t < 0.5) {
    r = 0;
    g = 1;
    b = 1 - (t - 0.25) * 4;
  } else if (t < 0.75) {
    r = (t - 0.5) * 4;
    g = 1;
    b = 0;
  } else {
    r = 1;
    g = 1 - (t - 0.75) * 4;
    b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Generate a full colormap array of N entries.
 */
export function generateColormapLUT(n: number = 256): Uint8Array {
  const lut = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = turboColormap(i / (n - 1));
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}
