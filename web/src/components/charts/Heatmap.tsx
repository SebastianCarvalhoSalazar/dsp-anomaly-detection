import { useEffect, useRef } from 'react';

/** Aproximación de la paleta Viridis con tramos lineales. */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

function viridis(t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i]!;
  const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)]!;
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

/** Heatmap canvas para el mel-spectrogram (filas = bandas, columnas = frames). */
export function Heatmap({ z, height = 320 }: { z: number[][]; height?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || z.length === 0) return;
    const rows = z.length;
    const cols = z[0]?.length ?? 0;
    if (cols === 0) return;

    let min = Infinity;
    let max = -Infinity;
    for (const row of z) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const span = max - min || 1;

    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      const row = z[r]!;
      // fila 0 arriba = banda más alta: invertimos para que las graves queden abajo
      const destRow = rows - 1 - r;
      for (let c = 0; c < cols; c++) {
        const t = ((row[c] ?? min) - min) / span;
        const color = viridis(t);
        const m = /rgb\((\d+),(\d+),(\d+)\)/.exec(color)!;
        const o = (destRow * cols + c) * 4;
        img.data[o] = Number(m[1]);
        img.data[o + 1] = Number(m[2]);
        img.data[o + 2] = Number(m[3]);
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [z]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Mel-spectrogram"
      className="w-full rounded-xl"
      style={{ height, imageRendering: 'pixelated' }}
    />
  );
}
