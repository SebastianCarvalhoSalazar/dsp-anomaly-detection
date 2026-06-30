/** KDE gaussiano simple sobre [0,1]. Equivalente ligero a scipy.gaussian_kde. */
export function gaussianKde(
  samples: number[],
  points = 120,
  bandwidth = 0.07,
): { x: number; y: number }[] {
  const xs = Array.from({ length: points }, (_, i) => i / (points - 1));
  const valid = samples.filter((s) => s > 0);
  if (valid.length < 2) return xs.map((x) => ({ x, y: 0 }));

  const norm = 1 / (valid.length * bandwidth * Math.sqrt(2 * Math.PI));
  return xs.map((x) => {
    let acc = 0;
    for (const v of valid) {
      const u = (x - v) / bandwidth;
      acc += Math.exp(-0.5 * u * u);
    }
    return { x, y: acc * norm };
  });
}
