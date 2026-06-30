import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface Props {
  score: number[];
  threshold?: number[];
  height?: number;
}

/** Wrapper uPlot para la serie de score en tiempo real + overlay de umbral. */
export function RealtimeChart({ score, threshold, height = 170 }: Props) {
  const elRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  const buildData = (): uPlot.AlignedData => {
    const n = score.length;
    const xs = Array.from({ length: n }, (_, i) => i);
    const th =
      threshold && threshold.length === n ? threshold : (new Array(n).fill(null) as null[]);
    return [xs, score, th as (number | null)[]];
  };

  useEffect(() => {
    const el = elRef.current;
    if (!el) return;
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height,
      legend: { show: false },
      scales: { x: { time: false }, y: { range: [0, 1] } },
      axes: [
        { show: false },
        { stroke: '#64748B', grid: { stroke: '#F1F5F9' }, size: 36 },
      ],
      series: [
        {},
        { stroke: '#7C3AED', width: 2, fill: 'rgba(124,58,237,0.12)' },
        { stroke: '#F59E0B', width: 1.5, dash: [4, 4] },
      ],
    };
    const plot = new uPlot(opts, buildData(), el);
    plotRef.current = plot;
    const ro = new ResizeObserver(() => plot.setSize({ width: el.clientWidth, height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height]);

  useEffect(() => {
    plotRef.current?.setData(buildData());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [score, threshold]);

  return <div ref={elRef} role="img" aria-label="Serie temporal del anomaly score" />;
}
