import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface Props {
  score: number[];
  threshold?: number[];
  height?: number;
}

/** Wrapper uPlot (tema oscuro) para la serie de score + overlay de umbral. */
export function RealtimeChart({ score, threshold, height = 180 }: Props) {
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
        {
          stroke: '#42546B',
          grid: { stroke: '#1E2A3A', width: 1 },
          ticks: { stroke: '#1E2A3A' },
          size: 38,
          font: '11px "JetBrains Mono", monospace',
        },
      ],
      series: [
        {},
        { stroke: '#22D3EE', width: 2, fill: 'rgba(34,211,238,0.14)' },
        { stroke: '#FBBF24', width: 1.5, dash: [4, 4] },
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
