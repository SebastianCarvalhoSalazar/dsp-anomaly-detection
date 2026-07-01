import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts';

import { gaussianKde } from '@/lib/kde';

export function KdeChart({ scores, threshold }: { scores: number[]; threshold: number }) {
  const data = gaussianKde(scores);
  return (
    <div role="img" aria-label="Distribución (KDE) de los scores recientes">
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 10, fill: '#42546B' }}
            stroke="#1E2A3A"
          />
          <YAxis hide />
          <Area
            dataKey="y"
            stroke="#22D3EE"
            strokeWidth={2}
            fill="rgba(34,211,238,0.12)"
            isAnimationActive={false}
          />
          {threshold > 0 && (
            <ReferenceLine x={threshold} stroke="#FBBF24" strokeDasharray="4 4" />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
