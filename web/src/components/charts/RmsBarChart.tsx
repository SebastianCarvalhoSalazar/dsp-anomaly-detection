import { Bar, BarChart, ResponsiveContainer, Tooltip, YAxis } from 'recharts';

export function RmsBarChart({ rms }: { rms: number[] }) {
  const data = rms.map((v, i) => ({ i, v }));
  return (
    <div role="img" aria-label="Amplitud RMS reciente">
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={data} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
          <YAxis hide domain={[0, 'auto']} />
          <Tooltip
            cursor={{ fill: 'rgba(34,211,238,0.08)' }}
            formatter={(v: number) => v.toFixed(4)}
            labelFormatter={() => ''}
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              background: '#0E141D',
              border: '1px solid #1E2A3A',
              color: '#E6EDF3',
            }}
          />
          <Bar dataKey="v" fill="rgba(34,211,238,0.6)" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
