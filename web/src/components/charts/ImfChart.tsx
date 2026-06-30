import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from 'recharts';

const COLORS = ['#7C3AED', '#10B981', '#EF4444', '#F59E0B', '#8B5CF6', '#EC4899', '#14B8A6'];

export function ImfChart({ imfs, sampleRate }: { imfs: number[][]; sampleRate: number }) {
  return (
    <div className="space-y-4">
      {imfs.map((imf, idx) => {
        const data = imf.map((y, i) => ({ t: i / sampleRate, y }));
        return (
          <div key={idx}>
            <div className="mb-1 text-xs font-semibold text-muted">IMF {idx + 1}</div>
            <ResponsiveContainer width="100%" height={90}>
              <LineChart data={data} margin={{ top: 2, right: 4, bottom: 0, left: 0 }}>
                <XAxis dataKey="t" hide />
                <YAxis hide domain={['auto', 'auto']} />
                <Line
                  dataKey="y"
                  stroke={COLORS[idx % COLORS.length]}
                  dot={false}
                  strokeWidth={1.3}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}
