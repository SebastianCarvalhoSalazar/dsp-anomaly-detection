export function MetricTile({
  label,
  value,
  help,
  valueClass = 'text-ink',
}: {
  label: string;
  value: string;
  help?: string;
  valueClass?: string;
}) {
  return (
    <div
      className="rounded-lg border border-line bg-surface px-4 py-3 shadow-panel"
      title={help}
    >
      <div className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted">
        {label}
      </div>
      <div className={`mt-1 font-mono text-xl font-semibold leading-none tnum ${valueClass}`}>
        {value}
      </div>
    </div>
  );
}
