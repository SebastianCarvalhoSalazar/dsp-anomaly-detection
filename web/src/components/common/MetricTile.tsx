export function MetricTile({
  label,
  value,
  help,
}: {
  label: string;
  value: string;
  help?: string;
}) {
  return (
    <div className="rounded-xl bg-surface px-4 py-3 shadow-sm" title={help}>
      <div className="text-xs font-semibold uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 text-xl font-bold leading-none text-ink">{value}</div>
    </div>
  );
}
