import { STATUS_META, type DetectorStatus } from '@/lib/status';

export function StatusChip({ status }: { status: DetectorStatus }) {
  const meta = STATUS_META[status];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold ${meta.chip}`}
      role="status"
    >
      <span className={`h-2.5 w-2.5 rounded-full ${meta.dot} animate-pulseDot`} aria-hidden />
      {meta.label}
    </span>
  );
}
