export const fmtScore = (v: number, digits = 3): string => v.toFixed(digits);

/** "2026-06-30T12:34:56.789" -> "2026-06-30 12:34:56" */
export const fmtTimestamp = (iso: string): string =>
  iso ? iso.slice(0, 19).replace('T', ' ') : '—';

export const fmtBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};
