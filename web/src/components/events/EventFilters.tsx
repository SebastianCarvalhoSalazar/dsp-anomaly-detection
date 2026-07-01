export interface FilterValues {
  minScore: number;
  limit: number;
  sort: 'recent' | 'score';
}

const selectCls =
  'rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-sm text-ink';
const labelCls = 'mb-1 block text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted';

export function EventFilters({
  values,
  onChange,
  onClearAll,
}: {
  values: FilterValues;
  onChange: (patch: Partial<FilterValues>) => void;
  onClearAll: () => void;
}) {
  return (
    <div className="flex flex-wrap items-end gap-4 rounded-lg border border-line bg-surface p-4 shadow-panel">
      <label className="text-sm">
        <span className={labelCls}>Score mínimo · {values.minScore.toFixed(2)}</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={values.minScore}
          onChange={(e) => onChange({ minScore: Number(e.target.value) })}
          className="w-48 accent-primary"
        />
      </label>

      <label className="text-sm">
        <span className={labelCls}>Mostrar</span>
        <select
          value={values.limit}
          onChange={(e) => onChange({ limit: Number(e.target.value) })}
          className={selectCls}
        >
          {[10, 25, 50, 100].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>

      <label className="text-sm">
        <span className={labelCls}>Orden</span>
        <select
          value={values.sort}
          onChange={(e) => onChange({ sort: e.target.value as FilterValues['sort'] })}
          className={selectCls}
        >
          <option value="recent">Más recientes</option>
          <option value="score">Mayor score</option>
        </select>
      </label>

      <div className="ml-auto">
        <button
          type="button"
          onClick={onClearAll}
          className="rounded-md border border-anomaly/40 px-3 py-2 text-sm font-semibold uppercase tracking-wide text-anomaly hover:bg-anomaly/10"
        >
          Borrar todo
        </button>
      </div>
    </div>
  );
}
