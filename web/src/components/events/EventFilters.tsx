export interface FilterValues {
  minScore: number;
  limit: number;
  sort: 'recent' | 'score';
}

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
    <div className="flex flex-wrap items-end gap-4 rounded-2xl border border-line bg-surface p-4">
      <label className="text-sm">
        <span className="mb-1 block font-medium text-muted">
          Score mínimo: {values.minScore.toFixed(2)}
        </span>
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
        <span className="mb-1 block font-medium text-muted">Mostrar</span>
        <select
          value={values.limit}
          onChange={(e) => onChange({ limit: Number(e.target.value) })}
          className="rounded-lg border border-line bg-surface px-3 py-2"
        >
          {[10, 25, 50, 100].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>

      <label className="text-sm">
        <span className="mb-1 block font-medium text-muted">Orden</span>
        <select
          value={values.sort}
          onChange={(e) => onChange({ sort: e.target.value as FilterValues['sort'] })}
          className="rounded-lg border border-line bg-surface px-3 py-2"
        >
          <option value="recent">Más recientes</option>
          <option value="score">Mayor score</option>
        </select>
      </label>

      <div className="ml-auto">
        <button
          type="button"
          onClick={onClearAll}
          className="rounded-lg border border-anomaly/40 px-3 py-2 text-sm font-medium text-anomaly hover:bg-red-50"
        >
          Borrar todo
        </button>
      </div>
    </div>
  );
}
