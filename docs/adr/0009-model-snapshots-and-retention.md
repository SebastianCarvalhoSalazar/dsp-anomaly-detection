# ADR-0009: Snapshots de modelos y retención

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 3

## Contexto y problema

No había forma de auditar qué modelo estaba activo en un momento dado ni de
reproducir/depurar una detección pasada. El requisito pide guardar snapshots
de PCA + IsolationForest (audio y video) en cada refit, junto con
`timestamp`, `drift_auc` y estadísticas del buffer, con una política de
retención configurable (no guardar todo indefinidamente).

## Opciones consideradas

1. **No guardar** — sin auditoría ni reproducibilidad.
2. **Guardar todos los snapshots** — crecimiento de disco ilimitado.
3. **Store con retención acotada** — guardar en cada refit y podar más allá de
   `max_snapshots`.

## Decisión

Opción 3. `SnapshotStore` (`src/detection/snapshots.py`) guarda por refit un
`model.pkl` (`{model, pca}`) + `metadata.json` (`timestamp`, `refit_count`,
`refit_reason`, `drift_auc`, `n_samples`, `buffer_mean`, `buffer_std`) en
`snapshot_NNNNNN/`, y poda los más antiguos manteniendo los últimos
`max_snapshots`. El guardado ocurre **fuera del hot-path** (tras el swap del
modelo, fuera del lock) y es best-effort (un fallo nunca interrumpe el scoring).

## Consecuencias

- **Positivas:** auditoría, debugging y reproducibilidad; uso de disco acotado;
  no impacta la latencia de scoring; opt-in por detector (se inyecta el store).
- **Negativas / costos:** IO en cada refit (mitigado: fuera del lock, poco
  frecuente); pickle de sklearn acopla la versión de la librería.
- **Riesgos y mitigaciones:** errores de IO atrapados y logueados a debug;
  retención configurable evita llenar el disco.

## Notas de implementación

- `src/detection/snapshots.py::SnapshotStore` (`save`, `list_snapshots`,
  `load_metadata`, `latest`, poda por `max_snapshots`).
- `src/detection/base.py`: `_save_snapshot` en la Fase B de `score()`.
- `src/pipeline.py`: stores en `data/snapshots/{audio,video}` (`SNAPSHOTS_DIR`).
- Tests: `tests/test_drift_refit.py` (guardado en refit, retención, metadata).
