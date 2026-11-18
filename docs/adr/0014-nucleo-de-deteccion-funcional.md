# ADR-0014: Núcleo de detección funcional — umbral estable, eventos segmentados y componentes opcionales

- **Estado:** Aceptado
- **Fecha:** 2026-11-18
- **Decisores:** mantenedor del proyecto
- **Fase:** plan de correcciones ([`docs/plan-correcciones.md`](../plan-correcciones.md))
- **Modifica:** [ADR-0002](0002-shared-base-detector-refactor.md) (fix C1), [ADR-0003](0003-timestamp-based-av-synchronization.md), [ADR-0004](0004-score-calibration-historical-percentiles.md), [ADR-0008](0008-drift-aware-adaptive-refit-policy.md), [ADR-0009](0009-model-snapshots-and-retention.md)

## Contexto y problema

Una revisión con datos sintéticos de verdad conocida (`scripts/replay.py`) mostró
que el núcleo no cumplía su propósito —detectar anomalías acústicas y guardarlas
como eventos útiles—:

- **F1** — Las 30 primeras ventanas del warmup entraban sin normalizar al buffer de
  entrenamiento (normalizador Welford aún sin datos); la PC1 del PCA explicaba
  ~100 % de la varianza.
- **F2** — El umbral era el percentil 2 de los scores *en vivo*: toda anomalía de
  más de ~10 ventanas (2 % de 500) se absorbía. Un tono de 15 s se detectaba en el
  14–18 % de sus ventanas.
- **F3** — El calibrador por percentiles llevaba los empates a rango 1.0: durante el
  warmup (scores = 0) la fusión reportaba `combined_score = 1.0`, `is_anomaly = True`.
- **F4** — La hora de cada ventana se tomaba al sacarla de la cola, no al capturarla.
- **F5** — Se guardaba un evento por **ventana** anómala (21 eventos para 7 sonidos
  en el escenario de prueba) y los embeddings corrían dentro del loop de audio.

Además, la detección de drift (C2ST), los refits por drift, los snapshots y el
detector de video calculaban en cada ventana resultados que no intervienen en la
decisión por defecto.

## Opciones consideradas

1. **Parchear cada síntoma** (p. ej. subir `adaptive_score_window`, cooldown fijo
   tras cada evento) — no corrige las causas (umbral relativo a datos que incluyen
   la anomalía; buffer contaminado).
2. **Reescribir el núcleo** — pierde almacenamiento, API, SPA y tests que funcionan.
3. **Corregir las causas en el código existente y apagar lo que no decide** —
   cambios localizados, cada uno con un test que reproduce el fallo.

## Decisión

Opción 3:

- **Cadena congelada por fit (F1).** El buffer guarda vectores crudos; cada fit
  aprende `scaler → PCA → IsolationForest` y se usa sin cambios hasta el siguiente.
  Se elimina el normalizador Welford (y `freeze_normalizer_on_anomaly`).
- **Umbral fijo por fit + cuarentena (F2).** Umbral = percentil
  `adaptive_percentile` de los scores del **buffer de entrenamiento**. Las ventanas
  anómalas no entran al buffer; si una anomalía persiste `accept_after_windows`
  (234 ≈ 60 s) se acepta como nuevo normal y se fuerza un refit.
- **Rango medio y fusión neutra en warmup (F3).**
- **Hora de captura (F4).** El callback encola `(muestras, punto medio de captura)`
  calculado con `inputBufferAdcTime`; el frame se busca con
  `nearest(ts, max_delta=0.2 s)` y, si no hay uno cercano, el evento no lleva frame.
- **Un evento por anomalía (F5).** `EventSegmenter` (IDLE → ACTIVE → IDLE): se cierra
  tras `EVENT_RELEASE_S` (2 s) sin anomalía o al llegar a `EVENT_MAX_DURATION_S`
  (60 s); guarda inicio, fin, duración y el frame/score del pico. `EventWriter`
  persiste y calcula embeddings en un hilo con cola acotada.
- **Opcionales, apagados por defecto:** `ENABLE_DRIFT` (C2ST + refits por drift),
  `ENABLE_SNAPSHOTS`, `ENABLE_VIDEO_DETECTOR` (detector de video + calibración +
  fusión). Ya lo estaban `FUSION_GATES_DECISION` y `ENABLE_SLOW_MODELS`.

## Consecuencias

- **Positivas:** ver resultados antes/después en
  [`docs/plan-correcciones.md`](../plan-correcciones.md). Menos cómputo por ventana
  y menos escrituras a disco por defecto.
- **Negativas / costos:**
  - Los `detector_state.pkl` anteriores (versión 1) se ignoran: un warmup al
    actualizar.
  - Con la cuarentena, un cambio permanente del entorno sigue alarmando hasta
    `accept_after_windows`; es deliberado (ajustable).
  - Cuarentena y refits por drift tiran en direcciones opuestas (resistir el cambio
    vs adaptarse rápido); por eso el drift queda opcional.
  - El refit programado ocurre en la ventana siguiente a la que completa
    `refit_every` muestras nuevas.
- **Riesgos y mitigaciones:** el C2ST tiene un sesgo conocido por buffers solapados
  (confirmado: AUC 0.99 → 0.52 con drift sostenido); se documenta y queda opcional.

## Notas de implementación

- `src/detection/base.py`, `src/detection/config.py` (F1/F2)
- `src/fusion/calibration.py`, `src/pipeline.py` (F3)
- `src/pipeline.py`: `capture_midpoint()`, `FRAME_MAX_DELTA_S` (F4)
- `src/events/` (`EventSegmenter`, `EventWriter`), `Pipeline.flush_events()`;
  columnas nullable `end_timestamp`, `duration_s` vía la migración de ADR-0011 (F5)
- Tests: `tests/test_detection_regressions.py`, `tests/test_events.py`,
  `tests/test_pipeline.py` (F3–F6), `tests/test_calibration.py`
- Banco: `poetry run python -m scripts.replay --scenario events|quiet`
