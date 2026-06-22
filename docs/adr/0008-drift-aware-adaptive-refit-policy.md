# ADR-0008: Política de refit sensible a drift

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 3

## Contexto y problema

El refit era de intervalo fijo (`refit_every`). Cuando el entorno cambia
(drift alto medido por el C2ST), conviene re-entrenar antes para adaptarse;
con el entorno estable, refits frecuentes son costo desperdiciado. El
requisito pide permitir refit anticipado bajo drift y **registrar
explícitamente** cuándo un refit fue provocado por drift.

## Opciones consideradas

1. **Intervalo fijo** (status quo) — simple, pero lento para adaptarse a drift.
2. **Intervalo adaptativo según `drift_auc`** — si `drift_auc >= umbral`,
   acortar el intervalo efectivo (`refit_every * factor`, con piso
   `min_refit_interval`); etiquetar el refit como `"drift"`.

## Decisión

Opción 2, en `BaseAnomalyDetector`. Banderas: `enable_drift_aware_refit`
(default off para preservar comportamiento), `drift_refit_threshold` (0.8),
`drift_refit_factor` (0.5) y `min_refit_interval` (50). El motivo del último
fit (`"initial"` | `"scheduled"` | `"drift"`) se rastrea en `_refit_reason`,
se persiste, se expone en `get_drift_metrics` y se guarda en cada snapshot.

## Consecuencias

- **Positivas:** adaptación más rápida ante cambios reales del entorno;
  auditable (se sabe *por qué* se reentrenó); configurable y desactivable.
- **Negativas / costos:** más refits bajo drift sostenido → más cómputo
  (acotado por `min_refit_interval`); el C2ST entre buffers consecutivos solo
  detecta drift mientras el buffer no se haya renovado del todo.
- **Riesgos y mitigaciones:** oscilación de frecuencia de refit → el piso
  `min_refit_interval` evita refits degenerados; el fit corre fuera del lock (H1).

## Notas de implementación

- `src/detection/config.py`: nuevas banderas drift-aware.
- `src/detection/base.py`: decisión de intervalo en la Fase A de `score()`;
  `_refit_reason`/`_pending_refit_reason`; `refit_reason` en métricas y estado.
- `src/api/schemas.py`: `refit_reason` en `AnomalyScoreMessage`.
- `src/pipeline.py`: audio detector con `enable_drift_aware_refit=True`.
- Test: `tests/test_drift_refit.py::test_drift_aware_refits_more_often_than_fixed`.
