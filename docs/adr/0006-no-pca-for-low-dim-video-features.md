# ADR-0006: Sin PCA para features de video de baja dimensión

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 2

## Contexto y problema

El requisito describe el detector visual con "PCA + IsolationForest"
por simetría con el de audio. Pero el detector de audio usa PCA porque su
vector tiene **~251 dimensiones** (scattering + wavelets + espectrales +
deltas), donde reducir a ~25 componentes elimina redundancia y acelera el IF.

El vector de features de video tiene **~7 dimensiones** (`motion_energy`,
`bbox_count`, `largest_bbox_area_ratio`, `total_foreground_area_ratio`,
`mean_bbox_area_ratio`, `max_temporal_weight`, `average_temporal_weight`).

## Opciones consideradas

1. **PCA también en video** (simetría con audio) — sobre 7 dims no reduce
   prácticamente nada útil, añade un componente que puede *descartar* varianza
   discriminativa y complica el pipeline sin beneficio.
2. **IsolationForest directo sobre las features** — el IF maneja 7 dimensiones
   sin problema; la normalización Welford ya pone las features en escala
   comparable.

## Decisión

Opción 2. `VideoAnomalyDetector` reusa `BaseAnomalyDetector` con
`enable_pca=False`. La normalización Z-score se mantiene; el IsolationForest
consume las 7 features directamente.

## Consecuencias

- **Positivas:** menos partes móviles; sin pérdida de señal por reducción
  innecesaria; misma base correcta que audio (sin clonar bugs).
- **Negativas / costos:** rompe la simetría literal con el requisito (decisión
  técnica justificada y documentada aquí).
- **Riesgos y mitigaciones:** si en el futuro el vector de video crece mucho
  (p.ej. al sumar features de flujo óptico o histogramas), PCA puede
  reactivarse con un único flag (`enable_pca=True`) sin tocar el detector.

## Notas de implementación

- `src/vision_detection/detector.py::default_video_config` → `enable_pca=False`.
- `src/vision_detection/types.py::VIDEO_FEATURE_NAMES` define el layout (7 dims),
  inyectado como `feature_names` para drift/explicabilidad correctos.
- Test: `tests/test_video_detection.py::test_video_detector_has_no_pca`.
