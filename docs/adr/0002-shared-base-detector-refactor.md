# ADR-0002: Refactor a `BaseAnomalyDetector` compartido (fix C1/C2/H1)

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 0

## Contexto y problema

El requisito de un detector de video pide "PCA + IsolationForest + warmup +
buffer + refits independientes" — es decir, **el mismo mecanismo** que el
detector de audio. Copiar `AnomalyDetector` duplicaría tres bugs detectados en
el análisis de v0.2.0:

- **C1** — el normalizador Welford se actualizaba en *cada* `score()`, incluso
  durante anomalías, contaminando media/varianza y desensibilizando el detector
  (train/serve skew creciente).
- **C2** — `score_min/score_max` se calibraban sobre el buffer de entrenamiento,
  así que puntos más anómalos que cualquiera visto en training saturaban todos
  en `1.0` y se volvían indistinguibles (perdiendo poder discriminativo, que
  además alimenta el ranking de fuente).
- **H1** — `_fit()` (PCA + IsolationForest + C2ST con RandomForest) corría
  **dentro del lock** en el hot-path de audio, bloqueando el scoring.

## Opciones consideradas

1. **Copiar el detector actual** para video (rápido, pero duplica C1/C2/H1 y el
   costo de mantenerlos).
2. **Extraer una base correcta compartida** (`BaseAnomalyDetector`) y arreglar
   los bugs una sola vez; audio y video pasan a ser subclases finas.

## Decisión

Opción 2. Se crea `src/detection/base.py::BaseAnomalyDetector` con toda la
mecánica modalidad-agnóstica. `AnomalyDetector` (audio) queda como subclase
fina; `VideoAnomalyDetector` (Fase 2) reusará la misma base.

Correcciones incorporadas:
- **C1:** una vez fitteado, el normalizador **no** se actualiza con ventanas
  marcadas como anomalía cruda (`freeze_normalizer_on_anomaly`, default `True`).
  El drift benigno se sigue absorbiendo.
- **C2:** la cota inferior de calibración se ensancha por
  `calibration_margin * (max - min)` (default `0.5`), dejando headroom para
  anomalías peores que las de training. Solo afecta el score de display `[0,1]`;
  las *decisiones* siguen usando el score crudo vs el umbral.
- **H1:** el fit se computa **fuera del lock** (`_compute_fit`, puro) sobre un
  snapshot del buffer y se intercambia atómicamente (`_apply_fit`) bajo el lock.
  El primer fit sigue siendo síncrono desde la vista del caller (determinismo de
  tests); los refits no bloquean lecturas concurrentes.

## Consecuencias

- **Positivas:** un solo punto de verdad para la lógica de detección; el detector
  de video nace sin los bugs; mejor latencia en refit; estado de persistencia
  completo (M3) y validación de dimensión (M4); nombres de features inyectables
  (H6) eliminando el layout hardcodeado.
- **Negativas / costos:** cambio estructural amplio en el módulo de detección;
  riesgo de regresión mitigado por la suite existente (198 tests) + nuevos tests
  de regresión (`tests/test_base_detector.py`).
- **Riesgos y mitigaciones:** la guarda `_fitting` evita refits concurrentes
  duplicados; los flags nuevos tienen defaults que preservan el comportamiento
  observable (salvo las correcciones intencionales).

## Notas de implementación

- `src/detection/base.py` (nuevo): `BaseAnomalyDetector`, `_WelfordNormalizer`.
- `src/detection/detector.py`: `AnomalyDetector(BaseAnomalyDetector)` fino;
  conserva el staticmethod `_build_feature_names` (usado por tests).
- `src/detection/config.py`: `freeze_normalizer_on_anomaly`, `calibration_margin`.
- `src/dsp/processor.py`: nueva propiedad `feature_names` (layout real, H6).
- `src/pipeline.py`: inyecta `feature_names=self.dsp.feature_names` al detector.
- Compatibilidad: `calibration_margin=0.0` + `freeze_normalizer_on_anomaly=False`
  restauran el comportamiento legacy exacto.
