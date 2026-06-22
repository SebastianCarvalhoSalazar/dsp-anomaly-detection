# ADR-0010: Explicabilidad por z-score vs baseline reciente

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 3

## Contexto y problema

El requisito pide explicar las anomalías mostrando los features que más
contribuyen (`top_audio_features`, `top_video_features`), **sin** intentar
interpretar coeficientes individuales de scattering (que no son legibles por
sí solos). Se necesita una explicación barata, online y por modalidad.

## Opciones consideradas

1. **SHAP / importancias del IsolationForest por instancia** — costoso, no
   trivial para IF, y para scattering los nombres no son interpretables.
2. **Desviación z-score por feature vs baseline reciente** — `z = (x - μ)/σ`
   usando la media/desv. del normalizador Welford; rankear por `|z|`.

## Decisión

Opción 2. `BaseAnomalyDetector.top_features(fv, k)` calcula el z-score de cada
feature contra el baseline reciente (media/σ del normalizador) y devuelve los
top-k formateados como `"<feature> ±X.Xσ"` (p.ej. `"spectral_centroid +4.2σ"`).
Reusa los `feature_names` inyectados (H6), así que las etiquetas son las reales
de cada modalidad. Durante el warmup devuelve lista vacía.

## Consecuencias

- **Positivas:** explicación barata (una resta/división por feature),
  interpretable, consistente entre modalidades; reusa estado ya disponible.
- **Negativas / costos:** el z-score es univariado (no captura interacciones
  entre features); es una aproximación de "qué cambió", no una atribución causal.
- **Riesgos y mitigaciones:** features altamente correlacionadas pueden
  co-aparecer en el top; aceptable para guía humana. Se calcula por ventana,
  con costo despreciable.

## Notas de implementación

- `src/detection/base.py::top_features` (+ accesores `is_ready`/`mean_std` en
  el normalizador).
- `src/pipeline.py`: `top_audio_features`/`top_video_features` en `mm`,
  expuestos en payload y persistidos (columnas `top_*_features`).
- `src/dashboard/pages/live_monitor.py`: caption "Top contributors".
- Tests: `tests/test_drift_refit.py` (warmup vacío, ranking por z-score).
