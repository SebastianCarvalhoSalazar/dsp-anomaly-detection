# ADR-0004: Calibración de scores por percentiles históricos

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 2

## Contexto y problema

Los scores crudos de los detectores de audio y video viven en **escalas
distintas y derivantes** (dependen del IsolationForest, del rango de
calibración de cada modalidad y del drift). Fusionarlos directamente
(`0.5·audio + 0.5·video`) es matemáticamente inválido: un 0.9 de audio no
significa lo mismo que un 0.9 de video. El requisito pide que `0.9 audio ≈
0.9 video` en términos relativos, y que la calibración se actualice sola.

## Opciones consideradas

1. **Min-max global fijo** — frágil ante drift; requiere conocer el rango a
   priori.
2. **Z-score sobre histórico** — comparable, pero sensible a outliers y asume
   normalidad, que no se cumple en scores de anomalía.
3. **Rango percentil sobre ventana deslizante** — mapea el score a su
   percentil empírico en el histórico reciente; robusto a outliers, sin
   supuestos de distribución, e interpretable ("más anómalo que el X% reciente").

## Decisión

Opción 3. `PercentileCalibrator` mantiene una ventana deslizante de scores
crudos por modalidad y `calibrate(score)` devuelve el rango percentil en
`[0,1]`. Durante el warmup (menos de `min_samples`) hace passthrough clampeado.
Se actualiza de forma continua (`calibrate_and_update`), no solo en refits, lo
que lo hace más responsivo al drift.

## Consecuencias

- **Positivas:** audio y video quedan en una escala común y autointerpretable;
  robusto a outliers y a drift; thread-safe; sin dependencias nuevas.
- **Negativas / costos:** ordena la ventana por llamada (O(n log n), n=500 →
  despreciable a cadencia de audio); el significado es *relativo a la ventana*,
  no absoluto.
- **Riesgos y mitigaciones:** ventana corta → calibración ruidosa; configurable
  vía `window`/`min_samples`. Distribuciones casi constantes → rangos saltan
  entre 0 y 1; aceptable porque indica ausencia de variación.

## Notas de implementación

- `src/fusion/calibration.py::PercentileCalibrator`.
- Pipeline: un calibrador por modalidad; `audio_score`/`video_score` expuestos
  ya calibrados; `anomaly_score` (audio crudo) se mantiene por compatibilidad.
- Tests: `tests/test_calibration.py` (passthrough, rango percentil, shift dinámico).
