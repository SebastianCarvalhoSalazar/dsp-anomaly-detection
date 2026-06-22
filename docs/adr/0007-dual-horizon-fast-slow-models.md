# ADR-0007: Doble horizonte temporal (modelo rápido / lento)

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 3

## Contexto y problema

El detector aprende solo del buffer reciente (~500 ventanas ≈ "estado
reciente"). El requisito pide un segundo horizonte "lento" (buffer grande,
p.ej. 10000) que refleje el comportamiento histórico, generando
`fast_*`/`slow_*` por modalidad. La decisión final puede seguir usando el
modelo rápido; los scores lentos se almacenan/exponen para observabilidad.

El costo es real: un modelo lento por modalidad duplica detectores y, con
buffers grandes, encarece el refit (que ya es el punto caliente, ver H1).

## Opciones consideradas

1. **Construir los 4 modelos (fast+slow × audio+video) siempre activos** —
   máximo costo de cómputo impuesto por defecto, riesgo de latencia real-time.
2. **Modelos lentos opt-in** — reusar `BaseAnomalyDetector` con buffer grande y
   refit poco frecuente; activables por configuración; decisión final usa fast.
3. **No implementarlos aún** — incumple el requisito.

## Decisión

Opción 2. Los modelos lentos (`slow_detector`, `slow_video_detector`) son
instancias de los mismos detectores con `buffer_size` grande y `refit_every`
alto, **opt-in vía `ENABLE_SLOW_MODELS`** (default off). Los campos
`fast_audio_score`/`slow_audio_score`/`fast_video_score`/`slow_video_score`
se exponen siempre (en payload, persistencia y dashboard); con los lentos
desactivados, `fast_* = score del detector rápido` y `slow_* = 0.0`. La
decisión de gating de eventos sigue usando el detector rápido de audio.

## Consecuencias

- **Positivas:** estructura y campos listos sin imponer el costo por defecto;
  reusa la base correcta (sin clonar lógica); activación trivial por env var.
- **Negativas / costos:** con los lentos activos, +2 detectores y refits sobre
  buffers grandes (mitigado por `refit_every` alto y `enable_drift_detection=False`
  en los lentos); los lentos aún no influyen en la decisión (intencional).
- **Riesgos y mitigaciones:** refit del modelo lento bajo H1 (fit fuera del
  lock) evita bloquear el scoring; buffers configurables para acotar memoria
  (10000×251 float32 ≈ 10 MB).

## Notas de implementación

- `src/pipeline.py`: `_enable_slow`, `slow_detector`, `slow_video_detector`;
  scores `fast_*`/`slow_*` propagados vía el dict `mm`.
- Campos en `AnomalyScoreMessage` y columnas en `AnomalyEvent` (Fase 1).
