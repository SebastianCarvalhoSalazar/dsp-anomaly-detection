# ADR-0005: Estrategias de fusión configurables (patrón Strategy)

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 2

## Contexto y problema

La fusión tardía ([ADR-0001](0001-late-fusion-multimodal-architecture.md))
requiere combinar los scores calibrados de audio y video en un
`combined_score` y una decisión. El requisito pide cuatro estrategias
(Weighted Average, Maximum, AND, OR) y dejar la puerta abierta a estrategias
futuras (p.ej. fusión aprendida) sin refactor.

Además hay que preservar el comportamiento actual: el gating de eventos no
debe regresionar por introducir la fusión.

## Opciones consideradas

1. **`if/elif` por estrategia en el pipeline** — simple pero rígido; agregar
   estrategias toca el pipeline y dificulta testear en aislamiento.
2. **Patrón Strategy** — cada estrategia es una clase con `combine(audio,
   video) -> FusionResult`; un registro permite instanciar por nombre.

## Decisión

Opción 2. `FusionStrategy` (ABC) + `WeightedAverage`/`Maximum`/`AndStrategy`/
`OrStrategy`, todas operando sobre scores **calibrados** en `[0,1]` con un
`threshold` único y consistente. `make_strategy(name, **kwargs)` resuelve por
nombre. `FusionResult` lleva `combined_score`, `is_anomaly` y
`dominant_modality` (`audio-driven`/`video-driven`/`multimodal`, con banda de
empate `_DOMINANCE_EPS`).

**Decisión de gating (preservación de comportamiento):** en esta fase el
pipeline sigue *gateando* el guardado de eventos con `is_anomaly` del detector
de audio (la ruta rápida y establecida). El `combined_score` de la fusión se
calibra, expone y persiste para observabilidad; el dashboard recomputa la
fusión en vivo (slider de peso + selector de estrategia) a partir de los scores
por modalidad recibidos por WebSocket, sin round-trip al pipeline. Promover la
fusión a decisión de gating es un follow-up configurable.

## Consecuencias

- **Positivas:** estrategias testeables en aislamiento; agregar una nueva no
  toca el pipeline; el dashboard reusa el mismo módulo para recomputar
  client-side; sin regresión del gating actual.
- **Negativas / costos:** la decisión final aún no usa la fusión (intencional
  en esta fase); dos lugares conocen la fusión (pipeline default + dashboard
  interactivo), mitigado al compartir `src/fusion`.
- **Riesgos y mitigaciones:** umbral en escala calibrada (percentil) — un
  `threshold=0.9` significa "top 10% reciente"; documentado y configurable.

## Notas de implementación

- `src/fusion/strategies.py`: estrategias + `FusionResult` + `make_strategy`.
- `src/pipeline.py`: default `WeightedAverage(audio_weight=0.5)`; expone
  `audio_score`/`video_score`/`combined_score`/`dominant_modality`.
- `src/dashboard/pages/live_monitor.py`: selector + slider recomputan en vivo.
- Tests: `tests/test_fusion.py` (weighted/max/and/or, dominante, registro).
