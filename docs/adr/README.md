# Architecture Decision Records (ADR)

Registro de decisiones de arquitectura del proyecto, en formato
[MADR](https://adr.github.io/madr/). Cada ADR captura una decisión
significativa: su contexto, las opciones evaluadas, la decisión tomada y
sus consecuencias. Usar [`0000-template.md`](0000-template.md) como base.

Las ADR 0001–0013 acompañaron la evolución a un detector multimodal de fusión
tardía (v0.3). [ADR-0014](0014-nucleo-de-deteccion-funcional.md) corrigió el
núcleo de detección y [ADR-0015](0015-destilacion-del-sistema.md) retiró los
componentes que no decidían eventos; las ADR reemplazadas se conservan como
registro histórico.

## Índice

| ADR | Título | Estado | Fase |
|-----|--------|--------|------|
| [0001](0001-late-fusion-multimodal-architecture.md) | Arquitectura multimodal de fusión tardía | Reemplazado (0015) | — |
| [0002](0002-shared-base-detector-refactor.md) | Refactor a `BaseAnomalyDetector` compartido (fix C1/C2/H1) | Aceptado · modificado (0014) | 0 |
| [0003](0003-timestamp-based-av-synchronization.md) | Sincronización audio-video por timestamp (ring buffer + nearest-frame) | Aceptado · modificado (0014) | 1 |
| [0004](0004-score-calibration-historical-percentiles.md) | Calibración de scores por percentiles históricos | Reemplazado (0015) | 2 |
| [0005](0005-configurable-fusion-strategies.md) | Estrategias de fusión configurables (patrón Strategy) | Reemplazado (0015) | 2 |
| [0006](0006-no-pca-for-low-dim-video-features.md) | Sin PCA para features de video de baja dimensión | Reemplazado (0015) | 2 |
| [0007](0007-dual-horizon-fast-slow-models.md) | Doble horizonte temporal (modelo rápido / lento) | Reemplazado (0015) | 3 |
| [0008](0008-drift-aware-adaptive-refit-policy.md) | Política de refit sensible a drift | Aceptado · opcional | 3 |
| [0009](0009-model-snapshots-and-retention.md) | Snapshots de modelos y retención | Aceptado · opcional | 3 |
| [0010](0010-explainability-via-zscore-baseline.md) | Explicabilidad por z-score vs baseline reciente | Aceptado · modificado (0014) | 3 |
| [0011](0011-backward-compatible-event-schema-migration.md) | Migración de esquema de eventos compatible hacia atrás | Aceptado | 1 |
| [0012](0012-dashboard-pipeline-live-control-channel.md) | Canal de control dashboard → pipeline en vivo | Reemplazado (0015) | post-v0.3 |
| [0013](0013-reemplazo-dashboard-streamlit-por-spa.md) | Reemplazo del dashboard Streamlit por un SPA (React + TS) | Aceptado · completado (0015) | Frontend |
| [0014](0014-nucleo-de-deteccion-funcional.md) | Núcleo de detección funcional: umbral estable, eventos segmentados, componentes opcionales | Aceptado | Plan de correcciones |
| [0015](0015-destilacion-del-sistema.md) | Destilación: retirar fusión, video, doble horizonte, EMD y Streamlit | Aceptado | Plan de destilación |

## Convenciones

- Numeración secuencial de 4 dígitos; nunca reusar números.
- Un ADR es inmutable una vez **Aceptado**: si cambia, se crea uno nuevo que
  lo *reemplaza* y se actualiza el estado del anterior.
- Enlazar los hallazgos del análisis (`Cx`/`Hx`/`Mx`) y los requisitos del
  plan que motivan cada decisión.
