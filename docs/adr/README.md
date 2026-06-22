# Architecture Decision Records (ADR)

Registro de decisiones de arquitectura del proyecto, en formato
[MADR](https://adr.github.io/madr/). Cada ADR captura una decisión
significativa: su contexto, las opciones evaluadas, la decisión tomada y
sus consecuencias. Usar [`0000-template.md`](0000-template.md) como base.

Estas decisiones acompañan la evolución a un **detector multimodal de
fusión tardía** (audio + video) descrita en el plan
`feature/multimodal-fusion-drift-aware`.

## Índice

| ADR | Título | Estado | Fase |
|-----|--------|--------|------|
| [0001](0001-late-fusion-multimodal-architecture.md) | Arquitectura multimodal de fusión tardía | Aceptado | — |
| [0002](0002-shared-base-detector-refactor.md) | Refactor a `BaseAnomalyDetector` compartido (fix C1/C2/H1) | Aceptado | 0 |
| [0003](0003-timestamp-based-av-synchronization.md) | Sincronización audio-video por timestamp (ring buffer + nearest-frame) | Aceptado | 1 |
| [0004](0004-score-calibration-historical-percentiles.md) | Calibración de scores por percentiles históricos | Aceptado | 2 |
| [0005](0005-configurable-fusion-strategies.md) | Estrategias de fusión configurables (patrón Strategy) | Aceptado | 2 |
| [0006](0006-no-pca-for-low-dim-video-features.md) | Sin PCA para features de video de baja dimensión | Aceptado | 2 |
| [0007](0007-dual-horizon-fast-slow-models.md) | Doble horizonte temporal (modelo rápido / lento) | Aceptado | 3 |
| [0008](0008-drift-aware-adaptive-refit-policy.md) | Política de refit sensible a drift | Aceptado | 3 |
| [0009](0009-model-snapshots-and-retention.md) | Snapshots de modelos y retención | Aceptado | 3 |
| [0010](0010-explainability-via-zscore-baseline.md) | Explicabilidad por z-score vs baseline reciente | Aceptado | 3 |
| [0011](0011-backward-compatible-event-schema-migration.md) | Migración de esquema de eventos compatible hacia atrás | Aceptado | 1 |

## Convenciones

- Numeración secuencial de 4 dígitos; nunca reusar números.
- Un ADR es inmutable una vez **Aceptado**: si cambia, se crea uno nuevo que
  lo *reemplaza* y se actualiza el estado del anterior.
- Enlazar los hallazgos del análisis (`Cx`/`Hx`/`Mx`) y los requisitos del
  plan que motivan cada decisión.
