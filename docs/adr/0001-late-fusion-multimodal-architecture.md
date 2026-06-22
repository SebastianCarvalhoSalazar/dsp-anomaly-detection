# ADR-0001: Arquitectura multimodal de fusión tardía

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** — (decisión transversal)

## Contexto y problema

El sistema actual es **unimodal de audio**:
`AudioProcessor → (z-score) → PCA → IsolationForest → AnomalyResult`. El video
solo se usa para correlación de fuente (bounding boxes) y embeddings. El
requerimiento pide convertirlo en un **detector multimodal real**: detección
independiente de audio y video, fusión configurable, calibración, sincronización
temporal y explicabilidad — diseñado para extenderse luego con un detector
semántico (DINOv2), modelos secuenciales y feedback humano sin rediseño mayor.

La pregunta central es **cómo combinar las modalidades**.

## Opciones consideradas

1. **Fusión temprana (early fusion):** concatenar features de audio y video en
   un único vector y entrenar un solo modelo. Simple, pero acopla las
   modalidades, exige sincronización perfecta por ventana, dificulta explicar
   qué modalidad disparó la anomalía y no permite pesos/estrategias por modalidad.
2. **Fusión tardía (late fusion):** un detector independiente por modalidad,
   cada uno con su propio score; una capa de calibración los lleva a escalas
   comparables y una capa de fusión configurable produce el `combined_score`.
3. **Fusión intermedia (modelo conjunto sobre embeddings):** requiere un modelo
   entrenable y datos etiquetados; fuera de alcance para un sistema no
   supervisado online.

## Decisión

Adoptar **fusión tardía** (opción 2). Cada modalidad tiene su propio detector
(`AnomalyDetector` de audio, `VideoAnomalyDetector`), sus scores se calibran y
se fusionan con una estrategia configurable (`WeightedAverage`/`Max`/`AND`/`OR`).

## Consecuencias

- **Positivas:** modalidades desacopladas; se puede agregar un tercer detector
  (DINOv2) o un modelo secuencial como "otra fuente de score" sin tocar las
  existentes; explicabilidad y `dominant_modality` naturales; tolerante a
  ausencia temporal de una modalidad (p.ej. sin frame).
- **Negativas / costos:** requiere una capa de calibración explícita (los scores
  crudos no son comparables) y sincronización temporal entre modalidades.
- **Riesgos y mitigaciones:** scores mal calibrados sesgan la fusión → se
  aborda en [ADR-0004](0004-score-calibration-historical-percentiles.md);
  desalineación temporal → [ADR-0003](0003-timestamp-based-av-synchronization.md).

## Notas de implementación

- Nuevos paquetes: `src/vision_detection/` (detector de video) y `src/fusion/`
  (calibración + estrategias).
- Se conserva intacto el detector de audio (Scattering, wavelets, RMS, ZCR,
  features espectrales, PCA, IsolationForest, C2ST).
- Los scores individuales (`audio_score`, `video_score`, `combined_score`) se
  exponen en API/WS/persistencia; nunca se descartan.
