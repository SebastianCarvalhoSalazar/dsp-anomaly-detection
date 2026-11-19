# ADR-0015: Destilación — retirar fusión multimodal, detector de video, doble horizonte, EMD y Streamlit

- **Estado:** Aceptado
- **Fecha:** 2026-11-19
- **Decisores:** mantenedor del proyecto
- **Fase:** plan de destilación ([`docs/plan-destilacion.md`](../plan-destilacion.md))
- **Reemplaza:** [ADR-0001](0001-late-fusion-multimodal-architecture.md), [ADR-0004](0004-score-calibration-historical-percentiles.md), [ADR-0005](0005-configurable-fusion-strategies.md), [ADR-0006](0006-no-pca-for-low-dim-video-features.md), [ADR-0007](0007-dual-horizon-fast-slow-models.md), [ADR-0012](0012-dashboard-pipeline-live-control-channel.md)
- **Completa:** [ADR-0013](0013-reemplazo-dashboard-streamlit-por-spa.md) (retiro de Streamlit)

## Contexto y problema

Tras [ADR-0014](0014-nucleo-de-deteccion-funcional.md) el núcleo (audio → detector → un evento
por anomalía → almacenamiento y búsqueda) funciona y está medido con `scripts/replay.py`. Quedaban
componentes que no intervenían en la decisión y costaban mantenimiento, dependencias y
superficie de API:

- **Fusión multimodal tardía** (detector de video, calibración por percentiles, 4 estrategias,
  control en vivo desde el dashboard): apagada por defecto y nunca validada con datos reales.
  El vector de video tenía features duplicadas (`motion_energy` = `total_foreground_area_ratio`)
  y la calibración por rangos disparaba a tasa fija (≈ 2–19 % de ventanas con ruido puro).
- **Doble horizonte** (modelos lentos): solo observabilidad.
- **EMD** (análisis offline por IMFs): sin uso para decidir ni para entender un evento.
- **Dashboard Streamlit**: duplicaba al SPA (ADR-0013).
- **"Fuente probable"**: el ranking multiplicaba por el score de audio, idéntico para todas las
  cajas de un frame, así que nunca relacionó movimiento con sonido.
- 13 dependencias directas sin uso o solo usadas por lo anterior.

## Opciones consideradas

1. **Mantener todo apagado por defecto** (estado tras ADR-0014) — conserva costo de
   mantenimiento, dependencias y una API/UI que sugieren capacidades no validadas.
2. **Retirar lo que no decide, conservando lo que da contexto** — menos código; la cámara
   sigue aportando frame y movimiento a cada evento.
3. **Retirar también la cámara** — pierde contexto útil en el historial y en la búsqueda.

## Decisión

Opción 2:

- Se eliminan `src/fusion/`, `src/vision_detection/`, los modelos lentos,
  `/internal/fusion-config` y sus controles en el SPA. La decisión de evento es
  `AnomalyResult.is_anomaly` del detector de audio.
- Se elimina el EMD; el espectrograma mel pasa a `GET /events/{id}/spectrogram` y se muestra en
  la tarjeta del evento. Desaparece la página "Análisis offline".
- Se elimina `src/dashboard/` (Streamlit).
- "Fuente probable" pasa a **"movimiento principal"**: `motion_score = area_ratio ×
  temporal_weight` (misma caja ganadora fuera del warmup). Contexto visual, sin relación con el
  audio.
- Se retiran 13 dependencias y el código sin uso (`AudioWindow`, `detect_with_result`,
  `MotionResult`, `draw_boxes`).

## Consecuencias

- **Positivas:** huella de detección (`flags_sha1`, `scores_sha1`) idéntica a la previa en todos
  los pasos; `poetry.lock` de 166 a 113 paquetes sin cambios de versión; API, WebSocket y SPA
  describen solo lo que existe. Resultados en [`docs/plan-destilacion.md`](../plan-destilacion.md).
- **Negativas / costos:**
  - Cambia el contrato: desaparecen `/internal/fusion-config`, `/events/{id}/offline_analysis` y
    los campos WS/REST de fusión, video y modelos lentos; la clave de las cajas pasa de
    `source_score` a `motion_score`.
  - Reintroducir una fusión audio + video exigiría rehacerla, idealmente validada con datos
    etiquetados.
- **Riesgos y mitigaciones:**
  - Bases de datos v0.3 conservan columnas de fusión/video: quedan nullable y sin mapear;
    lecturas e inserciones siguen funcionando (test dedicado).
  - Eventos guardados antes usan `source_score` en sus cajas: el frame anotado acepta ambas claves.

## Notas de implementación

- `src/pipeline.py`, `src/api/` (`main.py`, `schemas.py`, `routers/events.py`,
  `routers/websocket.py`), `src/storage/models.py` y `db.py`, `src/vision/`.
- SPA: `pages/LiveMonitor.tsx`, `components/events/EventCard.tsx`, `hooks/useSpectrogram.ts`,
  `hooks/useDetector.ts`, `components/common/BBoxChip.tsx`, `api/`.
- `zustand` queda sin uso en `web/package.json`; quitarlo exige regenerar `package-lock.json`
  contra el registro npm privado del proyecto.
