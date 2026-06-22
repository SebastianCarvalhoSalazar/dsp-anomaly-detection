# ADR-0003: Sincronización audio-video por timestamp

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 1

## Contexto y problema

El pipeline correlacionaba cada ventana de audio con `self._latest_frame` —
*el último frame capturado* por el thread de cámara. Problemas:

- **No hay garantía temporal:** el último frame puede ser de hace decenas de
  ms o estar desfasado respecto a la ventana de audio que se está puntuando.
- **C5:** `_latest_frame` guardaba la referencia cruda de `cv2.VideoCapture.read()`
  sin copiar; algunos backends reutilizan ese buffer → posible lectura de un
  frame siendo sobrescrito.
- La fusión multimodal (Fase 2) exige que `audio_score` y `video_score`
  describan **el mismo instante**; sin alineación explícita, fusionar es ruido.

## Opciones consideradas

1. **Mantener "último frame"** — simple pero temporalmente impreciso y con la
   condición de carrera C5.
2. **Buffer circular de frames con timestamp + búsqueda del más cercano** —
   cada frame se etiqueta con su instante de captura; la ventana de audio se
   empareja con `nearest(timestamp)`.
3. **Cola sincronizada productor-consumidor 1:1** — exige tasas de audio/video
   acopladas; frágil ante drops y fps variable.

## Decisión

Opción 2. Nuevo paquete `src/sync/` con:
- `AudioWindow(start_timestamp, end_timestamp)` y `CapturedFrame(timestamp, frame)`.
- `FrameRingBuffer` thread-safe: el thread de cámara hace `push(frame.copy(), ts)`
  y el loop de procesamiento consulta `nearest(ts_window, max_delta=None)`.

El loop principal sella la ventana de audio con `ts_window = time.time()` al
sacarla de la cola y recupera el frame más cercano en el tiempo.

## Consecuencias

- **Positivas:** correlación fuente-sonido y embeddings sobre frames realmente
  alineados; base correcta para la fusión multimodal; `max_delta` permite
  declarar "no hay frame alineado" en vez de usar uno arbitrario; C5 resuelto
  (se almacena una copia).
- **Negativas / costos:** memoria del buffer (N frames recientes) y una búsqueda
  lineal por ventana (N pequeño, despreciable).
- **Riesgos y mitigaciones:** los timestamps usan el reloj del proceso, no el
  del hardware de captura; suficiente para alineación a escala de ventana
  (~256 ms). El thread de cámara ahora atrapa errores por iteración (no muere
  ante un `read()` transitorio, H7).

## Notas de implementación

- `src/sync/buffer.py`: `FrameRingBuffer`, `AudioWindow`, `CapturedFrame`.
- `src/pipeline.py`: se elimina `_latest_frame`/`_frame_lock`; el camera loop
  pushea copias con timestamp; el main loop usa `nearest(ts_window)`.
- Tests: `tests/test_sync.py` (más cercano, vacío, fuera de rango, `max_delta`,
  evicción, latest/clear).
