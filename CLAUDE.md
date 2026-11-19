# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Entorno (requiere la librería de sistema PortAudio: brew install portaudio / apt-get install libportaudio2)
poetry install                                              # instalar dependencias
poetry shell                                                # activar entorno

# Ejecutar el sistema (tres procesos independientes)
poetry run python -m src.pipeline                          # pipeline de captura + detección + eventos
poetry run uvicorn src.api.main:app --reload               # API FastAPI

# Dashboard SPA (React + TS + Vite)
cd web && npm install && npm run dev                       # dashboard (http://localhost:5173)
cd web && npm test                                         # tests del SPA (Vitest)
cd web && npm run build                                    # typecheck (tsc) + build de producción

# Tests
poetry run pytest                                          # todos los tests
poetry run pytest tests/test_foo.py::test_bar -x          # un test específico
poetry run python -m scripts.replay --scenario events      # banco offline con verdad conocida (también: quiet, --wav)

# Lint
poetry run pylint src/
```

## Architecture

El sistema corre como **tres procesos independientes**: pipeline (Python), API (FastAPI) y
dashboard (SPA en `web/`). La decisión de anomalía la toma **solo el detector de audio**; la
cámara aporta contexto (frame del evento, energía de movimiento). Decisiones vigentes:
ADR-0014 (núcleo corregido) y ADR-0015 (módulos retirados: fusión multimodal, detector de
video, doble horizonte, EMD, Streamlit).

### 1. Pipeline (`src/pipeline.py`)
Proceso síncrono. `process_window(window, ts)` contiene toda la lógica por ventana y es lo que
usan los tests y `scripts/replay.py` sin micrófono.

**Flujo por ventana de audio (4096 muestras @ 16 kHz = 0.256 s, sin solapamiento):**
1. `sounddevice` callback → cola `(muestras, capture_midpoint)` (maxsize=64). La hora es la de **captura** (`capture_midpoint`, vía `inputBufferAdcTime`), no la de salida de la cola.
2. `AudioProcessor.process_window()` → vector de 251 features (scattering Kymatio + energías wavelet db4 + RMS/ZCR + centroid/flatness/rolloff/bandwidth + deltas)
3. `AnomalyDetector.score()` → `AnomalyResult`
4. Frame más cercano a la hora de captura (`FrameRingBuffer.nearest`, `max_delta=0.2 s`; si no hay, sin frame) → `MotionDetector.detect()` (MOG2, merge de cajas, IoU → `temporal_weight` 1.0 nueva / 0.5 persistente)
5. **Movimiento principal:** `motion_score = area_ratio × temporal_weight`, se conserva el top-1; `motion_energy` = fracción del frame cubierta. Solo contexto visual: no correlaciona con el audio.
6. HTTP POST a `/internal/score` (score, umbral, estado, cajas, `motion_energy`, `rms`, `top_audio_features`, métricas de drift) → broadcast WebSocket
7. `EventSegmenter` agrupa ventanas con `is_anomaly` consecutivas en **un** evento (cierre tras `EVENT_RELEASE_S`=2 s sin anomalía o a los `EVENT_MAX_DURATION_S`=60 s; inicio, fin, duración, frame/score del pico). Al cerrarse, `EventWriter` (hilo aparte, cola acotada) persiste: archivos → SQLite → embedding → FAISS. El loop de audio nunca espera.

**Model preloading:** al inicio de `run()`, Wav2Vec2 + DINOv2 se cargan en un hilo en paralelo al warmup.

**Detector online (`src/detection/base.py`):** Isolation Forest con buffer deslizante de 500 ventanas **crudas**; refit cada 200 ventanas nuevas. Cada fit aprende `scaler (z-score) → PCA (251→25) → IF` y la cadena queda congelada hasta el siguiente. Umbral de decisión **fijo por fit** = percentil 2.0 de los scores del buffer de entrenamiento; se confirma con 3 ventanas consecutivas (hysteresis). Las ventanas anómalas quedan en **cuarentena** fuera del buffer; si persisten `accept_after_windows` (234 ≈ 60 s) se aceptan como nuevo normal (refit forzado, `refit_reason="accepted"`). `anomaly_score` se suaviza con EMA (α=0.3) solo para visualización; `adaptive_threshold` del WS es el umbral de decisión en escala [0,1]. Estado persistido en `data/detector_state.pkl` (versión 2; versiones anteriores se ignoran).

**Opcionales, apagados por defecto (no deciden eventos):** `ENABLE_DRIFT` (C2ST con `RandomForestClassifier` entre buffers sucesivos + refits sensibles a drift; el AUC tiene un sesgo conocido por buffers solapados) y `ENABLE_SNAPSHOTS` (guarda el modelo en cada refit).

### 2. API (`src/api/`)
FastAPI async. Expone datos almacenados y recibe notificaciones del pipeline.

- `POST /internal/score` — el pipeline notifica cada ventana; el router WebSocket la difunde
- `POST /internal/reset-detector` — activa `app.state.detector_reset_pending = True`
- `GET /internal/reset-pending` — el pipeline lo sondea cada 50 ventanas; devuelve `{"pending": bool}` y limpia el flag
- `GET /health` — `{"status":"ok"}`
- `GET /events/` — lista eventos (min_score, fecha, paginación); incluye `end_timestamp`, `duration_s`
- `GET /events/{id}/audio|frame` — stream de archivos
- `GET /events/{id}/frame/annotated` — frame con la caja de movimiento principal (rojo + label `motion X.XXX`; acepta la clave antigua `source_score`)
- `GET /events/{id}/spectrogram` — espectrograma mel en dB (librosa), bajo demanda
- `DELETE /events/{id}` — elimina fila + archivos y su vector FAISS
- `DELETE /events/` — elimina todo y llama `FAISSStore.clear()`
- `POST /search/similar` — audio o imagen (máx. 10 MB) → embedding → k vecinos en FAISS
- `GET /search/similar/by-event/{event_id}` — vecinos de un evento usando su embedding guardado (instantáneo; excluye el propio evento)
- `WS /ws/stream` — `AnomalyScoreMessage` en tiempo real

**CORS:** `create_app()` registra `CORSMiddleware` con orígenes de `CORS_ORIGINS` (default `http://localhost:5173`).

**Model preloading:** el lifespan carga los modelos de embedding en un hilo (`_get_encoder()`, double-checked locking).

`Database`, `FAISSStore` y `EventStore` se inyectan vía `request.app.state` (`dependencies.py`); startup con `lifespan`.

### 3. Dashboard (`web/`)
SPA en **React + TypeScript + Vite**, lenguaje visual "Mission Control" (tema oscuro, Chakra
Petch + JetBrains Mono). Tiempo real vía `useAnomalyStream` (WebSocket con reconexión
backoff+jitter, heartbeat, *staleness*, ring buffers); estado de servidor con TanStack Query;
gráficos uPlot (streaming) + Recharts (KDE/RMS) y `Heatmap` (canvas) para el espectrograma.
Tests con Vitest. Contrato con la API: `docs/frontend-migration/03-contrato-api-frontend.md`.

Tres páginas:
- **Monitor en vivo** (`pages/LiveMonitor.tsx`) — score y umbral, estado, motion energy, drift AUC, RMS, KDE, movimiento principal, top contributors de la última anomalía, reiniciar historial/detector
- **Eventos** (`pages/EventFeed.tsx`) — tarjetas con frame anotado, audio y espectrograma bajo demanda (`useSpectrogram`); eliminar uno o todos
- **Búsqueda** (`pages/SimilaritySearch.tsx`) — por archivo subido o por evento existente

---

## Módulos de procesamiento

| Módulo | Clase principal | Output |
|--------|----------------|--------|
| `src/dsp/` | `AudioProcessor` | vector de features np.float32 (251 dims con la config por defecto) |
| `src/detection/` | `AnomalyDetector` (`BaseAnomalyDetector`) | `AnomalyResult` (raw_score, anomaly_score, is_anomaly, is_fitted) |
| `src/events/` | `EventSegmenter`, `EventWriter` | `SegmentedEvent` (inicio, fin, pico, audio) |
| `src/embeddings/` | `MultimodalEncoder` | np.ndarray 1536-dim L2-normalizado |
| `src/vision/` | `MotionDetector`, `FrameCapture` | `BoundingBox` (`temporal_weight`, `motion_score`) |
| `src/sync/` | `FrameRingBuffer` | frame más cercano a una hora dada |
| `src/storage/` | `EventStore`, `Database`, `FAISSStore` | Persistencia híbrida filesystem+SQLite+FAISS |

**Embeddings:** `AudioEncoder` (Wav2Vec2, 768-dim) + `ImageEncoder` (DINOv2, 768-dim) → concatenados y L2-normalizados → 1536-dim. Carga lazy con offline-first: intenta `local_files_only=True`, si falla descarga una sola vez. Sin frame, la mitad visual es cero.

**FAISS:** `IndexIDMap2` sobre `IndexFlatIP` (coseno sobre vectores L2-normalizados). El ID de cada vector es la PK de SQLite; escritura atómica con file-lock entre procesos.

**Persistencia de eventos:** `eventos/<timestamp>/audio.wav` (pre-roll ~3 s + evento), `frame.jpg` (frame del pico), `embedding.npy`, `metadata.json`. Las columnas nuevas de SQLite se añaden por migración nullable (`_ADDED_COLUMNS`); las bases v0.3 pueden conservar columnas de fusión/video que el ORM ya no mapea.

---

## Convenciones críticas

- Los tests **no deben requerir micrófono ni cámara** — usar arrays numpy sintéticos (`Pipeline.process_window` + encoder mockeado)
- Cambios en `src/detection/`, `src/dsp/` o el flujo de `process_window` deben conservar la **huella de detección**: comparar `flags_sha1`/`scores_sha1` de `scripts/replay.py` antes y después **en la misma máquina** (`scores_sha1` varía entre plataformas por punto flotante; `flags_sha1` no). Ver `docs/plan-destilacion.md`
- `src/dsp/` y `src/vision/` son síncronos por diseño; no introducir `async` en esos módulos
- `pytest-asyncio` configurado con `asyncio_mode = "auto"` — los tests async no necesitan `@pytest.mark.asyncio`
- pylint desactiva C0114/C0115/C0116 (docstrings) — no añadir docstrings en módulos que no los tenían
- Variables de entorno en `.env` de la raíz (copiar desde `.env.example`), **cargado automáticamente** por `python-dotenv` en el pipeline y la API: `EVENTS_DIR`, `DB_PATH`, `FAISS_PATH`, `SNAPSHOTS_DIR`, `API_BASE_URL`, `CORS_ORIGINS`, `ENABLE_DRIFT`, `ENABLE_SNAPSHOTS`, `EVENT_RELEASE_S`, `EVENT_MAX_DURATION_S`. Una variable exportada en el shell tiene prioridad. El SPA usa su propio `web/.env` (`VITE_API_BASE_URL`).
- Las dependencias del SPA se resuelven contra el npm público (`web/.npmrc`); el lockfile no debe apuntar a registros privados. El CI (`.github/workflows/ci.yml`) corre pylint + pytest y `npm ci && npm test && npm run build`.

---

## Flujo de trabajo Git

El repo sigue **GitHub Flow + tags SemVer** (ver sección "Flujo de trabajo y ramas" en `README.md`):

- Rama troncal única: **`main`** (default branch). No commitear directo; todo entra por PR.
- Ramas cortas que nacen de `main` y vuelven por PR: `feature/<slug>`, `fix/<slug>`, `docs/<slug>`. Se borran tras el merge.
- **No** crear ramas por versión (`dev/vX.Y.Z`) — las versiones se marcan con tags `vMAYOR.MENOR.PARCHE[-pre]` sobre `main`.
- Las decisiones de arquitectura se documentan como ADRs en `docs/adr/` (formato MADR).

---

## Prompt de implementación

Usar este prompt al iniciar una sesión de desarrollo del MVP:

```
Actúa como un senior ML engineer con experiencia en DSP, sistemas multimodales
y arquitecturas en tiempo real. Analiza exhaustivamente doc/ para entender el
diseño y objetivos del proyecto: un sistema de detección de anomalías
audiovisuales en tiempo real combinando DSP (wavelets, Kymatio), modelos no
supervisados y embeddings (Wav2Vec 2.0, DINOv2).

Implementa el MVP en Python siguiendo este orden de módulos:
1. src/dsp/ + src/detection/ — pipeline de features y scoring de anomalías
2. src/storage/ + src/api/ — persistencia con FAISS y SQLite, API FastAPI
3. src/embeddings/ — embeddings multimodales (audio + imagen)
4. src/vision/ — captura y procesamiento de video
5. web/ — dashboard (SPA React + TypeScript; originalmente Streamlit)

El sistema debe ejecutarse completamente offline. Los modelos HuggingFace
(Wav2Vec 2.0, DINOv2) deben descargarse una sola vez y cargarse con
local_files_only=True. Usa FAISS (no Qdrant) y SQLite (no PostgreSQL) para
el MVP.

El código debe ser modular y claro. Documenta con docstrings las funciones
públicas y añade comentarios únicamente donde la lógica técnica no sea evidente
(elección de wavelet, hiperparámetros del detector, decisiones de arquitectura).

Después de implementar cada módulo, valida con pytest usando datos simulados
(no requieras micrófono/cámara en tests). Corrige cualquier error antes de
continuar al siguiente módulo.
```
