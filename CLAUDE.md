# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Entorno
poetry install                                              # instalar dependencias
poetry shell                                                # activar entorno

# Ejecutar el sistema (tres procesos independientes)
poetry run python -m src.pipeline                          # pipeline de captura + detección
poetry run uvicorn src.api.main:app --reload               # API FastAPI
poetry run streamlit run src/dashboard/app.py              # dashboard

# Tests
poetry run pytest                                          # todos los tests
poetry run pytest tests/test_foo.py::test_bar -x          # un test específico

# Lint
poetry run pylint src/
```

## Architecture

El sistema corre como **tres procesos Python independientes**:

### 1. Pipeline (`src/pipeline.py`)
Proceso principal síncrono. Orquesta todos los módulos de procesamiento.

**Flujo interno por ventana de audio:**
1. `sounddevice` callback → cola de audio (maxsize=64)
2. `AudioProcessor.process_window()` → vector de 134 features (126 scattering Kymatio + 6 wavelet db4 energías + RMS + ZCR)
3. `AnomalyDetector.score()` → `AnomalyResult` con score normalizado [0,1]
4. `MotionDetector.detect()` sobre el frame más reciente (thread separado para cámara)
5. HTTP POST a `/internal/score` (fire-and-forget, notifica al API para broadcast WebSocket)
6. Si `is_anomaly=True` → `_handle_anomaly()`: guarda evento en filesystem + genera embeddings multimodales + indexa en FAISS + persiste en SQLite

**Detector online:** `AnomalyDetector` usa Isolation Forest con buffer deslizante de 200 ventanas (~25s). Se re-entrena cada 100 ventanas nuevas. Durante warmup devuelve `is_fitted=False`.

### 2. API (`src/api/`)
FastAPI async. Expone datos almacenados y recibe notificaciones del pipeline.

- `POST /internal/score` — pipeline notifica scores; el router WebSocket los broadcast a todos los clientes conectados
- `GET /events/` — lista eventos con filtros (min_score, fecha, paginación)
- `GET /events/{id}/audio|frame` — stream de archivos desde filesystem
- `GET /events/{id}/offline_analysis` — corre EMD (PyEMD) + mel-spectrogram (librosa) sobre el audio del evento
- `POST /search/similar` — recibe audio o imagen, genera embedding con `MultimodalEncoder`, busca k vecinos en FAISS, retorna metadata desde SQLite
- `WS /ws/stream` — WebSocket para el dashboard en tiempo real

Instancias de `Database`, `FAISSStore` y `EventStore` se inyectan vía `request.app.state` (dependency injection en `dependencies.py`).

### 3. Dashboard (`src/dashboard/`)
Streamlit app que consume el API vía `APIClient` (httpx síncrono).

Cuatro páginas: **Live Monitor** (WebSocket + waveform en tiempo real), **Event Feed** (grid con audio/frame reproducibles), **Similarity Search** (upload de query), **Offline Analysis** (IMFs de EMD + mel-spectrogram con Plotly).

---

## Módulos de procesamiento

| Módulo | Clase principal | Output |
|--------|----------------|--------|
| `src/dsp/` | `AudioProcessor` | `FeatureVector` (134-dim np.float32) |
| `src/detection/` | `AnomalyDetector` | `AnomalyResult` (score, is_anomaly, is_fitted) |
| `src/embeddings/` | `MultimodalEncoder` | np.ndarray 1536-dim L2-normalizado |
| `src/vision/` | `MotionDetector`, `FrameCapture` | `MotionResult` (boxes, annotated_frame) |
| `src/storage/` | `EventStore`, `Database`, `FAISSStore` | Persistencia híbrida filesystem+SQLite+FAISS |

**Embeddings:** `AudioEncoder` (Wav2Vec2, 768-dim) + `ImageEncoder` (DINOv2, 768-dim) → concatenados y L2-normalizados → 1536-dim. Carga lazy con offline-first: intenta `local_files_only=True`, si falla descarga una sola vez.

**FAISS:** `IndexFlatIP` (inner product = cosine sobre vectores L2-normalizados). IDs secuenciales ligados a `AnomalyEvent.faiss_index_id` en SQLite.

**Persistencia de eventos:** `eventos/<timestamp>/audio.wav`, `frame.jpg`, `embedding.npy`, `metadata.json`.

---

## Convenciones críticas

- Los tests **no deben requerir micrófono ni cámara** — usar arrays numpy sintéticos
- `src/dsp/` y `src/vision/` son síncronos por diseño; no introducir `async` en esos módulos
- `pytest-asyncio` configurado con `asyncio_mode = "auto"` — los tests async no necesitan `@pytest.mark.asyncio`
- pylint desactiva C0114/C0115/C0116 (docstrings) — no añadir docstrings en módulos que no los tenían
- Variables de entorno en `.env` (copiar desde `.env.example`): `DATABASE_URL`, `QDRANT_HOST`, `EVENTS_DIR`, `API_PORT`

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
5. src/dashboard/ — dashboard Streamlit

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
