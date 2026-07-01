# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Entorno
poetry install                                              # instalar dependencias
poetry shell                                                # activar entorno

# Ejecutar el sistema (tres procesos independientes)
poetry run python -m src.pipeline                          # pipeline de captura + detección
ENABLE_SLOW_MODELS=true poetry run python -m src.pipeline  # pipeline con doble horizonte (rápido + lento)
poetry run uvicorn src.api.main:app --reload               # API FastAPI
poetry run streamlit run src/dashboard/app.py              # dashboard Streamlit (legacy, en decomisión)

# Dashboard SPA (React + TS + Vite) — reemplaza a Streamlit; ver docs/frontend-migration/
cd web && npm install && npm run dev                       # dashboard SPA (http://localhost:5173)
cd web && npm test                                         # tests del SPA (Vitest)
cd web && npm run build                                    # typecheck (tsc) + build de producción

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
2. `AudioProcessor.process_window()` → vector de features dinámico (scattering Kymatio + wavelet db4 energías + temporal RMS/ZCR + spectral centroid/flatness/rolloff/bandwidth + delta features)
3. `AnomalyDetector.score()` → `AnomalyResult` con Z-score normalizado, EMA-smoothed, con hysteresis
4. `MotionDetector.detect()` sobre el frame más reciente (thread separado para cámara) → merge de cajas cercanas (`merge_gap`, `max_box_ratio`), IoU tracking contra frame previo y asignación de temporal weights (nuevo=1.0, persistente=0.5)
5. **Source scoring:** para cada box, calcula `source_score = anomaly_score × area_ratio × temporal_weight` y conserva solo el **top-1** (la caja más probable de ser la fuente de la anomalía acústica)
6. Cross-modal: calcula `motion_energy` (ratio de bounding-box area / frame area)
7. HTTP POST a `/internal/score` con `motion_energy` incluido (fire-and-forget, notifica al API para broadcast WebSocket)
8. Si `is_anomaly=True` → `_handle_anomaly()`: guarda evento con `motion_energy` y `source_score` en metadata → genera embeddings multimodales → indexa en FAISS → persiste en SQLite

**Model preloading:** Al inicio de `run()`, los modelos Wav2Vec2+DINOv2 se precargan en un background thread paralelo al warmup, eliminando la penalización de ~60s en la primera anomalía.

**Detector online:** `AnomalyDetector` usa Isolation Forest con buffer deslizante de 500 ventanas. Se re-entrena cada 200 ventanas nuevas. Features son Z-score normalizados (Welford online), luego reducidos de 251 a **25 dimensiones con PCA** antes de alimentar el IF. Scores pasan por EMA smoothing (α=0.3) y requieren 3 ventanas consecutivas de anomalía (hysteresis) para confirmar. Umbral adaptativo con percentil 2.0 sobre ventana de 500 scores. Umbral visual: percentil 98 de scores normalizados [0,1].

**C2ST drift detection:** En cada refit, un `RandomForestClassifier` (50 árboles, `max_depth=4`) intenta separar el buffer anterior del actual. AUC 3-fold CV simetrizado (`max(auc, 1-auc)`): 0.5 = sin drift, 1.0 = drift total. Las top-5 feature importances se mapean a nombres legibles vía `_build_feature_names()` (e.g., `scat_45`, `wavelet_band_3`, `spectral_centroid`) y se publican como `top_drift_features` en el payload WebSocket.

### 2. API (`src/api/`)
FastAPI async. Expone datos almacenados y recibe notificaciones del pipeline.

- `POST /internal/score` — pipeline notifica scores; el router WebSocket los broadcast a todos los clientes conectados
- `POST /internal/reset-detector` — activa flag `app.state.detector_reset_pending = True`
- `GET /internal/reset-pending` — pipeline pollea este endpoint; retorna `{"pending": bool}` y limpia el flag (one-shot)
- `POST /internal/fusion-config` — dashboard fija `strategy`/`audio_weight`/`gates` de fusión en vivo; el pipeline los pollea (ADR-0012). 422 si la estrategia es inválida
- `GET /internal/fusion-config` — config de fusión actual `{strategy, audio_weight, gates}`
- `GET /health` — healthcheck `{"status":"ok"}` (para el SPA y orquestación)

**CORS:** `create_app()` registra `CORSMiddleware` con orígenes de la env `CORS_ORIGINS`
(default `http://localhost:5173`) para permitir el SPA desacoplado.
- `GET /events/` — lista eventos con filtros (min_score, fecha, paginación)
- `GET /events/{id}/audio|frame` — stream de archivos desde filesystem
- `GET /events/{id}/frame/annotated` — devuelve el frame JPEG con el bounding box de la fuente más probable dibujado (rectángulo rojo + label `source X.XXX`)
- `GET /events/{id}/offline_analysis` — corre EMD (PyEMD) + mel-spectrogram (librosa) sobre el audio del evento
- `DELETE /events/{id}` — elimina DB row + filesystem; la entrada FAISS queda huérfana (ya es filtrada en búsqueda)
- `DELETE /events/` — elimina todos los eventos y llama `FAISSStore.clear()`
- `POST /search/similar` — recibe audio o imagen (máx. 10 MB), genera embedding con `MultimodalEncoder`, busca k vecinos en FAISS, retorna metadata desde SQLite. Usa `db.get_event_by_faiss_id()` para lookups O(1).
- `GET /search/similar/by-event/{event_id}` — busca eventos similares a uno ya almacenado usando su embedding pre-computado. No requiere carga de modelos → respuesta instantánea. Excluye el evento fuente de los resultados.
- `WS /ws/stream` — WebSocket para el dashboard en tiempo real (incluye `motion_energy`, `source_score`, `drift_auc`, `top_drift_features` en cada mensaje)

**Model preloading:** El lifespan handler lanza un background thread que pre-carga los modelos de embedding (Wav2Vec2+DINOv2) vía `_get_encoder()` con double-checked locking thread-safe. La primera búsqueda por upload no sufre cold-start de ~60s.

Instancias de `Database`, `FAISSStore` y `EventStore` se inyectan vía `request.app.state` (dependency injection en `dependencies.py`). El startup usa `lifespan` context manager (FastAPI 0.93+).

### 3. Dashboard

**Actual (recomendado): SPA en `web/`** — dashboard desacoplado en **React + TypeScript + Vite**
con lenguaje visual "Mission Control" (tema oscuro, tipografía técnica Chakra Petch + JetBrains
Mono, telemetría en mono tabular). Tiempo real *event-driven* vía el hook `useAnomalyStream`
(WebSocket con reconexión backoff+jitter, heartbeat, detección de *staleness*, ring buffers);
estado de servidor con TanStack Query; estado de fusión con Zustand; gráficos uPlot (streaming) +
Recharts (KDE/RMS/IMF/heatmap). Tests con Vitest. Reemplaza a Streamlit —
ver [ADR-0013](docs/adr/0013-reemplazo-dashboard-streamlit-por-spa.md) y
[`docs/frontend-migration/`](docs/frontend-migration/). Requiere `CORSMiddleware` en la API
(env `CORS_ORIGINS`, default `http://localhost:5173`).

**Legacy: `src/dashboard/`** — Streamlit app que consume el API vía `APIClient` (httpx síncrono);
permanece operativa como referencia hasta confirmar la paridad del SPA (fase F8).

Cuatro páginas; todas exponen `render(client: APIClient) -> None`:
- **Live Monitor** — scores en tiempo real vía WebSocket; historial de scores con overlay del umbral adaptativo (p98 normalizado, línea punteada); amplitud RMS; **distribución KDE** de scores recientes (gaussian_kde + umbral vertical); métrica **Drift AUC** (C2ST: 0.5=sin drift, 1.0=drift total) con caption ⚠️ de top drift features; indicador de motion_energy; chip **"Fuente probable"** con coordenadas y source_score de la caja top-1; botones "Reiniciar historial" y "Reiniciar detector"
- **Event Feed** — grid con audio y frame anotado (bounding box de fuente dibujado); botones "Eliminar evento" y "Borrar todo"
- **Similarity Search** — dos modos: (1) upload de query (audio o imagen) con encoding on-the-fly, (2) selección de evento existente que usa el embedding pre-computado (instantáneo). Resultados por similitud coseno. Frames mostrados con bounding box anotado.
- **Offline Analysis** — IMFs de EMD + mel-spectrogram con Plotly

Todo el HTML dinámico usa `st.html()` (no `st.markdown(unsafe_allow_html=True)`) para garantizar renderizado correcto en contextos de columnas.

---

## Módulos de procesamiento

| Módulo | Clase principal | Output |
|--------|----------------|--------|
| `src/dsp/` | `AudioProcessor` | `FeatureVector` (dynamic-dim np.float32) |
| `src/detection/` | `AnomalyDetector` | `AnomalyResult` (score, is_anomaly, is_fitted) |
| `src/embeddings/` | `MultimodalEncoder` | np.ndarray 1536-dim L2-normalizado |
| `src/vision/` | `MotionDetector`, `FrameCapture` | `MotionResult` (boxes con `source_score`, annotated_frame) |
| `src/storage/` | `EventStore`, `Database`, `FAISSStore` | Persistencia híbrida filesystem+SQLite+FAISS |

**Embeddings:** `AudioEncoder` (Wav2Vec2, 768-dim) + `ImageEncoder` (DINOv2, 768-dim) → concatenados y L2-normalizados → 1536-dim. Carga lazy con offline-first: intenta `local_files_only=True`, si falla descarga una sola vez.

**FAISS:** `IndexFlatIP` (inner product = cosine sobre vectores L2-normalizados). IDs secuenciales ligados a `AnomalyEvent.faiss_index_id` en SQLite.

**Persistencia de eventos:** `eventos/<timestamp>/audio.wav`, `frame.jpg`, `embedding.npy`, `metadata.json`.

---

## Source correlation — Correlación fuente-audio

El sistema identifica cuál bounding box de movimiento es más probable que sea la fuente de la anomalía acústica usando un ranking multi-señal:

1. **Detección de movimiento:** MOG2 → morfología → contornos → merge de cajas cercanas (solo si overlap/touching, limitado a ≤40% del frame)
2. **Temporal weights (IoU tracking):** cada box se compara con las del frame anterior vía Intersection-over-Union. Cajas nuevas (IoU<0.3) reciben peso=1.0; cajas persistentes (IoU≥0.3) reciben peso=0.5. Esto prioriza objetos que acaban de aparecer.
3. **Source score:** `source_score = anomaly_score × area_ratio × temporal_weight`, donde `area_ratio = box.area / frame_area`.
4. **Top-1 filtering:** se conserva únicamente la caja con mayor `source_score`.
5. **Visualización:** endpoint `GET /events/{id}/frame/annotated` dibuja la caja top-1 en rojo con label. Live Monitor muestra chip "Fuente probable" con coordenadas y score.

---

## Convenciones críticas

- Los tests **no deben requerir micrófono ni cámara** — usar arrays numpy sintéticos
- `src/dsp/` y `src/vision/` son síncronos por diseño; no introducir `async` en esos módulos
- `pytest-asyncio` configurado con `asyncio_mode = "auto"` — los tests async no necesitan `@pytest.mark.asyncio`
- pylint desactiva C0114/C0115/C0116 (docstrings) — no añadir docstrings en módulos que no los tenían
- Variables de entorno en `.env` de la raíz (copiar desde `.env.example`), **cargado automáticamente** por `python-dotenv` en el pipeline y la API: `EVENTS_DIR`, `DB_PATH`, `FAISS_PATH`, `API_BASE_URL`, `CORS_ORIGINS`, `ENABLE_SLOW_MODELS`. Una variable exportada en el shell tiene prioridad. El SPA usa su propio `web/.env` (`VITE_API_BASE_URL`).
- **Doble horizonte:** `ENABLE_SLOW_MODELS=true` activa el detector lento además del rápido (opt-in, cómputo extra). En línea: `ENABLE_SLOW_MODELS=true poetry run python -m src.pipeline`.

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
