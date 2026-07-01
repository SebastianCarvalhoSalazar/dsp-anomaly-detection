# Sistema Inteligente de Detección de Anomalías Audiovisuales

Sistema Python para detección de anomalías en señales de audio y video en tiempo real, sin necesidad de datos etiquetados. Combina procesamiento digital de señales clásico (DSP), modelos no supervisados y embeddings de modelos preentrenados de última generación para detectar, clasificar y almacenar eventos anómalos con soporte de búsqueda por similitud.

---

## Qué hace el sistema

El sistema procesa continuamente audio del micrófono y video de la cámara para:

1. **Extraer características** de cada ventana de audio usando wavelets y la Transformada de Scattering
2. **Detectar anomalías** en tiempo real con Isolation Forest (sin supervisión)
3. **Generar embeddings multimodales** cuando se confirma una anomalía, combinando audio (Wav2Vec 2.0) e imagen (DINOv2)
4. **Persistir el evento** con su audio, frame de video, embedding y metadatos
5. **Indexar el embedding** en una base vectorial FAISS para búsqueda por similitud
6. **Transmitir el score** en tiempo real al dashboard via WebSocket
7. **Exponer todo via API REST** para consulta, análisis offline con EMD y búsqueda semántica

---

## v0.3 — Detección multimodal de fusión tardía

A partir de la rama `feature/multimodal-fusion-drift-aware`, el sistema evoluciona
de un detector unimodal de audio a un **detector multimodal de fusión tardía**.
Las decisiones de arquitectura están documentadas como ADRs en
[`docs/adr/`](docs/adr/) y el informe técnico en [`docs/REPORT.md`](docs/REPORT.md).

```
Audio ─► AudioProcessor ─► AnomalyDetector (audio) ─► audio_score ┐
                                                                  ├─► Calibración ─► Fusión ─► combined_score
Video ─► VideoFeatureExtractor ─► VideoAnomalyDetector ─► video_score ┘     (percentil)   (weighted/max/and/or)
```

Capacidades nuevas:

- **Detectores independientes** de audio y video (`src/detection/`, `src/vision_detection/`)
  sobre una base común correcta (`BaseAnomalyDetector`).
- **Sincronización temporal** A/V por timestamp (`src/sync/`): cada ventana de audio se
  empareja con el frame más cercano en el tiempo (no "el último frame").
- **Calibración por percentiles** + **fusión configurable** (`src/fusion/`): Weighted
  Average / Maximum / AND / OR, con `dominant_modality`.
- **Refits sensibles a drift**, **snapshots de modelos** (auditoría) y **explicabilidad**
  por z-score (`top_audio_features`/`top_video_features`).
- **Doble horizonte** (modelo rápido/lento) opt-in vía `ENABLE_SLOW_MODELS`.

Compatibilidad hacia atrás: el esquema de eventos migra solo (columnas nuevas nullable)
y la decisión de gating sigue la ruta de audio establecida.

---

## Arquitectura general

El sistema corre como **tres procesos Python independientes** que se comunican via HTTP:

```
┌─────────────────────────────────────────────────────────────┐
│                     PROCESO PIPELINE                        │
│                                                             │
│  Micrófono ──► AudioProcessor ──► AnomalyDetector          │
│                  (DSP features)    (Isolation Forest)       │
│                                         │                   │
│  Cámara ──────► MotionDetector          │ is_anomaly?       │
│                  (MOG2 bboxes)          ▼                   │
│                               MultimodalEncoder             │
│                               (Wav2Vec2 + DINOv2)           │
│                                         │                   │
│                               EventStore + FAISSStore       │
│                               + Database (SQLite)           │
└─────────────────────────────┬───────────────────────────────┘
                              │ POST /internal/score
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PROCESO API (FastAPI)                   │
│                                                             │
│  REST: GET /events/  GET /events/{id}  POST /search/similar │
│  WebSocket: /ws/stream  (broadcast en tiempo real)          │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            PROCESO DASHBOARD (SPA React · web/)            │
│            (reemplaza al dashboard Streamlit)             │
│                                                             │
│  Live Monitor │ Event Feed │ Similarity Search │ EMD/Spec   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de datos

```
ventana de audio (4096 samples, float32)
  └─► AudioProcessor.process_window()
        ├─ Scattering1D (Kymatio)      →  coeficientes (dinámico según J, Q)
        ├─ Wavelet energy (db4, l=7)   →    8 valores
        ├─ RMS                         →    1 valor
        └─ ZCR                         →    1 valor
                                       + spectral (centroid, flatness, rolloff, bandwidth)
                                       + delta features
                                       = dynamic-dim FeatureVector
        └─► AnomalyDetector.score()
              └─► AnomalyResult (anomaly_score ∈ [0,1], is_anomaly)
                    └─[si anomalía]
                          ├─ Wav2Vec2(audio)  →  768-dim embedding
                          ├─ DINOv2(frame)    →  768-dim embedding
                          └─ concat + L2-norm →  1536-dim multimodal
                                │
                                ├─► eventos/<timestamp>/audio.wav
                                ├─► eventos/<timestamp>/frame.jpg
                                ├─► eventos/<timestamp>/embedding.npy
                                ├─► eventos/<timestamp>/metadata.json
                                ├─► FAISS IndexFlatIP (cosine similarity)
                                └─► SQLite (metadata + faiss_index_id)
```

---

## Técnicas de DSP

### Transformada Wavelet (PyWavelets, `db4`)

La Transformada Wavelet Discreta descompone la señal en múltiples escalas de resolución temporal-frecuencial. Se usa la wavelet **Daubechies-4** (`db4`) por su buen balance entre resolución en tiempo y frecuencia, y su capacidad para capturar transientes (inicio de sonidos, impactos) que las técnicas estacionarias como FFT no detectan bien.

Con `level=7` se obtienen 8 arrays de coeficientes (1 aproximación + 7 detalles), uno por escala. De cada array se calcula la **energía** (suma de cuadrados), produciendo un vector de 8 dimensiones que representa la distribución de energía a través de las escalas.

```python
coeffs = pywt.wavedec(window, 'db4', level=7)
# → [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1]
energies = [np.sum(c**2) for c in coeffs]
# → energía por escala: [E_approx, E_detail_7, ..., E_detail_1]
```

### Transformada de Scattering de Wavelets (Kymatio)

La **Wavelet Scattering Transform** genera representaciones invariantes a pequeñas deformaciones temporales y traslaciones en tiempo, lo cual las hace especialmente robustas para señales no estacionarias y variables.

El proceso aplica operadores de scattering en cascada:

```
Capa 0: |x * ψ_0|               (energía de baja frecuencia)
Capa 1: ||x * ψ_1| * ψ_2|       (interacciones entre escalas)
Capa 2: |||x * ψ_1| * ψ_2| * ψ_3|
```

Donde `ψ_j` son wavelets Morlet a escala `2^j`. Los parámetros usados son:

| Parámetro | Valor | Significado |
|---|---|---|
| `J=8` | 8 octavas | Escala máxima: `2^8 = 256` samples |
| `Q=8` | 8 filtros/octava | Balance resolución frecuencial vs. cómputo |
| `shape=4096` | tamaño de ventana | Potencia de 2 requerida por los filter banks |

La salida (n_coefs × T) se reduce haciendo **mean pooling sobre el eje temporal**, produciendo un vector de coeficientes (dimensión dinámica según J y Q) que son invariantes locales a traslaciones. Esta representación es mucho más discriminativa y estable que un espectrograma estándar para señales ruidosas.

> **Nota de compatibilidad:** `kymatio.numpy` falla en scipy ≥ 1.17 porque `scipy.special.sph_harm` fue eliminado. Se importa directamente desde el submodule: `from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D`.

### Características temporales adicionales

- **RMS (Root Mean Square):** amplitud media de la ventana, sensible a cambios de volumen
- **ZCR (Zero Crossing Rate):** fracción de samples consecutivos con cambio de signo; alto en sonidos de alta frecuencia (metales, vidrios) y bajo en vocales o tonos

### Segmentación en ventanas solapadas

```
|← 4096 →|
|████████|
    |████████|  ← hop = 1024 (75% overlap)
        |████████|
```

El solapamiento del 75% (hop=1024 sobre window=4096) asegura que transientes cortos no queden divididos entre dos ventanas y se capturan en al menos una con amplitud máxima.

### EMD — Descomposición en Modos Empíricos (análisis offline)

La **Empirical Mode Decomposition** (PyEMD) descompone señales no lineales y no estacionarias en componentes intrínsecas llamadas **IMFs (Intrinsic Mode Functions)**, cada una con frecuencia instantánea propia. A diferencia de la FFT o wavelets, es completamente adaptativa: no requiere base predefinida.

Se usa en análisis offline sobre los eventos almacenados para entender la estructura interna de las anomalías detectadas.

---

## Detección de anomalías

### Isolation Forest

El detector usa **Isolation Forest** (`scikit-learn`), un algoritmo de detección de anomalías no supervisado basado en el principio de que los puntos anómalos son más fáciles de aislar: requieren menos particiones aleatorias para quedar separados.

#### Protocolo de entrenamiento online (sin `partial_fit`)

IsolationForest no soporta aprendizaje incremental. Se usa un protocolo de **buffer ring + refit periódico**:

```
buffer = deque(maxlen=500)   # ventana deslizante

por cada ventana de audio:
    buffer.append(feature_vector)

    si len(buffer) == 500 y no fitted:
        model.fit(buffer)    # primer entrenamiento
        is_fitted = True

    si fitted y samples_since_refit >= 200:
        model.fit(buffer)    # reentrenamiento sobre los últimos 500 samples
        samples_since_refit = 0
```

Esto aproxima el aprendizaje online: el modelo se adapta al perfil estadístico reciente de la señal, permitiendo que lo que era anómalo en un contexto deje de serlo en otro.

#### Normalización del score

`IsolationForest.score_samples()` devuelve valores negativos (típicamente entre -0.7 y -0.3). Se normalizan a `[0, 1]` usando **min-max calibrado** sobre el buffer de entrenamiento:

```python
# 0 = más anómalo, 1 = más normal → se invierte
anomaly_score = 1.0 - (raw_score - buffer_min) / (buffer_max - buffer_min)
anomaly_score = clip(anomaly_score, 0.0, 1.0)
```

Un evento se considera anomalía cuando `raw_score < model.offset_` (threshold del Isolation Forest).

#### Fase de calentamiento

Durante las primeras ~500 ventanas (~2 min de audio a 16kHz con 4096/1024), el detector devuelve `is_fitted=False` y `anomaly_score=0.0`. El dashboard muestra "Calentando..." durante este período.

### PCA — Reducción de dimensionalidad

Antes de alimentar el Isolation Forest, el vector de features (251 dimensiones) se reduce a **25 componentes** mediante **PCA** (`sklearn.decomposition.PCA`). Se ajusta junto con el IF en cada refit.

```python
pca = PCA(n_components=25)
X_reduced = pca.fit_transform(X)  # 251 → 25 dims
model.fit(X_reduced)
```

Empíricamente, 9 componentes capturan >99% de la varianza; 25 es un margen de seguridad que preserva señales débiles sin perder la eficiencia del IF. El scoring aplica `pca.transform()` sobre cada vector individual antes de `score_samples()`.

### C2ST — Detección de drift por Classifier Two-Sample Test

Para medir si la distribución de features está cambiando entre refits sucesivos, se usa un **Classifier Two-Sample Test (C2ST)**:

1. Se etiqueta el buffer anterior como clase 0 y el actual como clase 1
2. Se entrena un `RandomForestClassifier` (50 árboles, `max_depth=4`) para distinguirlos
3. Se evalúa con **3-fold cross-validated AUC**
4. Se simetriza: `AUC = max(auc, 1 - auc)` para que siempre sea ≥ 0.5

| AUC | Interpretación |
|---|---|
| ≈ 0.5 | Sin drift (buffers indistinguibles) |
| 0.7–0.8 | Drift moderado |
| > 0.9 | Drift severo (distribución cambió drásticamente) |

Además, las **feature importances** del RF se mapean a nombres legibles (`scat_45`, `wavelet_band_3`, `spectral_centroid`, etc.) usando `_build_feature_names()`, y las **top-5** se reportan como `top_drift_features` para interpretar qué aspectos del audio cambiaron.

Esto reemplaza la métrica anterior de L2 norm entre medias de features, que no era sensible a cambios en varianza o forma de la distribución.

---

## Modelos de embeddings

### Wav2Vec 2.0 (audio)

**Modelo:** `facebook/wav2vec2-base`

Wav2Vec 2.0 es un modelo de representación de audio self-supervised preentrenado por Meta. Usa un encoder CNN para extraer features locales y un Transformer para modelar dependencias temporales largas. Se usa en modo de **extracción de features** (sin fine-tuning):

```python
outputs = model(input_values)  # last_hidden_state: (1, T, 768)
embedding = outputs.last_hidden_state.mean(dim=1)  # (768,)
embedding = L2_normalize(embedding)
```

El **mean pooling temporal** agrega todos los frames en un único vector de 768 dimensiones que captura el contenido semántico del audio de forma invariante a su duración exacta.

### DINOv2 (imagen)

**Modelo:** `facebook/dinov2-base`

DINOv2 es un Vision Transformer (ViT) preentrenado con aprendizaje auto-supervisado (distilación de features) por Meta. Produce representaciones visuales de alta calidad sin etiquetas. Se usa el **token CLS** de la última capa como embedding de la imagen:

```python
outputs = model(**inputs)  # last_hidden_state: (1, 197, 768)
embedding = outputs.last_hidden_state[:, 0, :]  # CLS token → (768,)
embedding = L2_normalize(embedding)
```

Los 196 patches restantes encodifican regiones locales; el CLS token agrega el contexto global de la imagen.

> **Nota:** Los frames de OpenCV vienen en formato BGR. Se convierte a RGB con `frame[:, :, ::-1]` antes de pasarlos al processor de HuggingFace.

### Embedding multimodal (1536-dim)

```python
multimodal = np.concatenate([audio_emb_768, image_emb_768])
multimodal = L2_normalize(multimodal)  # → (1536,)
```

La concatenación directa asume que ambos sub-espacios contribuyen por igual. La L2-normalización final garantiza que la similitud coseno en FAISS sea válida y comparable entre todos los eventos.

Si no hay frame disponible (pipeline solo-audio), la mitad visual se rellena con ceros y el vector resultante se normaliza para que la mitad audio siga siendo comparable.

---

## Base vectorial y persistencia

### FAISS — Búsqueda por similitud

Se usa **`IndexFlatIP`** (Inner Product exacto):

- Vectores insertados y consultados son L2-normalizados antes de cada operación
- Inner product de vectores unitarios = **similitud coseno**
- Búsqueda exacta (sin aproximación): óptima para MVPs con < 100K eventos
- Thread-safe mediante `threading.Lock`
- Persistencia automática a disco (`faiss.write_index`) tras cada inserción

```python
faiss_id = faiss_store.add(embedding)          # → id secuencial entero
distances, ids = faiss_store.search(query, k=5) # → cosine similarities ∈ [0,1]
```

### SQLite — Metadata de eventos

Esquema de la tabla `anomaly_events`:

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK | ID autoincremental |
| `timestamp` | DATETIME(tz) | Momento de detección (UTC) |
| `anomaly_score` | FLOAT | Score normalizado [0,1] |
| `event_dir` | TEXT UNIQUE | Ruta al directorio del evento |
| `faiss_index_id` | INTEGER | ID en el índice FAISS |
| `audio_path` | TEXT | Ruta a `audio.wav` |
| `frame_path` | TEXT | Ruta a `frame.jpg` (nullable) |
| `embedding_path` | TEXT | Ruta a `embedding.npy` |
| `source_region_json` | TEXT | Bounding boxes de movimiento (JSON) |
| `extra_json` | TEXT | Metadatos adicionales (JSON) |

SQLAlchemy **síncrono** con `run_in_executor` en FastAPI: no requiere `aiosqlite` y es suficiente para el throughput de un MVP.

### Filesystem — Artefactos de eventos

```
eventos/
  2024-01-15T10-30-00.123456+00-00/
    audio.wav       ← IEEE_FLOAT subtype (preserva rango float32 completo)
    frame.jpg       ← JPEG, solo si había frame disponible
    embedding.npy   ← array (1536,) float32
    metadata.json   ← score, timestamp, sample_rate, bounding_boxes, etc.
```

---

## Detección de movimiento en video

`MotionDetector` usa **MOG2 (Mixture of Gaussians v2)** de OpenCV:

1. **Background subtraction MOG2** → máscara de foreground binaria
2. **Morphological Open** (erosión + dilatación) → elimina ruido sal-y-pimienta
3. **Dilatación** → conecta regiones fragmentadas del mismo objeto
4. **`findContours`** → extrae contornos del foreground
5. Filtrado por `min_contour_area=500 px²` → descarta falsos positivos pequeños
6. `boundingRect` sobre cada contorno → `BoundingBox(x, y, w, h, area, source_score=0.0)`
7. **Merge de cajas cercanas:** cajas cuyas aristas están a ≤`merge_gap` px se fusionan en una sola (default: 0 = solo cajas que se tocan/superponen). Una caja fusionada no puede superar `max_box_ratio` (40%) del área del frame, evitando mega-cajas.
8. **Temporal weights (IoU tracking):** cada caja se compara con las del frame anterior por IoU. Cajas nuevas (IoU<0.3) reciben peso=1.0; persistentes (IoU≥0.3) reciben peso=0.5.

```python
result = motion_detector.detect(frame)
# result → lista de BoundingBox con source_score = temporal_weight (asignado por el detector)
# El pipeline luego calcula: source_score = anomaly_score × area_ratio × temporal_weight
```

---

## API REST

Base URL: `http://localhost:8000`

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/events/` | Lista eventos (filtros: `limit`, `offset`, `min_score`) |
| `GET` | `/events/{id}` | Detalle de un evento |
| `GET` | `/events/{id}/audio` | Stream del archivo WAV |
| `GET` | `/events/{id}/frame` | Stream del JPEG |
| `GET` | `/events/{id}/frame/annotated` | Frame JPEG con bounding box de la fuente más probable (rectángulo rojo + label `source X.XXX`) |
| `GET` | `/events/{id}/offline_analysis` | EMD + mel-spectrogram del audio del evento |
| `DELETE` | `/events/{id}` | Elimina un evento (DB + filesystem; entrada FAISS queda huérfana) |
| `DELETE` | `/events/` | Elimina todos los eventos y resetea el índice FAISS |
| `POST` | `/search/similar` | Upload audio/imagen → top-k eventos similares por cosine (máx. 10 MB) |
| `GET` | `/search/similar/by-event/{id}` | Busca eventos similares a uno ya almacenado usando su embedding pre-computado. Instantáneo (no carga modelos). Excluye el evento fuente. |
| `WS` | `/ws/stream` | WebSocket: push de `AnomalyScoreMessage` en tiempo real (incluye `motion_energy`, `source_score`, `drift_auc`, `top_drift_features`) |
| `POST` | `/internal/score` | Usado internamente por el pipeline para broadcast WS |
| `POST` | `/internal/reset-detector` | Señaliza al pipeline que resetee su `AnomalyDetector` |
| `GET` | `/internal/reset-pending` | Polling del pipeline: retorna `{"pending": bool}` y limpia el flag |
| `POST` | `/internal/fusion-config` | Dashboard → pipeline: fija `strategy`/`audio_weight`/`gates` de fusión en vivo (422 si estrategia inválida) |
| `GET` | `/internal/fusion-config` | Config de fusión actual `{strategy, audio_weight, gates}` |
| `GET` | `/health` | Healthcheck: `{"status":"ok"}` |

Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI).

**CORS:** habilitado vía `CORSMiddleware` para el SPA desacoplado. Orígenes permitidos por
la variable `CORS_ORIGINS` (default `http://localhost:5173`, el dev server de Vite).

---

## Dashboard web (SPA React · Mission Control)

A partir de la rama `feature/frontend-spa`, el dashboard principal es una **SPA
desacoplada en React + TypeScript + Vite** en [`web/`](web/), que consume la API
(REST + WebSocket) y reemplaza al dashboard Streamlit. Decisión:
[ADR-0013](docs/adr/0013-reemplazo-dashboard-streamlit-por-spa.md); plan y contrato en
[`docs/frontend-migration/`](docs/frontend-migration/).

**Lenguaje visual "Mission Control":** tema oscuro de sala de control, tipografía técnica
(Chakra Petch + JetBrains Mono), telemetría en mono tabular, traza tipo osciloscopio y un
estado de sistema inequívoco (NOMINAL / CALIBRANDO / ANOMALÍA).

Mejoras clave sobre Streamlit:

- **Tiempo real event-driven** (`useAnomalyStream`): sin polling de página; WebSocket con
  reconexión (backoff + jitter), heartbeat y detección de *staleness* → indicador de
  conexión honesto (ya no "siempre conectado").
- **Estado de fusión persistente** entre navegación (Zustand); estrategias de fusión
  reimplementadas en TS (sin acoplar `src.fusion`); el `combined_score` del backend sigue
  siendo la fuente autoritativa.
- **Acciones destructivas con confirmación**, accesibilidad (WCAG AA, `prefers-reduced-motion`)
  y diseño mobile-first.
- Gráficos ligeros: **uPlot** (streaming) + **Recharts** (KDE / RMS / IMF / heatmap),
  con *code-splitting* por ruta.

```bash
cd web
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                 # http://localhost:5173
```

Requiere la API con CORS habilitado (`CORS_ORIGINS`). Detalle en [`web/README.md`](web/README.md).

---

## Dashboard Streamlit (legacy · en decomisión)

> El dashboard Streamlit permanece operativo como referencia hasta confirmar la paridad del
> SPA (fase F8 del plan de migración). Se eliminará junto con `streamlit`/`plotly` una vez
> validado. Cuatro páginas accesibles desde el sidebar:

### Live Monitor
- Indicadores en tiempo real: anomaly score, estado del detector, fase de calentamiento, **motion energy**, **Drift AUC** (C2ST: 0.5 = sin drift, 1.0 = drift total)
- **Top drift features**: caption con ⚠️ mostrando las 3 features más relevantes del drift detectado (e.g., `scat_45, wavelet_band_5, spectral_centroid`)
- Chip **"Fuente probable"**: coordenadas y `source_score` de la caja top-1 que correlaciona con la anomalía
- **Historial de scores** con overlay del umbral adaptativo (percentil 98 de scores normalizados [0,1], línea punteada)
- **Amplitud RMS** (últimas 256 ventanas) con Plotly
- **Distribución de scores (KDE)**: densidad estimada por kernel gaussiano (`scipy.stats.gaussian_kde`) de los scores recientes, con línea vertical del umbral adaptativo
- Actualización automática cada segundo via WebSocket
- Botón **Reiniciar historial**: limpia los gráficos de la sesión actual
- Botón **Reiniciar detector**: envía señal al pipeline para resetear el Isolation Forest (reinicia warmup)

### Event Feed
- Lista de eventos con filtros por score mínimo y orden
- Reproducción de audio (`st.audio`) y visualización de frame anotado (con bounding box de fuente dibujado) por evento
- Botón **Eliminar evento** por cada evento individual
- Botón **Borrar todo**: elimina todos los eventos, filesystem y FAISS

### Similarity Search
- **Dos modos de búsqueda:**
  - 📁 **Subir archivo**: upload de audio o imagen → encoding on-the-fly (puede tardar ~60s la primera vez)
  - 📋 **Evento existente**: seleccionar un evento guardado → usa su embedding pre-computado → **respuesta instantánea** sin carga de modelos
- Resultados en grid con score de similitud coseno
- Frames mostrados con bounding box de fuente anotado
- Tarjeta de previsualización del evento de consulta (modo evento existente)

### Offline Analysis
- Selección de evento guardado
- Descomposición EMD: gráfica de cada IMF con `plotly` (subplots)
- Mel-spectrogram en escala dB con `plotly` heatmap

---

## Estructura del proyecto

```
dsp-idea/
├── src/
│   ├── dsp/
│   │   ├── config.py          # DSPConfig (window_size, J, Q, wavelet, etc.)
│   │   ├── types.py           # FeatureVector dataclass
│   │   └── processor.py       # AudioProcessor (Scattering + Wavelet + RMS/ZCR)
│   ├── detection/
│   │   ├── config.py          # DetectorConfig (buffer_size, refit_every, etc.)
│   │   ├── types.py           # AnomalyResult dataclass
│   │   └── detector.py        # AnomalyDetector (IsolationForest, thread-safe)
│   ├── storage/
│   │   ├── config.py          # StorageConfig (paths, embedding_dim)
│   │   ├── models.py          # SQLAlchemy ORM: AnomalyEvent
│   │   ├── db.py              # Database (SQLite, sync)
│   │   ├── event_store.py     # EventStore (filesystem: WAV, JPEG, NPY, JSON)
│   │   └── faiss_store.py     # FAISSStore (IndexFlatIP, L2-norm, thread-safe)
│   ├── api/
│   │   ├── main.py            # create_app() factory (inyección de dependencias)
│   │   ├── schemas.py         # Pydantic models para request/response
│   │   ├── dependencies.py    # get_db(), get_faiss_store(), get_event_store()
│   │   └── routers/
│   │       ├── events.py      # CRUD eventos + offline analysis (EMD)
│   │       ├── search.py      # POST /search/similar + GET /search/similar/by-event/{id}
│   │       └── websocket.py   # WS /ws/stream + POST /internal/score
│   ├── embeddings/
│   │   ├── config.py          # EmbeddingConfig (model IDs, dims, device)
│   │   ├── audio_encoder.py   # AudioEncoder (Wav2Vec2, lazy load, offline-first)
│   │   ├── image_encoder.py   # ImageEncoder (DINOv2, lazy load, offline-first)
│   │   └── encoder.py         # MultimodalEncoder (concat + L2-norm → 1536-dim)
│   ├── vision/
│   │   ├── config.py          # VisionConfig (MOG2 params, min_contour_area, merge_gap, max_box_ratio)
│   │   ├── types.py           # BoundingBox (con source_score), MotionResult dataclasses
│   │   ├── motion.py          # MotionDetector (MOG2 + morfología + contornos + merge + IoU temporal)
│   │   └── capture.py         # FrameCapture (OpenCV, context manager)
│   ├── dashboard/
│   │   ├── api_client.py      # APIClient (httpx sync)
│   │   ├── app.py             # Streamlit entry point + navegación
│   │   └── pages/
│   │       ├── live_monitor.py
│   │       ├── event_feed.py
│   │       ├── similarity_search.py
│   │       └── offline_analysis.py
│   └── pipeline.py            # Orquestador: loop principal audio+video+anomalía
├── web/                       # Dashboard SPA (React + TS + Vite) — "Mission Control"
│   ├── src/
│   │   ├── api/               # cliente tipado (types, client, endpoints, queryKeys, mediaUrls)
│   │   ├── hooks/             # useAnomalyStream (WS) + hooks TanStack Query
│   │   ├── store/             # fusionDraftStore (Zustand, persistente)
│   │   ├── lib/               # fusion, ringBuffer, kde, status, format, constants
│   │   ├── components/        # charts/ · common/ · fusion/ · events/ · similarity/
│   │   ├── pages/             # LiveMonitor · EventFeed · SimilaritySearch · OfflineAnalysis
│   │   └── layout/            # AppShell, NavBar
│   ├── package.json
│   ├── vite.config.ts · tailwind.config.ts · tsconfig.json
│   └── README.md
├── tests/
│   ├── conftest.py            # OMP_NUM_THREADS=1 (fix segfault macOS)
│   ├── test_dsp.py
│   ├── test_detection.py
│   ├── test_storage.py
│   ├── test_api.py
│   ├── test_embeddings.py
│   ├── test_vision.py
│   ├── test_dashboard.py
│   ├── test_pipeline.py
│   └── test_sample.py
├── doc/
│   ├── idea.md                # Especificación original del proyecto
│   └── ...                    # Papers de referencia
├── pyproject.toml
├── poetry.lock
└── .env.example
```

---

## Requisitos del sistema

- Python 3.11–3.12
- macOS, Linux o Windows (con WSL2 recomendado para Linux)
- Micrófono y cámara (solo para el pipeline en tiempo real; los tests no los requieren)
- ~1.5 GB de espacio para modelos HuggingFace en cache
- Conexión a internet solo para la descarga inicial de modelos

---

## Instalación y configuración

### 1. Clonar e instalar dependencias

```bash
git clone <repo>
cd dsp-idea
pip install poetry
poetry install
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` si se quieren cambiar los paths por defecto:

```env
EVENTS_DIR=./eventos      # directorio donde se guardan los eventos
DB_PATH=./data/events.db  # base de datos SQLite
FAISS_PATH=./data/faiss.index
API_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:5173   # orígenes permitidos para el SPA (coma-separados)
```

> El SPA usa su propia variable `VITE_API_BASE_URL` en `web/.env` (por defecto
> `http://localhost:8000`).

### 3. Descargar modelos de HuggingFace (una sola vez)

Requiere internet. Los modelos quedan en el cache local de HuggingFace (`~/.cache/huggingface/`). Todas las ejecuciones posteriores son completamente offline.

```bash
poetry run python - <<'EOF'
from src.embeddings import MultimodalEncoder
print("Descargando Wav2Vec2-base (~360MB) y DINOv2-base...")
MultimodalEncoder().ensure_downloaded()
print("Listo. El sistema funcionará offline a partir de ahora.")
EOF
```

---

## Ejecución

El sistema completo requiere tres procesos. Ábrelos en terminales separadas **en orden**:

### Terminal 1 — API

```bash
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Verificar que está activa:
```bash
curl http://localhost:8000/events/
# → []
```

Documentación interactiva: `http://localhost:8000/docs`

### Terminal 2 — Dashboard (SPA React · recomendado)

```bash
cd web
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                 # http://localhost:5173
```

Requiere que la API tenga `CORS_ORIGINS` incluyendo el origen del SPA (default
`http://localhost:5173`).

<details><summary>Alternativa legacy — dashboard Streamlit</summary>

```bash
poetry run streamlit run src/dashboard/app.py   # http://localhost:8501
```
</details>

### Terminal 3 — Pipeline (requiere micrófono y cámara)

```bash
poetry run python -m src.pipeline
```

El pipeline:
1. Abre el micrófono (dispositivo 0 por defecto)
2. Abre la cámara (dispositivo 0 por defecto)
3. Entra en la fase de calentamiento (~25s): acumula 200 ventanas de audio para entrenar el detector
4. Comienza a detectar anomalías y actualizar el dashboard en tiempo real

Para detener: `Ctrl+C`

---

## Tests

### Ejecutar la suite completa

```bash
poetry run pytest -v
# 240 tests, ~9s
```

### Ejecutar por módulo

```bash
poetry run pytest tests/test_dsp.py -v             # 21 tests — DSP features
poetry run pytest tests/test_detection.py -v       # 26 tests — Isolation Forest + PCA + C2ST drift
poetry run pytest tests/test_base_detector.py -v   #  9 tests — BaseAnomalyDetector (C1/C2/H1/H6 fixes)
poetry run pytest tests/test_storage.py -v         # 36 tests — FAISS, SQLite, EventStore
poetry run pytest tests/test_schema_migration.py -v#  2 tests — migración de esquema compat
poetry run pytest tests/test_api.py -v             # 34 tests — FastAPI endpoints + WebSocket
poetry run pytest tests/test_embeddings.py -v      # 19 tests — encoders multimodales (mockeados)
poetry run pytest tests/test_vision.py -v          # 34 tests — MOG2, IoU, temporal weights, box merge
poetry run pytest tests/test_video_detection.py -v #  6 tests — detector de video (sin PCA)
poetry run pytest tests/test_sync.py -v            #  8 tests — alineación temporal A/V
poetry run pytest tests/test_calibration.py -v     #  4 tests — calibración por percentiles
poetry run pytest tests/test_fusion.py -v          #  9 tests — estrategias de fusión
poetry run pytest tests/test_drift_refit.py -v     #  6 tests — drift-aware refit, snapshots, explicabilidad
poetry run pytest tests/test_dashboard.py -v       # 20 tests — APIClient + dashboard pages
poetry run pytest tests/test_pipeline.py -v        #  6 tests — pipeline orchestration
```

Todos los tests usan datos simulados (arrays numpy sintéticos, frames aleatorios, modelos mockeados). No requieren micrófono, cámara ni modelos descargados.

### Smoke test del pipeline DSP sin hardware

```bash
poetry run python - <<'EOF'
from src.dsp import AudioProcessor
from src.detection import AnomalyDetector
import numpy as np

proc = AudioProcessor()
det = AnomalyDetector()
signal = np.random.randn(16000 * 10).astype(np.float32)  # 10s sintéticos

for window in proc.segment_signal(signal):
    feat = proc.process_window(window)
    result = det.score(feat)
    if result.window_index % 20 == 0 or result.is_anomaly:
        status = "WARMUP" if not result.is_fitted else ("ANOMALY" if result.is_anomaly else "ok")
        print(f"win={result.window_index:3d}  score={result.anomaly_score:.3f}  [{status}]")
EOF
```

---

## Stack tecnológico

| Categoría | Librería | Versión | Uso |
|---|---|---|---|
| DSP | PyWavelets | ^1.9 | Wavelet DWT, energía por escala |
| DSP | Kymatio | ^0.3 | Scattering Transform 1D |
| DSP | librosa | ^0.11 | Mel-spectrogram (análisis offline) |
| DSP | PyEMD | ^1.9 | EMD offline sobre eventos |
| ML | scikit-learn | ^1.8 | IsolationForest |
| DL | PyTorch | ^2.11 | Backend de modelos |
| DL | Transformers | ^5.5 | Wav2Vec2, DINOv2 |
| Audio | sounddevice | ^0.5 | Captura de micrófono en tiempo real |
| Audio | soundfile | ^0.13 | Lectura/escritura WAV (IEEE FLOAT) |
| Audio | torchaudio | ^2.11 | Resampleo de audio |
| Video | OpenCV | ^4.13 | Captura de cámara, MOG2, morfología |
| Vectores | faiss-cpu | ^1.13 | IndexFlatIP, búsqueda coseno exacta |
| DB | SQLAlchemy | ^2.0 | ORM sync para SQLite |
| API | FastAPI | ^0.135 | REST + WebSocket |
| API | Uvicorn | ^0.43 | Servidor ASGI |
| Frontend SPA | React + TypeScript | 18 / 5 | Dashboard desacoplado (`web/`) |
| Frontend SPA | Vite | ^5 | Build y dev server del SPA |
| Frontend SPA | Tailwind CSS | ^3 | Estilos (tema "Mission Control") |
| Frontend SPA | TanStack Query | ^5 | Estado de servidor (REST) |
| Frontend SPA | Zustand | ^4 | Estado de fusión (persistente) |
| Frontend SPA | uPlot / Recharts | ^1 / ^2 | Gráficos realtime / estáticos |
| Frontend SPA | Vitest + MSW | ^2 | Tests del SPA |
| Frontend (legacy) | Streamlit + Plotly | ^1.56 / ^6.6 | Dashboard anterior (en decomisión) |
| HTTP | httpx | — | Cliente sync en dashboard Streamlit |

---

## Decisiones de diseño relevantes

### Pipeline síncrono, API asíncrona

El pipeline de captura y procesamiento (DSP, detección, visión) es **completamente síncrono**. Los callbacks de `sounddevice` tienen latencia mínima y no toleran el overhead del event loop de asyncio. El pipeline se comunica con la API (que sí es async) via HTTP POST al endpoint `/internal/score`, evitando cualquier compartición de estado entre procesos.

### Isolation Forest con buffer deslizante

Isolation Forest no tiene `partial_fit`. El buffer ring de 500 muestras con refit cada 200 nuevas muestras es un compromiso pragmático: el modelo se adapta al perfil de la señal reciente sin requerir arquitecturas de streaming complejas (River, Vowpal Wabbit). El buffer de 500 ventanas cubre ~2 minutos de audio real, suficiente para capturar variabilidad ambiental.

### SQLite síncrono en lugar de asyncpg

Para el MVP se eligió SQLite + SQLAlchemy síncrono porque:
- No requiere servidor externo
- El throughput de anomalías (eventos raros, no frecuentes) no justifica la complejidad de async DB
- FastAPI maneja las llamadas síncronas con `anyio.to_thread.run_sync()` sin bloquear el event loop
- Migrar a PostgreSQL solo requiere cambiar `DATABASE_URL`

### FAISS IndexFlatIP en lugar de HNSW o IVF

`IndexFlatIP` (búsqueda exacta por inner product) es la elección correcta para un MVP con pocos miles de eventos. Su ventaja principal: **sin fase de entrenamiento**, lo que permite insertar el primer vector sin configuración previa. Para escalar a millones de eventos, la migración a `IndexHNSWFlat` o `IndexIVFFlat` solo requiere cambiar la inicialización del índice.

### Embeddings lazy-load con offline-first

Los modelos de HuggingFace se cargan en el primer `encode()`:

```python
try:
    model = Wav2Vec2Model.from_pretrained(model_id, local_files_only=True)
except EnvironmentError:
    model = Wav2Vec2Model.from_pretrained(model_id)  # descarga
```

Esto evita penalizar el arranque del pipeline en ejecuciones normales (offline-first) sin requerir un flag manual que el usuario deba recordar cambiar.

### WAV con subtype IEEE_FLOAT

`soundfile` usa PCM_16 por defecto, que **recorta** amplitudes fuera de `[-1, 1]`. Las señales de audio del pipeline pueden tener valores de mayor amplitud (no normalizadas). Se fuerza `subtype='FLOAT'` para preservar el rango completo float32, necesario para reproducir fielmente los eventos y para que el análisis EMD offline sea correcto.

### conftest.py con OMP_NUM_THREADS=1

En macOS, PyTorch y OpenCV ambos linkan contra el framework Accelerate/OpenMP. Cuando coexisten en el mismo proceso, sus pools de threads nativos compiten y producen segmentation faults no deterministas. Establecer `OMP_NUM_THREADS=1` antes de cualquier import serializa OpenMP y elimina el problema. Esto solo afecta al rendimiento de operaciones matriciales en tests (irrelevante); en el pipeline real cada proceso carga solo sus librerías.

---

## Limitaciones conocidas y trabajo futuro

| Limitación | Descripción | Posible mejora |
|---|---|---|
| Sin GPU | Todo corre en CPU por defecto | Cambiar `device='cuda'` en `EmbeddingConfig` si hay GPU disponible |
| FAISS no distribuido | Índice en un solo archivo local | Migrar a Qdrant para escalado horizontal |
| Source correlation heurística | `source_score` usa IoU temporal + área ratio; no hay beamforming real | Audio-based localization con array de micrófonos |
| Validar fusión como gating en datos reales | El gating por fusión existe (`FUSION_GATES_DECISION`) pero está off por defecto | Medir FP/FN con datos etiquetados antes de activarlo por defecto |

> **Resuelto en v0.3:** sincronización audio-video por timestamp ([ADR-0003](docs/adr/0003-timestamp-based-av-synchronization.md)) y `motion_energy` acotado a `[0,1]`.
> **Resuelto en v0.3.1:** integridad FAISS↔SQLite cross-proceso (`IndexIDMap2` con IDs = PK + escritura atómica + file-lock; hallazgos C3/C4), path traversal en endpoints de streaming (H2), rutas de evento absolutas (H3), y gating por fusión disponible tras flag. El detalle de riesgos y trabajo futuro está en [`docs/REPORT.md`](docs/REPORT.md).

---

## Flujo de trabajo y ramas (Git)

El repositorio sigue **GitHub Flow + versionado semántico por tags**: una única
rama troncal de larga vida y ramas de trabajo cortas que se integran por Pull
Request. Las versiones se marcan con **tags inmutables**, no con ramas.

### Ramas

| Rama | Vida | Nace de | Se integra a | Propósito |
|------|------|---------|--------------|-----------|
| **`main`** | Permanente | — | — | Única rama troncal. Siempre desplegable; es el *default branch*. Solo recibe merges vía PR. |
| **`feature/<slug>`** | Corta | `main` | `main` (PR) | Una funcionalidad nueva. Ej: `feature/multimodal-fusion-drift-aware`. |
| **`fix/<slug>`** | Corta | `main` | `main` (PR) | Corrección de bug. Ej: `fix/faiss-id-desync`. |
| **`docs/<slug>`** | Corta | `main` | `main` (PR) | Cambios solo de documentación. Ej: `docs/branching-workflow`. |

**Reglas:**

- ✅ Toda rama de trabajo **nace de `main` actualizado** y vuelve a `main` por PR.
- ✅ Ramas cortas: se borran tras el merge (local y remoto).
- ✅ No se commitea directo a `main`; los cambios pasan por PR.
- ❌ **No** se crea una rama por versión (`dev/vX.Y.Z`) — eso es lo que hacen los tags.

### Tags (releases)

Cada versión publicada se marca con un tag `vMAYOR.MENOR.PARCHE[-pre]`
([SemVer](https://semver.org/lang/es/)) sobre el commit correspondiente de `main`:

```bash
git tag v0.3.0-alpha            # marca el commit actual
git push origin v0.3.0-alpha
```

Tags existentes: `v0.1.0-alpha`, `v0.2.0-alpha` (hitos previos preservados).

### Ciclo típico

```bash
# 1. Partir de main actualizado
git checkout main && git pull --ff-only

# 2. Crear la rama de trabajo
git checkout -b feature/mi-cambio

# 3. Trabajar y commitear
git add -A && git commit -m "feat: ..."

# 4. Subir y abrir PR contra main
git push -u origin feature/mi-cambio
gh pr create --base main

# 5. Tras el merge: limpiar
git checkout main && git pull --ff-only
git branch -d feature/mi-cambio
git push origin --delete feature/mi-cambio   # si no se borró desde la UI
```

> Las decisiones de arquitectura se registran como ADRs en
> [`docs/adr/`](docs/adr/) (formato MADR). Toda decisión técnica significativa
> debería acompañarse de un ADR nuevo.

---

## Lint

```bash
poetry run pylint src/
```

Configurado para ignorar `C0114/C0115/C0116` (docstrings obligatorios) y excluir `tests/`, `.venv/`, `.claude/`.
