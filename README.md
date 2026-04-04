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

## Arquitectura general

El sistema corre como dos procesos Python independientes que se comunican via HTTP:

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
│                   PROCESO DASHBOARD (Streamlit)             │
│                                                             │
│  Live Monitor │ Event Feed │ Similarity Search │ EMD/Spec   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de datos

```
ventana de audio (2048 samples, float32)
  └─► AudioProcessor.process_window()
        ├─ Scattering1D (Kymatio)      →  126 coeficientes
        ├─ Wavelet energy (db4, l=5)   →    6 valores
        ├─ RMS                         →    1 valor
        └─ ZCR                         →    1 valor
                                       = 134-dim FeatureVector
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

Con `level=5` se obtienen 6 arrays de coeficientes (1 aproximación + 5 detalles), uno por escala. De cada array se calcula la **energía** (suma de cuadrados), produciendo un vector de 6 dimensiones que representa la distribución de energía a través de las escalas.

```python
coeffs = pywt.wavedec(window, 'db4', level=5)
# → [cA5, cD5, cD4, cD3, cD2, cD1]
energies = [np.sum(c**2) for c in coeffs]
# → energía por escala: [E_approx, E_detail_5, ..., E_detail_1]
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
| `J=6` | 6 octavas | Escala máxima: `2^6 = 64` samples |
| `Q=8` | 8 filtros/octava | Balance resolución frecuencial vs. cómputo |
| `shape=2048` | tamaño de ventana | Potencia de 2 requerida por los filter banks |

La salida (n_coefs × T) se reduce haciendo **mean pooling sobre el eje temporal**, produciendo 126 coeficientes que son invariantes locales a traslaciones. Esta representación es mucho más discriminativa y estable que un espectrograma estándar para señales ruidosas.

> **Nota de compatibilidad:** `kymatio.numpy` falla en scipy ≥ 1.17 porque `scipy.special.sph_harm` fue eliminado. Se importa directamente desde el submodule: `from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D`.

### Características temporales adicionales

- **RMS (Root Mean Square):** amplitud media de la ventana, sensible a cambios de volumen
- **ZCR (Zero Crossing Rate):** fracción de samples consecutivos con cambio de signo; alto en sonidos de alta frecuencia (metales, vidrios) y bajo en vocales o tonos

### Segmentación en ventanas solapadas

```
|← 2048 →|
|████████|
    |████████|  ← hop = 512 (75% overlap)
        |████████|
```

El solapamiento del 75% (hop=512 sobre window=2048) asegura que transientes cortos no queden divididos entre dos ventanas y se capturan en al menos una con amplitud máxima.

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
buffer = deque(maxlen=200)   # ventana deslizante

por cada ventana de audio:
    buffer.append(feature_vector)

    si len(buffer) == 200 y no fitted:
        model.fit(buffer)    # primer entrenamiento
        is_fitted = True

    si fitted y samples_since_refit >= 100:
        model.fit(buffer)    # reentrenamiento sobre los últimos 200 samples
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

Durante las primeras ~200 ventanas (~25s de audio a 16kHz con 2048/512), el detector devuelve `is_fitted=False` y `anomaly_score=0.0`. El dashboard muestra "Calentando..." durante este período.

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
6. `boundingRect` sobre cada contorno → `BoundingBox(x, y, w, h, area)`

```python
result = motion_detector.detect_with_result(frame)
# result.boxes → lista de BoundingBox ordenada por área descendente
# result.annotated_frame → frame con rectángulos verdes dibujados
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
| `GET` | `/events/{id}/offline_analysis` | EMD + mel-spectrogram del audio del evento |
| `POST` | `/search/similar` | Upload audio/imagen → top-k eventos similares por cosine |
| `WS` | `/ws/stream` | WebSocket: push de `AnomalyScoreMessage` en tiempo real |
| `POST` | `/internal/score` | Usado internamente por el pipeline para broadcast WS |

Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI).

---

## Dashboard Streamlit

Cuatro páginas accesibles desde el sidebar:

### Live Monitor
- Indicadores en tiempo real: anomaly score, estado del detector, fase de calentamiento
- Historial de amplitud RMS (últimas 256 ventanas) vía `st.line_chart`
- Actualización automática cada segundo via WebSocket

### Event Feed
- Lista de eventos con filtros por score mínimo y cantidad
- Reproducción de audio (`st.audio`) y visualización de frame (`st.image`) por evento
- Score badge y timestamp por evento

### Similarity Search
- Upload de archivo de audio o imagen
- Búsqueda por similitud semántica en el índice FAISS
- Resultados en grid con score de similitud coseno

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
│   │       ├── search.py      # POST /search/similar
│   │       └── websocket.py   # WS /ws/stream + POST /internal/score
│   ├── embeddings/
│   │   ├── config.py          # EmbeddingConfig (model IDs, dims, device)
│   │   ├── audio_encoder.py   # AudioEncoder (Wav2Vec2, lazy load, offline-first)
│   │   ├── image_encoder.py   # ImageEncoder (DINOv2, lazy load, offline-first)
│   │   └── encoder.py         # MultimodalEncoder (concat + L2-norm → 1536-dim)
│   ├── vision/
│   │   ├── config.py          # VisionConfig (MOG2 params, min_contour_area, etc.)
│   │   ├── types.py           # BoundingBox, MotionResult dataclasses
│   │   ├── motion.py          # MotionDetector (MOG2 + morfología + contornos)
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
├── tests/
│   ├── conftest.py            # OMP_NUM_THREADS=1 (fix segfault macOS)
│   ├── test_dsp.py
│   ├── test_detection.py
│   ├── test_storage.py
│   ├── test_api.py
│   ├── test_embeddings.py
│   ├── test_vision.py
│   └── test_dashboard.py
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
```

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

### Terminal 2 — Dashboard

```bash
poetry run streamlit run src/dashboard/app.py
```

Abre automáticamente `http://localhost:8501` en el navegador.

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
# 104 tests, ~3s
```

### Ejecutar por módulo

```bash
poetry run pytest tests/test_dsp.py -v        # 17 tests — DSP features
poetry run pytest tests/test_detection.py -v  # 11 tests — Isolation Forest
poetry run pytest tests/test_storage.py -v    # 20 tests — FAISS, SQLite, EventStore
poetry run pytest tests/test_api.py -v        # 11 tests — FastAPI endpoints + WebSocket
poetry run pytest tests/test_embeddings.py -v # 19 tests — encoders multimodales (mockeados)
poetry run pytest tests/test_vision.py -v     # 15 tests — MOG2, FrameCapture
poetry run pytest tests/test_dashboard.py -v  #  9 tests — APIClient
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
| Frontend | Streamlit | ^1.56 | Dashboard web |
| Frontend | Plotly | ^6.6 | Gráficas interactivas (IMF, spectrogram) |
| HTTP | httpx | — | Cliente sync en dashboard |

---

## Decisiones de diseño relevantes

### Pipeline síncrono, API asíncrona

El pipeline de captura y procesamiento (DSP, detección, visión) es **completamente síncrono**. Los callbacks de `sounddevice` tienen latencia mínima y no toleran el overhead del event loop de asyncio. El pipeline se comunica con la API (que sí es async) via HTTP POST al endpoint `/internal/score`, evitando cualquier compartición de estado entre procesos.

### Isolation Forest con buffer deslizante

Isolation Forest no tiene `partial_fit`. El buffer ring de 200 muestras con refit cada 100 nuevas muestras es un compromiso pragmático: el modelo se adapta al perfil de la señal reciente sin requerir arquitecturas de streaming complejas (River, Vowpal Wabbit). El buffer de 200 ventanas cubre ~25 segundos de audio real, suficiente para capturar variabilidad ambiental.

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
| Fase de calentamiento | ~25s sin detección al inicio | Precargar un modelo guardado de sesiones anteriores |
| Embeddings on-demand | Los modelos Wav2Vec2/DINOv2 solo se cargan al confirmar anomalía | Pool de workers para encoding asíncrono |
| Búsqueda en SQLite | `list_events()` carga todos los eventos para encontrar por `faiss_index_id` | Índice en columna `faiss_index_id`, o tabla de mapeo ID↔evento |
| Sin GPU | Todo corre en CPU por defecto | Cambiar `device='cuda'` en `EmbeddingConfig` si hay GPU disponible |
| Sin sincronización audio-video | MOG2 y DSP corren en threads independientes sin timestamp compartido | Cola de pares (audio_window, frame) con timestamp aligned |
| Umbral fijo de anomalía | `IsolationForest.offset_` es fijo tras cada fit | Umbral adaptativo con percentil del score del buffer |
| FAISS no distribuido | Índice en un solo archivo local | Migrar a Qdrant para escalado horizontal |

---

## Lint

```bash
poetry run pylint src/
```

Configurado para ignorar `C0114/C0115/C0116` (docstrings obligatorios) y excluir `tests/`, `.venv/`, `.claude/`.
