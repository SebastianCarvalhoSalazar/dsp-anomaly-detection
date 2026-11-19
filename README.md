# Sistema de Detección de Anomalías Acústicas en Tiempo Real

Sistema Python que escucha un lugar de forma continua, aprende cómo suena normalmente y
registra como **evento** todo sonido fuera de lo habitual, sin datos etiquetados. Combina
procesamiento digital de señales (wavelets, Transformada de Scattering), un detector no
supervisado (Isolation Forest) y embeddings preentrenados (Wav2Vec 2.0 + DINOv2) para
buscar eventos parecidos. La cámara aporta contexto: el frame del momento y cuánto
movimiento había.

---

## Qué hace el sistema

1. **Extrae características** de cada ventana de audio (0.256 s) con wavelets, Scattering y
   descriptores espectrales.
2. **Detecta anomalías** con un Isolation Forest que se re-entrena en línea; un umbral fijo
   por entrenamiento y 3 ventanas consecutivas confirman la anomalía.
3. **Agrupa** las ventanas anómalas consecutivas en **un evento** con inicio, fin y duración.
4. **Guarda el evento** (audio con contexto previo, frame del pico, metadatos) y su
   **embedding multimodal** (Wav2Vec 2.0 + DINOv2) en un hilo aparte, sin frenar el audio.
5. **Indexa** el embedding en FAISS para **buscar eventos similares**.
6. **Transmite** el score de cada ventana al dashboard por WebSocket.
7. **Expone** eventos, audio, frames, espectrograma mel y búsqueda por similitud vía API REST.

Las decisiones de arquitectura están en [`docs/adr/`](docs/adr/); la evolución reciente,
en [`docs/plan-correcciones.md`](docs/plan-correcciones.md) (núcleo corregido, ADR-0014) y
[`docs/plan-destilacion.md`](docs/plan-destilacion.md) (módulos retirados, ADR-0015).

---

## Arquitectura general

El sistema corre como **tres procesos independientes** que se comunican por HTTP:

```
┌─────────────────────────────────────────────────────────────┐
│                     PROCESO PIPELINE (Python)               │
│                                                             │
│  Micrófono ──► AudioProcessor ──► AnomalyDetector           │
│                  (features DSP)    (Isolation Forest)       │
│                                         │ is_anomaly        │
│  Cámara ──► FrameRingBuffer             ▼                   │
│              └► MotionDetector ──► EventSegmenter           │
│                 (MOG2, mov. principal)  │ evento cerrado    │
│                                         ▼                   │
│                       EventWriter (hilo aparte):            │
│                       EventStore + MultimodalEncoder        │
│                       + FAISSStore + Database (SQLite)      │
└─────────────────────────────┬───────────────────────────────┘
                              │ POST /internal/score (cada ventana)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PROCESO API (FastAPI)                   │
│                                                             │
│  REST: /events/…  /events/{id}/spectrogram  /search/…       │
│  WebSocket: /ws/stream  (broadcast en tiempo real)          │
└─────────────────────────────┬───────────────────────────────┘
                              │ HTTP + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            DASHBOARD (SPA React + TypeScript · web/)        │
│                                                             │
│        Monitor en vivo │ Eventos │ Búsqueda por similitud   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de datos

```
ventana de audio (4096 muestras @ 16 kHz = 0.256 s, sin solapamiento, hora de captura)
  └─► AudioProcessor.process_window()
        ├─ Scattering1D (Kymatio)      →  234 coeficientes (J=8, Q=8)
        ├─ Wavelet energy (db4, l=7)   →    8 valores
        ├─ RMS, ZCR                    →    2 valores
        ├─ spectral centroid/flatness/rolloff/bandwidth → 4 valores
        └─ deltas (RMS, centroide, energía scattering)  → 3 valores   = 251 dims
  └─► AnomalyDetector.score()   scaler → PCA(25) → IsolationForest (cadena congelada por fit)
        └─► AnomalyResult (raw_score, anomaly_score ∈ [0,1], is_anomaly tras 3 ventanas)
  └─► EventSegmenter: ventanas anómalas consecutivas → UN evento (cierre tras 2 s sin anomalía)
        └─► EventWriter (hilo aparte)
              ├─ eventos/<timestamp>/audio.wav      (pre-roll de ~3 s + evento)
              ├─ eventos/<timestamp>/frame.jpg      (frame del pico)
              ├─ eventos/<timestamp>/metadata.json
              ├─ Wav2Vec2(audio) ⊕ DINOv2(frame) → 1536-dim → eventos/<timestamp>/embedding.npy
              ├─► FAISS IndexFlatIP (similitud coseno)
              └─► SQLite (metadata, inicio/fin/duración, faiss_index_id)
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

### Segmentación en ventanas

En vivo, `sounddevice` entrega bloques de `window_size = 4096` muestras **sin
solapamiento** (`blocksize = window_size`): una ventana cada 0.256 s a 16 kHz.
`DSPConfig.hop_size` y `AudioProcessor.segment_signal()` solo aplican al
procesamiento offline de señales completas.

### Espectrograma mel (por evento)

Para inspeccionar un evento guardado, `GET /events/{id}/spectrogram` calcula bajo demanda
el espectrograma mel en dB (`librosa`, 128 bandas) de su audio. El dashboard lo muestra en
la tarjeta del evento con el botón **Espectrograma**. No interviene en la detección.

---

## Detección de anomalías

### Isolation Forest

El detector usa **Isolation Forest** (`scikit-learn`), un algoritmo de detección de anomalías no supervisado basado en el principio de que los puntos anómalos son más fáciles de aislar: requieren menos particiones aleatorias para quedar separados.

#### Protocolo de entrenamiento online (sin `partial_fit`)

IsolationForest no soporta aprendizaje incremental. Se usa un protocolo de **buffer ring + refit periódico**:

```
buffer = deque(maxlen=500)   # vectores CRUDOS (sin normalizar)

por cada ventana de audio:
    si no fitted:
        buffer.append(x)
        si len(buffer) == 500: fit(buffer)          # primer entrenamiento
    si fitted:
        score = cadena_congelada(x)                 # scaler → PCA → IF del último fit
        si score no es anómalo: buffer.append(x)    # las anómalas quedan en cuarentena
        si llegaron 200 ventanas nuevas: fit(buffer)

fit(buffer): aprende scaler (z-score) → PCA(25) → IsolationForest
             y fija el umbral = percentil 2 de los scores de ese buffer
```

El modelo se adapta al perfil reciente de la señal sin aprender las anomalías: solo entran
al buffer las ventanas normales, salvo que una anomalía persista ~60 s y se acepte como
nuevo normal.

#### Normalización del score

`IsolationForest.score_samples()` devuelve valores negativos (típicamente entre -0.7 y -0.3). Se normalizan a `[0, 1]` usando **min-max calibrado** sobre el buffer de entrenamiento:

```python
# 0 = más anómalo, 1 = más normal → se invierte
anomaly_score = 1.0 - (raw_score - buffer_min) / (buffer_max - buffer_min)
anomaly_score = clip(anomaly_score, 0.0, 1.0)
```

Una ventana es anómala cuando `raw_score` cae bajo el **umbral fijado en el último fit**:
el percentil 2 de los scores del buffer de entrenamiento (con
`enable_adaptive_threshold=False`, `model.offset_`). Se confirma tras 3 ventanas
consecutivas (histéresis). Las ventanas anómalas quedan en **cuarentena** fuera del
buffer de entrenamiento; si la anomalía dura más de ~60 s
(`accept_after_windows`) se acepta como nuevo normal y se fuerza un refit
([ADR-0014](docs/adr/0014-nucleo-de-deteccion-funcional.md)).

Las ventanas anómalas consecutivas forman **un solo evento** (`src/events/`): se
cierra tras 2 s sin anomalía (`EVENT_RELEASE_S`) o a los 60 s
(`EVENT_MAX_DURATION_S`) y guarda inicio, fin, duración y el frame del pico. La
escritura y los embeddings corren en un hilo aparte.

#### Fase de calentamiento

Durante las primeras 500 ventanas (~128 s de audio a 16 kHz, ventanas de 4096), el detector devuelve `is_fitted=False` y `anomaly_score=0.0`. El dashboard muestra "Calentando..." durante este período.

### PCA — Reducción de dimensionalidad

Antes del Isolation Forest, el vector de 251 dimensiones se normaliza (z-score con la media
y la desviación del buffer de entrenamiento) y se reduce a **25 componentes** con **PCA**.
Scaler, PCA e IF se ajustan juntos en cada fit y se usan sin cambios hasta el siguiente.
La normalización existe por el PCA, que es sensible a la escala; el Isolation Forest por sí
solo no lo es.

```python
Xz = (X - mean) / std             # estadísticas del buffer de entrenamiento
pca = PCA(n_components=25)
model.fit(pca.fit_transform(Xz))  # 251 → 25 dims
```

### C2ST — Detección de drift por Classifier Two-Sample Test

> **Opcional, apagado por defecto** (`ENABLE_DRIFT=true` para activarlo). El AUC
> tiene un sesgo conocido: los buffers comparados se solapan (300 de 500 muestras
> con refits cada 200), lo que lo empuja hacia 0.5 aunque haya drift. Ver
> [ADR-0014](docs/adr/0014-nucleo-de-deteccion-funcional.md).

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
| `timestamp` | DATETIME(tz) | Inicio del evento (hora de captura, UTC) |
| `end_timestamp` | DATETIME(tz) | Fin del evento (nullable en eventos antiguos) |
| `duration_s` | FLOAT | Duración en segundos (nullable en eventos antiguos) |
| `anomaly_score` | FLOAT | Score normalizado [0,1] en el pico del evento |
| `event_dir` | TEXT UNIQUE | Ruta al directorio del evento |
| `faiss_index_id` | INTEGER | ID en el índice FAISS |
| `audio_path` | TEXT | Ruta a `audio.wav` |
| `frame_path` | TEXT | Ruta a `frame.jpg` (nullable) |
| `embedding_path` | TEXT | Ruta a `embedding.npy` |
| `source_region_json` | TEXT | Bounding boxes de movimiento (JSON) |
| `extra_json` | TEXT | Metadatos adicionales (JSON) |
| `top_audio_features` | TEXT | Features de audio que más se desviaron en el pico (JSON) |

Las columnas nuevas se añaden solas a bases existentes (migración nullable, ADR-0011). Una
base creada con v0.3 puede conservar columnas de fusión/video ya retiradas; quedan sin uso.

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

## Movimiento en video (contexto del evento)

`MotionDetector` usa **MOG2 (Mixture of Gaussians v2)** de OpenCV:

1. **Background subtraction MOG2** → máscara de foreground binaria
2. **Morphological Open** (erosión + dilatación) → elimina ruido sal-y-pimienta
3. **Dilatación** → conecta regiones fragmentadas del mismo objeto
4. **`findContours`** → extrae contornos del foreground
5. Filtrado por `min_contour_area=500 px²` → descarta falsos positivos pequeños
6. `boundingRect` sobre cada contorno → `BoundingBox(x, y, w, h, area)`
7. **Merge de cajas cercanas:** cajas cuyas aristas están a ≤`merge_gap` px se fusionan en una sola (default: 0 = solo cajas que se tocan/superponen). Una caja fusionada no puede superar `max_box_ratio` (40%) del área del frame, evitando mega-cajas.
8. **Novedad (IoU tracking):** cada caja se compara con las del frame anterior por IoU. Cajas nuevas (IoU<0.3) reciben `temporal_weight=1.0`; persistentes (IoU≥0.3), `0.5`.
9. **Movimiento principal:** el pipeline calcula `motion_score = area_ratio × temporal_weight` y conserva la caja de mayor score (movimiento grande y nuevo primero).

```python
boxes = motion_detector.detect(frame)       # BoundingBox con temporal_weight
# pipeline: b.motion_score = (b.area / frame_area) * b.temporal_weight → top-1
```

El movimiento principal y `motion_energy` (fracción del frame cubierta por cajas) son
**contexto visual**: no intervienen en la decisión y no indican qué produjo el sonido
([ADR-0015](docs/adr/0015-destilacion-del-sistema.md)). Cada evento guarda el frame del
momento de mayor score.

---

## API REST

Base URL: `http://localhost:8000`

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/events/` | Lista eventos (filtros: `limit`, `offset`, `min_score`) |
| `GET` | `/events/{id}` | Detalle de un evento |
| `GET` | `/events/{id}/audio` | Stream del archivo WAV |
| `GET` | `/events/{id}/frame` | Stream del JPEG |
| `GET` | `/events/{id}/frame/annotated` | Frame JPEG con la caja de movimiento principal (rectángulo rojo + label `motion X.XXX`) |
| `GET` | `/events/{id}/spectrogram` | Espectrograma mel (dB) del audio del evento, calculado bajo demanda |
| `DELETE` | `/events/{id}` | Elimina un evento (fila SQLite, archivos y su vector FAISS) |
| `DELETE` | `/events/` | Elimina todos los eventos y resetea el índice FAISS |
| `POST` | `/search/similar` | Upload audio/imagen → top-k eventos similares por cosine (máx. 10 MB) |
| `GET` | `/search/similar/by-event/{id}` | Busca eventos similares a uno ya almacenado usando su embedding pre-computado. Instantáneo (no carga modelos). Excluye el evento fuente. |
| `WS` | `/ws/stream` | WebSocket: push de `AnomalyScoreMessage` en tiempo real (score, umbral, estado, cajas con `motion_score`, `motion_energy`, `rms`, `top_audio_features`, `drift_auc`) |
| `POST` | `/internal/score` | Usado internamente por el pipeline para broadcast WS |
| `POST` | `/internal/reset-detector` | Señaliza al pipeline que resetee su `AnomalyDetector` |
| `GET` | `/internal/reset-pending` | Polling del pipeline: retorna `{"pending": bool}` y limpia el flag |
| `GET` | `/health` | Healthcheck: `{"status":"ok"}` |

Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI).

**CORS:** habilitado vía `CORSMiddleware` para el SPA desacoplado. Orígenes permitidos por
la variable `CORS_ORIGINS` (default `http://localhost:5173`, el dev server de Vite).

---

## Dashboard web (SPA React · Mission Control)

El dashboard es una **SPA en React + TypeScript + Vite** en [`web/`](web/) que consume la
API (REST + WebSocket). Decisión: [ADR-0013](docs/adr/0013-reemplazo-dashboard-streamlit-por-spa.md);
contrato con la API en [`docs/frontend-migration/03-contrato-api-frontend.md`](docs/frontend-migration/03-contrato-api-frontend.md).

**Lenguaje visual "Mission Control":** tema oscuro, tipografía técnica (Chakra Petch +
JetBrains Mono), telemetría en mono tabular y un estado de sistema inequívoco (NOMINAL /
CALIBRANDO / ANOMALÍA).

| Página | Contenido |
|---|---|
| **Monitor en vivo** | Score y umbral en tiempo real (uPlot), estado del detector, motion energy, drift AUC (si `ENABLE_DRIFT`), RMS, distribución KDE de scores, movimiento principal, *top contributors* de la última anomalía; botones "Reiniciar historial" y "Reiniciar detector" (con confirmación) |
| **Eventos** | Tarjetas con score, frame anotado, audio y **espectrograma** bajo demanda; eliminar uno o todos |
| **Búsqueda** | Subir audio o imagen, o elegir un evento guardado (embedding precalculado, instantáneo); resultados por similitud coseno |

Tiempo real *event-driven* (`useAnomalyStream`): WebSocket con reconexión (backoff +
jitter), heartbeat y detección de *staleness*. Estado de servidor con TanStack Query.
Accesibilidad WCAG AA y `prefers-reduced-motion`.

```bash
cd web
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                 # http://localhost:5173
```

Requiere la API con CORS habilitado (`CORS_ORIGINS`). Detalle en [`web/README.md`](web/README.md).

---

## Estructura del proyecto

```
dsp-anomaly-detection/
├── src/
│   ├── dsp/                   # AudioProcessor (Scattering + wavelet + RMS/ZCR + espectrales + deltas), DSPConfig
│   ├── detection/             # BaseAnomalyDetector (scaler→PCA→IF, umbral por fit, cuarentena),
│   │                          #   AnomalyDetector, DetectorConfig, SnapshotStore (opcional)
│   ├── events/                # EventSegmenter (un evento por anomalía) + EventWriter (hilo aparte)
│   ├── embeddings/            # Wav2Vec2 + DINOv2 → 1536-dim (carga lazy, offline-first)
│   ├── storage/               # EventStore (WAV/JPEG/NPY/JSON), Database (SQLite), FAISSStore
│   ├── vision/                # FrameCapture, MotionDetector (MOG2 + merge + IoU), BoundingBox
│   ├── sync/                  # FrameRingBuffer: frame más cercano a la hora de captura del audio
│   ├── api/                   # FastAPI: main.py (create_app), schemas.py, routers/ (events, search, websocket)
│   └── pipeline.py            # Orquestador: captura → features → detector → eventos → API
├── web/                       # Dashboard SPA (React + TS + Vite) — "Mission Control"
│   └── src/                   # api/ · hooks/ · lib/ · components/ · pages/ (LiveMonitor, EventFeed,
│                              #   SimilaritySearch) · layout/
├── scripts/
│   └── replay.py              # Banco offline: escenarios con verdad conocida + huella de detección
├── tests/                     # pytest, sin micrófono ni cámara (arrays sintéticos)
├── docs/
│   ├── adr/                   # Decisiones de arquitectura (MADR)
│   ├── architecture/          # Visión general de la arquitectura vigente
│   ├── plan-correcciones.md   # Núcleo corregido (ADR-0014): plan, bitácora, resultados
│   └── plan-destilacion.md    # Módulos retirados (ADR-0015): plan, bitácora, resultados
├── pyproject.toml · poetry.lock
└── .env.example
```

---

## Requisitos del sistema

- Python 3.11–3.14 (ver `pyproject.toml`)
- **PortAudio** (librería de sistema que usa `sounddevice`; sin ella el pipeline y sus tests
  no cargan): `brew install portaudio` en macOS, `sudo apt-get install libportaudio2` en
  Debian/Ubuntu
- Micrófono y cámara (solo para el pipeline en tiempo real; los tests no los requieren)
- ~1.5 GB de espacio para modelos HuggingFace en cache
- Conexión a internet solo para la descarga inicial de modelos
- Node.js para el dashboard web

---

## Instalación y configuración

### 1. Clonar e instalar dependencias

```bash
git clone <repo>
cd dsp-anomaly-detection
pip install poetry
poetry install
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` si se quieren cambiar los valores por defecto:

```env
EVENTS_DIR=./eventos      # directorio donde se guardan los eventos
DB_PATH=./data/events.db  # base de datos SQLite
FAISS_PATH=./data/faiss.index
API_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:5173   # orígenes permitidos para el SPA (coma-separados)
# Opcionales, apagados por defecto (no deciden eventos — ADR-0014):
ENABLE_DRIFT=false                   # C2ST + refits sensibles a drift
ENABLE_SNAPSHOTS=false               # guarda el modelo en disco en cada refit
# Segmentación de eventos:
EVENT_RELEASE_S=2.0                  # silencio que cierra un evento
EVENT_MAX_DURATION_S=60.0            # duración máxima de un evento
```

> El `.env` de la raíz se **carga automáticamente** al arrancar el pipeline y la API
> (`python-dotenv`). Una variable ya exportada en el shell tiene prioridad sobre el `.env`.
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
curl http://localhost:8000/health
# → {"status":"ok"}
```

Documentación interactiva: `http://localhost:8000/docs`

### Terminal 2 — Dashboard web

```bash
cd web
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                 # http://localhost:5173
```

Requiere que la API tenga `CORS_ORIGINS` incluyendo el origen del SPA (default
`http://localhost:5173`).

### Terminal 3 — Pipeline (requiere micrófono y cámara)

```bash
poetry run python -m src.pipeline
```

El pipeline:
1. Abre el micrófono y la cámara (dispositivo 0 por defecto)
2. Entra en la fase de calentamiento (~128 s): acumula 500 ventanas de audio y entrena el detector
3. Detecta anomalías, agrupa cada una en un evento y actualiza el dashboard en tiempo real

Al actualizar desde una versión anterior a ADR-0014, el estado guardado del detector se
descarta una vez (vuelve a calentar). Para detener: `Ctrl+C` (los eventos pendientes se
guardan antes de salir).

---

## Tests

### Ejecutar la suite completa

```bash
poetry run pytest          # 224 tests
```

### Ejecutar por módulo

```bash
poetry run pytest tests/test_dsp.py                  # 21 — features DSP
poetry run pytest tests/test_detection.py            # 26 — Isolation Forest, PCA, drift
poetry run pytest tests/test_base_detector.py        #  9 — BaseAnomalyDetector (cuarentena, calibración, persistencia)
poetry run pytest tests/test_detection_regressions.py #  4 — warmup limpio, anomalía sostenida, nuevo normal, falsas alarmas
poetry run pytest tests/test_drift_refit.py          #  6 — refits por drift, snapshots, explicabilidad
poetry run pytest tests/test_events.py               #  9 — segmentación de eventos y escritor asíncrono
poetry run pytest tests/test_storage.py              # 40 — FAISS, SQLite, EventStore
poetry run pytest tests/test_schema_migration.py     #  3 — compatibilidad de esquema (v0.2 y v0.3)
poetry run pytest tests/test_api.py                  # 37 — endpoints REST + WebSocket
poetry run pytest tests/test_embeddings.py           # 19 — encoders (mockeados)
poetry run pytest tests/test_vision.py               # 29 — MOG2, merge, IoU, movimiento principal
poetry run pytest tests/test_sync.py                 #  7 — frame más cercano por hora de captura
poetry run pytest tests/test_pipeline.py             # 14 — process_window, eventos, hora de captura
```

Todos los tests usan datos simulados (arrays numpy sintéticos, frames aleatorios, modelos
mockeados). No requieren micrófono, cámara ni modelos descargados; sí la librería PortAudio.

```bash
cd web && npm test         # tests del dashboard (Vitest)
cd web && npm run build    # typecheck + build
```

### Banco de reproducción offline (sin hardware)

`scripts/replay.py` pasa audio sintético —o un WAV propio— por `Pipeline.process_window`,
el mismo camino que en vivo, con embeddings simulados y almacenamiento temporal:

```bash
poetry run python -m scripts.replay --scenario events   # 8 min con 7 sonidos conocidos
poetry run python -m scripts.replay --scenario quiet    # 15 min de fondo (falsas alarmas)
poetry run python -m scripts.replay --wav grabacion.wav # WAV mono de 16 kHz, sin verdad conocida
```

Reporta, por sonido insertado, si se detectó, cuántos eventos generó y qué parte de sus
ventanas cubrió; además falsas alarmas por hora, tiempo por ventana, ventanas perdidas en una
cola simulada y una **huella** (`flags_sha1`, `scores_sha1`) que permite comprobar que un
cambio no altera las decisiones del detector.

---

## Stack tecnológico

| Categoría | Librería | Versión | Uso |
|---|---|---|---|
| DSP | PyWavelets | ^1.9 | Wavelet DWT, energía por escala |
| DSP | Kymatio | ^0.3 | Scattering Transform 1D |
| DSP | librosa | ^0.11 | Espectrograma mel de los eventos |
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
| Frontend SPA | uPlot / Recharts | ^1 / ^2 | Gráficos realtime / estáticos |
| Frontend SPA | Vitest + MSW | ^2 | Tests del SPA |
| HTTP | httpx | — | Notificación pipeline → API |

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

`soundfile` usa PCM_16 por defecto, que **recorta** amplitudes fuera de `[-1, 1]`. Las señales de audio del pipeline pueden tener valores de mayor amplitud (no normalizadas). Se fuerza `subtype='FLOAT'` para preservar el rango completo float32, necesario para reproducir fielmente los eventos y calcular bien su espectrograma.

### conftest.py con OMP_NUM_THREADS=1

En macOS, PyTorch y OpenCV ambos linkan contra el framework Accelerate/OpenMP. Cuando coexisten en el mismo proceso, sus pools de threads nativos compiten y producen segmentation faults no deterministas. Establecer `OMP_NUM_THREADS=1` antes de cualquier import serializa OpenMP y elimina el problema. Esto solo afecta al rendimiento de operaciones matriciales en tests (irrelevante); en el pipeline real cada proceso carga solo sus librerías.

---

## Limitaciones conocidas y trabajo futuro

| Limitación | Descripción | Posible mejora |
|---|---|---|
| Sin GPU | Todo corre en CPU por defecto | Cambiar `device='cuda'` en `EmbeddingConfig` si hay GPU disponible |
| FAISS no distribuido | Índice en un solo archivo local | Migrar a Qdrant para escalado horizontal |
| Sin localización de la fuente | El "movimiento principal" es solo área × novedad; no indica qué produjo el sonido | Localización acústica con un array de micrófonos |
| Embeddings de audio entrenados con voz | Wav2Vec2-base se preentrenó con habla; la búsqueda de sonidos ambientales no está medida | Medir con datos etiquetados; evaluar modelos entrenados en AudioSet |
| Refit dentro del loop de audio | Cada ~200 ventanas el fit tarda 0.2–0.3 s; hoy lo absorbe la cola sin perder audio | Entrenar en un hilo aparte reutilizando el swap atómico existente |

> **Resuelto en v0.3:** sincronización audio-video por timestamp ([ADR-0003](docs/adr/0003-timestamp-based-av-synchronization.md)) y `motion_energy` acotado a `[0,1]`.
> **Resuelto en v0.3.1:** integridad FAISS↔SQLite cross-proceso (`IndexIDMap2` con IDs = PK + escritura atómica + file-lock; hallazgos C3/C4), path traversal en endpoints de streaming (H2) y rutas de evento absolutas (H3).
> **Resuelto después:** núcleo de detección corregido —warmup, umbral, un evento por anomalía, hora de captura— ([ADR-0014](docs/adr/0014-nucleo-de-deteccion-funcional.md)); fusión multimodal, doble horizonte, EMD y Streamlit retirados ([ADR-0015](docs/adr/0015-destilacion-del-sistema.md)).

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
