# Guía de ejecución

Instrucciones paso a paso para correr el sistema completo en local.

---

## Requisitos previos

- Python 3.11–3.12 instalado
- Poetry instalado (`pip install poetry`)
- Micrófono disponible
- Cámara disponible (opcional — el sistema funciona sin ella)
- Conexión a internet para la descarga inicial de modelos (~400 MB, solo una vez)

---

## Paso 1 — Permisos de cámara en macOS

Ve a **Configuración del Sistema → Privacidad y Seguridad → Cámara** y activa el toggle para tu aplicación de terminal (Terminal, iTerm2, etc.).

Sin este permiso el pipeline lanza un warning pero continúa funcionando solo con audio.

---

## Paso 2 — Instalar dependencias

```bash
cd dsp-idea
poetry install
```

---

## Paso 3 — Crear el archivo `.env`

```bash
cat > .env << 'EOF'
EVENTS_DIR=./eventos
DB_PATH=./data/events.db
FAISS_PATH=./data/faiss.index
API_BASE_URL=http://localhost:8000
EOF
```

---

## Paso 4 — Descargar modelos de HuggingFace (opcional)

Los modelos se descargan automáticamente al iniciar el pipeline o la API (en background threads). Si prefieres descargarlos manualmente por adelantado:

```bash
poetry run python - << 'EOF'
from src.embeddings import MultimodalEncoder
print("Descargando Wav2Vec2-base y DINOv2-base (~400 MB)...")
MultimodalEncoder().ensure_downloaded()
print("Listo. El sistema funcionará offline a partir de ahora.")
EOF
```

---

## Paso 5 — Arrancar la API

Abre una terminal en el directorio del proyecto:

```bash
poetry run uvicorn src.api.main:app --reload
```

Cuando aparezca `Application startup complete.` está lista. Verifica que responde:

```bash
curl http://localhost:8000/events/
# → []
```

La documentación interactiva (Swagger UI) está en `http://localhost:8000/docs`.

---

## Paso 6 — Arrancar el dashboard

Abre una **segunda terminal**:

```bash
poetry run streamlit run src/dashboard/app.py
```

Se abre automáticamente `http://localhost:8501` en el navegador. El Live Monitor mostrará "Waiting for pipeline connection…" hasta que el pipeline esté corriendo.

---

## Paso 7 — Arrancar el pipeline

Abre una **tercera terminal**:

```bash
poetry run python -m src.pipeline
```

**Fase de calentamiento (~2 minutos / 500 ventanas):** el detector acumula ventanas de audio para entrenar el Isolation Forest. Si existe un estado guardado de una sesión anterior (`data/detector_state.pkl`), el warmup se salta completamente.

**Preloading de modelos:** al iniciar el pipeline, los modelos Wav2Vec2 y DINOv2 se descargan/cargan en un background thread en paralelo con el warmup, para que la primera anomalía se procese sin delay adicional.

**Después del calentamiento:** el detector puntúa cada ventana en tiempo real. Si el score supera el umbral (con hysteresis de 3 ventanas consecutivas), se guarda un evento completo (audio, frame, embedding, motion_energy) y el dashboard se actualiza.

Para detener el pipeline: `Ctrl+C`

---

## Resumen de terminales

| Terminal | Comando | URL |
|---|---|---|
| 1 — API | `poetry run uvicorn src.api.main:app --reload` | `http://localhost:8000` |
| 2 — Dashboard | `poetry run streamlit run src/dashboard/app.py` | `http://localhost:8501` |
| 3 — Pipeline | `poetry run python -m src.pipeline` | — |

---

## Qué probar en el dashboard

Una vez el pipeline está corriendo, abre `http://localhost:8501` y explora las cuatro páginas del sidebar:

**Live Monitor**
- Habla o haz ruido cerca del micrófono — verás el anomaly score subir en tiempo real
- El **historial de scores** muestra una línea punteada del **umbral adaptativo** (percentil 98 de scores normalizados); se actualiza cada segundo
- **Amplitud RMS**: barras que muestran la energía sonora reciente
- **Distribución KDE**: gráfico de densidad estimada por kernel de los scores recientes, con línea vertical del umbral. Permite ver si los scores tienden a concentrarse en zona normal o anormal
- **Drift AUC**: métrica del test C2ST (Classifier Two-Sample Test) que mide si la distribución de features está cambiando. 0.5 = sin drift, ≥ 0.8 = drift significativo
- **Top drift features**: cuando hay drift, muestra ⚠️ con las features más relevantes (e.g., `scat_45, wavelet_band_5, spectral_centroid`) para entender qué aspecto del audio cambió
- **Motion energy**: indicador que muestra el nivel de actividad visual detectada por la cámara
- **Fuente probable**: chip rojo que muestra las coordenadas y `source_score` de la caja que más correlaciona con la anomalía acústica
- **Reiniciar historial**: limpia los gráficos de la sesión (no afecta el detector)
- **Reiniciar detector**: envía señal al pipeline para reiniciar el Isolation Forest desde cero (reinicia el warmup)

**Event Feed**
- Lista de anomalías detectadas con audio reproducible y frame de video anotado (bounding box de la fuente dibujado en rojo)
- Filtra por score mínimo con el slider
- **Eliminar evento**: borra un evento individual (DB + archivos en disco)
- **Borrar todo**: elimina todos los eventos y resetea el índice FAISS

**Similarity Search**
- Dos modos de búsqueda:
  - 📁 **Subir archivo**: sube un WAV o imagen y busca los eventos más similares (la primera búsqueda puede tardar ~60s por carga de modelos; las siguientes son instantáneas gracias al preloading)
  - 📋 **Evento existente**: selecciona un evento del desplegable y busca similares usando su embedding pre-computado → **respuesta instantánea** sin necesidad de cargar modelos
- Resultados con frames anotados (bounding box de fuente) y score de similitud
- Requiere al menos un evento guardado en el índice FAISS
- Límite de 10 MB por archivo subido

**Offline Analysis**
- Selecciona un evento del desplegable y haz clic en "Ejecutar análisis"
- Muestra la descomposición EMD (IMFs) y el mel-spectrogram en dB del audio del evento

---

## Sin cámara

El pipeline funciona perfectamente sin cámara. Los eventos se guardan sin `frame.jpg` y el embedding multimodal usa solo la mitad de audio (la mitad visual se rellena con ceros). No hay ninguna configuración adicional necesaria.

---

## Tests automatizados

Para verificar que todo el código funciona correctamente sin necesidad de hardware:

```bash
poetry run pytest -v
# 198 tests, ~4 segundos
```

### Smoke test del pipeline DSP sin micrófono

```bash
poetry run python - << 'EOF'
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

## Solución de problemas

**`ModuleNotFoundError` al arrancar**
```bash
poetry install  # reinstalar dependencias
```

**La cámara no abre**
El pipeline continúa con solo audio. Para darle permiso en macOS: Configuración del Sistema → Privacidad y Seguridad → Cámara → activar la terminal.

**El dashboard no recibe datos en tiempo real**
Verifica que los tres procesos (API, dashboard, pipeline) están corriendo simultáneamente en terminales separadas.

**Error al descargar modelos**
```bash
# Limpiar el cache y reintentar
rm -rf ~/.cache/huggingface/hub/models--facebook*
poetry run python - << 'EOF'
from src.embeddings import MultimodalEncoder
MultimodalEncoder().ensure_downloaded()
EOF
```

**Segmentation fault en macOS al correr tests**
Ya está resuelto en `tests/conftest.py`. Si ocurre en otro contexto, antepón:
```bash
OMP_NUM_THREADS=1 poetry run pytest -v
```
