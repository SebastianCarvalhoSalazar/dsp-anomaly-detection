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

## Paso 4 — Descargar modelos de HuggingFace

Solo hay que hacerlo una vez. Los modelos quedan en el cache local y todas las ejecuciones posteriores son completamente offline.

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

**Fase de calentamiento (~25 segundos):** el detector acumula 200 ventanas de audio para entrenar el Isolation Forest. Durante este período no se detectan anomalías.

**Después del calentamiento:** el detector puntúa cada ventana en tiempo real. Si el score supera el umbral, se guarda un evento completo (audio, frame, embedding) y el dashboard se actualiza.

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
- El historial de scores y amplitud RMS se actualiza cada segundo
- **Reiniciar historial**: limpia los gráficos de la sesión (no afecta el detector)
- **Reiniciar detector**: envía señal al pipeline para reiniciar el Isolation Forest desde cero (reinicia el warmup de ~25s)

**Event Feed**
- Lista de anomalías detectadas con audio reproducible y frame de video (si hay cámara)
- Filtra por score mínimo con el slider
- **Eliminar evento**: borra un evento individual (DB + archivos en disco)
- **Borrar todo**: elimina todos los eventos y resetea el índice FAISS

**Similarity Search**
- Sube un archivo WAV o imagen y busca los eventos más similares en el índice
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
# 145+ tests, ~4 segundos
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
