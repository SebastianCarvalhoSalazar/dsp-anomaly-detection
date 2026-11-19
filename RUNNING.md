# Guía de ejecución

Instrucciones paso a paso para correr el sistema completo en local.

---

## Requisitos previos

- Python 3.11–3.14 y Poetry (`pip install poetry`)
- **PortAudio** (lo usa `sounddevice`): `brew install portaudio` en macOS,
  `sudo apt-get install libportaudio2` en Debian/Ubuntu
- Node.js (para el dashboard web)
- Micrófono; cámara opcional (sin ella los eventos se guardan sin frame)
- Internet solo para la descarga inicial de modelos (~400 MB, una vez)

---

## Paso 1 — Permisos en macOS

En **Configuración del Sistema → Privacidad y Seguridad**, activa **Micrófono** y
**Cámara** para tu terminal (Terminal, iTerm2, etc.). Sin permiso de cámara el pipeline
avisa y sigue solo con audio.

---

## Paso 2 — Instalar dependencias

```bash
cd dsp-anomaly-detection
poetry install
cd web && npm install && cd ..
```

---

## Paso 3 — Crear el archivo `.env`

```bash
cp .env.example .env
cp web/.env.example web/.env
```

Los valores por defecto sirven para correr en local. Variables disponibles en
`.env.example` (rutas, CORS, `ENABLE_DRIFT`, `ENABLE_SNAPSHOTS`, `EVENT_RELEASE_S`,
`EVENT_MAX_DURATION_S`).

---

## Paso 4 — Descargar modelos de HuggingFace (opcional)

El pipeline y la API los cargan en segundo plano al arrancar. Para descargarlos antes:

```bash
poetry run python - << 'EOF'
from src.embeddings import MultimodalEncoder
print("Descargando Wav2Vec2-base y DINOv2-base (~400 MB)...")
MultimodalEncoder().ensure_downloaded()
print("Listo. El sistema funcionará offline a partir de ahora.")
EOF
```

---

## Paso 5 — Arrancar la API (terminal 1)

```bash
poetry run uvicorn src.api.main:app --reload
```

Cuando aparezca `Application startup complete.`:

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

Swagger UI: `http://localhost:8000/docs`.

---

## Paso 6 — Arrancar el dashboard web (terminal 2)

```bash
cd web
npm run dev
```

Abre `http://localhost:5173`. Mientras el pipeline no esté corriendo, el indicador de
conexión mostrará que no llegan datos.

---

## Paso 7 — Arrancar el pipeline (terminal 3)

```bash
poetry run python -m src.pipeline
```

- **Calentamiento (~128 s / 500 ventanas):** el detector acumula audio para su primer
  entrenamiento; el dashboard muestra "CALIBRANDO". Si existe un estado guardado
  compatible (`data/detector_state.pkl`), se salta. Al actualizar desde una versión
  anterior a ADR-0014 ese estado se descarta una vez.
- **Después:** cada ventana (0.256 s) se puntúa en tiempo real. Tres ventanas anómalas
  seguidas confirman una anomalía; las ventanas anómalas consecutivas forman **un solo
  evento**, que se cierra tras 2 s sin anomalía y se guarda (audio, frame del pico,
  embedding) sin frenar el audio.
- **Detener:** `Ctrl+C`. Los eventos pendientes se guardan antes de salir.

---

## Resumen de terminales

| Terminal | Comando | URL |
|---|---|---|
| 1 — API | `poetry run uvicorn src.api.main:app --reload` | `http://localhost:8000` |
| 2 — Dashboard | `cd web && npm run dev` | `http://localhost:5173` |
| 3 — Pipeline | `poetry run python -m src.pipeline` | — |

---

## Qué probar en el dashboard

**Monitor en vivo**
- Haz un sonido de unos segundos (aplausos, música): el score sube por encima del umbral y
  el estado pasa a ANOMALÍA.
- La traza muestra el score y el **umbral de decisión** en la misma escala.
- **Top contributors** muestra las features de audio que más se desviaron en la última
  anomalía.
- **Motion energy** y **Movimiento principal** muestran la actividad de la cámara (contexto;
  no indican qué produjo el sonido).
- **Drift AUC** solo tiene valores si arrancas el pipeline con `ENABLE_DRIFT=true`.
- **Reiniciar historial** limpia los gráficos; **Reiniciar detector** vuelve al
  calentamiento (pide confirmación).

**Eventos**
- Un sonido de varios segundos debe producir **una sola tarjeta**, con su frame anotado y
  su audio.
- **Espectrograma** muestra el espectrograma mel del audio del evento.
- Se puede eliminar un evento o todos.

**Búsqueda**
- Sube un WAV o una imagen, o elige un evento guardado (instantáneo, usa su embedding ya
  calculado), y verás los eventos más parecidos por similitud coseno.
- Requiere al menos un evento guardado; máximo 10 MB por archivo.

---

## Sin cámara

El pipeline funciona sin cámara: los eventos se guardan sin `frame.jpg` y la mitad visual
del embedding se rellena con ceros.

---

## Tests y banco offline

```bash
poetry run pytest                                       # suite completa, sin hardware
cd web && npm test && npm run build && cd ..            # dashboard
poetry run python -m scripts.replay --scenario events   # banco con 7 sonidos conocidos
```

El banco pasa audio sintético por el mismo camino que en vivo y reporta detecciones,
eventos por sonido, falsas alarmas y una huella del detector (ver README, sección Tests).

---

## Solución de problemas

**`OSError: PortAudio library not found`**
Instala PortAudio (ver requisitos previos).

**`ModuleNotFoundError` al arrancar**
```bash
poetry install
```

**La cámara no abre**
El pipeline sigue solo con audio. En macOS revisa el permiso de cámara de la terminal.

**El dashboard no recibe datos en tiempo real**
Verifica que la API y el pipeline estén corriendo y que `CORS_ORIGINS` (en `.env`)
incluya `http://localhost:5173`.

**Error al descargar modelos**
```bash
rm -rf ~/.cache/huggingface/hub/models--facebook*
poetry run python - << 'EOF'
from src.embeddings import MultimodalEncoder
MultimodalEncoder().ensure_downloaded()
EOF
```

**Segmentation fault en macOS al correr tests**
Ya está resuelto en `tests/conftest.py`. Si ocurre en otro contexto:
```bash
OMP_NUM_THREADS=1 poetry run pytest
```
