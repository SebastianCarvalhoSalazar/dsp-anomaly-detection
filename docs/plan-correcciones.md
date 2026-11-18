# Plan de correcciones — hacer funcional el núcleo de detección

> Documento vivo. Se actualiza en cada fase con avances, hallazgos y resultados
> de validación. Rama de trabajo: `claude/repo-description-refgrl`.

## Contexto

Una revisión del código (noviembre 2026) encontró fallos que impiden que el
sistema cumpla su propósito básico —detectar anomalías acústicas y guardarlas
como eventos útiles—. Varios se confirmaron ejecutando el código del repo con
datos sintéticos:

| # | Fallo | Evidencia de la revisión |
|---|---|---|
| F1 | Las primeras 30 ventanas entran **sin normalizar** al buffer de entrenamiento | Norma media filas 0–29 = 7953 vs 4.3 del resto; PC1 del PCA explica el 100 % de la varianza |
| F2 | El umbral (percentil 2 de scores en vivo) **se auto-suprime** en anomalías largas | Tono de 1 kHz de 15 s detectado en 9 de 60 ventanas |
| F3 | El calibrador lleva los **empates a 1.0** y se fusiona durante el warmup | `calibrate(0.0)` tras 60 ceros = 1.0 → `combined=1.0`, `is_anomaly=True` |
| F4 | La sincronización A/V usa la hora de **salida de la cola**, no la de captura | `pipeline.py` toma `time.time()` al desencolar |
| F5 | **Un evento por ventana** anómala y embeddings **síncronos** en el loop de audio | Tono sostenido → 9 eventos casi idénticos; Wav2Vec2+DINOv2 bloquean el loop |

Además, varias piezas calculan resultados que no influyen en la decisión
(fusión, video, C2ST, snapshots) y la documentación describe un solapamiento
del 75 % que no existe en vivo.

## Fases

### Fase 0 — Línea base y banco de reproducción
- Instalar entorno completo; correr `pytest`, `pylint src/`, `npm test`, `npm run build` y anotar fallos previos.
- Extraer la lógica por ventana a `Pipeline.process_window(window, ts)` (refactor sin cambio de comportamiento).
- `scripts/replay.py`: pasa un WAV por ese método (encoder simulado, sin micrófono) y reporta eventos, falsas alarmas, latencia y tiempo por ventana.
- Escenario sintético: ruido de fondo con eventos conocidos insertados.
- Medir el estado **antes** de los arreglos.

### Fase 1 — Warmup limpio (F1) · `detection/base.py`
- Buffer de vectores **crudos**; en cada fit se ajusta `StandardScaler → PCA → IsolationForest` y se puntúa con la cadena congelada (sin Welford, sin desfase train/serve).
- Test: todas las filas del buffer a la misma escala; PC1 < 50 % de la varianza en ruido blanco.

### Fase 2 — Umbral estable (F2) · `detection/base.py`
- Umbral fijado en cada fit a partir de los scores del buffer de entrenamiento.
- Las ventanas anómalas no entran al buffer mientras haya anomalía; tras N s (configurable, 60 s) se aceptan como nuevo normal.
- Test: tono de 15 s detectado en ≥ 90 % de sus ventanas; ruido nuevo de 3 min deja de alarmar tras la aceptación; ruido estacionario ≤ 2 ráfagas / 2000 ventanas.

### Fase 3 — Calibrador y fusión (F3) · `fusion/calibration.py`, `pipeline.py`
- Rango medio para empates; el calibrador solo recibe scores de detectores entrenados; fusión neutra durante el warmup.
- Test: historial constante → 0.5; con `FUSION_GATES_DECISION=true`, cero eventos en warmup.

### Fase 4 — Sincronización A/V (F4) · `pipeline.py`
- El callback encola `(muestras, hora de captura)`; `nearest(ts, max_delta)`; sin frame cercano → evento sin frame.
- Test: con 3 s de retraso simulado se elige el frame del momento real (± 40 ms).

### Fase 5 — Un evento por anomalía (F5) · `EventSegmenter` + `EventWriter`
- Máquina de estados IDLE → ACTIVE → COOLDOWN; evento con pre-roll + duración, pico de score y frame del pico; columnas nullable `end_timestamp`, `duration_s` (migración ADR-0011).
- Escritura y embeddings en un hilo con cola acotada.
- Test: una ráfaga → 1 evento; dos cercanas → 1; tono sostenido → exactamente 1 evento; con encoder lento (2 s) cada ventana se procesa en < 50 ms.

### Fase 6 — Apagar por defecto lo que no decide
- Variables de entorno (apagadas por defecto): detector de video + calibración + fusión, C2ST + refits por drift, snapshots.
- ADR-0014; corregir README y comentarios erróneos.
- Test: `npm test` y `npm run build` del SPA siguen pasando.

## Criterios de aceptación (banco de reproducción, antes vs después)

| Métrica | Objetivo |
|---|---|
| Eventos del escenario detectados | 100 % de los de > 1 s; inicio con ≤ 1 s de error |
| Eventos guardados por anomalía real | exactamente 1 |
| Falsas alarmas en ruido estacionario | ≤ línea base |
| Tiempo por ventana (p99) | < 256 ms (duración de una ventana) |
| Ventanas de audio perdidas | 0 |
| Suites del repo | sin fallos nuevos |

Pendiente de validar fuera de este entorno (máquina del usuario): sesión real
con micrófono y cámara, y descarga de modelos de Hugging Face.

## Riesgos
- El `detector_state.pkl` previo deja de ser compatible y se descarta (un warmup único).
- Cambia la cantidad de eventos generados (es el objetivo).

---

## Bitácora

Formato: fecha · fase · qué se hizo / qué se encontró / resultado.

- **2026-11-18 · Plan** — Documento creado. Se inicia la Fase 0.
