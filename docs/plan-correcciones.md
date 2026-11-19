# Plan de correcciones — hacer funcional el núcleo de detección

> Documento vivo. Se actualiza en cada fase con avances, hallazgos y resultados
> de validación. Rama de trabajo: `claude/repo-description-refgrl`.

## Estado

| Fase | Estado | Commit |
|---|---|---|
| 0 — Línea base y banco | ✅ Hecha | `eb0c263` |
| 1 — Warmup limpio (F1) | ✅ Hecha | `844af74` |
| 2 — Umbral estable (F2) | ✅ Hecha | `844af74` |
| 3 — Calibrador y fusión (F3) | ✅ Hecha | `b155b90`, `267170a` |
| 4 — Sincronización A/V (F4) | ✅ Hecha | `267170a` |
| 5 — Un evento por anomalía (F5) | ✅ Hecha | `e76a336`, `267170a` |
| 6 — Apagar lo que no decide | ✅ Hecha (código + docs); ⚠️ SPA sin validar | `267170a`, `d389f40` |
| Validación final | ✅ Banco y suites Python; ⚠️ pendiente SPA y prueba real | — |

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
- *Real:* el test usa fondo de habitación (no ruido blanco) y exige PC1 < 90 %: ese fondo varía lentamente de volumen y eso mueve todas las energías juntas (PC1 ≈ 52 % legítimo). La firma del fallo era 99.99 %.

### Fase 2 — Umbral estable (F2) · `detection/base.py`
- Umbral fijado en cada fit a partir de los scores del buffer de entrenamiento.
- Las ventanas anómalas no entran al buffer mientras haya anomalía; tras N s (configurable, 60 s) se aceptan como nuevo normal.
- Test: tono de 15 s detectado en ≥ 90 % de sus ventanas; ruido nuevo de 3 min deja de alarmar tras la aceptación; ruido estacionario ≤ 2 ráfagas / 2000 ventanas.
- *Real:* en los tests unitarios, tono de 15 s (60 ventanas) ✅; aceptación con `accept_after_windows=40` y 160 ventanas de ruido nuevo (para que el test sea rápido) ✅; ≤ 2 ráfagas en 500 ventanas ✅. La medida larga de falsas alarmas la da el banco: 0 en 15 min (3515 ventanas).

### Fase 3 — Calibrador y fusión (F3) · `fusion/calibration.py`, `pipeline.py`
- Rango medio para empates; el calibrador solo recibe scores de detectores entrenados; fusión neutra durante el warmup.
- Test: historial constante → 0.5; con `FUSION_GATES_DECISION=true`, cero eventos en warmup.

### Fase 4 — Sincronización A/V (F4) · `pipeline.py`
- El callback encola `(muestras, hora de captura)`; `nearest(ts, max_delta)`; sin frame cercano → evento sin frame.
- Test: con 3 s de retraso simulado se elige el frame del momento real (± 40 ms).

### Fase 5 — Un evento por anomalía (F5) · `EventSegmenter` + `EventWriter`
- Máquina de estados IDLE → ACTIVE → COOLDOWN (*real:* IDLE → ACTIVE; `release_s` cumple el papel del cooldown); evento con pre-roll + duración, pico de score y frame del pico; columnas nullable `end_timestamp`, `duration_s` (migración ADR-0011).
- Escritura y embeddings en un hilo con cola acotada.
- Test: una ráfaga → 1 evento; dos cercanas → 1; tono sostenido → exactamente 1 evento; con encoder lento (2 s) cada ventana se procesa en < 50 ms.
- *Real:* el test exige < 200 ms para no ser frágil en máquinas lentas; el banco mide p99 = 36 ms con encoder de 2 s.

### Fase 6 — Apagar por defecto lo que no decide
- Variables de entorno (apagadas por defecto): detector de video + calibración + fusión, C2ST + refits por drift, snapshots.
- ADR-0014; corregir README y comentarios erróneos.
- Test: `npm test` y `npm run build` del SPA siguen pasando.
- *Real:* ⚠️ no ejecutado (ver "Pendientes").

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

## Resultados de la validación final

Banco `scripts/replay.py` (mismos escenarios y semilla) sobre el commit sin
arreglos (`eb0c263`) y sobre el código corregido. Embeddings simulados.

**Escenario `events` (8 min, 7 sonidos insertados)**

| Sonido | Antes: eventos guardados / cobertura de ventanas | Después |
|---|---|---|
| impulso 0.3 s | 1 / 0 % | 1 / 0 % (inicio ±0.06 s) |
| ráfaga 2 s | **8** / 78 % | **1** / 78 % |
| tono 15 s | **11** / **18 %** | **1** / **97 %** |
| doble ráfaga (2 × 1.5 s, 1 s de separación) | 1 + no detectada la 2.ª / 17 % y 0 % | **1 evento para ambas** (fusión intencional) / 67 % y 71 % |
| chirp 3 s | **no detectado** | **1** / 85 % |
| armónico 4 s | **no detectado** | **1** / 88 % |
| **Total de eventos guardados** | **21** | **6** |

**Criterios de aceptación**

| Métrica | Objetivo | Antes | Después |
|---|---|---|---|
| Sonidos de > 1 s detectados | 100 %, inicio ≤ 1 s | 3 de 5 (chirp y armónico no) | 5 de 5; error de inicio ≤ 0.11 s |
| Eventos por sonido real | exactamente 1 | hasta 11 | 1 |
| Falsas alarmas, `quiet` 15 min | ≤ línea base | 0 | 0 |
| Falsas alarmas, `events` | ≤ línea base | 0 | 0 |
| Tiempo por ventana p99 (encoder instantáneo) | < 256 ms | 39.6 ms | 40.1 ms |
| Tiempo por ventana p99 (encoder de 2 s) | < 256 ms | **2034 ms** | **36.4 ms** |
| Ventanas perdidas (cola simulada, encoder de 2 s) | 0 | **13** | **0** |
| `pytest` (suite completa) | sin fallos nuevos | 258 passed | 279 passed |
| `pylint src/` | sin regresión | 8.68/10 | 8.72/10 |
| `npm test` / `npm run build` (SPA) | sin fallos nuevos | ⚠️ no ejecutado | ⚠️ no ejecutado |

Los tests nuevos se ejecutaron también contra el código anterior: los de F1–F6
fallan ahí (3 de 4 en detección, 7 de 7 en pipeline) y pasan con el corregido.

## Pendientes y siguientes pasos

- **SPA sin validar.** `web/package-lock.json` resuelve los 383 paquetes contra un
  registro npm privado (`npm.artifacts.furycloud.io`) que no es accesible desde el
  entorno de trabajo; el intento de instalar desde el registro público fue
  bloqueado por la política de permisos de la sesión. Los cambios que tocan al SPA
  son aditivos (campos opcionales `end_timestamp`/`duration_s` en `/events`; el
  WebSocket sigue enviando todos los campos, con valores por defecto cuando un
  componente está apagado), pero hay que correr `npm test` y `npm run build` en una
  máquina con acceso a ese registro.
- **Prueba real** con micrófono y cámara, y con los modelos de Hugging Face
  descargados (no accesibles desde este entorno).
- **Refit dentro del loop de audio.** El fit (PCA + IsolationForest) corre en línea:
  max de 187–271 ms por ventana cada ~200 ventanas. Hoy lo absorbe la cola (0
  ventanas perdidas); se puede mover a un hilo aparte reutilizando el swap atómico
  que ya existe (H1).
- **Calidad de los embeddings** (Wav2Vec2 entrenado con voz) y features de audio
  redundantes: fuera del alcance de este plan; requieren medir la búsqueda por
  similitud con datos etiquetados.

## Riesgos
- El `detector_state.pkl` previo deja de ser compatible y se descarta (un warmup único).
- Cambia la cantidad de eventos generados (es el objetivo).

---

## Bitácora

Formato: fecha · fase · qué se hizo / qué se encontró / resultado.

- **2026-11-18 · Plan** — Documento creado. Se inicia la Fase 0.
- **2026-11-18 · Fase 0** — `Pipeline.process_window(window, ts)` extraído del loop
  de captura (refactor puro, commit `eb0c263`). `scripts/replay.py` creado: escenarios
  `events` (8 min, 7 eventos conocidos: impulso 0.3 s, ráfaga 2 s, tono 15 s, doble
  ráfaga con 1 s de separación, chirp 3 s, armónico 4 s) y `quiet` (15 min de fondo).
  Reporta cobertura por evento, eventos guardados por evento real, falsas alarmas/h,
  tiempo por ventana y ventanas perdidas en una simulación de cola en tiempo real
  (maxsize=64, una ventana cada 256 ms).
  - Hallazgo: el entorno completo instala torch con librerías CUDA (descarga pesada);
    mientras termina, los tests numéricos se corren en un venv mínimo.
- **2026-11-18 · Fases 1 y 2** — Implementadas en `detection/base.py` (commit `844af74`).
  - Buffer de vectores crudos; cada fit aprende `scaler → PCA → IsolationForest` y lo
    congela. Se elimina el normalizador Welford (y `freeze_normalizer_on_anomaly`).
  - Umbral fijo por fit = percentil 2 de los scores del buffer de entrenamiento
    (`enable_adaptive_threshold=False` sigue usando `offset_`).
  - Cuarentena: las ventanas anómalas (más una cola de `hysteresis_count` ventanas)
    no entran al buffer; si persisten `accept_after_windows=234` (≈60 s) se aceptan y
    se fuerza un refit (`refit_reason="accepted"`).
  - `adaptive_threshold` del WebSocket ahora es el mismo umbral de decisión llevado a
    la escala [0, 1] (antes: p98 de scores sin suavizar, otra cantidad).
  - Estado persistido versión 2; los `.pkl` viejos se ignoran (un warmup).
  - Comentarios corregidos: `contamination='auto'` → `offset_=-0.5`; 500 ventanas ≈ 128 s.
  - **Validación:** 4 tests nuevos en `tests/test_detection_regressions.py`. Contra el
    código anterior fallan 3 (PC1 = 99.99 %; tono detectado en 14 % de sus ventanas;
    no existe aceptación de nuevo normal); con el código nuevo pasan los 4. El cuarto
    (≤ 2 ráfagas de falsas alarmas en 500 ventanas de fondo) pasa en ambos: es una
    guarda contra regresiones. Suites de detección/DSP/drift: 66 passed.
  - Hallazgo 1: el propio test de F1 descubrió un bug en la primera versión del
    arreglo (la ventana que completa el warmup se añadía dos veces al buffer).
  - Hallazgo 2: los tests de drift inyectan cambios de distribución que ahora el
    detector trata como anomalías y deja en cuarentena; se les desactiva la cuarentena
    para probar la mecánica de drift aislada. **Cuarentena y refits por drift tiran en
    direcciones opuestas** (resistir el cambio vs adaptarse rápido): refuerza apagar
    el drift por defecto (Fase 6).
  - Hallazgo 3: confirmado en ejecución el sesgo del C2ST por buffers solapados: con
    refits cada 10 ventanas sobre un buffer de 40, el AUC cae de 0.99 a 0.52 aunque
    el drift continúa. El test que exigía que el *último* refit fuera "drift" dependía
    de la alineación de ventanas; ahora verifica que haya habido refits por drift.
  - Comportamiento nuevo: el refit programado ocurre en la ventana siguiente a la que
    completa `refit_every` muestras nuevas (antes, en la misma).
- **2026-11-18 · Fase 0 (línea base)** — Entorno completo instalado. Hizo falta la
  librería de sistema PortAudio (`libportaudio2`): sin ella `sounddevice` no carga y
  `tests/test_pipeline.py` no se puede ni importar. Con ella, la línea base da 258
  tests passed. Banco antes: ver "Resultados de la validación final".
- **2026-11-18 · Fase 3** — Rango medio en `PercentileCalibrator` (commit `b155b90`);
  en el pipeline los calibradores solo reciben scores de detectores entrenados y la
  fusión es neutra (`combined_score=0`) durante el warmup.
  - **Validación:** `test_ties_map_to_mid_rank_not_one` (historial constante → 0.5) y
    `test_f3_no_fused_events_during_warmup` (con `FUSION_GATES_DECISION=true`, 0
    eventos en 120 ventanas de warmup; con el código anterior falla).
- **2026-11-18 · Fase 4** — `capture_midpoint()` calcula el punto medio de captura con
  `currentTime − inputBufferAdcTime` del callback (si el host reporta ceros, asume
  que el bloque acaba de terminar). La cola transporta `(muestras, ts)`; el frame se
  busca con `max_delta = 0.2 s`.
  - **Validación:** 3 tests: el cálculo del punto medio, que el loop use la hora de
    captura aunque la ventana haya esperado en la cola, y que se elija el frame del
    momento real o ninguno si no hay uno cercano. Fallan con el código anterior.
- **2026-11-18 · Fase 5** — `src/events/` con `EventSegmenter` y `EventWriter`;
  `Pipeline.flush_events()` al apagar. Columnas `end_timestamp` y `duration_s`
  (migración nullable, ADR-0011) expuestas en `GET /events`.
  - Cambio respecto al plan: no hay un estado COOLDOWN aparte; el mismo
    `release_s` (2 s) que cierra el evento fusiona ráfagas separadas por menos que
    eso. Es más simple y cubre el caso.
  - **Validación:** 9 tests unitarios (`tests/test_events.py`) y
    `test_f5_sustained_tone_is_one_event_and_loop_never_blocks`: un tono de 10 s
    genera 1 evento (duración entre 8 y 12 s) y, con un encoder de 2 s, ninguna
    ventana tarda más de 0.2 s. Falla con el código anterior.
  - Hallazgo: dos expectativas mal planteadas en mis propios tests (la duración
    incluye la ventana que alcanza el tope; el evento se cierra en k=3, no k=4).
- **2026-11-18 · Fase 6** — `ENABLE_DRIFT`, `ENABLE_SNAPSHOTS`,
  `ENABLE_VIDEO_DETECTOR` apagados por defecto. Con el video apagado,
  `combined_score` es el score de audio calibrado. ADR-0014; README, CLAUDE.md y
  `.env.example` actualizados (incluye corregir el falso "solapamiento del 75 %").
  - **Validación:** 2 tests (valores por defecto y reactivación por variables).
  - **Bloqueo:** no se pudo instalar el SPA (ver "Pendientes").
- **2026-11-18 · Validación final** — Banco antes/después y suites completas. Todos
  los criterios del banco se cumplen; resultados arriba.
