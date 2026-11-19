# Plan de destilación — retirar lo que no decide sin tocar el núcleo

> Documento vivo. Se actualiza en cada paso con avances, hallazgos y resultados.
> Rama: `claude/repo-description-refgrl` (reiniciada desde `main` tras el merge del PR #8).
> Antecedente: [`plan-correcciones.md`](plan-correcciones.md) (núcleo de detección corregido).

## Estado

| Paso | Estado | Commit |
|---|---|---|
| 0 — Línea base y huella de detección | ✅ Hecho | (este commit) |
| 1 — EMD fuera, mel como detalle del evento | ✅ Hecho (SPA sin compilar aquí) | ver bitácora |
| 2 — Retirar fusión, detector de video y modelos lentos | ✅ Hecho (SPA sin compilar aquí) | ver bitácora |
| 3 — "Fuente probable" → "movimiento principal" | ✅ Hecho (SPA sin compilar aquí) | ver bitácora |
| 4 — Retirar Streamlit (adelantado, antes del 2) | ✅ Hecho | ver bitácora |
| 5 — Dependencias y código muerto | ⏳ | |
| 6 — Documentación | ⏳ | |
| Validación en el Mac del usuario (SPA + prueba real) | ⏳ pendiente | |

## Contexto

Con el núcleo ya corregido y medido, quedan módulos que no intervienen en la decisión y
cuestan mantenimiento: el EMD, el detector de video con calibración y 4 estrategias de
fusión (apagados por defecto, nunca validados), los modelos lentos, el dashboard Streamlit
(duplica al SPA) y dependencias sin importar. El recuadro "fuente probable" promete una
correlación con el audio que no existe (el score de audio era el mismo para todas las cajas).

Decisiones: quitar EMD y mover el espectrograma mel a un detalle del evento; eliminar
Streamlit ahora (con validación del SPA antes del merge); renombrar "fuente probable" a
"movimiento principal".

## Qué se queda y qué se va

| Se va | Se queda |
|---|---|
| EMD (`PyEMD`), página "Análisis offline" | Espectrograma mel, como detalle del evento |
| Detector de video, calibración, fusión, control de fusión en vivo | Cámara, frame por evento, `motion_energy` |
| Modelos lentos (doble horizonte) | Drift y snapshots (opcionales, apagados) |
| Dashboard Streamlit | Dashboard web (SPA) |
| Dependencias sin uso | `torchvision`, `torchaudio` |

## Garantía de calidad del núcleo

Cada paso debe pasar esta puerta antes del commit:

1. **Huella de detección idéntica** a la línea base: `scripts/replay.py` reporta
   `flags_sha1` (decisiones por ventana) y `scores_sha1` (raw scores a 6 decimales), además
   del informe por sonido. Escenarios `events` y `quiet`.
2. Los tests del núcleo (`test_detection_regressions.py`, `test_detection.py`,
   `test_base_detector.py`, `test_events.py`, `test_dsp.py`) pasan **sin modificarse**.
3. `pytest` completo verde; `pylint src/` **sin avisos nuevos** (criterio corregido en el Paso 2,
   ver bitácora); smoke de imports del pipeline y la API.
4. Sin restos de lo retirado (grep) fuera de documentos históricos.
5. Antes del merge, en el Mac del usuario: `npm test`, `npm run build` y una prueba real.

## Línea base (`main` en `47e2c13`)

| Medida | Valor |
|---|---|
| `pytest` | 279 passed |
| `pylint src/` | 8.72/10 |
| `events`: eventos / falsas alarmas | 6 / 0 |
| `events`: `flags_sha1` / `scores_sha1` | `b325166bac49…` / `6c0e6f5e939d…` |
| `quiet`: eventos / falsas alarmas | 0 / 0 |
| `quiet`: `flags_sha1` / `scores_sha1` | `ab6e36037b9e…` / `d0f2f8200e19…` |

Determinismo comprobado: dos corridas seguidas de `events` dan hashes e informe idénticos.

---

## Bitácora

- **2026-11-19 · Paso 0** — Rama reiniciada desde `main`. `scripts/replay.py` añade
  `flags_sha1` y `scores_sha1` al informe. Línea base registrada arriba.
- **2026-11-19 · Paso 1** — `GET /events/{id}/offline_analysis` (EMD + mel) reemplazado por
  `GET /events/{id}/spectrogram` (solo mel; `SpectrogramResponse {sample_rate, spectrogram}`).
  SPA: se borran la página "Análisis offline", `ImfChart` y la ruta `/offline`; la tarjeta de
  evento gana un botón "Espectrograma" que lo carga bajo demanda (`useSpectrogram`, reutiliza
  `Heatmap`). En Streamlit se quita la página EMD para no dejarla rota hasta el Paso 4.
  - **Puerta:** huella `events` y `quiet` idéntica; `pytest` 278 passed (sale el test del cliente
    Streamlit de EMD); `pylint` 8.76; imports OK.
  - **Limitación:** sin compilador de TypeScript en este entorno (las dependencias del SPA vienen
    de un registro npm privado): los cambios del SPA se revisaron a mano y quedan para la puerta
    en el Mac (`npm test`, `npm run build`).
- **2026-11-19 · Paso 4 (adelantado)** — El Streamlit importaba `src.fusion` directamente:
  retirar la fusión primero lo dejaba roto entre commits, y reescribirlo para borrarlo después
  no tenía sentido. Se adelanta: se borran `src/dashboard/` y `tests/test_dashboard.py`
  (las dependencias `streamlit`/`plotly` salen en el Paso 5).
  - **Puerta:** huella idéntica; `pytest` 254 passed (salen 24 tests del cliente Streamlit);
    `pylint` 8.72 (baja de 8.76 porque el código retirado puntuaba por encima de la media,
    sigue ≥ línea base); imports OK.
- **2026-11-19 · Paso 2** — Se borran `src/fusion/` y `src/vision_detection/`; el pipeline
  pierde detector de video, calibradores, estrategias de fusión, modelos lentos y el sondeo
  de `/internal/fusion-config`. La decisión de evento es `result.is_anomaly` (lo que ya
  ocurría por defecto). API: fuera `/internal/fusion-config` (GET/POST), `FusionConfigMessage`
  y los campos WS de fusión/video/lento; `EventResponse` sin campos de fusión. ORM: se dejan de
  mapear las columnas de fusión/lentos; las bases existentes las conservan (nullable) y
  SQLAlchemy las ignora — test nuevo con una BD v0.3 que lee y vuelve a insertar.
  SPA: fuera `FusionControls`, el store Zustand, `lib/fusion.ts` y los tiles lentos/video;
  `useResetDetector` pasa a `hooks/useDetector.ts`; los "top contributors" quedan solo con audio.
  - **Puerta:** huella idéntica; `pytest` 228 passed (salen `test_fusion`, `test_calibration`,
    `test_video_detection` y 5 tests de fusion-config del pipeline; entran 3 tests nuevos).
  - **Hallazgo / criterio corregido:** `pylint` baja de 8.72 a 8.62 **sin ningún aviso nuevo**
    (140 → 125 avisos): la nota es un promedio y el código borrado tenía pocos avisos por línea.
    La puerta "nota ≥ 8.72" no sirve para una tarea que borra código; pasa a ser "sin avisos
    nuevos" (diff de avisos antes/después, ignorando números de línea).
  - **Pendiente en el Mac:** `zustand` queda sin uso en `web/package.json`, pero quitarlo exige
    regenerar `package-lock.json` contra el registro npm privado (`npm uninstall zustand`).
- **2026-11-19 · Paso 3** — `BoundingBox.source_score` se divide en `temporal_weight` (1.0
  nueva / 0.5 persistente) y `motion_score = area_ratio × temporal_weight`. Se quita el factor
  `anomaly_score`: es el mismo para todas las cajas de un frame, así que no cambiaba el ganador.
  JSON de cajas y WebSocket usan la clave `motion_score`; `/events/{id}/frame/annotated` lee
  `motion_score` con respaldo a `source_score` para eventos antiguos y rotula `motion X.XXX`.
  SPA: el recuadro se llama "Movimiento principal".
  - **Diferencia de comportamiento (solo visual):** durante el warmup el score de audio era 0,
    todas las cajas empataban y ganaba la más grande; ahora gana la de mayor área × novedad.
  - **Puerta:** huella idéntica; `pytest` 230 passed (+2: el recuadro correcto se dibuja con
    la clave nueva y con la vieja, comprobado por color de píxel); `pylint` sin avisos nuevos.
