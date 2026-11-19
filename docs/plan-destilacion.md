# Plan de destilación — retirar lo que no decide sin tocar el núcleo

> Documento vivo. Se actualiza en cada paso con avances, hallazgos y resultados.
> Rama: `claude/repo-description-refgrl` (reiniciada desde `main` tras el merge del PR #8).
> Antecedente: [`plan-correcciones.md`](plan-correcciones.md) (núcleo de detección corregido).

## Estado

| Paso | Estado | Commit |
|---|---|---|
| 0 — Línea base y huella de detección | ✅ Hecho | `7413a53` |
| 1 — EMD fuera, mel como detalle del evento | ✅ Hecho (SPA sin compilar aquí) | `9cde0d7` |
| 2 — Retirar fusión, detector de video y modelos lentos | ✅ Hecho (SPA sin compilar aquí) | `b1ccb2a` |
| 3 — "Fuente probable" → "movimiento principal" | ✅ Hecho (SPA sin compilar aquí) | `7a88055` |
| 4 — Retirar Streamlit (adelantado, antes del 2) | ✅ Hecho | ver `git log` |
| 5 — Dependencias y código muerto | ✅ Hecho | `dfb9baf` |
| 6 — Documentación | ✅ Hecho | (este commit) |
| Validación en el Mac del usuario (SPA + prueba real) | ✅ Hecha (ver bitácora) | |

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

| Medida | Valor (Linux x86, este entorno) |
|---|---|
| `pytest` | 279 passed |
| `pylint src/` | 8.72/10 |
| `events`: eventos / falsas alarmas | 6 / 0 |
| `events`: `flags_sha1` / `scores_sha1` | `b325166bac49…` / `6c0e6f5e939d…` |
| `quiet`: eventos / falsas alarmas | 0 / 0 |
| `quiet`: `flags_sha1` / `scores_sha1` | `ab6e36037b9e…` / `d0f2f8200e19…` |

Determinismo comprobado: dos corridas seguidas de `events` dan hashes e informe idénticos.

## Resultados (línea base `main` → final)

| Medida | Antes | Después |
|---|---|---|
| Huella `events` / `quiet` (`flags_sha1`, `scores_sha1`) | — | **idéntica en todos los pasos** |
| `events`: eventos guardados / falsas alarmas | 6 / 0 | 6 / 0 |
| `quiet`: falsas alarmas en 15 min | 0 | 0 |
| Tests del núcleo y código de `detection`/`dsp`/`events` | — | **sin cambios** |
| `pytest` | 279 passed | 224 passed (los tests retirados cubrían código retirado; +5 nuevos) |
| Avisos de `pylint` | 140 | 123 (ninguno nuevo; nota 8.72 → 8.63, ver Paso 2) |
| Python en `src/` | 6269 líneas, 57 archivos | 3843 líneas, 41 archivos |
| SPA `web/src` (`.ts`/`.tsx`) | 2783 líneas | 2300 líneas |
| Paquetes en `poetry.lock` | 166 | 113 |
| Endpoints retirados | — | `/internal/fusion-config` (GET/POST), `/events/{id}/offline_analysis` |
| Endpoint nuevo | — | `/events/{id}/spectrogram` |

Diff total respecto a `main` en `src`, `tests`, `web/src` y `scripts`: 253 inserciones,
3823 borrados.

## Pendientes

- ✅ Dashboard: `npm test` (11) y `npm run build` en verde, también en CI (ver bitácora).
- ✅ `zustand` retirado.
- ✅ Carga real de Wav2Vec2 + DINOv2 con el entorno reducido (validado en el Mac del usuario).
- ✅ **Prueba real** en el Mac del usuario: un sonido → un evento con frame, audio, espectrograma
  y "Movimiento principal" (ver bitácora).

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
- **2026-11-19 · Paso 5** — `pyproject.toml` pierde 13 dependencias directas (`emd-signal`,
  `streamlit`, `plotly`, `qdrant-client`, `asyncpg`, `psycopg2-binary`, `seaborn`, `matplotlib`,
  `torchmetrics`, `einops`, `timm`, `accelerate`, `av`). `poetry lock` sin `--regenerate`:
  **166 → 113 paquetes, ninguna versión cambiada, ninguno añadido**. Entorno sincronizado
  (`poetry sync`) para que la puerta corra sin ellos. Código muerto retirado: `AudioWindow`,
  `MotionDetector.detect_with_result` + `MotionResult` + `draw_boxes` (solo los usaban sus tests).
  - Se mantiene `AudioProcessor.process_window_with_meta`: solo lo usa `tests/test_dsp.py`, que
    está en la lista de tests del núcleo que no se modifican.
  - Se mantienen `torchvision` y `torchaudio`: los procesadores de imagen de transformers pueden
    requerir `torchvision`, y cargar los modelos reales no es verificable aquí (Hugging Face
    bloqueado).
  - **Puerta:** huella idéntica; `pytest` 224 passed (salen 6 tests del código muerto); `pylint`
    125 → 123 avisos, ninguno nuevo; imports de transformers/librosa/torchaudio OK.
- **2026-11-19 · Paso 6** — Documentación al día: README (introducción, arquitectura, flujo,
  detector, espectrograma, movimiento principal, API, dashboard, estructura, requisitos con
  PortAudio, instalación, ejecución, tests con recuentos reales, banco de reproducción, stack,
  limitaciones), `RUNNING.md` reescrito, `CLAUDE.md` reescrito, `.env.example`, `web/README.md`,
  ADR-0015 nuevo, estados actualizados en ADR 0001–0014 y en el índice,
  `architecture/multimodal-overview.md` → `architecture/overview.md` (arquitectura vigente),
  contrato API ↔ frontend reescrito desde `types.ts`, y cabecera "documento histórico" en
  `REPORT.md` y en la documentación de la migración del frontend.
  - **Hallazgos al verificar contra el código:** el README decía que borrar un evento dejaba
    huérfano su vector FAISS (el código lo elimina por ID desde el fix C3); indicaba Python
    3.11–3.12 (el proyecto admite 3.11–3.14), un warmup de "~25 s / 200 ventanas" (son ~128 s /
    500) y una carpeta `doc/` que no existe. Corregidos.
  - **Verificación:** todos los enlaces relativos de la documentación apuntan a archivos
    existentes (salvo el marcador de la plantilla de ADR); los términos retirados solo aparecen
    en documentos marcados como históricos o en notas de compatibilidad.
- **2026-11-19 · Merge de `main`** — Entraron en `main` los PR #9 (licencia MIT, CI de
  GitHub Actions, recuentos de tests) y #10 (servidor PulseAudio ficticio en el CI).
  Conflictos solo en `README.md` y `RUNNING.md`: se conservan las insignias de `main` sobre la
  descripción nueva y el recuento de tests de esta rama (224). Comprobado como en el CI:
  `pylint src/ --fail-under=8.5` pasa (8.63) y `pytest` 224 passed; huella idéntica. Se
  actualiza el comentario del workflow que citaba la nota anterior (8.72).
- **2026-11-19 · Validación en el Mac del usuario** (otro agente) — Wav2Vec2 + DINOv2 cargan con
  el entorno reducido (primera carga 42 s con descarga; embedding de 1536 dims, norma 1, sin NaN;
  sin frame la mitad visual queda en ceros). Huella `main` = rama en ambos escenarios.
  - **Hallazgo:** en el Mac `flags_sha1` coincide con el de Linux, pero `scores_sha1` no
    (`events`: `5031774e…` vs `6c0e6f5e…`; `quiet`: `a46bcf5c…` vs `d0f2f820…`), aunque dentro
    de cada máquina `main` y la rama coinciden. El punto flotante varía entre CPU/BLAS:
    `scores_sha1` solo es comparable en la misma máquina. Documentado en README, `CLAUDE.md` y
    `scripts/replay.py`.
  - `npm` quedó bloqueado también allí: sin credenciales para el registro privado.
- **2026-11-19 · Dashboard al npm público** (autorizado por el usuario) — `web/package-lock.json`:
  solo cambian las 383 URLs `resolved` al registro público; versiones e `integrity` idénticas, y
  `npm ci` verificó cada tarball. `web/.npmrc` fija el registro público.
  - **Línea base** (`main`, worktree temporal): 7 archivos / 19 tests de Vitest, build OK.
  - **Rama:** 6 archivos / 11 tests, `tsc` + build OK **sin correcciones**: los cambios del SPA
    hechos a mano en los Pasos 1–3 compilan. Los 8 tests de menos son los 5 de `fusion.test.ts` y
    los 3 de `slowTileDisplay`, retirados con su código.
  - `npm uninstall zustand`: salen `zustand` y `use-sync-external-store`, nada más cambia
    (`useAnomalyStream` usa el `useSyncExternalStore` nativo de React). Tests y build en verde.
  - CI: nuevo job `web` (`npm ci`, `npm test`, `npm run build`).
- **2026-11-19 · Prueba real (Mac del usuario)** — Un sonido generó **un** evento: audio de 5 s
  (3 s de pre-roll + 2 s de cierre), frame y frame anotado (`motion 0.022`), espectrograma,
  embedding y vector FAISS. Duración 0.64 s = mínimo posible (una sola ventana confirmada).
- **2026-11-19 · Post-merge** (rama reiniciada desde `main` tras el PR #11):
  - **Duración de los eventos validada:** `scripts/replay.py` reporta ahora la duración guardada.
    Tono 15 s → 15.49 s; chirp 3 s → 3.46 s; doble ráfaga 4 s → 4.22 s (un solo evento);
    impulso 0.3 s → 0.64 s (mínimo).
  - **`window_index` congelado en 499** (detectado en la prueba real; bug anterior a ADR-0014):
    valía `len(buffer) - 1`. Ahora cuenta las ventanas desde el arranque o el reset. Huella
    idéntica.
  - **Sesgo del C2ST, corregido.** Las filas compartidas por dos buffers sucesivos aparecían en
    entrenamiento con una etiqueta y en prueba con la otra: AUC crudo ~0.10, simetrizado ~0.90.
    Medido con la configuración real (500/200) y datos estables: **0.84–0.86 antes, 0.52–0.54
    después**; drift sostenido con refits cada 10 ventanas: ~0.52 antes, 1.0 después. Afecta a
    lo observado en vivo con `ENABLE_DRIFT=true` (AUC 0.64–0.84 y un refit "drift"): eran
    compatibles con el artefacto. Huella idéntica (el drift está apagado por defecto).
  - **Debilidad de la cuarentena (ADR-0014), reproducida, pendiente:** en vivo, tras subir el
    ruido de fondo (RMS mediano 0.0041 → ~0.007) aparecieron 17 eventos en ~2 min. Con el banco
    (`--wav`, fondo sintético que sube 1.7× a los 4 min, sin sonidos): **25 eventos en 8 min**,
    sin adaptación, con drift apagado o encendido. Causa: la anomalía es intermitente; cada vez
    que se corta, la cuarentena se descarta y nunca llega a `accept_after_windows` seguidas; el
    buffer solo recibe las ventanas más tranquilas. Mitigación inmediata: "Reiniciar detector".
