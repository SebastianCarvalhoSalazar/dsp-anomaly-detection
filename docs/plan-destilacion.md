# Plan de destilación — retirar lo que no decide sin tocar el núcleo

> Documento vivo. Se actualiza en cada paso con avances, hallazgos y resultados.
> Rama: `claude/repo-description-refgrl` (reiniciada desde `main` tras el merge del PR #8).
> Antecedente: [`plan-correcciones.md`](plan-correcciones.md) (núcleo de detección corregido).

## Estado

| Paso | Estado | Commit |
|---|---|---|
| 0 — Línea base y huella de detección | ✅ Hecho | (este commit) |
| 1 — EMD fuera, mel como detalle del evento | ⏳ | |
| 2 — Retirar fusión, detector de video y modelos lentos | ⏳ | |
| 3 — "Fuente probable" → "movimiento principal" | ⏳ | |
| 4 — Retirar Streamlit | ⏳ | |
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
3. `pytest` completo verde; `pylint src/` ≥ 8.72; smoke de imports del pipeline y la API.
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
