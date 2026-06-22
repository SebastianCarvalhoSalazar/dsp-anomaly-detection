# ADR-0012: Canal de control dashboard → pipeline en vivo

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** post-v0.3

## Contexto y problema

La estrategia de fusión, el peso de audio y el gating por fusión solo se
podían fijar al **arrancar** el pipeline (env vars), y los controles del
dashboard (selector/slider) solo afectaban la **visualización**, no el
pipeline. El sistema corre como **tres procesos separados**: el dashboard no
comparte estado con el pipeline. Se necesita poder gobernar la fusión del
pipeline **en tiempo real** desde el frontend.

## Opciones consideradas

1. **Solo env vars** (status quo) — requiere reiniciar el pipeline; sin control
   en vivo.
2. **Canal de control vía la API** (reutilizar el patrón de "reset detector") —
   el dashboard hace POST a la API, que guarda la config en `app.state`; el
   pipeline pollea por GET y la aplica.
3. **Bus de mensajes / socket directo dashboard↔pipeline** — más infraestructura
   de la necesaria para un MVP local.

## Decisión

Opción 2, reusando el mecanismo ya existente del reset:

- `POST /internal/fusion-config` (dashboard → API): valida `strategy` contra el
  registro de estrategias (422 si es desconocida) y clampa `audio_weight` a
  `[0,1]`; guarda en `app.state.fusion_config`.
- `GET /internal/fusion-config` (pipeline → API): el pipeline lo pollea en el
  mismo intervalo que el reset y aplica la config con `_apply_fusion_config`
  **solo cuando cambió** (rebuild de la estrategia + toggle de gating).
- El dashboard hace POST cuando el usuario cambia selector / slider / toggle.

El polling corre en el hilo del `_process_loop`, que es el mismo que usa
`fusion_strategy`/`_fusion_gates`, así que no hace falta locking.

## Consecuencias

- **Positivas:** control en vivo de estrategia, peso y gating sin reiniciar;
  reusa infraestructura probada (mismo patrón que el reset); sin dependencias
  nuevas; el env var `FUSION_GATES_DECISION` sigue sirviendo como default inicial.
- **Negativas / costos:** estado en `app.state` es por-proceso (con
  `--workers N` no se comparte); el polling añade un GET ligero cada ~50 ventanas;
  el dashboard recalcula además la fusión client-side para feedback instantáneo
  (lógica duplicada, mitigada al compartir `src/fusion`).
- **Riesgos y mitigaciones:** config inválida → validada en la API (422) y
  re-validada en el pipeline (estrategia desconocida se ignora). Fallo de red en
  el poll → silencioso (debug log), el pipeline mantiene su config actual.

## Notas de implementación

- `src/api/schemas.py::FusionConfigMessage`; `src/api/main.py` inicializa
  `app.state.fusion_config`; endpoints en `src/api/routers/websocket.py`.
- `src/pipeline.py`: `_apply_fusion_config`, `_check_fusion_config` (polled).
- `src/dashboard/api_client.py`: `set_fusion_config`, `get_fusion_config`.
- `src/dashboard/pages/live_monitor.py`: toggle de gating + POST on-change.
- Tests: `test_api.py` (endpoints), `test_pipeline.py` (apply/check),
  `test_dashboard.py` (cliente).
