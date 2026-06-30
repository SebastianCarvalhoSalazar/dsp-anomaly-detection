# ADR-0013: Reemplazo del dashboard Streamlit por un SPA (React + TypeScript)

- **Estado:** Aceptado
- **Fecha:** 2026-06-30
- **Decisores:** Sebastian Carvalho
- **Fase:** Frontend (post-v0.3)

## Contexto y problema

El dashboard actual (`src/dashboard/`) está construido en Streamlit y, aunque funcional, arrastra
límites estructurales del modelo de ejecución de Streamlit que **no son arreglables de raíz** sin
salir de él (ver análisis detallado en
[`docs/frontend-migration/01-analisis-dashboard-actual.md`](../frontend-migration/01-analisis-dashboard-actual.md)):

- El "tiempo real" se implementa con `time.sleep(1)` + `st.rerun()` (B1): re-render de página
  completa cada segundo, techo de latencia de 1 s, pérdida de scroll y alto consumo de CPU.
- El WebSocket no se reconecta (B2) y el indicador de conexión es permanentemente verdadero tras el
  primer mensaje (B3): no hay detección de caída/*staleness*.
- El estado de los controles de fusión se pierde al navegar por el GC de estado de widgets de
  Streamlit (B4), forzando un *workaround* frágil (lo que parchea la rama
  `fix/dashboard-fusion-state-persist`).
- El dashboard importa `src.fusion.make_strategy` (B5), acoplándose al paquete Python del backend e
  impidiendo un despliegue independiente.
- Accesibilidad y diseño responsive deficientes (B10, B11).

La API FastAPI ya es un backend limpio y desacoplado (REST + WebSocket `/ws/stream`), por lo que el
dashboard es un mero consumidor del contrato y la migración a una SPA es directa.

## Opciones consideradas

1. **Seguir en Streamlit y parchear** — bajo costo inmediato, pero B1/B4/B5/B10/B11 son límites del
   *runtime* de Streamlit; el techo de calidad es bajo y la deuda persiste.
2. **Dash / Plotly** — sigue acoplando UI a Python y a Plotly (pesado para realtime); no resuelve el
   modelo de re-ejecución ni el despliegue independiente.
3. **SPA en React + TypeScript + Vite** *(elegida)* — desacople total vía el contrato HTTP/WS,
   tiempo real *event-driven*, accesibilidad y rendimiento controlables, ecosistema maduro
   (TanStack Query, gráficos uPlot/Recharts), despliegue independiente.
4. **Vue 3 + Vite** — equivalente técnico válido; menor familiaridad/ecosistema para este caso.
5. **SvelteKit** — *bundles* más pequeños y *stores* reactivos, pero ecosistema más chico.

## Decisión

Reemplazar el dashboard Streamlit por una **SPA en React 18 + TypeScript (strict) + Vite 5**, que
consume la API existente; gráficos ligeros para tiempo real (**uPlot** + **Recharts**), TanStack
Query para estado de servidor y un hook propio `useAnomalyStream` para el WebSocket. El plan completo
está en [`docs/frontend-migration/02-plan-migracion-spa.md`](../frontend-migration/02-plan-migracion-spa.md).

## Consecuencias

- **Positivas:** tiempo real sin polling ni re-render global; WebSocket con reconexión y detección de
  *staleness*; estado de fusión persistente; desacople del paquete Python; accesibilidad WCAG AA y
  responsive *mobile-first*; despliegue independiente del frontend; base testeable (Vitest/MSW).
- **Negativas / costos:** se introduce una *toolchain* de Node/Vite en el repo; trabajo de migración
  por fases; mantener temporalmente dos dashboards (Streamlit + SPA) hasta la paridad.
- **Riesgos y mitigaciones:** falta de CORS (→ añadir `CORSMiddleware`); tormentas de reconexión
  (→ backoff + jitter, una sola conexión); divergencia del recompute de fusión en TS (→ reimplementar
  solo las 4 estrategias triviales con tests de paridad; `combined_score` del backend autoritativo);
  `/internal/*` sin auth (→ documentar; proteger antes de red no confiable); retirar Streamlit antes
  de tiempo (→ decomisionar solo tras *sign-off* de paridad).

## Notas de implementación

- **Ubicación:** SPA en `web/` (hermano de `src/`), aislado de Poetry; `web/node_modules` y
  `web/dist` en `.gitignore`.
- **Backend:** añadir `CORSMiddleware` en `src/api/main.py` (orígenes vía env `CORS_ORIGINS`, default
  `http://localhost:5173`); endpoint `GET /health`; en producción servir `web/dist` con
  `StaticFiles` + *fallback* a `index.html` (single-origin → CORS no-op).
- **Contrato:** tipos TS espejo de `src/api/schemas.py`; el SPA replica el keep-alive del WS y el
  swap de esquema `http→ws`. Detalle en
  [`docs/frontend-migration/03-contrato-api-frontend.md`](../frontend-migration/03-contrato-api-frontend.md).
- **Compatibilidad / decomisión:** Streamlit permanece operativo hasta el cutover (F7). En F8 se
  elimina `src/dashboard/`, se quitan `streamlit`/`plotly` de `pyproject.toml` y se rompe el
  acoplamiento `src.fusion` del dashboard.
- **Continuidad visual:** reutilizar `PALETTE` de `src/dashboard/styles.py` como *tokens* de Tailwind.
