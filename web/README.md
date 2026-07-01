# DSP Anomaly Dashboard (SPA)

Dashboard web en **React + TypeScript + Vite** que reemplaza al dashboard Streamlit.
Consume la API FastAPI existente (REST + WebSocket). Ver el plan en
[`docs/frontend-migration/`](../docs/frontend-migration/) y la decisión en
[ADR-0013](../docs/adr/0013-reemplazo-dashboard-streamlit-por-spa.md).

## Diseño · "Mission Control"

Lenguaje visual de **sala de control / instrumento de laboratorio**, no una migración 1:1
del Streamlit:

- **Tema oscuro** (`#0A0E14`) con rejilla técnica sutil y glow por estado.
- **Tipografía:** display **Chakra Petch** (títulos) + **JetBrains Mono** (telemetría). Los
  números usan `tabular-nums` para que no "salten" al actualizarse.
- **Señal como protagonista:** `ScoreCard` instrumental con glow por estado + traza tipo
  osciloscopio; `SystemBanner` con el estado global inequívoco (NOMINAL / CALIBRANDO / ANOMALÍA);
  `DriftGauge` para el C2ST AUC.
- **Color señal:** cian (telemetría), esmeralda (normal), ámbar (warmup), rojo (anomalía).
- Tokens centralizados en `tailwind.config.ts` (reemplazan al morado/blanco genérico anterior).

## Requisitos

- Node 18+ (probado con Node 20).
- La API corriendo (`poetry run uvicorn src.api.main:app`) con **CORS** habilitado
  para el origen del SPA (por defecto `http://localhost:5173`, configurable con
  `CORS_ORIGINS`).

## Puesta en marcha

```bash
cd web
cp .env.example .env        # ajusta VITE_API_BASE_URL si la API no está en :8000
npm install
npm run dev                 # http://localhost:5173
```

## Scripts

| Script | Acción |
|--------|--------|
| `npm run dev` | Servidor de desarrollo (HMR). |
| `npm run build` | Typecheck (`tsc --noEmit`) + build de producción a `dist/`. |
| `npm run preview` | Sirve el build de producción localmente. |
| `npm test` | Suite de tests (Vitest). |
| `npm run typecheck` | Solo verificación de tipos. |

## Arquitectura

- **`src/api/`** — cliente tipado (`types.ts` espejo de `src/api/schemas.py`), `client.ts`,
  `endpoints.ts`, `mediaUrls.ts`, `queryKeys.ts`.
- **`src/hooks/useAnomalyStream.tsx`** — WebSocket con reconexión (backoff + jitter),
  heartbeat, detección de *staleness* y ring buffers; expuesto vía `useSyncExternalStore`.
- **`src/hooks/*`** — wrappers de TanStack Query para el REST.
- **`src/store/fusionDraftStore.ts`** — borrador de fusión (Zustand, persistente).
- **`src/lib/`** — utilidades puras: `fusion.ts` (reimplementación de las 4 estrategias),
  `ringBuffer.ts`, `kde.ts`, `status.ts`, `format.ts`, `constants.ts`.
- **`src/components/`** — `charts/` (uPlot + Recharts), `common/` (incl. `ScoreCard`,
  `SystemBanner`, `DriftGauge`, `StatusChip`, `ConnectionIndicator`, `ConfirmDialog`),
  `fusion/`, `events/`, `similarity/`.
- **`src/pages/`** — `LiveMonitor`, `EventFeed`, `SimilaritySearch`, `OfflineAnalysis`
  (cargadas con `React.lazy`).

## Despliegue en producción

Servir `dist/` como estático. Opción recomendada: FastAPI con `StaticFiles` + *fallback*
a `index.html` (single-origin → CORS no-op). Ver
[`docs/frontend-migration/02-plan-migracion-spa.md`](../docs/frontend-migration/02-plan-migracion-spa.md) §9.
