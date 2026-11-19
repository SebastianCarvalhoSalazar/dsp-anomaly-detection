# DSP Anomaly Dashboard (SPA)

Dashboard web en **React + TypeScript + Vite** del sistema de detección de anomalías. Consume la
API FastAPI (REST + WebSocket); contrato en
[`docs/frontend-migration/03-contrato-api-frontend.md`](../docs/frontend-migration/03-contrato-api-frontend.md).
Decisiones: [ADR-0013](../docs/adr/0013-reemplazo-dashboard-streamlit-por-spa.md) (reemplazo de
Streamlit) y [ADR-0015](../docs/adr/0015-destilacion-del-sistema.md) (fusión, modelos lentos y
EMD retirados).

## Diseño · "Mission Control"

Lenguaje visual de **sala de control / instrumento de laboratorio**:

- **Tema oscuro** (`#0A0E14`) con rejilla técnica sutil y glow por estado.
- **Tipografía:** display **Chakra Petch** (títulos) + **JetBrains Mono** (telemetría). Los
  números usan `tabular-nums` para que no "salten" al actualizarse.
- **Señal como protagonista:** `ScoreCard` instrumental con glow por estado + traza tipo
  osciloscopio; `SystemBanner` con el estado global inequívoco (NOMINAL / CALIBRANDO / ANOMALÍA);
  `DriftGauge` para el C2ST AUC.
- **Color señal:** cian (telemetría), esmeralda (normal), ámbar (warmup), rojo (anomalía).
- **Legibilidad de datos volátiles:** los *top contributors* (explicabilidad) se **congelan en la
  última anomalía** con su timestamp — no parpadean cada ventana.
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
- **`src/hooks/useSpectrogram.ts`** — espectrograma mel de un evento, bajo demanda.
- **`src/hooks/useDetector.ts`** — reinicio del detector (`POST /internal/reset-detector`).
- **`src/lib/`** — utilidades puras: `ringBuffer.ts`, `kde.ts`, `status.ts`, `format.ts`,
  `constants.ts`.
- **`src/components/`** — `charts/` (uPlot + Recharts), `common/` (incl. `ScoreCard`,
  `SystemBanner`, `DriftGauge`, `StatusChip`, `ConnectionIndicator`, `ConfirmDialog`,
  `BBoxChip` "Movimiento principal"), `events/` (`EventCard` con espectrograma), `similarity/`.
- **`src/pages/`** — `LiveMonitor`, `EventFeed`, `SimilaritySearch` (cargadas con `React.lazy`).

Las dependencias se instalan desde el npm público (`.npmrc`). El CI ejecuta `npm ci`,
`npm test` y `npm run build` en cada PR.

## Despliegue en producción

Servir `dist/` como estático. Opción recomendada: FastAPI con `StaticFiles` + *fallback*
a `index.html` (single-origin → CORS no-op). Ver
[`docs/frontend-migration/02-plan-migracion-spa.md`](../docs/frontend-migration/02-plan-migracion-spa.md) §9.
