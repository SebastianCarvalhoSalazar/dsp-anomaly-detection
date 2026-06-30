# 02 — Plan de migración a SPA (React + TypeScript + Vite)

Plan para llevar el dashboard al siguiente nivel reemplazando Streamlit por una SPA desacoplada
que consume la API existente. Ver el análisis del estado actual en
[doc 01](01-analisis-dashboard-actual.md) y el contrato de API en
[doc 03](03-contrato-api-frontend.md). Decisión formal en
[ADR-0013](../adr/0013-reemplazo-dashboard-streamlit-por-spa.md).

## 1. Objetivos y principios

- **Tiempo real de verdad:** la UI reacciona a cada mensaje del WebSocket, sin re-render de página
  completa ni polling.
- **Desacoplamiento total:** el frontend solo depende del contrato HTTP/WS; no importa nada del
  paquete Python.
- **Accesibilidad y responsive** como requisitos, no como adorno (WCAG AA, *mobile-first*).
- **Rendimiento:** *code-splitting*, gráficos ligeros y *ring buffers* para series streaming.
- **Continuidad visual:** reutilizar la paleta actual (`src/dashboard/styles.py::PALETTE`).
- **Migración incremental** con Streamlit funcionando hasta confirmar paridad.

## 2. Stack y justificación

| Preocupación | Elección | Justificación |
|--------------|----------|---------------|
| Build/dev | **Vite 5** + `@vitejs/plugin-react` | HMR rápido, ESM nativo, `import.meta.env.VITE_*`, *code-splitting* por ruta trivial. |
| Lenguaje | **TypeScript (strict)** | El contrato está bien definido; los tipos reflejan los esquemas del backend y detectan *drift*. `strict`, `noUncheckedIndexedAccess`. |
| Estilos | **Tailwind CSS** | Utilitario, *mobile-first* por defecto; la paleta como *design tokens* (CSS vars centralizadas). |
| Estado de servidor | **TanStack Query v5** | Todo el REST es estado de servidor: caché, *dedup*, reintentos/backoff, `staleTime`, invalidación tras *mutations* (delete/reset). |
| Estado realtime | **Hook WS propio + `useSyncExternalStore`** | El WS es un *push stream*, no request/response → TanStack Query no aplica. Un hook dedicado posee el socket; los componentes se suscriben. |
| Estado UI/local | **React local + Zustand mínimo** | Zustand solo para el **borrador de fusión** (debe sobrevivir navegación y alimentar el recompute local instantáneo). El resto es estado local o en la URL. |
| Routing | **React Router v6** | 4 rutas con `lazy()` por ruta; filtros en la URL vía `useSearchParams`. |
| Gráficos | **uPlot** (línea realtime + umbral) / **Recharts** (RMS, KDE, heatmap) | uPlot es canvas, *redraw* sub-ms, ideal para streaming con *ring buffer*. Recharts (SVG) es ergonómico y accesible para series estáticas/baja frecuencia. |
| Media | `<img>` / `<audio>` apuntando a las URLs de la API | Los endpoints de frame/audio son binarios; el navegador los carga directamente (sin pasar por JS, sin requerir CORS en binarios). |
| Testing | **Vitest + Testing Library + MSW** | Vitest comparte config con Vite; MSW *mockea* REST a nivel de red; `mock-socket` para el hook WS. |

## 3. Ubicación y estructura del proyecto

El SPA vive en **`web/`** (hermano de `src/`), aislado del paquete Python/Poetry. Se evita el
nombre `dashboard/` para no confundir con `src/dashboard/` (el Streamlit a retirar). `web/node_modules`
y `web/dist` van a `.gitignore`.

```
web/
  index.html
  package.json            vite.config.ts            tsconfig.json
  tailwind.config.ts      postcss.config.js         vitest.config.ts
  .env.example            # VITE_API_BASE_URL=http://localhost:8000
  src/
    main.tsx  App.tsx  routes.tsx  vite-env.d.ts
    api/        client.ts  types.ts  endpoints.ts  queryKeys.ts  mediaUrls.ts
    hooks/      useAnomalyStream.ts  useEvents.ts  useEvent.ts
                useOfflineAnalysis.ts  useSimilarByEvent.ts  useSimilarUpload.ts
                useFusionConfig.ts  useReducedMotion.ts
    store/      fusionDraftStore.ts
    lib/        fusion.ts  ringBuffer.ts  format.ts  status.ts  constants.ts
    components/
      charts/   RealtimeChart.tsx  RmsBarChart.tsx  KdeChart.tsx  Heatmap.tsx
      common/   ScoreCard  StatusChip  MetricTile  ConfirmDialog  EmptyState
                ErrorState  LoadingSkeleton  ConnectionIndicator  AudioPlayer
                AnnotatedFrame  BBoxChip
      fusion/   FusionControls.tsx
      events/   EventCard.tsx  EventFilters.tsx
      similarity/ SimilarityResultCard.tsx  SearchModeTabs.tsx  FileDropzone.tsx
    pages/      LiveMonitor.tsx  EventFeed.tsx  SimilaritySearch.tsx  OfflineAnalysis.tsx
    layout/     AppShell.tsx  NavBar.tsx
    styles/     theme.css  index.css
    test/       mocks/{handlers.ts,server.ts,fixtures.ts}
```

## 4. Pieza clave: hook `useAnomalyStream`

Es el corazón del tiempo real. Un único hook posee **un** WebSocket a `${WS_BASE}/ws/stream`,
donde `WS_BASE` se deriva de `VITE_API_BASE_URL` (`http→ws`, `https→wss`). Se monta en un
**provider a nivel de app** para que la conexión sobreviva a la navegación.

Responsabilidades:

- **Reconexión con backoff exponencial + jitter:** `min(1000 · 2^intento, 30000)` ± aleatorio;
  reset de intentos tras una apertura estable > N s. **Resuelve B2.**
- **Heartbeat:** enviar un *ping* de texto cada ~10 s (el servidor lo lee y descarta) para mantener
  viva la conexión y cumplir el protocolo documentado.
- **Detección de staleness:** registrar `lastMessageAt` y derivar estado por *recencia* de mensajes
  (`live` si llegó algo en < `staleThresholdMs`, si no `stale`; `connecting`/`reconnecting`/`closed`
  del ciclo de vida del socket). **Resuelve B3** (la salud ya no es "llegó algún mensaje alguna vez").
- **Ring buffers acotados** (p. ej. 300 muestras) para `anomaly_score`, `adaptive_threshold` y `rms`:
  *push* O(1), memoria acotada, sin reasignar arrays. **Resuelve B1/B13.**
- **Suscripción** vía `useSyncExternalStore` → solo re-renderizan los componentes suscritos.

```ts
type StreamStatus = 'connecting' | 'live' | 'stale' | 'reconnecting' | 'closed';

interface AnomalyStream {
  status: StreamStatus;
  lastMessage: AnomalyScoreMessage | null;
  lastMessageAt: number | null;
  scoreBuffer: RingBuffer<{ t: number; v: number }>;
  thresholdBuffer: RingBuffer<{ t: number; v: number }>;
  rmsBuffer: RingBuffer<{ t: number; v: number }>;
  reconnectNow: () => void;
}
```

## 5. Capa de API tipada

- **`api/types.ts`** — espejo 1:1 de `src/api/schemas.py` (ver [doc 03](03-contrato-api-frontend.md)).
- **`api/client.ts`** — *wrapper* sobre `fetch`: une `VITE_API_BASE_URL`, fija `Accept: application/json`,
  lanza `ApiError { status, message, body }` en respuestas no-2xx; maneja `204` (→ `void`), *multipart*
  (`FormData` sin fijar `Content-Type`) y mapea `413` a "archivo > 10 MB".
- **`api/endpoints.ts`** — una función tipada por ruta (`listEvents`, `getEvent`, `deleteEvent`,
  `deleteAllEvents`, `getOfflineAnalysis`, `searchSimilarUpload`, `searchSimilarByEvent`,
  `getFusionConfig`, `setFusionConfig`, `resetDetector`).
- **`api/mediaUrls.ts`** — `audioUrl(id)`, `frameUrl(id, annotated?)`.

## 6. Mapeo de páginas (Streamlit → React)

| Página Streamlit | Componente React | Notas de paridad y mejoras |
|------------------|------------------|----------------------------|
| **Monitor en vivo** (`live_monitor.py`) | `pages/LiveMonitor.tsx` | `ScoreCard` + `StatusChip` (de `is_fitted`/`is_anomaly`), `MetricTile`s (ventana, última detección, detector listo, conexión = estado del stream, motion_energy, refits, drift_auc + caption de `top_drift_features`), tiles audio/video/combined/dominante y doble horizonte rápido/lento, `top_audio/video_features`, `FusionControls`, `RealtimeChart` (score + umbral), `RmsBarChart`, `KdeChart`, `BBoxChip` ("fuente probable"). Reset historial / reset detector tras `ConfirmDialog`. |
| **Eventos** (`event_feed.py`) | `pages/EventFeed.tsx` | `EventFilters` (min_score, límite, orden) sincronizados a la URL; tiles de resumen; `EventCard` (frame anotado + `AudioPlayer` + scores + borrar). Borrar uno/todo tras `ConfirmDialog` con invalidación de la query. Estados empty/loading/error. |
| **Búsqueda** (`similarity_search.py`) | `pages/SimilaritySearch.tsx` | `SearchModeTabs` (subir archivo / evento existente) por valor tipado (no emoji); `FileDropzone` con guard de 10 MB en cliente; `k` slider; grid de `SimilarityResultCard`. UX de carga larga (~60 s) en el primer *upload*. |
| **Análisis offline** (`offline_analysis.py`) | `pages/OfflineAnalysis.tsx` | Selector de evento + botón ejecutar (`useOfflineAnalysis` con `enabled:false` hasta correr); IMFs como sub-gráficos de línea, `Heatmap` para el mel-spectrogram, `AudioPlayer`. Gráficos *lazy*. |

## 7. Cómo se resuelve cada bug

| Bug (doc 01) | Solución en React |
|--------------|-------------------|
| B1 polling 1 s, re-render total | WS *event-driven*: `useSyncExternalStore` re-renderiza solo lo suscrito; sin polling; scroll preservado; CPU casi nula en reposo. REST con `staleTime`, sin intervalos. |
| B2 WS sin reconexión | `WebSocket` nativo con **backoff + jitter** y reconexión automática; mensajes procesados al instante. |
| B3 "Conectado" permanente | **Staleness por `lastMessageAt`** + heartbeat; `ConnectionIndicator` refleja salud real. |
| B4 estado de fusión al navegar | **Zustand `fusionDraftStore`** persiste en memoria entre rutas; valor del servidor cacheado por TanStack Query; opcional espejo a `localStorage`/URL para sobrevivir *refresh*. Sin GC de estado. |
| B5 acoplamiento `src.fusion` | **Reimplementar las 4 estrategias en TS** (`lib/fusion.ts`): `weighted = w·audio+(1-w)·video`, `max`, `and`, `or` según semántica del backend. Recompute local = *preview*; el `combined_score` del backend sigue siendo autoritativo. Test de paridad sobre *fixtures*. |
| B6 acciones destructivas | `ConfirmDialog` accesible (`role="alertdialog"`, *focus trap*, ESC) con texto de consecuencias. |
| B8/B9/B12 magic strings/emoji | `lib/constants.ts` (enums) + `lib/status.ts` (derivación de campos tipados); base URL desde env, sin hardcode. |
| B10/B11 a11y/responsive | Texto+icono+color, ARIA, navegación por teclado; gráficos con `aria-label` y tabla oculta de respaldo; Tailwind *mobile-first* sin ancho fijo. |

## 8. Transversales

- **Accesibilidad (WCAG AA):** estado por texto+icono+color (nunca solo color); `aria-live="polite"`
  en el score; `ConfirmDialog` con *focus trap*; *skip link*; gráficos con resumen `aria-label` y
  tabla visualmente oculta; contraste verificado (cuidado con `#F59E0B` sobre blanco);
  `prefers-reduced-motion` desactiva el pulso y transiciones (`useReducedMotion`).
- **Responsive mobile-first:** *breakpoints* Tailwind; columna única en móvil, grid en `md+`; sin
  `max-width` fijo (usar `max-w-screen-2xl mx-auto`); `ResizeObserver`/`ResponsiveContainer` en gráficos.
- **Rendimiento / Core Web Vitals:** *code-splitting* por ruta (`React.lazy` + `Suspense`), gráficos
  *lazy*, *ring buffers* (sin *GC churn*), *redraw* limitado por `requestAnimationFrame`; imágenes
  `loading="lazy"`, audio `preload="none"`; caché de TanStack Query; *skeletons* (LCP), dimensiones
  reservadas (CLS).
- **Theming:** la paleta de `src/dashboard/styles.py` como CSS vars + *tokens* de Tailwind
  (`primary #7C3AED`, `anomaly #EF4444`, `normal #10B981`, `warning #F59E0B`), habilitando *dark mode*
  futuro por intercambio de *tokens*.

## 9. Cambios de backend requeridos

1. **CORS (obligatorio).** Añadir `CORSMiddleware` en `src/api/main.py`, orígenes vía env
   `CORS_ORIGINS` (default `http://localhost:5173`). Sin esto, el SPA en otro origen no puede llamar
   a la API.
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
   app.add_middleware(CORSMiddleware, allow_origins=origins,
                      allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                      allow_headers=["*"])
   ```
2. **`/health` (recomendado).** `GET /health → {status:"ok"}` para distinguir *backend alcanzable*
   de *salud del stream WS* y para orquestación de contenedores.
3. **Serving estático en prod.** En desarrollo: Vite dev server + CORS. En producción:
   FastAPI `StaticFiles` sirviendo `web/dist` con *fallback* a `index.html` (single-origin → CORS
   no-op). Mantener `CORS_ORIGINS` configurable.
4. **`/internal/*`** ya existe y el SPA lo necesita (`fusion-config` GET/POST, `reset-detector`).
   Hoy no tiene auth: aceptable en LAN/dev; **marcar** para proteger antes de exponer a red no confiable.

## 10. Estrategia de testing

- **Unitario (Vitest + RTL):** `lib/fusion.ts` (paridad de las 4 estrategias + bordes: `audio_weight`
  0/1, gates on/off, scores null); `lib/status.ts` (warmup/anomaly/normal); `lib/ringBuffer.ts`
  (capacidad/sobrescritura); componentes (`ScoreCard`, `StatusChip`, `FusionControls`, `ConfirmDialog`,
  `EventCard`, estados empty/error).
- **MSW:** *handlers* por endpoint; aserción de *query keys*, *optimistic updates* e invalidación tras
  delete/reset; rutas 413 (upload) y 404 (`getEvent`).
- **Hook WS (`mock-socket`):** `connecting→live` al primer mensaje; `live→stale` al avanzar *timers*
  sin mensajes; reconexión con *delays* crecientes; envío de *heartbeat*; acumulación/tope de buffers.
- **Integración:** LiveMonitor (secuencia de mensajes → score/estado/gráfico; reset abre diálogo y
  llama endpoints); EventFeed (filtros en URL → lista → delete invalida → empty); SimilaritySearch
  (ambos modos).
- **CI:** `vitest run --coverage` + ESLint + `tsc --noEmit` como *gates*; opcional *smoke* Playwright
  tras el cutover.

## 11. Roadmap por fases

Cada fase tiene **entregables** y **criterio de aceptación**. Streamlit sigue operativo hasta la F7.

| Fase | Entregables | Aceptación |
|------|-------------|------------|
| **QW** Quick-wins Streamlit | Los 4 *quick-wins* del [doc 01](01-analisis-dashboard-actual.md). | Indicador de conexión honesto; WS reconecta; confirmaciones en acciones destructivas. |
| **F0** Scaffolding | `web/` (Vite+TS+Tailwind+Router+Query+Vitest+MSW), `AppShell` + 4 rutas vacías, *tokens* de tema, CI lint/typecheck/test. **Backend:** CORS + `/health`. | `web` compila, dev server corre, 4 rutas navegables, 1 test pasa, SPA hace `GET /health` *cross-origin*. |
| **F1** API + realtime | `types/endpoints/client`; `useAnomalyStream` (reconnect/heartbeat/staleness + ring buffers); Live Monitor read-only (`ScoreCard`/`StatusChip`/`RealtimeChart`/`ConnectionIndicator`). Tests del hook. | Números y gráfico se actualizan desde el pipeline sin polling; cortar la red muestra `stale`→`reconnecting`→`live`. |
| **F2** Fusión + reset | `FusionControls` + `fusionDraftStore` + `lib/fusion.ts` (tests de paridad) + POST optimista a `/internal/fusion-config`; RMS + KDE + `BBoxChip`; `ConfirmDialog` + reset historial/detector. | Cambiar controles actualiza recompute local al instante y persiste entre rutas; el GET del backend refleja el POST; acciones destructivas piden confirmación. |
| **F3** Event Feed | `EventFilters` (URL), `useEvents`, `EventCard` (frame anotado + audio), delete uno/todo con invalidación, tiles de resumen, estados. | Filtrado/orden por URL; delete actualiza lista; vista filtrada *deep-linkable*. |
| **F4** Similarity Search | Modos upload + by-event, `FileDropzone` con guard 10 MB, `k` slider, grid de resultados, UX de carga del primer upload. Tests (incl. 413). | Ambos modos devuelven y renderizan; archivo *oversize* bloqueado en cliente con mensaje claro. |
| **F5** Offline Analysis | Selector + ejecutar, sub-gráficos IMF, `Heatmap` del spectrogram, audio; gráficos *lazy*. | Ejecutar análisis renderiza `n_imfs` paneles + spectrogram del evento elegido. |
| **F6** A11y/perf/polish | Pase ARIA/teclado, `prefers-reduced-motion`, verificación de *code-splitting*, Lighthouse/CWV, QA responsive. | Auditoría AA pasa; presupuestos Lighthouse a11y/perf cumplidos; funciona en viewport pequeño. |
| **F7** Empaquetado + cutover | Build prod servido por FastAPI `StaticFiles` + *fallback* SPA; CORS no-op en prod; `docs`/README actualizados; *smoke tests*. | Un solo proceso en prod sirve SPA + API; *sign-off* de paridad vs Streamlit. |
| **F8** Decomisión Streamlit | Eliminar `src/dashboard/`, quitar `streamlit`/`plotly` de `pyproject.toml`, romper acoplamiento `src.fusion` del dashboard, actualizar README/ADRs. | El repo compila sin Streamlit; quedan 2 procesos Python (pipeline, API) + SPA estático. |

## 12. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| CORS mal configurado bloquea todo en dev | Orígenes por env; documentar puerto Vite (5173); probar `/health` *cross-origin* en F0 antes de construir features. |
| Tormentas/fugas de reconexión WS | Backoff + jitter, *cap* de delay, reset de intentos tras apertura estable, limpieza estricta de socket+timers, una sola conexión a nivel de app. |
| Primer upload ~60 s parece colgado | El backend ya precarga el encoder en *startup*; el SPA muestra *skeleton* + copy de carga larga; opcional campo en `/health` reportando *encoder-ready*. |
| Recompute de fusión TS diverge del backend | Reimplementar solo las 4 estrategias triviales; `combined_score` del backend autoritativo; tests de paridad; etiquetar el valor local como *preview*. |
| Alta tasa de mensajes WS hace *jank* | Ring buffers + *redraw* por rAF; uPlot canvas; evitar re-render global vía selectores de `useSyncExternalStore`. |
| `/internal/*` sin auth expuesto al navegador | Aceptable en LAN/dev; documentar; añadir auth/gateway antes de despliegue en red no confiable. |
| *Drift* de esquema backend ↔ tipos TS | `types.ts` única fuente; considerar generar tipos desde OpenAPI (`openapi-typescript`) en fase posterior. |
| Quitar Streamlit antes de paridad | Mantenerlo operativo hasta F7; decomisionar (F8) solo tras *sign-off* explícito. |
| Media (`<img>`/`<audio>`) y CORS | Usar `src` directo en el elemento (no `fetch`) para no requerir CORS en los binarios. |

## 13. Verificación del SPA (al implementar)

1. `npm run dev` levanta el SPA; con pipeline+API arriba, el Monitor en vivo refleja datos sin polling.
2. Cortar y restaurar la API: el `ConnectionIndicator` transita `live→stale→reconnecting→live`.
3. `vitest run` verde (incl. paridad de fusión y comportamiento del hook WS); `tsc --noEmit` sin errores.
4. Auditoría Lighthouse: a11y y performance dentro de presupuesto; prueba en viewport móvil.
5. Paridad funcional con las 4 páginas Streamlit antes de la decomisión (F8).
