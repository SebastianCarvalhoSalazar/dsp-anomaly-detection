# 03 — Contrato API ↔ frontend

Contrato que consume el SPA. Es la referencia para `web/src/api/types.ts` y `endpoints.ts`,
verificado contra `src/api/schemas.py`, `src/api/routers/*` y `src/api/main.py` tras
[ADR-0015](../adr/0015-destilacion-del-sistema.md).

## Convención de base URL

- Variable **`API_BASE_URL`** (pipeline) y **`VITE_API_BASE_URL`** (frontend), default
  `http://localhost:8000`.
- **URL del WebSocket**: se deriva de la base cambiando el esquema (`http→ws`, `https→wss`) y
  añadiendo `/ws/stream`. Default: `ws://localhost:8000/ws/stream`.

## Call-outs para el frontend

- **CORS:** `create_app()` registra `CORSMiddleware` con los orígenes de `CORS_ORIGINS`
  (default `http://localhost:5173`).
- Los endpoints **`/internal/*`** tienen `include_in_schema=False` (no salen en Swagger) pero son
  rutas vivas; el SPA usa `POST /internal/reset-detector`.
- El SPA replica el **keep-alive del WS** (envía texto periódico; el servidor lo descarta).
- Los binarios (`/audio`, `/frame`, `/frame/annotated`) se cargan con `src` de `<img>`/`<audio>`.

## Endpoints REST

### Eventos — router `events.py` (prefijo `/events`)

| Método | Ruta | Params | Respuesta |
|--------|------|--------|-----------|
| GET | `/events/` | query: `limit` (1–200, def 50), `offset` (≥0, def 0), `min_score` (0–1, def 0.0) | `200` → `EventResponse[]` |
| GET | `/events/{id}` | path: `id` | `EventResponse` · `404` |
| GET | `/events/{id}/audio` | path: `id` | binario `audio/wav` · `404` |
| GET | `/events/{id}/frame` | path: `id` | binario `image/jpeg` · `404` |
| GET | `/events/{id}/frame/annotated` | path: `id` | `image/jpeg` con la caja de mayor `motion_score` dibujada (rojo + `motion X.XXX`); acepta `source_score` de eventos antiguos · `404` |
| GET | `/events/{id}/spectrogram` | path: `id` | `SpectrogramResponse` (mel en dB, 128 bandas) · `404` |
| DELETE | `/events/{id}` | path: `id` | `204` (borra fila + vector FAISS + archivos) · `404` |
| DELETE | `/events/` | — | `204` (borra todo + resetea índice FAISS) |

### Búsqueda — router `search.py` (prefijo `/search`)

| Método | Ruta | Params | Respuesta |
|--------|------|--------|-----------|
| POST | `/search/similar` | *multipart* `file` (**máx. 10 MB** → `413`); query: `modality` `audio\|image` (def `audio`), `k` (1–20, def 5) | `SimilarEventResponse[]` (vacío si el índice está vacío). La primera llamada puede tardar mientras cargan Wav2Vec2 + DINOv2. |
| GET | `/search/similar/by-event/{id}` | path: `id`; query: `k` (1–20, def 5) | `SimilarEventResponse[]` (excluye el evento fuente; embedding precalculado) · `404` |

### Control + WebSocket — router `websocket.py` (sin prefijo)

| Método | Ruta | Notas |
|--------|------|-------|
| WS | `/ws/stream` | El servidor empuja `AnomalyScoreMessage` (JSON) a todos los clientes. |
| POST | `/internal/reset-detector` | Activa el *flag* de reset; el pipeline lo recoge en su siguiente sondeo. `{ok:true}`. |

> Internos no usados por el frontend: `POST /internal/score` (pipeline → API) y
> `GET /internal/reset-pending` (sondeo del pipeline). `GET /health` → `{"status":"ok"}`.

## Esquemas → tipos TypeScript

Espejo de `src/api/schemas.py` (copia literal de `web/src/api/types.ts`):

```ts
export type Modality = 'audio' | 'image';

export interface EventResponse {
  id: number;
  timestamp: string; // ISO 8601
  anomaly_score: number;
  event_dir: string;
  faiss_index_id: number | null;
  has_audio: boolean;
  has_frame: boolean;
  has_embedding: boolean;
  // Segmentación de eventos (ADR-0014) — null en eventos antiguos
  end_timestamp: string | null;
  duration_s: number | null;
}

export interface SimilarEventResponse {
  event: EventResponse;
  cosine_similarity: number;
}

export interface SpectrogramResponse {
  sample_rate: number;
  spectrogram: number[][]; // mel en dB (bandas × frames)
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  motion_score: number; // área × novedad (sin relación con el audio)
}

export interface AnomalyScoreMessage {
  anomaly_score: number;
  is_anomaly: boolean;
  is_fitted: boolean;
  timestamp: string;
  window_index: number;
  bounding_boxes: BoundingBox[];
  motion_energy: number;
  rms: number;
  adaptive_threshold: number;
  score_mean: number;
  drift_auc: number;
  top_drift_features: string[];
  refit_count: number;
  refit_reason: string;
  top_audio_features: string[];
}
```

## Notas

- `motion_score` vive **dentro de cada caja** de `bounding_boxes[]`: área relativa × novedad
  (1.0 caja nueva, 0.5 persistente). Indica el **movimiento principal** del frame; no localiza la
  fuente del sonido.
- `adaptive_threshold` es el umbral de decisión del detector en la misma escala [0, 1] que
  `anomaly_score`. `drift_auc` vale 0.5 y `top_drift_features` está vacío salvo con
  `ENABLE_DRIFT=true`.
- Retirados en ADR-0015: `/internal/fusion-config`, `/events/{id}/offline_analysis` y los campos
  de fusión, video y modelos lentos (`audio_score`, `video_score`, `combined_score`, `fast_*`,
  `slow_*`, `top_video_features`, `dominant_modality`).

## Mantenimiento del contrato

`types.ts` es la única fuente en el frontend y debe revisarse contra `src/api/schemas.py` ante
cualquier cambio de esquema. A futuro: generar los tipos desde el OpenAPI con
`openapi-typescript`.
