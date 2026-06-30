# 03 — Contrato API ↔ frontend

Contrato que el SPA consumirá. Es la referencia para `web/src/api/types.ts` y `endpoints.ts`.
Verificado 1:1 contra `src/api/schemas.py`, `src/api/routers/*` y `src/api/main.py`.

## Convención de base URL

- Variable de entorno **`API_BASE_URL`** (backend/pipeline) y **`VITE_API_BASE_URL`** (frontend),
  default `http://localhost:8000`. **No existe** `API_PORT` (uvicorn usa 8000 por convención).
- **URL del WebSocket**: derivar de la base cambiando el esquema (`http→ws`, `https→wss`) y añadiendo
  `/ws/stream`. Default: `ws://localhost:8000/ws/stream`.

## Call-outs para el frontend

- ⚠️ **No hay CORS configurado** en `src/api/main.py`. Un SPA en otro origen (p. ej. `:5173`) será
  bloqueado hasta añadir `CORSMiddleware` (ver [doc 02 §9](02-plan-migracion-spa.md#9-cambios-de-backend-requeridos)).
- Los endpoints **`/internal/*`** tienen `include_in_schema=False` → no aparecen en OpenAPI/Swagger,
  pero son rutas HTTP vivas que el dashboard ya usa (fusion-config, reset-detector).
- El SPA debe **replicar el keep-alive del WS** (enviar texto periódico; el servidor lo lee y descarta)
  y el **swap de esquema** para construir la URL `ws(s)://`.
- Los binarios (`/audio`, `/frame`, `/frame/annotated`) se cargan vía `src` de `<img>`/`<audio>`
  (no `fetch`), evitando requerir CORS en ellos.

## Endpoints REST

### Eventos — router `events.py` (prefijo `/events`)

| Método | Ruta | Params | Respuesta |
|--------|------|--------|-----------|
| GET | `/events/` | query: `limit` (1–200, def 50), `offset` (≥0, def 0), `min_score` (0–1, def 0.0) | `200` → `EventResponse[]` |
| GET | `/events/{id}` | path: `id` | `EventResponse` · `404` si no existe |
| GET | `/events/{id}/audio` | path: `id` | binario `audio/wav` · `404` |
| GET | `/events/{id}/frame` | path: `id` | binario `image/jpeg` · `404` |
| GET | `/events/{id}/frame/annotated` | path: `id` | binario `image/jpeg` con la caja de mayor `source_score` dibujada (rojo + label) · `404` |
| DELETE | `/events/{id}` | path: `id` | `204` (borra DB + vector FAISS + filesystem) · `404` |
| DELETE | `/events/` | — | `204` (borra todo + resetea índice FAISS) |
| GET | `/events/{id}/offline_analysis` | path: `id` | `OfflineAnalysisResponse` (EMD + mel-spectrogram) · `404` |

### Búsqueda — router `search.py` (prefijo `/search`)

| Método | Ruta | Params | Respuesta |
|--------|------|--------|-----------|
| POST | `/search/similar` | *multipart* `file` (**máx. 10 MB** → `413`); query: `modality` `audio\|image` (def `audio`), `k` (1–20, def 5) | `SimilarEventResponse[]` (vacío si índice vacío). Primer llamado ~60 s (carga Wav2Vec2+DINOv2). |
| GET | `/search/similar/by-event/{id}` | path: `id`; query: `k` (1–20, def 5) | `SimilarEventResponse[]` (excluye el evento fuente; usa embedding precomputado, casi instantáneo) · `404` |

### Canal de control + WebSocket — router `websocket.py` (sin prefijo)

| Método | Ruta | Notas |
|--------|------|-------|
| WS | `/ws/stream` | El servidor empuja `AnomalyScoreMessage` (frames de texto JSON) a todos los clientes. El cliente debe enviar texto periódico (ping); cualquier mensaje recibido se descarta. |
| POST | `/internal/fusion-config` | body `FusionConfigMessage`; valida `strategy` (→ `422` si inválida), *clampa* `audio_weight` a [0,1]. Devuelve `{ok, strategy, audio_weight, gates}`. |
| GET | `/internal/fusion-config` | Devuelve `{strategy, audio_weight, gates}` (default `{"weighted", 0.5, false}`). |
| POST | `/internal/reset-detector` | Activa el *flag* de reset del detector. Devuelve `{ok:true}`. |

> Internos no usados por el frontend: `POST /internal/score` (pipeline→API) y
> `GET /internal/reset-pending` (poll del pipeline).

## Esquemas → tipos TypeScript

Espejo de `src/api/schemas.py`. Estos tipos van en `web/src/api/types.ts`.

```ts
export type FusionStrategy = 'weighted' | 'max' | 'and' | 'or';
export type Modality = 'audio' | 'image';

export interface EventResponse {
  id: number;
  timestamp: string;            // ISO 8601
  anomaly_score: number;
  event_dir: string;
  faiss_index_id: number | null;
  has_audio: boolean;
  has_frame: boolean;
  has_embedding: boolean;
  // Multimodal (v0.3) — null en eventos de versiones antiguas
  audio_score: number | null;
  video_score: number | null;
  combined_score: number | null;
  dominant_modality: string | null;
}

export interface SimilarEventResponse {
  event: EventResponse;
  cosine_similarity: number;
}

export interface OfflineAnalysisResponse {
  imfs: number[][];             // arrays IMF (series temporales)
  n_imfs: number;
  sample_rate: number;
  spectrogram: number[][];      // mel-spectrogram (freq × time)
}

export interface FusionConfig {
  strategy: FusionStrategy;
  audio_weight: number;         // 0..1
  gates: boolean;
}

export interface BoundingBox {
  x: number; y: number; w: number; h: number;
  source_score: number;         // score de localización por caja
}

// Mensaje empujado repetidamente por WS /ws/stream
export interface AnomalyScoreMessage {
  anomaly_score: number;
  is_anomaly: boolean;
  is_fitted: boolean;
  timestamp: string;            // ISO
  window_index: number;
  bounding_boxes: BoundingBox[];
  motion_energy: number;        // ratio de área en movimiento [0,1]
  rms: number;                  // amplitud RMS de la ventana de audio
  // Métricas de drift
  adaptive_threshold: number;
  score_mean: number;
  drift_auc: number;            // C2ST: 0.5 = sin drift, 1.0 = drift total
  top_drift_features: string[];
  refit_count: number;
  refit_reason: string;
  // Scores multimodales (v0.3)
  audio_score: number;
  video_score: number;
  combined_score: number;       // fuente autoritativa de la fusión
  fast_audio_score: number;
  slow_audio_score: number;
  fast_video_score: number;
  slow_video_score: number;
  top_audio_features: string[];
  top_video_features: string[];
  dominant_modality: string;    // "audio" por defecto
}
```

## Nota sobre `source_score`

No es un campo de nivel superior del mensaje WS: vive **dentro de cada elemento de
`bounding_boxes[]`**. El consumidor del frame anotado (`events.py`) usa la caja con mayor
`source_score` para dibujar la "fuente probable".

## Mantenimiento del contrato

Para evitar *drift* entre backend y frontend, considerar en una fase posterior generar los tipos
desde el OpenAPI del backend con `openapi-typescript`. Mientras tanto, `types.ts` es la única fuente
y debe revisarse contra `src/api/schemas.py` ante cualquier cambio de esquema.
