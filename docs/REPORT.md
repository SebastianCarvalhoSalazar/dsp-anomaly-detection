# Informe técnico — Evolución a detector multimodal (v0.3)

**Rama:** `feature/multimodal-fusion-drift-aware` · **Base:** `dev/v0.2.0-alpha`
**Estado:** 240 tests en verde · 4 fases entregadas (commits Fase 0–3).

Este informe resume la implementación de la evolución arquitectónica del
sistema a un **detector multimodal de fusión tardía** con sincronización
temporal, calibración, fusión configurable, refits sensibles a drift,
snapshots y explicabilidad. Las decisiones están registradas en
[`docs/adr/`](adr/).

---

## 1. Resumen de ejecución por fases

| Fase | Contenido | Commit |
|------|-----------|--------|
| **0** | Refactor `BaseAnomalyDetector` compartido + fix de bugs críticos (C1/C2/H1/H6/M3/M4) | `Fase 0` |
| **1** | Sincronización temporal A/V + esquema de eventos compatible hacia atrás | `Fase 1` |
| **2** | Detector de video + calibración + fusión (núcleo multimodal) | `Fase 2` |
| **3** | Doble horizonte + refits drift-aware + snapshots + explicabilidad | `Fase 3` |

Decisión de secuenciación clave: **arreglar la fundación antes de clonarla**.
El detector de video reusa `BaseAnomalyDetector`, por lo que no heredó los
bugs C1/C2/H1 del detector original (ver [ADR-0002](adr/0002-shared-base-detector-refactor.md)).

---

## 2. Archivos creados

| Archivo | Propósito |
|---------|-----------|
| `src/detection/base.py` | `BaseAnomalyDetector` — núcleo modalidad-agnóstico |
| `src/detection/snapshots.py` | `SnapshotStore` — snapshots de modelos con retención |
| `src/sync/buffer.py` | `AudioWindow`, `CapturedFrame`, `FrameRingBuffer` |
| `src/vision_detection/types.py` | `VideoFeatureVector` + `VIDEO_FEATURE_NAMES` |
| `src/vision_detection/extractor.py` | `VideoFeatureExtractor` (7 features de movimiento) |
| `src/vision_detection/detector.py` | `VideoAnomalyDetector` (sin PCA) |
| `src/fusion/calibration.py` | `PercentileCalibrator` |
| `src/fusion/strategies.py` | `FusionStrategy` + Weighted/Max/AND/OR + `make_strategy` |
| `docs/adr/*` | 12 ADRs (0000 template + 0001–0011) |
| `docs/architecture/multimodal-overview.md` | Diagrama de flujo |
| `tests/test_{base_detector,sync,schema_migration,video_detection,calibration,fusion,drift_refit}.py` | Cobertura nueva |

## 3. Archivos modificados (principales)

| Archivo | Cambio |
|---------|--------|
| `src/detection/detector.py` | `AnomalyDetector` → subclase fina de la base |
| `src/detection/config.py` | flags: freeze normalizer, calibration margin, drift-aware refit |
| `src/dsp/processor.py` | propiedad `feature_names` (layout real, H6) |
| `src/pipeline.py` | scoring de video, calibración, fusión, sync temporal, dual-horizon, snapshots |
| `src/storage/models.py`, `db.py` | columnas multimodales nullable + migración idempotente |
| `src/api/schemas.py`, `routers/events.py` | campos multimodales opcionales |
| `src/vision/motion.py` | M7: `area = w*h` consistente |
| `src/dashboard/pages/live_monitor.py` | scores A/V/combined, slider, selector, fast/slow, top features; fix M14/M15 |

---

## 4. Decisiones de diseño

Ver ADRs para el detalle. Las más relevantes:

- **Fusión tardía** sobre fusión temprana ([ADR-0001](adr/0001-late-fusion-multimodal-architecture.md)).
- **Base compartida** que arregla C1/C2/H1 una sola vez ([ADR-0002](adr/0002-shared-base-detector-refactor.md)).
- **Sincronización por timestamp** con ring buffer ([ADR-0003](adr/0003-timestamp-based-av-synchronization.md)).
- **Calibración por percentiles** para comparar modalidades ([ADR-0004](adr/0004-score-calibration-historical-percentiles.md)).
- **Estrategias de fusión** con patrón Strategy; gating sigue la ruta de audio ([ADR-0005](adr/0005-configurable-fusion-strategies.md)).
- **Sin PCA en video** (7 dims) ([ADR-0006](adr/0006-no-pca-for-low-dim-video-features.md)).
- **Doble horizonte opt-in** ([ADR-0007](adr/0007-dual-horizon-fast-slow-models.md)).
- **Refit drift-aware** con `refit_reason` ([ADR-0008](adr/0008-drift-aware-adaptive-refit-policy.md)).
- **Snapshots con retención** ([ADR-0009](adr/0009-model-snapshots-and-retention.md)).
- **Explicabilidad por z-score** ([ADR-0010](adr/0010-explainability-via-zscore-baseline.md)).
- **Migración compatible** del esquema ([ADR-0011](adr/0011-backward-compatible-event-schema-migration.md)).

---

## 5. Riesgos conocidos

- **La decisión de gating aún no usa la fusión** — por diseño en esta entrega
  (preservar comportamiento). El `combined_score` es observabilidad; promoverlo
  a gating requiere validación de tasas de falsos positivos.
- **Detector de video sobre estadísticas de movimiento gruesas** — `motion_energy`
  y derivados son señales coarse; un objeto entrando al cuadro eleva el score sin
  anomalía acústica. Las estrategias AND/OR dependen de umbrales bien elegidos.
- **C2ST entre buffers consecutivos** solo detecta drift mientras el buffer no se
  renueva del todo; un shift único se observa una vez (ver test de drift).
- **Costo de los modelos lentos** — desactivados por defecto; al activarlos,
  vigilar la latencia del refit (mitigado por fit fuera del lock, H1).
- **Pickle de sklearn en snapshots** acopla la versión de la librería para recargar.
- **Bugs de integridad NO abordados en este alcance** (siguen abiertos del análisis
  v0.2.0): C3/C4 (desync y race cross-proceso de FAISS) y H2/H3 (paths en endpoints
  de streaming). No fueron parte del requisito multimodal.

## 6. Limitaciones restantes

- Sin DINOv2 como detector semántico de anomalías, sin modelos secuenciales
  (LSTM/TCN/Transformer), sin Qdrant/PostgreSQL, sin feedback humano (excluidos
  explícitamente del alcance; la arquitectura los habilita sin rediseño mayor).
- La fusión interactiva del dashboard recomputa client-side; cambiar la estrategia
  *del pipeline* en caliente requeriría un canal de control dashboard→pipeline.
- Calibración relativa a la ventana (no absoluta).

## 7. Recomendaciones para la siguiente iteración

1. **Cerrar los críticos de integridad** C3/C4 (FAISS `IndexIDMap2` + escritura
   atómica/lock cross-proceso) y H2/H3 (validar `event_dir` en endpoints) antes de
   sumar más modalidades.
2. **Promover la fusión a gating** detrás de un flag, midiendo FP/FN contra el
   gating de audio actual.
3. **Detector semántico DINOv2** como tercera fuente de score (encaja como un
   `BaseAnomalyDetector`/encoder adicional en la capa de fusión).
4. **Canal de control dashboard→pipeline** para fijar estrategia/umbral en vivo.
5. **Migrar a Alembic** cuando el esquema requiera cambios más allá de añadir
   columnas.

---

## 8. Verificación

- `poetry run pytest` → **240 passed**.
- `poetry run pylint src/` sin regresiones atribuibles a este trabajo (los avisos
  remanentes son falsos positivos de cv2 y convenciones de naming ML preexistentes).
- Tests por capacidad: base detector, sync, migración, video detector, calibración,
  fusión, drift/snapshots/explicabilidad. Toda la suite usa datos sintéticos
  (sin micrófono/cámara), por convención del repo.
