# ADR-0011: Migración de esquema de eventos compatible hacia atrás

- **Estado:** Aceptado
- **Fecha:** 2026-06-21
- **Decisores:** Equipo del proyecto
- **Fase:** 1

## Contexto y problema

El sistema multimodal añade muchos campos por evento (`audio_score`,
`video_score`, `combined_score`, `fast/slow_*`, `top_audio/video_features`,
`dominant_modality`). El requisito exige **mantener compatibilidad con eventos
antiguos**. Limitaciones:

- SQLAlchemy `create_all` crea tablas nuevas pero **no altera** tablas que ya
  existen, así que una DB v0.2.0 no recibiría las columnas nuevas.
- El proyecto es un MVP sin framework de migraciones (no Alembic).
- La columna `extra_json` del modelo nunca se poblaba (hallazgo M13).

## Opciones consideradas

1. **Adoptar Alembic** — robusto pero pesado para el MVP; añade dependencia y
   flujo de versionado.
2. **Migración ligera idempotente en `Database.init()`** — inspeccionar la
   tabla viva y `ALTER TABLE ADD COLUMN` solo para las columnas faltantes
   (todas nullable).
3. **Recrear la tabla** — destruiría los eventos existentes; inaceptable.

## Decisión

Opción 2. `Database.init()` llama a `_migrate_schema()`, que usa
`sqlalchemy.inspect` para detectar columnas faltantes y las agrega vía
`ALTER TABLE ... ADD COLUMN` (mapa `_ADDED_COLUMNS`). Todas las columnas nuevas
son **nullable**, de modo que las filas viejas quedan con `NULL` y siguen siendo
legibles. Se corrige además M13 poblando `extra_json` en el pipeline.

## Consecuencias

- **Positivas:** DBs existentes se actualizan en el arranque sin pérdida de
  datos; idempotente (re-ejecutar `init()` no falla); sin dependencias nuevas.
- **Negativas / costos:** solución específica de columnas-añadidas; cambios más
  complejos (renombrar/borrar/índices) necesitarían Alembic más adelante.
- **Riesgos y mitigaciones:** SQLite soporta `ADD COLUMN` con default NULL de
  forma barata; la API expone los campos nuevos como `Optional[...] = None`
  para no romper clientes ni eventos viejos.

## Notas de implementación

- `src/storage/models.py`: columnas nuevas nullable en `AnomalyEvent`.
- `src/storage/db.py`: `_ADDED_COLUMNS` + `_migrate_schema()` en `init()`.
- `src/api/schemas.py`: `EventResponse` y `AnomalyScoreMessage` con campos
  multimodales opcionales/con default.
- `src/pipeline.py`: en operación mono-modalidad, `audio_score == combined_score
  == anomaly_score`, `dominant_modality = "audio"`; `extra_json` poblado (M13).
- Tests: `tests/test_schema_migration.py` (DB v0.2.0 → migra → filas viejas
  legibles con campos NULL; idempotencia).
