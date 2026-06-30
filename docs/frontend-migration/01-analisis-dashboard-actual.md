# 01 — Análisis del dashboard actual (Streamlit)

Análisis estático del dashboard Streamlit (`src/dashboard/`) realizado leyendo el código
completo de las cuatro páginas, `app.py`, `api_client.py` y `styles.py`, junto con el mapa de
la API y los ADRs. El objetivo es documentar **bugs**, **deuda técnica** y un conjunto de
**quick-wins** de bajo costo aplicables al Streamlit actual mientras se desarrolla el SPA.

> La inspección visual en vivo con *Claude in Chrome* no pudo completarse (la extensión no quedó
> emparejada con la sesión). Queda como paso de verificación opcional; los hallazgos visuales
> afinan UX pero no cambian la arquitectura.

## Resumen ejecutivo

El dashboard es funcional y visualmente cuidado, pero arrastra **limitaciones estructurales del
modelo de ejecución de Streamlit** (re-run de script completo, estado de widgets efímero) que no
se pueden resolver de raíz sin salir de Streamlit. Los problemas más serios están en el **Monitor
en vivo**: el "tiempo real" tiene techo de 1 s, el WebSocket no se reconecta y el indicador de
conexión es engañoso.

## Hallazgos

| # | Severidad | Área | Archivo:línea | Problema | Impacto |
|---|-----------|------|---------------|----------|---------|
| B1 | Alta | Realtime | `pages/live_monitor.py:566-567` | Bucle `time.sleep(1)` + `st.rerun()`: re-ejecuta TODO el script cada segundo. | Techo de latencia de 1 s en un panel "en vivo"; re-render completo (4 gráficos Plotly), parpadeo, se pierde el scroll, alto consumo de CPU. |
| B2 | Alta | WebSocket | `pages/live_monitor.py:79-96` | El listener WS corre en un thread→cola; ante error hace `break` silencioso. **Sin reconexión ni backoff.** | Si el WS cae (reinicio de API/pipeline), el panel deja de actualizar sin avisar y no se recupera solo. |
| B3 | Alta | UX/estado | `pages/live_monitor.py:314` | `connected = msg is not None`: una vez recibido el primer mensaje, `latest_msg` queda fijo. | "Pipeline: Conectado" permanece **siempre verdadero** tras el primer mensaje, aunque el pipeline esté caído. No hay detección de *staleness*/heartbeat. |
| B4 | Media | UX/estado | `pages/live_monitor.py:50-76,389-418` | Workaround `restore_fusion_state`/`persist_fusion_state` con clave-espejo no-widget para sobrevivir el GC de estado de widgets al navegar. | Síntoma del modelo de estado de Streamlit, no una solución real (es lo que parchea la rama `fix/dashboard-fusion-state-persist`). Frágil y difícil de extender. |
| B5 | Media | Acoplamiento | `pages/live_monitor.py:24,436-439` | El dashboard importa `src.fusion.make_strategy` y recomputa la fusión localmente para feedback instantáneo. | Acopla el dashboard al paquete Python del backend (no se puede desplegar de forma independiente) y el `combined_score` mostrado puede divergir del que calcula el pipeline. |
| B6 | Media | Seguridad/UX | `pages/event_feed.py:33-39,126-131`; `pages/live_monitor.py:366-377` | Acciones destructivas sin confirmación: "Borrar todo" (`DELETE /events/`), "Eliminar evento", "Reiniciar detector". | Pérdida de datos por un clic accidental; no hay paso de confirmación ni *undo*. |
| B7 | Media | Robustez | `pages/live_monitor.py:286-314` | Si el WS nunca entrega un mensaje (API caída al abrir), la página muestra ceros y solo el heurístico de B3 indica algo. | El primer arranque sin pipeline no comunica claramente "sin conexión". |
| B8 | Baja | Mantenibilidad | `pages/similarity_search.py:158,190` | Detección de modo/modalidad por prefijo de emoji: `mode.startswith("📁")`, `modality.startswith("🎤")`. | Frágil: cambiar el texto/emoji rompe la lógica. Debe derivarse de un valor tipado. |
| B9 | Baja | Mantenibilidad | `pages/live_monitor.py:393-403` | *Magic strings* de estrategias de fusión y sus etiquetas duplicadas en línea. | Duplicación; fácil de desincronizar con el backend. |
| B10 | Media | Accesibilidad | `styles.py` (global); `pages/*` (st.html) | Codificación de estado por color, sin ARIA, mucho HTML inline; Plotly no es accesible. | No cumple WCAG para lectores de pantalla; los chips sí incluyen texto (bien), pero los gráficos no tienen alternativa textual. |
| B11 | Baja | Responsive | `styles.py:33-37` | `max-width:1200px` fijo; las columnas de Streamlit no refluyen bien en móvil. | No es *mobile-first*; experiencia pobre en pantallas pequeñas. |
| B12 | Baja | Coherencia | `app.py:71` | Caption "API: localhost:8000" hardcodeado aunque `API_BASE_URL` es configurable. | Engañoso al desplegar contra otra URL. |
| B13 | Baja | Rendimiento | `pages/live_monitor.py:200-201,437` | `gaussian_kde` y `make_strategy(...).combine(...)` se recalculan en cada re-run (cada segundo). | Cómputo redundante en el hilo de UI cada segundo. |

## Quick-wins (sobre el Streamlit actual)

Cambios de **bajo costo** que mitigan los peores síntomas y de-riesgan la migración. Son
**puentes**, no refactors profundos (Streamlit se decomisiona al final del roadmap):

1. **Indicador de conexión honesto (B3).** Sustituir `connected = msg is not None` por una
   detección basada en *timestamp*: guardar la marca temporal del último mensaje y comparar contra
   *ahora* (p. ej. `stale` si pasaron > 5 s sin mensajes).
2. **Reconexión del WebSocket (B2).** Añadir reintento con *backoff* en `_ws_listener` y drenar la
   cola en cada iteración del bucle (no una vez por segundo).
3. **Confirmación en acciones destructivas (B6).** Doble paso (checkbox/expander de confirmación)
   antes de "Borrar todo" y "Reiniciar detector".
4. **Centralizar tokens y eliminar detección por emoji (B8, B9, B12).** Consolidar `PALETTE` como
   única fuente de color (será el origen de los *tokens* de Tailwind en el SPA) y derivar modo/
   modalidad/estrategia de valores tipados en vez de prefijos de emoji.

> Los problemas B1, B4, B5, B10 y B11 son **estructurales de Streamlit** y se resuelven
> propiamente solo en el SPA (ver [doc 02](02-plan-migracion-spa.md)).

## Verificación visual pendiente (opcional)

Cuando la extensión de *Claude in Chrome* esté emparejada: levantar pipeline + API + Streamlit,
navegar `http://localhost:8501`, capturar las cuatro páginas y la consola, y anexar aquí los
hallazgos visuales (parpadeo en re-run, pérdida de scroll, estados vacíos, contraste de color).
