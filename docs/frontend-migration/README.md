# Migración del dashboard: Streamlit → SPA (React + TypeScript)

Este paquete de documentos recoge el **plan para reemplazar el dashboard Streamlit**
(`src/dashboard/`) por una **Single Page Application** desacoplada en **React + TypeScript +
Vite**, que consume la API FastAPI existente (REST + WebSocket).

El material vive aislado en este sub-folder para no mezclarse con el resto de `docs/`. La
**decisión de arquitectura** formal está registrada como
[ADR-0013](../adr/0013-reemplazo-dashboard-streamlit-por-spa.md).

## Índice

| Doc | Contenido |
|-----|-----------|
| [01 — Análisis del dashboard actual](01-analisis-dashboard-actual.md) | Bugs, deuda técnica y *quick-wins* sobre el Streamlit actual (con `archivo:línea`). |
| [02 — Plan de migración a SPA](02-plan-migracion-spa.md) | Arquitectura del SPA, mapeo de páginas, solución de cada bug, roadmap por fases y verificación. |
| [03 — Contrato API ↔ frontend](03-contrato-api-frontend.md) | Endpoints REST, esquema del WebSocket y tipos TypeScript de referencia. |

## Contexto en una frase

La API ya es un backend limpio y desacoplado; el dashboard es un mero *consumidor*. Streamlit
impone límites estructurales (polling de página completa, WebSocket sin reconexión, estado que se
pierde al navegar, acoplamiento al paquete Python) que **no se arreglan de raíz dentro de
Streamlit**. Un SPA resuelve todo eso y habilita tiempo real real, accesibilidad y despliegue
independiente.

## Decisiones acordadas

- **Stack:** React 18 + TypeScript (strict) + Vite 5.
- **Gráficos:** ligeros para tiempo real — **uPlot** (series streaming) + **Recharts** (KDE/barras/heatmap).
- **Alcance:** migración completa **+** *quick-wins* inmediatos al Streamlit actual mientras se migra.

## Estado

**Implementado** en [`web/`](../../web/) (fases F0–F5 del roadmap): SPA React+TS+Vite funcional
con las 4 páginas, capa API tipada, `useAnomalyStream`, controles de fusión y tests (Vitest).
Backend con `CORSMiddleware` + `/health`. Pendientes: F6 (a11y/Lighthouse), F7 (serving estático
desde FastAPI) y F8 (decomisión de Streamlit tras confirmar paridad).

> **Nota de diseño:** la implementación **no** fue una migración 1:1 del Streamlit. Se adoptó un
> rediseño con lenguaje visual **"Mission Control"** (tema oscuro de sala de control, tipografía
> técnica, telemetría en mono tabular, la señal viva como protagonista), corrigiendo la jerarquía
> plana y la estética genérica del dashboard anterior. Ver [`web/README.md`](../../web/README.md).
