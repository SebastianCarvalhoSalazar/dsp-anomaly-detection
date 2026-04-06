"""Event feed — browse, filter and inspect stored anomaly events."""
from __future__ import annotations

import streamlit as st

from src.dashboard.api_client import APIClient
from src.dashboard.styles import PALETTE, page_header, score_color, score_pill


def _score_bar(score: float) -> str:
    pct = int(score * 100)
    color = score_color(score)
    return f"""
    <div class="sim-bar-wrap" style="margin-top:6px;">
      <div class="sim-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    """


def render(client: APIClient) -> None:
    st.html(page_header(
        "Eventos detectados",
        "Historial de anomalías capturadas con audio y evidencia visual",
    ))

    # ── Filters + acciones globales ──────────────────────────────────────────
    with st.container(border=True):
        fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
        min_score = fc1.slider("Score mínimo", 0.0, 1.0, 0.0, step=0.05, label_visibility="visible")
        limit     = fc2.selectbox("Mostrar", [10, 25, 50, 100], index=1, label_visibility="visible")
        sort_desc = fc3.selectbox("Orden", ["Más recientes", "Mayor score"], label_visibility="visible")
        fc4.html("<div style='height:1.6rem'></div>")
        if fc4.button("Borrar todo", type="secondary", use_container_width=True):
            try:
                client.clear_events()
                st.success("Todos los eventos han sido eliminados.")
                st.rerun()
            except Exception as exc:
                st.error(f"Error al borrar: {exc}")

    # ── Fetch ────────────────────────────────────────────────────────────────
    try:
        events = client.list_events(limit=int(limit), min_score=min_score)
    except Exception as exc:
        st.error(f"No se pudo conectar a la API: {exc}")
        return

    if sort_desc == "Mayor score":
        events = sorted(events, key=lambda e: e["anomaly_score"], reverse=True)

    # ── Empty state ──────────────────────────────────────────────────────────
    if not events:
        st.html("""
        <div class="empty-state">
          <div class="icon">🎙️</div>
          <p>No hay eventos todavía.<br>
             Ejecuta el pipeline y genera algún sonido cerca del micrófono.</p>
        </div>
        """)
        return

    # ── Summary row ──────────────────────────────────────────────────────────
    n_anomaly = sum(1 for e in events if e["anomaly_score"] >= 0.5)
    avg_score = sum(e["anomaly_score"] for e in events) / len(events)
    max_score = max(e["anomaly_score"] for e in events)

    s1, s2, s3, s4 = st.columns(4)
    for col, label, val in [
        (s1, "Total eventos",  str(len(events))),
        (s2, "Anomalías",      str(n_anomaly)),
        (s3, "Score promedio", f"{avg_score:.3f}"),
        (s4, "Score máximo",   f"{max_score:.3f}"),
    ]:
        col.html(f"""
        <div class="metric-box">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
        </div>
        """)

    st.html("<br>")

    # ── Event list ───────────────────────────────────────────────────────────
    for event in events:
        eid   = event["id"]
        score = event["anomaly_score"]
        ts    = event["timestamp"][:19].replace("T", " ")
        pill  = score_pill(score)
        bar   = _score_bar(score)
        emb   = "✓" if event.get("has_embedding") else "—"

        with st.expander(f"Evento #{eid}   ·   {ts}", expanded=False):
            ec1, ec2 = st.columns([1, 1], gap="large")

            with ec1:
                st.html(f"""
                <div style="margin-bottom:1rem;">
                  <div style="font-size:0.72rem;font-weight:600;
                       letter-spacing:0.08em;text-transform:uppercase;
                       color:{PALETTE['muted']};margin-bottom:4px;">Anomaly score</div>
                  <div style="font-size:2.2rem;font-weight:800;
                       color:{score_color(score)};line-height:1;">{score:.3f}</div>
                  {bar}
                </div>
                <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem;">
                  {pill}
                  <span style="background:{PALETTE['bg']};color:{PALETTE['muted']};
                        border-radius:999px;padding:3px 10px;font-size:0.78rem;">
                    #{eid}
                  </span>
                  <span style="background:{PALETTE['bg']};color:{PALETTE['muted']};
                        border-radius:999px;padding:3px 10px;font-size:0.78rem;">
                    embedding {emb}
                  </span>
                </div>
                """)

                if event.get("has_audio"):
                    st.html(f"""
                    <div style="font-size:0.72rem;font-weight:600;
                         letter-spacing:0.08em;text-transform:uppercase;
                         color:{PALETTE['muted']};margin-bottom:4px;">Audio</div>
                    """)
                    st.audio(client.get_audio_url(eid))

                if st.button("Eliminar evento", key=f"del_{eid}", type="secondary"):
                    try:
                        client.delete_event(eid)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

            with ec2:
                if event.get("has_frame"):
                    st.html(f"""
                    <div style="font-size:0.72rem;font-weight:600;
                         letter-spacing:0.08em;text-transform:uppercase;
                         color:{PALETTE['muted']};margin-bottom:4px;">Frame capturado</div>
                    """)
                    st.image(
                        client.get_annotated_frame_url(eid),
                        use_container_width=True,
                    )
                else:
                    st.html("""
                    <div class="empty-state" style="padding:1.5rem 0;">
                      <div class="icon" style="font-size:2rem;">📷</div>
                      <p style="font-size:0.8rem;">Sin frame<br>(cámara no disponible)</p>
                    </div>
                    """)
