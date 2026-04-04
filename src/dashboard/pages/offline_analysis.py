"""Offline analysis — EMD decomposition and mel spectrogram for stored events."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.dashboard.api_client import APIClient
from src.dashboard.styles import PALETTE, page_header, score_color


def _imf_figure(imfs: list[list[float]], sr: int) -> go.Figure:
    n = len(imfs)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"IMF {i + 1}" for i in range(n)],
    )
    colors = [PALETTE["primary"], PALETTE["normal"], PALETTE["anomaly"],
              PALETTE["warning"], "#8B5CF6", "#EC4899", "#14B8A6"]

    for i, imf in enumerate(imfs):
        t = [s / sr for s in range(len(imf))]
        fig.add_trace(
            go.Scatter(
                x=t, y=imf,
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
                name=f"IMF {i + 1}",
                showlegend=False,
                hovertemplate="%{y:.4f}<extra>IMF " + str(i + 1) + "</extra>",
            ),
            row=i + 1, col=1,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(180 * n, 300),
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color=PALETTE["muted"], size=11),
    )
    for i in range(1, n + 1):
        fig.update_xaxes(
            showgrid=False, zeroline=False,
            tickfont=dict(size=10), row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="#F1F5F9",
            zeroline=False, tickfont=dict(size=10),
            row=i, col=1,
        )
    fig.update_xaxes(title_text="Tiempo (s)", row=n, col=1)
    return fig


def _spec_figure(spec: list[list[float]], sr: int) -> go.Figure:
    z = np.array(spec)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="dB", font=dict(size=11)), thickness=14),
        hovertemplate="Bin %{y} · Frame %{x}<br>%{z:.1f} dB<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(title="Frame temporal", tickfont=dict(size=10),
                   showgrid=False, color=PALETTE["muted"]),
        yaxis=dict(title="Banda Mel", tickfont=dict(size=10),
                   showgrid=False, color=PALETTE["muted"]),
        font=dict(color=PALETTE["muted"]),
    )
    return fig


def render(client: APIClient) -> None:
    st.html(page_header(
        "Análisis offline",
        "Descomposición EMD e IMFs · Mel-spectrogram por evento",
    ))

    # ── Event selector ───────────────────────────────────────────────────────
    try:
        events = client.list_events(limit=100)
    except Exception as exc:
        st.error(f"No se pudo conectar a la API: {exc}")
        return

    if not events:
        st.html("""
        <div class="empty-state">
          <div class="icon">📊</div>
          <p>No hay eventos disponibles todavía.<br>
             Ejecuta el pipeline para detectar anomalías primero.</p>
        </div>
        """)
        return

    # Build options
    options = {
        f"#{e['id']}  ·  score {e['anomaly_score']:.3f}  ·  {e['timestamp'][:19].replace('T',' ')}": e["id"]
        for e in events
    }

    with st.container(border=True):
        sel_label = st.selectbox("Selecciona un evento", list(options.keys()), label_visibility="visible")
        event_id  = options[sel_label]

        # Preview metadata
        selected  = next(e for e in events if e["id"] == event_id)
        sc        = selected["anomaly_score"]

        mc1, mc2, mc3 = st.columns(3)
        mc1.html(f"""
        <div class="metric-box" style="margin-top:0.5rem;">
          <div class="label">Score</div>
          <div class="value" style="color:{score_color(sc)};">{sc:.3f}</div>
        </div>""")
        mc2.html(f"""
        <div class="metric-box" style="margin-top:0.5rem;">
          <div class="label">Timestamp</div>
          <div style="font-size:0.85rem;font-weight:600;color:{PALETTE['text']};margin-top:6px;">
            {selected['timestamp'][:19].replace('T',' ')}
          </div>
        </div>""")
        mc3.html(f"""
        <div class="metric-box" style="margin-top:0.5rem;">
          <div class="label">Embedding</div>
          <div style="font-size:0.85rem;font-weight:600;color:{PALETTE['text']};margin-top:6px;">
            {"✓ disponible" if selected.get("has_embedding") else "— no disponible"}
          </div>
        </div>""")

        run = st.button("Ejecutar análisis", type="primary")

    if not run:
        return

    # ── Analysis ─────────────────────────────────────────────────────────────
    with st.spinner("Ejecutando EMD y calculando mel-spectrogram…"):
        try:
            data = client.get_offline_analysis(event_id)
        except Exception as exc:
            st.error(f"Error en el análisis: {exc}")
            return

    imfs = data["imfs"]
    sr   = data["sample_rate"]
    spec = data["spectrogram"]

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_emd, tab_spec = st.tabs(["📈  Descomposición EMD", "🎨  Mel-Spectrogram"])

    with tab_emd:
        st.html(f"""
        <div style="font-size:0.82rem;color:{PALETTE['muted']};margin-bottom:1rem;">
          <strong>{data['n_imfs']}</strong> IMFs extraídas ·
          sample rate <strong>{sr} Hz</strong>
        </div>
        """)
        st.plotly_chart(_imf_figure(imfs, sr), use_container_width=True,
                        config={"displayModeBar": False})

    with tab_spec:
        st.html(f"""
        <div style="font-size:0.82rem;color:{PALETTE['muted']};margin-bottom:1rem;">
          Mel-spectrogram en escala logarítmica (dB) ·
          {len(spec)} bandas × {len(spec[0]) if spec else 0} frames
        </div>
        """)
        st.plotly_chart(_spec_figure(spec, sr), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Audio playback ───────────────────────────────────────────────────────
    if selected.get("has_audio"):
        st.html(f"""
        <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;
             text-transform:uppercase;color:{PALETTE['muted']};margin:1rem 0 4px;">
          Audio del evento
        </div>
        """)
        st.audio(client.get_audio_url(event_id))
