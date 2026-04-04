"""Live monitor — real-time anomaly score, status and history charts."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.api_client import APIClient
from src.dashboard.styles import (
    PALETTE,
    big_score_class,
    hex_to_rgba,
    page_header,
    status_chip,
)

_WS_URL  = "ws://localhost:8000/ws/stream"
_HISTORY = 256


def _ws_listener(msg_queue: queue.Queue, stop_event: threading.Event) -> None:
    try:
        import websockets, asyncio

        async def _connect() -> None:
            async with websockets.connect(_WS_URL) as ws:
                while not stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        msg_queue.put(json.loads(raw))
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

        asyncio.run(_connect())
    except Exception:
        pass


def _score_chart(history: list[float], is_anomaly: bool) -> go.Figure:
    color = PALETTE["anomaly"] if is_anomaly else PALETTE["normal"]
    fill  = hex_to_rgba(color, 0.12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=history,
        mode="lines",
        fill="tozeroy",
        line=dict(color=color, width=2),
        fillcolor=fill,
        hovertemplate="%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=0.5, line_dash="dot",
        line_color=PALETTE["warning"],
        annotation_text="umbral",
        annotation_position="top right",
        annotation_font_color=PALETTE["warning"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=160,
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#F1F5F9",
                   tickfont=dict(size=10, color=PALETTE["muted"]), zeroline=False),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False,
    )
    return fig


def _rms_chart(history: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(history))),
        y=history,
        marker_color=hex_to_rgba(PALETTE["primary"], 0.65),
        hovertemplate="%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=120,
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9",
                   tickfont=dict(size=10, color=PALETTE["muted"]), zeroline=False),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False,
        bargap=0.1,
    )
    return fig


def render(client: APIClient) -> None:
    st.html(page_header(
        "Monitor en vivo",
        "Scores de anomalía en tiempo real via WebSocket",
    ))

    # ── Session state ────────────────────────────────────────────────────────
    if "rms_history" not in st.session_state:
        st.session_state.rms_history   = deque([0.0] * _HISTORY, maxlen=_HISTORY)
        st.session_state.score_history = deque([0.0] * _HISTORY, maxlen=_HISTORY)
        st.session_state.ws_queue      = queue.Queue()
        st.session_state.ws_stop       = threading.Event()
        st.session_state.latest_msg    = None
        t = threading.Thread(
            target=_ws_listener,
            args=(st.session_state.ws_queue, st.session_state.ws_stop),
            daemon=True,
        )
        t.start()

    while not st.session_state.ws_queue.empty():
        try:
            msg = st.session_state.ws_queue.get_nowait()
            st.session_state.latest_msg = msg
            score = msg.get("anomaly_score", 0.0)
            st.session_state.score_history.append(score)
            st.session_state.rms_history.append(score * 0.3)
        except queue.Empty:
            break

    msg        = st.session_state.latest_msg
    score      = msg["anomaly_score"]  if msg else 0.0
    is_anomaly = msg.get("is_anomaly", False) if msg else False
    is_fitted  = msg.get("is_fitted",  False) if msg else False
    win_idx    = msg.get("window_index", 0)   if msg else 0
    ts_str     = (msg.get("timestamp", "")[:19].replace("T", " ")) if msg else "—"
    connected  = msg is not None

    score_cls  = big_score_class(score, is_fitted)
    chip_html  = status_chip(is_anomaly, is_fitted)

    # ── Top row ──────────────────────────────────────────────────────────────
    col_score, col_info = st.columns([1, 2], gap="large")

    with col_score:
        # Score card uses st.html() for the big number
        score_color_val = (
            PALETTE["warning"] if not is_fitted
            else (PALETTE["anomaly"] if is_anomaly else PALETTE["normal"])
        )
        st.html(f"""
        <div style="background:white;border-radius:14px;padding:2rem 1rem;
                    text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
          <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;
               text-transform:uppercase;color:{PALETTE['muted']};margin-bottom:0.75rem;">
            Anomaly Score
          </div>
          <div style="font-size:4.5rem;font-weight:800;line-height:1;
               letter-spacing:-0.04em;color:{score_color_val};">{score:.3f}</div>
          <div style="margin-top:1rem;">{chip_html}</div>
        </div>
        """)

    with col_info:
        m1, m2 = st.columns(2)
        m1.metric("Ventana", f"#{win_idx}")
        m2.metric("Última detección", ts_str)

        m3, m4 = st.columns(2)
        m3.metric("Detector", "Listo ✓" if is_fitted else "Calentando…")
        m4.metric("Pipeline", "Conectado" if connected else "Sin conexión")

        st.html("<div style='height:0.5rem'></div>")
        rc1, rc2 = st.columns(2)
        if rc1.button("Reiniciar historial", use_container_width=True):
            st.session_state.score_history = deque([0.0] * _HISTORY, maxlen=_HISTORY)
            st.session_state.rms_history   = deque([0.0] * _HISTORY, maxlen=_HISTORY)
            st.session_state.latest_msg    = None
        if rc2.button("Reiniciar detector", use_container_width=True):
            try:
                client.reset_detector()
                st.toast("Señal de reset enviada al pipeline.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.divider()

    # ── Score history ────────────────────────────────────────────────────────
    st.html(f"""
    <div style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
         text-transform:uppercase;color:{PALETTE['muted']};margin-bottom:0.25rem;">
      Historial de anomaly score
    </div>
    """)
    st.plotly_chart(
        _score_chart(list(st.session_state.score_history), is_anomaly),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── RMS amplitude ────────────────────────────────────────────────────────
    st.html(f"""
    <div style="font-size:0.78rem;font-weight:600;letter-spacing:0.06em;
         text-transform:uppercase;color:{PALETTE['muted']};margin-bottom:0.25rem;">
      Amplitud RMS
    </div>
    """)
    st.plotly_chart(
        _rms_chart(list(st.session_state.rms_history)),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Bounding boxes ───────────────────────────────────────────────────────
    if msg and msg.get("bounding_boxes"):
        boxes = msg["bounding_boxes"]
        chips = "".join(
            f'<span style="background:#FEE2E2;color:#B91C1C;border-radius:6px;'
            f'padding:3px 10px;font-size:0.8rem;font-weight:600;margin:2px;">'
            f'({b["x"]},{b["y"]}) {b["w"]}×{b["h"]}</span>'
            for b in boxes
        )
        st.html(f"""
        <div style="background:white;border-radius:12px;padding:1rem 1.5rem;
                    border-left:4px solid {PALETTE['anomaly']};margin-top:0.5rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);">
          <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;
               text-transform:uppercase;color:{PALETTE['muted']};margin-bottom:0.5rem;">
            Regiones de movimiento — {len(boxes)} bbox(s)
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:0.25rem;">{chips}</div>
        </div>
        """)

    time.sleep(1)
    st.rerun()
