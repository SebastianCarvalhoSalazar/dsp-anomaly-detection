"""Live monitor — real-time anomaly score, status and history charts."""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

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


def _score_chart(
    history: list[float],
    is_anomaly: bool,
    thresh_history: list[float] | None = None,
) -> go.Figure:
    color = PALETTE["anomaly"] if is_anomaly else PALETTE["normal"]
    fill = hex_to_rgba(color, 0.12)
    xs = list(range(len(history)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=history,
        mode="lines", fill="tozeroy",
        line=dict(color=color, width=2),
        fillcolor=fill, name="score",
        hovertemplate="%{y:.3f}<extra></extra>",
    ))

    # Overlay adaptive threshold if available
    has_thresh = (
        thresh_history
        and any(v > 0 for v in thresh_history)
    )
    if has_thresh:
        fig.add_trace(go.Scatter(
            x=xs, y=thresh_history,
            mode="lines",
            line=dict(
                color=PALETTE["warning"],
                width=1.5, dash="dot",
            ),
            name="umbral",
            hovertemplate="%{y:.3f}<extra></extra>",
        ))
    else:
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
        yaxis=dict(
            range=[0, 1], showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(
                size=10, color=PALETTE["muted"],
            ),
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=has_thresh,
        legend=dict(
            orientation="h", x=0, y=1.12,
            font=dict(size=10),
        ),
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


def _score_kde_chart(
    scores: list[float],
    threshold: float = 0.5,
) -> go.Figure:
    """KDE of recent anomaly scores + threshold line."""
    fig = go.Figure()
    xs = np.linspace(0, 1, 200)

    # Need at least 2 non-constant values for KDE
    valid = [s for s in scores if s > 0]
    if len(valid) >= 2 and np.std(valid) > 1e-9:
        kde = gaussian_kde(valid, bw_method=0.15)
        ys = kde(xs)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines", fill="tozeroy",
            line=dict(color="#8B5CF6", width=2),
            fillcolor=hex_to_rgba("#8B5CF6", 0.10),
            name="densidad",
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
        y_max = float(ys.max()) * 1.15
    else:
        y_max = 1.0

    # Threshold vertical line
    if threshold > 0:
        fig.add_trace(go.Scatter(
            x=[threshold, threshold],
            y=[0, y_max],
            mode="lines",
            line=dict(
                color=PALETTE["warning"],
                width=1.5, dash="dot",
            ),
            name="umbral",
            hoverinfo="skip",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=140,
        yaxis=dict(
            showgrid=True, gridcolor="#F1F5F9",
            tickfont=dict(
                size=10, color=PALETTE["muted"],
            ),
            zeroline=False,
            showticklabels=False,
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#F1F5F9",
            tickfont=dict(
                size=10, color=PALETTE["muted"],
            ),
            range=[0, 1],
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=1.15,
            font=dict(size=10),
        ),
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
        st.session_state.thresh_history = deque(
            [0.0] * _HISTORY, maxlen=_HISTORY,
        )
        st.session_state.drift_history = deque(
            [0.0] * _HISTORY, maxlen=_HISTORY,
        )
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
            st.session_state.rms_history.append(msg.get("rms", 0.0))
            # Drift metrics — normalize threshold to [0,1] score
            st.session_state.thresh_history.append(
                msg.get("adaptive_threshold", 0.0)
            )
            st.session_state.drift_history.append(
                msg.get("feature_mean_drift", 0.0)
            )
        except queue.Empty:
            break

    msg        = st.session_state.latest_msg
    score      = msg["anomaly_score"]  if msg else 0.0
    is_anomaly = msg.get("is_anomaly", False) if msg else False
    is_fitted  = msg.get("is_fitted",  False) if msg else False
    win_idx    = msg.get("window_index", 0)   if msg else 0
    motion_e   = msg.get("motion_energy", 0.0) if msg else 0.0
    refit_n    = msg.get("refit_count", 0)     if msg else 0
    drift_val  = msg.get("feature_mean_drift", 0.0) if msg else 0.0
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

        m3, m4, m5 = st.columns(3)
        m3.metric("Detector", "Listo ✓" if is_fitted else "Calentando…")
        m4.metric("Pipeline", "Conectado" if connected else "Sin conexión")
        m5.metric("Motion energy", f"{motion_e:.3f}")

        m6, m7 = st.columns(2)
        m6.metric("Refits", str(refit_n))
        m7.metric(
            "Feature drift",
            f"{drift_val:.4f}",
        )

        st.html("<div style='height:0.5rem'></div>")
        rc1, rc2 = st.columns(2)
        if rc1.button("Reiniciar historial", use_container_width=True):
            st.session_state.score_history = deque([0.0] * _HISTORY, maxlen=_HISTORY)
            st.session_state.rms_history   = deque([0.0] * _HISTORY, maxlen=_HISTORY)
            st.session_state.thresh_history = deque([0.0] * _HISTORY, maxlen=_HISTORY)
            st.session_state.drift_history = deque([0.0] * _HISTORY, maxlen=_HISTORY)
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
        _score_chart(
            list(st.session_state.score_history),
            is_anomaly,
            list(st.session_state.thresh_history),
        ),
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

    # ── Score distribution (KDE) ──────────────────────
    st.html(f"""
    <div style="font-size:0.78rem;font-weight:600;
         letter-spacing:0.06em;
         text-transform:uppercase;
         color:{PALETTE['muted']};
         margin-bottom:0.25rem;">
      Distribuci\u00f3n de scores (KDE)
    </div>
    """)
    _thresh = list(st.session_state.thresh_history)
    _tval = _thresh[-1] if _thresh else 0.5
    st.plotly_chart(
        _score_kde_chart(
            list(st.session_state.score_history),
            threshold=_tval,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Bounding boxes ───────────────────────────────────────────────────────
    if msg and msg.get("bounding_boxes"):
        boxes = msg["bounding_boxes"]
        best = max(
            boxes,
            key=lambda b: b.get("source_score", 0),
        )
        ss = best.get("source_score", 0)
        st.html(f"""
        <div style="background:white;
                    border-radius:12px;
                    padding:1rem 1.5rem;
                    border-left:4px solid
                    {PALETTE['anomaly']};
                    margin-top:0.5rem;
                    box-shadow:0 1px 3px
                    rgba(0,0,0,0.06);">
          <div style="font-size:0.72rem;
               font-weight:600;
               letter-spacing:0.08em;
               text-transform:uppercase;
               color:{PALETTE['muted']};
               margin-bottom:0.5rem;">
            Fuente probable
          </div>
          <span style="background:#FEE2E2;
                color:#B91C1C;
                border-radius:6px;
                padding:4px 12px;
                font-size:0.85rem;
                font-weight:700;">
            ({best['x']},{best['y']})
            {best['w']}&times;{best['h']}
            &mdash; score {ss:.3f}
          </span>
        </div>
        """)

    time.sleep(1)
    st.rerun()
