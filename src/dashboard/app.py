"""Streamlit dashboard entry point.

Run with:
    poetry run streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit executes scripts in a subprocess where the project root may not be
# on sys.path. Add it explicitly so `from src.*` imports resolve correctly.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from src.dashboard.api_client import APIClient
from src.dashboard.pages import event_feed, live_monitor, offline_analysis, similarity_search
from src.dashboard.styles import inject_global_css

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

PAGES = {
    "Monitor en vivo": ("📡", live_monitor),
    "Eventos":         ("🗂️",  event_feed),
    "Búsqueda":        ("🔍", similarity_search),
    "Análisis offline":("📊", offline_analysis),
}


def main() -> None:
    st.set_page_config(
        page_title="DSP Anomaly Detector",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(inject_global_css(), unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.25rem 0 1rem; text-align:center;">
          <div style="font-size:2rem;">📡</div>
          <div style="color:white;font-weight:700;font-size:1.05rem;margin-top:4px;">
            DSP Anomaly
          </div>
          <div style="color:#64748B;font-size:0.72rem;letter-spacing:0.08em;
                      text-transform:uppercase;margin-top:2px;">
            Sistema de detección
          </div>
        </div>
        <hr style="border-color:#1E293B;margin:0 0 1rem;">
        """, unsafe_allow_html=True)

        labels = list(PAGES.keys())
        icons  = [v[0] for v in PAGES.values()]
        display = [f"{icon}  {label}" for icon, label in zip(icons, labels)]

        choice_display = st.radio("Navegación", display, label_visibility="collapsed")
        choice = labels[display.index(choice_display)]

        st.markdown("""
        <hr style="border-color:#1E293B;margin:1.5rem 0 1rem;">
        <div style="color:#475569;font-size:0.72rem;text-align:center;
                    letter-spacing:0.05em;text-transform:uppercase;">
          API: localhost:8000
        </div>
        """, unsafe_allow_html=True)

    # ── Main area ─────────────────────────────────────────────────────────────
    client = APIClient(base_url=API_BASE_URL)
    _, module = PAGES[choice]
    module.render(client)


if __name__ == "__main__":
    main()
