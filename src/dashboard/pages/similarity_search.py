"""Similarity search — upload audio/image and find semantically similar events."""
from __future__ import annotations

import streamlit as st

from src.dashboard.api_client import APIClient
from src.dashboard.styles import PALETTE, page_header, score_color


def _sim_card(event: dict, sim: float, client: APIClient) -> None:
    eid = event["id"]
    ts  = event["timestamp"][:19].replace("T", " ")
    pct = int(sim * 100)
    bar_color = score_color(1 - sim)   # high similarity → green

    st.html(f"""
    <div class="dsp-card" style="padding:1.25rem;">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:0.75rem;">
        <div>
          <div style="font-weight:700;color:{PALETTE['text']};font-size:0.95rem;">
            Evento #{eid}
          </div>
          <div style="font-size:0.78rem;color:{PALETTE['muted']};margin-top:2px;">
            {ts}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:800;
               color:{PALETTE['normal'] if sim >= 0.7 else PALETTE['warning']};">
            {sim:.3f}
          </div>
          <div style="font-size:0.7rem;color:{PALETTE['muted']};
               text-transform:uppercase;letter-spacing:0.06em;">similitud</div>
        </div>
      </div>
      <div class="sim-bar-wrap">
        <div class="sim-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
      </div>
    </div>
    """)

    if event.get("has_frame"):
        st.image(client.get_frame_url(eid), use_container_width=True)
    if event.get("has_audio"):
        st.audio(client.get_audio_url(eid))


def render(client: APIClient) -> None:
    st.html(page_header(
        "Búsqueda por similitud",
        "Encuentra eventos similares usando embeddings multimodales (Wav2Vec2 + DINOv2)",
    ))

    # Persistir resultados y errores en session_state para sobrevivir reruns
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "sim_error" not in st.session_state:
        st.session_state.sim_error = None

    # ── Controls ─────────────────────────────────────────────────────────────
    with st.container(border=True):
        ctl1, ctl2 = st.columns([2, 1])
        modality = ctl1.radio(
            "Modalidad de búsqueda",
            ["🎤  Audio", "🖼️  Imagen"],
            horizontal=True,
            label_visibility="visible",
            key="sim_modality",
        )
        k = ctl2.slider("Resultados (k)", 1, 20, 5, key="sim_k")

        is_audio = modality.startswith("🎤")
        accept_types = ["wav", "mp3", "ogg"] if is_audio else ["jpg", "jpeg", "png", "webp"]

        uploaded = st.file_uploader(
            f"Sube un archivo de {'audio' if is_audio else 'imagen'}",
            type=accept_types,
            label_visibility="visible",
            key="sim_uploader",
        )

    # Limpiar resultados previos si el usuario quita el archivo
    if uploaded is None:
        st.session_state.sim_results = None
        st.session_state.sim_error = None
        st.html(f"""
        <div class="empty-state">
          <div class="icon">{'🎤' if is_audio else '🖼️'}</div>
          <p>Sube un archivo para buscar eventos similares en el índice FAISS.<br>
             <span style="font-size:0.8rem;">Necesitas al menos un evento guardado con embedding.</span>
          </p>
        </div>
        """)
        return

    # ── Search ───────────────────────────────────────────────────────────────
    if st.button("Buscar eventos similares", type="primary", key="sim_search_btn"):
        st.session_state.sim_results = None
        st.session_state.sim_error = None
        with st.spinner("Codificando y buscando en el índice vectorial… (la primera búsqueda carga los modelos, puede tardar ~60s)"):
            try:
                st.session_state.sim_results = client.search_similar(
                    file_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    modality="audio" if is_audio else "image",
                    k=k,
                )
            except Exception as exc:
                st.session_state.sim_error = str(exc)

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.sim_error:
        st.error(f"Error en la búsqueda: {st.session_state.sim_error}")
        return

    if st.session_state.sim_results is None:
        return

    if not st.session_state.sim_results:
        st.html("""
        <div class="empty-state">
          <div class="icon">🔍</div>
          <p>No se encontraron resultados.<br>El índice puede estar vacío.</p>
        </div>
        """)
        return

    results = st.session_state.sim_results
    st.html(f"""
    <div style="font-size:0.85rem;font-weight:600;color:{PALETTE['muted']};
         margin:1rem 0 0.75rem;">
      {len(results)} evento(s) más similares
    </div>
    """)

    n_cols = min(len(results), 3)
    cols   = st.columns(n_cols, gap="medium")
    for i, res in enumerate(results):
        with cols[i % n_cols]:
            _sim_card(res["event"], res["cosine_similarity"], client)
