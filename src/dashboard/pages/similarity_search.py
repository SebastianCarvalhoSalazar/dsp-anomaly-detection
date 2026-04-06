"""Similarity search — find semantically similar events.

Supports two search modes:

* **Upload file** — upload an audio clip or image and the server encodes it
  on-the-fly (requires model loading on the first call).
* **Select event** — pick an already-stored event from a dropdown; the
  server reuses its pre-computed embedding so the response is near-instant
  and never needs to load the encoder models.
"""
from __future__ import annotations

import streamlit as st

from src.dashboard.api_client import APIClient
from src.dashboard.styles import PALETTE, page_header, score_color


# ── Result card ──────────────────────────────────────────────────────────────

def _sim_card(event: dict, sim: float, client: APIClient) -> None:
    """Render a single similarity-result card with score bar."""
    eid = event["id"]
    ts  = event["timestamp"][:19].replace("T", " ")
    pct = int(sim * 100)
    bar_color = score_color(1 - sim)   # high similarity → green

    st.html(f"""
    <div class="dsp-card" style="padding:1.25rem;">
      <div style="display:flex;justify-content:space-between;
                  align-items:flex-start;margin-bottom:0.75rem;">
        <div>
          <div style="font-weight:700;color:{PALETTE['text']};
               font-size:0.95rem;">
            Evento #{eid}
          </div>
          <div style="font-size:0.78rem;color:{PALETTE['muted']};
               margin-top:2px;">{ts}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.5rem;font-weight:800;
               color:{PALETTE['normal'] if sim >= 0.7
                      else PALETTE['warning']};">
            {sim:.3f}
          </div>
          <div style="font-size:0.7rem;color:{PALETTE['muted']};
               text-transform:uppercase;letter-spacing:0.06em;">
            similitud</div>
        </div>
      </div>
      <div class="sim-bar-wrap">
        <div class="sim-bar-fill"
             style="width:{pct}%;background:{bar_color};"></div>
      </div>
    </div>
    """)

    if event.get("has_frame"):
        st.image(
            client.get_annotated_frame_url(eid),
            use_container_width=True,
        )
    if event.get("has_audio"):
        st.audio(client.get_audio_url(eid))


# ── Query-event preview card ────────────────────────────────────────────────

def _query_card(event: dict, client: APIClient) -> None:
    """Show a compact preview of the event used as search query."""
    eid = event["id"]
    ts  = event["timestamp"][:19].replace("T", " ")
    sc  = event.get("anomaly_score", 0)
    st.html(f"""
    <div style="background:{PALETTE['bg']};border-radius:10px;
                padding:0.75rem 1rem;
                border-left:4px solid {PALETTE['primary']};
                margin-bottom:0.75rem;">
      <div style="font-size:0.72rem;font-weight:600;
           letter-spacing:0.06em;text-transform:uppercase;
           color:{PALETTE['muted']};margin-bottom:0.25rem;">
        Evento de consulta
      </div>
      <span style="font-weight:700;color:{PALETTE['text']};">
        #{eid}</span>
      <span style="color:{PALETTE['muted']};font-size:0.82rem;
            margin-left:0.5rem;">
        {ts} · score {sc:.3f}</span>
    </div>
    """)
    col_f, col_a = st.columns(2)
    with col_f:
        if event.get("has_frame"):
            st.image(
                client.get_annotated_frame_url(eid),
                use_container_width=True,
            )
    with col_a:
        if event.get("has_audio"):
            st.audio(client.get_audio_url(eid))


# ── Shared results renderer ─────────────────────────────────────────────────

def _show_results(
    results: list[dict], client: APIClient
) -> None:
    """Render search results in a responsive 3-column grid."""
    if not results:
        st.html("""
        <div class="empty-state">
          <div class="icon">🔍</div>
          <p>No se encontraron resultados.<br>
             El índice puede estar vacío.</p>
        </div>
        """)
        return

    st.html(f"""
    <div style="font-size:0.85rem;font-weight:600;
         color:{PALETTE['muted']};margin:1rem 0 0.75rem;">
      {len(results)} evento(s) más similares
    </div>
    """)

    n_cols = min(len(results), 3)
    cols = st.columns(n_cols, gap="medium")
    for i, res in enumerate(results):
        with cols[i % n_cols]:
            _sim_card(
                res["event"],
                res["cosine_similarity"],
                client,
            )


# ── Main page render ─────────────────────────────────────────────────────────

def render(client: APIClient) -> None:
    st.html(page_header(
        "Búsqueda por similitud",
        "Encuentra eventos similares usando embeddings "
        "multimodales (Wav2Vec2 + DINOv2)",
    ))

    # Persistent state across Streamlit reruns
    for key in ("sim_results", "sim_error"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── Search-mode selector ─────────────────────────────────────────
    mode = st.radio(
        "Modo de búsqueda",
        ["📁  Subir archivo", "📋  Evento existente"],
        horizontal=True,
        key="sim_mode",
    )
    is_upload = mode.startswith("📁")

    k = st.slider("Resultados (k)", 1, 20, 5, key="sim_k")

    if is_upload:
        _render_upload_mode(client, k)
    else:
        _render_event_mode(client, k)

    # ── Results / errors ─────────────────────────────────────────────
    if st.session_state.sim_error:
        st.error(
            f"Error en la búsqueda: "
            f"{st.session_state.sim_error}"
        )
        return

    if st.session_state.sim_results is not None:
        _show_results(st.session_state.sim_results, client)


# ── Upload-file mode ─────────────────────────────────────────────────────────

def _render_upload_mode(client: APIClient, k: int) -> None:
    """Encode an uploaded file on-the-fly and search FAISS."""
    with st.container(border=True):
        modality = st.radio(
            "Modalidad",
            ["🎤  Audio", "🖼️  Imagen"],
            horizontal=True,
            key="sim_modality",
        )
        is_audio = modality.startswith("🎤")
        accept = (
            ["wav", "mp3", "ogg"] if is_audio
            else ["jpg", "jpeg", "png", "webp"]
        )
        uploaded = st.file_uploader(
            f"Sube un archivo de "
            f"{'audio' if is_audio else 'imagen'}",
            type=accept,
            key="sim_uploader",
        )

    if uploaded is None:
        st.session_state.sim_results = None
        st.session_state.sim_error = None
        st.html(f"""
        <div class="empty-state">
          <div class="icon">{'🎤' if is_audio else '🖼️'}</div>
          <p>Sube un archivo para buscar eventos similares
             en el índice FAISS.<br>
             <span style="font-size:0.8rem;">Necesitas al menos
             un evento guardado con embedding.</span></p>
        </div>
        """)
        return

    if st.button(
        "Buscar eventos similares",
        type="primary",
        key="sim_search_upload",
    ):
        st.session_state.sim_results = None
        st.session_state.sim_error = None
        with st.spinner(
            "Codificando y buscando en el índice vectorial… "
            "(la primera búsqueda carga los modelos, ~60 s)"
        ):
            try:
                st.session_state.sim_results = (
                    client.search_similar(
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        modality=(
                            "audio" if is_audio else "image"
                        ),
                        k=k,
                    )
                )
            except Exception as exc:
                st.session_state.sim_error = str(exc)


# ── Event-selection mode ─────────────────────────────────────────────────────

def _render_event_mode(client: APIClient, k: int) -> None:
    """Search using a stored event's pre-computed embedding."""
    with st.container(border=True):
        try:
            events = client.list_events(limit=50)
        except Exception:
            st.warning(
                "No se pudo conectar al API para listar eventos."
            )
            return

        if not events:
            st.html("""
            <div class="empty-state">
              <div class="icon">📋</div>
              <p>No hay eventos guardados todavía.<br>
                 Espera a que el pipeline detecte
                 anomalías.</p>
            </div>
            """)
            return

        options = {
            e["id"]: (
                f"#{e['id']} — "
                f"{e['timestamp'][:19].replace('T', ' ')} "
                f"(score: {e.get('anomaly_score', 0):.3f})"
            )
            for e in events
        }
        selected_id = st.selectbox(
            "Selecciona un evento",
            options.keys(),
            format_func=lambda eid: options[eid],
            key="sim_event_select",
        )

    # Show a preview of the selected event as "query card"
    selected_event = next(
        (e for e in events if e["id"] == selected_id), None
    )
    if selected_event:
        _query_card(selected_event, client)

    if st.button(
        "Buscar eventos similares",
        type="primary",
        key="sim_search_event",
    ):
        st.session_state.sim_results = None
        st.session_state.sim_error = None
        with st.spinner("Buscando en el índice vectorial…"):
            try:
                st.session_state.sim_results = (
                    client.search_by_event(
                        event_id=selected_id,
                        k=k,
                    )
                )
            except Exception as exc:
                st.session_state.sim_error = str(exc)
