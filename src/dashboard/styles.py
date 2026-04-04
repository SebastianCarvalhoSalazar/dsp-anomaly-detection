"""Shared CSS and HTML helpers for the dashboard."""
from __future__ import annotations


PALETTE = {
    "primary":   "#7C3AED",   # violet
    "anomaly":   "#EF4444",   # red
    "normal":    "#10B981",   # emerald
    "warning":   "#F59E0B",   # amber
    "surface":   "#FFFFFF",
    "bg":        "#F1F5F9",
    "text":      "#0F172A",
    "muted":     "#64748B",
    "border":    "#E2E8F0",
}


def inject_global_css() -> str:
    """Return the full CSS block to inject once per page."""
    return f"""
<style>
/* ── Reset / base ───────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
.stApp {{
    background: {PALETTE['bg']};
}}

/* ── Hide Streamlit chrome ──────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebarNav"] {{ display: none; }}
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {PALETTE['text']};
    border-right: none;
}}
section[data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
}}
section[data-testid="stSidebar"] .stRadio > label {{
    color: #94A3B8 !important;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 2px 0;
    transition: background 0.2s;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    background: rgba(124,58,237,0.25);
}}

/* ── Cards ──────────────────────────────────────────── */
.dsp-card {{
    background: {PALETTE['surface']};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}}
.dsp-card-accent-primary  {{ border-left: 4px solid {PALETTE['primary']}; }}
.dsp-card-accent-anomaly  {{ border-left: 4px solid {PALETTE['anomaly']}; }}
.dsp-card-accent-normal   {{ border-left: 4px solid {PALETTE['normal']}; }}
.dsp-card-accent-warning  {{ border-left: 4px solid {PALETTE['warning']}; }}

/* ── Page header ────────────────────────────────────── */
.dsp-page-header {{
    background: linear-gradient(135deg, {PALETTE['primary']} 0%, #4F46E5 100%);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
}}
.dsp-page-header h1 {{
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}}
.dsp-page-header p {{
    margin: 0.25rem 0 0;
    opacity: 0.82;
    font-size: 0.9rem;
}}

/* ── Score badge ────────────────────────────────────── */
.score-pill {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
.score-high  {{ background: #FEE2E2; color: #B91C1C; }}
.score-mid   {{ background: #FEF3C7; color: #92400E; }}
.score-low   {{ background: #D1FAE5; color: #065F46; }}

/* ── Big score display ──────────────────────────────── */
.big-score {{
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
}}
.big-score.anomaly {{ color: {PALETTE['anomaly']}; }}
.big-score.normal  {{ color: {PALETTE['normal']}; }}
.big-score.warmup  {{ color: {PALETTE['warning']}; }}

/* ── Live dot ───────────────────────────────────────── */
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.4; transform:scale(0.85); }}
}}
.live-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.6s ease-in-out infinite;
}}
.live-dot.red    {{ background: {PALETTE['anomaly']}; }}
.live-dot.green  {{ background: {PALETTE['normal']}; }}
.live-dot.amber  {{ background: {PALETTE['warning']}; }}

/* ── Status chip ────────────────────────────────────── */
.status-chip {{
    display: inline-flex;
    align-items: center;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    gap: 6px;
}}
.chip-anomaly {{ background:#FEE2E2; color:#B91C1C; }}
.chip-normal  {{ background:#D1FAE5; color:#065F46; }}
.chip-warmup  {{ background:#FEF3C7; color:#92400E; }}
.chip-offline {{ background:#F1F5F9; color:#475569; }}

/* ── Event card ─────────────────────────────────────── */
.event-row {{
    background: {PALETTE['surface']};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}}
.event-row:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.10); }}

/* ── Metric row ─────────────────────────────────────── */
.metric-box {{
    background: {PALETTE['surface']};
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.metric-box .label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {PALETTE['muted']};
    margin-bottom: 0.3rem;
}}
.metric-box .value {{
    font-size: 1.9rem;
    font-weight: 800;
    color: {PALETTE['text']};
    letter-spacing: -0.03em;
    line-height: 1;
}}

/* ── Similarity bar ─────────────────────────────────── */
.sim-bar-wrap {{
    background: {PALETTE['border']};
    border-radius: 999px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}}
.sim-bar-fill {{
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, {PALETTE['normal']}, {PALETTE['primary']});
}}

/* ── Empty state ────────────────────────────────────── */
.empty-state {{
    text-align: center;
    padding: 3rem 1rem;
    color: {PALETTE['muted']};
}}
.empty-state .icon {{ font-size: 3rem; margin-bottom: 0.5rem; }}
.empty-state p {{ font-size: 0.95rem; margin: 0; }}
</style>
"""


def score_class(score: float) -> str:
    if score >= 0.65:
        return "score-high"
    if score >= 0.35:
        return "score-mid"
    return "score-low"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB hex color to rgba(r,g,b,alpha) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def score_color(score: float) -> str:
    if score >= 0.65:
        return PALETTE["anomaly"]
    if score >= 0.35:
        return PALETTE["warning"]
    return PALETTE["normal"]


def big_score_class(score: float, is_fitted: bool) -> str:
    if not is_fitted:
        return "warmup"
    return "anomaly" if score >= 0.5 else "normal"


def page_header(title: str, subtitle: str = "") -> str:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return f"""
<div class="dsp-page-header">
  <h1>{title}</h1>
  {sub}
</div>
"""


def score_pill(score: float) -> str:
    cls = score_class(score)
    return f'<span class="score-pill {cls}">{score:.3f}</span>'


def status_chip(is_anomaly: bool, is_fitted: bool) -> str:
    if not is_fitted:
        return '<span class="status-chip chip-warmup"><span class="live-dot amber"></span>Calentando</span>'
    if is_anomaly:
        return '<span class="status-chip chip-anomaly"><span class="live-dot red"></span>ANOMALÍA</span>'
    return '<span class="status-chip chip-normal"><span class="live-dot green"></span>Normal</span>'
