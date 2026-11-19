"""Offline replay bench for the detection pipeline (no microphone/camera).

Feeds an audio signal window-by-window through ``Pipeline.process_window`` —
the exact code path used live — and reports detection metrics against a
known ground truth.

Usage::

    poetry run python -m scripts.replay --scenario events
    poetry run python -m scripts.replay --scenario quiet --minutes 15
    poetry run python -m scripts.replay --scenario events --encode-delay 2.0
    poetry run python -m scripts.replay --wav grabacion.wav   # sin ground truth

Embeddings are mocked by default (deterministic random vectors) so the bench
runs offline; ``--encode-delay`` simulates the cost of Wav2Vec2 + DINOv2.
All storage goes to a temporary directory; nothing in ``data/`` is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import numpy as np

SR = 16000
WIN = 4096
WIN_S = WIN / SR  # 0.256 s — live windows do not overlap (blocksize=WIN)


# --------------------------------------------------------------------------- #
#  Synthetic scenarios                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class TruthEvent:
    name: str
    start: float
    end: float


def _background(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary room-like background: low-passed noise + mains hum."""
    from scipy.signal import lfilter

    white = rng.normal(0.0, 1.0, n)
    # one-pole low-pass → reddish noise
    b = lfilter([0.05], [1.0, -0.95], white)
    b = b / (np.std(b) + 1e-12) * 0.01
    t = np.arange(n) / SR
    hum = 0.003 * np.sin(2 * np.pi * 60 * t)
    # slow loudness wander (±20 %) so the background is not perfectly flat
    wander = 1.0 + 0.2 * np.sin(2 * np.pi * t / 37.0)
    return ((b + hum) * wander).astype(np.float32)


def _insert(sig: np.ndarray, start_s: float, x: np.ndarray) -> None:
    i = int(start_s * SR)
    sig[i: i + len(x)] += x[: max(0, len(sig) - i)]


def make_scenario(name: str, minutes: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * SR)
    sig = _background(n, rng)
    truth: list[TruthEvent] = []
    if name == "quiet":
        return sig, truth

    def tone(dur, f, amp=0.2):
        t = np.arange(int(dur * SR)) / SR
        env = np.minimum(1.0, np.minimum(t, dur - t) / 0.02)
        return (amp * env * np.sin(2 * np.pi * f * t)).astype(np.float32)

    def burst(dur, amp=0.15):
        k = int(dur * SR)
        env = np.minimum(1.0, np.minimum(np.arange(k), k - np.arange(k)) / 320)
        return (amp * env * rng.normal(0, 1, k)).astype(np.float32)

    def chirp(dur, f0=300, f1=4000, amp=0.2):
        t = np.arange(int(dur * SR)) / SR
        ph = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * dur))
        return (amp * np.sin(ph)).astype(np.float32)

    def harmonic(dur, f0=180, amp=0.15):
        t = np.arange(int(dur * SR)) / SR
        x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
        am = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # syllable-like AM
        return (amp * am * x / 2).astype(np.float32)

    plan = [
        ("impulso_0.3s", 200.0, burst(0.3, amp=0.3)),
        ("rafaga_2s", 240.0, burst(2.0)),
        ("tono_15s", 280.0, tone(15.0, 1000)),
        ("doble_rafaga_a", 330.0, burst(1.5)),
        ("doble_rafaga_b", 332.5, burst(1.5)),
        ("chirp_3s", 380.0, chirp(3.0)),
        ("armonico_4s", 420.0, harmonic(4.0)),
    ]
    for label, st, x in plan:
        if st + len(x) / SR < minutes * 60:
            _insert(sig, st, x)
            truth.append(TruthEvent(label, st, st + len(x) / SR))
    return sig, truth


# --------------------------------------------------------------------------- #
#  Replay                                                                     #
# --------------------------------------------------------------------------- #

class _FakeEncoder:
    """Deterministic stand-in for MultimodalEncoder (offline, optional delay)."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._rng = np.random.default_rng(123)

    def encode(self, audio, frame=None, sample_rate=16000):
        if self.delay:
            time.sleep(self.delay)
        v = self._rng.normal(size=1536).astype(np.float32)
        return v / np.linalg.norm(v)

    def ensure_downloaded(self):
        return None


def replay(sig: np.ndarray, encode_delay: float, workdir: Path, env: dict):
    os.environ.update({
        "EVENTS_DIR": str(workdir / "eventos"),
        "DB_PATH": str(workdir / "events.db"),
        "FAISS_PATH": str(workdir / "faiss.index"),
        "SNAPSHOTS_DIR": str(workdir / "snapshots"),
        "API_BASE_URL": "http://127.0.0.1:9",  # unroutable: never reached
        **env,
    })
    # Detector state paths are relative to CWD: isolate them in workdir.
    os.chdir(workdir)
    from src.pipeline import Pipeline

    p = Pipeline()
    p.encoder = _FakeEncoder(encode_delay)
    p._notify_api = lambda *a, **k: None  # no API during replay
    p.db.init()
    p.faiss_store.init()

    saved_at: list[float] = []
    sim_now = {"t": 0.0}
    orig_handle = p._handle_anomaly

    def _wrapped(*a, **k):
        saved_at.append(sim_now["t"])
        return orig_handle(*a, **k)

    p._handle_anomaly = _wrapped

    n_win = len(sig) // WIN
    service = np.empty(n_win)
    flags = np.zeros(n_win, dtype=bool)
    raws = np.zeros(n_win)
    t0 = time.time()
    for k in range(n_win):
        sim_now["t"] = (k + 1) * WIN_S  # window end time (scenario clock)
        w = sig[k * WIN: (k + 1) * WIN]
        a = time.perf_counter()
        # ts = capture midpoint of the window, as the live callback reports.
        out = p.process_window(w, t0 + (k + 0.5) * WIN_S)
        service[k] = time.perf_counter() - a
        flags[k] = bool(out.get("gate", False))
        raws[k] = float(out["result"].raw_score)
    if hasattr(p, "flush_events"):
        # Async event writer: wait for pending writes, then take each event's
        # onset from its stored timestamp (scenario clock = ts - t0).
        p.flush_events()
        rows = sorted(p.db.list_events(limit=100000), key=lambda r: r.timestamp)
        # SQLite drops tzinfo: stored values are UTC, read back naive.
        saved_at = [
            r.timestamp.replace(tzinfo=timezone.utc).timestamp() - t0
            for r in rows
        ]
        durations = [r.duration_s for r in rows]
    else:
        durations = [None] * len(saved_at)
    n_db = len(p.db.list_events(limit=100000))
    return service, flags, saved_at, n_db, raws, durations


def simulate_drops(service: np.ndarray, period: float, qmax: int = 64) -> int:
    """Real-time queue simulation: windows arrive every ``period`` s into a
    bounded queue (maxsize=qmax, like the live pipeline); a full queue drops
    the window. Returns the number of dropped windows."""
    clock = 0.0       # time the worker becomes free
    queue_ends: list[float] = []  # finish times of queued/in-service items
    drops = 0
    for k, s in enumerate(service):
        arrival = (k + 1) * period
        queue_ends = [e for e in queue_ends if e > arrival]
        if len(queue_ends) >= qmax + 1:  # qmax waiting + 1 in service
            drops += 1
            continue
        start = max(arrival, clock)
        clock = start + s
        queue_ends.append(clock)
    return drops


def evaluate(truth, saved_at, flags, minutes, margin=1.0, durations=None):
    durations = durations or [None] * len(saved_at)
    rows = []
    used = set()
    for ev in truth:
        hits = [i for i, t in enumerate(saved_at)
                if ev.start - margin <= t <= ev.end + margin + 2.0]
        used.update(hits)
        win_lo, win_hi = int(ev.start / WIN_S), int(np.ceil(ev.end / WIN_S))
        cover = float(flags[win_lo:win_hi].mean()) if win_hi > win_lo else 0.0
        rows.append({
            "evento": ev.name,
            "duracion_s": round(ev.end - ev.start, 2),
            "detectado": bool(hits),
            "eventos_guardados": len(hits),
            "error_inicio_s": round(saved_at[hits[0]] - ev.start, 2) if hits else None,
            # Stored duration of the (first) event that covers this sound.
            "duracion_evento_s": (
                round(durations[hits[0]], 2)
                if hits and durations[hits[0]] is not None else None
            ),
            "cobertura_ventanas": round(cover, 2),
        })
    false_alarms = [t for i, t in enumerate(saved_at) if i not in used]
    return rows, false_alarms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", choices=["events", "quiet"], default="events")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--wav", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encode-delay", type=float, default=0.0)
    ap.add_argument("--json", type=str, default=None, help="write report JSON here")
    ap.add_argument("--env", action="append", default=[],
                    help="extra env var KEY=VALUE for the pipeline (repeatable)")
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))

    if args.wav:
        import soundfile as sf
        sig, sr = sf.read(args.wav, dtype="float32")
        if sig.ndim > 1:
            sig = sig[:, 0]
        if sr != SR:
            raise SystemExit(f"WAV must be {SR} Hz mono (got {sr} Hz)")
        truth, minutes = [], len(sig) / SR / 60
    else:
        minutes = args.minutes or (8.0 if args.scenario == "events" else 15.0)
        sig, truth = make_scenario(args.scenario, minutes, args.seed)

    env = dict(kv.split("=", 1) for kv in args.env)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        try:
            service, flags, saved_at, n_db, raws, durations = replay(
                sig, args.encode_delay, Path(tmp), env
            )
        finally:
            os.chdir(cwd)

    rows, fa = evaluate(truth, saved_at, flags, minutes, durations=durations)
    drops = simulate_drops(service, WIN_S)
    hours = minutes / 60
    report = {
        "escenario": args.wav or args.scenario,
        "minutos": round(minutes, 2),
        "ventanas": int(len(service)),
        "eventos_guardados_total": int(n_db),
        "eventos_por_verdad": rows,
        "falsas_alarmas": len(fa),
        "falsas_alarmas_por_hora": round(len(fa) / hours, 1) if hours else None,
        "falsas_alarmas_en_s": [round(t, 1) for t in fa[:20]],
        "tiempo_ventana_ms": {
            "p50": round(float(np.percentile(service, 50)) * 1e3, 1),
            "p99": round(float(np.percentile(service, 99)) * 1e3, 1),
            "max": round(float(service.max()) * 1e3, 1),
        },
        "ventanas_perdidas_simuladas": int(drops),
        "encode_delay_s": args.encode_delay,
        # Fingerprints of the detector's behaviour: identical hashes mean the
        # per-window decisions and raw scores did not change (refactor guard).
        # flags_sha1 is comparable across machines; scores_sha1 only on the
        # same machine (floating-point results differ across CPU/BLAS).
        "flags_sha1": hashlib.sha1(flags.astype(np.uint8).tobytes()).hexdigest(),
        "scores_sha1": hashlib.sha1(
            np.round(raws, 6).astype(np.float64).tobytes()
        ).hexdigest(),
    }
    txt = json.dumps(report, indent=2, ensure_ascii=False)
    print(txt)
    if args.json:
        Path(args.json).write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
