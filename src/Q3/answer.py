"""
Question 3
    Which brain area (if any) has the highest density of ripples (i.e. “hippocampal” ripples traditionally occurring during sharp wave-ripples)?
"""

import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Use the local, patched ripple_detection clone (stats computation stripped from
# the detectors for speed) instead of an installed copy. The package lives one
# level down inside the clone, so put the clone dir on the path ahead of it.
_RIPPLE_DETECTION = Path(__file__).resolve().parents[2] / "ripple_detection"
sys.path.insert(0, str(_RIPPLE_DETECTION))

from ripple_detection.detectors import (  # noqa: E402
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Roumis_ripple_detector,
)
from scipy.signal import butter, filtfilt

from src.data import ALL_SESSIONS, AREAS, DATA_DIR, SAMPLING_RATE

# Ripple band. Traditionally 150-250 Hz, but the LFP is sampled at 500 Hz so the
# Nyquist limit is 250 Hz: the library filters won't run here, and the top of
# the band is unresolvable. Clip just under Nyquist and expose it for revisiting.
RIPPLE_BAND = (150.0, 240.0)
DETECTORS = (
    Kay_ripple_detector,
    Karlsson_ripple_detector,
    Roumis_ripple_detector,
)


def filter_ripple_band(lfp: np.ndarray) -> np.ndarray:
    """Bandpass a [num_channels, num_timestamps] LFP into the ripple band.

    Returns [num_timestamps, num_channels], the layout the detectors expect.
    """
    b, a = butter(4, [f / (SAMPLING_RATE / 2) for f in RIPPLE_BAND], btype="band")
    return filtfilt(b, a, lfp, axis=1).T


def _count_events(
    detector, time: np.ndarray, filtered: np.ndarray, speed: np.ndarray
) -> int:
    """Run one detector on already-filtered LFP and return the event count.

    We have no running speed, so speed is zeroed (always below threshold), which
    disables the immobility gate and keeps every candidate event.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(detector(time, filtered, speed, SAMPLING_RATE))


def get_swr(detector, lfp: np.ndarray) -> int:
    """Bandpass one area's LFP and count events from a single detector."""
    filtered = filter_ripple_band(lfp)
    time = np.arange(filtered.shape[0]) / SAMPLING_RATE
    speed = np.zeros(filtered.shape[0])
    return _count_events(detector, time, filtered, speed)


def detect_session_area(
    task: tuple[int, int],
) -> tuple[int, int, dict[str, int], float]:
    """Worker: per-detector ripple counts and duration (s) for one (session, area).

    Loads just that area's LFP file rather than the whole Session, so each
    worker only holds one recording in memory, and filters it once before
    running every detector on the shared filtered signal. Duration comes from
    the number of timestamps (dim 1 of the [num_channels, num_timestamps] LFP),
    which lets the caller normalize counts into a rate.
    """
    session_id, area = task
    lfp = np.load(DATA_DIR / str(session_id) / f"lfp_{area}.npy")
    duration_s = lfp.shape[1] / SAMPLING_RATE
    filtered = filter_ripple_band(lfp)
    time = np.arange(filtered.shape[0]) / SAMPLING_RATE
    speed = np.zeros(filtered.shape[0])
    counts = {
        det.__name__: _count_events(det, time, filtered, speed) for det in DETECTORS
    }
    return session_id, area, counts, duration_s


if __name__ == "__main__":
    sessions = ALL_SESSIONS
    sessions.pop(11)  # remove duplicated lfp

    # One independent job per (session, area). Each LFP recording is hundreds of
    # MB and filtering doubles that, so default to a modest worker count to stay
    # within RAM; override with Q3_WORKERS for machines with more headroom.
    tasks = [(sid, area) for sid in sessions for area in AREAS]
    max_workers = int(os.environ.get("Q3_WORKERS", 4))
    print(
        f"Detecting ripples in {len(tasks)} (session, area) jobs "
        f"across {max_workers} workers"
    )

    # counts[detector_name][area][session_id] -> count; durations[area][session_id].
    counts: dict[str, dict[int, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    durations: dict[int, dict[int, float]] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(detect_session_area, t) for t in tasks]
        for done, fut in enumerate(as_completed(futures), start=1):
            session_id, area, det_counts, duration_s = fut.result()
            durations[area][session_id] = duration_s
            for det_name, count in det_counts.items():
                counts[det_name][area][session_id] = count
            print(
                f"  [{done:>2}/{len(tasks)}] session {session_id:>2} area {area} done"
            )

    # Build a duration-normalized report (ripples/min). Sessions differ in
    # length, so a raw count favors longer recordings; every number below is a
    # rate. Print it and write it to results.txt next to this script.
    lines: list[str] = []
    names = [det.__name__ for det in DETECTORS]

    lines.append("Per-session ripple rate (ripples/min, normalized by duration)")
    lines.append(
        f"{'sess':>4} {'area':>4} {'minutes':>8}  "
        + "  ".join(f"{name.split('_')[0]:>10}" for name in names)
    )
    for sid in sessions:
        for area in AREAS:
            dur_min = durations[area][sid] / 60
            rates = "  ".join(
                f"{counts[name][area][sid] / dur_min:>10.2f}" for name in names
            )
            lines.append(f"{sid:>4} {area:>4} {dur_min:>8.1f}  {rates}")

    lines.append("")
    lines.append("Ripple density by brain area, pooled over sessions (ripples/min)")
    for name in names:
        lines.append(f"\n{name}:")
        lines.append(
            f"{'area':>4}  {'ripples':>8}  {'minutes':>8}  {'ripples/min':>12}"
        )
        for area in AREAS:
            total_count = sum(counts[name][area].values())
            total_min = sum(durations[area].values()) / 60
            lines.append(
                f"{area:>4}  {total_count:>8}  {total_min:>8.1f}  {total_count / total_min:>12.2f}"
            )

    report = "\n".join(lines)
    print("\n" + report)
    results_path = Path(__file__).parent / "results.txt"
    results_path.write_text(report + "\n")
    print(f"\nWrote results to {results_path}")
