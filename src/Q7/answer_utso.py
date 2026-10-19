"""
Question 7
    Which brain area has the highest density of putative fast-spiking
    interneurons based on spike waveform and/or spike train features?

Otsu variant of ``answer_waveform.py``: the narrow/broad cut is picked
automatically by Otsu's method instead of by eye at the histogram valley, and
the result is checked against the valley classification.

Otsu maximises the between-class variance, which is biased toward splitting the
larger class when the two classes are unbalanced (here ~1:7 narrow:broad). It
is therefore run on both the raw and the log durations; the log transform
compresses the broad mode's right tail and pulls the cut back toward the valley.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from skimage.filters import threshold_otsu
from sklearn.metrics import cohen_kappa_score

from src.data import ALL_SESSIONS, AREAS, WAVEFORM_SAMPLING_RATE, Dataloader
from src.Q7.answer_waveform import (
    MIN_UNITS,
    THRESHOLD_MS,
    is_narrow,
    waveform_durations,
)
from src.stats import compare_paired, sig_stars


def count_narrow_by_area(
    durations: list[np.ndarray], areas: list[np.ndarray], threshold_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(n_narrow, n_units)``, each [n_sessions, n_areas]."""
    n_narrow = np.zeros((len(durations), len(AREAS)), dtype=int)
    n_units = np.zeros_like(n_narrow)
    for i, (dur, ar) in enumerate(zip(durations, areas)):
        for j, area in enumerate(AREAS):
            d = dur[(ar == area) & np.isfinite(dur)]
            n_narrow[i, j], n_units[i, j] = (
                int(is_narrow(d, threshold_ms).sum()),
                d.size,
            )
    return n_narrow, n_units


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)

    all_durations, all_areas = [], []
    for s in dl:
        all_durations.append(waveform_durations(s.waveforms))
        all_areas.append(s.brain_areas)

    durations = np.concatenate(all_durations)
    areas = np.concatenate(all_areas)
    ok = np.isfinite(durations)
    values, counts = np.unique(durations[ok], return_counts=True)
    log_values = np.log(values)
    k = np.searchsorted(values, threshold_otsu(hist=(counts, values)))
    k_log = np.searchsorted(log_values, threshold_otsu(hist=(counts, log_values)))
    thresholds = {
        "valley": THRESHOLD_MS,
        "Otsu": float((values[k] + values[k + 1]) / 2),
        "Otsu (log)": float(np.exp((log_values[k_log] + log_values[k_log + 1]) / 2)),
    }
    sample_ms = 1e3 / WAVEFORM_SAMPLING_RATE

    print("=== narrow/broad thresholds ===")
    for name, thr in thresholds.items():
        print(f"  {name:<11} {thr:.3f} ms  ({thr / sample_ms:.1f} samples)")

    valley_narrow = is_narrow(durations[ok])
    pooled = {}
    for name, thr in thresholds.items():
        n_narrow, n_units = count_narrow_by_area(all_durations, all_areas, thr)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(n_units >= MIN_UNITS, n_narrow / n_units, np.nan)
        res = compare_paired(fraction, AREAS)
        pooled_narrow, pooled_units = n_narrow.sum(axis=0), n_units.sum(axis=0)
        pooled[name] = pooled_narrow / pooled_units
        _, p_chi2, _, _ = chi2_contingency(
            np.vstack([pooled_narrow, pooled_units - pooled_narrow])
        )
        kappa = cohen_kappa_score(valley_narrow, is_narrow(durations[ok], thr))

        print(f"\n=== {name}: narrow < {thr:.3f} ms ===")
        for j, area in enumerate(AREAS):
            print(
                f"area {area}: pooled {pooled_narrow[j]}/{pooled_units[j]} = {pooled[name][j]:.1%};  "
                f"per-session fraction = {res.mean[j]:.3f} +/- {res.sem[j]:.3f} SEM"
            )
        print(f"Chi-square pooled p = {p_chi2:.4g};  Friedman p = {res.p_omnibus:.4g}")
        for (i, j), hp in zip(res.pairs, res.holm_p):
            print(
                f"  area {AREAS[i]} vs area {AREAS[j]}: p_holm = {hp:.4g} {sig_stars(hp)}"
            )
        print(f"Agreement with valley classification: Cohen's kappa = {kappa:.2f}")
        print(f"-> highest density: area {res.winner('max')}")

    # ---- Figure: duration histogram with the cuts + pooled fractions --------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    line_styles = ["--", "-", ":"]

    # One bin per waveform sample (see answer_waveform.py).
    bins = (np.arange(36) + 0.5) * sample_ms
    axL.hist(durations[ok], bins=bins, color="0.6")
    for (name, thr), ls in zip(thresholds.items(), line_styles):
        axL.axvline(thr, color="k", ls=ls, lw=1, label=f"{name} ({thr:.2f} ms)")
    axL.set_xlabel("Trough-to-peak duration (ms)")
    axL.set_ylabel("Units")
    axL.set_title("Waveform duration, all sessions pooled")
    axL.legend(fontsize=9)

    x = np.arange(len(AREAS))
    width = 0.8 / len(thresholds)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for k, (name, frac) in enumerate(pooled.items()):
        axR.bar(x + (k - 1) * width, frac, width, color=colors[k], label=name)
    axR.set_xticks(x)
    axR.set_xticklabels([f"area {a}" for a in AREAS])
    axR.set_ylabel("Pooled fraction of narrow-spiking units")
    axR.set_title("Fast-spiking density by threshold method")
    axR.legend(fontsize=9)

    fig.tight_layout()
    out = Path("figures/Q7")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "fast_spiking_density_otsu.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
