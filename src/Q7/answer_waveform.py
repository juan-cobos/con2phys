"""
Question 7
    Which brain area has the highest density of putative fast-spiking
    interneurons based on spike waveform and/or spike train features?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

from src.data import ALL_SESSIONS, AREAS, WAVEFORM_SAMPLING_RATE, Dataloader, Session
from src.stats import compare_paired, sig_stars

# Narrow/broad cut, picked from the data: pooled over all sessions the
# trough-to-peak durations are bimodal, with a narrow mode at 6-8 samples, a
# broad mode at 14-16 samples and a clear valley at ~10.5 samples. Expressed in
# ms with the waveform sampling rate (30 kHz); THRESHOLD_SWEEP shows how much
# the answer depends on the exact value.
VALLEY_SAMPLES = 10.5
THRESHOLD_MS = VALLEY_SAMPLES / WAVEFORM_SAMPLING_RATE * 1e3
THRESHOLD_SWEEP = tuple(
    n / WAVEFORM_SAMPLING_RATE * 1e3 for n in (7.5, 9.0, 10.5, 12.0, 13.5)
)
MIN_UNITS = 5


def waveform_durations(
    waveforms: np.ndarray, fs: float = WAVEFORM_SAMPLING_RATE
) -> np.ndarray:
    """Trough-to-peak duration (ms) of each mean waveform. [n_units]"""
    w = waveforms.astype(float)
    trough = w.argmin(axis=1)
    peak = np.array([t + row[t:].argmax() for row, t in zip(w, trough)])
    duration = (peak - trough) / fs * 1e3

    negative_spike = np.abs(w.min(axis=1)) > np.abs(w.max(axis=1))
    valid = negative_spike & (peak > trough)
    return np.where(valid, duration, np.nan)


def is_narrow(
    duration_ms: np.ndarray, threshold_ms: float = THRESHOLD_MS
) -> np.ndarray:
    """Boolean mask of putative fast-spiking units (NaN durations -> False)."""
    return duration_ms < threshold_ms


def count_by_area(
    session: Session, duration_ms: np.ndarray
) -> dict[int, tuple[int, int]]:
    """``{area: (n_narrow, n_classified)}`` for one session's units."""
    out = {}
    for area in AREAS:
        d = duration_ms[session.area_mask(area)]
        d = d[np.isfinite(d)]
        out[area] = (int(is_narrow(d).sum()), int(d.size))
    return out


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)

    # Every unit's duration and area, pooled for the histogram / chi-square.
    all_durations: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    # [n_sessions, n_areas] narrow / classified counts per session.
    n_narrow = np.zeros((len(dl), len(AREAS)), dtype=int)
    n_units = np.zeros_like(n_narrow)

    print(
        f"=== putative fast-spiking units (trough-to-peak < {THRESHOLD_MS:.2f} ms) ==="
    )
    for i, s in enumerate(dl):
        duration = waveform_durations(s.waveforms)
        all_durations.append(duration)
        all_areas.append(s.brain_areas)
        cells = []
        for j, (area, (k, n)) in enumerate(count_by_area(s, duration).items()):
            n_narrow[i, j], n_units[i, j] = k, n
            frac = k / n if n else float("nan")
            cells.append(f"area {area}: {k:>3}/{n:<3} ({frac:5.1%})")
        dropped = int(np.isnan(duration).sum())
        print(
            f"session {s.id:>2}  " + "  ".join(cells) + f"   (unclassifiable {dropped})"
        )

    durations = np.concatenate(all_durations)
    areas = np.concatenate(all_areas)
    ok = np.isfinite(durations)

    # Per-session fraction, NaN where the area has too few units; compare_paired
    # drops those sessions so every remaining row has all three areas.
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n_units >= MIN_UNITS, n_narrow / n_units, np.nan)
    res = compare_paired(fraction, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

    pooled_narrow, pooled_units = n_narrow.sum(axis=0), n_units.sum(axis=0)
    pooled_frac = pooled_narrow / pooled_units
    # Chi-square on the pooled narrow/broad counts: are the proportions the
    # same in the three areas? Complements the paired test, which weighs every
    # session equally regardless of how many units it contributed.
    table = np.vstack([pooled_narrow, pooled_units - pooled_narrow])
    chi2, p_chi2, _, _ = chi2_contingency(table)

    print("\n=== fast-spiking density by area ===")
    for j, area in enumerate(AREAS):
        print(
            f"area {area}: pooled {pooled_narrow[j]}/{pooled_units[j]} = {pooled_frac[j]:.1%};  "
            f"per-session fraction = {mean[j]:.3f} +/- {sem[j]:.3f} SEM"
        )
    print(
        f"(n = {res.n} sessions with >= {MIN_UNITS} classifiable units in every area; "
        f"{int(ok.sum())}/{durations.size} units classifiable)"
    )
    print(f"\nChi-square on pooled counts: chi2 = {chi2:.3f}, p = {p_chi2:.4g}")
    print(
        f"Friedman across areas (per-session fractions): chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}"
    )
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    print("\nThreshold sensitivity (pooled fraction narrow per area):")
    for thr in THRESHOLD_SWEEP:
        fr = [is_narrow(durations[ok & (areas == a)], thr).mean() for a in AREAS]
        best_thr = AREAS[int(np.argmax(fr))]
        print(
            f"  < {thr:.2f} ms: "
            + "  ".join(f"area {a}: {f:.1%}" for a, f in zip(AREAS, fr))
            + f"   -> area {best_thr}"
        )

    best = res.winner("max")
    print(f"\nHighest density of putative fast-spiking interneurons: area {best}")

    # ---- Figure: duration histograms per area + per-session fractions ------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # One bin per waveform sample: durations are whole samples, so finer or
    # misaligned bins alias into a comb and coarser ones hide the narrow mode.
    bins = (np.arange(36) + 0.5) / WAVEFORM_SAMPLING_RATE * 1e3
    for j, area in enumerate(AREAS):
        axL.hist(
            durations[ok & (areas == area)],
            bins=bins,
            color=colors[j],
            alpha=0.55,
            label=f"area {area} (n={int((ok & (areas == area)).sum())})",
        )
    axL.axvline(THRESHOLD_MS, color="k", ls="--", lw=1)
    axL.text(
        THRESHOLD_MS, axL.get_ylim()[1] * 0.95, f" {THRESHOLD_MS:.2f} ms", va="top"
    )
    axL.set_xlabel("Trough-to-peak duration (ms)")
    axL.set_ylabel("Units")
    axL.set_title(
        "Waveform duration, all sessions pooled\n(narrow-spiking = left of the line)"
    )
    axL.legend(fontsize=9)

    x = np.arange(len(AREAS))
    axR.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
    for row, dx in zip(paired, jitter):
        axR.plot(x + dx, row, color="k", lw=0.5, alpha=0.2, zorder=3)
    for j in range(len(AREAS)):
        axR.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.5, zorder=4)
    axR.scatter(
        x, pooled_frac, marker="_", s=600, color="k", lw=2, zorder=5, label="pooled"
    )

    top = max((mean + sem).max(), paired.max())
    step = top * 0.08
    for k, ((i, j), hp) in enumerate(zip(pairs, holm)):
        y = top + step * (k + 1.2)
        axR.plot(
            [x[i], x[i], x[j], x[j]],
            [y, y + step * 0.2, y + step * 0.2, y],
            c="k",
            lw=1,
        )
        axR.text(
            (x[i] + x[j]) / 2, y + step * 0.2, sig_stars(hp), ha="center", va="bottom"
        )
    axR.set_xticks(x)
    axR.set_xticklabels([f"area {a}" for a in AREAS])
    axR.set_ylabel("Fraction of narrow-spiking units")
    axR.set_ylim(0, top + step * (len(pairs) + 2))
    axR.set_title(
        f"Fast-spiking density per session\nFriedman p = {p_omnibus:.3g}, pooled chi2 p = {p_chi2:.3g}"
    )
    axR.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = Path("figures/Q7")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "fast_spiking_density.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
