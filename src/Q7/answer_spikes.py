"""
Question 7
    Which brain area has the highest density of putative fast-spiking
    interneurons based on spike waveform and/or spike train features?

Spike-train-only variant of ``answer_waveform.py``: units are classified from
their firing statistics without looking at the waveform, and the result is
checked unit-by-unit against the waveform classification.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

from src.data import ALL_SESSIONS, AREAS, Dataloader, Session
from src.Q7.answer_waveform import THRESHOLD_MS, waveform_durations
from src.stats import compare_paired, sig_stars

RATE_HZ = 10.0
RATE_SWEEP = (5.0, 10.0, 15.0, 20.0)
# ISIs shorter than this count as burst/high-frequency firing.
BURST_ISI_S = 0.010
# Units with fewer spikes than this have too few ISIs for stable statistics.
MIN_SPIKES = 100
# Same per-session sample-size guard as the waveform variant.
MIN_UNITS = 5


def spike_features(
    spike_times: np.ndarray, duration_s: float
) -> tuple[float, float, float]:
    """(firing rate Hz, ISI coefficient of variation, fraction of ISIs < BURST_ISI_S)
    of one unit; all NaN when the train has fewer than MIN_SPIKES spikes."""
    if spike_times.size < MIN_SPIKES:
        return (np.nan, np.nan, np.nan)
    isi = np.diff(np.sort(spike_times))
    return (
        spike_times.size / duration_s,
        float(isi.std() / isi.mean()),
        float(np.mean(isi < BURST_ISI_S)),
    )


def is_fast_spiking(rate_hz: np.ndarray, threshold_hz: float = RATE_HZ) -> np.ndarray:
    """Boolean mask of putative fast-spiking units (NaN rates -> False)."""
    return rate_hz > threshold_hz


def session_features(session: Session) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-unit (rate, cv, burst) arrays aligned with the session's unit order."""
    rate = np.full(session.n_units, np.nan)
    cv = np.full(session.n_units, np.nan)
    burst = np.full(session.n_units, np.nan)
    for area in AREAS:
        units = np.flatnonzero(session.area_mask(area))
        for u, train in zip(units, session.get_unit_spikes_in(area)):
            rate[u], cv[u], burst[u] = spike_features(train, session.duration_s)
    return rate, cv, burst


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)

    all_rate, all_cv, all_burst, all_dur, all_area = [], [], [], [], []
    n_fs = np.zeros((len(dl), len(AREAS)), dtype=int)
    n_units = np.zeros_like(n_fs)

    print(f"=== putative fast-spiking units (firing rate > {RATE_HZ:g} Hz) ===")
    for i, s in enumerate(dl):
        rate, cv, burst = session_features(s)
        all_rate.append(rate)
        all_cv.append(cv)
        all_burst.append(burst)
        all_dur.append(waveform_durations(s.waveforms))  # for the agreement check only
        all_area.append(s.brain_areas)
        cells = []
        for j, area in enumerate(AREAS):
            r = rate[s.area_mask(area)]
            r = r[np.isfinite(r)]
            n_fs[i, j], n_units[i, j] = int(is_fast_spiking(r).sum()), r.size
            frac = n_fs[i, j] / n_units[i, j] if n_units[i, j] else float("nan")
            cells.append(
                f"area {area}: {n_fs[i, j]:>3}/{n_units[i, j]:<3} ({frac:5.1%})"
            )
        print(f"session {s.id:>2}  " + "  ".join(cells))

    rate, cv, burst, duration, areas = map(
        np.concatenate, (all_rate, all_cv, all_burst, all_dur, all_area)
    )
    ok = np.isfinite(rate)
    fs = is_fast_spiking(rate)

    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n_units >= MIN_UNITS, n_fs / n_units, np.nan)
    res = compare_paired(fraction, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

    pooled_fs, pooled_units = n_fs.sum(axis=0), n_units.sum(axis=0)
    pooled_frac = pooled_fs / pooled_units
    chi2, p_chi2, _, _ = chi2_contingency(
        np.vstack([pooled_fs, pooled_units - pooled_fs])
    )

    print("\n=== fast-spiking density by area (spike-train criterion) ===")
    for j, area in enumerate(AREAS):
        print(
            f"area {area}: pooled {pooled_fs[j]}/{pooled_units[j]} = {pooled_frac[j]:.1%};  "
            f"per-session fraction = {mean[j]:.3f} +/- {sem[j]:.3f} SEM;  "
            f"median rate {np.median(rate[ok & (areas == area)]):.2f} Hz"
        )
    print(
        f"(n = {res.n} sessions; {int(ok.sum())}/{rate.size} units with >= {MIN_SPIKES} spikes)"
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

    print("\nThreshold sensitivity (pooled fraction fast-spiking per area):")
    for thr in RATE_SWEEP:
        fr = [is_fast_spiking(rate[ok & (areas == a)], thr).mean() for a in AREAS]
        print(
            f"  > {thr:>4.1f} Hz: "
            + "  ".join(f"area {a}: {f:.1%}" for a, f in zip(AREAS, fr))
            + f"   -> area {AREAS[int(np.argmax(fr))]}"
        )

    best = res.winner("max")
    print(
        f"\nHighest density of putative fast-spiking interneurons (spike train): area {best}"
    )

    # ---- Consistency with the waveform classification, unit by unit --------
    both = ok & np.isfinite(duration)
    narrow = duration < THRESHOLD_MS
    kappa = cohen_kappa_score(narrow[both], fs[both])
    print(
        f"\n=== agreement with the waveform criterion (< {THRESHOLD_MS} ms), {int(both.sum())} units ==="
    )
    print(f"Cohen's kappa = {kappa:.3f}")
    print(
        f"P(narrow | rate > {RATE_HZ:g} Hz) = {narrow[both & fs].mean():.2f}   "
        f"P(narrow | rate <= {RATE_HZ:g} Hz) = {narrow[both & ~fs].mean():.2f}   "
        f"(base rate {narrow[both].mean():.2f})"
    )
    print(
        f"P(rate > {RATE_HZ:g} Hz | narrow) = {fs[both & narrow].mean():.2f}   "
        f"P(rate > {RATE_HZ:g} Hz | broad) = {fs[both & ~narrow].mean():.2f}"
    )
    print("spike-train features by waveform class (median):")
    for name, feat in (
        ("rate (Hz)", rate),
        ("ISI CV", cv),
        (f"ISI < {BURST_ISI_S * 1e3:g} ms", burst),
    ):
        print(
            f"  {name:<12} narrow {np.nanmedian(feat[both & narrow]):.3f}   "
            f"broad {np.nanmedian(feat[both & ~narrow]):.3f}"
        )

    # ---- Figure: rate histograms + rate vs duration + per-session fractions --
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bins = np.logspace(-1, 2, 40)
    for j, area in enumerate(AREAS):
        sel = ok & (areas == area)
        axL.hist(
            rate[sel],
            bins=bins,
            color=colors[j],
            alpha=0.55,
            label=f"area {area} (n={int(sel.sum())})",
        )
    axL.axvline(RATE_HZ, color="k", ls="--", lw=1)
    axL.text(RATE_HZ, axL.get_ylim()[1] * 0.95, f" {RATE_HZ:g} Hz", va="top")
    axL.set_xscale("log")
    axL.set_xlabel("Mean firing rate (Hz)")
    axL.set_ylabel("Units")
    axL.set_title(
        "Firing rate, all sessions pooled\n(fast-spiking = right of the line)"
    )
    axL.legend(fontsize=9)

    for j, area in enumerate(AREAS):
        sel = both & (areas == area)
        axM.scatter(
            duration[sel],
            rate[sel],
            s=10,
            color=colors[j],
            alpha=0.5,
            label=f"area {area}",
        )
    axM.axhline(RATE_HZ, color="k", ls="--", lw=1)
    axM.axvline(THRESHOLD_MS, color="k", ls="--", lw=1)
    axM.set_yscale("log")
    axM.set_xlabel("Trough-to-peak duration (ms)")
    axM.set_ylabel("Mean firing rate (Hz)")
    axM.set_title(f"Spike-train vs waveform criterion\nCohen's kappa = {kappa:.2f}")
    axM.legend(fontsize=9)

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
    axR.set_ylabel(f"Fraction of units firing > {RATE_HZ:g} Hz")
    axR.set_ylim(0, top + step * (len(pairs) + 2))
    axR.set_title(
        f"Fast-spiking density per session\nFriedman p = {p_omnibus:.3g}, pooled chi2 p = {p_chi2:.3g}"
    )
    axR.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = Path("figures/Q7")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "fast_spiking_density_spikes.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
