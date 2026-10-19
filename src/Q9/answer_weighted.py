"""
Question 9
    Which brain area has the highest excitation-inhibition ratio (i.e., the
    strongest relative excitation)?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader
from src.Q7.answer_waveform import (
    MIN_UNITS,
    THRESHOLD_MS,
    THRESHOLD_SWEEP,
    is_narrow,
    waveform_durations,
)
from src.Q9.answer import N_BOOT, pooled_ratio
from src.stats import compare_paired, sig_stars


def summed_rates(
    durations: list[np.ndarray],
    rates: list[np.ndarray],
    areas: list[np.ndarray],
    threshold_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(exc_hz, inh_hz, n_units)``, each [n_sessions, n_areas].

    Summed firing rate of broad (E) and narrow (I) units, plus the number of
    classifiable units, per session and area.
    """
    exc = np.zeros((len(durations), len(AREAS)))
    inh = np.zeros_like(exc)
    n_units = np.zeros_like(exc, dtype=int)
    for i, (dur, rate, ar) in enumerate(zip(durations, rates, areas)):
        for j, area in enumerate(AREAS):
            ok = (ar == area) & np.isfinite(dur)
            narrow = is_narrow(dur[ok], threshold_ms)
            inh[i, j] = rate[ok][narrow].sum()
            exc[i, j] = rate[ok][~narrow].sum()
            n_units[i, j] = ok.sum()
    return exc, inh, n_units


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)

    all_durations, all_rates, all_areas = [], [], []
    for s in dl:
        all_durations.append(waveform_durations(s.waveforms))
        all_rates.append(s.spike_counts / s.duration_s)
        all_areas.append(s.brain_areas)

    exc, inh, n_units = summed_rates(all_durations, all_rates, all_areas, THRESHOLD_MS)

    print(
        f"=== summed firing rate, E / I (narrow < {THRESHOLD_MS:.2f} ms = inhibitory) ==="
    )
    for i, s in enumerate(dl.ids):
        cells = [
            f"area {a}: E {exc[i, j]:6.1f} / I {inh[i, j]:5.1f} Hz"
            for j, a in enumerate(AREAS)
        ]
        print(f"session {s:>2}  " + "  ".join(cells))

    # Per-session E share; sessions with too few units in an area are dropped
    # by compare_paired so every remaining row has all three areas.
    with np.errstate(invalid="ignore"):
        share = np.where(n_units >= MIN_UNITS, exc / (exc + inh), np.nan)
    res = compare_paired(share, AREAS)

    # Pooled ratio with a session-bootstrap CI.
    pooled = pooled_ratio(exc, inh)
    rng = np.random.default_rng(0)
    boot = np.array(
        [
            pooled_ratio(exc[idx], inh[idx])
            for idx in rng.integers(0, len(dl), (N_BOOT, len(dl)))
        ]
    )
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)

    durations = np.concatenate(all_durations)
    rates = np.concatenate(all_rates)
    areas = np.concatenate(all_areas)
    ok = np.isfinite(durations)
    narrow = is_narrow(durations)

    print("\n=== rate-weighted E/I by area ===")
    for j, area in enumerate(AREAS):
        in_area = ok & (areas == area)
        print(
            f"area {area}: pooled E/I = {pooled[j]:.2f} [95% CI {ci_lo[j]:.2f}, {ci_hi[j]:.2f}];  "
            f"per-session E share = {res.mean[j]:.3f} +/- {res.sem[j]:.3f} SEM;  "
            f"median rate E {np.median(rates[in_area & ~narrow]):.2f} Hz, "
            f"I {np.median(rates[in_area & narrow]):.2f} Hz"
        )
    print(
        f"(n = {res.n} sessions with >= {MIN_UNITS} classifiable units in every area)"
    )
    print(
        f"\nFriedman across areas (per-session E share): chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}"
    )
    print("Post-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    print("\nThreshold sensitivity (pooled rate-weighted E/I per area):")
    for thr in THRESHOLD_SWEEP:
        e, i_, _ = summed_rates(all_durations, all_rates, all_areas, thr)
        ratios = pooled_ratio(e, i_)
        best_thr = AREAS[int(np.argmax(ratios))]
        print(
            f"  < {thr:.2f} ms: "
            + "  ".join(f"area {a}: {r:5.2f}" for a, r in zip(AREAS, ratios))
            + f"   -> area {best_thr}"
        )

    best = res.winner("max")
    print(f"\nHighest rate-weighted excitation-inhibition ratio: area {best}")

    # ---- Figure: pooled ratio with CI + per-session E share ----------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(AREAS))

    axL.bar(
        x,
        pooled,
        yerr=[pooled - ci_lo, ci_hi - pooled],
        capsize=6,
        color=colors,
        alpha=0.85,
    )
    axL.set_xticks(x)
    axL.set_xticklabels([f"area {a}" for a in AREAS])
    axL.set_ylabel("E/I ratio (summed rate, broad / narrow units)")
    axL.set_title("Pooled rate-weighted E/I\n(95% bootstrap CI over sessions)")

    paired = res.paired
    axR.bar(x, res.mean, yerr=res.sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
    for row, dx in zip(paired, jitter):
        axR.plot(x + dx, row, color="k", lw=0.5, alpha=0.2, zorder=3)
    for j in range(len(AREAS)):
        axR.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.5, zorder=4)

    top = 1.0
    step = 0.06
    for k, ((i, j), hp) in enumerate(zip(res.pairs, res.holm_p)):
        y = top + step * (k + 0.5)
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
    axR.set_ylabel("E share of output, E / (E + I)")
    axR.set_ylim(0, top + step * (len(res.pairs) + 1))
    axR.set_title(f"Per-session E share\nFriedman p = {res.p_omnibus:.3g}")

    fig.tight_layout()
    out = Path("figures/Q9")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "ei_ratio_weighted.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
