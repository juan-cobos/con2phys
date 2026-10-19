"""
Question 9
    Which brain area has the highest excitation-inhibition ratio (i.e., the
    strongest relative excitation)?

Built on the Q7 waveform classification (``src/Q7/answer_waveform.py``):
broad-spiking units are taken as putative excitatory (pyramidal) cells and
narrow-spiking units as putative inhibitory (fast-spiking) interneurons. The
E/I ratio of an area is then n_broad / n_narrow.

Area 1 has no narrow units at all in many sessions, so the per-session ratio is
often infinite. Two complementary estimates are reported:

* pooled ratio per area (all sessions' counts summed), with a 95% bootstrap CI
  from resampling sessions;
* per-session log ratio with a +0.5 (Haldane) correction on both counts, which
  keeps every session finite for the paired Friedman / Wilcoxon tests.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader
from src.Q7.answer_waveform import (
    MIN_UNITS,
    THRESHOLD_MS,
    THRESHOLD_SWEEP,
    count_by_area,
    is_narrow,
    waveform_durations,
)
from src.stats import compare_paired, sig_stars

PSEUDOCOUNT = 0.5
N_BOOT = 10_000


def pooled_ratio(n_exc: np.ndarray, n_inh: np.ndarray) -> np.ndarray:
    """Per-area E/I from counts summed over sessions. [n_areas]"""
    with np.errstate(divide="ignore"):
        return n_exc.sum(axis=0) / n_inh.sum(axis=0)


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)

    # [n_sessions, n_areas] putative inhibitory (narrow) / excitatory (broad) counts.
    n_inh = np.zeros((len(dl), len(AREAS)), dtype=int)
    n_exc = np.zeros_like(n_inh)
    all_durations, all_areas = [], []

    print(f"=== E/I unit counts (narrow < {THRESHOLD_MS:.2f} ms = inhibitory) ===")
    for i, s in enumerate(dl):
        duration = waveform_durations(s.waveforms)
        all_durations.append(duration)
        all_areas.append(s.brain_areas)
        cells = []
        for j, (area, (k, n)) in enumerate(count_by_area(s, duration).items()):
            n_inh[i, j], n_exc[i, j] = k, n - k
            cells.append(f"area {area}: E {n - k:>3} / I {k:<3}")
        print(f"session {s.id:>2}  " + "  ".join(cells))

    durations = np.concatenate(all_durations)
    areas = np.concatenate(all_areas)
    ok = np.isfinite(durations)

    # Per-session log E/I; sessions with too few units in an area are dropped
    # by compare_paired so every remaining row has all three areas.
    log_ratio = np.log((n_exc + PSEUDOCOUNT) / (n_inh + PSEUDOCOUNT))
    log_ratio = np.where(n_exc + n_inh >= MIN_UNITS, log_ratio, np.nan)
    res = compare_paired(log_ratio, AREAS)

    # Pooled ratio with a session-bootstrap CI.
    pooled = pooled_ratio(n_exc, n_inh)
    rng = np.random.default_rng(0)
    boot = np.array(
        [
            pooled_ratio(n_exc[idx], n_inh[idx])
            for idx in rng.integers(0, len(dl), (N_BOOT, len(dl)))
        ]
    )
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)

    print("\n=== E/I ratio by area ===")
    for j, area in enumerate(AREAS):
        print(
            f"area {area}: pooled E/I = {n_exc[:, j].sum()}/{n_inh[:, j].sum()} = {pooled[j]:.2f} "
            f"[95% CI {ci_lo[j]:.2f}, {ci_hi[j]:.2f}];  "
            f"per-session E/I (geometric mean) = {np.exp(res.mean[j]):.2f}"
        )
    print(
        f"(n = {res.n} sessions with >= {MIN_UNITS} classifiable units in every area; "
        f"pseudocount {PSEUDOCOUNT} on E and I)"
    )
    print(
        f"\nFriedman across areas (per-session log E/I): chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}"
    )
    print("Post-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    print("\nThreshold sensitivity (pooled E/I per area):")
    for thr in THRESHOLD_SWEEP:
        ratios = []
        for a in AREAS:
            narrow = is_narrow(durations[ok & (areas == a)], thr)
            ratios.append((~narrow).sum() / narrow.sum())
        best_thr = AREAS[int(np.argmax(ratios))]
        print(
            f"  < {thr:.2f} ms: "
            + "  ".join(f"area {a}: {r:5.2f}" for a, r in zip(AREAS, ratios))
            + f"   -> area {best_thr}"
        )

    best = res.winner("max")
    print(f"\nHighest excitation-inhibition ratio: area {best}")

    # ---- Figure: pooled ratio with CI + per-session log ratios -------------
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
    axL.set_ylabel("E/I ratio (broad / narrow units)")
    axL.set_title("Pooled E/I ratio\n(95% bootstrap CI over sessions)")

    paired = res.paired
    axR.bar(x, res.mean, yerr=res.sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
    for row, dx in zip(paired, jitter):
        axR.plot(x + dx, row, color="k", lw=0.5, alpha=0.2, zorder=3)
    for j in range(len(AREAS)):
        axR.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.5, zorder=4)

    top = paired.max()
    step = (top - min(paired.min(), 0)) * 0.08
    for k, ((i, j), hp) in enumerate(zip(res.pairs, res.holm_p)):
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
    axR.set_ylabel(f"log E/I per session (+{PSEUDOCOUNT} pseudocount)")
    axR.set_ylim(top=top + step * (len(res.pairs) + 2))
    axR.set_title(f"Per-session E/I\nFriedman p = {res.p_omnibus:.3g}")

    fig.tight_layout()
    out = Path("figures/Q9")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "ei_ratio.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
