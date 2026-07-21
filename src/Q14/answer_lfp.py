"""
Question 14
    In which brain area is modularity the lowest? Use the entirety of the
    recording to answer this question.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader
from src.stats import compare_paired, sig_stars
from src.utils import get_lfp, get_modularity

if __name__ == "__main__":
    dl = Dataloader(list(ALL_SESSIONS))

    mod_per_area: dict[int, list[float]] = defaultdict(list)

    for s in dl:
        cells = []
        for area in AREAS:
            lfp = get_lfp(s, area)
            mod = get_modularity(lfp)
            mod_per_area[area].append(mod)
            cells.append(f"area {area}: Q={mod:>6.3f} (ch={lfp.shape[0]:>3})")
        print(f"session {s.id:>2}  " + "  ".join(cells))

    # [n_sessions, n_areas]; areas stay paired session-by-session. compare_paired
    # drops sessions missing an area, then runs the standard Friedman omnibus +
    # Holm-corrected pairwise Wilcoxon comparison (see src/stats.py).
    mods = np.column_stack([mod_per_area[a] for a in AREAS])
    res = compare_paired(mods, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

    print("\n=== LFP correlation-graph modularity by area ===")
    for j, area in enumerate(AREAS):
        print(f"area {area}: mean Q = {mean[j]:.4f} +/- {sem[j]:.4f} SEM")
    print(f"(n = {res.n} sessions with all three areas)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")

    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    lowest = res.winner("min")
    print(f"\nLowest modularity: area {lowest}")

    # Bar plot: mean +/- SEM per area, with per-session points and significance
    # brackets for each pairwise comparison.
    fig, ax = plt.subplots(figsize=(5, 5))
    x = np.arange(len(AREAS))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    # Overlay individual sessions (jittered) to show the spread behind the mean.
    rng = np.random.default_rng(0)
    for j in range(len(AREAS)):
        jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
        ax.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.4, zorder=3)

    # Significance brackets, stacked above the tallest bar.
    top = (mean + sem).max()
    step = top * 0.12
    for k, ((i, j), hp) in enumerate(zip(pairs, holm)):
        y = top + step * (k + 1.2)
        ax.plot(
            [x[i], x[i], x[j], x[j]],
            [y, y + step * 0.2, y + step * 0.2, y],
            c="k",
            lw=1,
        )
        ax.text(
            (x[i] + x[j]) / 2, y + step * 0.2, sig_stars(hp), ha="center", va="bottom"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Area {a}" for a in AREAS])
    ax.set_ylabel("Modularity Q (LFP correlation graph)")
    ax.set_title(f"LFP modularity by area\nFriedman p = {p_omnibus:.3g}")
    ax.set_ylim(0, top + step * (len(pairs) + 2))
    fig.tight_layout()

    out = Path("figures/Q14")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "modularity.png", dpi=150)
    print(f"\nSaved plot to {out / 'modularity.png'}")
