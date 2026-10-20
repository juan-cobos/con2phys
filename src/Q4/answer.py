"""
Question 4
    In which brain area are pairwise spike train interactions strongest at the 100 ms timescale?
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader
from src.stats import compare_paired, sig_stars
from src.utils import BIN_SIZE, binned_trains

dl = Dataloader(ALL_SESSIONS)


def interaction_strength(trains: np.ndarray) -> float:
    """Mean absolute pairwise correlation between cluster spike trains."""
    if trains.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(trains, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    return float(np.mean(np.abs(corr[iu])))


strength_by_ba: dict[int, list[float]] = defaultdict(list)
units_by_ba: dict[int, list[int]] = defaultdict(list)
for s in dl:
    for area in AREAS:
        trains = binned_trains(s, area, BIN_SIZE)
        strength = interaction_strength(trains)
        strength_by_ba[area].append(strength)
        units_by_ba[area].append(trains.shape[1])
        print(
            f"session {s.id} area {area}: {trains.shape[1]} clusters, "
            f"trains {trains.shape}, "
            f"mean |pairwise r| = {strength:.4f}"
        )

# Stack into a [n_sessions, n_areas] matrix. Every session contributes one
# value per area, so the areas are paired session-by-session.
strength = np.column_stack([strength_by_ba[a] for a in AREAS])
# Drop sessions missing an area, then run the standard Friedman + Holm-Wilcoxon
# paired comparison (see src/stats.py). The plotting below reuses these fields.
res = compare_paired(strength, AREAS)
paired, mean, sem = res.paired, res.mean, res.sem
pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

print("\n=== pairwise interaction strength across sessions ===")
for j, area in enumerate(AREAS):
    units = units_by_ba[area]
    print(
        f"area {area}: mean |r| = {mean[j]:.4f} +/- {sem[j]:.4f} SEM, "
        f"95% CI [{mean[j] - res.ci[j]:.4f}, {mean[j] + res.ci[j]:.4f}], "
        f"units/session {min(units)}-{max(units)} (median {np.median(units):g})"
    )
print(f"(n = {res.n} sessions with all three areas)")

print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")

print("\nPost-hoc Wilcoxon (Holm-corrected):")
for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
    print(
        f"  area {AREAS[i]} vs area {AREAS[j]}: "
        f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
    )

print(f"\nStrongest pairwise interactions at 100 ms: {res.describe_winner('max')}")


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
        [x[i], x[i], x[j], x[j]], [y, y + step * 0.2, y + step * 0.2, y], c="k", lw=1
    )
    ax.text((x[i] + x[j]) / 2, y + step * 0.2, sig_stars(hp), ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels([f"Area {a}" for a in AREAS])
ax.set_ylabel("Mean |pairwise r| (100 ms bins)")
ax.set_title(f"Pairwise spike-train interaction by area\nFriedman p = {p_omnibus:.3g}")
ax.set_ylim(0, top + step * (len(pairs) + 2))
fig.tight_layout()

out = Path("figures/Q4")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "interaction_strength.png", dpi=150)
print(f"\nSaved plot to {out / 'interaction_strength.png'}")
