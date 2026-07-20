"""
Question 4
    In which brain area are pairwise spike train interactions strongest at the 100 ms timescale?
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from src.data import ALL_SESSIONS, AREAS, Dataloader

dl = Dataloader(ALL_SESSIONS)


BIN = 0.100  # 100 ms


def binned_trains(
    spikes: np.ndarray,
    clusters: np.ndarray,
    area_clusters: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Spike-count matrix [n_bins, n_clusters] for one brain area."""
    trains = np.empty((len(bin_edges) - 1, len(area_clusters)))
    for j, c in enumerate(area_clusters):
        trains[:, j] = np.histogram(spikes[clusters == c], bins=bin_edges)[0]
    return trains


def interaction_strength(trains: np.ndarray) -> float:
    """Mean absolute pairwise correlation between cluster spike trains."""
    if trains.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(trains, rowvar=False)
    iu = np.triu_indices_from(corr, k=1)
    return float(np.mean(np.abs(corr[iu])))


strength_by_ba: dict[int, list[float]] = defaultdict(list)
for s in dl:
    spike_bin = np.arange(s.spikes.min(), s.spikes.max() + 1, BIN)
    for area in AREAS:
        area_clusters = s.cluster_ids[s.brain_areas == area]
        trains = binned_trains(s.spikes, s.clusters, area_clusters, spike_bin)
        strength = interaction_strength(trains)
        strength_by_ba[area].append(strength)
        print(
            f"session {s.id} area {area}: {trains.shape[1]} clusters, "
            f"trains {trains.shape}, "
            f"mean |pairwise r| = {strength:.4f}"
        )

# Stack into a [n_sessions, n_areas] matrix. Every session contributes one
# value per area, so the areas are paired session-by-session.
strength = np.column_stack([strength_by_ba[a] for a in AREAS])
# A session is usable only if all three areas are non-nan (paired tests need it).
paired = strength[~np.isnan(strength).any(axis=1)]
mean = paired.mean(axis=0)
sem = paired.std(axis=0, ddof=1) / np.sqrt(paired.shape[0])

print("\n=== pairwise interaction strength across sessions ===")
for j, area in enumerate(AREAS):
    print(f"area {area}: mean |r| = {mean[j]:.4f} +/- {sem[j]:.4f} SEM")
print(f"(n = {paired.shape[0]} sessions with all three areas)")

# Omnibus test: does interaction strength differ across the three areas?
# Friedman is the non-parametric repeated-measures test (paired, no normality
# assumption), appropriate given small-N areas and skewed |r| values.
chi2, p_omnibus = friedmanchisquare(*paired.T)
print(f"\nFriedman across areas: chi2 = {chi2:.3f}, p = {p_omnibus:.4g}")

# Post-hoc pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.
pairs = [(0, 1), (0, 2), (1, 2)]
raw_p = [wilcoxon(paired[:, i], paired[:, j]).pvalue for i, j in pairs]
order = np.argsort(raw_p)
holm = np.empty(len(pairs))
m = len(pairs)
for rank, idx in enumerate(order):
    holm[idx] = min(1.0, raw_p[idx] * (m - rank))
holm = np.maximum.accumulate(holm[order])[np.argsort(order)]  # enforce monotonic
print("\nPost-hoc Wilcoxon (Holm-corrected):")
for (i, j), rp, hp in zip(pairs, raw_p, holm):
    sig = "*" if hp < 0.05 else "ns"
    print(
        f"  area {AREAS[i]} vs area {AREAS[j]}: p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig}"
    )

best = AREAS[int(np.argmax(mean))]
print(f"\nStrongest pairwise interactions at 100 ms: area {best}")


def sig_label(p: float) -> str:
    """Star notation for a p-value."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


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
    ax.text((x[i] + x[j]) / 2, y + step * 0.2, sig_label(hp), ha="center", va="bottom")

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
