"""
Question 13
    In which brain area is the dimensionality of neural activity highest? Use
    the segments from stimulus presentation to outcome to answer this question.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader
from src.stats import compare_paired, sig_stars
from src.utils import get_stim_lfp, matched_draws

# Reference variance level marked on the cumulative-variance curves.
EXPLAINED_VARIANCE = 95
# Drop trials whose stimulus->outcome window is above this percentile of the
# session's own durations; a few aborted trials per session span ~1000 s and
# would otherwise swamp the concatenated segments.
MAX_SEGMENT_PCT = 95
N_DRAWS = 20
SEED = 0


def eig_spectrum(lfp: np.ndarray) -> np.ndarray:
    """Sorted (descending) positive eigenvalues of the channel covariance."""
    eigenvals = np.sort(np.linalg.eigvalsh(np.cov(lfp)))[::-1]
    return eigenvals[eigenvals > 0]


def participation_ratio(eigenvals: np.ndarray) -> float:
    """Effective dimensionality of an eigenspectrum: (sum l)^2 / sum(l^2).

    A threshold-free summary of the whole curve: ~1 when one component
    dominates, up to n when variance is spread evenly across n components.
    """
    return float(eigenvals.sum() ** 2 / np.square(eigenvals).sum())


def explained_variance_curve(eigenvals: np.ndarray) -> np.ndarray:
    """Cumulative percent variance explained by the first k components."""
    return np.cumsum(eigenvals) / eigenvals.sum() * 100


if __name__ == "__main__":
    dl = Dataloader(list(ALL_SESSIONS))

    pr_per_area: dict[int, list[float]] = defaultdict(list)
    curves_per_area: dict[int, list[np.ndarray]] = defaultdict(list)

    for s in dl:
        lfps = {a: get_stim_lfp(s, a, max_segment_pct=MAX_SEGMENT_PCT) for a in AREAS}
        draws, k = matched_draws(
            {a: lfp.shape[0] for a, lfp in lfps.items()}, N_DRAWS, [SEED, s.id]
        )
        cells = []
        for area in AREAS:
            spectra = [eig_spectrum(lfps[area][idx]) for idx in draws[area]]
            pr = float(np.mean([participation_ratio(e) for e in spectra]))
            pr_per_area[area].append(pr)
            # Every draw has k channels, so the curves line up and can be
            # averaged (truncated to the shortest, in case a draw lost a
            # non-positive eigenvalue).
            curves = [explained_variance_curve(e) for e in spectra]
            n_comp = min(len(c) for c in curves)
            curves_per_area[area].append(np.mean([c[:n_comp] for c in curves], axis=0))
            cells.append(f"area {area}: PR={pr:>5.2f} (ch={lfps[area].shape[0]:>3})")
        print(f"session {s.id:>2} [{k:>2} ch matched]  " + "  ".join(cells))

    # [n_sessions, n_areas]; areas stay paired session-by-session. compare_paired
    # drops sessions missing an area, then runs the standard Friedman omnibus +
    # Holm-corrected pairwise Wilcoxon comparison (see src/stats.py).
    pr_matrix = np.column_stack([pr_per_area[a] for a in AREAS])
    res = compare_paired(pr_matrix, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

    print(
        f"\n=== neural dimensionality (participation ratio, channel-matched, "
        f"mean of {N_DRAWS} draws) by area ==="
    )
    for j, area in enumerate(AREAS):
        print(f"area {area}: mean PR = {mean[j]:.3f} +/- {sem[j]:.3f} SEM")
    print(f"(n = {res.n} sessions with all three areas)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")

    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    highest = res.winner("max")
    print(f"\nHighest neural dimensionality: area {highest}")

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
    ax.set_ylabel("Effective dimensionality (participation ratio)")
    ax.set_title(
        f"LFP dimensionality by area (channel-matched)\nFriedman p = {p_omnibus:.3g}"
    )
    ax.set_ylim(0, top + step * (len(pairs) + 2))
    fig.tight_layout()

    out = Path("figures/Q13")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "dimensionality.png", dpi=150)
    print(f"\nSaved plot to {out / 'dimensionality.png'}")

    # Second figure: number of components needed as a function of the variance
    # explained. Thin lines are individual sessions; the bold line per area is
    # the mean, interpolated onto a common variance grid (sessions have
    # different channel counts, so their raw curves have different lengths).
    # Each curve is anchored at (0%, 0 components). The dashed line marks
    # EXPLAINED_VARIANCE, where a curve's height is exactly that area's dim.
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    grid = np.linspace(0, 100, 201)
    for j, area in enumerate(AREAS):
        interp = []
        for cum_pct in curves_per_area[area]:
            xp = np.concatenate([[0.0], cum_pct])
            yp = np.arange(len(xp))  # 0, 1, ..., n_components
            ax2.plot(xp, yp, color=colors[j], alpha=0.25, lw=1, zorder=2)
            interp.append(np.interp(grid, xp, yp))
        ax2.plot(
            grid,
            np.mean(interp, axis=0),
            color=colors[j],
            lw=2.5,
            label=f"Area {area}",
            zorder=3,
        )
    ax2.axvline(EXPLAINED_VARIANCE, color="k", ls="--", lw=1, zorder=1)
    ax2.text(
        EXPLAINED_VARIANCE - 1,
        ax2.get_ylim()[1],
        f"{EXPLAINED_VARIANCE}% ",
        ha="right",
        va="top",
        fontsize=9,
    )
    ax2.set_xlim(0, 100)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Cumulative variance explained (%)")
    ax2.set_ylabel("Number of components")
    ax2.set_title("Components vs variance explained (per session + area mean)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out / "variance_curves.png", dpi=150)
    print(f"Saved plot to {out / 'variance_curves.png'}")
