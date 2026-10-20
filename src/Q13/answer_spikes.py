"""
Question 13
    In which brain area is the dimensionality of neural activity highest? Use
    the segments from stimulus presentation to outcome to answer this question.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader, Session
from src.Q13.answer import (
    MAX_SEGMENT_PCT,
    eig_spectrum,
    explained_variance_curve,
    participation_ratio,
)
from src.stats import compare_paired, sig_stars

# Each stim->outcome segment is cut into this many equal parts of its own
# duration, so every trial contributes the same number of bins.
N_BINS = 10


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)


def segment_edges(session: Session) -> np.ndarray:
    """Time-warped bin edges (s) of every kept stim->outcome segment.

    Trials above the MAX_SEGMENT_PCT percentile of segment duration are
    dropped, as in the LFP version. [n_trials, N_BINS + 1]
    """
    windows = session.trial_data.select("stim_start", "outcome").drop_nulls()
    stim = windows["stim_start"].to_numpy()
    outcome = windows["outcome"].to_numpy()
    keep = outcome - stim <= np.percentile(outcome - stim, MAX_SEGMENT_PCT)
    return np.linspace(stim[keep], outcome[keep], N_BINS + 1, axis=1)


def firing_rates(session: Session, area: int) -> np.ndarray:
    """Firing rate (Hz) of every unit in every bin. [n_units, n_trials * N_BINS]

    Spike counts per bin are divided by the bin's duration, so short and long
    segments are on the same scale.
    """
    edges = segment_edges(session)
    widths = np.diff(edges, axis=1)  # [n_trials, N_BINS]
    rates = []
    for spike_times in session.get_unit_spikes_in(area):
        idx = np.searchsorted(np.sort(spike_times), edges)
        rates.append((np.diff(idx, axis=1) / widths).ravel())
    return np.array(rates)


if __name__ == "__main__":
    dl = Dataloader(list(ALL_SESSIONS))

    pr_per_area: dict[int, list[float]] = defaultdict(list)
    units_per_area: dict[int, list[int]] = defaultdict(list)
    curves_per_area: dict[int, list[np.ndarray]] = defaultdict(list)

    for s in dl:
        cells = []
        for a in AREAS:
            rates = firing_rates(s, a)
            rates = rates[rates.std(axis=1) > 0]  # silent units can't be z-scored
            z = zscore(rates)
            eigenvals = eig_spectrum(z)
            pr = participation_ratio(eigenvals)
            pr_per_area[a].append(pr)
            units_per_area[a].append(z.shape[0])
            curves_per_area[a].append(explained_variance_curve(eigenvals))
            cells.append(f"area {a}: PR={pr:>5.2f} / {z.shape[0]:>2} units")
        print(f"session {s.id:>2} ({z.shape[1]} bins)  " + "  ".join(cells))

    # [n_sessions, n_areas]; areas stay paired session-by-session. The raw PR
    # grows with the number of units, so the answer is read from PR / n units,
    # the fraction of the available dimensions in use.
    pr_matrix = np.column_stack([pr_per_area[a] for a in AREAS])
    norm_matrix = pr_matrix / np.column_stack([units_per_area[a] for a in AREAS])

    for name, matrix in (("PR", pr_matrix), ("PR / n units", norm_matrix)):
        res = compare_paired(matrix, AREAS)
        print(f"\n=== {name} of unit firing rates by area ===")
        for j, a in enumerate(AREAS):
            print(
                f"area {a}: mean = {res.mean[j]:.3f} +/- {res.sem[j]:.3f} SEM, "
                f"95% CI [{res.mean[j] - res.ci[j]:.3f}, {res.mean[j] + res.ci[j]:.3f}]"
            )
        print(f"(n = {res.n} sessions with all three areas)")
        print(f"Friedman across areas: chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}")
        print("Post-hoc Wilcoxon (Holm-corrected):")
        for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
            print(
                f"  area {AREAS[i]} vs area {AREAS[j]}: "
                f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
            )
    # res is now the normalised comparison, which answers the question.
    print(
        f"\nHighest neural dimensionality (spikes, PR / n units): {res.describe_winner('max')}"
    )

    # Left: PR per area, one dot per session. Middle: PR against the number of
    # units, since the PR can be at most the unit count (dashed line). Right:
    # PR / n units, the fraction of the available dimensions in use.
    fig, (axL, axR, axN) = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(AREAS))
    rng = np.random.default_rng(0)
    for j, a in enumerate(AREAS):
        pr = np.array(pr_per_area[a])
        axL.bar(x[j], pr.mean(), color=colors[j], alpha=0.85, zorder=2)
        jitter = rng.uniform(-0.12, 0.12, pr.size)
        axL.scatter(x[j] + jitter, pr, color="k", s=14, alpha=0.4, zorder=3)
        axR.scatter(units_per_area[a], pr, color=colors[j], s=24, label=f"Area {a}")
        axN.bar(
            x[j],
            res.mean[j],
            yerr=res.sem[j],
            capsize=6,
            color=colors[j],
            alpha=0.85,
            zorder=2,
        )
        axN.scatter(
            x[j] + jitter, norm_matrix[:, j], color="k", s=14, alpha=0.4, zorder=3
        )
    axL.set_xticks(x)
    axL.set_xticklabels([f"Area {a}" for a in AREAS])
    axL.set_ylabel("Participation ratio")
    axL.set_title("Firing-rate dimensionality, stim -> outcome\n(mean over sessions)")

    n_max = max(max(u) for u in units_per_area.values())
    axR.plot([0, n_max], [0, n_max], color="k", ls="--", lw=1, label="PR = n units")
    axR.set_xlabel("Number of units")
    axR.set_ylabel("Participation ratio")
    axR.set_title("Participation ratio vs number of units\n(one point per session)")
    axR.legend(fontsize=9)

    axN.set_xticks(x)
    axN.set_xticklabels([f"Area {a}" for a in AREAS])
    axN.set_ylabel("Participation ratio / number of units")
    # Significance brackets stacked above the bars, inside the [0, 1] range.
    step = 0.06
    for k, ((i, j), hp) in enumerate(zip(res.pairs, res.holm_p)):
        y = 1.0 + step * (k + 0.5)
        axN.plot(
            [x[i], x[i], x[j], x[j]],
            [y, y + step * 0.2, y + step * 0.2, y],
            c="k",
            lw=1,
        )
        axN.text(
            (x[i] + x[j]) / 2, y + step * 0.2, sig_stars(hp), ha="center", va="bottom"
        )
    axN.set_ylim(0, 1.0 + step * (len(res.pairs) + 1))
    axN.set_title(f"Normalised dimensionality\nFriedman p = {res.p_omnibus:.3g}")

    fig.tight_layout()
    out = Path("figures/Q13")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "dimensionality_spikes.png", dpi=150)
    print(f"\nSaved plot to {out / 'dimensionality_spikes.png'}")

    # Second figure: cumulative variance explained against the fraction of
    # components used (k / n units), so populations of different sizes share
    # one axis. Thin lines are sessions, bold lines the area mean on a common
    # grid. The diagonal is what fully independent units give (every component
    # explains 1/n of the variance); the further a curve bows above it, the
    # more the variance is concentrated in a few shared components. The dotted
    # verticals mark each area's mean PR / n units on the same axis.
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    grid = np.linspace(0, 1, 201)
    for j, a in enumerate(AREAS):
        interp = []
        for cum_pct in curves_per_area[a]:
            xp = np.arange(len(cum_pct) + 1) / len(cum_pct)  # 0, 1/n, ..., 1
            yp = np.concatenate([[0.0], cum_pct])
            ax2.plot(xp, yp, color=colors[j], alpha=0.25, lw=1, zorder=2)
            interp.append(np.interp(grid, xp, yp))
        ax2.plot(
            grid,
            np.mean(interp, axis=0),
            color=colors[j],
            lw=2.5,
            label=f"Area {a} (PR / n = {res.mean[j]:.2f})",
            zorder=3,
        )
        ax2.axvline(res.mean[j], color=colors[j], ls=":", lw=1.5, zorder=1)
    ax2.plot([0, 1], [0, 100], color="k", ls="--", lw=1, label="independent units")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Fraction of components (k / n units)")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_title("Firing-rate variance curves, stim -> outcome\n(per session + area mean)")
    ax2.legend(fontsize=9, loc="lower right")
    fig2.tight_layout()
    fig2.savefig(out / "variance_curves_spikes.png", dpi=150)
    print(f"Saved plot to {out / 'variance_curves_spikes.png'}")
