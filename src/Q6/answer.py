"""
Question 6
    Which brain area pair has the strongest directed functional connectivity?
"""

from itertools import combinations
from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import friedmanchisquare, t, wilcoxon

from src.data import ALL_SESSIONS, SAMPLING_RATE, Dataloader


def sig_label(p: float) -> str:
    """Star notation for a p-value."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


MAX_LAG_S = 1


def region_signal(lfp: np.ndarray) -> np.ndarray:
    """Collapse one area's LFP to a single trace: mean over channels.

    LFP is [num_channels, num_timestamps]; averaging across channels gives a
    [num_timestamps] representative signal for the region.
    """
    return lfp.mean(axis=0)


def directed_connectivity(
    x: np.ndarray, y: np.ndarray, max_lag_s: float = MAX_LAG_S
) -> tuple[float, float]:
    """Peak of the normalized cross-correlation between two region signals.

    z-scores both traces so the zero-lag value equals Pearson r, then finds the
    lag (within +/- ``max_lag_s``) that maximises |cross-correlation|.

    Returns ``(peak_r, lag_s)``. With the convention ``correlate(x, y)``, a
    positive ``lag_s`` means ``y`` leads ``x``; negative means ``x`` leads ``y``.
    """
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    xc = signal.correlate(x, y, mode="full", method="fft") / x.size
    lags = signal.correlation_lags(x.size, y.size, mode="full")

    max_lag = int(max_lag_s * SAMPLING_RATE)
    keep = np.abs(lags) <= max_lag
    xc, lags = xc[keep], lags[keep]

    k = int(np.argmax(np.abs(xc)))
    return float(xc[k]), lags[k] / SAMPLING_RATE


def functional_connectivity(signals: dict[int, np.ndarray]) -> dict[tuple, tuple]:
    """Pairwise directed connectivity between region signals.

    Returns {(area_a, area_b): (peak_r, lag_s)} for each unordered pair.
    """
    fc = {}
    for a, b in combinations(sorted(signals), 2):
        fc[(a, b)] = directed_connectivity(signals[a], signals[b])
    return fc


if __name__ == "__main__":
    sessions = ALL_SESSIONS
    sessions.pop(11)  # remove duplicated lfp
    dl = Dataloader(sessions)

    per_pair: dict[tuple, list[tuple]] = {}
    for s in dl:
        signals = {
            1: region_signal(s.lfp1),
            2: region_signal(s.lfp2),
            3: region_signal(s.lfp3),
        }
        for pair, rl in functional_connectivity(signals).items():
            per_pair.setdefault(pair, []).append(rl)

    pairs = sorted(per_pair)  # (1, 2), (1, 3), (2, 3)
    n = len(next(iter(per_pair.values())))  # sessions per pair (paired design)

    # Connectivity strength = |peak cross-correlation|. The peak lag was chosen
    # by maximal |xc|, so peak_r can be negative; magnitude is what ranks the
    # coupling. Stack into a [n_sessions, n_pairs] matrix so pairs stay paired
    # session-by-session for the repeated-measures tests below.
    strength = np.column_stack([np.abs([r for r, _ in per_pair[p]]) for p in pairs])
    lag = np.column_stack([[lag for _, lag in per_pair[p]] for p in pairs])

    mean = strength.mean(axis=0)
    sem = strength.std(axis=0, ddof=1) / np.sqrt(n)
    # 95% CI on the mean via the t distribution (small N, unknown variance).
    ci = t.ppf(0.975, n - 1) * sem

    # Signed mean lag cancels opposing leads across sessions, so also report
    # the mean |lag| (typical delay magnitude) and the sign breakdown: for pair
    # (a, b), lag > 0 means b leads a, lag < 0 means a leads b, 0 is synchronous.
    abs_lag = np.abs(lag).mean(axis=0)
    b_leads = (lag > 0).sum(axis=0)
    a_leads = (lag < 0).sum(axis=0)
    synced = (lag == 0).sum(axis=0)

    fc = {
        pair: {
            "peak_r": float(mean[j]),
            "sem": float(sem[j]),
            "ci95": (float(mean[j] - ci[j]), float(mean[j] + ci[j])),
            "lag_s": float(lag[:, j].mean()),
            "abs_lag_s": float(abs_lag[j]),
        }
        for j, pair in enumerate(pairs)
    }
    pp(fc)

    print(f"\n=== directed connectivity |peak r| across {n} sessions ===")
    for j, (a, b) in enumerate(pairs):
        # One-sample Wilcoxon: is this pair's coupling above zero at all?
        w_p = wilcoxon(strength[:, j]).pvalue
        print(
            f"pair {(a, b)}: |r| = {mean[j]:.4f} +/- {sem[j]:.4f} SEM, "
            f"95% CI [{mean[j] - ci[j]:.4f}, {mean[j] + ci[j]:.4f}], "
            f"p(>0) = {w_p:.4g}"
        )
        print(
            f"           mean |lag| = {abs_lag[j] * 1e3:.1f} ms; "
            f"leads: {b} leads {a} in {b_leads[j]}/{n}, "
            f"{a} leads {b} in {a_leads[j]}/{n}, "
            f"synced in {synced[j]}/{n}"
        )

    # Omnibus test: does connectivity strength differ across the three pairs?
    # Friedman is the non-parametric repeated-measures test (paired, no
    # normality assumption), matching the small-N, skewed |r| values.
    chi2, p_omnibus = friedmanchisquare(*strength.T)
    print(f"\nFriedman across pairs: chi2 = {chi2:.3f}, p = {p_omnibus:.4g}")

    comparisons = list(combinations(range(len(pairs)), 2))
    raw_p = [wilcoxon(strength[:, i], strength[:, j]).pvalue for i, j in comparisons]
    order = np.argsort(raw_p)
    holm = np.empty(len(comparisons))
    m = len(comparisons)
    for rank, idx in enumerate(order):
        holm[idx] = min(1.0, raw_p[idx] * (m - rank))
    holm = np.maximum.accumulate(holm[order])[np.argsort(order)]  # enforce monotonic
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(comparisons, raw_p, holm):
        sig = "*" if hp < 0.05 else "ns"
        print(f"  {pairs[i]} vs {pairs[j]}: p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig}")

    best = pairs[int(np.argmax(mean))]
    print(f"\nStrongest directed functional connectivity: pair {best}")

    # ---- Figure: strength (with stats) + directed-flow schematic ----------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Left: mean |r| +/- SEM per pair, per-session points, sig brackets.
    x = np.arange(len(pairs))
    axL.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    for j in range(len(pairs)):
        jitter = rng.uniform(-0.12, 0.12, n)
        axL.scatter(x[j] + jitter, strength[:, j], color="k", s=14, alpha=0.4, zorder=3)

    top = (mean + sem).max()
    step = top * 0.12
    for k, ((i, j), hp) in enumerate(zip(comparisons, holm)):
        y = top + step * (k + 1.2)
        axL.plot(
            [x[i], x[i], x[j], x[j]],
            [y, y + step * 0.2, y + step * 0.2, y],
            c="k",
            lw=1,
        )
        axL.text(
            (x[i] + x[j]) / 2, y + step * 0.2, sig_label(hp), ha="center", va="bottom"
        )

    axL.set_xticks(x)
    axL.set_xticklabels([f"{a}–{b}" for a, b in pairs])
    axL.set_xlabel("Area pair")
    axL.set_ylabel("|peak cross-correlation|")
    axL.set_ylim(0, top + step * (len(comparisons) + 2))
    axL.set_title(f"Connectivity strength\nFriedman p = {p_omnibus:.3g}")

    # Right: directed-flow schematic. Nodes laid out left-to-right in the
    # propagation order; each edge points leader -> lagger, labelled with the
    # mean |lag| and how many sessions show that lead.
    node_pos = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}
    node_color = {1: colors[0], 2: colors[1], 3: colors[2]}
    for area, (px, py) in node_pos.items():
        axR.scatter(px, py, s=2200, color=node_color[area], alpha=0.9, zorder=3)
        axR.text(
            px,
            py,
            f"Area {area}",
            ha="center",
            va="center",
            color="w",
            fontweight="bold",
            zorder=4,
        )

    for j, (a, b) in enumerate(pairs):
        # Leader = side that leads in more sessions (lower index leads here).
        leader, lagger = (a, b) if a_leads[j] >= b_leads[j] else (b, a)
        lead_ct = max(a_leads[j], b_leads[j])
        (sx, sy), (tx, ty) = node_pos[leader], node_pos[lagger]
        arc = 0.35 if abs(a - b) == 2 else 0.0  # bow the 1-3 edge below node 2
        axR.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            zorder=2,
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.5 + 3 * mean[j],  # thicker = stronger coupling
                color="0.35",
                shrinkA=26,
                shrinkB=26,
                connectionstyle=f"arc3,rad={arc}",
            ),
        )
        mx, my = (sx + tx) / 2, (sy + ty) / 2 + (-0.62 if arc else 0.16)
        axR.text(
            mx,
            my,
            f"{abs_lag[j] * 1e3:.1f} ms\n{lead_ct}/{n} sess.",
            ha="center",
            va="center",
            fontsize=9,
        )

    axR.set_xlim(-0.6, 2.6)
    axR.set_ylim(-1.05, 0.55)
    axR.axis("off")
    axR.set_title("Directed flow  (arrow = leader → lagger,\nwidth ∝ |r|)")

    fig.tight_layout()
    out = Path("figures/Q6")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "directed_connectivity.png", dpi=150)
    print(f"\nSaved plot to {out / 'directed_connectivity.png'}")
