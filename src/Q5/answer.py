"""
Question 5
    Which brain area pair has the strongest undirected functional connectivity?
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader
from src.stats import compare_paired, one_sample_wilcoxon, sig_stars

# Welch segment length for coherence: 2048 samples = ~4 s at 500 Hz, giving a
# ~0.24 Hz resolution with hundreds of averaged segments per session.
NPERSEG = 2048
# Band the coherence is averaged over. Below 1 Hz is slow drift shared by every
# channel; above ~45 Hz the LFP carries little power here, so both ends would
# add noise rather than coupling.
COH_BAND = (1.0, 45.0)
FMAX_PLOT = 100.0  # Hz shown in the spectrum panel


def undirected_connectivity(x: np.ndarray, y: np.ndarray) -> float:
    """Zero-lag Pearson correlation between two region signals.

    The undirected counterpart of Q6's peak cross-correlation: no lag is
    searched, so neither region is allowed to lead, and the value is symmetric
    in ``x`` and ``y`` by construction.
    """
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return float((x * y).mean())


def coherence_spectrum(
    x: np.ndarray, y: np.ndarray, nperseg: int = NPERSEG
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude-squared coherence between two region signals."""
    return signal.coherence(x, y, fs=SAMPLING_RATE, nperseg=nperseg)


def functional_connectivity(
    signals: dict[int, np.ndarray],
) -> dict[tuple, tuple[float, np.ndarray, np.ndarray]]:
    """Pairwise undirected connectivity between region signals.

    Returns {(area_a, area_b): (r, freqs, coherence)} for each unordered pair.
    """
    fc = {}
    for a, b in combinations(sorted(signals), 2):
        r = undirected_connectivity(signals[a], signals[b])
        freqs, coh = coherence_spectrum(signals[a], signals[b])
        fc[(a, b)] = (r, freqs, coh)
    return fc


if __name__ == "__main__":
    sessions = ALL_SESSIONS
    dl = Dataloader(sessions)

    per_pair_r: dict[tuple, list[float]] = defaultdict(list)
    per_pair_coh: dict[tuple, list[np.ndarray]] = defaultdict(list)
    freqs = np.array([])
    for s in dl:
        signals = {a: s.get_lfp_in(a).mean(axis=0) for a in AREAS}
        for pair, (r, freqs, coh) in functional_connectivity(signals).items():
            per_pair_r[pair].append(r)
            per_pair_coh[pair].append(coh)

    pairs = sorted(per_pair_r)  # (1, 2), (1, 3), (2, 3)
    n = len(next(iter(per_pair_r.values())))  # sessions per pair (paired design)

    # Connectivity strength = |zero-lag r|. A negative correlation is still a
    # coupling (the regions move together, in antiphase), so magnitude is what
    # ranks the pairs. Stack into [n_sessions, n_pairs] so pairs stay paired
    # session-by-session for the repeated-measures tests below.
    strength = np.column_stack([np.abs(per_pair_r[p]) for p in pairs])
    signed_r = np.column_stack([per_pair_r[p] for p in pairs])

    # Second, independent undirected measure: coherence averaged over COH_BAND.
    band = (freqs >= COH_BAND[0]) & (freqs <= COH_BAND[1])
    coh_stack = {p: np.vstack(per_pair_coh[p]) for p in pairs}  # [n_sessions, n_freqs]
    band_coh = np.column_stack([coh_stack[p][:, band].mean(axis=1) for p in pairs])

    # Standard paired comparison across pairs (nothing is dropped here -- every
    # pair is present in every session by construction). See src/stats.py.
    res = compare_paired(strength, pairs)
    mean, sem, ci = res.mean, res.sem, res.ci
    # One-sample Wilcoxon per pair: is this pair's coupling above zero at all?
    w_gt0 = one_sample_wilcoxon(strength)
    res_coh = compare_paired(band_coh, pairs)

    fc = {
        pair: {
            "r": float(signed_r[:, j].mean()),
            "abs_r": float(mean[j]),
            "sem": float(sem[j]),
            "ci95": (float(mean[j] - ci[j]), float(mean[j] + ci[j])),
            "coherence": float(res_coh.mean[j]),
        }
        for j, pair in enumerate(pairs)
    }
    pp(fc)

    print(f"\n=== undirected connectivity |zero-lag r| across {n} sessions ===")
    for j, (a, b) in enumerate(pairs):
        neg = (signed_r[:, j] < 0).sum()
        print(
            f"pair {(a, b)}: |r| = {mean[j]:.4f} +/- {sem[j]:.4f} SEM, "
            f"95% CI [{mean[j] - ci[j]:.4f}, {mean[j] + ci[j]:.4f}], "
            f"p(>0) = {w_gt0[j]:.4g}"
        )
        print(
            f"           signed r = {signed_r[:, j].mean():+.4f} "
            f"(negative in {neg}/{n} sessions); "
            f"mean coherence ({COH_BAND[0]:g}-{COH_BAND[1]:g} Hz) = "
            f"{res_coh.mean[j]:.4f} +/- {res_coh.sem[j]:.4f} SEM"
        )

    p_omnibus = res.p_omnibus
    print(f"\nFriedman across pairs: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")

    comparisons, holm = res.pairs, res.holm_p
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(comparisons, res.raw_p, holm):
        print(
            f"  {pairs[i]} vs {pairs[j]}: p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    print(
        f"\nSame comparison on band coherence: "
        f"Friedman chi2 = {res_coh.chi2:.3f}, p = {res_coh.p_omnibus:.4g}"
    )
    for (i, j), hp in zip(res_coh.pairs, res_coh.holm_p):
        print(f"  {pairs[i]} vs {pairs[j]}: p_holm = {hp:.4g} {sig_stars(hp)}")

    best, best_coh = res.winner("max"), res_coh.winner("max")
    agree = "AGREE" if best == best_coh else "DIFFER"
    print(f"\nStrongest undirected functional connectivity: pair {best}")
    print(f"  (coherence ranks pair {best_coh} first -- the two measures {agree})")

    # ---- Figure: strength (with stats) + coherence spectra ------------------
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
            (x[i] + x[j]) / 2, y + step * 0.2, sig_stars(hp), ha="center", va="bottom"
        )

    axL.set_xticks(x)
    axL.set_xticklabels([f"{a}–{b}" for a, b in pairs])
    axL.set_xlabel("Area pair")
    axL.set_ylabel("|zero-lag correlation|")
    axL.set_ylim(0, top + step * (len(comparisons) + 2))
    axL.set_title(f"Connectivity strength\nFriedman p = {p_omnibus:.3g}")

    # Right: coherence spectrum per pair, mean across sessions with SEM band --
    # where in frequency the coupling lives, with the averaging band shaded.
    show = freqs <= FMAX_PLOT
    for j, pair in enumerate(pairs):
        curves = coh_stack[pair][:, show]
        m = curves.mean(axis=0)
        se = curves.std(axis=0, ddof=1) / np.sqrt(n)
        axR.plot(freqs[show], m, color=colors[j], lw=1.5, label=f"{pair[0]}–{pair[1]}")
        axR.fill_between(freqs[show], m - se, m + se, color=colors[j], alpha=0.2, lw=0)

    axR.axvspan(*COH_BAND, color="0.85", alpha=0.4, zorder=0)
    axR.set_xlim(0, FMAX_PLOT)
    axR.set_ylim(0, 1)
    axR.set_xlabel("Frequency (Hz)")
    axR.set_ylabel("Magnitude-squared coherence")
    axR.legend(title="Area pair", fontsize=9)
    axR.set_title(
        "Coherence spectra (mean +/- SEM over sessions)\n"
        f"shaded: {COH_BAND[0]:g}-{COH_BAND[1]:g} Hz averaging band"
    )

    fig.tight_layout()
    out = Path("figures/Q5")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "undirected_connectivity.png", dpi=150)
    print(f"\nSaved plot to {out / 'undirected_connectivity.png'}")
