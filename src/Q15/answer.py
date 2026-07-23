"""
Question 15
    In which brain area is the neural signal most complex? Use the segments
    from stimulus presentation to trial end to answer this question.
"""

from collections import defaultdict
from pathlib import Path

import antropy as ant
import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader, Session
from src.stats import compare_paired, sig_stars

SEGMENT = ("stim_start", "trial_end")

# Trials whose segment is shorter than this are dropped: every complexity
# estimate needs a minimum run of samples to be meaningful, and a handful of
# trials have near-zero-length or inverted (end before stim) segments. 250
# samples = 0.5 s at 500 Hz.
MIN_SAMPLES = 250

# Permutation-entropy embedding dimension. order=3 (delay=1, i.e. 2 ms steps at
# 500 Hz) counts the 3! = 6 orderings of consecutive triples; large enough to
# separate structured from noisy LFP, small enough that even a short segment
# fills the 6 patterns.
PERM_ORDER = 3
# Welch window for spectral entropy. A fixed segment length gives every trial
# the same frequency resolution regardless of its duration, so the entropy
# compares like with like across trials that span 1 s to >1000 s.
WELCH_NPERSEG = 256


def perm_complexity(trace: np.ndarray) -> float:
    """Normalised permutation entropy of an LFP trace, in [0, 1].

    Built from the ordinal patterns of the signal, so it is invariant to any
    monotonic rescaling of amplitude -- exactly what is needed to compare areas
    whose LFP is on different scales (different channel counts, gains). 0 is a
    perfectly ordered signal, 1 is maximally irregular.
    """
    return float(ant.perm_entropy(trace, order=PERM_ORDER, delay=1, normalize=True))


def spectral_complexity(trace: np.ndarray) -> float:
    """Normalised spectral entropy of an LFP trace, in [0, 1].

    Flatness of the Welch power spectrum: a signal with power concentrated in a
    few rhythms scores low, one spread evenly across frequencies (white-ish)
    scores near 1. A second, frequency-domain view of complexity to cross-check
    the ordinal :func:`perm_complexity`.
    """
    nperseg = min(trace.size, WELCH_NPERSEG)
    return float(
        ant.spectral_entropy(
            trace, sf=SAMPLING_RATE, method="welch", nperseg=nperseg, normalize=True
        )
    )


# Primary measure decides the answer; the rest are reported as robustness checks.
MEASURES = {
    "permutation entropy": perm_complexity,
    "spectral entropy": spectral_complexity,
}


def area_mean_lfp(session: Session) -> np.ndarray:
    """One LFP trace per area, averaged over that area's channels. [n_areas, T]

    Channel counts differ per area and per session, so collapsing each area to
    its mean trace gives every area a single comparable "neural signal" and
    matches how the rest of the repo reduces an area's LFP (see Q8/Q12).
    """
    return np.stack([session.get_lfp_in(area).mean(axis=0) for area in AREAS])


def window_lfp_by_area(session: Session) -> dict[int, list[np.ndarray]]:
    """LFP restricted to each stim->end segment, split by area.

    Returns ``{area: [trace_per_trial, ...]}`` where each trace is the area's
    channel-averaged LFP over one valid trial's stim->end window. Trials are
    windowed once (same ``lo``/``hi`` for every area) so a trial that is kept in
    one area is kept in all three, keeping the areas paired trial-by-trial.
    """
    lfp = area_mean_lfp(session)
    lo, hi, keep = session.event_bounds(*SEGMENT, min_samples=MIN_SAMPLES)
    kept = np.flatnonzero(keep)
    return {area: [lfp[a, lo[t] : hi[t]] for t in kept] for a, area in enumerate(AREAS)}


def session_complexity(
    windows: dict[int, list[np.ndarray]],
) -> dict[str, dict[int, float]]:
    """Mean complexity per measure and area for one session.

    Complexity is computed per trial and then averaged, so every trial counts
    once and the (length-normalised) measures are never distorted by gluing
    variable-length segments into one trace. Returns
    ``{measure_name: {area: mean_complexity}}``.
    """
    out: dict[str, dict[int, float]] = {name: {} for name in MEASURES}
    for area, traces in windows.items():
        for name, measure in MEASURES.items():
            out[name][area] = float(np.mean([measure(t) for t in traces]))
    return out


if __name__ == "__main__":
    dl = Dataloader(list(ALL_SESSIONS))

    # measure -> area -> [one mean per session]; areas stay paired session-wise.
    per_area: dict[str, dict[int, list[float]]] = {
        name: defaultdict(list) for name in MEASURES
    }

    print(f"Complexity of {SEGMENT[0]}->{SEGMENT[1]} LFP (mean over trials, per area)")
    for s in dl:
        windows = window_lfp_by_area(s)
        comp = session_complexity(windows)
        kept = len(windows[AREAS[0]])
        cells = []
        for area in AREAS:
            for name in MEASURES:
                per_area[name][area].append(comp[name][area])
            pe = comp["permutation entropy"][area]
            se = comp["spectral entropy"][area]
            cells.append(f"area {area}: PE={pe:.4f} SE={se:.4f}")
        print(f"  session {s.id:>2} ({kept:>3} trials)  " + "  ".join(cells))

    # Run the standard paired comparison (Friedman + Holm-Wilcoxon) per measure.
    results = {}
    for name in MEASURES:
        matrix = np.column_stack([per_area[name][a] for a in AREAS])
        results[name] = compare_paired(matrix, AREAS)

    for name, res in results.items():
        print(f"\n=== {name} by area ===")
        for j, area in enumerate(AREAS):
            lo, hi = res.mean[j] - res.ci[j], res.mean[j] + res.ci[j]
            print(
                f"area {area}: mean = {res.mean[j]:.4f} +/- {res.sem[j]:.4f} SEM, "
                f"95% CI [{lo:.4f}, {hi:.4f}]"
            )
        print(f"(n = {res.n} sessions with all three areas)")
        print(f"Friedman across areas: chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}")
        print("Post-hoc Wilcoxon (Holm-corrected):")
        for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
            print(
                f"  area {AREAS[i]} vs area {AREAS[j]}: "
                f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
            )
        print(f"Most complex ({name}): area {res.winner('max')}")

    primary = next(iter(MEASURES))
    print(
        f"\nMost complex neural signal ({primary}): area {results[primary].winner('max')}"
    )

    # ---- Figure: one bar panel per measure, session points + significance ----
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(AREAS))
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(5.5 * len(MEASURES), 5))
    axes = np.atleast_1d(axes)

    for ax, (name, res) in zip(axes, results.items()):
        ax.bar(x, res.mean, yerr=res.sem, capsize=6, color=colors, alpha=0.85, zorder=2)
        rng = np.random.default_rng(0)
        for j in range(len(AREAS)):
            jitter = rng.uniform(-0.12, 0.12, res.paired.shape[0])
            ax.scatter(
                x[j] + jitter, res.paired[:, j], color="k", s=14, alpha=0.4, zorder=3
            )

        top = res.paired.max()
        step = (top - res.paired.min()) * 0.15 or top * 0.02
        for k, ((i, j), hp) in enumerate(zip(res.pairs, res.holm_p)):
            y = top + step * (k + 1.2)
            ax.plot(
                [x[i], x[i], x[j], x[j]],
                [y, y + step * 0.2, y + step * 0.2, y],
                c="k",
                lw=1,
            )
            ax.text(
                (x[i] + x[j]) / 2,
                y + step * 0.2,
                sig_stars(hp),
                ha="center",
                va="bottom",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Area {a}" for a in AREAS])
        ax.set_ylabel(f"{name} (session mean over trials)")
        ax.set_ylim(res.paired.min() - step, top + step * (len(res.pairs) + 2))
        ax.set_title(f"{name}\nFriedman p = {res.p_omnibus:.3g}")

    fig.suptitle(f"Neural complexity by area, {SEGMENT[0]} -> {SEGMENT[1]} segment")
    fig.tight_layout()
    out = Path("figures/Q15")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "complexity_by_area.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
