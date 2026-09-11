"""
Question 11
    Which brain area contains most information about variable A? Exclude
    inter-trial intervals. The specific trial segment on which to base the
    analysis is not given because it cannot be determined a priori for
    variable A.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader, Session
from src.stats import compare_paired, one_sample_wilcoxon, sig_stars

SEED = 43
target_variable = "variable_A"
CHANCE = 0.5

events = ("trial_start", "stim_start", "outcome", "trial_end")
segment_names = ("pre-stim", "stim->outcome", "post-outcome")
N_BINS = 10  # bins per segment, so 3 * N_BINS features per area
PRE_S = 0.4
POST_S = 1.0
PRE_SAMPLES = round(PRE_S * SAMPLING_RATE)
POST_SAMPLES = round(POST_S * SAMPLING_RATE)
MAX_SEGMENT_PCT = 95

N_FOLDS = 5
CLASS_WEIGHT = "balanced"
SCORING = "balanced_accuracy"


def decode(X: np.ndarray, y: np.ndarray, n_folds: int, random_state) -> np.ndarray:
    """Cross-validated linear-SVM balanced accuracy, one score per fold.

    The scaler is inside the pipeline, so each fold standardises with its own
    training statistics only.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return cross_val_score(
        make_pipeline(
            StandardScaler(), SVC(kernel="linear", class_weight=CLASS_WEIGHT)
        ),
        X,
        y,
        cv=cv,
        scoring=SCORING,
    )


def trial_windows(session: Session) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-index bounds ``[lo, hi)`` of the three windows of every trial,
    each [n_trials, 3], plus a boolean mask of usable trials.

    Event times (s) are mapped onto the LFP sample grid with ``searchsorted``.
    A trial is kept only if all four events are present and ordered, the
    baseline is at least PRE_S, the post-outcome period at least POST_S (so the
    window never runs into the inter-trial interval), and the reaction-time
    segment has at least N_BINS samples and is within its class's
    MAX_SEGMENT_PCT -- so every retained trial yields a complete feature vector.
    """
    df = session.trial_data
    labels = df[target_variable].to_numpy().astype(int)
    t = np.arange(session.lfp1.shape[1]) / SAMPLING_RATE
    times = {e: df[e].to_numpy().astype(float) for e in events}
    idx = {e: np.searchsorted(t, v) for e, v in times.items()}  # NaN -> t.size

    lo = np.column_stack(
        [idx["stim_start"] - PRE_SAMPLES, idx["stim_start"], idx["outcome"]]
    )
    hi = np.column_stack(
        [idx["stim_start"], idx["outcome"], idx["outcome"] + POST_SAMPLES]
    )
    keep = np.all([np.isfinite(v) for v in times.values()], axis=0)
    keep &= idx["stim_start"] - idx["trial_start"] >= PRE_SAMPLES
    keep &= idx["outcome"] - idx["stim_start"] >= N_BINS
    keep &= idx["trial_end"] - idx["outcome"] >= POST_SAMPLES
    keep &= (lo.min(axis=1) >= 0) & (hi.max(axis=1) <= t.size)

    duration = idx["outcome"] - idx["stim_start"]
    for label in np.unique(labels[keep]):
        in_class = keep & (labels == label)
        cutoff = np.percentile(duration[in_class], MAX_SEGMENT_PCT)
        keep[in_class] = duration[in_class] <= cutoff
    return lo, hi, keep


def build_features(session: Session) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Binned per-channel LFP of every usable trial, per area, plus the labels."""
    lo, hi, keep = trial_windows(session)
    labels = session.trial_data[target_variable].to_numpy().astype(int)[keep]
    n_kept = keep.sum()
    lo, hi = lo[keep], hi[keep]
    out = {}
    for area in AREAS:
        lfp = session.get_lfp_in(area)
        x = np.empty((n_kept, len(segment_names), lfp.shape[0], N_BINS))
        for i in range(n_kept):
            for j in range(len(segment_names)):
                parts = np.array_split(lfp[:, lo[i, j] : hi[i, j]], N_BINS, axis=1)
                x[i, j] = np.column_stack([part.mean(axis=1) for part in parts])
        out[area] = x
    return out, labels


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)
    n_seg = len(segment_names)

    print(
        f"=== decoding {target_variable} per area: {N_FOLDS}-fold CV balanced "
        f"accuracy, linear SVM on {n_seg} segments x {N_BINS} bins x channels ==="
    )

    whole = np.full((len(dl), len(AREAS)), np.nan)
    # Same, restricted to one segment at a time: [n_sessions, n_areas, n_segments].
    by_segment = np.full((len(dl), len(AREAS), n_seg), np.nan)
    n_trials = np.zeros(len(dl), dtype=int)

    for i, s in enumerate(dl):
        feats, y = build_features(s)
        n_trials[i] = y.size
        rs = int(np.random.default_rng([SEED, s.id]).integers(2**31))
        cells = []
        for j, area in enumerate(AREAS):
            # [n_trials, n_segments, n_channels, N_BINS] -> flat feature rows
            x = feats[area]
            whole[i, j] = decode(x.reshape(y.size, -1), y, N_FOLDS, rs).mean()
            for k in range(n_seg):
                x_k = x[:, k].reshape(y.size, -1)
                by_segment[i, j, k] = decode(x_k, y, N_FOLDS, rs).mean()
            cells.append(f"area {area}: {whole[i, j]:.3f}")
        counts = np.bincount(y, minlength=2)
        print(
            f"session {s.id:>2}  "
            + "  ".join(cells)
            + f"   (trials {y.size}, {target_variable}=0/1: {counts[0]}/{counts[1]})"
        )

    # Paired across areas within session (the absolute level depends on the
    # session's trial count and label split, which all three areas share).
    res = compare_paired(whole, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus
    # Does each area carry any information at all? One-sample Wilcoxon of the
    # per-session scores against chance.
    p_chance = one_sample_wilcoxon(paired - CHANCE)

    print(f"\n=== {target_variable} decoding by area (whole trial) ===")
    for j, area in enumerate(AREAS):
        above = (paired[:, j] > CHANCE).sum()
        print(
            f"area {area}: balanced accuracy = {mean[j]:.3f} +/- {sem[j]:.3f} SEM, "
            f"95% CI [{mean[j] - res.ci[j]:.3f}, {mean[j] + res.ci[j]:.3f}], "
            f"> chance in {above}/{res.n} sessions, p(vs {CHANCE}) = {p_chance[j]:.4g}"
        )
    print(f"(n = {res.n} sessions; {n_trials.min()}-{n_trials.max()} trials each)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    best = res.winner("max")
    print(f"\nBrain area with most information about {target_variable}: area {best}")

    # Where in the trial the information sits, per area (single-segment decoders).
    seg_mean = np.nanmean(by_segment, axis=0)
    seg_sem = np.nanstd(by_segment, axis=0, ddof=1) / np.sqrt(len(dl))
    print("\n=== by segment (single-segment decoders, mean +/- SEM) ===")
    header = "".join(f"{name:>26}" for name in segment_names)
    print(f"{'':8}{header}")
    for j, area in enumerate(AREAS):
        row = "".join(
            f"{seg_mean[j, k]:>18.3f} +/- {seg_sem[j, k]:.3f}" for k in range(n_seg)
        )
        print(f"area {area:<3}{row}")
    for j, area in enumerate(AREAS):
        res_seg = compare_paired(by_segment[:, j, :], segment_names)
        print(
            f"area {area}: best segment {res_seg.winner('max')} "
            f"(Friedman across segments p = {res_seg.p_omnibus:.3g})"
        )

    # ---- Figure: whole-trial decoding by area (stats) + segment breakdown ----
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Left: mean +/- SEM per area; one dot per (session, area), joined across
    # areas by thin lines to show the pairing.
    x = np.arange(len(AREAS))
    axL.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
    for row, dx in zip(paired, jitter):
        axL.plot(x + dx, row, color="k", lw=0.5, alpha=0.2, zorder=3)
    for j in range(len(AREAS)):
        axL.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.5, zorder=4)
    axL.axhline(CHANCE, color="k", ls="--", lw=1, zorder=1)
    axL.text(x[-1] + 0.45, CHANCE, " chance", va="bottom", ha="right", fontsize=9)

    top = max((mean + sem).max(), paired.max())
    step = (top - CHANCE) * 0.12
    for k, ((i, j), hp) in enumerate(zip(pairs, holm)):
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
    axL.set_xticklabels(
        [f"area {a}\n{sig_stars(p)} vs chance" for a, p in zip(AREAS, p_chance)]
    )
    axL.set_ylabel(f"Balanced accuracy, {target_variable} ({N_FOLDS}-fold CV)")
    axL.set_ylim(min(CHANCE - step, paired.min() - step), top + step * (len(pairs) + 2))
    axL.set_title(f"Whole-trial decoding by area\nFriedman p = {p_omnibus:.3g}")

    # Right: the same score from each segment alone, grouped by area.
    width = 0.25
    for k, name in enumerate(segment_names):
        xs = x + (k - 1) * width
        axR.bar(
            xs,
            seg_mean[:, k],
            width,
            yerr=seg_sem[:, k],
            capsize=3,
            color=colors,
            alpha=0.35 + 0.3 * k,
            edgecolor="k",
            lw=0.5,
            zorder=2,
            label=name,
        )
        for j in range(len(AREAS)):
            axR.scatter(
                xs[j] + jitter * 0.6,
                by_segment[:, j, k],
                color="k",
                s=8,
                alpha=0.4,
                zorder=3,
            )
    axR.axhline(CHANCE, color="k", ls="--", lw=1, zorder=1)
    axR.set_xticks(x)
    axR.set_xticklabels([f"area {a}" for a in AREAS])
    axR.set_ylim(axL.get_ylim())
    axR.legend(title="segment (alpha: early → late)", fontsize=8, loc="upper right")
    axR.set_title("Single-segment decoders")

    fig.tight_layout()
    out = Path("figures/Q11")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "decoding_by_area.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
