"""
Question 12
    During which trial segment is variable C best decoded?

Linear-SVM / cross-validation variant of ``answer.py``: the same per-segment
question, decoded with the pipeline used in Q11 (per-channel binned LFP,
standardised linear SVM, stratified k-fold balanced accuracy) instead of a
single train/valid/test split and a tuned logistic regression.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader, Session
from src.stats import compare_paired, sig_stars

SEED = 43
events = ("trial_start", "stim_start", "outcome", "trial_end")
trial_segments = (
    ("trial_start", "stim_start"),
    ("stim_start", "outcome"),
    ("outcome", "trial_end"),
)
segment_names = tuple(f"{a}->{b}" for a, b in trial_segments)
target_variable = "variable_C"
output_size = 3  # 0, 1, 2

N_BINS = 10
MAX_SEGMENT_PCT = 95
N_FOLDS = 5
MIN_CLASS_TRIALS = N_FOLDS
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


def segment_windows(session: Session) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-index bounds ``[lo, hi)`` of every segment of every trial, each
    [n_trials, n_segments], plus a boolean ``keep`` of the same shape."""

    df = session.trial_data
    labels = df[target_variable].to_numpy().astype(int)
    t = np.arange(session.lfp1.shape[1]) / SAMPLING_RATE
    times = {e: df[e].to_numpy().astype(float) for e in events}
    idx = {e: np.searchsorted(t, v) for e, v in times.items()}

    lo = np.column_stack([idx[a] for a, _ in trial_segments])
    hi = np.column_stack([idx[b] for _, b in trial_segments])
    finite = np.column_stack(
        [np.isfinite(times[a]) & np.isfinite(times[b]) for a, b in trial_segments]
    )
    duration = hi - lo
    keep = finite & (duration >= N_BINS) & (lo >= 0) & (hi <= t.size)

    for j in range(len(trial_segments)):
        for label in np.unique(labels[keep[:, j]]):
            in_class = keep[:, j] & (labels == label)
            cutoff = np.percentile(duration[in_class, j], MAX_SEGMENT_PCT)
            keep[in_class, j] = duration[in_class, j] <= cutoff
        for label in np.unique(labels):
            in_class = keep[:, j] & (labels == label)
            if in_class.sum() < MIN_CLASS_TRIALS:
                keep[in_class, j] = False
    return lo, hi, keep


def build_features(session: Session) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{segment name: (X, y)}`` for one session.

    ``X`` is [n_trials, n_channels_all_areas * N_BINS]: every LFP channel of
    every area, averaged within each of the N_BINS parts of the window.
    Channels are kept separate (no area averaging), as in Q11.
    """
    lo, hi, keep = segment_windows(session)
    labels = session.trial_data[target_variable].to_numpy().astype(int)
    lfps = [session.get_lfp_in(area) for area in AREAS]
    n_channels = sum(lfp.shape[0] for lfp in lfps)

    out = {}
    for j, name in enumerate(segment_names):
        trials = np.flatnonzero(keep[:, j])
        x = np.empty((trials.size, n_channels, N_BINS))
        for i, trial in enumerate(trials):
            window = np.concatenate(
                [lfp[:, lo[trial, j] : hi[trial, j]] for lfp in lfps]
            )
            parts = np.array_split(window, N_BINS, axis=1)
            x[i] = np.column_stack([part.mean(axis=1) for part in parts])
        out[name] = (x.reshape(trials.size, -1), labels[trials])
    return out


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)
    n_seg = len(segment_names)

    print(
        f"=== decoding {target_variable} per segment: {N_FOLDS}-fold CV balanced "
        f"accuracy, linear SVM on all channels x {N_BINS} time-warped bins ==="
    )
    # Mean CV score per (session, segment): [n_sessions, n_segments]. Folds of
    # the same session are not independent, so they are averaged, and the
    # session is the unit of the paired comparison below.
    scores = np.full((len(dl), n_seg), np.nan)
    chance = np.full((len(dl), n_seg), np.nan)  # 1 / n_classes decoded
    counts = np.zeros(output_size, dtype=int)

    for i, s in enumerate(dl):
        feats = build_features(s)
        # One CV split seed per session, shared by the three segments so they
        # are scored on comparable partitions.
        rs = int(np.random.default_rng([SEED, s.id]).integers(2**31))
        cells = []
        for j, name in enumerate(segment_names):
            x, y = feats[name]
            classes = np.unique(y)
            if classes.size < 2:
                cells.append(f"{name}: n/a (1 class)")
                continue
            scores[i, j] = decode(x, y, N_FOLDS, rs).mean()
            chance[i, j] = 1 / classes.size
            if j == 0:
                counts += np.bincount(y, minlength=output_size)
            n_dropped = y.size and len(s.trial_data) - y.size
            cells.append(f"{name}: {scores[i, j]:.3f} (n={y.size}, -{n_dropped})")
        print(f"session {s.id:>2}  " + "  ".join(f"{c:<38}" for c in cells))

    print(
        f"\nlabel counts pooled over sessions ({target_variable}, first segment): {counts.tolist()}"
    )

    # [n_sessions, n_segments]; segments stay paired session-by-session, which
    # is what makes them comparable at all -- the absolute level depends on the
    # session's own class balance and trial count.
    res = compare_paired(scores, segment_names)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

    print("\n=== balanced accuracy by trial segment ===")
    for j, name in enumerate(segment_names):
        print(
            f"{name:<24} {mean[j]:.3f} +/- {sem[j]:.3f} SEM, "
            f"95% CI [{mean[j] - res.ci[j]:.3f}, {mean[j] + res.ci[j]:.3f}]   "
            f"(chance {np.nanmean(chance[:, j]):.3f})"
        )
    print(f"(n = {res.n} sessions)")

    print(f"\nFriedman across segments: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
        print(
            f"  {segment_names[i]} vs {segment_names[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    best = res.winner("max")
    print(f"\nVariable C is best decoded during: {best}")

    # Bar plot: mean +/- SEM per segment, per-session points joined across
    # segments, significance brackets, and the mean chance level.
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(n_seg)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
    for row, dx in zip(paired, jitter):
        ax.plot(x + dx, row, color="k", lw=0.5, alpha=0.2, zorder=3)
    for j in range(n_seg):
        ax.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.5, zorder=4)

    chance_level = float(np.nanmean(chance))
    ax.axhline(chance_level, color="k", ls="--", lw=1, zorder=1)
    ax.text(x[-1] + 0.45, chance_level, " chance", va="bottom", ha="right", fontsize=9)

    top = max((mean + sem).max(), paired.max())
    step = (top - chance_level) * 0.12
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
    ax.set_xticklabels([n.replace("->", "\n->") for n in segment_names])
    ax.set_ylabel(f"Balanced accuracy, {target_variable} ({N_FOLDS}-fold CV)")
    ax.set_title(
        f"Decoding by trial segment (linear SVM)\nFriedman p = {p_omnibus:.3g}"
    )
    ax.set_ylim(
        min(chance_level - step, paired.min() - step), top + step * (len(pairs) + 2)
    )
    fig.tight_layout()

    out = Path("figures/Q12")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "decoding_by_segment_svm.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
