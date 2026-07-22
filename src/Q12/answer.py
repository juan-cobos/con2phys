"""
Question 12
    During which trial segment is variable C best decoded?
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader, Session
from src.stats import compare_paired, sig_stars

SEED = 43
events = ("trial_start", "stim_start", "outcome", "trial_end")
trial_segments = (
    ("trial_start", "stim_start"),
    ("stim_start", "outcome"),
    ("outcome", "trial_end"),
)
target_variable = "variable_C"

output_size = 3  # 0, 1, 2

n_sessions = len(ALL_SESSIONS)
splits = ("train", "valid", "test")
split_pct = (80, 10, 10)

# Each segment is summarised by this many time bins. Segment durations vary
# from trial to trial (and the three segments have very different typical
# lengths), so bins are time-warped: every segment is cut into N_BINS equal
# parts of its own duration. That keeps the feature vector fixed-size and makes
# bin k mean "the same relative moment of the segment" across trials.
N_BINS = 10
# Drop trials whose segment is longer than this percentile of the session's own
# durations for that segment; a few aborted trials span hundreds of seconds and
# would otherwise be averaged into 10 meaningless bins.
MAX_SEGMENT_PCT = 95

# L2 strengths swept for the logistic-regression decoder. Inverse penalty, so
# small = heavily regularised; the 30 binned-LFP features are correlated within
# an area, which an unpenalised fit would happily overfit.
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
# Weight each class by its inverse frequency in the training loss, matching the
# balanced-accuracy scoring: sessions run as lopsided as 364/94, and without
# this the fit is rewarded for simply following the majority class.
CLASS_WEIGHT = "balanced"


@dataclass
class SegmentDataset:
    """Intrasession decoding dataset for one (session, trial segment) pair.

    ``x[split]`` is [n_trials, n_areas * N_BINS] of binned LFP amplitude and
    ``y[split]`` the matching [n_trials] ``variable_C`` labels. Splits are
    disjoint sets of *trials* from a single session -- the decoder is trained
    and tested within the same recording, so channel counts and LFP scale never
    have to be reconciled across mice.
    """

    session_id: int
    segment: tuple[str, str]
    x: dict[str, np.ndarray] = field(default_factory=dict)
    y: dict[str, np.ndarray] = field(default_factory=dict)
    n_dropped: int = 0

    @property
    def name(self) -> str:
        return f"{self.segment[0]}->{self.segment[1]}"

    @property
    def n_features(self) -> int:
        return self.x["train"].shape[1]

    def baseline(self, split: str = "test") -> float:
        """Majority-class accuracy on ``split``, judged from the train labels.

        Class balance is uneven and session-specific (one session is 364/94),
        so 1/output_size is not the chance level to beat.
        """
        if self.y[split].size == 0:
            return float("nan")
        counts = np.bincount(self.y["train"], minlength=output_size)
        return float(np.mean(self.y[split] == counts.argmax()))


def area_mean_lfp(session: Session) -> np.ndarray:
    """One LFP trace per area, averaged over that area's channels. [n_areas, T]

    Channel counts differ per area and per session (10 to 35), so averaging
    first gives every session the same feature layout and keeps the decoder's
    input dimension fixed at ``n_areas * N_BINS``.
    """
    return np.stack([session.get_lfp_in(area).mean(axis=0) for area in AREAS])


def segment_windows(
    session: Session, segment: tuple[str, str], n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-index window ``[lo, hi)`` per trial, plus a mask of usable trials.

    Trials are dropped when the segment is empty or inverted (a few trials have
    ``outcome`` after ``trial_end``), when it is longer than MAX_SEGMENT_PCT of
    the session's durations, when it falls outside the recording, or when it is
    too short to fill ``n_bins`` bins with at least one sample each.
    """
    start_col, stop_col = segment
    df = session.trial_data
    start = df[start_col].to_numpy().astype(float)
    stop = df[stop_col].to_numpy().astype(float)
    n_samples = session.get_lfp_in(AREAS[0]).shape[1]

    duration = stop - start
    keep = np.isfinite(duration) & (duration > 0)
    if keep.any():
        cutoff = np.percentile(duration[keep], MAX_SEGMENT_PCT)
        keep &= duration <= cutoff

    safe_start = np.where(np.isfinite(start), start, 0.0)
    safe_stop = np.where(np.isfinite(stop), stop, 0.0)
    lo = np.floor(safe_start * SAMPLING_RATE).astype(int)
    hi = np.ceil(safe_stop * SAMPLING_RATE).astype(int)
    keep &= (lo >= 0) & (hi <= n_samples) & ((hi - lo) >= n_bins)
    return lo, hi, keep


def bin_windows(
    signals: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_bins: int = N_BINS
) -> np.ndarray:
    """Mean signal within each of ``n_bins`` equal parts of every window.

    Returns [n_windows, n_signals, n_bins]. Uses a running sum over the whole
    recording so each bin is two lookups and a subtraction, instead of slicing
    hundreds of windows out of a multi-million-sample array.
    """
    running = np.zeros((signals.shape[0], signals.shape[1] + 1))
    np.cumsum(signals, axis=1, dtype=float, out=running[:, 1:])

    binned = np.empty((lo.size, signals.shape[0], n_bins))
    for i, (a, b) in enumerate(zip(lo, hi)):
        edges = np.linspace(a, b, n_bins + 1).round().astype(int)
        binned[i] = np.diff(running[:, edges], axis=1) / np.diff(edges)
    return binned


def stratified_split(y: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shuffle trials and cut them into disjoint train/valid/test index sets.

    Splitting happens *within* each class so the rare label (``variable_C == 1``
    occurs 1-13 times, and only in 5 sessions) cannot land entirely in one
    split and leave the others without it. Classes too small to reach a whole
    trial in valid/test simply stay in train.
    """
    per_split: dict[str, list[np.ndarray]] = {s: [] for s in splits}
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        # Round the two small splits down; train takes whatever is left over.
        n_valid = int(idx.size * split_pct[1] / 100)
        n_test = int(idx.size * split_pct[2] / 100)
        per_split["valid"].append(idx[:n_valid])
        per_split["test"].append(idx[n_valid : n_valid + n_test])
        per_split["train"].append(idx[n_valid + n_test :])

    out = {}
    for split, chunks in per_split.items():
        merged = np.concatenate(chunks) if chunks else np.empty(0, dtype=int)
        rng.shuffle(merged)  # undo the class-major ordering
        out[split] = merged
    return out


def build_dataset(
    session: Session,
    segment: tuple[str, str],
    lfp: np.ndarray,
    rng: np.random.Generator,
) -> SegmentDataset:
    """Binned-LFP -> ``variable_C`` dataset for one segment of one session."""
    lo, hi, keep = segment_windows(session, segment)
    labels = session.trial_data[target_variable].to_numpy().astype(int)[keep]
    features = bin_windows(lfp, lo[keep], hi[keep]).reshape(keep.sum(), -1)

    ds = SegmentDataset(
        session_id=session.id, segment=segment, n_dropped=int((~keep).sum())
    )
    index = stratified_split(labels, rng)

    # Standardise with the train split's statistics only: valid/test must not
    # contribute to the scaling the decoder sees. Constant features (possible
    # with very few train trials) keep a unit scale instead of dividing by 0.
    mean = features[index["train"]].mean(axis=0)
    std = features[index["train"]].std(axis=0)
    std[std == 0] = 1.0

    for split in splits:
        ds.x[split] = (features[index[split]] - mean) / std
        ds.y[split] = labels[index[split]]
    return ds


def build_all(dl: Dataloader) -> dict[tuple[int, str], SegmentDataset]:
    """Every (session, segment) dataset, keyed by ``(session_id, segment name)``.

    Sessions are streamed one at a time (the Dataloader caches only weakly), and
    the LFP is reduced to area means once per session and reused by all three
    segments, so only the small feature matrices are retained.
    """
    datasets: dict[tuple[int, str], SegmentDataset] = {}
    for s in dl:
        lfp = area_mean_lfp(s)
        # One generator per session: the split of a session must not depend on
        # how many sessions were processed before it.
        rng = np.random.default_rng([SEED, s.id])
        cells = []
        for segment in trial_segments:
            ds = build_dataset(s, segment, lfp, rng)
            datasets[(s.id, ds.name)] = ds
            sizes = "/".join(str(ds.y[split].size) for split in splits)
            cells.append(f"{ds.name}: {sizes} (-{ds.n_dropped})")
        print(f"session {s.id:>2}  " + "  ".join(cells))
    return datasets


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes actually present in ``y_true``.

    Sessions are imbalanced (one is 364/94), so plain accuracy mostly measures
    the majority class. Averaging recall weights every class equally instead.
    Classes absent from ``y_true`` are skipped rather than scored 0, since a
    split that contains no such trial says nothing about that class.
    """
    labels = np.unique(y_true)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in labels]))


@dataclass
class DecodeResult:
    """Held-out decoding performance for one (session, segment) pair."""

    session_id: int
    segment: str
    c: float  # L2 strength picked on the valid split
    valid_score: float
    accuracy: float
    balanced: float
    baseline: float
    n_predicted_1: int  # how often the rare class was guessed on test

    @property
    def above_baseline(self) -> float:
        return self.accuracy - self.baseline


def fit_decoder(ds: SegmentDataset) -> DecodeResult:
    """Fit a multinomial logistic decoder, tuning L2 on the valid split.

    The valid split picks the penalty and the test split is scored once with
    the model chosen that way, so the reported number never saw its own data.
    The model is left fit on train alone -- refitting on train+valid would use
    a scaling (see :func:`build_dataset`) estimated from train only.
    """
    best_score, best_c, best_model = -np.inf, C_GRID[0], None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=1000, class_weight=CLASS_WEIGHT)
        model.fit(ds.x["train"], ds.y["train"])
        score = balanced_accuracy(ds.y["valid"], model.predict(ds.x["valid"]))
        if score > best_score:
            best_score, best_c, best_model = score, c, model

    predicted = best_model.predict(ds.x["test"])
    return DecodeResult(
        session_id=ds.session_id,
        segment=ds.name,
        c=best_c,
        valid_score=best_score,
        accuracy=float(np.mean(predicted == ds.y["test"])),
        balanced=balanced_accuracy(ds.y["test"], predicted),
        baseline=ds.baseline("test"),
        n_predicted_1=int((predicted == 1).sum()),
    )


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)
    segment_names = [f"{a}->{b}" for a, b in trial_segments]

    print(f"=== building {target_variable} datasets ({'/'.join(splits)} trials) ===")
    datasets = build_all(dl)

    print(
        f"\nfeatures per trial: {len(AREAS)} area-mean LFP traces x {N_BINS} "
        f"time-warped bins = {len(AREAS) * N_BINS}"
    )

    counts = np.zeros(output_size, dtype=int)
    for (sid, name), ds in datasets.items():
        if name == segment_names[0]:
            for split in splits:
                counts += np.bincount(ds.y[split], minlength=output_size)
    print(f"label counts pooled over sessions ({target_variable}): {counts.tolist()}")

    print("\n=== decoding (balanced accuracy on held-out test trials) ===")
    results: dict[tuple[int, str], DecodeResult] = {}
    for sid in dl.ids:
        cells = []
        for name in segment_names:
            res = fit_decoder(datasets[(sid, name)])
            results[(sid, name)] = res
            cells.append(f"{res.balanced:.3f} (acc {res.accuracy:.3f}, C={res.c:g})")
        print(f"session {sid:>2}  " + "  ".join(f"{c:<32}" for c in cells))

    # [n_sessions, n_segments]; segments stay paired session-by-session, which is
    # what makes them comparable at all -- the absolute level depends on the
    # session's own class balance and trial count.
    scores = np.array(
        [[results[(sid, name)].balanced for name in segment_names] for sid in dl.ids]
    )
    res = compare_paired(scores, segment_names)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus

    print("\n=== balanced accuracy by trial segment ===")
    for j, name in enumerate(segment_names):
        raw = np.mean([results[(sid, name)].accuracy for sid in dl.ids])
        base = np.mean([results[(sid, name)].baseline for sid in dl.ids])
        print(
            f"{name:<24} balanced = {mean[j]:.3f} +/- {sem[j]:.3f} SEM   "
            f"(accuracy {raw:.3f} vs majority baseline {base:.3f})"
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

    # variable_C == 1 is far too rare to be learnable (a handful of trials in 5
    # sessions, usually none in test). CLASS_WEIGHT gives its ~9 training trials
    # a ~150x weight, so the decoder guesses it far more often than it occurs;
    # both counts are printed so the scores above can be read with that in mind.
    n_class1_test = sum(
        int((datasets[(sid, name)].y["test"] == 1).sum())
        for sid in dl.ids
        for name in segment_names
    )
    n_class1_pred = sum(r.n_predicted_1 for r in results.values())
    print(
        f"(class 1: {n_class1_test} trials in test across all sessions/segments, "
        f"predicted {n_class1_pred} times)"
    )

    # Bar plot: mean +/- SEM per segment, per-session points, significance
    # brackets, and each session's majority baseline as a reference line.
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(segment_names))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    ax.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    for j in range(len(segment_names)):
        jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
        ax.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.4, zorder=3)

    chance = np.mean(
        [
            1 / np.unique(datasets[(sid, name)].y["test"]).size
            for sid in dl.ids
            for name in segment_names
        ]
    )
    ax.axhline(chance, color="k", ls="--", lw=1, zorder=1)
    ax.text(len(segment_names) - 0.4, chance, " chance", va="bottom", fontsize=9)

    # Above the highest session point, not just the highest bar, so the
    # brackets never sit on top of the scatter.
    top = max((mean + sem).max(), paired.max())
    step = top * 0.06
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
    ax.set_ylabel(f"Balanced accuracy, {target_variable} (held-out trials)")
    ax.set_title(f"Decoding by trial segment\nFriedman p = {p_omnibus:.3g}")
    ax.set_ylim(0, top + step * (len(pairs) + 2))
    fig.tight_layout()

    out = Path("figures/Q12")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "decoding_by_segment.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
