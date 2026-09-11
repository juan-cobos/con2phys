"""
Question 11
    Which brain area contains most information about variable A? Exclude
    inter-trial intervals. The specific trial segment on which to base the
    analysis is not given because it cannot be determined a priori for
    variable A.

Spike-count variant of ``answer.py``: the decoder sees each area's single
units instead of its LFP channels. Trial windows, binning scheme, decoder and
statistics are imported from ``answer.py`` so the two feature sets are compared
on identical trials and folds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader, Session
from src.Q11.answer import (
    CHANCE,
    N_BINS,
    N_FOLDS,
    SEED,
    decode,
    segment_names,
    target_variable,
    trial_windows,
)
from src.stats import compare_paired, one_sample_wilcoxon, sig_stars


def area_spikes(session: Session, area: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Spike times (s) of every unit in ``area`` and, per spike, the unit's
    index within the area; plus the number of units.

    Unit index follows ``cluster_ids[area_mask(area)]`` order, i.e. the same
    order as ``get_unit_spikes_in(area)`` and the waveform rows. Spike times
    are sorted (as in ``session.spikes``), which lets each trial window be cut
    out with ``searchsorted`` rather than a scan over millions of spikes.
    """
    clusters = session.cluster_ids[session.area_mask(area)]
    in_area = np.isin(session.clusters, clusters)
    times = session.spikes[in_area]
    # cluster id -> position in ``clusters`` (ids are unique within a session)
    order = np.argsort(clusters)
    unit = order[np.searchsorted(clusters[order], session.clusters[in_area])]
    return times, unit, clusters.size


def build_features(session: Session) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Binned firing rate of every unit, per area, plus the labels.

    Returns ``{area: [n_trials, n_segments, n_units, N_BINS]}`` (Hz) and the
    [n_trials] ``variable_A`` labels, for the same trials and windows as the
    LFP version. Each window is cut into N_BINS equal parts and the spikes of
    every unit are counted per part with one ``histogram2d`` call (time x
    unit), then divided by the part's duration so the time-warped
    stim -> outcome bins do not encode reaction time through raw counts.
    """
    lo, hi, keep = trial_windows(session)
    labels = session.trial_data[target_variable].to_numpy().astype(int)[keep]
    # window bounds in seconds, [n_trials, n_segments]
    t_lo, t_hi = lo[keep] / SAMPLING_RATE, hi[keep] / SAMPLING_RATE

    out = {}
    for area in AREAS:
        times, unit, n_units = area_spikes(session, area)
        unit_edges = np.arange(n_units + 1) - 0.5
        x = np.empty((t_lo.shape[0], len(segment_names), n_units, N_BINS))
        for i in range(t_lo.shape[0]):
            for j in range(len(segment_names)):
                edges = np.linspace(t_lo[i, j], t_hi[i, j], N_BINS + 1)
                a, b = np.searchsorted(times, (edges[0], edges[-1]))
                counts, _, _ = np.histogram2d(
                    times[a:b], unit[a:b], bins=[edges, unit_edges]
                )  # [N_BINS, n_units]
                x[i, j] = counts.T / np.diff(edges)
        out[area] = x
    return out, labels


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)
    n_seg = len(segment_names)

    print(
        f"=== decoding {target_variable} per area: {N_FOLDS}-fold CV balanced "
        f"accuracy, linear SVM on {n_seg} segments x {N_BINS} bins x units ==="
    )
    # Mean CV score per (session, area): [n_sessions, n_areas], and the same
    # restricted to one segment at a time: [n_sessions, n_areas, n_segments].
    whole = np.full((len(dl), len(AREAS)), np.nan)
    by_segment = np.full((len(dl), len(AREAS), n_seg), np.nan)
    n_units = np.zeros((len(dl), len(AREAS)), dtype=int)

    for i, s in enumerate(dl):
        feats, y = build_features(s)
        # Same CV seed as the LFP version -> identical fold assignment.
        rs = int(np.random.default_rng([SEED, s.id]).integers(2**31))
        cells = []
        for j, area in enumerate(AREAS):
            x = feats[area]
            n_units[i, j] = x.shape[2]
            whole[i, j] = decode(x.reshape(y.size, -1), y, N_FOLDS, rs).mean()
            for k in range(n_seg):
                x_k = x[:, k].reshape(y.size, -1)
                by_segment[i, j, k] = decode(x_k, y, N_FOLDS, rs).mean()
            cells.append(f"area {area}: {whole[i, j]:.3f} ({n_units[i, j]:>2} units)")
        print(f"session {s.id:>2}  " + "  ".join(cells) + f"   (trials {y.size})")

    res = compare_paired(whole, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem
    pairs, holm, p_omnibus = res.pairs, res.holm_p, res.p_omnibus
    p_chance = one_sample_wilcoxon(paired - CHANCE)

    print(f"\n=== {target_variable} decoding by area (whole trial, spike counts) ===")
    for j, area in enumerate(AREAS):
        above = (paired[:, j] > CHANCE).sum()
        print(
            f"area {area}: balanced accuracy = {mean[j]:.3f} +/- {sem[j]:.3f} SEM, "
            f"95% CI [{mean[j] - res.ci[j]:.3f}, {mean[j] + res.ci[j]:.3f}], "
            f"> chance in {above}/{res.n} sessions, p(vs {CHANCE}) = {p_chance[j]:.4g}, "
            f"units/session {n_units[:, j].min()}-{n_units[:, j].max()}"
        )
    print(f"(n = {res.n} sessions)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {p_omnibus:.4g}")
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, res.raw_p, holm):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    best = res.winner("max")
    print(
        f"\nBrain area with most information about {target_variable} (spikes): area {best}"
    )

    seg_mean = np.nanmean(by_segment, axis=0)
    seg_sem = np.nanstd(by_segment, axis=0, ddof=1) / np.sqrt(len(dl))
    print("\n=== by segment (single-segment decoders, mean +/- SEM) ===")
    print(f"{'':8}" + "".join(f"{name:>26}" for name in segment_names))
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

    # ---- Figure: same layout as the LFP version ------------------------------
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    colors = ["#4C72B0", "#DD8452", "#55A868"]

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
    axL.set_title(
        f"Whole-trial decoding by area, spike counts\nFriedman p = {p_omnibus:.3g}"
    )

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
    save_path = out / "decoding_by_area_spikes.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
