"""
Question 8
    In which brain area are spikes most strongly phase-locked to its own LFP
    in the 4-10 Hz range?
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, filtfilt, hilbert

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader
from src.stats import compare_paired, sig_stars

BAND_RANGE = (4.0, 10.0)  # Hz
FILTER_ORDER = 4
# Units with few spikes give a very noisy phase estimate; PPC is unbiased in
# expectation but its variance explodes for small n, so require a floor.
MIN_SPIKES = 50


def band_phase(lfp: np.ndarray) -> np.ndarray:
    """Instantaneous 4-10 Hz phase of one area's LFP. [num_timestamps]

    Channels are averaged into a single region trace first: filtering and
    averaging are both linear, so this is the same signal as filtering every
    channel and then averaging, at a fraction of the cost. The band-pass is
    zero-phase (``filtfilt``), which matters here -- a causal filter would
    rotate every phase by its own group delay and bias the locking angle.

    ``hilbert`` is FFT-based, so the trace is padded to a fast length (a few
    million samples at 500 Hz would otherwise hit a slow prime-ish transform)
    and the analytic signal is cropped back afterwards.
    """
    region = lfp.mean(axis=0)
    b, a = butter(
        FILTER_ORDER, [f / (SAMPLING_RATE / 2) for f in BAND_RANGE], btype="band"
    )
    filtered = filtfilt(b, a, region)
    analytic = hilbert(filtered, N=next_fast_len(filtered.size))[: filtered.size]
    return np.angle(analytic)


def spike_phases(spike_times: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """LFP phase sampled at each spike time.

    The LFP is the shared timeline: spike times (s) are rounded to the nearest
    LFP sample index, so both signals are read on the same 500 Hz grid. Spikes
    falling outside the LFP window are dropped.
    """
    idx = np.round(spike_times * SAMPLING_RATE).astype(int)
    return phase[idx[(idx >= 0) & (idx < phase.size)]]


def ppc(phases: np.ndarray) -> float:
    """Pairwise phase consistency of a set of phases (Vinck et al., 2010).

    Mean cosine distance over all spike pairs, computed in closed form from the
    resultant vector: ``(|sum e^(i*theta)|^2 - n) / (n * (n - 1))``. Unlike the
    phase-locking value (the resultant length itself), PPC is unbiased by spike
    count, which matters when comparing areas whose units fire at very
    different rates. 0 means uniformly distributed phases, 1 perfect locking;
    it can go slightly negative by chance.
    """
    n = phases.size
    if n < MIN_SPIKES:
        return float("nan")
    resultant = np.exp(1j * phases).sum()
    return float((np.abs(resultant) ** 2 - n) / (n * (n - 1)))


def preferred_phase(phases: np.ndarray) -> float:
    """Circular mean of a unit's spike phases: the LFP phase it fires at."""
    return float(np.angle(np.exp(1j * phases).mean()))


if __name__ == "__main__":
    dl = Dataloader(list(ALL_SESSIONS))

    phase_lock_per_area: dict[int, list[float]] = defaultdict(list)
    # Per-unit values pooled over sessions, for the distribution panels.
    unit_ppc_per_area: dict[int, list[float]] = defaultdict(list)
    unit_phase_per_area: dict[int, list[float]] = defaultdict(list)

    print(f"Per-session spike-field locking ({BAND_RANGE[0]:g}-{BAND_RANGE[1]:g} Hz)")
    for s in dl:
        cells = []
        for area in AREAS:
            # Spikes and LFP from the *same* area: this is within-area locking,
            # not cross-area coupling.
            phase = band_phase(s.get_lfp_in(area))

            unit_ppcs = []
            for spike_times in s.get_unit_spikes_in(area):
                phases = spike_phases(spike_times, phase)
                value = ppc(phases)
                if np.isnan(value):  # unit below MIN_SPIKES
                    continue
                unit_ppcs.append(value)
                unit_phase_per_area[area].append(preferred_phase(phases))

            # Session-level locking = mean over units, so every session counts
            # once regardless of how many units it contributed.
            phase_lock = float(np.mean(unit_ppcs)) if unit_ppcs else float("nan")
            phase_lock_per_area[area].append(phase_lock)
            unit_ppc_per_area[area].extend(unit_ppcs)
            cells.append(f"area {area}: PPC={phase_lock:>8.5f} (n={len(unit_ppcs):>3})")
        print(f"  session {s.id:>2}  " + "  ".join(cells))

    # [n_sessions, n_areas]; areas stay paired session-by-session. compare_paired
    # drops sessions missing an area, then runs the standard Friedman omnibus +
    # Holm-corrected pairwise Wilcoxon comparison (see src/stats.py).
    ppc_matrix = np.column_stack([phase_lock_per_area[a] for a in AREAS])
    res = compare_paired(ppc_matrix, AREAS)
    paired, mean, sem = res.paired, res.mean, res.sem

    band = f"{BAND_RANGE[0]:g}-{BAND_RANGE[1]:g} Hz"
    print(f"\n=== spike-LFP phase locking (PPC, {band}) ===")
    for j, area in enumerate(AREAS):
        lo, hi = mean[j] - res.ci[j], mean[j] + res.ci[j]
        print(
            f"area {area}: mean PPC = {mean[j]:.5f} +/- {sem[j]:.5f} SEM, "
            f"95% CI [{lo:.5f}, {hi:.5f}]  ({len(unit_ppc_per_area[area])} units)"
        )
    print(f"(n = {res.n} sessions with all three areas)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}")

    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    strongest = res.winner("max")
    print(f"\nStrongest 4-10 Hz spike-LFP phase locking: area {strongest}")

    # ---- Figure: session-level PPC (with stats) + per-unit preferred phases --
    fig = plt.figure(figsize=(11, 5))
    axL = fig.add_subplot(1, 2, 1)
    axR = fig.add_subplot(1, 2, 2, projection="polar")
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = np.arange(len(AREAS))
    axL.bar(x, mean, yerr=sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    # Overlay individual sessions (jittered) to show the spread behind the mean.
    rng = np.random.default_rng(0)
    for j in range(len(AREAS)):
        jitter = rng.uniform(-0.12, 0.12, paired.shape[0])
        axL.scatter(x[j] + jitter, paired[:, j], color="k", s=14, alpha=0.4, zorder=3)

    # Significance brackets, stacked above the tallest bar.
    top = (mean + sem).max()
    step = top * 0.12
    for k, ((i, j), hp) in enumerate(zip(res.pairs, res.holm_p)):
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

    axL.axhline(0, color="k", lw=0.8, zorder=1)
    axL.set_xticks(x)
    axL.set_xticklabels([f"Area {a}" for a in AREAS])
    axL.set_ylabel("Pairwise phase consistency (session mean over units)")
    axL.set_ylim(min(0, paired.min() * 1.1), top + step * (len(res.pairs) + 2))
    axL.set_title(
        f"Spike-LFP locking, {BAND_RANGE[0]:g}-{BAND_RANGE[1]:g} Hz\n"
        f"Friedman p = {res.p_omnibus:.3g}"
    )

    # Right: where in the cycle each unit fires. A concentrated rose means the
    # area's units share a preferred phase; a flat one means they spread out.
    # Shown as a percentage of the area's units, since areas differ in how many
    # units clear MIN_SPIKES and raw counts would just track that.
    bins = np.linspace(-np.pi, np.pi, 25)
    centers = (bins[:-1] + bins[1:]) / 2
    for j, area in enumerate(AREAS):
        counts, _ = np.histogram(unit_phase_per_area[area], bins=bins)
        pct = counts / counts.sum() * 100
        axR.plot(
            np.append(centers, centers[0]),
            np.append(pct, pct[0]),
            color=colors[j],
            lw=2,
            label=f"Area {area} (n={counts.sum()})",
        )
    axR.set_theta_zero_location("E")
    axR.set_rlabel_position(112)
    axR.set_title(
        "Preferred LFP phase per unit (% of units)\n(0 = peak of the 4-10 Hz cycle)"
    )
    axR.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1), fontsize=8)

    fig.tight_layout()
    out = Path("figures/Q8")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "phase_locking.png", dpi=150)
    print(f"\nSaved plot to {out / 'phase_locking.png'}")
