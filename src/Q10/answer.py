"""
Question 10
    Which brain area has the shortest intrinsic neuronal timescale during
    baseline activity?
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader, Session
from src.stats import compare_paired, sig_stars

BASELINE = ("trial_start", "stim_start")
# Fixed-length window ending at stim_start. The baseline is short and varies
# trial to trial (median ~0.65 s, minimum ~0.44 s in most sessions), so taking
# the last BASELINE_S seconds before the stimulus gives every trial the same
# number of bins -- required for an across-trial autocorrelation -- while still
# keeping >=98% of the trials in every session.
BASELINE_S = 0.4
MIN_SAMPLES = round(BASELINE_S * SAMPLING_RATE)
BIN_S = 0.025  # 16 bins per trial; lags of 25 ms up to 200 ms
N_BINS = round(BASELINE_S / BIN_S)
# The fit starts at the peak of the autocorrelation rather than at lag 0 (whose
# value is trivially 1) or lag 1: refractoriness and burst structure often push
# the first lags *below* the ones after them, and fitting a decay through that
# dip is what Murray et al. avoid by starting at the first non-decreasing lag.
# The search is capped so at least half the lags always enter the fit.
MAX_START_LAG = N_BINS // 2

# A unit needs enough baseline spikes for its bin-to-bin correlations to mean
# anything; below this the autocorrelation is mostly sampling noise.
MIN_SPIKES = 100
# A fit is rejected when tau runs to either bound: below a fifth of a bin the
# decay is faster than the binning can express, and at the window length the
# curve is flat and the "timescale" is just the fit running away.
TAU_BOUNDS = (BIN_S / 5, BASELINE_S)


def baseline_starts(session: Session) -> np.ndarray:
    """Start time (s) of every usable baseline window in a session.

    Each window is the ``BASELINE_S`` seconds immediately preceding
    ``stim_start``. :meth:`Session.event_bounds` supplies the trial_start ->
    stim_start bounds and drops trials with a missing event, a window outside
    the recording, or a baseline shorter than ``MIN_SAMPLES`` -- so every window
    returned here fits inside its own trial's pre-stimulus period and never
    spills into the previous trial.
    """
    _, hi, keep = session.event_bounds(*BASELINE, min_samples=MIN_SAMPLES)
    return hi[keep] / SAMPLING_RATE - BASELINE_S


def spike_counts(spike_times: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Spike counts of one unit in every baseline window. [n_trials, N_BINS]

    The window of each trial is cut into ``N_BINS`` bins of ``BIN_S`` seconds.
    Counting is done with ``searchsorted`` on the unit's sorted spike train, so
    the whole trial x bin matrix costs one binary search per edge instead of a
    pass over the spikes per bin.
    """
    if spike_times.size == 0:
        return np.zeros((starts.size, N_BINS))
    sorted_times = np.sort(spike_times)
    edges = starts[:, None] + np.arange(N_BINS + 1) * BIN_S
    idx = np.searchsorted(sorted_times, edges.ravel()).reshape(edges.shape)
    return np.diff(idx, axis=1).astype(float)


def baseline_counts(session: Session, area: int) -> list[np.ndarray]:
    """Baseline spike-count matrices of every unit in ``area``.

    Returns one ``[n_trials, N_BINS]`` matrix per unit that clears
    :data:`MIN_SPIKES` inside the baseline windows. All units share the same
    windows, so the areas of a session are compared over identical trials.
    """
    starts = baseline_starts(session)
    counts = []
    for spike_times in session.get_unit_spikes_in(area):
        c = spike_counts(spike_times, starts)
        if c.sum() >= MIN_SPIKES:
            counts.append(c)
    return counts


def autocorrelation(counts: np.ndarray) -> np.ndarray:
    """Across-trial spike-count autocorrelation per time lag (Murray et al., 2014).

    Correlates bin i with bin j *across trials*, then averages every pair with
    the same separation, giving one value per lag 0..N_BINS-1. Correlating
    across trials (not within a single trace) is what makes this a measure of
    how long the area's baseline fluctuations persist, rather than of the
    stimulus-locked response shape. Bins where the unit never fires have no
    variance and are left out of the average.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(counts, rowvar=False)
    lags = np.abs(np.subtract.outer(np.arange(N_BINS), np.arange(N_BINS)))
    return np.array(
        [
            np.nanmean(corr[lags == lag])
            if np.isfinite(corr[lags == lag]).any()
            else np.nan
            for lag in range(N_BINS)
        ]
    )


def decay(lag_s: np.ndarray, a: float, tau: float, b: float) -> np.ndarray:
    """Exponential decay with an offset: ``A * (exp(-lag / tau) + B)``."""
    return a * (np.exp(-lag_s / tau) + b)


def fit_decay(ac: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Fit :func:`decay` to an autocorrelation-vs-lag curve.

    The fit runs from the curve's peak lag (see :data:`MAX_START_LAG`) to the
    longest lag. Returns ``(params, start_lag)`` with ``params`` = ``(A, tau,
    B)``, or ``(None, start_lag)`` when the curve has missing lags, the fit does
    not converge, or ``tau`` escapes :data:`TAU_BOUNDS` -- a flat curve, whose
    timescale this window cannot resolve, ends up rejected rather than
    reported as a spurious number.
    """
    if not np.isfinite(ac[1:]).all():
        return None, 0
    start = int(1 + np.argmax(ac[1 : MAX_START_LAG + 1]))
    lags = np.arange(start, N_BINS)
    y = ac[lags]
    try:
        params, _ = curve_fit(
            decay,
            lags * BIN_S,
            y,
            p0=(max(y[0], 1e-3), BIN_S * 2, 0.5),
            bounds=([0.0, TAU_BOUNDS[0], -1.0], [np.inf, TAU_BOUNDS[1], 1.0]),
            maxfev=10000,
        )
    except RuntimeError:  # no convergence within maxfev
        return None, start
    # curve_fit reports the boundary itself when the optimum lies outside it.
    if not TAU_BOUNDS[0] * 1.001 < params[1] < TAU_BOUNDS[1] * 0.999:
        return None, start
    return params, start


def timescale(ac: np.ndarray) -> float:
    """Intrinsic timescale ``tau`` (s) of an autocorrelation curve, else NaN."""
    params, _ = fit_decay(ac)
    return float("nan") if params is None else float(params[1])


def area_autocorrelation(session: Session, area: int) -> tuple[np.ndarray, int]:
    """Unit-averaged baseline autocorrelation of one area, plus its unit count.

    Autocorrelations are averaged over the area's units *before* fitting: single
    units fire sparsely in a 0.4 s baseline, so their individual curves are
    noisy, while the population-averaged curve is the quantity Murray et al. fit
    per area. The curve is all-NaN if the area has no usable unit.
    """
    counts = baseline_counts(session, area)
    if not counts:
        return np.full(N_BINS, np.nan), 0
    acs = np.vstack([autocorrelation(c) for c in counts])
    return np.nanmean(acs, axis=0), len(counts)


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)

    per_area: dict[int, list[float]] = defaultdict(list)  # area -> one tau per session
    per_area_units: dict[int, list[int]] = defaultdict(list)
    per_area_ac: dict[int, list[np.ndarray]] = defaultdict(list)

    print(
        f"Intrinsic timescale during baseline ({BASELINE[0]}->{BASELINE[1]}, "
        f"last {BASELINE_S:g} s, {BIN_S * 1e3:g} ms bins)"
    )
    for s in dl:
        n_trials = baseline_starts(s).size
        cells = []
        for area in AREAS:
            ac, n_units = area_autocorrelation(s, area)
            tau = timescale(ac)
            per_area[area].append(tau)
            per_area_units[area].append(n_units)
            per_area_ac[area].append(ac)
            cells.append(f"area {area}: tau={tau * 1e3:>6.1f} ms (n={n_units:>3})")
        print(f"  session {s.id:>2} ({n_trials:>3} trials)  " + "  ".join(cells))

    # [n_sessions, n_areas] in ms; areas stay paired session-by-session.
    # compare_paired drops sessions where any area went unfitted, then runs the
    # standard Friedman omnibus + Holm-corrected Wilcoxon (see src/stats.py).
    tau_ms = np.column_stack([per_area[a] for a in AREAS]) * 1e3
    res = compare_paired(tau_ms, AREAS)

    print(
        f"\n=== intrinsic timescale by area (n = {res.n} sessions, all areas fit) ==="
    )
    for j, area in enumerate(AREAS):
        lo, hi = res.mean[j] - res.ci[j], res.mean[j] + res.ci[j]
        fitted = np.isfinite(tau_ms[:, j])
        print(
            f"area {area}: tau = {res.mean[j]:6.1f} +/- {res.sem[j]:.1f} ms SEM, "
            f"95% CI [{lo:.1f}, {hi:.1f}], median {np.median(res.paired[:, j]):6.1f} ms  "
            f"(fit succeeded in {fitted.sum()}/{len(ALL_SESSIONS)} sessions, "
            f"{int(np.mean(per_area_units[area]))} units/session)"
        )

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}")

    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    shortest = res.winner("min")
    print(f"\nShortest intrinsic neuronal timescale during baseline: area {shortest}")

    # Sessions where an area's curve never decayed are dropped above, and they
    # are not missing at random -- a rejected fit means a *long* (unresolvable)
    # timescale, so dropping them can only work against the area being called
    # shortest. Report the rejections so the ranking can be read with that in
    # mind, plus the ranking over every session that did fit, unpaired.
    print("\nUnfitted sessions per area (flat autocorrelation within the window):")
    for j, area in enumerate(AREAS):
        missing = [ALL_SESSIONS[i] for i in np.flatnonzero(~np.isfinite(tau_ms[:, j]))]
        column = tau_ms[np.isfinite(tau_ms[:, j]), j]
        print(
            f"  area {area}: {len(missing)} dropped {missing}; "
            f"median over all fitted sessions = {np.median(column):.1f} ms"
        )

    # ---- Figure: timescale per area + the autocorrelation curves behind it ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Left: mean tau +/- SEM per area, per-session points, significance brackets.
    x = np.arange(len(AREAS))
    axL.bar(x, res.mean, yerr=res.sem, capsize=6, color=colors, alpha=0.85, zorder=2)
    rng = np.random.default_rng(0)
    for j in range(len(AREAS)):
        jitter = rng.uniform(-0.12, 0.12, res.n)
        axL.scatter(
            x[j] + jitter, res.paired[:, j], color="k", s=14, alpha=0.4, zorder=3
        )

    top = max((res.mean + res.sem).max(), res.paired.max())
    step = top * 0.08
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

    axL.set_xticks(x)
    axL.set_xticklabels([f"Area {a}" for a in AREAS])
    axL.set_ylabel("Intrinsic timescale tau (ms)")
    axL.set_ylim(0, top + step * (len(res.pairs) + 2))
    axL.set_title(
        f"Baseline intrinsic timescale (n = {res.n} sessions)\n"
        f"Friedman p = {res.p_omnibus:.3g}"
    )

    # Right: the measurement itself -- each area's autocorrelation averaged over
    # sessions, with the exponential fitted to that grand-average curve. Filled
    # markers are the lags that entered the fit, open ones the leading lags the
    # peak-start rule excluded.
    lags_ms = np.arange(N_BINS) * BIN_S * 1e3
    for j, area in enumerate(AREAS):
        grand = np.nanmean(np.vstack(per_area_ac[area]), axis=0)
        params, start = fit_decay(grand)
        axR.scatter(
            lags_ms[1:start],
            grand[1:start],
            facecolors="none",
            edgecolors=colors[j],
            s=26,
            zorder=3,
        )
        axR.scatter(
            lags_ms[start:], grand[start:], color=colors[j], s=26, zorder=3, label=None
        )
        if params is not None:
            fine = np.linspace(start * BIN_S, (N_BINS - 1) * BIN_S, 200)
            axR.plot(
                fine * 1e3,
                decay(fine, *params),
                color=colors[j],
                lw=1.8,
                label=f"Area {area}: tau = {params[1] * 1e3:.0f} ms",
            )
        else:
            axR.plot([], [], color=colors[j], lw=1.8, label=f"Area {area}: no fit")

    axR.set_xlabel("Lag (ms)")
    axR.set_ylabel("Across-trial spike-count autocorrelation")
    axR.set_xlim(0, lags_ms[-1] + BIN_S * 1e3)
    axR.legend(fontsize=9)
    axR.set_title(
        "Autocorrelation averaged over sessions\n"
        "(open markers: lags excluded by the peak-start rule)"
    )

    fig.tight_layout()
    out = Path("figures/Q10")
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "intrinsic_timescale.png"
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
