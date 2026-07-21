import contextlib
import os
import warnings
from itertools import combinations

import numpy as np
from scipy.signal import decimate
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.tsa.stattools import grangercausalitytests

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader
from src.Q6.answer import directed_connectivity, region_signal
from src.stats import compare_paired, holm_correct, one_sample_wilcoxon, sig_stars

# Granger on raw 500 Hz LFP is expensive, so decimate (anti-alias FIR) before
# fitting. A coarse rate (e.g. 100 Hz, 10 ms/lag) cannot resolve the 1-9 ms peak
# lags cross-correlation found -- the direction ends up floored to a single bin
# and is dominated by noise. Decimate only 2x to 250 Hz so one lag = 4 ms, and
# test 4/8/12 ms histories that bracket those leads. Each lag is fit
# independently, not cumulatively.
DECIM = 2
FS = SAMPLING_RATE // DECIM
LAGS = [1, 2, 3]


def prep(sig: np.ndarray) -> np.ndarray:
    """Decimate a region trace to ``FS`` Hz and z-score it for the VAR fit."""
    d = decimate(sig, DECIM, ftype="fir")
    return (d - d.mean()) / d.std()


def granger_test(cause: np.ndarray, effect: np.ndarray, lags: list = LAGS) -> tuple:
    """Strongest Granger effect of ``cause -> effect`` over the given lags.

    ``grangercausalitytests`` tests whether column 2 Granger-causes column 1, so
    we stack ``[effect, cause]``. Passing an explicit lag list fits each lag's
    model independently. On ~10^5-sample LFP the p-values underflow to 0 in every
    direction, so we rank the direction by *effect size* — the SSR F-statistic,
    which measures how much ``cause``'s past shrinks ``effect``'s residual
    variance — and keep the lag with the largest F. Returns ``(F, p, lag)``.
    """
    data = np.column_stack([effect, cause])
    try:
        with (
            warnings.catch_warnings(),
            contextlib.redirect_stdout(open(os.devnull, "w")),
        ):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=list(lags))
    except InfeasibleTestError:
        return float("nan"), float("nan"), 0
    fp = {lag: res[lag][0]["ssr_ftest"][:2] for lag in res}  # lag -> (F, p)
    lag = max(fp, key=lambda k: fp[k][0])
    f, p = fp[lag]
    return float(f), float(p), lag


if __name__ == "__main__":
    dl = Dataloader(ALL_SESSIONS)
    per_pair_F: dict[tuple, list[tuple]] = {}  # (a, b) -> [(F_a->b, F_b->a), ...]
    xc: dict[tuple, list[float]] = {}  # (a, b) -> cross-correlation peak lags (s)

    for s in dl:
        full = {a: region_signal(getattr(s, f"lfp{a}")) for a in AREAS}
        dec = {a: prep(full[a]) for a in AREAS}
        for a, b in combinations(AREAS, 2):
            f_fwd = granger_test(dec[a], dec[b])[0]  # a -> b
            f_rev = granger_test(dec[b], dec[a])[0]  # b -> a
            per_pair_F.setdefault((a, b), []).append((f_fwd, f_rev))
            # Cross-correlation on the full-rate traces (directed_connectivity
            # uses SAMPLING_RATE internally, so it must see 500 Hz signals).
            _, lag_s = directed_connectivity(full[a], full[b])
            xc.setdefault((a, b), []).append(lag_s)
        print(f"session {s.id} done")

    pairs = sorted(per_pair_F)
    n = len(next(iter(per_pair_F.values())))
    Fpair = {p: np.array(per_pair_F[p]) for p in pairs}  # each [n_sessions, 2]

    # Per-session directionality index DI = (F_a->b - F_b->a)/(F_a->b + F_b->a),
    # in [-1, 1]: >0 favours a->b. Normalised, so it is comparable across
    # sessions of different length. Undirected Granger coupling = the larger of
    # the two directed F's, used to rank pairs the way |peak_r| does in answer.py.
    di = np.column_stack(
        [(F[:, 0] - F[:, 1]) / (F[:, 0] + F[:, 1]) for F in Fpair.values()]
    )
    gstr = np.column_stack([F.max(axis=1) for F in Fpair.values()])

    lag_ms = [lg * 1000 // FS for lg in LAGS]
    print(f"\n=== Granger causality ({FS} Hz, lags {lag_ms} ms; n = {n} sessions) ===")

    # --- Directionality: is each pair's net Granger flow reliable across sessions?
    # One-sample Wilcoxon of DI vs 0 over sessions, Holm-corrected across pairs.
    di_p = holm_correct(one_sample_wilcoxon(di))
    print("\nNet directionality (Wilcoxon DI vs 0, Holm across pairs):")
    for j, (a, b) in enumerate(pairs):
        m = di[:, j].mean()
        sem = di[:, j].std(ddof=1) / np.sqrt(n)
        net = f"{a}->{b}" if m >= 0 else f"{b}->{a}"
        print(
            f"  pair {a}-{b}: DI = {m:+.3f} +/- {sem:.3f} SEM, "
            f"net {net}, p_holm = {di_p[j]:.4g} {sig_stars(di_p[j])}"
        )

    # --- Strength: does Granger coupling magnitude differ across pairs?
    # Friedman + post-hoc Wilcoxon on session-paired F, mirroring answer.py.
    res = compare_paired(gstr, pairs)
    print(
        f"\nGranger coupling strength (max directed F): "
        f"Friedman p = {res.p_omnibus:.4g}"
    )
    for (i, j), hp in zip(res.pairs, res.holm_p):
        print(
            f"  {pairs[i]} (med F={np.median(gstr[:, i]):.0f}) vs "
            f"{pairs[j]} (med F={np.median(gstr[:, j]):.0f}): "
            f"p_holm = {hp:.4g} {sig_stars(hp)}"
        )
    print(f"Strongest Granger coupling: pair {res.winner('max')}")

    # --- Cross-method agreement, both now judged at the session level.
    print("\n=== cross-correlation vs Granger direction (per pair) ===")
    for j, (a, b) in enumerate(pairs):
        # Cross-corr: lag>0 => b leads a; lag<0 => a leads b (answer.py convention).
        mean_lag = float(np.mean(xc[(a, b)]))
        xc_leader, xc_lagger = (b, a) if mean_lag > 0 else (a, b)
        g_leader, g_lagger = (a, b) if di[:, j].mean() >= 0 else (b, a)
        agree = "AGREE" if g_leader == xc_leader else "DIFFER"
        print(
            f"  pair {a}-{b}: xcorr {xc_leader}->{xc_lagger} "
            f"(mean |lag| {abs(mean_lag) * 1e3:.1f} ms) vs "
            f"Granger {g_leader}->{g_lagger} (DI {di[:, j].mean():+.3f})  [{agree}]"
        )
