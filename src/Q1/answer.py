"""
Question 1
    Which brain area has the lowest firing rate over the entire recording?
"""

from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, t, wilcoxon

from src.data import ALL_SESSIONS, AREAS, Dataloader

if __name__ == "__main__":
    sessions = list(ALL_SESSIONS)  # copy: pop() below would mutate the shared list
    dl = Dataloader(sessions)

    # fr_per_area[area] -> one mean firing rate per session. Averaging over
    # units within a session first keeps each session equally weighted,
    # regardless of how many units it contributed.
    fr_per_area: dict[int, list[float]] = defaultdict(list)

    print("Per-session mean firing rate (spikes/s)")
    for s in dl:
        cells = []
        for area in AREAS:
            # Rates are per unit rather than pooled over the area: a pooled
            # count scales with how many units the probe happened to isolate,
            # which is a recording artifact, not a property of the area.
            rates = s.spike_counts[s.area_mask(area)] / s.duration_s
            # An area with no units contributes nan; the paired filter below
            # drops that session rather than biasing the mean.
            mean_rate = float(np.mean(rates)) if rates.size else float("nan")
            fr_per_area[area].append(mean_rate)
            cells.append(f"area {area}: {mean_rate:>8.4f} (n={rates.size:>3})")
        print(
            f"  session {s.id:>2} [{s.duration_s / 60:>6.1f} min]  " + "  ".join(cells)
        )

    # [n_sessions, n_areas]; areas stay paired session-by-session.
    fr = np.column_stack([fr_per_area[a] for a in AREAS])
    # Paired tests need all three areas present, so keep complete sessions only.
    paired = fr[~np.isnan(fr).any(axis=1)]
    n = paired.shape[0]

    mean = paired.mean(axis=0)
    sem = paired.std(axis=0, ddof=1) / np.sqrt(n)
    # 95% CI on the mean via the t distribution (small N, unknown variance).
    ci = t.ppf(0.975, n - 1) * sem

    print(f"\n=== firing rate by brain area across {n} sessions ===")
    for j, area in enumerate(AREAS):
        print(
            f"area {area}: mean = {mean[j]:.4f} +/- {sem[j]:.4f} SEM spikes/s, "
            f"95% CI [{mean[j] - ci[j]:.4f}, {mean[j] + ci[j]:.4f}]"
        )
    print(f"(n = {n} sessions with all three areas)")

    # Omnibus test: does firing rate differ across the three areas? Friedman is
    # the non-parametric repeated-measures test (paired, no normality
    # assumption), appropriate for small N and skewed firing rates.
    chi2, p_omnibus = friedmanchisquare(*paired.T)
    print(f"\nFriedman across areas: chi2 = {chi2:.3f}, p = {p_omnibus:.4g}")

    # Post-hoc pairwise Wilcoxon signed-rank tests with Holm-Bonferroni.
    pairs = [(0, 1), (0, 2), (1, 2)]
    raw_p = [wilcoxon(paired[:, i], paired[:, j]).pvalue for i, j in pairs]
    order = np.argsort(raw_p)
    holm = np.empty(len(pairs))
    m = len(pairs)
    for rank, idx in enumerate(order):
        holm[idx] = min(1.0, raw_p[idx] * (m - rank))
    holm = np.maximum.accumulate(holm[order])[np.argsort(order)]  # enforce monotonic
    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(pairs, raw_p, holm):
        sig = "*" if hp < 0.05 else "ns"
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig}"
        )

    lowest = AREAS[int(np.argmin(mean))]
    print(f"\nLowest firing rate over the entire recording: area {lowest}")
