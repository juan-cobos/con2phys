"""
Question 1
    Which brain area has the lowest firing rate over the entire recording?
"""

from collections import defaultdict

import numpy as np

from src.data import ALL_SESSIONS, AREAS, Dataloader
from src.stats import compare_paired, sig_stars

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

    # [n_sessions, n_areas]; areas stay paired session-by-session. compare_paired
    # drops sessions missing an area, then runs the standard Friedman omnibus +
    # Holm-corrected pairwise Wilcoxon comparison (see src/stats.py).
    fr = np.column_stack([fr_per_area[a] for a in AREAS])
    res = compare_paired(fr, AREAS)

    print(f"\n=== firing rate by brain area across {res.n} sessions ===")
    for j, area in enumerate(AREAS):
        lo, hi = res.mean[j] - res.ci[j], res.mean[j] + res.ci[j]
        print(
            f"area {area}: mean = {res.mean[j]:.4f} +/- {res.sem[j]:.4f} SEM spikes/s, "
            f"95% CI [{lo:.4f}, {hi:.4f}]"
        )
    print(f"(n = {res.n} sessions with all three areas)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}")

    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    print(f"\nLowest firing rate over the entire recording: area {res.winner('min')}")
