"""
Question 2
    Which brain area has the highest broadband LFP power (1-100 Hz) over the
    entire recording?
"""

from collections import defaultdict

import numpy as np
from scipy.signal import butter, filtfilt

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader
from src.stats import compare_paired, sig_stars

BROADBAND = (1.0, 100.0)
V_TO_UV = 1e6


def band_power(lfp: np.ndarray) -> float:
    b, a = butter(4, [f / (SAMPLING_RATE / 2) for f in BROADBAND], btype="band")
    filtered = filtfilt(b, a, lfp * V_TO_UV, axis=1)
    per_channel = np.mean(filtered**2, axis=1)  # [num_channels], uV^2
    return float(np.mean(per_channel))


if __name__ == "__main__":
    dl = Dataloader(list(ALL_SESSIONS))
    power_per_area: dict[int, list[float]] = defaultdict(list)

    print(f"Per-session broadband power ({BROADBAND[0]:g}-{BROADBAND[1]:g} Hz)")
    for s in dl:
        cells = []
        for area in AREAS:
            lfp = s.get_lfp_in(area)
            power = band_power(lfp) if lfp.size else float("nan")
            power_per_area[area].append(power)
            cells.append(f"area {area}: {power:>8.2f} uV^2 (ch={lfp.shape[0]:>3})")
        print(
            f"  session {s.id:>2} [{s.duration_s / 60:>6.1f} min]  " + "  ".join(cells)
        )

    # [n_sessions, n_areas]; areas stay paired session-by-session. compare_paired
    # drops sessions missing an area, then runs the standard Friedman omnibus +
    # Holm-corrected pairwise Wilcoxon comparison (see src/stats.py).
    power = np.column_stack([power_per_area[a] for a in AREAS])
    res = compare_paired(power, AREAS)

    print(f"\n=== broadband LFP power by brain area across {res.n} sessions ===")
    for j, area in enumerate(AREAS):
        lo, hi = res.mean[j] - res.ci[j], res.mean[j] + res.ci[j]
        print(
            f"area {area}: mean = {res.mean[j]:.2f} +/- {res.sem[j]:.2f} SEM uV^2, "
            f"95% CI [{lo:.2f}, {hi:.2f}]"
        )
    print(f"(n = {res.n} sessions with all three areas)")

    print(f"\nFriedman across areas: chi2 = {res.chi2:.3f}, p = {res.p_omnibus:.4g}")

    print("\nPost-hoc Wilcoxon (Holm-corrected):")
    for (i, j), rp, hp in zip(res.pairs, res.raw_p, res.holm_p):
        print(
            f"  area {AREAS[i]} vs area {AREAS[j]}: "
            f"p_raw = {rp:.4g}, p_holm = {hp:.4g} {sig_stars(hp)}"
        )

    highest = res.winner("max")
    print(f"\nHighest broadband LFP power over the entire recording: area {highest}")
