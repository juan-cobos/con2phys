"""
Question 2
    Which brain area has the highest broadband LFP power (1-100 Hz) over the
    entire recording?
"""

from collections import defaultdict

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import friedmanchisquare, t, wilcoxon

from src.data import ALL_SESSIONS, AREAS, SAMPLING_RATE, Dataloader
from src.utils import get_lfp

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
            lfp = get_lfp(s, area)
            power = band_power(lfp) if lfp.size else float("nan")
            power_per_area[area].append(power)
            cells.append(f"area {area}: {power:>8.2f} uV^2 (ch={lfp.shape[0]:>3})")
        print(
            f"  session {s.id:>2} [{s.duration_s / 60:>6.1f} min]  " + "  ".join(cells)
        )

    # [n_sessions, n_areas]; areas stay paired session-by-session.
    power = np.column_stack([power_per_area[a] for a in AREAS])
    # Paired tests need all three areas present, so keep complete sessions only.
    paired = power[~np.isnan(power).any(axis=1)]
    n = paired.shape[0]

    mean = paired.mean(axis=0)
    sem = paired.std(axis=0, ddof=1) / np.sqrt(n)
    # 95% CI on the mean via the t distribution (small N, unknown variance).
    ci = t.ppf(0.975, n - 1) * sem

    print(f"\n=== broadband LFP power by brain area across {n} sessions ===")
    for j, area in enumerate(AREAS):
        print(
            f"area {area}: mean = {mean[j]:.2f} +/- {sem[j]:.2f} SEM uV^2, "
            f"95% CI [{mean[j] - ci[j]:.2f}, {mean[j] + ci[j]:.2f}]"
        )
    print(f"(n = {n} sessions with all three areas)")

    # Omnibus test: does broadband power differ across the three areas? Friedman
    # is the non-parametric repeated-measures test (paired, no normality
    # assumption), appropriate for small N and skewed power distributions.
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

    highest = AREAS[int(np.argmax(mean))]
    print(f"\nHighest broadband LFP power over the entire recording: area {highest}")
