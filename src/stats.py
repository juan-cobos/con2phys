from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import friedmanchisquare, t, wilcoxon
from statsmodels.stats.multitest import multipletests


def drop_incomplete(matrix: np.ndarray) -> np.ndarray:
    """Keep only rows (sessions) that are non-nan in every column (group).

    Paired tests need every group present in a session, so a session missing
    any group is dropped rather than biasing that group's mean.
    """
    return matrix[~np.isnan(matrix).any(axis=1)]


def summarize(
    paired: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column mean, SEM, and half-width of the (1 - ``alpha``) t CI.

    The CI uses the t distribution (small N, unknown variance); add/subtract
    the returned half-width from the mean to get the interval.
    """
    n = paired.shape[0]
    mean = paired.mean(axis=0)
    sem = paired.std(axis=0, ddof=1) / np.sqrt(n)
    ci = t.ppf(1 - alpha / 2, n - 1) * sem
    return mean, sem, ci


def holm_correct(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction for a family of p-values."""
    return multipletests(pvalues, method="holm")[1]


def one_sample_wilcoxon(paired: np.ndarray) -> np.ndarray:
    """Per-column Wilcoxon signed-rank p-value against a location of zero.

    Answers "is this group's value different from zero at all?" for each
    column independently (e.g. does a coupling exist), separate from the
    across-group comparison.
    """
    return np.array([wilcoxon(paired[:, j]).pvalue for j in range(paired.shape[1])])


@dataclass
class PairedComparison:
    """Result of comparing several paired groups across sessions.

    ``labels`` are the caller's group identifiers (e.g. ``(1, 2, 3)`` for areas
    or ``[(0, 1), ...]`` for area-pairs); every array below is column-aligned
    with them. ``pairs`` indexes ``labels`` for each post-hoc comparison, so
    ``labels[i]`` vs ``labels[j]`` names the comparison at ``holm_p[k]``.
    """

    labels: tuple
    n: int  # sessions surviving the complete-case filter
    paired: np.ndarray  # [n, n_groups] complete-case matrix
    mean: np.ndarray
    sem: np.ndarray
    ci: np.ndarray  # half-width of the 95% t CI
    chi2: float
    p_omnibus: float
    pairs: list[tuple[int, int]]
    raw_p: np.ndarray
    holm_p: np.ndarray

    def winner(self, which: str = "max"):
        """Label of the group with the largest (``"max"``) or smallest mean."""
        idx = np.argmax(self.mean) if which == "max" else np.argmin(self.mean)
        return self.labels[int(idx)]

    def tied_winners(self, which: str = "max", alpha: float = 0.05) -> list:
        """:meth:`winner` plus every group the post-hoc tests cannot separate
        from it (Holm p >= ``alpha``), winner first.

        The extreme mean only answers "which group is highest/lowest" if it is
        significantly different from every other group; otherwise those
        groups tie with it.
        """
        best = self.winner(which)
        b = self.labels.index(best)
        return [best] + [
            self.labels[j if i == b else i]
            for (i, j), hp in zip(self.pairs, self.holm_p)
            if b in (i, j) and hp >= alpha
        ]

    def describe_winner(self, which: str = "max", alpha: float = 0.05) -> str:
        """``"area 2"``, or ``"area 2 and area 3 (tied; ...)"`` when the
        winner is not significantly different from some other group."""
        tied = self.tied_winners(which, alpha)
        if len(tied) == 1:
            return f"area {tied[0]}"
        names = " and ".join(f"area {g}" for g in sorted(tied))
        word = "highest" if which == "max" else "lowest"
        return (
            f"{names} (tied; area {tied[0]} has the {word} mean but is not "
            f"significantly different from the others)"
        )


def compare_paired(matrix: np.ndarray, labels, alpha: float = 0.05) -> PairedComparison:
    """Run the standard paired across-group comparison.

    Drops incomplete sessions, summarizes each group (mean/SEM/CI), runs the
    Friedman omnibus test across groups, and follows up with all pairwise
    Wilcoxon signed-rank tests under Holm-Bonferroni correction.

    ``matrix`` is ``[n_sessions, n_groups]`` and column-aligned with ``labels``.
    """
    labels = tuple(labels)
    paired = drop_incomplete(matrix)
    n = paired.shape[0]

    mean, sem, ci = summarize(paired, alpha)
    chi2, p_omnibus = friedmanchisquare(*paired.T)

    pairs = list(combinations(range(len(labels)), 2))
    raw_p = np.array([wilcoxon(paired[:, i], paired[:, j]).pvalue for i, j in pairs])
    holm_p = holm_correct(raw_p)

    return PairedComparison(
        labels=labels,
        n=n,
        paired=paired,
        mean=mean,
        sem=sem,
        ci=ci,
        chi2=float(chi2),
        p_omnibus=float(p_omnibus),
        pairs=pairs,
        raw_p=raw_p,
        holm_p=holm_p,
    )


def sig_stars(p: float, alpha: float = 0.05) -> str:
    """Star notation for a p-value: ``***`` <1e-3, ``**`` <1e-2, ``*`` <alpha."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < alpha else "ns"
