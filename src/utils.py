"""Shared helpers for accessing session data and building correlation graphs."""

import mne
import networkx as nx
import numpy as np

from src.data import SAMPLING_RATE, Session


def get_stim_lfp(
    session: Session, region: int, max_segment_pct: float | None = None
) -> np.ndarray:
    """LFP for ``region`` restricted to each trial's stimulus->outcome window.

    Keeps only the samples between ``stim_start`` and ``outcome`` of every
    trial and concatenates those segments along time, so downstream measures
    reflect stimulus-evoked activity rather than the whole recording. Trials
    missing either timestamp are skipped.

    ``max_segment_pct`` drops trials whose stimulus->outcome window is longer
    than this percentile of the session's own window durations -- a handful of
    aborted/unterminated trials per session have windows of ~1000 s that would
    otherwise dominate the concatenation.

    Returns [num_channels, num_kept_timestamps].
    """
    lfp = session.get_lfp_in(region)
    n = lfp.shape[1]
    windows = session.trial_data.select("stim_start", "outcome").drop_nulls()
    stim = windows["stim_start"].to_numpy()
    outcome = windows["outcome"].to_numpy()

    keep = np.ones(len(stim), dtype=bool)
    if max_segment_pct is not None:
        durations = outcome - stim
        keep = durations <= np.percentile(durations, max_segment_pct)

    mask = np.zeros(n, dtype=bool)
    for lo, hi in zip(
        np.floor(stim[keep] * SAMPLING_RATE).astype(int),
        np.ceil(outcome[keep] * SAMPLING_RATE).astype(int),
    ):
        mask[max(0, lo) : min(n, hi)] = True
    return lfp[:, mask]


def get_raw(session: Session, region: int) -> mne.io.RawArray:
    """Wrap a region's LFP array in an MNE Raw object."""
    lfp = session.get_lfp_in(region)
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(lfp.shape[0])],
        sfreq=SAMPLING_RATE,
        ch_types="eeg",
    )
    return mne.io.RawArray(lfp, info, verbose=False)


def binned_trains(session: Session, area: int, bin_s: float) -> np.ndarray:
    """Spike-count matrix [n_bins, n_units] for one area's clusters.

    Each column is a unit's spike train over the whole recording, histogrammed
    onto shared ``bin_s``-second bins. Silent units (no spikes -> constant,
    zero-variance train) are dropped, as they carry no correlation.
    """
    clusters = session.cluster_ids[session.brain_areas == area]
    edges = np.arange(session.spikes.min(), session.spikes.max() + bin_s, bin_s)
    trains = np.empty((len(edges) - 1, len(clusters)))
    for j, c in enumerate(clusters):
        trains[:, j] = np.histogram(session.spikes[session.clusters == c], bins=edges)[
            0
        ]
    return trains[:, trains.std(axis=0) > 0]


def correlation_graph(signals: np.ndarray) -> nx.Graph:
    """Weighted graph over the rows of ``signals``; edge weight = |Pearson r|.

    ``signals`` is [n_variables, n_samples] (e.g. LFP channels or unit spike
    trains). Every variable is a node and every pair is connected with weight
    equal to the absolute correlation of their two rows (self-loops removed).
    The absolute value keeps every weight non-negative, as the modularity of a
    weighted graph assumes.
    """
    corr = np.abs(np.corrcoef(signals))
    np.fill_diagonal(corr, 0.0)
    return nx.from_numpy_array(corr)


def get_modularity(signals: np.ndarray, seed: int = 0) -> float:
    """Modularity of the correlation graph over the rows of ``signals``.

    Partitions the variables into communities with Louvain (seeded, since it is
    randomized), then scores that partition's modularity: high when the rows
    split into strongly-coupled groups, near zero when they form one
    homogeneously connected blob. Returns NaN for fewer than two variables.
    """
    graph = correlation_graph(signals)
    if graph.number_of_nodes() < 2:
        return float("nan")
    communities = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    return nx.community.modularity(graph, communities, weight="weight")


def matched_draws(
    counts: dict[int, int], n_draws: int, seed
) -> tuple[dict[int, np.ndarray], int]:
    """Random index subsets that equalise the channel/unit count across areas."""
    k = min(counts.values())
    rng = np.random.default_rng(seed)
    draws = {
        area: np.vstack(
            [
                rng.choice(n, k, replace=False) if n > k else np.arange(k)
                for _ in range(n_draws)
            ]
        )
        for area, n in counts.items()
    }
    return draws, k
