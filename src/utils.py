"""Shared helpers for accessing session LFP data."""

import mne
import numpy as np

from src.data import SAMPLING_RATE, Session


def get_lfp(session: Session, region: int) -> np.ndarray:
    """LFP array for ``region`` (1/2/3): [num_channels, num_timestamps]."""
    return getattr(session, f"lfp{region}")


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
    lfp = get_lfp(session, region)
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
    lfp = get_lfp(session, region)
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(lfp.shape[0])],
        sfreq=SAMPLING_RATE,
        ch_types="eeg",
    )
    return mne.io.RawArray(lfp, info, verbose=False)
