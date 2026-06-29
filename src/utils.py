"""Shared helpers for accessing session LFP data."""

import mne
import numpy as np

from src.data import SAMPLING_RATE, Session


def get_lfp(session: Session, region: int) -> np.ndarray:
    """LFP array for ``region`` (1/2/3): [num_channels, num_timestamps]."""
    return getattr(session, f"lfp{region}")


def get_raw(session: Session, region: int) -> mne.io.RawArray:
    """Wrap a region's LFP array in an MNE Raw object."""
    lfp = get_lfp(session, region)
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(lfp.shape[0])],
        sfreq=SAMPLING_RATE,
        ch_types="eeg",
    )
    return mne.io.RawArray(lfp, info, verbose=False)
