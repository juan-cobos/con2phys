from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import loadmat

DATA_DIR = Path("dataset")
VALID_IDS = set(range(1, 19))
VERTICAL_SPACING = 20  # um
SAMPLING_RATE = 500  # Hz


def _load_mat(path: Path, key: str) -> np.ndarray:
    """Load a single named variable from a MATLAB .mat file."""
    return loadmat(path)[key]


class Session:
    """A single recording session (one mouse, folder ``dataset/<id>``).

    Holds the raw arrays for the whole session. Spikes/clusters are pooled
    across all units; per-unit access is provided by :meth:`unit`.
    """

    def __init__(self, id: int):
        if id not in VALID_IDS:
            raise ValueError(f"id must be one of {sorted(VALID_IDS)}, got {id}")

        self.id = id
        self.data_path = DATA_DIR / f"{id}"

        # Spike times (s) and the cluster (unit) id each spike belongs to.
        # Both are [num_spikes] and index-aligned.
        self.spikes = _load_mat(self.data_path / "spikes.mat", "spikes").ravel()
        self.clusters = _load_mat(self.data_path / "clusters.mat", "clusters").ravel()

        # Per-unit metadata, ordered the same as waveforms: [num_units].
        ba = _load_mat(self.data_path / "brain_area.mat", "brain_areas")
        self.cluster_ids = ba["cluster_id"][0, 0].ravel()
        self.brain_areas = ba["brain_area"][0, 0].ravel()

        # Average waveform per unit: [num_units, 128].
        self.waveforms = _load_mat(self.data_path / "waveforms.mat", "wf")

        # LFP per brain area, [num_channels, num_timestamps]. Channel counts
        # differ between areas, so each area is kept as its own array.
        self.lfp1 = _load_mat(self.data_path / "lfp_1.mat", "lfp1")
        self.lfp2 = _load_mat(self.data_path / "lfp_2.mat", "lfp2")
        self.lfp3 = _load_mat(self.data_path / "lfp_3.mat", "lfp3")

        self.trial_data = pl.read_excel(self.data_path / "trial_data.xlsx")

        print(f"Session {id} loaded: {self.n_units} units, {self.spikes.size} spikes")

    @property
    def n_units(self) -> int:
        return self.cluster_ids.size


class Dataloader:
    """Loads recording sessions by id, lazily and cached.

    Each session holds hundreds of MB of LFP, so sessions are loaded on first
    access rather than all up front.

    Examples::

        dl = Dataloader()              # all 18 sessions
        dl = Dataloader([1, 2, 3])     # a subset
        session = dl[1]                # load (and cache) one session
        for session in dl:             # iterate, loading one at a time
            ...
    """

    def __init__(self, ids: Iterable[int] | None = None):
        ids = sorted(VALID_IDS) if ids is None else list(dict.fromkeys(ids))
        invalid = set(ids) - VALID_IDS
        if invalid:
            raise ValueError(
                f"invalid session ids {sorted(invalid)}; valid: {sorted(VALID_IDS)}"
            )
        self.ids = ids
        self._cache: dict[int, Session] = {}

    def load(self, id: int) -> Session:
        """Load session ``id`` (from cache if already loaded)."""
        if id not in self.ids:
            raise KeyError(f"session {id} not in this loader ({self.ids})")
        if id not in self._cache:
            self._cache[id] = Session(id)
        return self._cache[id]

    def __getitem__(self, id: int) -> Session:
        return self.load(id)

    def __iter__(self) -> Iterator[Session]:
        for id in self.ids:
            yield self.load(id)

    def __len__(self) -> int:
        return len(self.ids)


if __name__ == "__main__":
    dl = Dataloader([1, 2])
    print(f"Dataloader with {len(dl)} sessions: {dl.ids}")

    s = dl[1]
    print(f"spikes:     {s.spikes.shape} {s.spikes.dtype}")
    print(f"clusters:   {s.clusters.shape} {s.clusters.dtype}")
    print(f"cluster_ids:{s.cluster_ids.shape}  brain_areas:{np.unique(s.brain_areas)}")
    print(f"waveforms:  {s.waveforms.shape}")
    print(f"lfp:        {s.lfp1.shape}, {s.lfp2.shape}, {s.lfp3.shape}")
    print(f"trial_data: {s.trial_data.shape} cols={s.trial_data.columns}")

    for session in dl:
        print(f"  session {session.id}: {session.n_units} units")
