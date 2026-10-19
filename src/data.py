import weakref
from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path("dataset")
VALID_IDS = set(range(1, 19))
VERTICAL_SPACING = 20  # um
SAMPLING_RATE = 500  # Hz, LFP and spike-time binning
WAVEFORM_SAMPLING_RATE = 30_000  # Hz, mean spike waveforms
ALL_SESSIONS = list(VALID_IDS)
AREAS = (1, 2, 3)


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
        self.spikes = np.load(self.data_path / "spikes.npy").ravel()
        self.clusters = np.load(self.data_path / "clusters.npy").ravel()

        # Per-unit metadata, ordered the same as waveforms: [num_units].
        ba = np.load(self.data_path / "brain_area.npy", allow_pickle=True).item()
        self.cluster_ids = ba["cluster_id"].ravel()
        self.brain_areas = ba["brain_area"].ravel()

        # Average waveform per unit: [num_units, 128].
        self.waveforms = np.load(self.data_path / "waveforms.npy")

        # LFP per brain area, [num_channels, num_timestamps]. Channel counts
        # differ between areas, so each area is kept as its own array.
        self.lfp1 = np.load(self.data_path / "lfp_1.npy")
        self.lfp2 = np.load(self.data_path / "lfp_2.npy")
        self.lfp3 = np.load(self.data_path / "lfp_3.npy")

        self.trial_data = pl.read_csv(self.data_path / "trial_data.csv")

        print(f"Session {id} loaded: {self.n_units} units, {self.spikes.size} spikes")

    @property
    def n_units(self) -> int:
        return self.cluster_ids.size

    @property
    def duration_s(self) -> float:
        """Length of the whole recording, in seconds."""
        return self.lfp1.shape[1] / SAMPLING_RATE

    def area_mask(self, region: int) -> np.ndarray:
        """Boolean mask over this session's units, selecting those in ``region``.

        Indexes the per-unit arrays (``cluster_ids``, ``brain_areas``,
        ``waveforms``, :attr:`spike_counts`, and anything aligned with them).
        """
        return self.brain_areas == region

    @cached_property
    def spike_counts(self) -> np.ndarray:
        """Spike count per unit, aligned with :attr:`cluster_ids`. [num_units]

        Cluster ids are only unique *within* a session -- different mice can
        reuse the same id -- so this is a per-session array positionally
        aligned with this session's units, never an id-keyed mapping that
        could silently collide across sessions.

        Units that never fire are absent from the spike train and get a count
        of 0, so they survive as silent units instead of being dropped.

        Cached: the ``np.unique`` pass is over every spike in the session
        (millions), and the result is reused by each area.
        """
        uniq, counts = np.unique(self.clusters, return_counts=True)
        count_by_cluster = dict(zip(uniq.tolist(), counts.tolist()))
        return np.array(
            [count_by_cluster.get(c, 0) for c in self.cluster_ids.tolist()],
            dtype=float,
        )

    def get_spikes_in(self, area: int) -> np.ndarray:
        """Spike times (s) of every unit in ``area``, pooled. [num_spikes]

        Cluster ids repeat across sessions but not within one, so selecting by
        this session's own ids is unambiguous. Order follows :attr:`spikes`.
        """
        clusters = self.cluster_ids[self.area_mask(area)]
        return self.spikes[np.isin(self.clusters, clusters)]

    def get_unit_spikes_in(self, area: int) -> list[np.ndarray]:
        """Spike times (s) of each unit in ``area``: one array per cluster.

        Same spikes as :meth:`get_spikes_in`, kept split by unit for measures
        that are only meaningful per neuron (e.g. spike-field phase locking,
        where pooling units with different preferred phases cancels the effect).
        Silent units contribute empty arrays rather than disappearing.
        """
        clusters = self.cluster_ids[self.area_mask(area)]
        return [self.spikes[self.clusters == c] for c in clusters]

    def get_lfp_in(self, area: int) -> np.ndarray:
        """LFP array for ``area`` (1/2/3): [num_channels, num_timestamps]."""
        return getattr(self, f"lfp{area}")

    def event_bounds(
        self, start_event: str, stop_event: str, min_samples: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample-index window ``[lo, hi)`` per trial for a ``start_event`` ->
        ``stop_event`` segment, plus a boolean mask of usable trials.
        """
        start = self.trial_data[start_event].to_numpy().astype(float)
        stop = self.trial_data[stop_event].to_numpy().astype(float)
        n_samples = self.lfp1.shape[1]

        safe_start = np.where(np.isfinite(start), start, 0.0)
        safe_stop = np.where(np.isfinite(stop), stop, 0.0)
        lo = np.floor(safe_start * SAMPLING_RATE).astype(int)
        hi = np.ceil(safe_stop * SAMPLING_RATE).astype(int)

        keep = np.isfinite(start) & np.isfinite(stop)
        keep &= (lo >= 0) & (hi <= n_samples) & ((hi - lo) >= min_samples)
        return lo, hi, keep


class Dataloader:
    """Loads recording sessions by id, lazily and cached.

    Each session holds hundreds of MB of LFP, so sessions are loaded on first
    access rather than all up front.

    The cache holds sessions only *weakly*: a session stays cached while the
    caller keeps a reference to it, and is freed automatically once no longer
    referenced. So ``for session in dl:`` streams one session at a time —
    each iteration frees the previous one — while ``s = dl[1]`` stays cached
    for as long as ``s`` is alive.

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
        self._cache: weakref.WeakValueDictionary[int, Session] = (
            weakref.WeakValueDictionary()
        )

    def load(self, id: int) -> Session:
        """Load session ``id`` (from cache if already loaded)."""
        if id not in self.ids:
            raise KeyError(f"session {id} not in this loader ({self.ids})")
        session = self._cache.get(id)
        if session is None:
            session = Session(id)
            self._cache[id] = session
        return session

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
    print(f"cluster_ids:{s.cluster_ids.shape}  brain_areas:{s.brain_areas.shape}")
    print(f"waveforms:  {s.waveforms.shape}")
    print(f"lfp:        {s.lfp1.shape}, {s.lfp2.shape}, {s.lfp3.shape}")
    print(f"trial_data: {s.trial_data.shape} cols={s.trial_data.columns}")

    for session in dl:
        print(f"  session {session.id}: {session.n_units} units")
