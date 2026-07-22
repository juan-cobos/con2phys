"""
Scratch script for inspecting how spikes, clusters, units and LFP relate.
"""

import sys

import numpy as np

from src.data import DATA_DIR, SAMPLING_RATE, Session


def inspect(session_id: int) -> None:
    path = DATA_DIR / str(session_id)
    session = Session(session_id)

    print(f"\n{'=' * 70}\nSession {session_id}  ({path})\n{'=' * 70}")

    # ---- per-spike arrays ------------------------------------------------
    spikes = session.spikes
    clusters = session.clusters
    print("\n-- per-spike (index-aligned) --")
    print(
        f"spikes    {spikes.shape} {spikes.dtype}  [{spikes.min():.2f} .. {spikes.max():.2f}] s"
    )
    print(f"clusters  {clusters.shape} {clusters.dtype}")
    print(f"aligned:  {spikes.shape == clusters.shape}")
    spike_span_s = float(spikes.max() - spikes.min())

    # ---- per-unit arrays -------------------------------------------------
    cluster_ids = session.cluster_ids
    brain_areas = session.brain_areas
    waveforms = session.waveforms
    print("\n-- per-unit --")
    print(f"cluster_ids {cluster_ids.shape} {cluster_ids.dtype}")
    print(
        f"brain_areas {brain_areas.shape} {brain_areas.dtype}  unique={np.unique(brain_areas)}"
    )
    print(f"waveforms   {waveforms.shape}")
    print(f"units == waveform rows: {cluster_ids.size == waveforms.shape[0]}")

    # ---- do the two index spaces agree? ----------------------------------
    # A cluster id appearing in the spike train but not in the unit metadata
    # would have no brain area (unassignable); a unit with no spikes is simply
    # silent and should still count as a 0 Hz unit.
    in_spikes = set(np.unique(clusters).tolist())
    in_meta = set(cluster_ids.tolist())
    orphans = in_spikes - in_meta
    silent = in_meta - in_spikes
    print("\n-- clusters vs units --")
    print(f"distinct clusters in spike train: {len(in_spikes)}")
    print(f"units in metadata:                {len(in_meta)}")
    print(f"spike clusters with NO unit/area: {len(orphans)} {sorted(orphans)[:10]}")
    print(f"units with NO spikes (silent):    {len(silent)} {sorted(silent)[:10]}")

    # ---- per-area breakdown ---------------------------------------------
    uniq, counts = np.unique(clusters, return_counts=True)
    count_by_cluster = dict(zip(uniq.tolist(), counts.tolist()))

    print("\n-- per area --")
    print(
        f"{'area':>4} {'units':>6} {'spikes':>10} {'lfp chans':>10} "
        f"{'lfp samples':>12} {'lfp dur (s)':>12} {'mean Hz/unit':>13}"
    )
    for area in np.unique(brain_areas):
        area_clusters = cluster_ids[brain_areas == area]
        n_spk = np.array(
            [count_by_cluster.get(c, 0) for c in area_clusters.tolist()], dtype=float
        )

        # Memory-map the LFP: we only want the shape, and these arrays are
        # hundreds of MB each.
        lfp = session.get_lfp_in(area)
        n_chan, n_samp = lfp.shape
        lfp_dur = n_samp / SAMPLING_RATE
        chans, samps, dur = f"{n_chan}", f"{n_samp}", f"{lfp_dur:.1f}"
        rate = n_spk.mean() / lfp_dur if lfp_dur == lfp_dur else float("nan")

        print(
            f"{area:>4} {area_clusters.size:>6} {int(n_spk.sum()):>10} {chans:>10} "
            f"{samps:>12} {dur:>12} {rate:>13.4f}"
        )

    # ---- how long is "the entire recording"? -----------------------------
    # Matters for any rate denominator: the spike span can be noticeably
    # shorter than the LFP if spiking starts late or stops early.
    lfp1_dur = session.lfp1.shape[0]
    lfp2_dur = session.lfp2.shape[0]
    lfp3_dur = session.lfp3.shape[0]
    print("\n-- recording duration --")
    print(f"spike span (max-min): {spike_span_s:>10.1f} s")
    print(f"lfp_1 timestamps:     {lfp1_dur:>10.1f} s")
    print(f"lfp_2 timestamps:     {lfp2_dur:>10.1f} s")
    print(f"lfp_3 timestamps:     {lfp3_dur:>10.1f} s")
    print(f"difference:           {lfp_dur - spike_span_s:>10.1f} s")

    print(session.trial_data.head())


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or [1]
    for sid in ids:
        inspect(sid)
