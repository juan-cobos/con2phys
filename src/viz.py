"""Visualize per-session, per-region waveforms and LFP.

Produces two figures per session under ``figures/<id>/``:

* ``waveforms.png`` - average spike waveform of every unit, grouped into one
  panel per brain region (region mean overlaid).
* ``psd.png`` - Welch power spectral density (via MNE) of every LFP channel,
  one panel per brain region (region mean overlaid).

Run::

    uv run python -m src.viz                 # all sessions
    uv run python -m src.viz --sessions 1 2  # a subset
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
import numpy as np

from src.data import Session
from src.utils import get_raw

# Units are labelled by an integer brain-area id (1/2/3), which also indexes the
# matching LFP array. No human-readable names ship with the data.
REGIONS = (1, 2, 3)
FIG_DIR = Path("figures")


def plot_waveforms(session: Session, out_dir: Path) -> Path:
    """One panel per region: every unit's mean waveform plus the region mean."""
    fig, axes = plt.subplots(
        1, len(REGIONS), figsize=(4 * len(REGIONS), 4), sharey=True
    )

    for ax, region in zip(np.atleast_1d(axes), REGIONS):
        wfs = session.waveforms[session.brain_areas == region]
        for wf in wfs:
            ax.plot(wf, color="0.7", lw=0.5, alpha=0.6)
        if len(wfs):
            ax.plot(wfs.mean(0), color="C3", lw=1.8, label="region mean")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Region {region}  (n={len(wfs)} units)")
        ax.set_xlabel("sample")
    np.atleast_1d(axes)[0].set_ylabel("amplitude")

    fig.suptitle(f"Session {session.id} - mean waveforms by region")
    fig.tight_layout()
    path = out_dir / "waveforms.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_spikes_over_time(session: Session, out_dir: Path, bin_s: float) -> Path:
    """Population firing rate over time, one line per region.

    For each region the spikes of all its units are pooled and histogrammed
    onto shared ``bin_s``-second bins (same region/cluster selection as Q2),
    then divided by the bin width to give a spike rate in Hz.
    """
    edges = np.arange(session.spikes.min(), session.spikes.max() + bin_s, bin_s)
    centers = edges[:-1] + bin_s / 2

    fig, ax = plt.subplots(figsize=(12, 4))
    for region, color in zip(REGIONS, ("C0", "C1", "C2")):
        clusters = session.cluster_ids[session.brain_areas == region]
        in_region = np.isin(session.clusters, clusters)
        counts = np.histogram(session.spikes[in_region], bins=edges)[0]
        ax.plot(
            centers,
            counts / bin_s,
            color=color,
            lw=1,
            label=f"Region {region}  (n={len(clusters)} units)",
        )

    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"population rate (spikes/s, {bin_s:g} s bins)")
    ax.set_title(f"Session {session.id} - pooled spike rate by region")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out_dir / "spikes_over_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_psd(session: Session, out_dir: Path, fmin: float, fmax: float) -> Path:
    """One panel per region: PSD via Raw.compute_psd().plot().

    Defaults to the sharp-wave-ripple band (~150-250 Hz in mouse). 250 Hz is
    also the Nyquist limit at the 500 Hz LFP sampling rate, so it is the
    highest frequency the data can resolve.
    """
    fig, axes = plt.subplots(
        1, len(REGIONS), figsize=(4 * len(REGIONS), 4), sharey=True
    )

    for ax, region in zip(np.atleast_1d(axes), REGIONS):
        raw = get_raw(session, region)
        raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False).plot(axes=ax, show=False)
        ax.set_title(f"Region {region}  ({raw.info['nchan']} channels)")

    fig.suptitle(f"Session {session.id} - LFP power spectral density by region")
    fig.tight_layout()
    path = out_dir / "psd.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions", type=int, nargs="+", help="session ids (default: all 1-18)"
    )
    parser.add_argument("--out", type=Path, default=FIG_DIR, help="output directory")
    parser.add_argument("--fmin", type=float, default=0, help="min PSD frequency (Hz)")
    parser.add_argument(
        "--fmax", type=float, default=250, help="max PSD frequency (Hz, <= Nyquist)"
    )
    parser.add_argument(
        "--spike-bin", type=float, default=1.0, help="spike-rate bin width (s)"
    )
    args = parser.parse_args()

    ids = args.sessions if args.sessions else range(1, 19)
    for sid in ids:
        session = Session(sid)
        out_dir = args.out / str(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        wf_path = plot_waveforms(session, out_dir)
        spk_path = plot_spikes_over_time(session, out_dir, bin_s=args.spike_bin)
        psd_path = plot_psd(session, out_dir, fmin=args.fmin, fmax=args.fmax)
        print(f"  wrote {wf_path}, {spk_path} and {psd_path}")


if __name__ == "__main__":
    main()
