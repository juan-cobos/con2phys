# Dataset Structure

This document describes the organization of the CON²PHYS dataset.

## Overview

The dataset:

- Contains **18 compressed files**, one for each mouse.
- Includes **single-unit activity (SUA)** and **local field potentials (LFP)** recorded with **Neuropixels probes** during a behavioral task.
- Provides only minimal information about the behavioral task and electrode placement to reduce experimental bias.
- Contains recordings from **3 simultaneously recorded brain areas**.
- Each recording includes:
  - **LFP:** at least 20 channels in total (minimum 10 channels per brain area)
  - **SUA:** at least 5 units per brain area
  - **1,449 total units** across the complete dataset.
- Recording durations range from **55 to 101 minutes**.

> **Note**
>
> If you downloaded the dataset before **15 July 2026**, download the corrected version.

---

# Files Included for Each Mouse

Each mouse folder contains the following data:

| File | Description |
|------|-------------|
| `trial_data.csv` | Behavioral trial information |
| `brain_area.npy` / `.mat` | Brain area and cluster mapping |
| `spikes.npy` / `.mat` | Spike timestamps |
| `clusters.npy` / `.mat` | Spike-to-cluster assignments |
| `waveforms.npy` / `.mat` | Average spike waveforms |
| `lfp*.npy` / `.mat` | Local field potential recordings |

---

# Trial Data

**Format:** `.csv`

Behavioral information aligned to electrophysiology recordings.

| Column | Description |
|--------|-------------|
| `trial_start` | Trial onset (seconds) |
| `stim_start` | Stimulus presentation (seconds) |
| `outcome` | Reward or punishment time (seconds) |
| `trial_end` | End of trial (seconds) |
| `Variable A` | Binary behavioral variable |
| `Variable B` | Binary behavioral variable |
| `Variable C` | Behavioral variable with values 1–3 |

---

# Brain Area

**Format:** `.npy`, `.mat`

Contains two arrays:

1. **Brain area IDs**

   Shape:

   ```
   [1 × num_units]
   ```

   Integer values:

   - `1`
   - `2`
   - `3`

   indicating the brain area from which each unit was recorded.

2. **Cluster IDs**

   Shape:

   ```
   [1 × num_units]
   ```

   Integer identifier for every recorded unit.

---

# Spikes

**Format:** `.npy`, `.mat`

Spike timestamps.

Shape:

```
[1 × num_spikes]
```

Contents:

- Spike times (seconds)

---

# Clusters

**Format:** `.npy`, `.mat`

Maps every spike to a cluster.

Shape:

```
[1 × num_spikes]
```

Contents:

- Cluster ID corresponding to each spike.

---

# Waveforms

**Format:** `.npy`, `.mat`

Average extracellular waveform for each unit.

Shape:

```
[num_units × 128]
```

Details:

- Recorded at **30 kHz**
- Extracted from the **best detection channel**
- Ordering matches the entries in:
  - `brain_area`
  - cluster IDs

---

# Local Field Potentials (LFP)

**Format:** `.npy`, `.mat`

Three arrays are provided:

- `lfp1`
- `lfp2`
- `lfp3`

Each has shape:

```
[num_channels × num_timepoints]
```

## Notes

- The number of channels varies across brain areas and mice.
- Each brain area contains **at least 10 channels**.
- Channels belonging to the same brain area are spatially contiguous.
- Different brain areas occupy separate probe regions.
- Within each brain area, channels are ordered from:
  - deepest
  - →
  - most superficial.
- Every other Neuropixels recording site is included.
- Vertical spacing between recording sites is **20 μm**.
- Signals were:
  - externally referenced
  - preprocessed before release.
- Sampling rate: **500 Hz**.
