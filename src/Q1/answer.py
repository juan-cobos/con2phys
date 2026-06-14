"""

Question 1
    Which brain area (if any) has the highest density of ripples (i.e. “hippocampal” ripples traditionally occurring during sharp wave-ripples)?

"""

# First, define different ways to compute SWR
# Stratify data for each region
# waveforms shape matches cluster ids and brain_areas

#
from collections import defaultdict

import numpy as np

from src.data import Dataloader


def get_swr(): ...


dl = Dataloader([1, 2])
print(f"Dataloader with {len(dl)} sessions: {dl.ids}")

# Classify waveforms by brain_area
wf_by_ba = defaultdict(list)
for s in dl:
    for wf, ba in zip(s.waveforms, s.brain_areas):
        wf_by_ba[ba].append(wf)

# np.stack each of the (batch_size, 128)
for k, v in wf_by_ba.items():
    wf_by_ba[k] = np.stack(v)

# TODO: consider and analyze the lfp for each brain_area (lfp1, lfp2 and lfp3)
