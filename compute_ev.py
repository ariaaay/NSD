import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from util.util import ev

# extract each subject's index for 1000 images
def extract_subject_trials_index_common(stim, subj):
    index = list()
    for i in range(3):
        col = 'subject%01d_rep%01d' % (subj, i)
        assert len(stim[col][stim['shared1000']]) == 1000
        index.append(list(stim[col][stim['shared1000']]))
    assert len(index) == 3
    return index


def compute_ev(stim, subj, roi=""):
    l = extract_subject_trials_index_common(stim, subj)
    data = np.load("output/cortical_voxel_across_sessions_subj%02d%s.npy" % (subj, roi))
    ev_list = []
    for v in tqdm(range(data.shape[1])):
        repeat = np.array([data[np.array(l[i]),v] for i in range(3)]).T
        try:
            assert repeat.shape == (1000,3)
        except AssertionError:
            print(repeat.shape)
        ev_list.append(ev(repeat))

    return ev_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_only")
    parser.add_argument("--subj", type=int)

    args = parser.parse_args()
    if args.roi_only:
        roi = "_roi_only"
    else:
        roi = ""

    stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")

    all_evs = compute_ev(stim, args.subj, roi)
    np.save("output/evs_subj%02d%s.npy" % (args.subj, roi))
    plt.hist(all_evs)
    plt.savefig("figures/evs_subj%02d%s.npy")