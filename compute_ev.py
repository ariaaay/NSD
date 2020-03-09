import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from util.util import ev

# extract each subject's index for 1000 images
def extract_subject_trials_index_shared1000(stim, subj):
    index = list()
    for i in range(3):
        col = 'subject%01d_rep%01d' % (subj, i)
        assert len(stim[col][stim['shared1000']]) == 1000
        index.append(list(stim[col][stim['shared1000']]))
    assert len(index) == 3
    return index


def extract_subject_trials_index(stim, subj):
    # load trial index
    trial_index_path = "output/trials_subj%02d.pkl" % args.subj
    trial_lists = pickle.load(open(trial_index_path, "rb"))
    return trial_lists,

def compute_ev(stim, subj, roi="", biascorr=False, zscored_input=False):
    l = extract_subject_trials_index(stim, subj)
    repeat_n = len(l[-1])
    if zscored_input:
        data = np.load("output/cortical_voxel_across_sessions_zscored_by_run_subj%02d%s.npy" % (subj, roi))
    else:
        data = np.load("output/cortical_voxel_across_sessions_subj%02d%s.npy" % (subj, roi))

    ev_list = []
    for v in tqdm(range(data.shape[1])): #loop over voxels
        repeat = np.array([data[np.array(l[i]), v] for i in range(3)]).T
        try:
            assert repeat.shape == (repeat_n,3)
        except AssertionError:
            print(repeat.shape)
        ev_list.append(ev(repeat, biascorr=biascorr))
    return np.array(ev_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_only", action="store_true")
    parser.add_argument("--subj", type=int)
    parser.add_argument("--biascorr", action="store_true")
    parser.add_argument("--zscored_input", action="store_true")

    args = parser.parse_args()

    tag = ""

    if args.roi_only:
        roi = "_roi_only"
        tag += roi
    else:
        roi = ""

    if args.biascorr:
        tag += "_biascorr"

    if args.zscored_input:
        tag += "_zscored"

    stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
    try:
        all_evs = np.load("output/evs_subj%02d%s.npy" % (args.subj, tag))
    except FileNotFoundError:
        print("computing EVs")
        all_evs = compute_ev(stim, args.subj, roi, args.biascorr, args.zscored_input)
        np.save("output/evs_subj%02d%s.npy" % (args.subj, tag), all_evs)

    plt.figure()
    plt.hist(all_evs)
    plt.title("Explainable Variance across Voxels (subj%02d%s)" % (args.subj, tag))
    plt.savefig("figures/evs_subj%02d%s.png" % (args.subj, tag))