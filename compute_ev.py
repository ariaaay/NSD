import argparse
import numpy as np
import pandas as pd

from util.util import ev

# extract each subject's index for 1000 images
def extract_subject_trials_index_common(subj):
    index = list()
    for i in range(3):
        col = 'subject%01d_rep%01d' % (subj, i)
        assert len(stim[col][stim['shared1000']]) == 1000
        index.append(list(stim[col][stim['shared1000']]))
    assert len(index) == 3
    return index


def extract_common_data_into_matrix(subj, roi_only=False):
    if roi_only:
        roi = "_roi_only"
    else:
        roi = ""
    l = extract_subject_trials_index_common(subj)
    data = np.load("output/cortical_voxel_across_sessions_subj%02d%s.npy" % (subj, roi))
    repeat = np.hstack((data[np.array(l[0])], data[np.array(l[1])], data[np.array(l[2])]))
    print(repeat.shape)
    assert repeat.shape == (1000,3)
    return repeat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_only")
    parser.add_argument("--subj", type=int)

    args = parser.parse_args()
    stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")

    repeat_matrix = extract_common_data_into_matrix(args.subj, roi_only=args.roi_only)
    out = ev(repeat_matrix)
    print(out)