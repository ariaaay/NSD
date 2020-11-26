"""
Regress a feature out of another and predict with residual
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from featureprep.feature_prep import get_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", default=[], nargs="+", type=str)
    parser.add_argument("--subjs", default=[1])

    args = parser.parse_args()
    stim = pd.read_pickle(
        "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
    )

    stim_list = stim.cocoId

    tasks = ["taskrepr_" + task for task in args.tasks]
    X = get_features(0, stim_list, tasks[0])
    Y = get_features(0, stim_list, tasks[1])
    assert X.shape[0] == 73000
    assert Y.shape[0] == 73000

    reg = LinearRegression().fit(X, Y)

    for s in args.subjs:
        X_subj = get_features(s, stim_list, tasks[0])
        Y_true = get_features(s, stim_list, tasks[1])

        assert X_subj.shape[0] == 10000
        assert Y_true.shape[0] == 10000
        Y_pred = reg.predict(X_subj)

        Y_res = Y_true - Y_pred
        np.save(
            "features/subj%d/pred_of_%s_from_%s.npy" % (s, tasks[1], tasks[0]), Y_pred
        )
        np.save(
            "features/subj%d/res_of_%s_from_%s.npy" % (s, tasks[1], tasks[0]), Y_res
        )
