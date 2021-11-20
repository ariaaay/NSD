"""
This scripts runs FDR correction on given p-values.
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
from glob import glob

from util.model_config import *


def compute_adjust_p(
    model,
    feature="",
    subj=None,
    correction="fdr",
):
    out = pickle.load(
        open(
            "output/encoding_results/subj%d/corr_%s_%s_whole_brain.p"
            % (subj, model, feature),
            "rb",
        )
    )
    # print(len(out))
    pvalues = np.array(out)[:, 1]
    if correction == "fdr":
        adj_p = fdrcorrection(pvalues)[1]
    elif correction == "bonferroni":
        adj_p = pvalues * len(pvalues)
    return pvalues, adj_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--model", default="taskrepr")
    parser.add_argument("--correction", default="fdr")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--use_empirical_p", action="store_true")
    args = parser.parse_args()

    all_ps = list()
    try:
        flist = model_features[args.model]
    except KeyError:
        flist = [""]
    for f in flist:
        if args.use_empirical_p:
            adj_p = pickle.load(
                open(
                    "output/baseline/empirical_p_values_taskrepr_%s_subj%d_whole_brain_test_permute_fdr.p"
                    % (
                        f,
                        args.subj,
                    ),
                    "rb",
                )
            )
        else:
            p, adj_p = compute_adjust_p(
                args.model,
                f,
                subj=args.subj,
                correction=args.correction,
            )
        # print(len(adj_p))
        print(f + ": " + str(np.sum(np.array(adj_p) < args.alpha)))

        mask_dir = "output/voxels_masks/subj{}".format(args.subj)
        if not os.path.isdir(mask_dir):
            os.makedirs(mask_dir)
        if args.use_empirical_p:
            np.save(
                "%s/%s_%s_emp_%s_%s"
                % (mask_dir, args.model, f, args.correction, str(args.alpha)),
                adj_p < args.alpha,
            )
        else:
            np.save(
                "%s/%s_%s_%s_%s"
                % (mask_dir, args.model, f, args.correction, str(args.alpha)),
                adj_p < args.alpha,
            )

        all_ps.append(adj_p)
