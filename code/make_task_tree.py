"""
This scripts computes the dendrogram of tasks based on prediction
on all voxels of the whole brain.
"""


import argparse

# import plotly
# import plotly.figure_factory as ff

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from util.model_config import *

import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


def sim(voxel_mat, metric=None):
    if metric is None:
        dist = squareform(pdist(voxel_mat))
    else:
        dist = squareform(pdist(voxel_mat, metric))
    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int)
    parser.add_argument("--method", type=str, default="masked_corr")
    parser.add_argument("--roi", default=None, type=str, help="layer specific ROI mask")
    parser.add_argument(
        "--exclude_roi", default=None, type=str, help="exclude specidic ROI"
    )

    args = parser.parse_args()
    roi_tag = ""
    if args.roi:
        roi_tag += "_only_%s" % args.roi
    elif args.exclude_roi:
        roi_tag += "_exclude_%s" % args.exclude_roi


    matrix_path = "output/task_matrix"
    if args.method == "sig":
        mat = np.load("%s/sig_mask_subj%d.npy" % (matrix_path, args.subj)).astype(int)
    elif args.method == "masked_corr":
        mat = np.load(
            "output/task_matrix/mask_corr_subj%d_emp_fdr%s.npy"
            % (args.subj, roi_tag)
        )
        # mat = np.load("output/task_matrix/mask_corr_subj%d_emp_fdr.npy" % args.subj)

    for linkage in ["average", "ward"]:
        Z = hierarchy.linkage(mat, linkage)
        # X = sim(mat,  metric=lambda u, v: u @ v)
        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 1, 1)
        labs = list(task_label.values())[:21]
        dn = hierarchy.dendrogram(
            Z,
            ax=ax,
            labels=labs,
            leaf_font_size=15,
            color_threshold=0,
            above_threshold_color="gray",
        )
        plt.xticks(rotation="vertical")

        # post hoc hand code node colors
        # if args.subj == 1:
        color_list = ["blue"] * 9 + ["green"] * 10 + ["purple"] * 2

        [t.set_color(i) for (i, t) in zip(color_list, ax.xaxis.get_ticklabels())]

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.margins(0.2)
        plt.subplots_adjust(bottom=0.5)
        plt.savefig(
            "figures/task_tree/dendrogram_subj%d_%s_%s.pdf"
            % (args.subj, args.method, linkage)
        )
