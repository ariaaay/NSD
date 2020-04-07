"""
This scripts computes the similarity or distance matrix of tasks based on prediction
on all voxels of the whole brain.
"""

import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import AgglomerativeClustering

from util.model_config import taskrepr_features, task_label
from visualize_corr_in_pycortex import load_data


sns.set(style="whitegrid", font_scale=1)


# def cross_subject_corrs(mat_list):
#     rs = list()
#     for i in range(len(mat_list) - 1):
#         for j in range(len(mat_list) - 1):
#             if i != j + 1:
#                 r = pearsonr(mat_list[i].flatten(), mat_list[j + 1].flatten())
#                 rs.append(r[0])
#     return rs

def get_voxels(model_list, subj):
    datamat = list()
    for l in taskrepr_features:

        data = load_data("taskrepr", task=l, subj=subj, measure="corr")
        datamat.append(data)
    datamat = np.array(datamat)
    return datamat

def get_sig_mask(model_list, correction, alpha, subj):
    maskmat = list()
    for l in model_list:
        mask = np.load(
            "output/voxels_masks/subj%d/taskrepr_{}_{}_{}_whole_brain.npy"
            % (subj, l, correction, alpha)
        )
        maskmat.append(mask)
    maskmat = np.array(maskmat)
    return maskmat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="please specify subject to show")

    parser.add_argument(
        "--use_prediction", default=False, action="store_true",
    )
    parser.add_argument(
        "--use_significance",
        default=False,
        action="store_true",
        help="use the overlap of significant voxels",
    )
    parser.add_argument(
        "--use_mask_corr",
        default=False,
        action="store_true",
        help="use the masked correlation matrix",
    )

    parser.add_argument("--subj", default=1, type=int, help="define which subject")
    parser.add_argument(
        "--method",
        default="cosine",
        help="define what metric should be used to generate task matrix",
    )
    parser.add_argument(
        "--empirical",
        default=False,
        action="store_true",
        help="use masked results with permutation p values",
    )

    # parser.add_argument("--compute_correlation_across_subject", action="store_true")

    args = parser.parse_args()

    if args.empirical:
        p_method = "emp_fdr"
    else:
        p_method = "fdr"

    n_tasks = len(taskrepr_features)  # 21 tasks

    #mask correlation based on significance
    voxel_mat = get_voxels(list(task_label.keys()), subj=args.subj)
    sig_mat = get_sig_mask(
        list(task_label.keys()), correction=p_method, alpha=0.05, subj=args.subj
    )
    assert voxel_mat.shape == sig_mat.shape

    voxel_mat[~sig_mat] = 0
    print(voxel_mat.shape)
    np.save(
        "output/task_matrix/mask_corr_subj%d_%s.npy" % (args.subj, p_method),
        voxel_mat,
    )
    #

    print(voxel_mat.shape)
    if args.method == "l2":
        sim = euclidean_distances(voxel_mat)
    elif args.method == "cosine":
        sim = cosine_similarity(voxel_mat, dense_output=False)

    np.save(
        "../outputs/task_matrix/task_matrix_subj%d_%s.npy" % (args.subj, args.method),
        sim,
    )

    model = AgglomerativeClustering(
        n_clusters=3, linkage="single", affinity="cosine"
    ).fit(sim)

    # print(model.labels_)
    order = np.argsort(model.labels_)
    fit_data = sim[order]
    fit_data = fit_data[:, order]
    # fit_data = fit_data / 3

    # all_sim = all_sim/3
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    all_task_label = list(task_label.values())
    # active_labels = all_task_label[:14] + all_task_label[15:]
    labs = np.array(all_task_label)

    ax = sns.heatmap(
        fit_data,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        xticklabels=labs[order],
        yticklabels=labs[order],
    )

    ax.set_ylim(0, 21.1)
    plt.subplots_adjust(bottom=0.3)

    plt.savefig(
        "figures/taskmatrix/task_matrix_subj%d_%s_%s.pdf" % (args.subj, args.method, p_method)
    )
