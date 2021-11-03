# import sys, os
# sys.path.append("/Users/ariaw/Projects/cortilities/")

import os
import pickle
import argparse
import json
import numpy as np

from util.util import negative_tail_fdr_threshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Task of Interest
TOI = [
    "edge2d",
    "edge3d",
    "class_places",
    "class_1000",
    "vanishing_point",
    "room_layout",
    "inpainting_whole",
    "rgb2sfnorm",
    "segment2d",
]


def plot_pc_variance(pca):
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel("PCs")
    plt.ylabel("Explained Variance Ratio")
    plt.savefig("figures/PCA/pca_of_acc_subj%d.png" % args.subj)


def plot_pca_proj(pca, acc_mat):
    acc_transform = pca.transform(acc_mat)
    nt = len(TOI)
    plt.figure(figsize=(7, 7))
    plt.imshow(acc_transform)
    plt.yticks(range(0, nt), labels=TOI)
    plt.colorbar()
    plt.xlabel("PCs")
    plt.ylabel("tasks")
    for i in range(nt):
        for j in range(nt):
            _ = plt.text(
                j, i, round(acc_transform[i, j], 2), ha="center", va="center", color="w"
            )
    plt.tight_layout()
    plt.savefig("figures/PCA/pca_proj_of_acc_subj%d.png" % args.subj)


def plot_MDS_of_proj(pca, acc_mat, PC_n):
    from sklearn.manifold import MDS

    acc_transform = pca.transform(acc_mat)
    embedding = MDS(n_components=2)
    out_mds = embedding.fit_transform(acc_transform[:, :PC_n])
    plt.figure(figsize=(6, 6))
    plt.scatter(
        out_mds[:, 0], out_mds[:, 1],
    )
    for i, t in enumerate(TOI):
        plt.annotate(t, (out_mds[i, 0], out_mds[i, 1]))
    plt.title("MDS on task projection onto first %d PCs" % PC_n)
    plt.savefig("figures/PCA/MDS_%d_pc_proj_of_acc_subj%d.png" % (PC_n, args.subj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1, help="specify subjects")
    parser.add_argument("--dir", type=str, default="output/encoding_results")

    args = parser.parse_args()

    acc_mat = []
    for task in TOI:
        output = pickle.load(
            open(
                "%s/subj%d/corr_taskrepr_%s_whole_brain.p"
                % (args.dir, args.subj, task),
                "rb",
            )
        )
        corrs = np.array(output)[:, 0]
        acc_mat.append(corrs)

    acc_mat = np.array(acc_mat)
    np.save("output/pca/subj%d/accuracy_mat.npy" % args.subj, acc_mat)

    assert acc_mat.shape[0] == len(TOI)
    pca = PCA(n_components=9)
    pca.fit(acc_mat)
    np.save("output/pca/subj%d/pca_components.npy" % args.subj, pca.components_)
    with open("output/pca/subj%d/pca.pkl" % args.subj) as pickle_file:
        pickle.dump(pca, pickle_file)

    plot_pc_variance(pca)
    plot_pca_proj(pca, acc_mat)
    plot_MDS_of_proj(pca, acc_mat, 6)
    plot_MDS_of_proj(pca, acc_mat, 3)
