import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from util.data_util import fill_in_nan_voxels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj",
        type=int,
        default=1,
        help="Specify which subject to build model on. Currently it supports subject 1, 2, 7",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD",
        help="Specify the path to the output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip"
    )
    parser.add_argument(
        "--spectral_clustering",
        action="store_true"
    )
    args = parser.parse_args()

if args.spectral_clustering:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering
    subj_w = np.load(
                "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
                % (args.output_root, args.subj, args.model)
            )
    subj_w = fill_in_nan_voxels(subj_w, args.subj, args.output_root)
    subj_mask = np.load(
            "%s/output/pca/%s/pca_voxels/pca_voxels_subj%02d_best_20000.npy"
            % (args.output_root, args.model, args.subj)
        )
    subj_w = subj_w[:, subj_mask]
    print(subj_w.shape)
    # w_sim = cosine_similarity(subj_w.T)
    # print(np.sum(np.isnan(subj_w)))
    # print(np.sum(np.isinf(subj_w)))
    
    clustering = SpectralClustering(
        n_clusters=4,
        assign_labels='kmeans',
        affinity="rbf",
        random_state=0).fit(subj_w.T)

    # print(clustering.labels_)
    if not os.path.exists("%s/output/clustering" % args.output_root):
        os.makedirs("%s/output/clustering" % args.output_root)
    
    if not os.path.exists("figures/clustering"):
        os.makedirs("figures/clustering")
    
    np.save("%s/output/clustering/spectral_subj%01d.npy" % (args.output_root, args.subj) , clustering.labels_)
    plt.hist(clustering.labels_)
    plt.savefig("figures/clustering/spectral_subj%01d.png" % args.subj)