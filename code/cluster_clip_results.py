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
    args = parser.parse_args()

if args.spectral_clustering:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import SpectralClustering
    subj_w = np.load(
                "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
                % (args.output_root, args.subj, args.model)
            )
    subj_w = fill_in_nan_voxels(subj_w, args.subj, args.output_root)
    print(subj_w.shape)
    w_sim = cosine_similarity(subj_w)
    clustering = SpectralClustering(
        assign_labels='cluster_qr',
        affinity="precomputed",
        random_state=0).fit(w_sim)

    print(clustering.labels_)
    if not os.path.exists("%s/output/clustering" % args.output_dir):
        os.makedir("%s/output/clustering" % args.output_dir)
    
    if not os.path.exists("figures/clustering"):
        os.makedir("figures/clustering")
    
    np.save("%s/output/clustering/spectral_cluster_qr_subj%01d.npy" % (args.output_dir, args.subj) , clustering.labels_)
    plt.hist(clustering.labels_)
    plt.savefig("figures/clustering/spectral_cluster_qr_subj%01d.png" % args.subj)