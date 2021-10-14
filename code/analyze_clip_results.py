import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from util.data_util import load_model_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj",
        type=int,
        default=1,
        help="Specify which subject to build model on. Currently it supports subject 1, 2, 7",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD",
        help="Specify the path to the output directory",
    )
args = parser.parse_args()

corr_i = load_model_performance("clip", None, args.output_dir, subj=args.subj)
corr_t = load_model_performance("clip_text", None, args.output_dir, subj=args.subj)

# Plotting text performance vs image performances
# plt.scatter(corr_i, corr_t, alpha=0.05)
# plt.plot([-0.1, 1], [-0.1, 1], "r")
# plt.xlabel("image")
# plt.ylabel("text")
# plt.savefig("figures/CLIP/image_vs_text_acc.png")

w_i = np.load(
    "%s/output/encoding_results/subj%d/weights_clip_whole_brain.npy"
    % (args.output_dir, args.subj)
)
w_t = np.load(
    "%s/output/encoding_results/subj%d/weights_clip_text_whole_brain.npy"
    % (args.output_dir, args.subj)
)

pca_i = PCA(n_components=5)
pca_i.fit(w_i)
np.save(
    "%s/pca/subj%d/clip_pca_components.npy" % (args.output_dir, args.subj),
    pca_i.components_,
)

pca_t = PCA(n_components=5)
pca_t.fit(w_t)
np.save(
    "%s/pca/subj%d/clip_text_pca_components.npy" % (args.output_dir, args.subj),
    pca_t.components_,
)
