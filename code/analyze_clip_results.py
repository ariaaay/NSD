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
    parser.add_argument(
        "--plot_voxel_wise_performance",
        action="store_true"
    )
    parser.add_argument(
        "--plot_image_wise_performance",
        action="store_true"
    )
    parser.add_argument(
        "--weight_analysis",
        action="store_true"
    )
    parser.add_argument(
        "--mask",
        default=False,
        action="store_true"
    )
args = parser.parse_args()

# scatter plot per voxels
if args.plot_voxel_wise_performance:
    corr_i = load_model_performance("convnet_res50_person_subset", None, args.output_dir, subj=args.subj)
    # corr_t = load_model_performance("clip_text", None, args.output_dir, subj=args.subj)
    corr_j = load_model_performance("clip_person_subset", None, args.output_dir, subj=args.subj)

    # roi_cmap = np.zeros(corr_i.shape)
    colors = np.load("output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy" % (args.subj, args.subj, args.roi))
    if args.mask:
        mask = colors > 0
        plt.figure(figsize=(7,7))
    else:
        mask = colors > -1
        plt.figure(figsize=(30,15))
    # Plotting text performance vs image performances

    plt.scatter(corr_i[mask], corr_j[mask], alpha=0.07, c=colors[mask])
    plt.plot([-0.1, 1], [-0.1, 1], "r")
    plt.xlabel("resnet_person_subset")
    plt.ylabel("clip_person_subset")
    plt.savefig("figures/CLIP/resnet_person_vs_clip_person_acc_%s.png" % args.roi)

if args.weight_analysis:
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

if args.plot_image_wise_performance:
    # scatter plot by images
    from scipy.stats import pearsonr

    model1 = "clip_person_subset"
    model2 = "convnet_res50_person_subset"

    def compute_sample_corrs(model, output_dir):
        yhat, ytest = load_model_performance(model, output_root=output_dir, measure="pred")
        pvalues = load_model_performance(model, output_root=output_dir, measure="pvalue")
        sig_mask = pvalues <= 0.05

        sample_corrs = [pearsonr(ytest[:,sig_mask][i, :], yhat[:,sig_mask][i, :]) for i in range(ytest.shape[0])]
        return sample_corrs

    sample_corr1 = compute_sample_corrs(model=model1, output_dir=args.output_dir)
    sample_corr2 = compute_sample_corrs(model=model2, output_dir=args.output_dir)

    plt.scatter(sample_corr1, sample_corr2)
    plt.plot([-0.1, 1], [-0.1, 1], "r")
    plt.xlabel(model1)
    plt.ylabel(model2)
    plt.savefig("figures/CLIP/%s_vs_%s_samplewise.png" % (model1, model2))