import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import clip

from sklearn.decomposition import PCA

from util.data_util import load_model_performance, extract_test_image_ids


def compute_sample_corrs(model, output_dir):
    try:
        sample_corrs = np.load(
            "%s/output/clip/%s_sample_corrs.npy" % (output_dir, model)
        )
    except FileNotFoundError:
        yhat, ytest = load_model_performance(model, output_root=output_dir, measure="pred")
        pvalues = load_model_performance(model, output_root=output_dir, measure="pvalue")
        sig_mask = pvalues <= 0.05

        sample_corrs = [
            pearsonr(ytest[:, sig_mask][i, :], yhat[:, sig_mask][i, :])
            for i in range(ytest.shape[0])
        ]
        np.save(
            "%s/output/clip/%s_sample_corrs.npy" % (output_dir, model)
        )
    
    return sample_corrs


def plot_image_wise_performance(model1, model2):
    sample_corr1 = compute_sample_corrs(model=model1, output_dir=args.output_dir)
    sample_corr2 = compute_sample_corrs(model=model2, output_dir=args.output_dir)

    plt.figure()
    plt.scatter(sample_corr1[:, 0], sample_corr2[:, 0], alpha=0.3)
    plt.plot([-0.1, 1], [-0.1, 1], "r")
    plt.xlabel(model1)
    plt.ylabel(model2)
    plt.savefig("figures/CLIP/%s_vs_%s_samplewise.png" % (model1, model2))


def find_corner_images(model1, model2, upper_thr=0.5, lower_thr=0.03):
    sc1 = np.load("%s/output/clip/%s_sample_corrs.npy" % (args.output_dir, model1))[
        :, 0
    ]
    sc2 = np.load("%s/output/clip/%s_sample_corrs.npy" % (args.output_dir, model2))[
        :, 0
    ]
    diff = sc1 - sc2
    indexes = np.argsort(diff)
    br = indexes[:20]
    tl = indexes[::-1][:20]
    tr = np.where(((sc1 > upper_thr) * 1 * ((sc2 > upper_thr) * 1).T))
    bl = np.where(((sc1 > lower_thr) * 1 * ((sc2 > lower_thr) * 1).T))
    corner_idxes = [br, tl, tr, bl]

    test_image_id, _ = extract_test_image_ids(subj=1)
    image_ids = [test_image_id[idx] for idx in corner_idxes]
    with open(
        "%s/output/clip/%s_vs_%s_corner_image_ids.npy"
        % (args.output_dir, model1, model2),
        "wb",
    ) as f:
        pickle.dump(image_ids, f)

    image_labels = ["%s Better" % model1, "%s Better" % model2, "Both Good", "Both Bad"]

    for i, idx in enumerate(image_ids):
        plt.figure()
        for j, id in enumerate(idx[:16]):
            print(id)
            plt.subplot(4, 4, j + 1)
            try:
                imgIds = coco_train.getImgIds(imgIds=[id])
                # img = coco.loadImgs(imgIds)[0]
                # print(imgIds)
                img = coco_train.loadImgs(imgIds)[0]
            except KeyError:
                imgIds = coco_val.getImgIds(imgIds=[id])
                # img = coco.loadImgs(imgIds)[0]
                # print(imgIds)
                img = coco_val.loadImgs(imgIds)[0]
            I = io.imread(img["coco_url"])
            plt.axis("off")
            plt.imshow(I)
        plt.title(image_labels[i])
        plt.tight_layout()
        plt.savefig("figures/CLIP/sample_corr_images_%s.png" % image_labels[i])


def compare_model_and_brain_performance_on_COCO(subj=1):
    _, test_idx = extract_test_image_ids(subj)
    clip_feature = np.load("%s/features/subj%01d/clip.npy" % (args.output_dir, subj))[test_idx, :]
    clip_text_feature = np.load("%s/features/subj%01d/clip.npy" % (args.output_dir, subj))[test_idx, :]
    scores = clip_feature @ clip_text_feature.T

    sample_corr_clip = compute_sample_corrs("clip", args.output_dir)
    sample_corr_clip_text = compute_sample_corrs("clip_text", args.output_dir)
    
    plt.figure()
    plt.plot(sample_corr_clip, "g")
    plt.plot(sample_corr_clip_text, "b")
    plt.plot(scores, "r")
    return scores



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
    parser.add_argument("--plot_voxel_wise_performance", action="store_true")
    parser.add_argument("--plot_image_wise_performance", action="store_true")
    parser.add_argument("--compare_brain_and_clip_performance", action="store_true")
    parser.add_argument("--weight_analysis", action="store_true")
    parser.add_argument("--mask", default=False, action="store_true")
    parser.add_argument("--roi", type=str)
    args = parser.parse_args()

    # scatter plot per voxels

    if args.plot_voxel_wise_performance:
        model1 = "convnet_res50"
        model2 = "clip_visual_resnet"
        corr_i = load_model_performance(model1, None, args.output_dir, subj=args.subj)
        # corr_t = load_model_performance("clip_text", None, args.output_dir, subj=args.subj)
        corr_j = load_model_performance(model2, None, args.output_dir, subj=args.subj)

        if args.roi is not None:
            colors = np.load(
                "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (args.output_dir, args.subj, args.subj, args.roi)
            )
            if args.mask:
                mask = colors > 0
                plt.figure(figsize=(7, 7))
            else:
                mask = colors > -1
                plt.figure(figsize=(30, 15))
            # Plotting text performance vs image performances

            plt.scatter(corr_i[mask], corr_j[mask], alpha=0.07, c=colors[mask])
        else:
            plt.scatter(corr_i, corr_j, alpha=0.02)

        plt.plot([-0.1, 1], [-0.1, 1], "r")
        plt.xlabel(model1)
        plt.ylabel(model2)
        plt.savefig("figures/CLIP/%s_vs_%s_acc_%s.png" % (model1, model2, args.roi))

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
        from pycocotools.coco import COCO
        import skimage.io as io

        annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
        annFile_val = (
            "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
        )
        coco_train = COCO(annFile_train)
        coco_val = COCO(annFile_val)

        # plot_image_wise_performance("clip", "convnet_res50")
        find_corner_images("clip", "convnet_res50")
        # plot_image_wise_performance("clip", "bert_layer_13")
        find_corner_images("clip", "bert_layer_13")
    
    if args.compare_brain_and_clip_performance:
        compare_model_and_brain_performance_on_COCO(subj=1)

    # trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
    # valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"

    # train_caps = COCO(trainFile)
    # val_caps = COCO(valFile)
