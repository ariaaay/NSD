import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA

import torch
import clip

from util.data_util import load_model_performance, extract_test_image_ids
from util.model_config import COCO_cat

from extract_clip_features import load_captions

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_sample_corrs(model, output_dir):
    from scipy.stats import pearsonr

    try:
        sample_corrs = np.load(
            "%s/output/clip/%s_sample_corrs.npy" % (output_dir, model)
        )
    except FileNotFoundError:
        yhat, ytest = load_model_performance(
            model, output_root=output_dir, measure="pred"
        )
        pvalues = load_model_performance(
            model, output_root=output_dir, measure="pvalue"
        )
        sig_mask = pvalues <= 0.05

        sample_corrs = [
            pearsonr(ytest[:, sig_mask][i, :], yhat[:, sig_mask][i, :])
            for i in range(ytest.shape[0])
        ]
        np.save(
            "%s/output/clip/%s_sample_corrs.npy" % (output_dir, model), sample_corrs
        )

    return sample_corrs


def extract_text_scores(word_lists, weight):
    phrase_lists = ["photo of " + w[:-1] for w in word_lists]
    model, _ = clip.load("ViT-B/32", device=device)
    activations = []
    for phrase in phrase_lists:
        text = clip.tokenize([phrase]).to(device)
        with torch.no_grad():
            activations.append(model.encode_text(text).data.numpy())
        
    scores = np.mean(activations.squeeze() @ weight, axis=1)
    print(np.array(word_lists)[np.argsort(scores)[:30]])
    print(np.array(word_lists)[np.argsort(scores)[::-1][:30]])
    return np.array(scores)


def plot_image_wise_performance(model1, model2):
    sample_corr1 = compute_sample_corrs(model=model1, output_dir=args.output_root)
    sample_corr2 = compute_sample_corrs(model=model2, output_dir=args.output_root)

    plt.figure()
    plt.scatter(sample_corr1[:, 0], sample_corr2[:, 0], alpha=0.3)
    plt.plot([-0.1, 1], [-0.1, 1], "r")
    plt.xlabel(model1)
    plt.ylabel(model2)
    plt.savefig("figures/CLIP/%s_vs_%s_samplewise.png" % (model1, model2))


def find_corner_images(model1, model2, upper_thr=0.5, lower_thr=0.03):
    sc1 = np.load("%s/output/clip/%s_sample_corrs.npy" % (args.output_root, model1))[
        :, 0
    ]
    sc2 = np.load("%s/output/clip/%s_sample_corrs.npy" % (args.output_root, model2))[
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
        % (args.output_root, model1, model2),
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
    stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"

    test_image_id, _ = extract_test_image_ids(subj)
    all_images_paths = list()
    all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in test_image_id]

    print("Number of Images: {}".format(len(all_images_paths)))

    captions = [
        load_captions(cid)[0] for cid in test_image_id
    ]  # pick the first caption
    model, preprocess = clip.load("ViT-B/32", device=device)

    preds = list()
    for i, p in enumerate(all_images_paths):
        image = preprocess(Image.open(p)).unsqueeze(0).to(device)
        text = clip.tokenize(captions).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            preds.append(probs[0][i])

    sample_corr_clip = compute_sample_corrs("clip", args.output_root)
    sample_corr_clip_text = compute_sample_corrs("clip_text", args.output_root)

    plt.figure()
    plt.plot(sample_corr_clip, "g", alpha=0.3)
    plt.plot(sample_corr_clip_text, "b", alpha=0.3)
    plt.plot(preds, "r", alpha=0.3)
    plt.savefig("figures/CLIP/model_brain_comparison.png")


def coarse_level_semantic_analysis(subj=1):
    from sklearn.metrics.pairwise import cosine_similarity

    image_supercat = np.load("data/NSD_supcat_feat.npy")
    # image_cat = np.load("data/NSD_cat_feat.npy")
    cocoId_subj = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )
    nsd2coco = np.load("%s/output/NSD2cocoId.npy" % args.output_root)
    img_ind = [list(nsd2coco).index(i) for i in cocoId_subj]
    image_supercat_subsample = image_supercat[img_ind, :]
    max_cat = np.argmax(image_supercat_subsample, axis=1)
    max_cat_order = np.argsort(max_cat)
    sorted_image_supercat = image_supercat_subsample[max_cat_order, :]
    sorted_image_supercat_sim_by_image = cosine_similarity(sorted_image_supercat)
    # image_cat_subsample = image_cat[img_ind,:]
    # sorted_image_cat = image_cat_subsample[np.argsort(max_cat),:]
    models = [
        "clip",
        "clip_text",
        "convnet_res50",
        "bert_layer_13",
        "clip_visual_resnet",
    ]
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(sorted_image_supercat_sim_by_image)
    for i, m in enumerate(models):
        plt.subplot(2, 3, i + 2)
        rdm = np.load("%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, m))
        plt.imshow(rdm)
    plt.savefig("figures/CLIP/coarse_category_RDM_comparison.png")


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
    parser.add_argument("--roi", type=str)
    parser.add_argument("--plot_voxel_wise_performance", default=False, action="store_true")
    parser.add_argument("--plot_image_wise_performance", default=False, action="store_true")
    parser.add_argument(
        "--coarse_level_semantic_analysis", default=False, action="store_true"
    )
    parser.add_argument(
        "--compare_brain_and_clip_performance", default=False, action="store_true"
    )
    parser.add_argument("--weight_analysis", default=False, action="store_true")
    parser.add_argument("--mask", default=False, action="store_true")
    args = parser.parse_args()

    # scatter plot per voxels

    if args.plot_voxel_wise_performance:
        model1 = "convnet_res50"
        model2 = "clip_visual_resnet"
        corr_i = load_model_performance(model1, None, args.output_root, subj=args.subj)
        corr_j = load_model_performance(model2, None, args.output_root, subj=args.subj)

        if args.roi is not None:
            colors = np.load(
                "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (args.output_root, args.subj, args.subj, args.roi)
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
        models = ["clip_text", "clip", "convnet_res50", "clip_visual_resnet", "bert_layer_13"]
        for m in models:
            print(m)
            w = np.load(
                "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
                % (args.output_root, args.subj, m)
            )
            print(w.shape)
            print("NaNs? Finite?:")
            print(np.any(np.isnan(w)))
            print(np.all(np.isfinite(w)))
            pca = PCA(n_components=5)
            pca.fit(w)
            np.save(
                "%s/output/pca/subj%d/clip_pca_components.npy"
                % (args.output_root, args.subj),
                pca.components_,
            )

        
    if args.plot_image_wise_performance:
        # scatter plot by images
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

    if args.coarse_level_semantic_analysis:
        coarse_level_semantic_analysis(subj=1)

    # trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
    # valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"

    # train_caps = COCO(trainFile)
    # val_caps = COCO(valFile)
