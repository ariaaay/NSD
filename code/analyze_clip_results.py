import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.decomposition import PCA

import torch
import clip

from util.data_util import load_model_performance, extract_test_image_ids
from util.model_config import COCO_cat, roi_name_dict

from extract_clip_features import load_captions

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_sample_performance(model, output_dir, masking="sig", subj=1, measure="corrs"):
    if measure == "corrs":
        from scipy.stats import pearsonr
        metric = pearsonr
    elif measure == "rsq":
        from sklearn.metrics import r2_score
        metric = r2_score
    
    try:
        sample_corrs = np.load(
            "%s/output/clip/%s_sample_%s_%s.npy" % (output_dir, model, measure, masking)
        )
        if len(sample_corrs.shape) == 2:
            sample_corrs = np.array(sample_corrs)[:,0]
            np.save("%s/output/clip/%s_sample_corrs_%s.npy" % (output_dir, model, masking), sample_corrs)
    except FileNotFoundError:
        yhat, ytest = load_model_performance(
            model, output_root=output_dir, measure="pred"
        )
        if masking == "sig":
            pvalues = load_model_performance(
                model, output_root=output_dir, measure="pvalue"
            )
            sig_mask = pvalues <= 0.05

            sample_corrs = [
                metric(ytest[:, sig_mask][i, :], yhat[:, sig_mask][i, :])
                for i in range(ytest.shape[0])
            ]

        else:
            roi = np.load("%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy" % (output_dir, subj, subj, masking))
            roi_mask = roi > 0
            sample_corrs = [
                metric(ytest[:, roi_mask][i, :], yhat[:, roi_mask][i, :])[0]
                for i in range(ytest.shape[0])
            ]
        
        if measure == "corr":
            sample_corrs = np.array(sample_corrs)[:,0]
        np.save(
            "%s/output/clip/%s_sample_%s_%s.npy" % (output_dir, model, measure, masking), sample_corrs
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


def plot_image_wise_performance(model1, model2, masking="sig", measure="corrs"):
    sample_corr1 = compute_sample_performance(model=model1, output_dir=args.output_root, masking=masking, measure=measure)
    sample_corr2 = compute_sample_performance(model=model2, output_dir=args.output_root, masking=masking, measure=measure)
    plt.figure()
    plt.scatter(sample_corr1, sample_corr2, alpha=0.3)
    plt.plot([-0.1, 1], [-0.1, 1], "r")
    plt.xlabel(model1)
    plt.ylabel(model2)
    plt.savefig("figures/CLIP/image_wise_performance/%s_vs_%s_samplewise_%s_%s.png" % (model1, model2, measure, masking))


def find_corner_images(model1, model2, upper_thr=0.5, lower_thr=0.03, masking="sig", measure="corrs"):
    sp1 = compute_sample_performance(model=model1, output_dir=args.output_root, masking=masking, measure=measure)
    sp2 = compute_sample_performance(model=model2, output_dir=args.output_root, masking=masking, measure=measure)
    diff = sp1 - sp2
    indexes = np.argsort(diff) # from where model 2 does the best to where model 1 does the best
    br = indexes[:20] # model2 > 1
    tl = indexes[::-1][:20] # model1 > 2

    best1 = np.argsort(sp1)[::-1][:20]
    best2 = np.argsort(sp2)[::-1][:20]
    worst1 = np.argsort(sp1)[:20]
    worst2 = np.argsort(sp2)[:20]

    tr = [idx for idx in best1 if idx in best2]
    bl = [idx for idx in worst1 if idx in worst2]
    corner_idxes = [br, tl, tr, bl]

    test_image_id, _ = extract_test_image_ids(subj=1)
    corner_image_ids = [test_image_id[idx] for idx in corner_idxes]
    with open(
        "%s/output/clip/%s_vs_%s_corner_image_ids_%s_sample_%s.npy"
        % (args.output_root, model1, model2, masking, measure),
        "wb",
    ) as f:
        pickle.dump(corner_image_ids, f)

    image_labels = ["%s+" % model1, "%s+" % model2, "%s+%s+" % (model1, model2), "%s-%s-" % (model1, model2)]

    for i, idx in enumerate(corner_image_ids):
        plt.figure()
        for j, id in enumerate(idx[:16]):
            # print(id)
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
        # plt.title(image_labels[i])
        plt.tight_layout()
        plt.savefig("figures/CLIP/corner_images/sample_%s_images_%s_%s.png" % (measure, image_labels[i], masking))
        plt.close()


def compare_model_and_brain_performance_on_COCO(subj=1):
    from scipy.stats import pearsonr
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
            # print(logits_per_image.shape)
            probs = logits_per_image.squeeze().softmax(dim=-1).cpu().numpy()
            # print(probs.shape)
            preds.append(probs[i])

    sample_corr_clip = compute_sample_performance("clip", args.output_root)
    sample_corr_clip_text = compute_sample_performance("clip_text", args.output_root)

    plt.figure(figsize=(30, 10))
    plt.plot(sample_corr_clip, "g", alpha=0.3)
    plt.plot(sample_corr_clip_text, "b", alpha=0.3)
    plt.plot(preds, "r", alpha=0.3)
    print(pearsonr(sample_corr_clip, preds)[0])
    print(pearsonr(sample_corr_clip_text, preds)[0])


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
    plt.tight_layout()
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
        plt.savefig("figures/CLIP/voxel_wise_performance/%s_vs_%s_acc_%s.png" % (model1, model2, args.roi))

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
        plot_image_wise_performance("clip", "convnet_res50")
        find_corner_images("clip", "convnet_res50")

        roi_list = list(roi_name_dict.keys())
        roi_list = ["floc-faces", "floc-bodies", "prf-visualrois", "floc-places"]
        for roi in roi_list:
            plot_image_wise_performance("clip", "convnet_res50", masking=roi)
            find_corner_images("clip", "convnet_res50", masking=roi)
            plot_image_wise_performance("clip", "bert_layer_13", masking=roi)
            find_corner_images("clip", "bert_layer_13", masking=roi)

            plot_image_wise_performance("clip", "convnet_res50", masking=roi, measure="rsq")
            find_corner_images("clip", "convnet_res50", masking=roi, measure="rsq")
            plot_image_wise_performance("clip", "bert_layer_13", masking=roi, measure="rsq")
            find_corner_images("clip", "bert_layer_13", masking=roi, measure="rsq")

    if args.compare_brain_and_clip_performance:
        compare_model_and_brain_performance_on_COCO(subj=1)

    if args.coarse_level_semantic_analysis:
        coarse_level_semantic_analysis(subj=1)

    # trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
    # valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"

    # train_caps = COCO(trainFile)
    # val_caps = COCO(valFile)
