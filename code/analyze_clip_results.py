import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import argparse

# from msilib.schema import File
import pickle

import pandas as pd
import seaborn as sns
import nibabel as nib
import numpy as np

# from shinyutils.matwrap import sns, plt, MatWrap as mw
import matplotlib.pyplot as plt
import skimage.io as io

from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA

# import torch
import clip

from util.util import fdr_correct_p
from util.data_util import load_model_performance, extract_test_image_ids
from util.model_config import *

# device = "cuda" if torch.cuda.is_available() else "cpu"

from pycocotools.coco import COCO

# mw.configure(backend="Agg")

# annFile_train = (
#     "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
# )
# annFile_val = (
#     "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
# )
# coco_train = COCO(annFile_train)
# coco_val = COCO(annFile_val)

# annFile_train_caps = (
#     "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
# )
# annFile_val_caps = (
#     "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"
# )
# coco_train_caps = COCO(annFile_train_caps)
# coco_val_caps = COCO(annFile_val_caps)


def compute_sample_performance(model, subj, output_dir, masking="sig", measure="corrs"):
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
            sample_corrs = np.array(sample_corrs)[:, 0]
            np.save(
                "%s/output/clip/%s_sample_corrs_%s.npy" % (output_dir, model, masking),
                sample_corrs,
            )
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
            roi = np.load(
                "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                % (output_dir, subj, subj, masking)
            )
            roi_mask = roi > 0
            sample_corrs = [
                metric(ytest[:, roi_mask][i, :], yhat[:, roi_mask][i, :])
                for i in range(ytest.shape[0])
            ]

        if measure == "corr":
            sample_corrs = np.array(sample_corrs)[:, 0]
        np.save(
            "%s/output/clip/%s_sample_%s_%s.npy"
            % (output_dir, model, measure, masking),
            sample_corrs,
        )

    return sample_corrs


def extract_text_activations(model, word_lists):
    activations = []
    for word in word_lists:
        text = clip.tokenize([word]).to(device)
        with torch.no_grad():
            activations.append(model.encode_text(text).data.numpy())
    return np.array(activations)


def extract_keywords_for_roi(w, roi_name, roi_vals, activations, common_words):
    roi_mask = np.load(
        "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
        % (args.output_root, args.subj, args.subj, roi_name)
    )
    roi_selected_vox = np.zeros((roi_mask.shape))
    for v in roi_vals:
        roi_selected_vox += roi_mask == v
    roi_selected_vox = roi_selected_vox > 0

    roi_w = w[:, roi_selected_vox]

    scores = np.mean(activations.squeeze() @ roi_w, axis=1)
    print(roi_name)
    best_list = list(np.array(common_words)[np.argsort(scores)[::-1][:30]])
    worst_list = list(np.array(common_words)[np.argsort(scores)[:30]])
    print(best_list)
    print(worst_list)
    pickle.dump(
        best_list,
        open(
            "%s/output/clip/roi_maximization/best_in_%s.json"
            % (args.output_root, roi_name),
            "wb",
        ),
    )
    pickle.dump(
        worst_list,
        open(
            "%s/output/clip/roi_maximization/worst_in_%s.json"
            % (args.output_root, roi_name),
            "wb",
        ),
    )


def extract_captions_for_voxel(roi, n=3):
    """
    voxel that are selected by the mask will be assigned integer values
    """
    save_path = "%s/output/clip/roi_maximization" % args.output_root

    try:
        activations = np.load(
            "%s/output/clip/roi_maximization/subj1_caption_activation.npy"
            % args.output_root
        )
        all_captions = pickle.load(
            open(
                "%s/output/clip/roi_maximization/all_captions_subj1.pkl"
                % args.output_root,
                "rb",
            )
        )
    except FileNotFoundError:
        import clip
        import torch
        from util.coco_utils import load_captions

        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_captions = []
        all_coco_ids = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, args.subj)
        )
        model, _ = clip.load("ViT-B/32", device=device)
        activations = []
        for cid in tqdm(all_coco_ids):
            with torch.no_grad():
                captions = load_captions(cid)
                all_captions += captions
                for caption in captions:
                    text = clip.tokenize(caption).to(device)
                    activations.append(model.encode_text(text).cpu().data.numpy())

        np.save(
            "%s/output/clip/roi_maximization/subj1_caption_activation.npy"
            % args.output_root,
            activations,
        )
        pickle.dump(
            all_captions,
            open(
                "%s/output/clip/roi_maximization/all_captions_subj1.pkl"
                % args.output_root,
                "wb",
            ),
        )

    activations = np.array(activations)
    print(activations.shape)

    w = np.load(
        "%s/output/encoding_results/subj%d/weights_clip_whole_brain.npy"
        % (args.output_root, args.subj)
    )

    best_caption_dict = dict()

    roi_mask = np.load(
        "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
        % (args.output_root, args.subj, args.subj, roi)
    )

    try:  # take out zero voxels
        non_zero_mask = np.load(
            "%s/output/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (args.output_root, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        roi_mask = roi_mask[non_zero_mask]
    except FileNotFoundError:
        pass

    if args.roi_value != 0:
        mask = roi_mask == args.roi_value
    else:
        mask = roi_mask > 0
    print(str(sum(mask)) + " voxels for optimization")
    vox_w = w[:, mask]

    vindx = np.arange(sum(mask))

    scores = activations.squeeze() @ vox_w  # of captions x # voxels

    for i in vindx:
        best_caption_dict[str(i)] = list(
            np.array(all_captions)[np.argsort(scores[:, i])[::-1][:n]]
        )

    import json

    with open("%s/max_caption_per_voxel_in_%s.json" % (save_path, roi), "w") as f:
        json.dump(best_caption_dict, f)
    return best_caption_dict


def extract_emb_keywords(embedding, activations, common_words, n=15):
    scores = activations.squeeze() @ embedding
    if len(embedding.shape) > 1:
        scores = np.mean(scores, axis=1)

    best_list = list(np.array(common_words)[np.argsort(scores)[::-1][:n]])
    worst_list = list(np.array(common_words)[np.argsort(scores)[:n]])
    best_list_word_only = [w.split(" ")[-1] for w in best_list]
    worst_list_word_only = [w.split(" ")[-1] for w in worst_list]
    return best_list_word_only, worst_list_word_only


def plot_image_wise_performance(model1, model2, masking="sig", measure="corrs"):
    subjs = np.arange(1, 9)
    plt.figure()

    for subj in subjs:
        sample_corr1 = compute_sample_performance(
            model=model1,
            subj=subj,
            output_dir=args.output_root,
            masking=masking,
            measure=measure,
        )
        sample_corr2 = compute_sample_performance(
            model=model2,
            subj=subj,
            output_dir=args.output_root,
            masking=masking,
            measure=measure,
        )
        plt.subplot(2, 4, subj)
        plt.scatter(sample_corr1, sample_corr2, alpha=0.3)
        plt.plot([-0.1, 1], [-0.1, 1], "r")
        plt.xlabel(model1)
        plt.ylabel(model2)

    plt.savefig(
        "figures/CLIP/image_wise_performance/%s_vs_%s_samplewise_%s_%s_all_subjs.png"
        % (model1, model2, measure, masking)
    )


def get_coco_image(id):
    try:
        img = coco_train.loadImgs([id])[0]
    except KeyError:
        img = coco_val.loadImgs([id])[0]
    I = io.imread(img["coco_url"])
    return I


def get_coco_anns(id):
    try:
        annIds = coco_train.getAnnIds([id])
        anns = coco_train.loadAnns(annIds)
    except KeyError:
        annIds = coco_val.getAnnIds([id])
        anns = coco_val.loadAnns(annIds)

    cats = [ann["category_id"] for ann in anns]
    return cats


def get_coco_caps(id):
    try:
        annIds = coco_train_caps.getAnnIds([id])
        anns = coco_train_caps.loadAnns(annIds)
    except KeyError:
        annIds = coco_val_caps.getAnnIds([id])
        anns = coco_val_caps.loadAnns(annIds)

    caps = [ann["caption"] for ann in anns]
    return caps


def find_corner_images(
    model1, model2, subj, upper_thr=0.5, lower_thr=0.03, masking="sig", measure="corrs"
):
    sp1 = compute_sample_performance(
        model=model1,
        subj=subj,
        output_dir=args.output_root,
        masking=masking,
        measure=measure,
    )
    sp2 = compute_sample_performance(
        model=model2,
        subj=subj,
        output_dir=args.output_root,
        masking=masking,
        measure=measure,
    )
    diff = sp1 - sp2
    indexes = np.argsort(
        diff
    )  # from where model 2 does the best to where model 1 does the best
    br = indexes[:20]  # model2 > 1
    tl = indexes[::-1][:20]  # model1 > 2

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
        "%s/output/clip/%s_vs_%s_corner_image_ids_%s_sample_%s_%s.npy"
        % (args.output_root, model1, model2, masking, measure, subj),
        "wb",
    ) as f:
        pickle.dump(corner_image_ids, f)

    image_labels = [
        "%s+%s-" % (model1, model2),
        "%s+%s-" % (model2, model1),
        "%s+%s+" % (model1, model2),
        "%s-%s-" % (model1, model2),
    ]

    for i, idx in enumerate(corner_image_ids):
        plt.figure()
        for j, id in enumerate(idx[:16]):
            # print(id)
            plt.subplot(4, 4, j + 1)
            I = get_coco_image(id)
            plt.axis("off")
            plt.imshow(I)
        # plt.title(image_labels[i])
        plt.tight_layout()
        plt.savefig(
            "figures/CLIP/corner_images/sample_%s_images_%s_%s_%s.png"
            % (measure, image_labels[i], masking, subj)
        )
        plt.close()


def compare_model_and_brain_performance_on_COCO():
    import torch
    from scipy.stats import pearsonr
    from utils.coco_utils import load_captions

    stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"

    corrs_v, corrs_t = [], []
    for subj in np.arange(1, 9):
        test_image_id, _ = extract_test_image_ids(subj)
        all_images_paths = list()
        all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in test_image_id]

        print("Number of Images: {}".format(len(all_images_paths)))

        captions = [
            load_captions(cid)[0] for cid in test_image_id
        ]  # pick the first caption
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

        sample_corr_clip = compute_sample_performance("clip", i, args.output_root)
        sample_corr_clip_text = compute_sample_performance(
            "clip_text", i, args.output_root
        )

        corrs_v.append(pearsonr(sample_corr_clip, preds)[0])
        corrs_t.append(pearsonr(sample_corr_clip_text, preds)[0])

    fig = plt.figure()
    plt.plot(corrs_v, color="red", label="clip visual")
    plt.plot(corrs_t, color="blue", label="clip text")
    plt.legend()

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
    plt.figure(figsize=(30, 30))
    plt.subplot(2, 3, 1)
    plt.imshow(sorted_image_supercat_sim_by_image)
    plt.colorbar()
    for i, m in enumerate(models):
        plt.subplot(2, 3, i + 2)
        rdm = np.load("%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, m))
        plt.imshow(rdm[max_cat_order, :][:, max_cat_order])
        r = np.corrcoef(rdm.flatten(), sorted_image_supercat_sim_by_image.flatten())[
            1, 1
        ]
        plt.title("%s (r=%.2g)" % (m, r))
        plt.colorbar()
    plt.tight_layout()
    plt.savefig("figures/CLIP/coarse_category_RDM_comparison.png")


def sample_level_semantic_analysis(
    subj=1, model1="clip", model2="resnet50_bottleneck", print_distance=False
):
    cocoId_subj = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )
    rdm1 = np.load("%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1))
    rdm2 = np.load("%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2))

    diff1 = rdm1 - rdm2  # close in 1, far in 2
    diff2 = rdm2 - rdm1
    ind_1 = np.unravel_index(np.argsort(diff1, axis=None), diff1.shape)
    ind_2 = np.unravel_index(np.argsort(diff2, axis=None), diff2.shape)

    # b/c symmetry of RDM, every two pairs are the same

    trial_id_pair_1 = [(ind_1[0][::-1][i], ind_1[1][::-1][i]) for i in range(0, 20, 2)]
    trial_id_pair_2 = [(ind_2[0][::-1][i], ind_2[1][::-1][i]) for i in range(0, 20, 2)]

    plt.figure(figsize=(10, 30))
    for i in range(10):
        plt.subplot(10, 2, i * 2 + 1)
        id = cocoId_subj[trial_id_pair_1[i][0]]
        I = get_coco_image(id)
        plt.imshow(I)
        if print_distance:
            plt.title(
                "Diff:%.2f; Sim1:%.2f; Sim2:%.2f"
                % (
                    diff1[trial_id_pair_1[i]],
                    rdm1[trial_id_pair_1[i]],
                    rdm2[trial_id_pair_1[i]],
                )
            )
        plt.axis("off")

        plt.subplot(10, 2, i * 2 + 2)
        id = cocoId_subj[trial_id_pair_1[i][1]]
        I = get_coco_image(id)
        plt.imshow(I)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        "figures/CLIP/RDM_max/RDM_max_images_close_in_%s_far_in_%s.png"
        % (model1, model2)
    )

    for i in range(10):
        plt.subplot(10, 2, i * 2 + 1)
        id = cocoId_subj[trial_id_pair_2[i][0]]
        I = get_coco_image(id)
        plt.imshow(I)
        if print_distance:
            plt.title(
                "Diff:%.2f; Sim1:%.2f; Sim2:%.2f"
                % (
                    diff2[trial_id_pair_2[i]],
                    rdm1[trial_id_pair_2[i]],
                    rdm2[trial_id_pair_2[i]],
                )
            )
        plt.axis("off")

        plt.subplot(10, 2, i * 2 + 2)
        id = cocoId_subj[trial_id_pair_2[i][1]]
        I = get_coco_image(id)
        plt.imshow(I)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        "figures/CLIP/RDM_max/RDM_max_images_close_in_%s_far_in_%s.png"
        % (model2, model1)
    )


def image_level_scatter_plot(model1="clip", model2="resnet50_bottleneck", subj=1, i=1):
    from compute_feature_rdm import computeRSM
    from scipy.stats import pearsonr
    from util.util import zscore

    # cocoId_subj = np.load(
    #     "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    # )
    try:
        rsm1 = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1)
        )
    except FileNotFoundError:
        rsm1 = computeRSM(model1, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model1),
            rsm1,
        )

    try:
        rsm2 = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2)
        )
    except FileNotFoundError:
        rsm2 = computeRSM(model2, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model2),
            rsm2,
        )

    tmp = np.ones(rsm1.shape)
    triu_flag = np.triu(tmp, k=1).astype(bool)
    # plt.figure(figsize=(20, 20))
    plt.box(False)
    # subsample 1000 point for plotting
    sampling_idx = np.random.choice(len(rsm1[triu_flag]), size=10000, replace=False)
    x = zscore(rsm1[triu_flag][sampling_idx])
    y = zscore(rsm2[triu_flag][sampling_idx])
    plt.subplot(2, 3, i)
    plt.scatter(x, y, alpha=0.2, s=2, label=model2)
    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(0, 1, num=100)
    plt.plot(xseq, a + b * xseq, lw=1, color="k", label=model2 + "_fit")

    r = pearsonr(rsm1[triu_flag][sampling_idx], rsm2[triu_flag][sampling_idx])
    # plt.xlim(0, 1)
    # plt.ylim(-0.25, 1)
    # ax = plt.gca()
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # plt.axis("off")
    print(model2 + " r: " + str(r[0]))
    plt.legend()
    # rdm1[~triu_flag] = 0
    # rdm2[~triu_flag] = 0
    # ind = np.unravel_index(np.argsort(rdm1, axis=None), x.shape)


def category_based_similarity_analysis(model, threshold, subj=1):
    from compute_feature_rdm import computeRSM
    from scipy.stats import pearsonr
    from util.util import zscore

    try:
        rsm = np.load(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model)
        )
    except FileNotFoundError:
        rsm = computeRSM(model1, args.feature_dir)
        np.save(
            "%s/output/rdms/subj%02d_%s.npy" % (args.output_root, subj, model),
            rsm,
        )

    tmp = np.ones(rsm.shape)
    triu_flag = np.triu(tmp, k=1).astype(bool)

    from featureprep.feature_prep import get_preloaded_features

    stimulus_list = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
    )
    COCO_cat_feat = get_preloaded_features(
        1,
        stimulus_list,
        "cat",
        features_dir="%s/features" % args.output_root,
    )
    print(COCO_cat_feat.shape)
    person_flag = COCO_cat_feat[:, 0] > threshold
    person_n = np.sum(person_flag)
    cluster_flag = np.outer(person_flag, person_flag).astype(bool)
    within_flag = cluster_flag * triu_flag
    cross_flag = ~cluster_flag * triu_flag

    within_cluster_score = np.mean(rsm[within_flag])
    cross_cluster_score = np.mean(rsm[cross_flag])

    # print(within_cluster_score)
    # print(cross_cluster_score)

    return within_cluster_score / (cross_cluster_score + within_cluster_score), person_n


def make_roi_df(roi_names, subjs, update=False):
    if update:
        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
    else:
        df = pd.DataFrame()

    for subj in subjs:
        try:
            subj_df = pd.read_csv(
                "%s/output/clip/performance_by_roi_df_subj%02d.csv"
                % (args.output_root, subj)
            )
        except FileNotFoundError:
            subj_df = pd.DataFrame(
                columns=[
                    "voxel_idx",
                    "var_clip",
                    "var_resnet",
                    "uv_clip",
                    "uv_resnet",
                    "uv_diff",
                    # "uv_diff_nc",
                    "joint",
                    "subj",
                    "nc",
                ]
                + roi_names
            )

            joint_var = load_model_performance(
                model=[
                    "resnet50_bottleneck_clip_visual_resnet",
                    "clip_visual_resnet_resnet50_bottleneck",
                ],
                output_root=args.output_root,
                subj=subj,
                measure="rsq",
            )
            clip_var = load_model_performance(
                model="clip_visual_resnet",
                output_root=args.output_root,
                subj=subj,
                measure="rsq",
            )
            resnet_var = load_model_performance(
                model="resnet50_bottleneck",
                output_root=args.output_root,
                subj=subj,
                measure="rsq",
            )
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
                % (args.output_root, subj, subj)
            )

            u_clip = joint_var - resnet_var
            u_resnet = joint_var - clip_var

            for i in tqdm(range(len(joint_var))):
                vd = dict()
                vd["voxel_idx"] = i
                vd["var_clip"] = clip_var[i]
                vd["var_resnet"] = resnet_var[i]
                vd["uv_clip"] = u_clip[i]
                vd["uv_resnet"] = u_resnet[i]
                vd["uv_diff"] = u_clip[i] - u_resnet[i]
                # vd["uv_diff_nc"] = u_clip[i] / (nc[i]/100) - u_resnet[i] / (nc[i]/100)
                vd["joint"] = joint_var[i]
                vd["nc"] = nc[i]
                vd["subj"] = subj
                subj_df = subj_df.append(vd, ignore_index=True)

            cortical_mask = np.load(
                "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
                % (args.output_root, subj, subj)
            )

            for roi_name in roi_names:
                if roi_name == "language":
                    lang_ROI = np.load(
                        "%s/output/voxels_masks/language_ROIs.npy" % args.output_root,
                        allow_pickle=True,
                    ).item()
                    roi_volume = lang_ROI["subj%02d" % subj]
                    roi_volume = np.swapaxes(roi_volume, 0, 2)

                else:
                    roi = nib.load(
                        "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/subj%02d/func1pt8mm/roi/%s.nii.gz"
                        % (subj, roi_name)
                    )
                    roi_volume = roi.get_fdata()
                roi_vals = roi_volume[cortical_mask]
                roi_label_dict = roi_name_dict[roi_name]
                roi_label_dict[-1] = "non-cortical"
                roi_label_dict["-1"] = "non-cortical"
                try:
                    roi_labels = [roi_label_dict[int(i)] for i in roi_vals]
                except KeyError:
                    roi_labels = [roi_label_dict[str(int(i))] for i in roi_vals]
                # print(np.array(list(df["voxel_idx"])).astype(int))
                subj_df[roi_name] = np.array(roi_labels)[
                    np.array(list(subj_df["voxel_idx"])).astype(int)
                ]

            subj_df.to_csv(
                "%s/output/clip/performance_by_roi_df_subj%02d.csv"
                % (args.output_root, subj)
            )
        df = pd.concat([df, subj_df])

    df.to_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
    return df


def compute_ci_cutoff(n, ci=0.95):
    ci_ends = np.array([0.0 + (1 - ci) / 2.0, 1 - (1 - ci) / 2.0])
    ci_ind = (ci_ends * n).astype(np.int32)
    return ci_ind


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
        "--feature_dir", default="/user_data/yuanw3/project_outputs/NSD/features"
    )
    parser.add_argument("--roi", type=str)
    parser.add_argument(
        "--plot_voxel_wise_performance", default=False, action="store_true"
    )
    parser.add_argument(
        "--plot_image_wise_performance", default=False, action="store_true"
    )
    parser.add_argument(
        "--coarse_level_semantic_analysis", default=False, action="store_true"
    )
    parser.add_argument(
        "--sample_level_semantic_analysis", default=False, action="store_true"
    )
    parser.add_argument(
        "--compare_brain_and_clip_performance", default=False, action="store_true"
    )
    parser.add_argument(
        "--compare_to_human_judgement", default=False, action="store_true"
    )
    parser.add_argument(
        "--performance_analysis_by_roi", default=False, action="store_true"
    )
    parser.add_argument(
        "--performance_analysis_by_roi_subset", default=False, action="store_true"
    )
    parser.add_argument(
        "--image_level_scatter_plot", default=False, action="store_true"
    )
    parser.add_argument(
        "--category_based_similarity_analysis", default=False, action="store_true"
    )
    parser.add_argument("--rerun_df", default=False, action="store_true")
    parser.add_argument("--weight_analysis", default=False, action="store_true")
    parser.add_argument(
        "--extract_keywords_for_roi", default=False, action="store_true"
    )
    parser.add_argument("--group_analysis_by_roi", default=False, action="store_true")
    parser.add_argument("--summary_statistics", default=False, action="store_true")
    parser.add_argument("--group_weight_analysis", default=False, action="store_true")
    parser.add_argument("--clip_rsq_across_subject", default=False, action="store_true")
    parser.add_argument("--extract_captions_for_roi", default=None, type=str)
    parser.add_argument(
        "--process_bootstrap_results", default=False, action="store_true"
    )
    parser.add_argument("--cross_model_comparison", default=False, action="store_true")
    parser.add_argument("--nc_scatter_plot", default=False, action="store_true")
    parser.add_argument("--mask", default=False, action="store_true")
    parser.add_argument("--roi_value", default=0, type=int)
    args = parser.parse_args()

    if args.process_bootstrap_results:

        joint_rsq = np.load(
            "%s/output/bootstrap/subj%01d/rsq_dist_clip_visual_resnet_resnet50_bottleneck_whole_brain.npy"
            % (args.output_root, args.subj)
        )
        clipv_rsq = np.load(
            "%s/output/bootstrap/subj%01d/rsq_dist_clip_visual_resnet_whole_brain.npy"
            % (args.output_root, args.subj)
        )
        resnet_rsq = np.load(
            "%s/output/bootstrap/subj%01d/rsq_dist_resnet50_bottleneck_whole_brain.npy"
            % (args.output_root, args.subj)
        )

        fdr_p = fdr_correct_p(resnet_rsq)
        print(np.sum(fdr_p[1] < 0.05))
        np.save(
            "output/ci_threshold/resnet50_bottleneck_unique_var_fdr_p_subj%01d.npy"
            % args.subj,
            fdr_p,
        )

        fdr_p = fdr_correct_p(clipv_rsq)
        print(np.sum(fdr_p[1] < 0.05))
        np.save(
            "output/ci_threshold/clip_visual_resnet_fdr_p_subj%01d.npy" % args.subj,
            fdr_p,
        )

        clip_unique_var = joint_rsq - resnet_rsq
        resnet_unique_var = joint_rsq - clipv_rsq
        del joint_rsq
        del resnet_rsq
        fdr_p = fdr_correct_p(clip_unique_var)
        print(np.sum(fdr_p[1] < 0.05))
        np.save(
            "output/ci_threshold/clip_unique_var_fdr_p_subj%01d.npy" % args.subj, fdr_p
        )

        fdr_p = fdr_correct_p(resnet_unique_var)
        print(np.sum(fdr_p[1] < 0.05))
        np.save(
            "output/ci_threshold/resnet_unique_var_fdr_p_subj%01d.npy" % args.subj,
            fdr_p,
        )

        for model in ["clip", "clip_text"]:
            rsq = np.load(
                "%s/output/bootstrap/subj%01d/rsq_dist_%s_whole_brain.npy"
                % (args.output_root, args.subj, model)
            )
            fdr_p = fdr_correct_p(rsq)
            print(np.sum(fdr_p[1] < 0.05))
            np.save(
                "output/ci_threshold/%s_fdr_p_subj%01d.npy" % (model, args.subj), fdr_p
            )

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
        plt.savefig(
            "figures/CLIP/voxel_wise_performance/%s_vs_%s_acc_%s.png"
            % (model1, model2, args.roi)
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

        m1 = "clip"
        m2 = "resnet50_bottleneck"

        plot_image_wise_performance(m1, m2, measure="rsq")

        for subj in np.arange(1, 9):
            find_corner_images("clip", "convnet_res50", subj=subj, measure="rsq")

        roi_list = list(roi_name_dict.keys())
        roi_list = ["floc-faces", "floc-bodies", "prf-visualrois", "floc-places"]

        for roi in roi_list:
            plot_image_wise_performance(m1, m2, masking=roi, measure="rsq")

            for subj in np.arange(1, 9):
                find_corner_images(m1, m2, masking=roi, measure="rsq", subj=subj)

    if args.compare_brain_and_clip_performance:
        compare_model_and_brain_performance_on_COCO()

    if args.coarse_level_semantic_analysis:
        coarse_level_semantic_analysis(subj=1)

    if args.image_level_scatter_plot:
        models = [
            "clip",
            "YFCC_clip",
            "YFCC_slip",
            "YFCC_simclr",
            "resnet50_bottleneck",
        ]
        plt.figure(figsize=(10, 10))
        for m, model in enumerate(models):
            image_level_scatter_plot(model1="bert_layer_13", model2=model, i=m + 1)
        # plt.legend()
        plt.savefig("figures/CLIP/manifold_distance/bert_vs_others.png", dpi=400)

    if args.extract_keywords_for_roi:
        with open(
            "%s/output/clip/roi_maximization/1000eng.txt" % args.output_root
        ) as f:
            out = f.readlines()
        common_words = ["photo of " + w[:-1] for w in out]
        try:
            activations = np.load(
                "%s/output/clip/roi_maximization/1000eng_activation.npy"
                % args.output_root
            )
        except FileNotFoundError:
            from nltk.corpus import wordnet
            import clip
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load("ViT-B/32", device=device)
            activations = extract_text_activations(model, common_words)
            np.save(
                "%s/output/clip/roi_maximization/1000eng_activation.npy"
                % args.output_root,
                activations,
            )

        w = np.load(
            "%s/output/encoding_results/subj%d/weights_clip_whole_brain.npy"
            % (args.output_root, args.subj)
        )

        extract_keywords_for_roi(w, "floc-faces", [2, 3], activations, common_words)
        extract_keywords_for_roi(w, "floc-bodies", [1, 2, 3], activations, common_words)
        extract_keywords_for_roi(
            w, "floc-places", [1, 2, 3, 4], activations, common_words
        )

    if args.extract_captions_for_roi is not None:
        extract_captions_for_voxel(args.extract_captions_for_roi)

    if args.sample_level_semantic_analysis:
        from pycocotools.coco import COCO
        import skimage.io as io

        annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
        annFile_val = (
            "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
        )
        coco_train = COCO(annFile_train)
        coco_val = COCO(annFile_val)

        sample_level_semantic_analysis(
            subj=args.subj,
            model1="clip",
            model2="resnet50_bottleneck",
            print_distance=True,
        )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="clip", model2="bert_layer_13"
        # )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="visual_layer_11", model2="resnet50_bottleneck"
        # )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="clip", model2="visual_layer_1"
        # )
        # sample_level_semantic_analysis(
        #     subj=args.subj, model1="clip", model2="clip_text"
        # )

    if args.compare_to_human_judgement:
        human_emb_path = (
            "data/human_similarity_judgement/spose_embedding_49d_sorted.txt"
        )
        word_path = "data/human_similarity_judgement/unique_id.txt"

        human_emb = np.loadtxt(human_emb_path)
        emb_label = np.loadtxt(word_path, dtype="S")
        emb_label = [w.decode("utf-8") for w in emb_label]

        # checked that the label and emb matches
        from util.model_config import COCO_cat, COCO_super_cat

        print(len(COCO_cat))
        count = 0
        COCO_HJ_overlap = []
        # all_coco = COCO_cat + COCO_super_cat
        for w in COCO_cat:
            if "_" in w:
                word = "_".join(w.split(" "))
            else:
                word = w

            if word in emb_label:
                count += 1
                COCO_HJ_overlap.append(w)

        print(count)
        # print(COCO_HJ_overlap)
        # ['car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        # 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        # 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'tie', 'suitcase',
        # 'frisbee', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle',
        # 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        # 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair',
        # 'couch', 'bed', 'toilet', 'laptop', 'keyboard', 'microwave', 'oven',
        # 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']

        # compare clip, bert, and human judgement
        # clip
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_features = []
        for word in COCO_HJ_overlap:
            with torch.no_grad():
                expression = "a photo of " + word
                text = clip.tokenize(expression).to(device)
                emb = clip_model.encode_text(text).cpu().data.numpy()
                clip_features.append(emb)
        clip_features = np.array(clip_features).squeeze()
        rsm_clip = np.corrcoef(clip_features)

        # bert
        bert_features = []
        from transformers import BertTokenizer, BertModel

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )

        bert_model.eval()
        for word in COCO_HJ_overlap:
            text = "[CLS] a photo of " + word + " [SEP]"
            tokenized_text = bert_tokenizer.tokenize(text)
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                outputs = bert_model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
                # print(len(hidden_states[-1][0][4]))
                # print(hidden_states[-1][0][4])
                # print(hidden_states)
                # print(hidden_states.shape)
                bert_features.append(
                    hidden_states[-1][0][4].numpy()
                )  # embedding of the word, from first layer of bert
        rsm_bert = np.corrcoef(bert_features)

        # human judgement model
        hj_features = []
        for word in COCO_HJ_overlap:
            ind = list(emb_label).index(word)
            hj_features.append(human_emb[ind])
        hj_features = np.array(hj_features)
        rsm_hj = np.corrcoef(hj_features)

        print(np.corrcoef(rsm_hj.flatten(), rsm_clip.flatten()))
        print(np.corrcoef(rsm_hj.flatten(), rsm_bert.flatten()))
        print(np.corrcoef(rsm_bert.flatten(), rsm_clip.flatten()))

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(rsm_clip)
        plt.colorbar()
        plt.title("CLIP")

        plt.subplot(1, 3, 2)
        plt.imshow(rsm_bert)
        plt.colorbar()
        plt.title("BERT")

        plt.subplot(1, 3, 3)
        plt.imshow(rsm_hj)
        plt.colorbar()
        plt.title("Human Behavior")

        plt.tight_layout()

        plt.savefig("figures/CLIP/human_judgement_rsm_comparison_bert13.png")

    if args.performance_analysis_by_roi:
        sns.set(style="whitegrid", font_scale=4.5)

        roi_names = list(roi_name_dict.keys())
        if not args.rerun_df:
            df = pd.read_csv(
                "%s/output/clip/performance_by_roi_df.csv" % args.output_root
            )
        else:
            df = make_roi_df(roi_names, subjs=np.arange(1, 9))

        for roi_name in roi_names:
            plt.figure(figsize=(max(len(roi_name_dict[roi_name].values()) / 4, 50), 30))
            ax = sns.boxplot(
                x=roi_name,
                y="uv_diff",
                data=df,
                dodge=True,
                order=list(roi_name_dict[roi_name].values()),
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.ylabel("Difference in Unique Var.")
            plt.savefig("figures/CLIP/performances_by_roi/uv_diff_%s.png" % roi_name)

        # for roi_name in roi_names:
        #     plt.figure(figsize=(max(len(roi_name_dict[roi_name].values()) / 4, 50), 30))
        #     ax = sns.boxplot(
        #         x=roi_name,
        #         y="uv_diff",
        #         data=df,
        #         dodge=True,
        #         order=list(roi_name_dict[roi_name].values()),
        #     )
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        #     plt.ylabel("Difference in Unique Var. (NC)")
        #     plt.savefig("figures/CLIP/performances_by_roi/uv_nc_diff_%s.png" % roi_name)

    if args.performance_analysis_by_roi_subset:
        roa_list = [
            ("prf-visualrois", "V1v"),
            ("prf-visualrois", "h4v"),
            ("floc-places", "RSC"),
            ("floc-places", "PPA"),
            ("floc-places", "OPA"),
            ("floc-bodies", "EBA"),
            ("floc-faces", "FFA-1"),
            ("floc-faces", "FFA-2"),
            ("HCP_MMP1", "TPOJ1"),
            ("HCP_MMP1", "TPOJ2"),
            ("HCP_MMP1", "TPOJ3"),
            ("language", "AG"),
        ]
        axis_labels = [v for _, v in roa_list]
        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
        new_df = pd.DataFrame()
        for i, (roi_name, roi_lab) in enumerate(roa_list):
            roi_df = df[df[roi_name] == roi_lab].copy()
            roi_df["roi_labels"] = roi_lab
            roi_df["roi_names"] = roi_name
            new_df = pd.concat((new_df, roi_df))

        # plt.figure(figsize=(12, 5))
        # ax = sns.boxplot(
        #     x="roi_labels",
        #     y="uv_diff",
        #     hue="roi_names",
        #     data=new_df,
        #     dodge=False,
        #     order=axis_labels,
        # )
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # plt.ylabel("Difference in Unique Var. (NC)")
        # plt.xlabel("ROIs")
        # plt.legend([],[], frameon=False)
        # plt.savefig("figures/CLIP/performances_by_roi/uv_diff_roi_subset.png")

        # plt.figure(figsize=(12, 5))
        # ax = sns.boxplot(
        #     x="roi_labels",
        #     y="uv_diff_nc",
        #     hue="roi_names",
        #     data=new_df,
        #     dodge=False,
        #     order=axis_labels,
        # )
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # plt.xlabel("ROIs")
        # plt.ylabel("Difference in Unique Var. (NC)")
        # plt.legend([],[], frameon=False)
        # plt.savefig("figures/CLIP/performances_by_roi/uv_nc_diff_roi_subset.png")

        # plt.figure()
        # sns.relplot(x="uv_resnet", y="uv_clip", data=new_df, alpha=0.5)
        # plt.plot([-0.08, 0.3], [-0.08, 0.3], linewidth=1, color="red")
        # plt.ylabel("CLIP")
        # plt.xlabel("ResNet")
        # plt.savefig("figures/CLIP/performances_by_roi/unique_var_roi_subset.png")

        # plt.figure()
        # sns.relplot(x="var_resnet", y="var_clip", data=new_df, alpha=0.5)
        # plt.plot([-0.05, 0.8], [-0.05, 0.8], linewidth=1, color="red")
        # plt.ylabel("CLIP")
        # plt.xlabel("ResNet")
        # plt.savefig("figures/CLIP/performances_by_roi/var_roi_subset.png")

        fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(15, 6))
        for roi, ax in zip(axis_labels, axes.T.flatten()):
            # plt.subplot(3, 4, i+1)
            sns.scatterplot(
                x="uv_resnet",
                y="uv_clip",
                data=new_df[new_df["roi_labels"] == roi],
                alpha=0.5,
                ax=ax,
            )
            sns.lineplot([-0.08, 0.25], [-0.08, 0.25], linewidth=1, color="red", ax=ax)
            ax.set_title(roi)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            # plt.gca().title.set_text(roi)
        fig.supylabel("Unique Var. of " + r"$ResNet_{CLIP}$", size=18)
        fig.supxlabel("Unique Var. of " + r"$ResNet_{ImageNet}$", size=18)

        plt.tight_layout()
        plt.savefig(
            "figures/CLIP/performances_by_roi/unique_var_scatterplot_by_roi.png"
        )

        fig.supylabel("Unique Var. of " + r"$ResNet_{CLIP}$", size=20)
        fig.supxlabel("Unique Var. of " + r"$ResNet_{ImageNet}$", size=20)
        plt.savefig(
            "figures/CLIP/performances_by_roi/unique_var_scatterplot_by_roi_poster.png"
        )

    if args.group_analysis_by_roi:
        from scipy.stats import ttest_rel
        from util.util import ztransform

        roa_list = [
            ("floc-bodies", "EBA"),
            ("floc-faces", "FFA-1"),
            ("floc-places", "RSC"),
            ("floc-words", "VWFA-1"),
            ("HCP_MMP1", "MST"),
            ("HCP_MMP1", "MT"),
            ("HCP_MMP1", "PH"),
            ("HCP_MMP1", "TPOJ1"),
            ("HCP_MMP1", "TPOJ2"),
            ("HCP_MMP1", "TPOJ3"),
            ("HCP_MMP1", "PGp"),
            ("HCP_MMP1", "V4t"),
            ("HCP_MMP1", "FST"),
            ("Kastner2015", "TO1"),
            ("Kastner2015", "TO2"),
            ("language", "AG"),
            ("language", "ATL"),
            ("prf-visualrois", "V1v"),
        ]

        # roa_list = []
        # roi_names = list(roi_name_dict.keys())
        # for roi_name in roi_names:
        #     if df[roi]

        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
        subjs = np.arange(1, 9)
        roi_by_subj_mean_clip = np.zeros((8, len(roa_list)))
        roi_by_subj_mean_resnet = np.zeros((8, len(roa_list)))
        for s, subj in enumerate(subjs):
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
                % (args.output_root, subj, subj)
            )
            varc = df[df["subj"] == subj]["var_clip"] / (nc[nc >= 10] / 100)
            varr = df[df["subj"] == subj]["var_resnet"] / (nc[nc >= 10] / 100)
            tmp_c = ztransform(varc)
            tmp_r = ztransform(varr)

            means_c, means_r = [], []
            for i, (roi_name, roi_lab) in enumerate(roa_list):
                roiv = df[roi_name] == roi_lab
                roi_by_subj_mean_clip[s, i] = np.mean(tmp_c[roiv])
                roi_by_subj_mean_resnet[s, i] = np.mean(tmp_r[roiv])

        stats = ttest_rel(
            roi_by_subj_mean_clip,
            roi_by_subj_mean_resnet,
            axis=0,
            nan_policy="propagate",
            alternative="two-sided",
        )
        print(stats)
        results = {}
        for i, r in enumerate(roa_list):
            results[r] = (stats[0][i], stats[1][i])
        for k, v in results.items():
            print(k, v)
        # print(roa_list)

    if args.summary_statistics:
        roi_names = list(roi_name_dict.keys())
        df = pd.read_csv(
            "%s/output/clip/performance_by_roi_df_nc_corrected.csv" % args.output_root
        )
        for roi_name in roi_names:
            sns.set(style="whitegrid", font_scale=4.5)
            plt.figure(figsize=(50, 20))
            ax = sns.boxplot(
                x=roi_name,
                y="var_resnet",
                data=df,
                dodge=True,
                order=list(roi_name_dict[roi_name].values()),
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig("figures/CLIP/performances_by_roi/var_resnet_%s.png" % roi_name)

            plt.figure(figsize=(50, 20))
            ax = sns.boxplot(
                x=roi_name,
                y="var_clip",
                data=df,
                dodge=True,
                order=list(roi_name_dict[roi_name].values()),
            )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig("figures/CLIP/performances_by_roi/var_clip_%s.png" % roi_name)

    if args.clip_rsq_across_subject:
        means = []
        for subj in range(8):
            subj_var = clip_var = load_model_performance(
                model="clip", output_root=args.output_root, subj=subj + 1, measure="rsq"
            )
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
                % (args.output_root, subj + 1, subj + 1)
            )
            means.append(
                np.mean(subj_var / (nc / 100), where=np.isfinite(subj_var / (nc / 100)))
            )
        print(means)

    if args.nc_scatter_plot:
        # nc_array, var_array = [], []
        # plt.figure(figsize=(10, 10))
        # for subj in np.arange(8) + 1:
        #     clip_var = load_model_performance(
        #         model="clip",
        #         output_root=args.output_root,
        #         subj=subj,
        #         measure="rsq",
        #     )
        #     nc = np.load(
        #             "%s/output/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
        #             % (args.output_root, subj, subj)
        #         ) / 100
        #     nc_array += list(nc)
        #     var_array += list(clip_var)

        #     plt.subplot(4,2, subj)
        #     sns.scatterplot(x=nc, y=clip_var, alpha=0.5, size=0.5)
        #     sns.lineplot([-0.05, 1.05], [-0.05, 1.05], linewidth=1, color="red")
        #     plt.title("subj %d" % subj)
        #     plt.ylabel("Model performance")
        #     plt.xlabel("Noise ceiling")
        # plt.tight_layout()
        # plt.savefig("figures/CLIP/var_clip_vs_nc_per_subj.png")

        # plt.figure()
        # sns.scatterplot(x=nc_array, y=var_array, alpha=0.5)
        # sns.lineplot([-0.05, 1.05], [-0.05, 1.05], linewidth=1, color="red")
        # plt.savefig("figures/CLIP/var_clip_vs_nc.png")

        df = pd.read_csv("%s/output/clip/performance_by_roi_df.csv" % args.output_root)
        df["nc"] = df["nc"] / 100
        # percentiles = [75, 90, 95, 99]
        # markers = ["*", "+", "x", "o"]

        # sns.set_theme(style="whitegrid")
        # plt.figure(figsize=(5,10))
        # labels = ["(0, 0.1]", "(0.1, 0.2]", "(0.2, 0.3]", "(0.3, 0.4]", "(0.4, 0.5]", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]"]
        # df5['nc_bins'] = pd.cut(df5['nc'], 9, precision=1, labels=labels)
        # df5['perc_nc'] = df5["var_clip"] / df5["nc"]

        # # df5 = df5[df5["nc"]>0.1]

        # # sns.relplot(x="nc", y="perc_nc", data=df5, alpha=0.5)
        # # sns.lineplot([-0.05, 1.05], [-0.05, 1.05], linewidth=1, color="red", label="Noise Ceiling")

        # sns.boxplot(data=df5, x="nc_bins", y="perc_nc")
        # plt.ylim((-0.1, 1.1))
        # plt.xticks(rotation = 45)

        # # n = len(df5["nc"])
        # # for i, p in enumerate(percentiles):
        # #     px = np.percentile(df["nc"], p)
        # #     py = np.percentile(df["var_clip"], p)
        # #     # print(px, py)
        # #     plt.scatter(x=px, y=py, marker=markers[i], s=100, color="red", label="%d%% (n=%d)" % (p, (100-p)*n/100))
        # plt.xlabel("Noise Ceiling")
        # plt.ylabel("Model Performance as % in noise ceiling")
        # # plt.legend()
        # plt.savefig("figures/CLIP/var_clip_vs_nc_subj5.png")
        import matplotlib as mpl
        import matplotlib.pylab as plt

        # PLOTTING SUBJ5
        plt.figure()
        df5 = df[df["subj"] == 5]
        sns.lineplot(
            [-0.05, 1], [-0.05, 1], linewidth=2, color="red", label="noise ceiling"
        )
        sns.lineplot(
            [-0.05, 1],
            [-0.05, 0.85],
            linewidth=2,
            color="orange",
            linestyle="--",
            label="85% noise ceiling",
        )

        plt.hist2d(
            df5["nc"],
            df5["var_clip"],
            bins=100,
            norm=mpl.colors.LogNorm(),
            cmap="magma",
        )
        plt.colorbar()
        plt.xlabel("Noise Ceiling")
        plt.ylabel("Model Performance $(R^2)$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/CLIP/var_clip_vs_nc_subj5_2dhist.png")

        plt.figure()
        sns.lineplot(
            [-0.05, 1], [-0.05, 1], linewidth=2, color="red", label="noise ceiling"
        )
        sns.lineplot(
            [-0.05, 1],
            [-0.05, 0.85],
            linewidth=2,
            color="orange",
            linestyle="--",
            label="85% noise ceiling",
        )

        plt.hist2d(df5["nc"], df5["var_resnet"], bins=100, norm=mpl.colors.LogNorm())

        plt.colorbar()
        plt.xlabel("Noise Ceiling")
        plt.ylabel("Model Performance $(R^2)$")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig("figures/CLIP/var_rn_vs_nc_subj5_2dhist.png")

        plt.figure(figsize=(5, 8))
        plt.subplot(2, 1, 1)
        sns.lineplot([-0.05, 1], [-0.05, 1], linewidth=1, color="red")

        plt.hist2d(
            df5["var_resnet"], df5["var_clip"], bins=100, norm=mpl.colors.LogNorm()
        )

        plt.colorbar()
        plt.xlabel("$ResNet_{ImageNet}$", size=20)
        plt.ylabel("$ResNet_{CLIP}$", size=20)
        plt.xlim(-0.05, 0.9)
        plt.ylim(-0.05, 0.9)
        plt.grid(True)
        plt.title("Model Performance $(R^2)$", size=24)

        plt.subplot(2, 1, 2)
        sns.lineplot([-0.1, 1], [-0.1, 1], linewidth=1, color="red")
        plt.hist2d(
            df5["uv_resnet"],
            df5["uv_clip"],
            bins=100,
            norm=mpl.colors.LogNorm(),
            cmap="magma",
        )

        plt.colorbar()
        plt.xlabel("$ResNet_{ImageNet}$", size=20)
        plt.ylabel("$ResNet_{CLIP}$", size=20)
        plt.xlim(-0.15, 0.4)
        plt.ylim(-0.15, 0.4)
        plt.grid(True)
        plt.title("Unique Variance", size=24)
        plt.tight_layout()
        plt.savefig("figures/CLIP/var_rn_vs_clip_subj5_2dhist.png", dpi=300)

        # PLOTTING ALL SUBJECTS
        fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 15))
        for s, ax in zip(np.arange(8) + 1, axes.T.flatten()):
            dfs = df[df["subj"] == s]
            h = ax.hist2d(
                dfs["nc"],
                dfs["var_clip"],
                bins=100,
                norm=mpl.colors.LogNorm(),
                cmap="magma",
            )

            sns.lineplot(
                [-0.05, 1.05],
                [-0.05, 1.05],
                linewidth=2,
                color="red",
                label="Noise Ceiling",
                ax=ax,
            )
            sns.lineplot(
                [-0.05, 1],
                [-0.05, 0.85],
                linewidth=2,
                color="orange",
                linestyle="--",
                label="85% noise ceiling",
                ax=ax,
            )

            ax.grid()
            fig.colorbar(h[3], ax=ax)
            ax.set_title("subj %d" % s)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.legend().set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left")
        fig.supylabel("Model Performance $(R^2)$")
        fig.supxlabel("Noise Ceiling")
        plt.tight_layout()

        plt.savefig("figures/CLIP/var_clip_vs_nc_all_subj.png")


if args.category_based_similarity_analysis:
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    models = ["clip", "YFCC_clip", "YFCC_slip", "YFCC_simclr", "resnet50_bottleneck"]
    for threshold in thresholds:
        scores = []
        for model in models:
            score, person_n = category_based_similarity_analysis(
                model, threshold=threshold
            )
            # print(model)
            # print(score)
            scores.append(score)
        plt.figure()
        plt.bar(np.arange(len(scores)), scores)
        plt.xticks(ticks=np.arange(len(scores)), labels=models, rotation="45")
        plt.ylabel("within/(within+cross)")
        plt.title("# of pics with person: %d/10000" % person_n)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig("figures/CLIP/category_based_sim/person_%.1f.png" % threshold)


if args.cross_model_comparison:
    from util.model_config import *
    # plot ROI averaged prediction across models
    df = pd.DataFrame()
    models = [
        # "resnet50_bottleneck",
        "YFCC_clip",
        "YFCC_simclr",
        "YFCC_slip",
        "laion400m_clip",
        "clip",
        "laion2b_clip",
    ]
    # model_sizes = ["1m", "15m", "15m", "15m", "400m", "400m", "2b"]
    model_sizes = ["15m", "15m", "15m", "400m", "400m", "2b"]
    model_size_for_plot = {"1m": 100, "15m": 200, "400m": 400, "2b": 600}
    subjs = [1, 2, 5, 7]
    # rois = [
    #     "prf-visualrois": 'all'],
    #     "floc-faces",
    #     "floc-places",
    #     "floc-bodies": ,
    #     "HCP": "TPOJ1"
    # ]
    # rois_name = {
    #     "prf-visualrois": "Early Visual",
    #     "floc-bodies": "Body",
    #     "floc-faces": "Face",
    #     "floc-places": "Place",
    # }

    roa_list = [
            ("prf-visualrois", "V1v"),
            ("prf-visualrois", "h4v"),
            ("floc-places", "RSC"),
            ("floc-places", "PPA"),
            ("floc-places", "OPA"),
            ("floc-bodies", "EBA"),
            ("floc-faces", "FFA-1"),
            ("floc-faces", "FFA-2"),
            ("HCP_MMP1", "TPOJ1"),
            ("HCP_MMP1", "TPOJ2"),
            ("HCP_MMP1", "TPOJ3"),
            # ("language", "AG"),
        ]
    # rois_name = [v for _, v in roa_list]


    # l = (4, 2, 0)
    # s = (4, 2, 45)
    # sl = (8, 2, 0)
    # l = "P"
    # s = "X"
    # sl = "*"

    model_type = {
        "clip": "Lang (OpenAI)",
        # "resnet50_bottleneck": "Labels",
        "laion400m_clip": "Lang (Laion)",
        "laion2b_clip": "Lang (Laion)",
        "YFCC_clip": "Lang (YFCC)",
        "YFCC_simclr": "SSL",
        "YFCC_slip": "SSL + Lang (YFCC)",
    }

    marker_type={
        "Lang (OpenAI)": "1",
        # "resnet50_bottleneck": "Labels",
        "Lang (Laion)": "2",
        "Lang (YFCC)": "3",
        "SSL": "x",
        "SSL + Lang (YFCC)": (8, 2, 0), 
    }

    for i, model in enumerate(tqdm(models)):
        for (roi_type, roi_lab) in roa_list:
            # rsqs = []
            for subj in subjs:
                rsq = load_model_performance(
                    model, output_root=args.output_root, subj=subj, measure="rsq"
                )
                roi_mask = np.load(
                    "%s/output/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
                    % (args.output_root, subj, subj, roi_type)
                )
                roi_dict = roi_name_dict[roi_type]
                roi_val = list(roi_dict.keys())[list(roi_dict.values()).index(roi_lab)]
                # print(roi_lab, roi_val)
                roi_selected_vox = roi_mask == int(roi_val)
                # print(np.sum(roi_selected_vox))
                # import pdb; pdb.set_trace()
                # rsqs.append(np.mean(rsq[roi_selected_vox]))
                rsq_mean = np.mean(rsq[roi_selected_vox])

                vd = {}
                vd["Regions"] = roi_lab
                vd["Model"] = model
                vd["Dataset size"] = model_sizes[i]
                vd["Model type"] = model_type[model]
                vd["Mean Performance"] = rsq_mean
                vd["Subject"] = str(subj)
            # vd["perf_std"] = np.std(rsqs[roi_selected_vox])
            # 

                df = df.append(vd, ignore_index=True)
    df.to_csv("%s/output/clip/cross_model_comparison.csv" % args.output_root)

    import seaborn.objects as so
    from seaborn import axes_style

    (
        so.Plot(
            df,
            x="Regions",
            y="Mean Performance",
            # color="Subject",
            color="Dataset size",
            marker="Model type",
            pointsize="Dataset size",
            # ymin=0,
            # ymax=0.25,
        )
        .add(
            so.Dot(),
            so.Agg(),
            so.Jitter(),
            # so.Dodge(by=["color"]),
            so.Dodge(),
        )
        # .add(so.Range(), so.Est(errorbar="sd"), so.Dodge())
        .scale(
            marker=marker_type, 
            pointsize=(13, 6))
        .theme(
            {
                **axes_style("white"),
                **{
                    "legend.frameon": False,
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                    # "axes.grid": True,
                    # "axes.grid.axis": "x",
                },
            }
        )
        .save("figures/model_comp/subj1257.png", bbox_inches="tight")
    )
