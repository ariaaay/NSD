import os
import argparse
# from msilib.schema import File
import pickle

import pandas as pd
import seaborn as sns
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA

import torch
import clip

from util.data_util import load_model_performance, extract_test_image_ids
from util.model_config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_sample_performance(
    model, output_dir, masking="sig", subj=1, measure="corrs"
):
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

    roi_pca = PCA(n_components=5, svd_solver="full")
    roi_pca.fit(roi_w)

    scores = np.mean(activations.squeeze() @ roi_w, axis=1)
    print(roi_name)
    best_list = list(np.array(common_words)[np.argsort(scores)[::-1][:30]])
    worst_list = list(np.array(common_words)[np.argsort(scores)[:30]])
    print(best_list)
    print(worst_list)
    pickle.dump(
        best_list,
        open(
            "%s/output/clip/word_interpretation/best_in_%s.json"
            % (args.output_root, roi_name),
            "wb",
        ),
    )
    pickle.dump(
        worst_list,
        open(
            "%s/output/clip/word_interpretation/worst_in_%s.json"
            % (args.output_root, roi_name),
            "wb",
        ),
    )


def plot_image_wise_performance(model1, model2, masking="sig", measure="corrs"):
    sample_corr1 = compute_sample_performance(
        model=model1, output_dir=args.output_root, masking=masking, measure=measure
    )
    sample_corr2 = compute_sample_performance(
        model=model2, output_dir=args.output_root, masking=masking, measure=measure
    )
    plt.figure()
    plt.scatter(sample_corr1, sample_corr2, alpha=0.3)
    plt.plot([-0.1, 1], [-0.1, 1], "r")
    plt.xlabel(model1)
    plt.ylabel(model2)
    plt.savefig(
        "figures/CLIP/image_wise_performance/%s_vs_%s_samplewise_%s_%s.png"
        % (model1, model2, measure, masking)
    )


def get_coco_image(id):
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
    return I


def find_corner_images(
    model1, model2, upper_thr=0.5, lower_thr=0.03, masking="sig", measure="corrs"
):
    sp1 = compute_sample_performance(
        model=model1, output_dir=args.output_root, masking=masking, measure=measure
    )
    sp2 = compute_sample_performance(
        model=model2, output_dir=args.output_root, masking=masking, measure=measure
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
        "%s/output/clip/%s_vs_%s_corner_image_ids_%s_sample_%s.npy"
        % (args.output_root, model1, model2, masking, measure),
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
            "figures/CLIP/corner_images/sample_%s_images_%s_%s.png"
            % (measure, image_labels[i], masking)
        )
        plt.close()


def compare_model_and_brain_performance_on_COCO(subj=1):
    from scipy.stats import pearsonr
    from extract_clip_features import load_captions


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


def sample_level_semantic_analysis(subj=1, model1="clip", model2="resnet50_bottleneck"):
    cocoId_subj = np.load(
        "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, subj)
    )
    # models = [
    #     "clip",
    #     "clip_text",
    #     "convnet_res50",
    #     "bert_layer_13",
    #     "clip_visual_resnet",
    # ]
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

        plt.subplot(10, 2, i * 2 + 2)
        id = cocoId_subj[trial_id_pair_2[i][1]]
        I = get_coco_image(id)
        plt.imshow(I)

    plt.tight_layout()
    plt.savefig(
        "figures/CLIP/RDM_max/RDM_max_images_close_in_%s_far_in_%s.png"
        % (model2, model1)
    )

def make_roi_df(roi_names, subjs, update=False):
    if update:
        df = pd.read_csv("%s/output/clip/performance_by_roi_df_nc_corrected.csv" % args.output_root)
    else:
        df = pd.DataFrame()

    for subj in subjs:
        try:
            subj_df = pd.read_csv("%s/output/clip/performance_by_roi_df_subj%02d_nc_corrected.csv" % (args.output_root, subj))
        except FileNotFoundError:
            subj_df = pd.DataFrame(
                columns = [
                    "voxel_idx",
                    "var_clip",
                    "var_resnet",
                    "uv_clip",
                    "uv_resnet",
                    "uv_diff",
                    "uv_diff_nc",
                    "joint",
                    "subj"
                ] + roi_names
            )
                
            joint_var = load_model_performance(model="resnet50_bottleneck", output_root=args.output_root, subj=subj, measure="rsq")
            clip_var = load_model_performance(model="clip", output_root=args.output_root, subj=subj, measure="rsq")
            resnet_var = load_model_performance(model="resnet50_bottleneck", output_root=args.output_root, subj=subj, measure="rsq")
            nc = np.load("%s/output/noise_ceiling/subj%01d/ncsnr_1d_subj%02d.npy" % (args.output_root, subj, subj))

            u_clip = joint_var - resnet_var
            u_resnet = joint_var - clip_var

            for i in tqdm(range(len(joint_var))):
                if nc[i] < 0.1:
                    continue
                else:
                    vd = dict()
                    vd["voxel_idx"] = i
                    vd["var_clip"] = clip_var[i]
                    vd["var_resnet"] = resnet_var[i]
                    vd["uv_clip"] = u_clip[i]
                    vd["uv_resnet"] = u_resnet[i]
                    vd["uv_diff"] = u_clip[i] - u_resnet[i]
                    vd["uv_diff_nc"] = u_clip[i]/nc[i] - u_resnet[i]/nc[i]
                    vd["joint"] = joint_var[i]
                    vd["subj"] = subj
                    subj_df = subj_df.append(vd, ignore_index=True)

            cortical_mask = np.load(
                "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
                % (args.output_root, subj, subj)
            )

            for roi_name in roi_names:
                if roi_name == "language":
                    lang_ROI = np.load("%s/output/voxels_masks/language_ROIs.npy" % args.output_root, allow_pickle=True).item()
                    roi_volume = lang_ROI['subj%02d' % subj]
                    roi_volume = np.swapaxes(roi_volume, 0, 2)

                else:            
                    roi = nib.load(
                        "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/subj%02d/func1pt8mm/roi/%s.nii.gz" % (subj, roi_name)
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
                subj_df[roi_name] = np.array(roi_labels)[np.array(list(subj_df["voxel_idx"])).astype(int)]
            
            subj_df.to_csv("%s/output/clip/performance_by_roi_df_subj%02d_nc_corrected.csv" % (args.output_root, subj))
        df = pd.concat([df, subj_df])
        
    df.to_csv("%s/output/clip/performance_by_roi_df_nc_corrected.csv" % args.output_root)
    return df


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
    parser.add_argument("--rerun_df", default=False, action="store_true")
    parser.add_argument("--weight_analysis", default=False, action="store_true")
    parser.add_argument(
        "--extract_keywords_for_roi", default=False, action="store_true"
    )
    parser.add_argument("--group_analysis_by_roi", default=False, action="store_true")
    parser.add_argument("--group_weight_analysis", default=False, action="store_true")
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
        plt.savefig(
            "figures/CLIP/voxel_wise_performance/%s_vs_%s_acc_%s.png"
            % (model1, model2, args.roi)
        )

    if args.weight_analysis:
        models = ["clip", "resnet50_bottleneck", "bert_layer_13"]
        # models = ["convnet_res50", "clip_visual_resnet", "bert_layer_13"]
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
            pca = PCA(n_components=5, svd_solver="full")
            pca.fit(w)
            np.save(
                "%s/output/pca/subj%d/%s_pca_components.npy"
                % (args.output_root, args.subj, m),
                pca.components_,
            )
    
    if args.group_weight_analysis:
        from util.data_util import load_model_performance, fill_in_nan_voxels
        from util.util import zscore
        
        models = ["clip"]
        subjs = [1,2,5,7]
        num_pc = 20
        best_voxel_n = 10000
        # models = ["convnet_res50", "clip_visual_resnet", "bert_layer_13"]
        for m in models:
            print(m)
            try:
                group_w = np.load("%s/output/pca/weight_matrix_best5000.npy" % args.output_root)
            except FileNotFoundError:
                group_w = []
                for subj in subjs:
                    w = np.load(
                        "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
                        % (args.output_root, subj, m)
                    )
                    w = fill_in_nan_voxels(w, subj, args.output_root)
                    # print(w.shape) # 512 x $voxel
                    # print("NaNs? Finite?:")
                    # print(np.any(np.isnan(w)))
                    # print(np.all(np.isfinite(w)))
                    rsq = load_model_performance(m, output_root=args.output_root, subj=subj, measure="rsq")
                    nc = np.load("%s/output/noise_ceiling/subj%01d/ncsnr_1d_subj%02d.npy" % (args.output_root, subj, subj))
                    corrected_rsq = rsq / nc
                    threshold = corrected_rsq[np.argsort(corrected_rsq)[-best_voxel_n]] # get the threshold for the best 10000 voxels
                    print(threshold)
                    group_w.append(w[:, corrected_rsq>=threshold])
                    np.save("%s/output/pca/pca_voxels_subj%02d.npy" % (args.output_root, subj), corrected_rsq>=threshold)
                group_w = np.hstack(group_w)
                np.save("%s/output/pca/weight_matrix_best_%d.npy" % (args.output_root, best_voxel_n), group_w)
            pca = PCA(n_components=num_pc, svd_solver="full")
            pca.fit(group_w)
            np.save(
                "%s/output/pca/%s_pca_group_components.npy"
                % (args.output_root, m),
                pca.components_,
            )
            idx = 0
            for subj in subjs:
                subj_mask = np.load("%s/output/pca/pca_voxels_subj%02d.npy" % (args.output_root, subj))
                subj_pca = np.zeros((num_pc, len(subj_mask)))
                subj_pca[:, subj_mask] = zscore(pca.components_[:, idx : idx + np.sum(subj_mask)], axis=1)
                if not os.path.exists("%s/output/pca/subj%02d" % (args.output_root, subj)):
                    os.mkdir("%s/output/pca/subj%02d" % (args.output_root, subj))
                np.save("%s/output/pca/subj%02d/%s_pca_group_components.npy" % (args.output_root, subj, m), subj_pca)
                idx += np.sum(subj_mask)
            

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
            plot_image_wise_performance("convnet_res50", "clip", masking=roi)
            find_corner_images("clip", "convnet_res50", masking=roi)
            plot_image_wise_performance("bert_layer_13", "clip", masking=roi)
            find_corner_images("clip", "bert_layer_13", masking=roi)

            plot_image_wise_performance(
                "convnet_res50", "clip", masking=roi, measure="rsq"
            )
            find_corner_images("clip", "convnet_res50", masking=roi, measure="rsq")
            plot_image_wise_performance(
                "bert_layer_13", "clip", masking=roi, measure="rsq"
            )
            find_corner_images("clip", "bert_layer_13", masking=roi, measure="rsq")

    if args.compare_brain_and_clip_performance:
        compare_model_and_brain_performance_on_COCO(subj=1)

    if args.coarse_level_semantic_analysis:
        coarse_level_semantic_analysis(subj=1)

    if args.extract_keywords_for_roi:
        with open("output/1000eng.txt") as f:
            out = f.readlines()
        common_words = ["photo of " + w[:-1] for w in out]
        try:
            activations = np.load(
                "%s/output/clip/word_interpretation/1000eng_activation.npy"
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
                "%s/output/clip/word_interpretation/1000eng_activation.npy"
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
            subj=args.subj, model1="clip", model2="resnet50_bottleneck"
        )
        sample_level_semantic_analysis(
            subj=args.subj, model1="clip", model2="bert_layer_13"
        )
        sample_level_semantic_analysis(
            subj=args.subj, model1="visual_layer_11", model2="resnet50_bottleneck"
        )
        sample_level_semantic_analysis(
            subj=args.subj, model1="clip", model2="visual_layer_1"
        )
        sample_level_semantic_analysis(
            subj=args.subj, model1="clip", model2="clip_text"
        )

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
        sns.set(style="whitegrid", font_scale=1.5)

        roi_names = list(roi_name_dict.keys())
        if not args.rerun_df:
            df = pd.read_csv("%s/output/clip/performance_by_roi_df_nc_corrected.csv" % args.output_root)
        else:
            df = make_roi_df(roi_names, subjs=[1, 2, 5, 7])

        for roi_name in roi_names:  
            plt.figure(figsize=(50, 20))
            ax = sns.barplot(x=roi_name, y="uv_diff", data=df, dodge=True, order=list(roi_name_dict[roi_name].values()))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig("figures/CLIP/performances_by_roi/uv_diff_%s.png" % roi_name)
        
        for roi_name in roi_names:  
            plt.figure(figsize=(50, 20))
            ax = sns.barplot(x=roi_name, y="uv_diff_nc", data=df, dodge=True, order=list(roi_name_dict[roi_name].values()))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            plt.savefig("figures/CLIP/performances_by_roi/uv_nc_diff_%s.png" % roi_name)
    
    if args.group_analysis_by_roi:
        from scipy.stats import ttest_rel
        from util.util import ztransform

        roa_list = [("floc-bodies", "EBA"),
                ("floc-faces", "FFA-1"),
                ("floc-places", "RSC"),
                ("floc-words", "VWFA-1"),
                ("HCP_MMP1", "MST"),
                ("HCP_MMP1", "MT"),
                ("HCP_MMP1", "PH"),
                ("HCP_MMP1", "TPOJ2"),
                ("HCP_MMP1", "TPOJ3"),
                ("HCP_MMP1", "PGp"),
                ("HCP_MMP1", "V4t"),
                ("HCP_MMP1", "FST"),
                ("language", "AG"),
                ("prf-visualrois", "V1v")
                ]

        # roa_list = [] 
        # roi_names = list(roi_name_dict.keys())
        # for roi_name in roi_names:
        #     if df[roi]

        df = pd.read_csv("%s/output/clip/performance_by_roi_df_nc_corrected.csv" % args.output_root)
        subjs = [1,2,5,7]
        roi_by_subj_mean_clip = np.zeros((4, len(roa_list)))
        roi_by_subj_mean_resnet = np.zeros((4, len(roa_list)))
        for s, subj in enumerate(subjs):
            nc = np.load("%s/output/noise_ceiling/subj%01d/ncsnr_1d_subj%02d.npy" % (args.output_root, subj, subj))
            varc = df[df["subj"]==subj]["var_clip"] / nc[nc>=0.1]
            varr = df[df["subj"]==subj]["var_resnet"] / nc[nc>=0.1]
            tmp_c = ztransform(varc)
            tmp_r = ztransform(varr)

            means_c, means_r = [], []
            for i, (roi_name, roi_lab) in enumerate(roa_list):
                roiv = df[roi_name]==roi_lab
                roi_by_subj_mean_clip[s, i] = np.mean(tmp_c[roiv])
                roi_by_subj_mean_resnet[s, i] = np.mean(tmp_r[roiv])

        stats = ttest_rel(roi_by_subj_mean_clip, roi_by_subj_mean_resnet, axis=0, nan_policy='propagate', alternative='two-sided')
        print(stats)
        results = {}
        for i, r in enumerate(roa_list):
            results[r] = (stats[0][i], stats[1][i])
        for k,v in results.items():
            print(k, v)
        # print(roa_list)
    

            





        
