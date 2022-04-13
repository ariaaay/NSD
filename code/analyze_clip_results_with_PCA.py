import os
import argparse

# from msilib.schema import File
import pickle

import seaborn as sns
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA

# import torch
import clip

from util.data_util import load_model_performance, extract_test_image_ids
from util.model_config import *
from analyze_clip_results import extract_text_activations, extract_emb_keywords, get_coco_anns, get_coco_image, get_coco_caps

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
    
    parser.add_argument("--weight_analysis", default=False, action="store_true")
    parser.add_argument(
        "--extract_keywords_for_roi", default=False, action="store_true"
    )
    parser.add_argument("--group_analysis_by_roi", default=False, action="store_true")
    parser.add_argument("--group_weight_analysis", default=False, action="store_true")
    parser.add_argument("--pc_text_visualization", default=False, action="store_true")
    parser.add_argument("--pc_image_visualization", default=False, action="store_true")
    parser.add_argument("--proj_feature_pc_to_subj", default=False, action="store_true")
    parser.add_argument("--analyze_PCproj_consistency", default=False, action="store_true")
    args = parser.parse_args()
    
    
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

        # models = ["clip"]
        models = ["resnet50_bottleneck", "clip_visual_resnet"]
        subjs = [1, 2, 5, 7]
        num_pc = 20
        best_voxel_n = 20000
        
        for m in models:
            print(m)
            try:
                group_w = np.load(
                    "%s/output/pca/%s/weight_matrix_best_%d.npy"
                    % (args.output_root, m, best_voxel_n)
                )
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
                    rsq = load_model_performance(
                        m, output_root=args.output_root, subj=subj, measure="rsq"
                    )
                    nc = np.load(
                        "%s/output/noise_ceiling/subj%01d/ncsnr_1d_subj%02d.npy"
                        % (args.output_root, subj, subj)
                    )
                    corrected_rsq = rsq / nc
                    threshold = corrected_rsq[
                        np.argsort(corrected_rsq)[-best_voxel_n]
                    ]  # get the threshold for the best 10000 voxels
                    print(threshold)
                    print(w.shape)
                    group_w.append(w[:, corrected_rsq >= threshold])
                    np.save(
                        "%s/output/pca/%s/pca_voxels_subj%02d_best_%d.npy"
                        % (args.output_root, m, subj, best_voxel_n),
                        corrected_rsq >= threshold,
                    )
                group_w = np.hstack(group_w)
                np.save(
                    "%s/output/pca/%s/weight_matrix_best_%d.npy"
                    % (args.output_root, m, best_voxel_n),
                    group_w,
                )

            pca = PCA(n_components=num_pc, svd_solver="full")
            pca.fit(group_w)
            np.save(
                "%s/output/pca/%s/%s_pca_group_components.npy" % (args.output_root, m, m),
                pca.components_,
            )
            idx = 0
            for subj in subjs:
                subj_mask = np.load(
                    "%s/output/pca/%s/pca_voxels_subj%02d_best_%d.npy"
                    % (args.output_root, m, subj, best_voxel_n)
                )
                print(len(subj_mask))
                subj_pca = np.zeros((num_pc, len(subj_mask)))
                subj_pca[:, subj_mask] = zscore(
                    pca.components_[:, idx : idx + np.sum(subj_mask)], axis=1
                )
                if not os.path.exists(
                    "%s/output/pca/%s/subj%02d" % (args.output_root, m, subj)
                ):
                    os.mkdir("%s/output/pca/%s/subj%02d" % (args.output_root, m, subj))
                np.save(
                    "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
                    % (args.output_root, m, subj, m),
                    subj_pca,
                )
                idx += np.sum(subj_mask)

    if args.pc_text_visualization:
        subjs = [1, 2, 5, 7]
        num_pc = 20
        best_voxel_n = 20000

        with open("%s/output/clip/word_interpretation/1000eng.txt" % args.output_root) as f:
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
        group_w = np.load("%s/output/pca/weight_matrix_best_%d.npy" % (args.output_root, best_voxel_n))
        pca = PCA(n_components=num_pc, svd_solver="full")
        pca.fit(group_w.T)
        np.save(
                "%s/output/pca/clip_pca_group_components_by_feature.npy" % args.output_root,
                pca.components_,
            )
        # each components should be 20 x 512?
        keywords = dict()
        for i in range(pca.components_.shape[0]):
            keywords[i] = extract_emb_keywords(pca.components_[i, :], activations, common_words)
        
        for k, v in keywords.items():
            print("****** PC " + str(k) + " ******")
            print("-Best:")
            print(v[0])
            print("-Worst:")
            print(v[1])
            
        np.save("%s/output/clip/word_interpretation/group_pc_keywords.json" % (args.output_root), keywords)

    
    if args.pc_image_visualization:
        from featureprep.feature_prep import get_preloaded_features

        model = "clip"
        plotting = True
        # model = "resnet50_bottleneck_rep_only"
        num_pc = 20
        best_voxel_n = 20000

        stimulus_list = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
        )

        activations = get_preloaded_features(
            1,
            stimulus_list,
            "%s" % model.replace("_rep_only", ""),
            features_dir="%s/features" % args.output_root,
        )

        # load PCs
        if "rep_only" in model:
            try:
                PCs = np.load("%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model))
            except FileNotFoundError:
                pca = PCA(n_components=num_pc, svd_solver="full")
                pca.fit(activations)
                PCs = pca.components_
                np.save(
                        "%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model), 
                        PCs,
                    )
        else:
            try:
                PCs = np.load("%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model))
            except FileNotFoundError:
                group_w = np.load("%s/output/pca/%s/weight_matrix_best_%d.npy" % (args.output_root, model, best_voxel_n))
                pca = PCA(n_components=num_pc, svd_solver="full")
                pca.fit(group_w.T)
                PCs = pca.components_
                np.save(
                        "%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model), 
                        PCs,
                    )
        
        # getting scores and plotting
        
        
        cats = coco_train.loadCats(coco_train.getCatIds())
        id2cat = {}
        for cat in cats:
            id2cat[cat['id']] = cat['name']

        # each components should be 20 x 512?
        COCO_cat_feat = get_preloaded_features(
                    1,
                    stimulus_list,
                    "cat",
                    features_dir="%s/features" % args.output_root,
                )
        best_label_corrs, worst_label_corrs = [], []
        for i in tqdm(range(PCs.shape[0])):
            scores = activations.squeeze() @ PCs[i, :]
            best_img_ids = stimulus_list[np.argsort(scores)[::-1][:20]]
            worst_img_ids = stimulus_list[np.argsort(scores)[:20]]
        
            if plotting:
                # plot images
                plt.figure()
                for j, id in enumerate(best_img_ids):
                    plt.subplot(4, 5, j + 1)
                    I = get_coco_image(id)
                    plt.axis("off")
                    plt.imshow(I)
                plt.tight_layout()
                plt.savefig("figures/PCA/image_vis/%s_pc%d_best_images.png" % (model, i))
                plt.close()

                plt.figure()
                for j, id in enumerate(worst_img_ids):
                    plt.subplot(4, 5, j + 1)
                    I = get_coco_image(id)
                    plt.axis("off")
                    plt.imshow(I)
                plt.tight_layout()
                plt.savefig("figures/PCA/image_vis/%s_pc%d_worst_images.png" % (model, i))
                plt.close()

            #find corresponding captions of best image 
            best_cats, worst_cats = [], []
            for j, id in enumerate(best_img_ids):
                captions = get_coco_caps(id)
                best_cats.append(captions)

            for j, id in enumerate(worst_img_ids):
                captions = get_coco_caps(id)
                worst_cats.append(captions)

            print(best_cats)
            print(worst_cats)

            # calculate label consistency
            cat_feats = []
            for j, id in enumerate(best_img_ids):
                idx = np.where(stimulus_list == id)[0]
                cat_feats.append(COCO_cat_feat[idx, :])

            cat_feats = np.array(cat_feats).squeeze()
            # corr = (np.sum(np.corrcoef(cat_feats)) - num_pc) / (num_pc^2-num_pc) 
            corr = np.mean(np.corrcoef(cat_feats))
            best_label_corrs.append(corr)

            cat_feats = []
            for j, id in enumerate(worst_img_ids):
                idx = np.where(stimulus_list == id)[0]
                cat_feats.append(COCO_cat_feat[idx, :])

            cat_feats = np.array(cat_feats).squeeze()
            print(cat_feats.shape)
            # corr = (np.sum(np.corrcoef(cat_feats)) - num_pc) / (num_pc^2-num_pc) 
            corr = np.mean(np.corrcoef(cat_feats))
            worst_label_corrs.append(corr)
        
        plt.figure()
        plt.plot(np.arange(20), worst_label_corrs, label="Worst")
        plt.plot(np.arange(20), best_label_corrs, label="Best")

        plt.ylabel("Mean Pairwise Correlation")
        plt.xlabel("PCs")
        plt.legend()
        plt.savefig("figures/PCA/image_vis/%s_pc_label_corr.png" % model)


    if args.proj_feature_pc_to_subj:
        from util.util import zscore
        # Calculate weight projection onto PC space
        model = "clip"
        subjs = [1, 2, 5, 7]
        num_pc = 20
        best_voxel_n = 20000
        PC_feat = np.load("%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model))
        group_w = np.load(
                    "%s/output/pca/%s/weight_matrix_best_%d.npy"
                    % (args.output_root, model, best_voxel_n)
                )
        w_transformed = np.dot(group_w.T, PC_feat.T) # (80,000x512 x 512x20)
        print(w_transformed.shape) 
        proj = w_transformed.T # should be (# of PCs) x (# of voxels) 

        idx = 0
        for subj in subjs:
            subj_mask = np.load(
                "%s/output/pca/%s/pca_voxels_subj%02d_best_%d.npy"
                % (args.output_root, model, subj, best_voxel_n)
            )
            subj_proj = np.zeros((num_pc, len(subj_mask)))
            subj_proj[:, subj_mask] = zscore(
                proj[:, idx : idx + np.sum(subj_mask)], axis=1
            )
            if not os.path.exists(
                "%s/output/pca/%s/subj%02d" % (args.output_root, model, subj)
            ):
                os.mkdir("%s/output/pca/%s/subj%02d" % (args.output_root, model, subj))
            np.save(
                "%s/output/pca/%s/subj%02d/%s_feature_pca_projections.npy"
                % (args.output_root, model, subj, model),
                subj_proj,
            )
            idx += np.sum(subj_mask)


    if args.analyze_PCproj_consistency:
        from analyze_in_mni import analyze_data_correlation_in_mni

        subjs = [1, 2, 5, 7]
        model = "clip"
        # load all PC projection from all four subjs
        all_PC_projs = []
        for subj in subjs:
            all_PC_projs.append(np.load(
                    "%s/output/pca/%s/subj%02d/%s_feature_pca_projections.npy"
                    % (args.output_root, model, subj, model)
                ))


        analyze_data_correlation_in_mni(all_PC_projs, model, dim=20, save_name = "PC_proj", subjs=subjs)