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

def make_word_cloud(text, saving_fname):
    text = " ".join(t for t in text)
    # print(text)
    wordcloud = WordCloud(background_color="white").generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file(saving_fname)

def get_PCs(model="clip", data=None, num_pc=20):
    try:
        PCs = np.load("%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model))
    except FileNotFoundError:
        pca = PCA(n_components=num_pc, svd_solver="full")
        pca.fit(data)
        PCs = pca.components_
        np.save(
                "%s/output/pca/%s/%s_pca_group_components_by_feature.npy" % (args.output_root, model, model), 
                PCs,
            )

        import pickle
        with open("%s/output/pca/%s/%s_pca_group_by_feature.pkl" % (args.output_root, model, model), "wb") as f:
            pickle.dumps(pca, f)
        
        plt.plot(pca.explained_variance_ratio_)
        plt.savefig("figures/PCA/ev/%s_explained_var_ratio.png")
        
    return PCs

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
    parser.add_argument("--image2pc", default=False, action="store_true")
    parser.add_argument("--load_and_show_all_word_clouds", default=False, action="store_true")
    parser.add_argument("--clustering_on_brain_pc", default=False, action="store_true")
    parser.add_argument("--maximize_clustering_on_brain", default=False, action="store_true")

    args = parser.parse_args()
    
    
    # if args.weight_analysis:
    #     models = ["clip", "resnet50_bottleneck", "bert_layer_13"]
    #     # models = ["convnet_res50", "clip_visual_resnet", "bert_layer_13"]
    #     for m in models:
    #         print(m)
    #         w = np.load(
    #             "%s/output/encoding_results/subj%d/weights_%s_whole_brain.npy"
    #             % (args.output_root, args.subj, m)
    #         )
    #         print(w.shape)
    #         print("NaNs? Finite?:")
    #         print(np.any(np.isnan(w)))
    #         print(np.all(np.isfinite(w)))
    #         pca = PCA(n_components=5, svd_solver="full")
    #         pca.fit(w)
    #         np.save(
    #             "%s/output/pca/subj%d/%s_pca_components.npy"
    #             % (args.output_root, args.subj, m),
    #             pca.components_,
    #         )

    if args.group_weight_analysis:
        from util.data_util import load_model_performance, fill_in_nan_voxels
        from util.util import zscore

        models = ["clip"]
        # models = ["resnet50_bottleneck", "clip_visual_resnet"]
        subjs = np.arange(1, 9)
        num_pc = 20
        best_voxel_n = 15000
        
        for m in models:
            print(m)
            
            if not os.path.exists("%s/output/pca/%s" % (args.output_root, m)):
                os.makedirs("%s/output/pca/%s" % (args.output_root, m))

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
                    ]  # get the threshold for the best n voxels
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

            # pca = PCA(n_components=num_pc, svd_solver="full")
            # pca.fit(group_w)
            # np.save(
            #     "%s/output/pca/%s/%s_pca_group_components.npy" % (args.output_root, m, m),
            #     pca.components_,
            # )

            # import pickle
            # with open("%s/output/pca/%s/%s_pca_group.pkl" % (args.output_root, m, m), "wb") as f:
            #     pickle.dumps(pca, f)
            
            # plt.plot(pca.explained_variance_ratio_)
            # plt.savefig("figures/PCA/ev/%s_explained_var_ratio.png")

            # idx = 0
            # for subj in subjs:
            #     subj_mask = np.load(
            #         "%s/output/pca/%s/pca_voxels_subj%02d_best_%d.npy"
            #         % (args.output_root, m, subj, best_voxel_n)
            #     )
            #     print(len(subj_mask))
            #     subj_pca = np.zeros((num_pc, len(subj_mask)))
            #     subj_pca[:, subj_mask] = zscore(
            #         pca.components_[:, idx : idx + np.sum(subj_mask)], axis=1
            #     )
            #     if not os.path.exists(
            #         "%s/output/pca/%s/subj%02d" % (args.output_root, m, subj)
            #     ):
            #         os.mkdir("%s/output/pca/%s/subj%02d" % (args.output_root, m, subj))
            #     np.save(
            #         "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
            #         % (args.output_root, m, subj, m),
            #         subj_pca,
            #     )
            #     idx += np.sum(subj_mask)

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
        from analyze_clip_results import extract_text_activations, extract_emb_keywords, get_coco_anns, get_coco_image, get_coco_caps
        from featureprep.feature_prep import get_preloaded_features

        model = "clip"
        plotting = False
        # model = "resnet50_bottleneck_rep_only"
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
        if "rep_only" in model:
            data_to_fit = activations
        else:
            group_w = np.load("%s/output/pca/%s/weight_matrix_best_%d.npy" % (args.output_root, model, best_voxel_n))
            data_to_fit = group_w.T
        # load PCs
        PCs = get_PCs(model=model, data=data_to_fit)
        
        # getting scores and plotting
        from pycocotools.coco import COCO

        annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
        # annFile_val = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
        coco_train = COCO(annFile_train)
        # coco_val = COCO(annFile_val)

        # annFile_train_cap = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
        # annFile_val_cap = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"
        # coco_train_cap = COCO(annFile_train_cap)
        # coco_val_cap = COCO(annFile_val_cap)
        
        cats = coco_train.loadCats(coco_train.getCatIds())
        id2cat = {}
        for cat in cats:
            id2cat[cat['id']] = cat['name']

        # compute label embedding correlation
        best_label_corrs, worst_label_corrs = [], []
        COCO_cat_feat = get_preloaded_features(
                    1,
                    stimulus_list,
                    "cat",
                    features_dir="%s/features" % args.output_root,
                )

        plt.figure(figsize=(30, 30))
        for i in tqdm(range(PCs.shape[0])):
            n_samples = int(len(stimulus_list) / 20)
            sample_idx = np.arange(0, len(stimulus_list), n_samples)
            print(len(sample_idx))
            scores = activations.squeeze() @ PCs[i, :]
            sampled_img_ids = stimulus_list[np.argsort(scores)[::-1][sample_idx]]
        
            # plot images        
            for j, id in enumerate(sampled_img_ids):
                plt.subplot(20, 20, i*20+j+1)
                I = get_coco_image(id)
                plt.axis("off")
                plt.imshow(I)
        plt.tight_layout()
        plt.savefig("figures/PCA/image_vis/%s_pc_sampled_images.png" % (model))

        # for i in tqdm(range(PCs.shape[0])):
        #     scores = activations.squeeze() @ PCs[i, :]
        #     best_img_ids = stimulus_list[np.argsort(scores)[::-1][:20]]
        #     worst_img_ids = stimulus_list[np.argsort(scores)[:20]]
        
        #     if plotting:
        #         # plot images
        #         plt.figure()
        #         for j, id in enumerate(best_img_ids):
        #             plt.subplot(4, 5, j + 1)
        #             I = get_coco_image(id)
        #             plt.axis("off")
        #             plt.imshow(I)
        #         plt.tight_layout()
        #         plt.savefig("figures/PCA/image_vis/%s_pc%d_best_images.png" % (model, i))
        #         plt.close()

        #         plt.figure()
        #         for j, id in enumerate(worst_img_ids):
        #             plt.subplot(4, 5, j + 1)
        #             I = get_coco_image(id)
        #             plt.axis("off")
        #             plt.imshow(I)
        #         plt.tight_layout()
        #         plt.savefig("figures/PCA/image_vis/%s_pc%d_worst_images.png" % (model, i))
        #         plt.close()

        #     #find corresponding captions of best image 
        #     best_caps, worst_caps = [], []
        #     for j, id in enumerate(best_img_ids):
        #         captions = get_coco_caps(id)
        #         best_caps += captions

        #     for j, id in enumerate(worst_img_ids):
        #         captions = get_coco_caps(id)
        #         worst_caps += captions

        #     # print(best_caps)
        #     # print(worst_caps)

        #     make_word_cloud(best_caps, saving_fname="./figures/PCA/image_vis/word_clouds/PC%d_best_captions.png" % i)
        #     make_word_cloud(worst_caps, saving_fname="./figures/PCA/image_vis/word_clouds/PC%d_worst_captions.png" % i)
           

        #     # calculate label consistency
        #     cat_feats = []
        #     for j, id in enumerate(best_img_ids):
        #         idx = np.where(stimulus_list == id)[0]
        #         cat_feats.append(COCO_cat_feat[idx, :])

        #     cat_feats = np.array(cat_feats).squeeze()
        #     # corr = (np.sum(np.corrcoef(cat_feats)) - num_pc) / (num_pc^2-num_pc) 
        #     corr = np.mean(np.corrcoef(cat_feats))
        #     best_label_corrs.append(corr)

        #     cat_feats = []
        #     for j, id in enumerate(worst_img_ids):
        #         idx = np.where(stimulus_list == id)[0]
        #         cat_feats.append(COCO_cat_feat[idx, :])

        #     cat_feats = np.array(cat_feats).squeeze()
        #     # print(cat_feats.shape)
        #     # corr = (np.sum(np.corrcoef(cat_feats)) - num_pc) / (num_pc^2-num_pc) 
        #     corr = np.mean(np.corrcoef(cat_feats))
        #     worst_label_corrs.append(corr)
        
        # plt.figure()
        # plt.plot(np.arange(20), worst_label_corrs, label="Worst")
        # plt.plot(np.arange(20), best_label_corrs, label="Best")

        # plt.ylabel("Mean Pairwise Correlation")
        # plt.xlabel("PCs")
        # plt.legend()
        # plt.savefig("figures/PCA/image_vis/%s_pc_label_corr.png" % model)


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


    if args.image2pc:
        from featureprep.feature_prep import get_preloaded_features
        from analyze_clip_results import extract_text_activations, extract_emb_keywords, get_coco_anns, get_coco_image, get_coco_caps

        
        model = "clip"
        stimulus_list = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
        )

        activations = get_preloaded_features(
            1,
            stimulus_list,
            "%s" % model.replace("_rep_only", ""),
            features_dir="%s/features" % args.output_root,
        )

        PCs = get_PCs(model=model)
        nPC = PCs.shape[0]
        pc_proj = np.dot(activations, PCs.T) # returns a 10000 by 20 matrix
        top2_pcs = np.argsort(np.abs(pc_proj), axis=1)[:, -2:] # returns a 10000 by 2 matrix
        pc_counter = np.zeros((nPC, nPC))
        for i in range(top2_pcs.shape[0]):
            j, k = top2_pcs[i, :]
            pc_counter[j, k] += 1
            pc_counter[k, j] += 1
        
        plt.imshow(pc_counter, cmap="Blues")
        plt.xlabel("PCs")
        plt.ylabel("PCs")
        plt.xticks(np.arange(nPC))
        plt.yticks(np.arange(nPC))
        plt.colorbar()
        plt.savefig("figures/PCA/top2pc/top2pc.png")

        ind = np.unravel_index(np.argsort(pc_counter, axis=None), pc_counter.shape)
        best_2_pcs = [(ind[0][::-1][i], ind[1][::-1][i]) for i in np.arange(0, 10, 2)]
        print(best_2_pcs)
        for p in best_2_pcs:
            proj = np.vstack((pc_proj[:, p[0]], pc_proj[:, p[1]])).T
            proj_norm = np.linalg.norm(proj, axis=1)
            img_rank = np.argsort(proj_norm)[::-1][:20]
            plt.figure(figsize=(20,20))
            for i, idx in enumerate(img_rank):
                coco_id = stimulus_list[idx]
                I = get_coco_image(coco_id)
                plt.subplot(4, 5, i+1)
                plt.imshow(I)
                plt.title("proj: %.2f, %.2f" % (pc_proj[idx, p[0]], pc_proj[idx, p[1]]))
            plt.savefig("figures/PCA/top2pc/top_images_for_PC%d&%d.png" % (p[0], p[1]))
        

        # proj_norm = np.linalg.norm(pc_proj, axis=1)
        # img_rank = np.argsort(proj_norm)[::-1][:20]
        # plt.figure(figsize=(30,10))
        # for i, idx in enumerate(img_rank):
        #     coco_id = stimulus_list[idx]
        #     I = get_coco_image(coco_id)
        #     pref_pc = np.argsort(pc_proj[idx,:])[::-1][:2]
        #     first2 = ["%d:%.2f" % (pc, pc_proj[idx, pc]) for pc in pref_pc]
        #     plt.subplot(4, 5, i + 1)
        #     plt.axis("off")
        #     plt.imshow(I)
        #     plt.title(first2)
        # plt.tight_layout()
        # plt.savefig("figures/PCA/image_vis/image2PC/image_PC_proj_%s_l2max.png" % model)

        # img_rank = np.argsort(proj_norm)[:20]
        # plt.figure(figsize=(30,10))
        # for i, idx in enumerate(img_rank):
        #     coco_id = stimulus_list[idx]
        #     I = get_coco_image(coco_id)
        #     pref_pc = np.argsort(pc_proj[idx,:])[::-1][:3]
        #     first3 = ["%d:%.2f" % (pc, pc_proj[idx, pc]) for pc in pref_pc]
        #     plt.subplot(4, 5, i + 1)
        #     plt.axis("off")
        #     plt.imshow(I)
        #     plt.title(first3)
        # plt.tight_layout()
        # plt.savefig("figures/PCA/image_vis/image2PC/image_PC_proj_%s_l2min.png" % model)
        # plt.close()

        # proj_norm = np.linalg.norm(pc_proj, ord=-np.inf, axis=1)
        # img_rank = np.argsort(proj_norm)[:20]
        # plt.figure(figsize=(30,10))
        # for i, idx in enumerate(img_rank):
        #     coco_id = stimulus_list[idx]
        #     I = get_coco_image(coco_id)
        #     pref_pc = np.argsort(pc_proj[idx,:])[::-1][:3]
        #     first3 = ["%d:%.2f" % (pc, pc_proj[idx, pc]) for pc in pref_pc]
        #     plt.subplot(4, 5, i + 1)
        #     plt.axis("off")
        #     plt.imshow(I)
        #     plt.title(first3)
        # plt.tight_layout()
        # plt.savefig("figures/PCA/image_vis/image2PC/image_PC_proj_%s_-inf.png" % model)
        # plt.close()

    if args.load_and_show_all_word_clouds:
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        import matplotlib.image as img
        plt.figure(figsize=(10, 50))
        for i in range(20):
            plt.subplot(20, 2, i*2+1)
            im = img.imread("./figures/PCA/image_vis/word_clouds/PC%d_best_captions.png" % i)
            plt.imshow(im)
            plt.title("PC %d" % i)
            plt.subplot(20, 2, i*2+2)
            im = img.imread("./figures/PCA/image_vis/word_clouds/PC%d_worst_captions.png" % i)
            plt.imshow(im)
        plt.tight_layout()
        plt.savefig("./figures/PCA/image_vis/word_clouds/all_word_clouds.png")

    if args.clustering_on_brain_pc:
        from sklearn.cluster import KMeans
        model = "clip"
        subj = [1,2,5,7]
        for s in subj:
            PCs = np.load(
                "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
                % (args.output_root, model, s, model)
            )
            print(PCs.shape)

            inertia = []
            for k in range(1,21):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(PCs.T)
                inertia.append(kmeans.inertia_)
            plt.plot(inertia, label="subj %d" % s)
        plt.legend()
        plt.ylabel("Sum of squared distances")
        plt.xlabel("# of clusters")
        plt.savefig("figures/PCA/clustering/inertia.png")
    
    if args.maximize_clustering_on_brain:
        # verify they are in a patch?
        from analyze_clip_results import extract_text_activations, extract_emb_keywords, get_coco_anns, get_coco_image, get_coco_caps
        from featureprep.feature_prep import get_preloaded_features

        model = "clip"
        plotting = False
        best_voxel_n = 20000
        n_clusters = 15

        stimulus_list = np.load(
            "%s/output/coco_ID_of_repeats_subj%02d.npy" % (args.output_root, 1)
        )

        activations = get_preloaded_features(
            1,
            stimulus_list,
            "%s" % model.replace("_rep_only", ""),
            features_dir="%s/features" % args.output_root,
        )

        group_w = np.load("%s/output/pca/%s/weight_matrix_best_%d.npy" % (args.output_root, model, best_voxel_n))
        print(group_w.shape)

        # getting scores and plotting
        from pycocotools.coco import COCO
        annFile_train = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_train2017.json"
        # annFile_val = "/lab_data/tarrlab/common/datasets/coco_annotations/instances_val2017.json"
        coco_train = COCO(annFile_train)
        # coco_val = COCO(annFile_val)
        PCs = np.load(
            "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
            % (args.output_root, model, args.subj, model)
        )
        kmeans = KMeans(n_clusters=k, random_state=0).fit(PCs.T)
        
        for i in tqdm(range(n_clusters)):
            vox = kmeans.labels_==i
            scores = activations.squeeze() @ group_w[:, vox]
            best_img_ids = stimulus_list[np.argsort(scores)[::-1][:20]]
        
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
