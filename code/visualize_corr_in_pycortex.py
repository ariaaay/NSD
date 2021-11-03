"This scripts visualize prediction performance with Pycortex."

import pickle
from nibabel.volumeutils import working_type
import numpy as np

import argparse
from tqdm import tqdm

from util.model_config import model_features
from util.data_util import load_model_performance


def project_vals_to_3d(vals, mask):
    # print(np.sum(mask))
    # print(len(vals))
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals


def visualize_layerwise_max_corr_results(
    model,
    layer_num,
    subj=1,
    task=None,
    threshold=85,
    mask_with_significance=False,
    start_with_zero=True,
):
    val_array = list()
    for i in range(layer_num):
        if not start_with_zero:  # layer starts with 1
            continue
        val_array.append(
            load_model_performance(
                model="%s_%d" % (model, i), output_root=output_root, subj=args.subj
            )
        )

    val_array = np.array(val_array)

    threshold_performance = np.max(val_array, axis=0) * (threshold / 100)
    layeridx = np.zeros(threshold_performance.shape) - 1
    for v in tqdm(range(len(threshold_performance))):
        if threshold_performance[v] > 0:
            layeridx[v] = (
                int(np.nonzero(val_array[:, v] >= threshold_performance[v])[0][0]) + 1
            )
            # print(layeridx[i])
    try:
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
            % (output_root, args.subj, args.subj)
        )
    except FileNotFoundError:
        print("loading old mask...")
        # cortical_mask = np.load(
        #     "%s/output/voxels_masks/subj%d/old/cortical_mask_subj%02d.npy"
        #     % (output_root, args.subj, args.subj)
        # )
    except:
        pass

    if mask_with_significance:
        if args.sig_method == "negtail_fdr":
            sig_mask = np.load(
                "%s/output/voxels_masks/subj%d/%s_%s_%s_%0.2f.npy"
                % (output_root, subj, model, task, "negtail_fdr", 0.05)
            )
        elif args.sig_method == "pvalue":
            pvalues = load_model_performance(
                model="%s_%d" % (model, layer_num - 1),
                output_root=output_root,
                subj=args.subj,
                measure="pvalue",
            )
            sig_mask = pvalues <= 0.05

    layeridx[~sig_mask] = -1

    # projecting value back to 3D space
    all_vals = project_vals_to_3d(layeridx, cortical_mask)

    layerwise_volume = cortex.Volume(
        all_vals,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=1,
        vmax=layer_num,
    )
    return layerwise_volume


def make_volume(
    subj,
    model=None,
    vals=None,
    model2=None,
    task=None,
    mask_with_significance=False,
    output_root=".",
    measure="corr",
):
    if measure == "corr":
        vmax = 0.8
    else:
        vmax = 0.6

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (output_root, subj, subj)
    )

    # load correlation scores of cortical voxels
    if vals is None:
        if type(model) == list: # for different naming convention for variance partitioning (only 1 should exist)
            model_list = model
            for model in model_list:
                try:
                    vals = load_model_performance(
                        model, task, output_root=output_root, subj=subj, measure=measure
                    )
                    break
                except FileNotFoundError:
                    continue
        else:
            vals = load_model_performance(
                    model, task, output_root=output_root, subj=subj, measure=measure
                )
                
        if model2 is not None: #for variance paritioning
            vals2 = load_model_performance(
                model2, task, output_root=output_root, subj=subj, measure=measure
            )
            vals = vals-vals2
    
        print("model:" + model)
        print("max:" + str(max(vals)))


    if mask_with_significance:
        if args.sig_method == "negtail_fdr":
            sig_mask = np.load(
                "%s/output/voxels_masks/subj%d/%s_%s_%s_%0.2f.npy"
                % (output_root, subj, model, task, "negtail_fdr", 0.05)
            )
        elif args.sig_method == "pvalue":
            pvalues = load_model_performance(
                model, task, output_root=output_root, subj=subj, measure="pvalue"
            )
            sig_mask = pvalues <= 0.05

        vals[~sig_mask] = 0
    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap="hot",
        vmin=0,
        vmax=vmax,
    )
    return vol_data


def make_pc_volume(subj, vals, mask_with_significance=False, output_root="."):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    try:
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
            % (output_root, subj, subj)
        )
    except FileNotFoundError:
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/old/cortical_mask_subj%02d.npy"
            % (output_root, subj, subj)
        )

    if mask_with_significance:
        sig_mask = np.load(
            "%s/output/voxels_masks/subj%d/taskrepr_superset_mask_%s_%0.02f.npy"
            % (output_root, subj, "negtail_fdr", 0.05)
        )
        vals[~sig_mask] = -999
    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap="RdPu",
        vmin=-3,
        vmax=3,
    )
    return vol_data


def make_3pc_volume(subj, PCs, mask_with_significance=False, output_root="."):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    try:
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
            % (output_root, subj, subj)
        )
    except FileNotFoundError:
        cortical_mask = np.load(
            "%s/output/voxels_masks/subj%d/old/cortical_mask_subj%02d.npy"
            % (output_root, subj, subj)
        )

    pc_3d = []
    for i in range(3):
        tmp = PCs[i, :] / np.max(PCs_zscore[i, :]) * 255
        if mask_with_significance:
            sig_mask = np.load(
                "%s/output/voxels_masks/subj%d/taskrepr_superset_mask_%s_%0.02f.npy"
                % (output_root, subj, "negtail_fdr", 0.05)
            )
            tmp[~sig_mask] = 0
        # projecting value back to 3D space
        pc_3d.append(project_vals_to_3d(tmp, cortical_mask))

    red = cortex.Volume(
        pc_3d[0].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    green = cortex.Volume(
        pc_3d[1].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    blue = cortex.Volume(
        pc_3d[2].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )

    vol_data = cortex.VolumeRGB(
        red,
        green,
        blue,
        "subj%02d" % subj,
        channel1color=(194, 30, 86),
        channel2color=(50, 205, 50),
        channel3color=(30, 144, 255),
    )

    return vol_data


def make_roi_volume(roi_name):
    roi = nib.load(
        "%s/output/voxels_masks/subj%d/%s.nii.gz" % (output_root, args.subj, roi_name)
    )
    roi_data = roi.get_fdata()
    roi_data = np.swapaxes(roi_data, 0, 2)

    roi_volume = cortex.Volume(
        roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=0,
        vmax=np.max(roi_data),
    )
    return roi_volume


if __name__ == "__main__":
    import cortex
    import nibabel as nib

    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    parser.add_argument("--mask_sig", default=False, action="store_true")
    parser.add_argument("--sig_method", default="negtail_fdr")
    parser.add_argument("--alpha", default=0.05)
    parser.add_argument("--show_pcs", default=False, action="store_true")
    parser.add_argument("--on_cluster", action="store_true")

    args = parser.parse_args()

    if args.on_cluster:
        output_root = "/user_data/yuanw3/project_outputs/NSD"
    else:
        output_root = "."


    visual_roi_volume = make_roi_volume("prf-visualrois")
    ecc_roi_volume = make_roi_volume("prf-eccrois")
    place_roi_volume = make_roi_volume("floc-places")
    face_roi_volume = make_roi_volume("floc-faces")
    body_roi_volume = make_roi_volume("floc-bodies")
    word_roi_volume = make_roi_volume("floc-words")
    kastner_volume = make_roi_volume("Kastner2015")
    hcp_volume = make_roi_volume("HCP_MMP1")
    sulc_volume = make_roi_volume("corticalsulc")

    ev_vals = np.load(
        "%s/output/evs_subj%02d_zscored.npy" % (output_root, args.subj)
    )
    ev_volume = make_volume(subj=args.subj, vals=ev_vals, measure="rsq")

    volumes = {
        "Visual ROIs": visual_roi_volume,
        "Eccentricity ROIs": ecc_roi_volume,
        "Places ROIs": place_roi_volume,
        "Faces ROIs": face_roi_volume,
        "Bodies ROIs": body_roi_volume,
        "Words ROIs": word_roi_volume,
        "Kastner2015": kastner_volume,
        "HCP": hcp_volume,
        "sulcus": sulc_volume,
        "EV": ev_volume,
    }

    # for i in range(12):
    #     volumes["clip-ViT-%s" % str(i + 1)] = make_volume(
    #         subj=args.subj,
    #         model="visual_layer_%d" % i,
    #         output_root=output_root,
    #         mask_with_significance=args.mask_sig,
    #     )

    # volumes["clip-ViT-last"] = make_volume(
    #     subj=args.subj,
    #     model="clip",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    # for i in range(12):
    #     volumes["clip-text-%s" % str(i + 1)] = make_volume(
    #         subj=args.subj,
    #         model="text_layer_%d" % i,
    #         output_root=output_root,
    #         mask_with_significance=args.mask_sig,
    #     )

    # volumes["clip-text-last"] = make_volume(
    #     subj=args.subj,
    #     model="clip_text",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    volumes["clip_top1_object"] = make_volume(
        subj=args.subj,
        model="clip_top1_object",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
    )

    volumes["clip_all_objects"] = make_volume(
        subj=args.subj,
        model="clip_object",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
    )

    volumes["COCO categories"] = make_volume(
        subj=args.subj,
        model="cat",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
    )

    volumes["COCO super categories"] = make_volume(
        subj=args.subj,
        model="supcat",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
    )

    # volumes["CLIP+Cat"] = make_volume(
    #     subj=args.subj,
    #     model="clip_cat",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["CLIP+Resnet50"] = make_volume(
    #     subj=args.subj,
    #     model="clip_resnet50_bottleneck",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    # for model in ["resnet50_bottleneck", "clip", "cat"]:
    #     for subset in ["person", "giraffe", "toilet", "train"]:
    #         model_name = "%s_%s_subset" % (model, subset)

    #         volumes[model_name] = make_volume(
    #             subj=args.subj,
    #             model=model_name,
    #             output_root=output_root,
    #             mask_with_significance=args.mask_sig,
    #         )

    # volumes["clip-person-subset"] = make_volume(
    #     subj=args.subj,
    #     model="clip_person_subset",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["resnet-person-subset"] = make_volume(
    #     subj=args.subj,
    #     model="resnet50_bottleneck_person_subset",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["cat-person-subset"] = make_volume(
    #     subj=args.subj,
    #     model="cat_person_subset",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["clip-top1-person-subset"] = make_volume(
    #     subj=args.subj,
    #     model="clip_top1_object_person_subset",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    volumes["resnet50"] = make_volume(
        subj=args.subj,
        model="resnet50_bottleneck",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
    )

    # for i in range(7):
    #     volumes["clip-RN-%s" % str(i + 1)] = make_volume(
    #         subj=args.subj,
    #         model="visual_layer_resnet_%d" % i,
    #         output_root=output_root,
    #         mask_with_significance=args.mask_sig,
    #     )
    # volumes["clip-RN-last"] = make_volume(
    #     subj=args.subj,
    #     model="clip_visual_resnet",
    #     output_root=output_root,
    #     mask_with_significance=args.mask_sig,
    # )

    volumes["BERT-last"] = make_volume(
        subj=args.subj,
        model="bert_layer_13",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
    )
    
    volumes["clip-ViT-last R^2"] = make_volume(
        subj=args.subj,
        model="clip",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["bert-13 R^2"] = make_volume(
        subj=args.subj,
        model="bert_layer_13",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["resnet50 R^2"] = make_volume(
        subj=args.subj,
        model="resnet50_bottleneck",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["clip&resnet50-clip R^2"] = make_volume(
        subj=args.subj,
        model=["resnet50_bottleneck_clip", "clip_resnet50_bottleneck"],
        model2 = "clip",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["clip&resnet50-resnet50 R^2"] = make_volume(
        subj=args.subj,
        model=["resnet50_bottleneck_clip", "clip_resnet50_bottleneck"],
        model2 = "resnet50_bottleneck",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["clip&bert13-bert13 R^2"] = make_volume(
        subj=args.subj,
        model=["clip_bert_layer_13", "bert_layer_13_clip"],
        model2="bert_layer_13",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["clip&bert13-clip R^2"] = make_volume(
        subj=args.subj,
        model=["clip_bert_layer_13","bert_layer_13_clip"],
        model2="clip",
        output_root=output_root,
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    # for i in range(13):
    #     volumes["bert-%s" % str(i + 1)] = make_volume(
    #         subj=args.subj,
    #         model="bert_layer_%d" % (i + 1),
    #         output_root=output_root,
    #         mask_with_significance=args.mask_sig,
    #     )

    # volumes["clip-ViT-layerwise"] = visualize_layerwise_max_corr_results(
    #     "visual_layer", 12, threshold=85, mask_with_significance=args.mask_sig
    # )
    # volumes["clip-RN-layerwise"] = visualize_layerwise_max_corr_results(
    #     "visual_layer_resnet", 7, threshold=85, mask_with_significance=args.mask_sig
    # )
    # volumes["clip-text-layerwise"] = visualize_layerwise_max_corr_results(
    #     "text_layer", 12, threshold=85, mask_with_significance=args.mask_sig
    # )
    # volumes["bert-layerwise"] = visualize_layerwise_max_corr_results(
    #     "bert_layer", 13, threshold=85, mask_with_significance=args.mask_sig, start_with_zero=False
    # )

    if args.show_pcs:
        pc_vols = []
        PCs = np.load(
            "%s/output/pca/subj%d/pca_components.npy" % (output_root, args.subj)
        )
        # Normalize the PCs

        from util.util import zscore

        PCs_zscore = zscore(PCs, axis=1)

        # norm_PCs = PCs / np.sum(PCs, axis=1, keepdims=True)
        for i in range(PCs.shape[0]):
            key = "PC" + str(i)
            volumes[key] = make_pc_volume(
                args.subj,
                PCs_zscore[i, :],
                mask_with_significance=args.mask_sig,
                output_root=output_root,
            )

        volumes["3PC"] = make_3pc_volume(
            args.subj,
            PCs_zscore,
            mask_with_significance=args.mask_sig,
            output_root=output_root,
        )
    subj_port = "1111" + str(args.subj)
    cortex.webgl.show(data=volumes, autoclose=False, port=int(subj_port))

    import pdb

    pdb.set_trace()
