"This scripts visualize prediction performance with pycortex."

from nibabel.volumeutils import working_type
import numpy as np

import argparse
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm

from util.data_util import load_model_performance

OUTPUT_ROOT = "/user_data/yuanw3/project_outputs/NSD"


def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals


def project_vols_to_mni(subj, vol):
    import cortex

    xfm = "func1pt8_to_anat0pt8_autoFSbbr"
    # template = "func1pt8_to_anat0pt8_autoFSbbr"
    mni_transform = cortex.db.get_mnixfm("subj%02d" % subj, xfm)
    mni_vol = cortex.mni.transform_to_mni(vol, mni_transform)
    mni_data = mni_vol.get_fdata().T
    return mni_data


def visualize_layerwise_max_corr_results(
    model,
    layer_num,
    subj=1,
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
                model="%s_%d" % (model, i), output_root=OUTPUT_ROOT, subj=args.subj
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
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    )

    if mask_with_significance:
        if args.sig_method == "negtail_fdr":
            sig_mask = np.load(
                "%s/output/voxels_masks/subj%d/%s_%s_%0.2f.npy"
                % (OUTPUT_ROOT, subj, model, "negtail_fdr", 0.05)
            )
        else:
            pvalues = load_model_performance(
                model="%s_%d" % (model, layer_num - 1),
                output_root=OUTPUT_ROOT,
                subj=args.subj,
                measure="pvalue",
            )
            sig_mask = pvalues <= 0.05

    layeridx[~sig_mask] = -1

    # # projecting value back to 3D space
    all_vals = project_vals_to_3d(layeridx, cortical_mask)

    layerwise_volume = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
        vmin=0,
        vmax=layer_num,
    )
    return layerwise_volume


def make_volume(
    subj,
    model=None,
    vals=None,
    model2=None,
    mask_with_significance=False,
    measure="corr",
    noise_corrected=False,
):
    if measure == "rsq":
        vmax = 0.6
    else:
        vmax = 0.8
    if model2 is not None:
        vmax -= 0.3
    if noise_corrected:
        vmax = 0.85
    if measure == "pvalue":
        vmax = 0.06

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    # load correlation scores of cortical voxels
    if vals is None:
        if (
            type(model) == list
        ):  # for different naming convention for variance partitioning (only 1 should exist)
            model_list = model
            for model in model_list:
                try:
                    vals = load_model_performance(
                        model, output_root=OUTPUT_ROOT, subj=subj, measure=measure
                    )
                    break
                except FileNotFoundError:
                    continue
        else:
            vals = load_model_performance(
                model, output_root=OUTPUT_ROOT, subj=subj, measure=measure
            )

        if model2 is not None:  # for variance paritioning
            vals2 = load_model_performance(
                model2, output_root=OUTPUT_ROOT, subj=subj, measure=measure
            )
            vals = vals - vals2

        print("model:" + model)
        print("max:" + str(max(vals)))
    if mask_with_significance:
        if args.sig_method == "negtail_fdr":
            sig_mask = np.load(
                "%s/output/voxels_masks/subj%d/%s_%s_%0.2f.npy"
                % (OUTPUT_ROOT, subj, model, "negtail_fdr", 0.05)
            )

        elif args.sig_method == "nc":
            nc = np.load(
                "%s/output/noise_ceiling/subj%01d/ncsnr_1d_subj%02d.npy"
                % (OUTPUT_ROOT, subj, subj)
            )
            sig_mask = nc >= 0.1

        else:
            pvalues = load_model_performance(
                model, output_root=OUTPUT_ROOT, subj=subj, measure="pvalue"
            )
            sig_mask = pvalues <= 0.05

        if measure == "pvalue":
            vals[~sig_mask] = np.nan
        else:
            vals[~sig_mask] = np.nan

    if (measure == "rsq") and (noise_corrected):
        vals = vals / nc
        vals[np.isnan(vals)] = np.nan

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


def make_pc_volume(subj, vals, vmin=-2, vmax=2, cmap="BrBG_r"):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


def make_3pc_volume(subj, PCs):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )

    pc_3d = []
    for i in range(3):
        tmp = PCs[i, :] / np.max(PCs[i, :]) * 255
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
    roi = nib.load("%s/%s.nii.gz" % (ROI_FILE_ROOT, roi_name))
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
    # parser.add_argument("--with_noise_ceiling", default=False, action="store_true")
    parser.add_argument("--show_more", action="store_true")
    parser.add_argument("--vis_method", type=str, default="webgl")

    args = parser.parse_args()

    if args.on_cluster:
        ROI_FILE_ROOT = (
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/subj%02d/func1pt8mm/roi"
            % args.subj
        )
    else:
        OUTPUT_ROOT = "."
        ROI_FILE_ROOT = "./roi_data/subj%02d" % args.subj

    # visual_roi_volume = make_roi_volume("prf-visualrois")
    # ecc_roi_volume = make_roi_volume("prf-eccrois")
    # place_roi_volume = make_roi_volume("floc-places")
    # face_roi_volume = make_roi_volume("floc-faces")
    # body_roi_volume = make_roi_volume("floc-bodies")
    # word_roi_volume = make_roi_volume("floc-words")
    # kastner_volume = make_roi_volume("Kastner2015")
    # hcp_volume = make_roi_volume("HCP_MMP1")
    # sulc_volume = make_roi_volume("corticalsulc")

    # lang_ROI = np.load(
    #     "./output/voxels_masks/language_ROIs.npy", allow_pickle=True
    # ).item()
    # language_vals = lang_ROI["subj%02d" % args.subj]
    # language_volume = cortex.Volume(
    #     language_vals,
    #     "subj%02d" % args.subj,
    #     "func1pt8_to_anat0pt8_autoFSbbr",
    #     mask=cortex.utils.get_cortical_mask(
    #         "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
    #     ),
    #     vmin=np.min(language_vals),
    #     vmax=np.max(language_vals),
    # )

    # ev_vals = np.load("%s/output/evs_subj%02d_zscored.npy" % (OUTPUT_ROOT, args.subj))
    # ev_volume = make_volume(subj=args.subj, vals=ev_vals, measure="rsq")

    # old_ev_vals = np.load("%s/output/evs_old_subj%02d_zscored.npy" % (OUTPUT_ROOT, args.subj))
    # old_ev_volume = make_volume(subj=args.subj, vals=old_ev_vals, measure="rsq")
    # nc = np.load(
    #     "%s/output/noise_ceiling/subj%01d/ncsnr_1d_subj%02d.npy"
    #     % (OUTPUT_ROOT, args.subj, args.subj)
    # )
    # nc_volume = make_volume(subj=args.subj, vals=nc, measure="rsq")

    # food = np.load("%s/output/subj01_food_v_all_FDR.npy" % (OUTPUT_ROOT))
    # food_volume = make_volume(subj=args.subj, vals=food, measure="pvalue", mask_with_significance=True)

    # Food maks
    # regions_nums_to_include = [136, 138, 163, 7, 22, 154, 6] #"TE2p", "PH", "VVC", "v8", "PIT", "VMV3", "v4"
    # food_mask = np.zeros(hcp_volume.data.shape)
    # for region_num in regions_nums_to_include:
    #     food_mask[np.where(hcp_volume.data == region_num)] = 1
    # food_mask_volume = cortex.Volume(
    #     food_mask,
    #     "subj%02d" % args.subj,
    #     "func1pt8_to_anat0pt8_autoFSbbr",
    #     mask=cortex.utils.get_cortical_mask(
    #         "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
    #     ),
    #     vmin=np.min(food_mask),
    #     vmax=np.max(food_mask),
    # )

    volumes = {
        # "Visual ROIs": visual_roi_volume,
        # "Eccentricity ROIs": ecc_roi_volume,
        # "Places ROIs": place_roi_volume,
        # "Faces ROIs": face_roi_volume,
        # "Bodies ROIs": body_roi_volume,
        # "Words ROIs": word_roi_volume,
        # "Kastner2015": kastner_volume,
        # "HCP": hcp_volume,
        # "sulcus": sulc_volume,
        # "Language ROIs": language_volume,
        # "Noise Ceiling": nc_volume,
        # "EV": ev_volume,
        # "EV - old": old_ev_volume,
        # "food": food_volume,
        # "food_mask": food_mask_volume
    }

    volumes["clip-ViT-last r"] = make_volume(
        subj=args.subj,
        model="clip",
        mask_with_significance=args.mask_sig,
    )

    # volumes["clip-RN50-last r"] = make_volume(
    #     subj=args.subj,
    #     model="clip_visual_resnet",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["clip-text-last r"] = make_volume(
    #     subj=args.subj,
    #     model="clip_text",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["resnet50 r"] = make_volume(
    #     subj=args.subj,
    #     # model="convnet_res50",
    #     model="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    # )

    # volumes["BERT-last r"] = make_volume(
    #     subj=args.subj,
    #     model="bert_layer_13",
    #     mask_with_significance=args.mask_sig,
    # )

    # rsquare
    volumes["clip-ViT-last R^2 NC"] = make_volume(
        subj=args.subj,
        model="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=True,
    )

    volumes["clip-ViT-last R^2"] = make_volume(
        subj=args.subj,
        model="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
        noise_corrected=False,
    )

    # volumes["clip-text-last R^2"] = make_volume(
    #     subj=args.subj,
    #     model="clip_text",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    #     noise_corrected=False,
    # )

    # volumes["clip-RN50-last R^2"] = make_volume(
    #     subj=args.subj,
    #     model="clip_visual_resnet",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["bert-13 R^2"] = make_volume(
    #     subj=args.subj,
    #     model="bert_layer_13",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     # model="convnet_res50",
    #     model="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&resnet50-clip ViT R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip",
    #     ],
    #     model2="clip",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&resnet50-clip RN50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_visual_resnet_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip_visual_resnet",
    #     ],
    #     model2="clip_visual_resnet",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip ViT&resnet50-resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip",
    #     ],
    #     model2="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip RN50&resnet50-resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=[
    #         # "convnet_res50_clip",
    #         # "clip_convnet_res50",
    #         "clip_visual_resnet_resnet50_bottleneck",
    #         "resnet50_bottleneck_clip_visual_resnet",
    #     ],
    #     model2="resnet50_bottleneck",
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    volumes["clip&bert13-bert13 R^2"] = make_volume(
        subj=args.subj,
        model=["clip_bert_layer_13", "bert_layer_13_clip"],
        model2="bert_layer_13",
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    volumes["clip&bert13-clip R^2"] = make_volume(
        subj=args.subj,
        model=["clip_bert_layer_13", "bert_layer_13_clip"],
        model2="clip",
        mask_with_significance=args.mask_sig,
        measure="rsq",
    )

    # volumes["clip&bert13"] = make_volume(
    #     subj=args.subj,
    #     model=["clip_bert_layer_13", "bert_layer_13_clip"],
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # volumes["clip&resnet50 R^2"] = make_volume(
    #     subj=args.subj,
    #     model=["clip_resnet50_bottleneck", "resnet50_bottleneck_clip"],
    #     mask_with_significance=args.mask_sig,
    #     measure="rsq",
    # )

    # clipRN50_resnet50_unique = cortex.Volume2D(vol1, vol2, cmap="PU_PinkBlue_covar", vmin=0, vmax=0.15, vmin2=0, vmax2=0.15)
    # volumes["clipViT_v_resnet50_unique"] = cortex.Volume2D(volumes["clip&resnet50-clip ViT R^2"], volumes["clip ViT&resnet50-resnet50 R^2"], cmap="PU_BuOr_covar_alpha", vmin=0.02, vmax=0.1, vmin2=0.02, vmax2=0.1)
    # volumes["clipRN50_v_resnet50_unique"] = cortex.Volume2D(volumes["clip&resnet50-clip RN50 R^2"], volumes["clip RN50&resnet50-resnet50 R^2"], cmap="PU_BuOr_covar_alpha", vmin=0.02, vmax=0.1, vmin2=0.02, vmax2=0.1)
    volumes["clip_v_bert_unique"] = cortex.Volume2D(
        volumes["clip&bert13-clip R^2"],
        volumes["clip&bert13-bert13 R^2"],
        cmap="PU_BuOr_covar_alpha",
        vmin=0.02,
        vmax=0.1,
        vmin2=0.02,
        vmax2=0.1,
    )

    if args.subj == 1 & args.show_more:
        volumes["clip_top1_object"] = make_volume(
            subj=args.subj,
            model="clip_top1_object",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip_all_objects"] = make_volume(
            subj=args.subj,
            model="clip_object",
            mask_with_significance=args.mask_sig,
        )

        volumes["COCO categories -r^2"] = make_volume(
            subj=args.subj,
            model="cat",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["COCO super categories"] = make_volume(
            subj=args.subj,
            model="supcat",
            mask_with_significance=args.mask_sig,
        )

        volumes["CLIP&CLIPtop1 - top1"] = make_volume(
            subj=args.subj,
            model="clip_clip_top1_object",
            model2="clip_top1_object",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["CLIP&CLIPtop1 - CLIP"] = make_volume(
            subj=args.subj,
            model="clip_clip_top1_object",
            model2="clip",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["CLIP&Resnet50"] = make_volume(
            subj=args.subj,
            model="clip_resnet50_bottleneck",
            mask_with_significance=args.mask_sig,
        )

        # for model in ["resnet50_bottleneck", "clip", "cat"]:
        #     for subset in ["person", "giraffe", "toilet", "train"]:
        #         model_name = "%s_%s_subset" % (model, subset)

        #         volumes[model_name] = make_volume(
        #             subj=args.subj,
        #             model=model_name,
        #             mask_with_significance=args.mask_sig,
        #         )

        volumes["clip-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip-no-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_no_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["resnet-person-subset"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["resnet-no-person-subset"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_no_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["cat-person-subset"] = make_volume(
            subj=args.subj,
            model="cat_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip-top1-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_top1_object_person_subset",
            mask_with_significance=args.mask_sig,
        )

        volumes["clip-top1-no-person-subset"] = make_volume(
            subj=args.subj,
            model="clip_top1_object_no_person_subset",
            mask_with_significance=args.mask_sig,
        )

        # for i in range(12):
        #     volumes["clip-ViT-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="visual_layer_%d" % i,
        #         mask_with_significance=args.mask_sig,
        #     )

        # for i in range(12):
        #     volumes["clip-text-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="text_layer_%d" % i,
        #         mask_with_significance=args.mask_sig,
        #     )

        # for i in range(7):
        #     volumes["clip-RN-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="visual_layer_resnet_%d" % i,
        #         mask_with_significance=args.mask_sig,
        #     )
        volumes["clip-RN-last"] = make_volume(
            subj=args.subj,
            model="clip_visual_resnet",
            mask_with_significance=args.mask_sig,
        )

        # volumes["clip&bert13 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="clip_bert_layer_13_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&resnet50-clip R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["resnet50_bottleneck_clip", "clip_resnet50_bottleneck"],
        #     model2="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip-ViT-last R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip-ViT-last R^2 - old(rerun)"] = make_volume(
        #     subj=args.subj,
        #     model="clip_old_rerun",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&resnet50 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="clip_convnet_res50",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )
        # volumes["resnet50 - old"] = make_volume(
        #     subj=args.subj,
        #     model="resnet50_bottleneck",
        #     mask_with_significance=args.mask_sig,
        # )

        # volumes["bert-13 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="bert_layer_13_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["resnet50 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model="resnet50_bottleneck",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&resnet50-resnet50 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["resnet50_bottleneck_clip", "clip_resnet50_bottleneck"],
        #     model2="resnet50_bottleneck",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )
        # volumes["clip&bert13-bert13 R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["clip_bert_layer_13_old", "bert_layer_13_clip_old"],
        #     model2="bert_layer_13",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip&bert13-clip R^2 - old"] = make_volume(
        #     subj=args.subj,
        #     model=["clip_bert_layer_13_old", "bert_layer_13_clip_old"],
        #     model2="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip - clip(old) R^2"] = make_volume(
        #     subj=args.subj,
        #     model="clip",
        #     model2="clip_old",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        # volumes["clip - clip(old) R^2"] = make_volume(
        #     subj=args.subj,
        #     model="clip",
        #     model2="clip_old_rerun",
        #     mask_with_significance=args.mask_sig,
        #     measure="rsq",
        # )

        volumes["oscar"] = make_volume(
            subj=args.subj,
            model="oscar",
            mask_with_significance=args.mask_sig,
            measure="corr",
        )

        volumes["clip&oscar - oscar"] = make_volume(
            subj=args.subj,
            model="oscar_clip",
            model2="oscar",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["clip&oscar - clip"] = make_volume(
            subj=args.subj,
            model="oscar_clip",
            model2="clip",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["ResNet&oscar - ResNet"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_oscar",
            model2="resnet50_bottleneck",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        volumes["ResNet&oscar - oscar"] = make_volume(
            subj=args.subj,
            model="resnet50_bottleneck_oscar",
            model2="oscar",
            mask_with_significance=args.mask_sig,
            measure="rsq",
        )

        # for i in range(13):
        #     volumes["bert-%s" % str(i + 1)] = make_volume(
        #         subj=args.subj,
        #         model="bert_layer_%d" % (i + 1),
        #         mask_with_significance=args.mask_sig,
        #     )

        volumes["clip-ViT-layerwise"] = visualize_layerwise_max_corr_results(
            "visual_layer", 12, threshold=85, mask_with_significance=args.mask_sig
        )
        volumes["clip-RN-layerwise"] = visualize_layerwise_max_corr_results(
            "visual_layer_resnet", 7, threshold=85, mask_with_significance=args.mask_sig
        )
        # volumes["clip-text-layerwise"] = visualize_layerwise_max_corr_results(
        #     "text_layer", 12, threshold=85, mask_with_significance=args.mask_sig
        # )
        # volumes["bert-layerwise"] = visualize_layerwise_max_corr_results(
        #     "bert_layer", 13, threshold=85, mask_with_significance=args.mask_sig, start_with_zero=False
        # )

    if args.show_pcs:
        model = "clip"

        # name_modifier = "acc_0.3_minus_prf-visualrois"
        name_modifier = "best_20000"
        pc_vols = []
        PCs = np.load(
            "%s/output/pca/%s/subj%02d/%s_pca_group_components_%s.npy"
            % (OUTPUT_ROOT, model, args.subj, model, name_modifier)
        )
        subj_mask = np.load(
            "%s/output/pca/%s/pca_voxels/pca_voxels_subj%02d_%s.npy"
            % (OUTPUT_ROOT, model, args.subj, name_modifier)
        )
        PCs[:, ~subj_mask] = np.nan
        PC_val_only = PCs[:, subj_mask]

        # norm_PCs = PCs / np.sum(PCs, axis=1, keepdims=True)
        for i in range(PCs.shape[0]):
            key = "PC" + str(i)
            volumes[key] = make_pc_volume(
                args.subj,
                PCs[i, :],
            )

        # visualize PC projections
        subj_proj = np.load(
            "%s/output/pca/%s/subj%02d/%s_feature_pca_projections_%s.npy"
            % (OUTPUT_ROOT, model, args.subj, model, name_modifier)
        )

        for i in range(subj_proj.shape[0]):
            key = "PC Proj " + str(i)
            volumes[key] = make_pc_volume(
                args.subj,
                subj_proj[i, :],
            )

        

        # volumes["3PC"] = make_3pc_volume(
        #     args.subj,
        #     PCs_zscore,
        # )

        # basis?
        def kmean_sweep_on_PC(n_pc):
            from sklearn.cluster import KMeans

            inertia = []
            for k in range(3, 11):
                kmeans = KMeans(n_clusters=k, random_state=0).fit(
                    PC_val_only[:n_pc, :].T
                )
                labels = PCs[0, :].copy()
                labels[subj_mask] = kmeans.labels_ + 1
                volumes["basis %d-%d" % (n_pc, k)] = make_pc_volume(
                    args.subj, labels, vmin=1, vmax=k, cmap="J4s"
                )
                inertia.append(kmeans.inertia_)
            return inertia

        import matplotlib.pyplot as plt

        plt.figure()
        n_pcs = [3, 4, 5]
        for n in n_pcs:
            inertia = kmean_sweep_on_PC(n)
            plt.plot(inertia, label="%d PCS" % n)
        plt.savefig("figures/pca/clustering/inertia_across_pc_num.png")

        # MNI
        # mni_data = project_vols_to_mni(args.subj, volume)

        # mni_vol = cortex.Volume(
        #     mni_data,
        #     "fsaverage",
        #     "atlas",
        #     cmap="BrBG_r",
        # )
        # volumes["PC1 - MNI"] = mni_vol

        # # visualize PC projections
        # subj_proj = np.load(
        #             "%s/output/pca/%s/subj%02d/%s_feature_pca_projections.npy"
        #             % (OUTPUT_ROOT, model, args.subj, model)
        #         )
        # for i in range(PCs.shape[0]):
        #     key = "PC Proj " + str(i)
        #     volumes[key] = make_pc_volume(
        #         args.subj,
        #         subj_proj[i, :],
        #     )

        # cortex.quickflat.make_figure(mni_vol, with_roi=False)
        # print("***********")
        # print(volumes["PC1"])

    if args.show_clustering:
        model = "clip"

        # name_modifier = "acc_0.3_minus_prf-visualrois"
        name_modifier = "best_20000"
        labels_vals = np.load(
            "%s/output/clustering/spectral_subj%01d.npy"
            % (OUTPUT_ROOT, args.subj)
        )
        subj_mask = np.load(
            "%s/output/pca/%s/pca_voxels/pca_voxels_subj%02d_%s.npy"
            % (OUTPUT_ROOT, model, args.subj, name_modifier)
        )

        labels = np.zeros(subj_mask.shape)
        labels[~subj_mask] = np.nan
        labels[subj_mask] = labels_vals

    if args.vis_method == "webgl":
        subj_port = "4111" + str(args.subj)
        # cortex.webgl.show(data=volumes, autoclose=False, port=int(subj_port))
        cortex.webgl.show(data=volumes, port=int(subj_port), recache=False)

    elif args.vis_method == "quickflat":
        roi_list = ["RSC", "PPA", "OPA", "EarlyVis", "FFA-1", "FFA-2"]
        for k in volumes.keys():
            # vol_name = k.replace(" ", "_")
            filename = "./figures/flatmap/subj%d/%s.png" % (args.subj, k)
            _ = cortex.quickflat.make_png(
                filename,
                volumes[k],
                linewidth=3,
                with_curvature=True,
                recache=False,
                roi_list=roi_list,
            )

    elif args.vis_method == "3d_views":
        from save_3d_views import save_3d_views

        for k, v in volumes.items():
            root = "figures/3d_views/subj%s" % args.subj
            _ = save_3d_views(
                v,
                root,
                k,
                list_views=["lateral", "bottom", "back"],
                list_surfaces=["inflated"],
                with_labels=True,
                size=(1024 * 4, 768 * 4),
                trim=True,
            )

    import pdb

    pdb.set_trace()
