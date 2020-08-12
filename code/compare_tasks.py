"""
This scripts computes the similarity or distance matrix of tasks based on prediction
on all voxels of the whole brain.
"""
import pickle
import argparse
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import ranksums

from visualize_corr_in_pycortex import load_data
from util.model_config import visual_roi_names, place_roi_names, ecc_roi_names

sns.set(style="whitegrid", font_scale=1)


# def cross_subject_corrs(mat_list):
#     rs = list()
#     for i in range(len(mat_list) - 1):
#         for j in range(len(mat_list) - 1):
#             if i != j + 1:
#                 r = pearsonr(mat_list[i].flatten(), mat_list[j + 1].flatten())
#                 rs.append(r[0])
#     return rs


def load_prediction(model, task, subj=1):
    output = pickle.load(
        open(
            "output/encoding_results/subj%d/pred_%s_%s_whole_brain.p"
            % (subj, model, task),
            "rb",
        )
    )
    out = np.array(output)
    return out


def get_voxels(model_list, subj):
    datamat = list()
    for l in model_list:
        data = load_data("taskrepr", task=l, subj=subj, measure="corr")
        datamat.append(data)
    datamat = np.array(datamat)
    return datamat


def get_sig_mask(model_list, correction, alpha, subj):
    maskmat = list()
    for l in model_list:
        mask = np.load(
            "output/voxels_masks/subj%d/taskrepr_%s_%s_%s.npy"
            % (subj, l, correction, alpha)
        )
        maskmat.append(mask)
    maskmat = np.array(maskmat)
    return maskmat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="please specify subject to show")

    parser.add_argument(
        "--use_voxel_prediction", default=False, action="store_true",
    )
    parser.add_argument("--subj", default=1, type=int, help="define which subject")

    parser.add_argument(
        "--method",
        default="cosine",
        help="define what metric should be used to generate task matrix",
    )

    parser.add_argument(
        "--no_masked", action="store_true", help="do not use significance mask"
    )

    parser.add_argument(
        "--task_list",
        nargs="+",
        type=str,
        default=["vanishing_point", "inpainting_whole", "edge2d", "edge3d", "room_layout", "class_places"],
    )

    args = parser.parse_args()

    voxel_mat = get_voxels(args.task_list, subj=args.subj)
    print(voxel_mat.shape)

    if not args.no_masked:
        sig_mat = get_sig_mask(
            args.task_list, correction="emp_fdr", alpha=0.05, subj=args.subj
        )
        assert voxel_mat.shape == sig_mat.shape
        # masking out insignificant voxels in prediction
        voxel_mat[~sig_mat] = 0
        sig_mask_tag = "_emp_fdr_0.05"
    else:
        sig_mask_tag = "_no_sig_mask"

    roi_tag = ""
    cortical_mask = np.load(
        "output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (args.subj, args.subj)
    )

    visual_rois = np.load(
        "output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
        % (args.subj, args.subj, "prf-visualrois")
    )
    ecc_rois = np.load(
        "output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
        % (args.subj, args.subj, "prf-eccrois")
    )
    place_rois = np.load(
        "output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
        % (args.subj, args.subj, "floc-places")
    )
    print(voxel_mat.shape[1])
    print(len(visual_rois))
    print(len(ecc_rois))
    assert voxel_mat.shape[1] == len(visual_rois)
    assert voxel_mat.shape[1] == len(ecc_rois)
    assert voxel_mat.shape[1] == len(place_rois)

    # create dataframe
    dfpath = "output/dataframes/correlations_subj%d%s.csv" % (args.subj, sig_mask_tag)

    try:
        df = pd.read_csv(dfpath)
        assert voxel_mat.shape[1] == len(df)
        for i, task in enumerate(args.task_list):
            if task not in df.columns:
                df[task] = voxel_mat[i, :]

        df.to_csv(dfpath)

    except FileNotFoundError:
        print("Making a new dataframe...")
        task_cols = [task for task in args.task_list]
        cols = ["ecc_rois", "place_rois", "visual_rois"] + task_cols
        df = pd.DataFrame(columns=cols)

        for vox_num in range(voxel_mat.shape[1]):
            vd = dict()
            vd["ecc_rois"] = ecc_roi_names[ecc_rois[vox_num]]
            vd["place_rois"] = place_roi_names[place_rois[vox_num]]
            vd["visual_rois"] = visual_roi_names[visual_rois[vox_num]]

            for i, task in enumerate(args.task_list):
                vd[task] = voxel_mat[i, vox_num]
                df = df.append(vd, ignore_index=True)

        df.to_csv(dfpath)

    if args.use_voxel_prediction:
        assert len(args.task_list) == 2
        pred1 = load_prediction(
            "taskrepr", args.task_list[0], subj=args.subj
        )
        # print(pred1.shape)
        pred2 = load_prediction(
            "taskrepr", args.task_list[1], subj=args.subj
        )
        print("number of voxels is: " + str(pred1[0].shape[1]))
        assert pred1[1].all() == pred2[1].all() #make sure the testing examples are the same
        rs_output = list()
        for i in range(pred1[0].shape[1]):
            rs_output.append(ranksums(pred1[0][:, i], pred2[0][:, i])[0])

        np.save(
            "output/comparisons/ranksums_of_pred_of_%s_and_%s_subj%d.npy"
            % (args.task_list[0], args.task_list[1], args.subj),
            np.array(rs_output),
        )
