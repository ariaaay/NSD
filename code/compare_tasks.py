"""
This scripts computes the similarity or distance matrix of tasks based on prediction
on all voxels of the whole brain.
"""
import pickle
import argparse
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import ranksums

from visualize_corr_in_pycortex import load_data
from util.model_config import taskrepr_features, task_label

sns.set(style="whitegrid", font_scale=1)


# def cross_subject_corrs(mat_list):
#     rs = list()
#     for i in range(len(mat_list) - 1):
#         for j in range(len(mat_list) - 1):
#             if i != j + 1:
#                 r = pearsonr(mat_list[i].flatten(), mat_list[j + 1].flatten())
#                 rs.append(r[0])
#     return rs


def load_prediction(model, task, subj=1, measure="pred"):
    output = pickle.load(
        open(
            "output/encoding_results/subj%d/%s_%s_%s_whole_brain.pkl"
            % (subj, measure, model, task),
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
        "--task_list", type=list, default=["inpainting_whole", "edge3d", "room_layout"]
    )
    parser.add_argument(
        "--use_voxel_prediction", default=False, action="store_true",
    )
    parser.add_argument("--subj", default=1, type=int, help="define which subject")

    parser.add_argument(
        "--method",
        default="cosine",
        help="define what metric should be used to generate task matrix",
    )

    args = parser.parse_args()

    voxel_mat = get_voxels(args.task_list, subj=args.subj)

    sig_mat = get_sig_mask(
        args.task_list, correction="emp_fdr", alpha=0.05, subj=args.subj
    )
    assert voxel_mat.shape == sig_mat.shape

    roi_tag = ""
    cortical_mask = np.load(
        "output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (args.subj, args.subj)
    )

    # masking out insignificant voxels in prediction
    voxel_mat[:,~sig_mat] = 0

    # load ROI mask and mask brain prediction if using
    # if args.roi is not None:
    #     roi_1d_mask = np.load(
    #         "output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
    #         % (args.subj, args.subj, args.roi)
    #     )
    #
    #     assert voxel_mat.shape == roi_1d_mask.shape
    #
    #     voxel_mat[roi_1d_mask < 1] = 0
    #     roi_tag += "_only_%s" % args.roi
    #
    # elif args.exclude_roi is not None:
    #     roi_excluded_1d_mask = np.load(
    #         "output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy"
    #         % (args.subj, args.subj, args.exclude_roi)
    #     )
    #
    #     assert voxel_mat.shape[1] == len(roi_excluded_1d_mask)
    #     voxel_mat[:, roi_excluded_1d_mask] = 0
    #     roi_tag += "_exclude_%s" % args.exclude_roi

    visual_rois = np.load("output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy" % (args.subj, args.subj, "prf-visualrois"))
    ecc_rois = np.load("output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy" % (args.subj, args.subj, "prf-eccrois"))
    place_rois = np.load("output/voxels_masks/subj%d/roi_1d_mask_subj%02d_%s.npy" % (args.subj, args.subj, "floc-places"))
    assert voxel_mat.shape == visual_rois.shape
    assert voxel_mat.shape == ecc_rois.shape
    assert voxel_mat.shape == place_rois.shape

    # create dataframe
    dfpath = "output/dataframes/correlations_subj%d.csv" % args.subj

    try:
        df = pd.read_csv(dfpath)
    except FileNotFoundError:
        print("Making a new dataframe...")
        task_cols = [task for task in args.task_list]
        cols = ["ecc_rois", "place_rois", "visual_rois"] + task_cols
        df = pd.DataFrame(columns=cols)

        for vox_num in range(voxel_mat.shape[1]):
            vd = dict()
            vd["ecc_rois"] = ecc_rois[vox_num]
            vd["place_rois"] = place_rois[vox_num]
            vd["visual_rois"] = visual_rois[vox_num]

            for i, task in enumerate(args.task_list):
                vd[task] = voxel_mat[i, vox_num]

        pd.to_csv(dfpath)

