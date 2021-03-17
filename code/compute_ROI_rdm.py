import argparse
import numpy as np

from util.model_config import *


def add_roi_to_voxel_selected(roi_list, voxel_mask):
    for roi_name in roi_list:
        roi_mask = np.load(
            "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_dir, args.subj, args.subj, roi_name)
        )
        for i in roi_name_dict[roi_name].keys():
            if i > 0:
                voxel_mask[roi_mask == i] = voxel_mask[roi_mask == i] + 1

    return voxel_mask

def extract_single_roi(roi_name):
    output_masks, roi_labels = list(), list()
    roi_mask = np.load(
            "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy"
            % (args.output_dir, args.subj, args.subj, roi_name)
        )
    roi_dict = roi_name_dict[roi_name]
    for k, v in roi_dict.items():
        if k > 0:
            output_masks.append(roi_mask == k)
            roi_labels.append(v)
    return output_masks, roi_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output"
    )
    parser.add_argument("--combined_roi", action="store_true")
    parser.add_argument("--single_roi", type=str, nargs="+")

    args = parser.parse_args()

    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )

    br_data = np.load(brain_path)
    voxel_selected = np.zeros(br_data.shape[1], dtype=bool)
    
    if args.combined_roi:
        roi_list = ["floc-words", "floc-faces", "floc-places", "prf-visualrois"]
        voxel_selected = add_roi_to_voxel_selected(roi_list, voxel_selected)
        selected_brain_data = br_data[:, voxel_selected]
        rdm = np.corrcoef(selected_brain_data)
        np.save(
            "%s/rdms/subj%02d_%s.npy" % (args.output_dir, args.subj, "_".join(roi_list)),
            rdm,
        )
    else:
        print("extracting RDMs for these rois: " + str(args.single_roi))
        for roi in args.single_roi:
            roi_masks, roi_labels = extract_single_roi(roi)
            for i, m in enumerate(roi_masks):
                brain_data = br_data[:, m]
                rdm = np.corrcoef(brain_data)
                np.save("%s/rdms/subj%02d_%s" % (args.output_dir, args.subj, roi_labels), rdm)