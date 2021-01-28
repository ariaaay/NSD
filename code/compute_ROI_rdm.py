import argparse
import numpy as np

from util.model_config import *

def add_roi_to_voxel_selected(roi_list, voxel_mask):
    for roi_name in roi_list:
        roi_mask = np.load("%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_%s.npy" % (args.output_dir, args.subj, args.subj, roi_name))
        for i in roi_name_dict[roi_name].keys():
            if i > 0:
                voxel_mask[roi_mask == i] = voxel_mask[roi_mask == i] + 1
    
    return voxel_mask
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output")

    args = parser.parse_args()

    brain_path = (
            "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
            % (args.output_dir, args.subj)
        )

    roi_list = ["floc-words", "floc-faces", "floc-places", "prf-visualrois"]

    br_data = np.load(brain_path)
    voxel_selected = np.zeros(br_data.shape[1], dtype=bool)
    voxel_selected = add_roi_to_voxel_selected(roi_list, voxel_selected)

    selected_brain_data = br_data[:, voxel_selected]
    rdm = np.corrcoef(selected_brain_data)

    np.save("%s/rdms/subj%02d_%s.npy" % (args.output_dir, args.subj, "_".join(roi_list)), rdm)
    
            