import argparse
import numpy as np

from util.model_config import place_roi_names, face_roi_names
from util.util import pearson_corr

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output")

    args = parser.parse_args()

    brain_path = (
            "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
            % (args.output_dir, args.subj)
        )

    br_data = np.load(brain_path)

    place = np.load("%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_floc-places.npy" % (args.output_dir, args.subj, args.subj))

    place_rdm = list()
    for i in place_roi_names.keys():
        if i > 0:
            roi_response = br_data[:, place == i]
            place_rdm.append(np.corrcoef(roi_response))
    
    np.save("%s/rdms/subj%02d_places.npy" % (args.output_dir, args.subj), np.array(place_rdm))

    face = np.load("%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_floc-faces.npy" % (args.output_dir, args.subj, args.subj))
    face_rdm = list()
    for i in face_roi_names.keys():
        if i > 0:
            roi_response = br_data[:, face == i]
            face_rdm.append(np.corrcoef(roi_response))
    np.save("%s/rdms/subj%02d_faces.npy" % (args.output_dir, args.subj), np.array(face_rdm))


    # stimulus_list = np.load(
    #         "%s/coco_ID_of_repeats_subj%02d.npy" % (args.output_dir, args.subj)
    #     )