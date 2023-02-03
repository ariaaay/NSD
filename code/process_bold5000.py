from genericpath import exists
import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.stats import zscore

BOLD5000_root = "/lab_data/tarrlab/common/datasets/BOLD5000_GLMsingle"
roi_path = BOLD5000_root + "/betas/BOLD5000_ROIs"
beta_path = BOLD5000_root + "/betas/BOLD5000_complete"

# def extract_cortical_mask(subj, mask_name="", output_dir=None):
#     if output_dir is None:
#         output_dir = args.output_dir

#     path = "%s/masks/CSI%d_%s.nii.gz" % (
#         BOLD5000_root,
#         subj,
#         mask_name
#     )
#     anat_mat = nib.load(path).get_fdata()
#     cortical_mask = anat_mat > 0

#     print(
#         "Output cortical mask from %s, cortical voxel number is: %d."
#         % (mask_name, np.sum(cortical_mask))
#     )

#     output_dir_subj = "%s/voxels_masks/CSI%d/" % (output_dir, subj)
#     if not os.path.exists(output_dir_subj):
#         os.makedirs(output_dir_subj)

#     np.save(
#         "%s/cortical_mask_CSI%d_%s.npy" % (output_dir_subj, subj, mask_name),
#         cortical_mask,
#     )

#     return cortical_mask


def extract_voxels(
    subj,
):

    output_path = (
        "%s/cortical_voxels/cortical_voxel_across_sessions_zscored_CSI%d.npy"
        % (args.output_dir, subj)
    )

    mask = np.load("%s/voxels_masks/CSI%d/cortical_mask.npy" % (args.output_dir, subj))
    print("Subj:" + str(subj))
    print("Mask Shape:")
    print(mask.shape)

    cortical_beta_mat = None
    for ses in tqdm(range(1, 16)):
        try:
            beta_file = nib.load(
                "%s/CSI%d_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_ses-%02d.nii.gz"
                % (beta_path, subj, ses)
            )
        except FileNotFoundError:
            break
        beta = beta_file.get_fdata()
        cortical_beta = beta.T[:, mask]
        cortical_beta = zscore(cortical_beta)

        if cortical_beta_mat is None:
            cortical_beta_mat = cortical_beta  # TODO: check this divide by 300 step
        else:
            cortical_beta_mat = np.vstack((cortical_beta_mat, cortical_beta))

    print("NaN Values:" + str(np.any(np.isnan(cortical_beta_mat))))
    print("Is finite:" + str(np.all(np.isfinite(cortical_beta_mat))))

    if np.any(np.isnan(cortical_beta_mat)):
        print("Generating nonzero mask...")
        non_zero_mask = np.sum(np.isnan(cortical_beta_mat), axis=0) < 1
        np.save(
            "%s/voxels_masks/CSI%d/nonzero_voxels.npy" % (args.output_dir, subj),
            non_zero_mask,
        )

    print("NaN Values:" + str(np.any(np.isnan(cortical_beta_mat))))
    print("Is finite:" + str(np.all(np.isfinite(cortical_beta_mat))))

    np.save(output_path, cortical_beta_mat)
    return cortical_beta_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj", type=int, default=1, help="Subject number (from 1 to 4)"
    )
    parser.add_argument(
        "--all_subj",
        action="store_true",
        help="extract cortical voxel for all subjects",
    )
    # parser.add_argument(
    #     "--roi",
    #     type=str,
    #     default="",
    #     help="extract voxels related to rois. Choices: general, face, words, kastner2015. "
    #     "Input arguments are files names of ROIs in "
    #     "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/subj01/func1pt8mm/roi",
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/BOLD5000/outputs",
    )

    args = parser.parse_args()

    if args.all_subj:
        subjs = np.arange(1, 4)
    else:
        subjs = [args.subj]

    for subj in subjs:
        if not os.path.isdir("%s/voxels_masks/CSI%d" % (args.output_dir, subj)):
            os.makedirs("%s/voxels_masks/CSI%d" % (args.output_dir, subj))

    extract_voxels(subj)
