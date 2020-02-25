import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

roi_path = "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/"
beta_path = "/lab_data/tarrlab/common/datasets/NSD/nsddata_betas/ppdata/"

def extract_cortical_mask(subj, roi_only):
    roi_subj_path = "%s/subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (roi_path, subj)
    anat = nib.load(roi_subj_path)
    anat_mat = anat.get_data()
    if roi_only:
        mask = anat_mat > 0
        np.save("output/cortical_mask_subj%02d_roi_only.npy" % subj, mask)
        return mask
    else:
        mask = anat_mat > -1
        np.save("output/cortical_mask_subj%02d.npy" % subj, mask)
        return mask

def extract_voxels(subj, roi_only):
    beta_subj_dir = "%s/subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR" % (beta_path, subj)
    if roi_only:
        mask_tag = "_roi_only"
    else:
        mask_tag = ""

    try:
        mask = np.load("output/cortical_mask_subj%02d%s" % (subj, mask_tag))
    except FileNotFoundError:
        mask = extract_cortical_mask(subj, roi_only)

    cortical_beta_mat = None
    for ses in tqdm(range(1,41)):
        beta_file = nib.load("%s/betas_session%02d.nii.gz" % (beta_subj_dir, ses))
        beta = beta_file.get_data()
        cortical_beta = (beta[mask]).T #verify the mask with array

        if cortical_beta_mat is None:
            cortical_beta_mat = cortical_beta/300
        else:
            cortical_beta_mat = np.vstack((cortical_beta_mat, cortical_beta/300))

    np.save(
        "output/cortical_voxel_across_sessions_subj%02d%s.npy" % (subj, mask_tag), cortical_beta_mat
    )
    return cortical_beta_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, help="Subject number (from 1 to 8)")
    parser.add_argument("--all_subj", type=bool, help="extract cortical voxel for all subjects")
    parser.add_argument("--use_roi_only", type=bool, help="only extract voxels related to rois")

    args = parser.parse_args()
    if args.all_subj:
        subj = ["0" + str(i) for i in np.arange(1,9)]
    else:
        subj = [args.subj]

    for s in subj:
        extract_voxels(s, args.use_roi_only)

