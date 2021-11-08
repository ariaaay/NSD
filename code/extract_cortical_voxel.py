import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

roi_path = "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/"
beta_path = "/lab_data/tarrlab/common/datasets/NSD/nsddata_betas/ppdata/"


def zscore_by_run(mat, run_n=480):
    from scipy.stats import zscore

    run_n = np.ceil(mat.shape[0] / 62.5) # should be 480 for subject with full experiment\

    zscored_mat = np.zeros(mat.shape)
    index_so_far = 0
    for i in tqdm(range(run_n)):
        if i % 2 == 0:
            zscored_mat[index_so_far : index_so_far + 62, :] = zscore(
                mat[index_so_far : index_so_far + 62, :]
            )
            index_so_far += 62
        else:
            zscored_mat[index_so_far : index_so_far + 63, :] = zscore(
                mat[index_so_far : index_so_far + 63, :]
            )
            index_so_far += 63

    return zscored_mat


def extract_cortical_mask(
    subj, roi=""
):
    if roi != "":
        roi_tag = "_" + roi
    else:
        roi_tag = ""

    nsd_general_path = "%s/subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (
        roi_path,
        subj,
    )
    nsd_general = nib.load(nsd_general_path)
    nsd_cortical_mat = nsd_general.get_fdata()

    if roi == "general" or roi == "":
        anat_mat = nsd_cortical_mat
    else:
        roi_subj_path = "%s/subj%02d/func1pt8mm/roi/%s.nii.gz" % (roi_path, subj, roi)
        anat = nib.load(roi_subj_path)
        anat_mat = anat.get_fdata()

    if roi == "":  # cortical
        mask = anat_mat > -1
    else:  # roi
        mask = anat_mat > 0

        # save a 1D version as well
        cortical = nsd_cortical_mat > -1
        print("from NSD general, cortical voxel number is: %d." % np.sum(cortical))
        roi_1d_mask = anat_mat[cortical]
        # assert np.sum(roi_1d_mask) == np.sum(mask)
        print("Number of non-zero ROI voxels: " + str(np.sum(roi_1d_mask > 0)))
        print("Number of cortical voxels is: " + str(len(roi_1d_mask)))
        assert len(roi_1d_mask) == np.sum(
            cortical
        )  # check the roi 1D length is same as cortical numbers in nsd general
        np.save(
            "%s/voxels_masks/subj%d/roi_1d_mask_subj%02d%s.npy"
            % (args.output_dir, subj, subj, roi_tag),
            roi_1d_mask,
        )

    np.save(
        "%s/voxels_masks/subj%d/cortical_mask_subj%02d%s.npy"
        % (args.output_dir, subj, subj, roi_tag),
        mask,
    )

    return mask


def extract_voxels(
    subj,
    roi,
    zscore,
    mask=None,
    mask_tag="",
):
    tag = roi

    if zscore:
        zscore_tag = "zscored_by_run_"
    else:
        zscore_tag = ""

    output_path = (
        "%s/cortical_voxels/cortical_voxel_across_sessions_%ssubj%02d%s.npy"
        % (args.output_dir, zscore_tag, subj, mask_tag)
    )

    beta_subj_dir = "%s/subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR" % (
        beta_path,
        subj,
    )
    if mask is None:
        try:
            mask = np.load(
                "%s/voxels_masks/subj%d/cortical_mask_subj%02d%s.npy"
                % (args.output_dir, subj, subj, tag)
            )
        except FileNotFoundError:
            mask = extract_cortical_mask(subj, roi)

    cortical_beta_mat = None
    for ses in tqdm(range(1, 41)):
        beta_file = nib.load("%s/betas_session%02d.nii.gz" % (beta_subj_dir, ses))
        beta = beta_file.get_fdata()
        cortical_beta = (beta[mask]).T  # verify the mask with array

        if cortical_beta_mat is None:
            cortical_beta_mat = cortical_beta / 300
        else:
            cortical_beta_mat = np.vstack((cortical_beta_mat, cortical_beta / 300))
    
    print("NaN Values:" + str(np.any(np.isnan(cortical_beta_mat))))
    print("Is finite:" + str(np.all(np.isfinite(cortical_beta_mat))))

    if zscore:
        print("Zscoring...")
        cortical_beta_mat = zscore_by_run(cortical_beta_mat)
        finite_flag = np.all(np.isfinite(cortical_beta_mat))
        print("Is finite:" + str(finite_flag))

        if finite_flag == False:
            nonzero_mask = np.sum(np.isfinite(cortical_beta_mat), axis=0) != cortical_beta_mat.shape[0]
            np.save("%s/cortical_voxels/nonzero_mask_subj%02d.npy" % (args.output_dir, subj), nonzero_mask)
        
    np.save(args.output_path, cortical_beta_mat)
    return cortical_beta_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj", type=int, default=1, help="Subject number (from 1 to 8)"
    )
    parser.add_argument(
        "--all_subj", type=bool, help="extract cortical voxel for all subjects"
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help="extract voxels related to rois. Choices: general, face, words, kastner2015. "
        "Input arguments are files names of ROIs in "
        "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/subj01/func1pt8mm/roi",
    )
    parser.add_argument(
        "--zscore_by_run", default=False, action="store_true", help="zscore brain data by runs"
    )
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help="only extract roi mask but not voxel response",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output"
    )

    args = parser.parse_args()

    if args.all_subj:
        subj = ["0" + str(i) for i in np.arange(1, 9)]
    else:
        subj = [args.subj]

    for s in subj:
        if not os.path.isdir("%s/voxels_masks/subj%d" % (args.output_dir, subj)):
            os.makedirs("%s/voxels_masks/subj%d" % (args.output_dir, subj))
        if args.mask_only:
            print("Extracting ROI %s for subj%d" % (args.roi, s))
            extract_cortical_mask(s, roi=args.roi)
        else:
            extract_voxels(s, args.roi, args.zscore_by_run)
