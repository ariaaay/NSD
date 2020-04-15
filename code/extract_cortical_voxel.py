import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm

roi_path = "/lab_data/tarrlab/common/datasets/NSD/nsddata/ppdata/"
beta_path = "/lab_data/tarrlab/common/datasets/NSD/nsddata_betas/ppdata/"


def zscore_by_run(mat, run_n=480):
    try:
        assert mat.shape[0] / run_n == 62.5
    except AssertionError:
        print("data has the wrong shape or run_number is wrong for zscoring by run.")

    from scipy.stats import zscore

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


def extract_cortical_mask(subj, roi=""):
    if roi == "general" or "":
        roi_subj_path = "%s/subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz" % (
            roi_path,
            subj,
        )
    else:
        roi_subj_path = "%s/subj%02d/func1pt8mm/roi/%s.nii.gz" % (roi_path, subj, roi)

    anat = nib.load(roi_subj_path)
    anat_mat = anat.get_data()

    if roi == "":
        mask = anat_mat > -1
    else:
        mask = anat_mat > 0

    np.save("output/cortical_mask_subj%02d_%s.npy" % (subj, roi), mask)
    return mask


def extract_voxels(subj, roi, zscore):
    tag = roi

    if zscore:
        tag += "_zscore"

    output_path = "output/cortical_voxel_across_sessions_subj%02d%s.npy" % (subj, tag,)

    try:
        cortical_beta_mat = np.load(output_path)
        print("Cortical voxels file already existed...")
    except FileNotFoundError:
        beta_subj_dir = "%s/subj%02d/func1pt8mm/betas_fithrf_GLMdenoise_RR" % (
            beta_path,
            subj,
        )

        try:
            mask = np.load("output/cortical_mask_subj%02d%s" % (subj, tag))
        except FileNotFoundError:
            mask = extract_cortical_mask(subj, roi)

        cortical_beta_mat = None
        for ses in tqdm(range(1, 41)):
            beta_file = nib.load("%s/betas_session%02d.nii.gz" % (beta_subj_dir, ses))
            beta = beta_file.get_data()
            cortical_beta = (beta[mask]).T  # verify the mask with array

            if cortical_beta_mat is None:
                cortical_beta_mat = cortical_beta / 300
            else:
                cortical_beta_mat = np.vstack((cortical_beta_mat, cortical_beta / 300))

        if zscore_by_run:
            cortical_beta_mat = zscore_by_run(cortical_beta_mat)

        np.save(output_path, cortical_beta_mat)
    return cortical_beta_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, help="Subject number (from 1 to 8)")
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
        "--zscore_by_run", action="store_true", help="zscore brain data by runs"
    )
    parser.add_argument(
        "--mask_only",
        action="store_true",
        help="only extract roi mask but not voxel response",
    )

    args = parser.parse_args()

    if args.all_subj:
        subj = ["0" + str(i) for i in np.arange(1, 9)]
    else:
        subj = [args.subj]

    for s in subj:
        if args.mask_only:
            print("Extracting ROI %s" % args.roi)
            extract_cortical_mask(subj, roi=args.roi)
        else:
            extract_voxels(s, args.roi, args.zscore_by_run)
