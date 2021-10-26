import argparse
import numpy as np
import matplotlib.pyplot as plt

from util.data_util import find_trial_indexes

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output",
    )
    parser.add_argument("--subj", type=int, default=1)

    args = parser.parse_args()

    pidx, npidx = find_trial_indexes(subj=args.subj, cat="person")
    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )

    # Load brain data
    br_data = np.load(brain_path)
    print("Brain response size is: " + str(br_data.shape))

    roi_mask = np.load(
        "%s/voxels_masks/subj%01d/roi_1d_mask_subj%02d_floc-places.npy"
        % (args.output_dir, args.subj, args.subj)
    )
    roi_list = ["OPA", "PPA", "RSC"]
    masks = [roi_mask == i + 1 for i in range(3)]

    barplot_vals, errs = [], []
    for i, roi in enumerate(roi_list):
        roi_value_p = np.mean(br_data[pidx, :][:, masks[i]], axis=1)
        roi_value_np = np.mean(br_data[npidx, :][:, masks[i]], axis=1)
        errs.append(np.std(roi_value_np - roi_value_p))
        barplot_vals.append(np.mean(roi_value_np - roi_value_p))

    plt.bar(np.arange(3), barplot_vals, tick_label=roi_list, yerr=errs)
    plt.savefig("figures/person_vs_no_person/subj%02d.png" % args.subj)
