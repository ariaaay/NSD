import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.data_util import extract_single_roi
from encodingmodel.encoding_model import fit_encoding_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output"
    )

    args = parser.parse_args()

    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )
    br_data = np.load(brain_path)

    try:
        non_zero_mask = np.load(
            "%s/voxels_masks/subj%d/nonzero_voxels_subj%02d.npy"
            % (args.output_dir, args.subj, args.subj)
        )
        print("Masking zero voxels...")
        br_data = br_data[:, non_zero_mask]
    except FileNotFoundError:
        pass

    roi_list = ["floc-words", "floc-faces", "floc-places", "prf-visualrois"]

    roi_masks_list, roi_label_list = [], []

    for roi in roi_list:
        roi_masks, roi_labels = extract_single_roi(roi, args.output_dir, args.subj)
        roi_masks_list.append(roi_masks)
        roi_label_list.append(roi_labels)
    print(len(roi_masks_list))

    corrs_array = np.zeros((len(roi_masks_list, roi_masks_list)))
    for i, m1 in enumerate(tqdm(roi_masks)):
        for j, m2 in enumerate(tqdm(roi_masks)):
            roi1 = roi_label_list[i]
            roi2 = roi_label_list[j]
            if roi1 == roi2:
                corrs_array[i, j] = 1
            else:
                model_name_to_save = "%s_to_%s" % (roi1, roi2)
                feature = br_data[:, m1]
                bdata = br_data[:, m2]
                corrs, _ = fit_encoding_model(
                    feature,
                    bdata,
                    model_name=model_name_to_save,
                    subj=args.subj,
                    fix_testing=True,
                    cv=False,
                    saving=False,
                    output_dir=args.output_dir,
                )

                corrs_array[i, j] = np.mean(corrs[:, 0])
    np.save("%s/roi_gen/corrs_mean.npy")

    plt.imshow(corrs_array)
    plt.colorbar()
    plt.xlabel(roi_label_list)
    plt.savefig("figures/roi_gen.png")
