import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.data_util import extract_single_roi
from encodingmodel.encoding_model import fit_encoding_model

from scipy.stats import zscore

# def corr(X,Y,axis=0):
#     # computes the correlation of x1 with y1, x2 with y2, and so on
#     return np.mean(zscore(X,axis=axis)*zscore(Y,axis=axis),axis=axis)

# def crosscorr(X,Y):
#     # computes the pair-wise correlation of all variables in X with all variables in Y

#     nvars_x = X.shape[-1]
#     nvars_y = Y.shape[-1]

#     # num_samples = X.shape[0]

#     rep = np.float32(np.repeat(X,nvars_y,axis=1))
#     rep = np.reshape(rep, [-1, nvars_x, nvars_y])
#     rep2 = np.float32(np.repeat(Y,nvars_x,axis=1))
#     rep2 = np.reshape(rep2, [-1, nvars_y, nvars_x])
#     rep2 = np.swapaxes(rep2, 1, 2)

#     return corr(rep, rep2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output"
    )
    parser.add_argument("--data_to_data", type=bool, default=False)
    parser.add_argument("--pred_to_data", type=str, default=None)
    parser.add_argument("--roi", type=str, nargs="+", default="floc")

    args = parser.parse_args()

    if args.roi == "floc":
        roi_list = ["floc-faces", "floc-places", "floc-bodies", "prf-visualrois"]
    else:
        roi_list = list(args.roi)
    print(roi_list)

    if args.data_to_data:
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

        roi_masks_list, roi_label_list = [], []

        for roi in roi_list:
            roi_masks, roi_labels = extract_single_roi(roi, args.output_dir, args.subj)
            roi_masks_list += roi_masks
            roi_label_list += roi_labels
        print(len(roi_masks_list))
        print(roi_label_list)
        try:
            corrs_array = np.load(
                "%s/roi_gen/corrs_mean_subj%02d_%s.npy"
                % (args.output_dir, args.subj, args.roi)
            )
        except FileNotFoundError:
            corrs_array = np.zeros((len(roi_masks_list), len(roi_masks_list)))
            for i, m1 in enumerate(tqdm(roi_masks_list)):
                for j, m2 in enumerate(tqdm(roi_masks_list)):
                    roi1 = roi_label_list[i]
                    roi2 = roi_label_list[j]
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
                    np.save(
                        "%s/roi_gen/corrs_mean_subj%02d_%s.npy"
                        % (args.output_dir, args.subj, args.roi),
                        corrs_array,
                    )

        plt.imshow(corrs_array)
        plt.colorbar()
        plt.xticks(np.arange(len(roi_label_list)), roi_label_list, rotation=75)
        plt.yticks(np.arange(len(roi_label_list)), roi_label_list)
        plt.savefig("figures/roi_generalization/roi_gen.png")

    if args.pred_to_data is not None:
        nc = np.load(
            "%s/noise_ceiling/subj%01d/noise_ceiling_1d_subj%02d.npy"
            % (args.output_dir, args.subj, args.subj)
        )
        pred_path = "%s/encoding_results/subj%01d/pred_%s_whole_brain.p" % (
            args.output_dir,
            args.subj,
            args.pred_to_data,
        )
        pred, test_data = np.load(pred_path, allow_pickle=True)

        nc_thre = 20
        zone_indices, roi_label_list, tick_pos = [], [], []

        for roi in roi_list:
            roi_masks, roi_labels = extract_single_roi(roi, args.output_dir, args.subj)
            print(roi_labels)
            for i, m in enumerate(roi_masks):
                # print(m)
                m[nc < nc_thre] = 0
                zone_mask = list(np.where(m > 0)[0])

                zone_indices += zone_mask
                roi_label_list.append(roi_labels[i])
                tick_pos.append(len(zone_mask))

        predictions = pred[:, zone_indices]
        test_data = test_data[:, zone_indices]
        print("data shape: ")
        print(test_data.shape)
        print(np.cumsum(tick_pos))

        # compute the encoding model performance
        # from scipy.stats import pearsonr
        corr = np.corrcoef(predictions.T, test_data.T)[
            : len(zone_indices), len(zone_indices) :
        ]
        # np.save("%s/roi_gen/corrs_%s_subj%02d.npy" % (args.output_dir, args.pred_to_data, args.subj), {'generalizations':corr, 'zone indices':zone_indices, 'roi_labels': roi_label_list})

        plt.imshow(corr, cmap="RdBu_r")
        plt.colorbar()
        plt.xticks(np.cumsum(tick_pos), roi_label_list, rotation="vertical")
        plt.yticks(np.cumsum(tick_pos), roi_label_list)

        plt.savefig(
            "figures/roi_generalization/roi_gen_%s_subj%02d_%s_nc%d.png"
            % (args.pred_to_data, args.subj, args.roi, nc_thre)
        )
