import argparse
import numpy as np
import matplotlib.pyplot as plt
from compute_feature_rdm import computeID
from sklearn.cluster import SpectralClustering, SpectralBiclustering

from util.model_config import *
import json


def load_brain_data(args):
    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )

    br_data = np.load(brain_path)
    voxel_selected = np.zeros(br_data.shape[1], dtype=bool)
    return br_data, voxel_selected


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
    parser.add_argument("--ID", action="store_true")
    parser.add_argument("--rdm", action="store_true")
    parser.add_argument("--biclustering", action="store_true")
    parser.add_argument("--spectral_clustering", action="store_true")

    args = parser.parse_args()

    if args.rdm:
        br_data, voxel_selected = load_brain_data(args)
        if args.combined_roi:
            roi_list = ["floc-words", "floc-faces", "floc-places", "prf-visualrois"]
            voxel_selected = add_roi_to_voxel_selected(roi_list, voxel_selected)
            selected_brain_data = br_data[:, voxel_selected]
            rdm = np.corrcoef(selected_brain_data)
            np.save(
                "%s/rdms/subj%02d_%s.npy"
                % (args.output_dir, args.subj, "_".join(roi_list)),
                rdm,
            )
        else:
            print("extracting RDMs for these rois: " + str(args.single_roi))
            for roi in args.single_roi:
                roi_masks, roi_labels = extract_single_roi(roi)
                for i, m in enumerate(roi_masks):
                    try:
                        rdm = np.load(
                            "%s/rdms/subj%02d_%s.npy"
                            % (args.output_dir, args.subj, roi_labels[i])
                        )
                    except FileNotFoundError:
                        brain_data = br_data[:, m]
                        rdm = np.corrcoef(brain_data)
                        np.save(
                            "%s/rdms/subj%02d_%s.npy"
                            % (args.output_dir, args.subj, roi_labels[i]),
                            rdm,
                        )
                    if args.biclustering:
                        model = SpectralBiclustering(n_clusters=10, method="scale")
                        model.fit(rdm)
                        try:
                            assert (model.row_labels_ == model.column_labels_).all()
                        except AssertionError:
                            print("not match")
                            print(model.row_labels_[:10])
                            print(model.column_labels_[:10])
                        bc_idx = np.argsort(model.row_labels_)
                        np.save(
                            "%s/rdms/subj%02d_%s_biclustering_indexes.npy"
                            % (args.output_dir, args.subj, roi_labels[i]),
                            bc_idx,
                        )

    if args.spectral_clustering:
        for roi in args.single_roi:
            roi_masks, roi_labels = extract_single_roi(roi)
            for i, m in enumerate(roi_masks):
                nc = 10
                brain_data = br_data[:, m]
                clustering = SpectralClustering(n_clusters=nc).fit(brain_data)
                np.save(
                    "%s/rdms/cluster_%d_labels_subj%02d_%s.npy"
                    % (args.output_dir, args.subj, roi_labels[i]),
                    clustering.labels_,
                )

    if args.ID:
        sample_size = 3000
        for roi in args.single_roi:
            try:
                with open(
                    "../Cats/outputs/brain_ID_dict_%s_s%d.json" % (roi, sample_size),
                    "r",
                ) as f:
                    ID_dict = json.load(f)
            except FileNotFoundError:
                br_data, _ = load_brain_data(args)
                subsample_idx = np.random.choice(
                    np.arange(br_data.shape[0]), size=sample_size, replace=False
                )
                ID_dict = {}
                roi_masks, roi_labels = extract_single_roi(roi)
                for i, m in enumerate(roi_masks):
                    brain_data = br_data[:, m]
                    brain_data = brain_data[subsample_idx, :]

                    mean, error = computeID(brain_data.squeeze())
                    print(roi_labels[i])
                    print("Mean ID is: %d Error of ID is: %d" % (mean, error))
                    ID_dict[roi_labels[i]] = (mean, error)

                with open(
                    "../Cats/outputs/brain_ID_dict_%s_s%d.json" % (roi, sample_size),
                    "w",
                ) as f:
                    json.dump(ID_dict, f)

            plt.figure()
            plt.bar(
                np.arange(len(ID_dict.keys())),
                np.array(list(ID_dict.values()))[:, 0],
                yerr=np.array(list(ID_dict.values()))[:, 1],
            )
            plt.xlabel("ROIs")
            plt.ylabel("Intrinsic Dimensions")
            plt.xticks(
                ticks=np.arange(len(ID_dict.keys())), labels=list(ID_dict.keys())
            )
            plt.savefig("../Cats/figures/brain_ID_dict_%s_s%d.png" % (roi, sample_size))
