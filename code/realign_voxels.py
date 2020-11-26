import numpy as np
import argparse
import pickle

from util.model_config import taskrepr_features


def realign_cortex_mask(old_mask, new_mask):
    """
    :param old_mask: old surface mask (3D matrix)
    :param new_mask: new surface mask (3D matrix)
    :param old_results: old correlation scores (1D array)
    :return: a mask for new voxels under the new cortical mask
    """
    new_mask[old_mask] = False
    return new_mask


def fit_new_results(
    old_mask, new_mask, old_results, additional_results, additional_voxel_mask
):
    """
    :param old_mask: old surface mask (3D matrix)
    :param new_mask: new surface mask (3D matrix)
    :param old_results: old correlation scores (1D array)
    :param additional_results: new correlation scores (1D array)
    :return:
    """
    assert np.sum(old_mask) == len(old_results)
    assert np.sum(additional_voxel_mask) == np.sum(additional_results)

    vol = np.zeros(old_mask.shape)
    vol[old_mask] = old_results
    vol[additional_voxel_mask] = additional_results

    corrs = vol[new_mask]

    return corrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1)
    parser.add_argument("--value", default="corr")
    parser.add_argument("--mode", default="")

    args = parser.parse_args()

    old_mask = np.load(
        "output/cortical_masks/subj%/old/cortical_mask_subj%02d.npy"
        % (args.subj, args.subj)
    )
    new_mask = np.load(
        "output/cortical_masks/subj%/cortical_mask_subj%02d.npy"
        % (args.subj, args.subj)
    )

    # make sure the two masks are not the same
    assert np.sum(old_mask == new_mask) != len(old_mask.flatten())

    if args.mode == "make_additional_mask":
        mask_for_additional_voxels = realign_cortex_mask(old_mask, new_mask)
        np.save(
            "output/voxel_masks/subj%d/cortical_mask_additional_subj%02d.npy",
            mask_for_additional_voxels,
        )

    elif args.mode == "combine_new_results":
        if args.value == "corr":
            for task in taskrepr_features:
                output = pickle.load(
                    open(
                        "output/encoding_results/subj%d/%s_%s_%s_whole_brain.p"
                        % (args.subj, args.value, "taskrepr", task),
                        "rb",
                    )
                )
                out = np.array(output)[:, 0]
