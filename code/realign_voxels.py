import numpy as np
import argparse

def realign_cortex_mask(old_mask, new_mask):
    """
    :param old_mask: old surface mask (3D matrix)
    :param new_mask: new surface mask (3D matrix)
    :param old_results: old correlation scores (1D array)
    :return: a mask for new voxels under the new cortical mask
    """
    new_mask[old_mask] = False
    return new_mask

def fit_new_results(old_mask, new_mask, old_results, additional_results, additional_voxel_mask):
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

    old_mask = np.load("output/cortical_masks/subj%")