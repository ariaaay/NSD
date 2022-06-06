
import argparse
import numpy as np

import cortex
from visualize_in_pycortex import project_vals_to_3d


def split(original, pca_voxel_idxes, i_PC, cortical_n, branch="", split_threshold=5):
    """
    Input:
        original: PC projection matrix to be split by the ith PC (20000 x 20)
        cortical_mask: cortical mask (3d mask)
        pca_voxel_idxes: indexes to select the "best 20000 voxels" from cortical voxels (dim:20000 x 1)
    Returns:
        matrix_A: subset of projection matrix with the thresholded voxels by PCs
        idx_A: coritical length mask to pick for matrix A
        vol_A: the volume correspond to matrix_A
    """
    if i_PC < split_threshold:
    # positive
        idx = np.where(original[:,i_PC]>0)[0] # n x 1 integer (n = # of chosen voxels)
        if len(idx) == 0:
            return
        matrix_A = original[idx] # n x 20
        mask_A= np.zeros(cortical_n).astype(bool) # 100k x 1
        idx_A = pca_voxel_idxes[idx] # n x 1 integers, used to index cortical length array to pick out voxels relevant in this pc split
        mask_A [idx_A] = True
        branch += "A"
        VOLS["PC %d %s" % (i_PC, branch)] = make_volume(args.subj, matrix_A[:,i_PC], mask_A)
        split(matrix_A, idx_A, i_PC+1, cortical_n, branch)

        # negative
        idx = np.where(original[:,i_PC]<0)[0]
        if len(idx) == 0:
            return
        matrix_B = original[idx]
        mask_B = np.zeros(cortical_n).astype(bool)
        idx_B = pca_voxel_idxes[idx]
        mask_B [idx_B] = True
        branch += "B"
        VOLS["PC %d %s" % (i_PC, branch)] = make_volume(args.subj, matrix_B[:,i_PC], mask_B)
        split(matrix_B, idx_B, i_PC+1, cortical_n, branch)    


def make_volume(subj, vals, pca_mask, vmin=-2, vmax=2, cmap="BrBG_r", descale=True):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )
    cortical_vals = np.zeros(np.sum(cortical_mask))*np.nan
    # print(len(vals))
    # print(np.sum(pca_mask))
    cortical_vals[pca_mask] = vals
    

    # projecting value back to 3D space
    three_d_vals = project_vals_to_3d(cortical_vals, cortical_mask)

    vol_data = cortex.Volume(
        three_d_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return vol_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument("--model", default="clip")
    parser.add_argument("--name_modifier", default="best_20000_nc")
    parser.add_argument

    args = parser.parse_args()
    
    OUTPUT_ROOT = "."

    # visualize PC projections
    subj_proj = np.load(
        "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
        % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
    ).T # 100k x 20 vectors
    subj_mask = np.load(
        "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
        % (OUTPUT_ROOT, args.model, args.name_modifier, args.subj)
    ) # 1 x 100k vectors

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, args.subj, args.subj)
    ) # 1 x 100k vectors

    proj_val_only = subj_proj[subj_mask, :]
    proj_val_only /= proj_val_only.std(axis=0)
    pca_voxel_idxes = np.where(subj_mask==True)[0]

    VOLS = {}
    split(subj_proj[subj_mask, :], pca_voxel_idxes, i_PC=0, cortical_n=np.sum(cortical_mask), split_threshold=5)
    subj_port = "7111" + str(args.subj)
    # cortex.webgl.show(data=volumes, autoclose=False, port=int(subj_port))
    cortex.webgl.show(data=VOLS, port=int(subj_port), recache=False)


    import pdb

    pdb.set_trace()

