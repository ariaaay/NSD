
import argparse
import numpy as np

import cortex
from torch import true_divide
from visualize_in_pycortex import project_vals_to_3d


def split(subj, original, pca_voxel_idxes, i_PC, cortical_n, branch="", split_threshold=5, split_ratio=9):
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
    if len(pca_voxel_idxes) < 50:
        return
    if i_PC < split_threshold:
        idx = np.where(original[:,i_PC]>0)[0] # n x 1 integer (n = # of chosen voxels)
        matrix_A = original[idx] # n x 20
        idx_A = pca_voxel_idxes[idx] # n x 1 integers, used to index cortical length array to pick out voxels relevant in this pc split
        idx = np.where(original[:,i_PC]<0)[0]
        matrix_B = original[idx]
        idx_B = pca_voxel_idxes[idx]
        if not split_here(idx_A, idx_B, ratio=split_ratio): # this split is ineffective and should skip to the next PC
            split(subj,original, pca_voxel_idxes, i_PC+1, cortical_n, branch + "X")
        

        # positive
        mask_A= np.zeros(cortical_n).astype(bool) # 100k x 1
        mask_A[idx_A] = True
        VOLS["PC %d %s (n=%d)" % (i_PC, branch + "A", len(idx_A))] = make_volume(subj, matrix_A[:,i_PC], mask_A)
        
        # negative
        mask_B = np.zeros(cortical_n).astype(bool)
        mask_B[idx_B] = True
        VOLS["PC %d %s (n=%d)" % (i_PC, branch + "B", len(idx_B))] = make_volume(subj, matrix_B[:,i_PC], mask_B)

        split(subj, matrix_A, idx_A, i_PC+1, cortical_n, branch + "A")
        split(subj, matrix_B, idx_B, i_PC+1, cortical_n, branch + "B")


def split_here(idx_A, idx_B, ratio):
    if (len(idx_A) < 5) or (len(idx_B) < 5):
        return False
    elif len(idx_A) / len(idx_B) > ratio:
        return False
    elif len(idx_B) / len(idx_A) > ratio:
        return False
    else:
        return True


def make_subj_tree(subj, split_ratio=999, visualize=False):
    subj_proj = np.load(
        "%s/output/pca/%s/%s/subj%02d/pca_projections.npy"
        % (OUTPUT_ROOT, args.model, args.name_modifier, subj)
    ).T # 100k x 20 vectors
    subj_mask = np.load(
        "%s/output/pca/%s/%s/pca_voxels/pca_voxels_subj%02d.npy"
        % (OUTPUT_ROOT, args.model, args.name_modifier, subj)
    ) # 1 x 100k vectors

    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    ) # 1 x 100k vectors

    proj_val_only = subj_proj[subj_mask, :]
    proj_val_only /= proj_val_only.std(axis=0)
    pca_voxel_idxes = np.where(subj_mask==True)[0]

    split(subj, subj_proj[subj_mask, :], pca_voxel_idxes, i_PC=0, cortical_n=np.sum(cortical_mask), split_threshold=5, split_ratio=split_ratio)
    if visualize:
        subj_port = "2111" + str(subj)
        cortex.webgl.show(data=VOLS, port=int(subj_port), recache=False)
        import pdb
        pdb.set_trace()


def make_volume(subj, vals, pca_mask, vmin=-0.1, vmax=0.1, cmap="BrBG_r"):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    cortical_mask = np.load(
        "%s/output/voxels_masks/subj%d/cortical_mask_subj%02d.npy"
        % (OUTPUT_ROOT, subj, subj)
    )
    cortical_vals = np.zeros(np.sum(cortical_mask))*np.nan
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
    parser.add_argument("--on_cluster", default=False, action="store_true")
    parser.add_argument

    args = parser.parse_args()
    
    OUTPUT_ROOT = "."
    if args.on_cluster:
        OUTPUT_ROOT = "/user_data/yuanw3/project_outputs/NSD"
    
    VOLS = {}
    # visualize single subject
    # make_subj_tree(args.subj, visualize=True)

    # compute consistency of trees
    from analyze_in_mni import analyze_data_correlation_in_mni
    subjs = np.arange(8)
    all_volumes = []
    for s in subjs:
        make_subj_tree(s+1, split_ratio=999)
        subj_volumes = list(VOLS.values())
        labels = list(VOLS.keys())
        all_volumes.append(subj_volumes)
        VOLS = {}
    # remember to run `module load fsl-6.0.3` on cluster
    analyze_data_correlation_in_mni(all_volumes, args.model, save_name = "PC_tree_%s" % args.name_modifier, subjs=np.arange(1,9), volumes=True, xtick_label=labels)

    

