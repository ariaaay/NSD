
import argparse
import numpy as np
from tqdm import tqdm

import cortex
from visualize_in_pycortex import project_vals_to_3d


def split(subj, original, pca_voxel_idxes, i_PC, cortical_n, branch="", split_threshold=4, split_ratio=9):
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
        VOLS["PC %d %s" % (i_PC, branch + "A")] = make_volume(subj, matrix_A[:,i_PC], mask_A)
        
        # negative
        mask_B = np.zeros(cortical_n).astype(bool)
        mask_B[idx_B] = True
        VOLS["PC %d %s" % (i_PC, branch + "B")] = make_volume(subj, matrix_B[:,i_PC], mask_B)

        split(subj, matrix_A, idx_A, i_PC+1, cortical_n, branch + "A", split_ratio=split_ratio)
        split(subj, matrix_B, idx_B, i_PC+1, cortical_n, branch + "B", split_ratio=split_ratio)


def split_here(idx_A, idx_B, ratio):
    if ratio == 999: #not limiting splits
        return True
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
    from visualize_in_pycortex import project_vols_to_mni

    save_name = "PC_split_" + args.name_modifier
    n_subj = 8
    all_volumes = []
    for s in np.arange(n_subj):
        make_subj_tree(s+1, split_ratio=999)
        vols = VOLS.copy()
        all_volumes.append(vols)
        VOLS = {}


    # make them pycortex volume if they are not and project them to mni
    corrs, corrs_mean, labels = [], [], []
    vol_mask = cortex.db.get_mask("fsaverage", "atlas")

    n_nodes = len(all_volumes[0].keys()) # some subject might not have this many nodes
    for i in tqdm(range(n_nodes)):
        vals = np.zeros((n_subj, np.sum(vol_mask)))
        split_name = list(all_volumes[0].keys())[i] # get the split name from subj 1
        for s in range(n_subj):
            try: 
                vol = all_volumes[s][split_name]
                vol.data[np.isnan(vol.data)] = 0
                mni_vol = project_vols_to_mni(s+1, vol)
                # mask the values and compute correlations across subj
                vals[s, :] = mni_vol[vol_mask]
                skip_this_split = False
            except KeyError:
                skip_this_split = True
                break

        if not skip_this_split:
            labels.append(split_name)
            corr = np.corrcoef(vals)
            corrs.append(corr)
            corrs_mean.append(np.sum(np.triu(corr, k=1)) / (n_subj*(n_subj-1)/2))

    np.save(
        "%s/output/pca/%s/%s_corr_across_subjs.npy" % (OUTPUT_ROOT, args.model, save_name),
        corrs,
    )

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(labels)), corrs_mean)
    plt.xlabel("Splits")
    plt.xticks(np.arange(len(labels)), labels=labels, rotation=45)
    plt.ylabel("correlation")
    plt.savefig("figures/PCA/%s_%s_corr_across_subjs.png" % (args.model, save_name))
    

