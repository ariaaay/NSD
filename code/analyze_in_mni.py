import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cortex

from visualize_in_pycortex import project_vols_to_mni, make_pc_volume, project_vals_to_3d


subjs = [1, 2, 5, 7]
OUTPUT_ROOT = "/user_data/yuanw3/project_outputs/NSD/"
MODEL = "clip"
n_subj = len(subjs)
n_PCs = 20

# load all PCS from all four subjs
all_PCs = []
for subj in subjs:
    all_PCs.append(np.load(
            "%s/output/pca/%s/subj%02d/%s_pca_group_components.npy"
            % (OUTPUT_ROOT, MODEL, subj, MODEL)
        ))

# make them pycortex volume and project them to mni
corrs, corrs_mean, corrs_12, corrs_13, corrs_14, corrs_23, corrs_24, corrs_34 = [], [], [], [], [], [], [], []
vol_mask = cortex.db.get_mask("fsaverage", "atlas")
print(vol_mask.shape)
print(np.sum(vol_mask))


for i in tqdm(range(n_PCs)):
    vals = np.zeros((n_subj, np.sum(vol_mask)))
    for s in range(n_subj):
        vol = make_pc_volume(subjs[s], all_PCs[s][i, :])
        mni_vol = project_vols_to_mni(subjs[s], vol)

        # mask the values and compute correlations across subj
        vals[s, :] = mni_vol[vol_mask]
    
    corr = np.corrcoef(vals)
    corrs.append(corr)
    corrs_mean.append(np.triu(corr, k=1) / 6) 
    corrs_12.append(corr[0, 1])
    corrs_13.append(corr[0, 2])
    corrs_14.append(corr[0, 3])
    corrs_23.append(corr[1, 2])
    corrs_24.append(corr[1 ,3])
    corrs_34.append(corr[2, 3])

np.save("%s/output/pca/%s/PC_corr_across_subjs.npy" % (OUTPUT_ROOT, MODEL), corrs)

plt.plot(np.arange(n_PCs), corrs_mean, label="Average")
plt.plot(np.arange(n_PCs), corrs_12, label="1-2", alpha=0.6)
plt.plot(np.arange(n_PCs), corrs_13, label="1-5", alpha=0.6)
plt.plot(np.arange(n_PCs), corrs_14, label="1-7", alpha=0.6)
plt.plot(np.arange(n_PCs), corrs_23, label="2-5", alpha=0.6)
plt.plot(np.arange(n_PCs), corrs_24, label="2-7", alpha=0.6)
plt.plot(np.arange(n_PCs), corrs_34, label="5-7", alpha=0.6)
plt.xlabel("PC nums")
plt.ylabel("correlation")
plt.legend()
plt.savefig("figures/PCA/%s_PC_corr_across_subjs.png" % MODEL)