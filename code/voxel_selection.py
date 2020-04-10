import pickle
import numpy as np

model = "convnet_res50"
subj = 1
num_voxel = 100

perf = pickle.load(open("output/encoding_results/subj%d/corr_%s_whole_brain.p" % (subj, model), "rb"))
corrs = [out[0] for out in perf]

#select the index for the best predicted voxels
best_vox_ind = np.argsort(corrs)[::-1][:num_voxel]
np.save("output/voxels_masks/subj%d/best_%d_voxel_inds_%s.npy" % (num_voxel, subj, model), best_vox_ind)