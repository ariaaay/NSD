import argparse
import numpy as np
import pandas as pd

from featureprep.feature_prep import extact_feature_by_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--output_path", type=str, default="/user_data/yuanw3/project_outputs/NSD/output")
    args = parser.parse_args()

    # coco ID - common 1000
    try:
        feature_mat = np.load("%s/%s_common_features.npy" % (args.output_path, args.task))
    except FileNotFoundError:
        stim = pd.read_pickle("/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
        stimulus_list = stim["cocoId"][stim["shared1000"]]
        assert len(stimulus_list) == 1000
        feature_mat = extact_feature_by_imgs(stimulus_list, args.task)
        np.save("%s/%s_common_features.npy" % (args.output_path, args.task), feature_mat)
    
    cov = np.cov(feature_mat)
    eigv = np.linalg.eigvals(cov)
    print(np.sum(eigv))


