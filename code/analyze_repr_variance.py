import argparse
import numpy as np
import pandas as pd

from featureprep.feature_prep import extact_feature_by_imgs


def load_common_feature_matrix(task):
    try:
        feature_mat = np.load("%s/%s_common_features.npy" % (args.output_path, task))
    except FileNotFoundError:
        stim = pd.read_pickle(
            "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
        )
        stimulus_list = stim["cocoId"][stim["shared1000"]]
        assert len(stimulus_list) == 1000
        feature_mat = extact_feature_by_imgs(stimulus_list, task)
        np.save("%s/%s_common_features.npy" % (args.output_path, task), feature_mat)
    return feature_mat


def measure_variance(feature_matrix):
    cov = np.cov(feature_matrix)
    eigv = np.linalg.eigvals(cov)
    return np.sum(eigv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--task_list",
        default=[
            "edge2d",
            "edge3d",
            "keypoint2d",
            "keypoint3d",
            "class_places",
            "class_1000",
            "autoencoder",
            "vanishing_point",
            "colorization",
            "rgb2sfnorm",
        ],
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output",
    )
    args = parser.parse_args()

    # coco ID - common 1000

    if args.task is not None:
        fm = load_common_feature_matrix(args.task)
        print(measure_variance(fm))
    
    else:
        tasks = ["taskrepr_" + t for t in args.task_list]
        var_dict = dict()
        for t in tasks:
            print("loading " + t)
            fm = load_common_feature_matrix(t)
            var_dict[t] = measure_variance(fm)
        print(var_dict)
        

        