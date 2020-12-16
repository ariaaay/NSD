import argparse
import numpy as np

from featureprep.feature_prep import extact_feature_by_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    # coco ID - common 10000
    stimulus_list = np.load("output/trials_subj01.npy")[:,0]
    feature_mat = extact_feature_by_imgs(stimulus_list, args.task)
    np.save("%s/taskrepr_%s_common_features.npy" % (args.output_path, task), feature_mat)
    
    cov = np.cov(feature_mat)
    eigv = np.linalg.eigvals(cov)
    print(np.sum(eigv))


