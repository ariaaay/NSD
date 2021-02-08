import argparse
import numpy as np

from util.model_config import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default="/user_data/yuanw3/project_outputs/NSD/output"
    )
    parser.add_argument("--feature")
    args = parser.parse_args()

    feature_path = "/user_data/yuanw3/project_outputs/NSD/features/subj%d/%s.npy" % (args.subj, args.feature)
    feature = np.load(feature_path).squeeze()
    rsm = np.corrcoef(feature)

    np.save(
        "%s/rdms/subj%02d_%s.npy" % (args.output_dir, args.subj,args.feature),
        rsm,
    )
