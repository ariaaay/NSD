"""
This scripts loads the permutation results and computed per roi or per voxels p-values.
"""

import seaborn as sns
import argparse
from glob import glob
from statsmodels.stats.multitest import fdrcorrection

from util.model_config import *
from util.util import *

perm_dir = "output/permutation_results"
acc_dir = "outputs/encoding_results"
nrep = 5000


def load_data(dir, subj, model, feature, type="pvalue"):
    fname = (
        "%s/subj%d/permutation_test_on_test_data_%s_%s_%s_whole_brain_whole_brain.p"
        % (dir, subj, type, model, feature)
    )
    f = open(fname, "rb")
    output = pickle.load(f)
    return output


def get_per_voxel_p(feature, subj):
    try:
        p = load_data(
            perm_dir, subj=subj, model="taskrepr", feature=feature, type="pvalue"
        )
    except FileNotFoundError:
        print("task " + feature + " results doesn't exists yet.")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please specific subjects/features to load"
    )
    parser.add_argument("--subj", type=int, default=1)
    # parser.add_argument("--comparison", action="store_true", default=False)
    # parser.add_argument("--reload", action="store_true", default=False)

    args = parser.parse_args()

    for feature in taskrepr_features:
        print("Loading results for task {}".format(feature))
        p = get_per_voxel_p(feature, args.subj)
        pickle.dump(
            p,
            open(
                "output/baseline/empirical_p_values_taskrepr_%s_subj%d_whole_brain_test_permute.p"
                % (feature, args.subj),
                "wb",
            ),
        )
        fdr_p = fdrcorrection(p)[1]
        pickle.dump(
            fdr_p,
            open(
                "output/baseline/empirical_p_values_taskrepr_%s_subj%d_whole_brain_test_permute_fdr.p"
                % (feature, args.subj),
                "wb",
            ),
        )
        # print(p)
