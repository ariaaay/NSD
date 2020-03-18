import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.model_config import taskrepr_features

parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)

args = parser.parse_args()

for task in tqdm(taskrepr_features):
    corrs = pickle.load(open("output/encoding_results/subj%d/corr_taskrepr_*s_whole_brain.p" % (args.subj, task), "rb"))
    plt.figure()
    plt.hist(corrs, bins=50)
    plt.savefig("figures/encoding_results/subj%d/corr_taskrepr_*s_whole_brain.png" % (args.subj, task))
