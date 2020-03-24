import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.model_config import taskrepr_features, convnet_structures

def read_and_plot_performances(model, feature, subj):
    perf = pickle.load(
        open("output/encoding_results/subj%d/corr_%s_%s_whole_brain.p" % (subj, model, feature), "rb"))
    corrs = [item[0] for item in perf]
    plt.figure()
    plt.hist(corrs, bins=50)
    plt.xlabel("Correlations")
    plt.title("Histogram of encoding model performances across all cortical voxels (subj %s, %s_%s)" % (subj, model, feature))
    plt.savefig("figures/encoding_results/subj%d/corr_%s_%s_whole_brain.png" % (subj, model, feature))


parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--model", type=str, default="taskrepr")

args = parser.parse_args()

if args.model == "taskrepr":
    for task in tqdm(taskrepr_features):
        try:
            read_and_plot_performances(args.model, task, args.subj)
        except FileNotFoundError:
            continue
elif args.model == "convnet":
    for structure in convnet_structures:
        try:
            read_and_plot_performances(args.model, structure, args.subj)
        except FileNotFoundError:
            continue
