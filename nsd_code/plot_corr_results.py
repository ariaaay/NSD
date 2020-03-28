import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.model_config import taskrepr_features, convnet_structures

def read_and_plot_performances(model, feature, subj, type="corr"):
    perf = pickle.load(
        open("output/encoding_results/subj%d/%s_%s_%s_whole_brain.p" % (subj, type, model, feature), "rb"))
    try:
        out = [item[0] for item in perf]
    except IndexError:
        out = perf
    print(max(out))
    plt.figure()
    plt.hist(out, bins=50)
    plt.xlabel(type)
    plt.title("Histogram of encoding model performances across all cortical voxels (%s, subj %s, %s_%s)" % (type, subj, model, feature))
    plt.savefig("figures/encoding_results/subj%d/%s_%s_%s_whole_brain.png" % (subj, type, model, feature))


parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--model", type=str, default="taskrepr")
parser.add_argument("--type", type=str, default="corr")

args = parser.parse_args()

if args.model == "taskrepr":
    for task in tqdm(taskrepr_features):
        try:
            read_and_plot_performances(args.model, task, args.subj, args.type)
        except FileNotFoundError:
            continue
elif args.model == "convnet":
    for structure in convnet_structures:
        try:
            read_and_plot_performances(args.model, structure, args.subj, args.type)
        except FileNotFoundError:
            continue
