import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.model_config import taskrepr_features, convnet_structures


def read_and_plot_performances(model, feature, subj, type="corr", model_note=""):
    perf = pickle.load(
        open(
            "output/encoding_results/subj%d/%s_%s_%s_whole_brain%s.p"
            % (subj, type, model, feature, model_note),
            "rb",
        )
    )
    try:
        out = [item[0] for item in perf]
    except IndexError:
        out = perf
    print(max(out))
    plt.figure()
    plt.hist(out, bins=50)
    plt.xlabel(type)
    plt.title(
        "Histogram of encoding model performances across all cortical voxels (%s, subj %s, %s_%s%s)"
        % (type, subj, model, feature, model_note)
    )
    plt.savefig(
        "figures/encoding_results/subj%d/%s_%s_%s_whole_brain%s.png"
        % (subj, type, model, feature, model_note)
    )


def read_and_plot_compared_performances(
    model, feature, subj, type="corr", model_note=""
):
    perf = pickle.load(
        open(
            "output/encoding_results/subj%d/%s_%s_%s_whole_brain.p"
            % (subj, type, model, feature),
            "rb",
        )
    )
    try:
        out = [item[0] for item in perf]
    except IndexError:
        out = perf
    print(max(out))
    plt.figure()
    plt.hist(out, bins=50, label=10000, alpha=0.5)

    perf = pickle.load(
        open(
            "output/encoding_results/subj%d/%s_%s_%s_whole_brain%s.p"
            % (subj, type, model, feature, model_note),
            "rb",
        )
    )
    try:
        out = [item[0] for item in perf]
    except IndexError:
        out = perf
    plt.hist(out, bins=50, label=5000, alpha=0.5)
    plt.xlabel(type)
    plt.legend()

    plt.title(
        "Comparison Histogram across all cortical voxels (%s, subj %s, %s_%s%s)"
        % (type, subj, model, feature, model_note)
    )
    plt.savefig(
        "figures/encoding_results/subj%d/Comparison_%s_%s_%s_whole_brain%s.png"
        % (subj, type, model, feature, model_note)
    )


parser = argparse.ArgumentParser()
parser.add_argument("--subj", type=int, default=1)
parser.add_argument("--model", type=str, default="taskrepr")
parser.add_argument("--type", type=str, default="corr")
parser.add_argument("--model_note", type=str, default="")
parser.add_argument("--compare", type=bool)

args = parser.parse_args()

# if args.model == "taskrepr":
#     for task in tqdm(taskrepr_features):
#         try:
#             read_and_plot_performances(args.model, task, args.subj, args.type, args.model_note)
#         except FileNotFoundError:
#             continue
# elif args.model == "convnet":
#     for structure in convnet_structures:
#         try:
#             read_and_plot_performances(args.model, structure, args.subj, args.type, args.model_note)
#         except FileNotFoundError:
#             continue

for structure in convnet_structures:
    try:
        read_and_plot_compared_performances(
            args.model, structure, args.subj, args.type, args.model_note
        )
    except FileNotFoundError:
        continue
