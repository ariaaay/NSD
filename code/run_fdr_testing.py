# import sys, os
# sys.path.append("/Users/ariaw/Projects/cortilities/")

import os
import pickle
import argparse
import json
import numpy as np

from util.util import negative_tail_fdr_threshold

# Task of Interest
TOI = [
    "edge2d",
    "edge3d",
    "class_places",
    "class_1000",
    "vanishing_point",
    "room_layout",
    "inpainting_whole",
    "rgb2sfnorm",
    "segment2d",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1, help="specify subjects")
    parser.add_argument("--dir", type=str, default="output/encoding_results")

    args = parser.parse_args()
    threshold_dict = dict()

for task in TOI:
    output = pickle.load(
        open(
            "%s/subj%d/corr_taskrepr_%s_whole_brain.p" % (args.dir, args.subj, task),
            "rb",
        )
    )
    corrs = np.array(output)[:, 0]
    threshold_dict[task] = negative_tail_fdr_threshold(corrs, 0, alpha=0.01, axis=0)

output_path = os.path.join(args.dir, "subj" + str(args.subj), "fdr_threshold.json")
with open(output_path, "w") as f:
    json.dump(threshold_dict, f)
