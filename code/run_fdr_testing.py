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
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()
    threshold_dict = dict()

    superset_mask = None

for task in TOI:
    output = pickle.load(
        open(
            "%s/subj%d/corr_taskrepr_%s_whole_brain.p" % (args.dir, args.subj, task),
            "rb",
        )
    )
    corrs = np.array(output)[:, 0]
    task_threshold = negative_tail_fdr_threshold(corrs, 0, alpha=args.alpha, axis=0)
    threshold_dict[task] = task_threshold

    sig_mask = corrs > task_threshold
    np.save("output/voxels_masks/subj%d/taskrepr_%s_negtail_fdr_%0.2f.npy" % (args.subj, task, args.alpha), sig_mask)

    # Compute a significance mask using superset of mask from all tasks
    if superset_mask is None:
        superset_mask = sig_mask.astype(int)
    else:
        superset_mask += sig_mask.astype(int)

superset_mask = superset_mask > 0
np.save("output/voxels_masks/subj%d/taskrepr_superset_mask_negtail_fdr_%0.2f.npy" % (args.subj, args.alpha), superset_mask)

output_path = os.path.join(args.dir, "subj" + str(args.subj), "fdr_threshold.json")
with open(output_path, "w") as f:
    json.dump(threshold_dict, f)
