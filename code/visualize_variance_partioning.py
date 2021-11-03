import argparse
import pickle
import numpy as np
import cortex

from visualize_corr_in_pycortex import project_vals_to_3d
from util.data_util import load_model_performance


def make_volume(vals, subj, mask_with_significance=False):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    try:
        cortical_mask = np.load(
            "output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (subj, subj)
        )
    except FileNotFoundError:
        cortical_mask = np.load(
            "output/voxels_masks/subj%d/old/cortical_mask_subj%02d.npy" % (subj, subj)
        )
    if mask_with_significance:
        sig_mask = np.load(
            "output/voxels_masks/subj%d/taskrepr_superset_mask_%s_%0.2f.npy"
            % (subj, "negtail_fdr", 0.05)
        )
        vals[~sig_mask] = 0
    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap="hot",
        vmin=0,
        vmax=0.5,
    )
    return vol_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--mask_sig", default=False)

    args = parser.parse_args()

    if len(args.tasks) == 2:
        t1 = load_model_performance("taskrepr_" + args.tasks[0], args.subj)
        t2 = load_model_performance("taskrepr_" + args.tasks[1], args.subj)
        t12 = load_model_performance(
            "taskrepr_" + args.tasks[0] + "_taskrepr_" + args.tasks[1], args.subj
        )
        print(max(t12))

        var_unique1 = t12 - t2
        var_unique2 = t12 - t1
        var_shared = t1 + t2 - t12

    elif len(args.tasks) == 3:
        t1 = load_model_performance("taskrepr_" + args.tasks[0], args.subj)
        t2 = load_model_performance("taskrepr_" + args.tasks[1], args.subj)
        t3 = load_model_performance("taskrepr_" + args.tasks[2], args.subj)

        t12 = load_model_performance(
            "taskrepr_" + args.tasks[0] + "_taskrepr_" + args.tasks[1], args.subj
        )
        t13 = load_model_performance(
            "taskrepr_" + args.tasks[0] + "_taskrepr_" + args.tasks[2], args.subj
        )
        t23 = load_model_performance(
            "taskrepr_" + args.tasks[1] + "_taskrepr_" + args.tasks[2], args.subj
        )

        t123 = load_model_performance(
            "taskrepr_"
            + args.tasks[0]
            + "_taskrepr_"
            + args.tasks[1]
            + "_taskrepr_"
            + args.tasks[2],
            args.subj,
        )

        print(max(t123))

        var_unique1 = t123 - t2 - t3 + t23
        var_unique2 = t123 - t1 - t3 + t13
        var_unique3 = t123 - t1 - t2 + t12

    volumes = {
        "Unique Var - 2D Edges": make_volume(
            var_unique1, subj=args.subj, mask_with_significance=args.mask_sig,
        ),
        "Unique Var - 3D Edges": make_volume(
            var_unique2, subj=args.subj, mask_with_significance=args.mask_sig,
        ),
        "Unique Var - Semantic": make_volume(
            var_unique3, subj=args.subj, mask_with_significance=args.mask_sig,
        ),
        "Total Variance": make_volume(
            t123, subj=args.subj, mask_with_significance=args.mask_sig,
        ),
    }

    cortex.webgl.show(data=volumes, autoclose=False)

    import pdb

    pdb.set_trace()
