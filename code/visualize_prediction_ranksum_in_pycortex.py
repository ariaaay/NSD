"This scripts visualize prediction performance with Pycortex."

import pickle
import numpy as np

import argparse

from util.model_config import model_features


def load_data(task1, task2, subj=1):
    return np.load(
        "output/comparisons/ranksums_of_pred_of_%s_and_%s_subj%d.npy"
        % (task1, task2, subj)
    )


def make_volume(subj, task1, task2):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    vals = load_data(task1, task2, subj=subj)

    cortical_mask = np.load(
        "output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (subj, subj)
    )
    all_vals = np.zeros(cortical_mask.shape)
    all_vals[cortical_mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)

    # np.save("output/volumetric_results/subj%d/%s_%s.npy" % (subj, model, task), all_vals)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap="hot",
        vmax=np.max(all_vals),
    )
    return vol_data


# def model_selection(subj, model_dict, TR="_TRavg", version=11):
#     datamat = list()
#     for m in model_dict.keys():
#         if model_dict[m] is not None:
#             for l in model_dict[m]:
#                 data = load_data(m, model_subtype=l, subj=subj, measure="corr")
#                 datamat.append(data)
#     datamat = np.array(datamat)
#     threshold_mask = np.max(datamat, axis=0) > 0.13
#     best_model = np.argmax(datamat, axis=0)[threshold_mask]
#     mask = cortex.utils.get_cortical_mask("sub-CSI{}".format(subj), "full")
#     mask[mask == True] = threshold_mask
#
#     vol_data = cortex.Volume(
#         best_model, "sub-CSI{}".format(subj), "full", mask=mask, cmap="nipy_spectral"
#     )
#     return vol_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    args = parser.parse_args()

    # subjport = int("1111{}".format(args.subj))

    volumes = {
        "edge2d vs. edge3d": make_volume(
            subj=args.subj, task1="edge2d", task2="edge3d"
        ),
        "edge3d vs. class_places": make_volume(
            subj=args.subj, task1="edge3d", task2="class_places"
        ),
        "vanishing pts vs. layout": make_volume(
            subj=args.subj, task1="vanishing_point", task2="room_layout"
        ),
    }
    import cortex

    cortex.webgl.show(data=volumes, autoclose=False)

    import pdb

    pdb.set_trace()
