"This scripts visualize prediction performance with Pycortex."

import pickle
import numpy as np

import argparse
from util.model_config import model_features


def load_data(model, task, subj=1, measure="corr"):
    output = pickle.load(
        open(
            "output/encoding_results/subj%d/%s_%s_%s_whole_brain.p"
            % (subj, measure, model, task),
            "rb",
        )
    )
    if measure == "corr":
        out = np.array(output)[:, 0]
    else:
        out = np.array(output)
    return out


def make_volume(subj, model, task, mask_with_significance=False):
    import cortex

    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )
    vals = load_data(model, task, subj=subj)

    if mask_with_significance:
        correction = "emp_fdr"
        alpha = 0.05
        sig_mask = np.load(
            "output/voxels_masks/subj%d/%s_%s_%s_%s_%s.npy"
            % (subj, model, task, correction, str(alpha))
        )
        # print(sig_mask.shape)
        # print(np.sum(sig_mask))
        if np.sum(sig_mask) > 0:
            mask[mask == True] = sig_mask
            vals = vals[sig_mask]

    cortical_mask = np.load("output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (subj, subj))
    all_vals = np.zeros(cortical_mask.shape)
    all_vals[cortical_mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)

    np.save("output/volumetric_results/subj%d/%s_%s.npy" % (subj, model, task), all_vals)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap="hot",
        vmax=np.max(all_vals)
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
    parser.add_argument("--mask_sig", default=False, action="store_true")
    args = parser.parse_args()

    # subjport = int("1111{}".format(args.subj))

    volumes = {
        # "vgg16": make_volume(
        #     subj=args.subj,
        #     model="convnet",
        #     task="vgg16",
        #     mask_with_significance=args.mask_sig,
        # ),
        "resnet50": make_volume(
            subj=args.subj,
            model="convnet",
            task="res50",
            mask_with_significance=args.mask_sig,
        ),
    }
    import cortex

    cortex.webgl.show(data=volumes, autoclose=False)

    import pdb

    pdb.set_trace()
