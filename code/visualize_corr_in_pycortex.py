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
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    #load correlation scores of cortical voxels
    vals = load_data(model, task, subj=subj)

    #TODO: fix the following piece of code

    # if mask_with_significance:
    #     correction = "emp_fdr"
    #     alpha = 0.05
    #     sig_mask = np.load(
    #         "output/voxels_masks/subj%d/%s_%s_%s_%s_%s.npy"
    #         % (subj, model, task, correction, str(alpha))
    #     )
    #     # print(sig_mask.shape)
    #     # print(np.sum(sig_mask))
    #     if np.sum(sig_mask) > 0:
    #         mask[mask == True] = sig_mask
    #         vals = vals[sig_mask]

    cortical_mask = np.load(
        "output/voxels_masks/subj%d/cortical_mask_subj%02d.npy" % (subj, subj)
    )
    # projecting value back to 3D space
    all_vals = np.zeros(cortical_mask.shape)
    all_vals[cortical_mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)

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
    import cortex
    import nibabel as nib

    parser = argparse.ArgumentParser(description="please specific subject to show")
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )
    parser.add_argument("--mask_sig", default=False, action="store_true")
    args = parser.parse_args()

    vroi = nib.load("output/voxels_masks/subj%d/prf-visualrois.nii.gz" % args.subj)
    vroi_data = vroi.get_fdata()
    vroi_data = np.swapaxes(vroi_data, 0, 2)

    visual_roi_volume = cortex.Volume(
        vroi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
    )

    ecc_roi = nib.load("output/voxels_masks/subj%d/prf-eccrois.nii.gz" % args.subj)
    ecc_roi_data = ecc_roi.get_fdata()
    ecc_roi_data = np.swapaxes(ecc_roi_data, 0, 2)

    ecc_roi_volume = cortex.Volume(
        ecc_roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
    )

    place_roi = nib.load("output/voxels_masks/subj%d/floc-places.nii.gz" % args.subj)
    place_roi_data = place_roi.get_fdata()
    place_roi_data = np.swapaxes(place_roi_data, 0, 2)

    place_roi_volume = cortex.Volume(
        place_roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
    )

    # subjport = int("1111{}".format(args.subj))

    volumes = {
        # "Curvature": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="curvature",
        #     mask_with_significance=args.mask_sig,
        # ),
        "2D Edges": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="edge2d",
            mask_with_significance=args.mask_sig,
        ),
        "3D Edges": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="edge3d",
            mask_with_significance=args.mask_sig,
        ),
        # "2D Keypoint": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="keypoint2d",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "3D Keypoint": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="keypoint3d",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Depth": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="rgb2depth",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Reshade": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="reshade",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Distance": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="rgb2mist",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Surface Normal": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="rgb2sfnorm",
        #     mask_with_significance=args.mask_sig,
        # ),
        "Object Class": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="class_1000",
            mask_with_significance=args.mask_sig,
        ),
        "Scene Class": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="class_places",
            mask_with_significance=args.mask_sig,
        ),
        # "Autoencoder": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="autoencoder",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Denoising": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="denoise",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "2.5D Segm.": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="segment25d",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "2D Segm.": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="segment2d",
        #     mask_with_significance=args.mask_sig,
        # ),
        "Semantic Segm": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="segmentsemantic",
            mask_with_significance=args.mask_sig,
        ),
        "Vanishing Point": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="vanishing_point",
            mask_with_significance=args.mask_sig,
        ),
        "Room Layout": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="room_layout",
            mask_with_significance=args.mask_sig,
        ),
        # "Color": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="colorization",
        #     mask_with_significance=args.mask_sig,
        # ),
        "Inpainting Whole": make_volume(
            subj=args.subj,
            model="taskrepr",
            task="inpainting_whole",
            mask_with_significance=args.mask_sig,
        ),
        # "Jigsaw": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="jigsaw",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Taskrepr model comparison": model_selection(
        #     subj=args.subj, model_dict=model_features
        # ),
        "Visual ROIs": visual_roi_volume,
        "Eccentricity ROIs" : ecc_roi_volume,
        "Places ROIs": place_roi_volume
    }
    import cortex

    cortex.webgl.show(data=volumes, autoclose=False)

    import pdb

    pdb.set_trace()
