"This scripts visualize prediction performance with Pycortex."

import pickle
import numpy as np

import argparse
from util.model_config import model_features
from util.data_util import load_data


def project_vals_to_3d(vals, mask):
    all_vals = np.zeros(mask.shape)
    all_vals[mask] = vals
    all_vals = np.swapaxes(all_vals, 0, 2)
    return all_vals


def make_volume(subj, model, task=None, mask_with_significance=False):
    mask = cortex.utils.get_cortical_mask(
        "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
    )

    # load correlation scores of cortical voxels
    vals = load_data(model, task, output_root, subj=subj)
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
            "output/voxels_masks/subj%d/%s_%s_%s_%0.2f.npy"
            % (subj, model, task, "negtail_fdr", 0.05)
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


def make_pc_volume(subj, vals, mask_with_significance=False):
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
            "output/voxels_masks/subj%d/taskrepr_superset_mask_%s_%0.02f.npy"
            % (subj, "negtail_fdr", 0.05)
        )
        vals[~sig_mask] = -999
    # projecting value back to 3D space
    all_vals = project_vals_to_3d(vals, cortical_mask)

    vol_data = cortex.Volume(
        all_vals,
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
        cmap="RdPu",
        vmin=-3,
        vmax=3,
    )
    return vol_data


def make_3pc_volume(subj, PCs, mask_with_significance=False):
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

    pc_3d = []
    for i in range(3):
        tmp = PCs[i, :] / np.max(PCs_zscore[i, :]) * 255
        if mask_with_significance:
            sig_mask = np.load(
                "output/voxels_masks/subj%d/taskrepr_superset_mask_%s_%0.02f.npy"
                % (subj, "negtail_fdr", 0.05)
            )
            tmp[~sig_mask] = 0
        # projecting value back to 3D space
        pc_3d.append(project_vals_to_3d(tmp, cortical_mask))

    red = cortex.Volume(
        pc_3d[0].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    green = cortex.Volume(
        pc_3d[1].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )
    blue = cortex.Volume(
        pc_3d[2].astype(np.uint8),
        "subj%02d" % subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=mask,
    )

    vol_data = cortex.VolumeRGB(
        red,
        green,
        blue,
        "subj%02d" % subj,
        channel1color=(194, 30, 86),
        channel2color=(50, 205, 50),
        channel3color=(30, 144, 255),
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
    parser.add_argument("--sig_method", default="negtail_fdr")
    parser.add_argument("--alpha", default=0.05)
    parser.add_argument("--show_pcs", default=False, action="store_true")
    parser.add_argument("--on_cluster", action="store_true")

    args = parser.parse_args()

    if args.on_cluster:
        output_root = "/user_data/yuanw3/project_outputs/NSD"
    else:
        output_root = "."

    vroi = nib.load("%s/output/voxels_masks/subj%d/prf-visualrois.nii.gz" % (output_root, args.subj))
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

    ecc_roi = nib.load("%s/output/voxels_masks/subj%d/prf-eccrois.nii.gz" % (output_root, args.subj))
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

    place_roi = nib.load("%s/output/voxels_masks/subj%d/floc-places.nii.gz" % (output_root, args.subj))
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

    face_roi = nib.load("%s/output/voxels_masks/subj%d/floc-faces.nii.gz" % (output_root, args.subj))
    face_roi_data = face_roi.get_fdata()
    face_roi_data = np.swapaxes(face_roi_data, 0, 2)

    face_roi_volume = cortex.Volume(
        face_roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
    )

    body_roi = nib.load("%s/output/voxels_masks/subj%d/floc-bodies.nii.gz" % (output_root, args.subj))
    body_roi_data = body_roi.get_fdata()
    body_roi_data = np.swapaxes(body_roi_data, 0, 2)

    body_roi_volume = cortex.Volume(
        body_roi_data,
        "subj%02d" % args.subj,
        "func1pt8_to_anat0pt8_autoFSbbr",
        mask=cortex.utils.get_cortical_mask(
            "subj%02d" % args.subj, "func1pt8_to_anat0pt8_autoFSbbr"
        ),
    )

    word_roi = nib.load("%s/output/voxels_masks/subj%d/floc-words.nii.gz" % (output_root, args.subj))
    word_roi_data = word_roi.get_fdata()
    word_roi_data = np.swapaxes(word_roi_data, 0, 2)

    word_roi_volume = cortex.Volume(
        word_roi_data,
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
        "clip": make_volume(
            subj=args.subj,
            model="clip",
        ),
        "clip-text": make_volume(
            subj=args.subj,
            model="clip_text",
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
        # "Semantic Segm": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="segmentsemantic",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Vanishing Point": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="vanishing_point",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Room Layout": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="room_layout",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Color": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="colorization",
        #     mask_with_significance=args.mask_sig,
        # ),
        # "Inpainting Whole": make_volume(
        #     subj=args.subj,
        #     model="taskrepr",
        #     task="inpainting_whole",
        #     mask_with_significance=args.mask_sig,
        # ),
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
        "Eccentricity ROIs": ecc_roi_volume,
        "Places ROIs": place_roi_volume,
        "Faces ROIs": face_roi_volume,
    }

    for i in range(12):
        volumes["clip-vision-%d" % i+1] = make_volume(subj=args.subj, model="visual_layer_%d" % i)
    
    for i in range(12):
        volumes["clip-text-%d" % i+1] = make_volume(subj=args.subj, model="text_layer_%d" % i)

    if args.show_pcs:
        pc_vols = []
        PCs = np.load("%s/output/pca/subj%d/pca_components.npy" % (output_root, args.subj))
        # Normalize the PCs

        from util.util import zscore

        PCs_zscore = zscore(PCs, axis=1)

        # norm_PCs = PCs / np.sum(PCs, axis=1, keepdims=True)
        for i in range(PCs.shape[0]):
            key = "PC" + str(i)
            volumes[key] = make_pc_volume(
                args.subj, PCs_zscore[i, :], mask_with_significance=args.mask_sig
            )

        volumes["3PC"] = make_3pc_volume(
            args.subj, PCs_zscore, mask_with_significance=args.mask_sig
        )

    cortex.webgl.show(data=volumes, autoclose=False, port=24124)

    import pdb
    pdb.set_trace()
