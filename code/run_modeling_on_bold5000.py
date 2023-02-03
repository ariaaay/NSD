"""
This scripts takes an features space and runs encoding models (ridge regression) to
predict BOLD5000 data.
"""
import re
import pickle
import json
import argparse
import numpy as np
from encodingmodel.encoding_model import fit_encoding_model, bootstrap_test
from featureprep.feature_prep import (
    get_preloaded_features,
    extract_feature_with_image_order,
)
from util.data_util import load_subset_trials

def zero_strip(s):
    if s[0] == "0":
        s = s[1:]
        return zero_strip(s)
    else:
        return s


def extract_dataset_index(stim_list, dataset="all", rep=False, return_filename=False):
    with open("%s/stimuli_info/imgnet_imgsyn_dict.pkl" % args.output_dir, "rb") as f:
        stim_imgnet = pickle.load(f)

    with open("%s/stimuli_info/COCO_img2cats.json" % args.output_dir) as f:
        COCO_img2cats = json.load(f)  # categories info

    dataset_labels = stim_list.copy()
    COCO_idx, imagenet_idx, SUN_idx = list(), list(), list()
    COCO_cat_list, imagenet_cat_list, SUN_cat_list = list(), list(), list()
    for i, n in enumerate(stim_list):
        if "COCO_" in n:
            if "rep_" in n and rep is False:
                continue
            dataset_labels[i] = "COCO"
            COCO_idx.append(i)
            n.split()  # takeout \n
            COCO_id = zero_strip(str(n[21:-5]))
            COCO_cat_list.append(COCO_img2cats[COCO_id])
        elif "JPEG" in n:  # imagenet
            dataset_labels[i] = "imagenet"
            if "rep_" in n:
                if not rep:
                    continue
                else:
                    n = n[4:]
            syn = stim_imgnet[n[:-1]]
            imagenet_idx.append(i)
            imagenet_cat_list.append(syn)
        else:
            dataset_labels[i] = "SUN"
            name = n.split(".")[0]
            if "rep_" in name:
                if not rep:
                    continue
                else:
                    name = name[4:]
            SUN_idx.append(i)
            if return_filename:
                SUN_cat_list.append(name)
            else:
                SUN_cat_list.append(re.split("[0-9]", name)[0])
    if rep:
        assert len(stim_list) == len(COCO_idx) + len(imagenet_idx) + len(SUN_idx)

    if dataset == "COCO":
        return COCO_idx, COCO_cat_list
    elif dataset == "imagenet":
        return imagenet_idx, imagenet_cat_list
    elif dataset == "SUN":
        return SUN_idx, SUN_cat_list
    else:
        return dataset_labels


def get_features(subj, model, dataset="", br_subset_idx=None):
    with open("%s/BOLD5000/CSI%02d_stim_lists.txt" % (args.proj_dir, subj)) as f:
        sl = f.readlines()
    stim_list = [item.strip("\n") for item in sl]

    imgnet_idx, imgnet_cats = extract_dataset_index(sl, dataset="imagenet", rep=False)
    scene_idx, scene_cats = extract_dataset_index(sl, dataset="SUN", rep=False)
    COCO_idx, COCO_cats = extract_dataset_index(sl, dataset="COCO", rep=False)

    # Load features list generated with the whole brain data. This dictionary includes: image names, valence responses,
    # reaction time, session number, etc.
    with open("{}/CSI{}_events.json".format(cortical_dir, subj)) as f:
        events = json.load(f)

    # events also has a stim list, it is same as the "stim_lists.txt"; but repetition is not indicated in the file name.

    if (
            dataset is not ""
    ):  # only an argument for features spaces that applies to all
        if dataset == "ImageNet":
            br_subset_idx = imgnet_idx
            stim_list = np.array(stim_list)[imgnet_idx]
        elif dataset == "COCO":
            br_subset_idx = COCO_idx
            stim_list = np.array(stim_list)[COCO_idx]
        elif dataset == "SUN":
            br_subset_idx = scene_idx
            stim_list = np.array(stim_list)[scene_idx]

    if br_subset_idx is None:
        br_subset_idx = get_nonrep_index(stim_list)

    featmat = [] #TODO

    return featmat, br_subset_idx

def run(
    fm,
    br,
    model_name,
    test,
    fix_testing,
    cv,
    output_dir,
):
    if test:
        print("Running Bootstrap Test")
        bootstrap_test(
            fm,
            br,
            model_name=model_name,
            subj=args.subj,
            output_dir=output_dir,
        )

    else:
        print("Fitting Encoding Models")
        fit_encoding_model(
            fm,
            br,
            model_name=model_name,
            subj=args.subj,
            fix_testing=fix_testing,
            cv=cv,
            saving=True,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Please specify features to model from and parameters of the encoding model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        nargs="+",
        help="input the names of the features.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="input name of the layer. e.g. input_layer1",
    )
    parser.add_argument("--test", action="store_true", help="Run bootstrap testing.")
    parser.add_argument(
        "--subj",
        type=int,
        default=1,
        help="Specify which subject to build model on. Currently it supports subject 1, 2, 3.",
    )
    parser.add_argument(
        "--fix_testing",
        action="store_true",
        help="Use fixed sampling for training and testing (for model performance comparison purpose)",
    )
    parser.add_argument(
        "--cv", action="store_true", default=False, help="run cross-validation."
    )
    parser.add_argument(
        "--get_features_only",
        action="store_true",
        default=False,
        help="only generate and save the feature matrix but not running the encoding models (for preloaded features)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/BOLD5000/outputs",
        help="Specify the path to the output directory",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/BOLD5000/features",
        help="Specify the path to the features directory",
    )
    parser.add_argument(
        "--feature_matrix",
        type=str,
        default=None,
        help="Specify the path to the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--feature_order",
        type=str,
        default=None,
        help="Specify the path to the ordering of the feature matrix (should be a numpy array)",
    )
    parser.add_argument(
        "--model_name_to_save",
        type=str,
        default=None,
        help="Specify a name to save the performance with",
    )
    parser.add_argument(
        "--subset_data",
        type=str,
        default=None,
        help="specify a category to subset training and testing data",
    )

    args = parser.parse_args()

    feature_mat, br_subset_idx = get_features(
        args.subj,
        args.model,
        layer=args.layer,
        dim=args.dim,
        )

    brain_path = (
        "%s/cortical_voxels/cortical_voxel_across_sessions_zscored_CSI%d.npy"
        % (args.output_dir, args.subj)
    )

    # Load brain data
    br_data = np.load(brain_path)
    br_data = br_data[br_subset_idx, :]

    # deal with voxels that are zeros in runs and therefore cause nan values in zscoring
    # only happens in some subjects (e.g. subj5)
    try:
        non_zero_mask = np.load(
            "%s/voxels_masks/CSI%d/nonzero_voxels.npy.npy"
            % (args.output_dir, args.subj)
        )
        print("Masking zero voxels...")
        br_data = br_data[:, non_zero_mask]
    except FileNotFoundError:
        pass

    print("NaNs? Finite?:")
    print(np.any(np.isnan(br_data)))
    print(np.all(np.isfinite(br_data)))
    print("Brain response size is: " + str(br_data.shape))

   

    # Load feature spaces
    if args.feature_matrix is not None:  # for general design matrix input
        feature_mat_unordered = np.load(args.feature_matrix)
        image_order = np.laod(args.image_order)
        model_name_to_save = args.model_name_to_save
        feature_mat = extract_feature_with_image_order(
            stimulus_list, feature_mat_unordered, image_order
        )
    else:
        if args.layer is not None:
            model_name_to_save = args.model[0] + "_" + args.layer
        else:
            model_name_to_save = args.model[0]

        feature_mat = get_preloaded_features(
            args.subj,
            stimulus_list,
            args.model[0],
            layer=args.layer,
            features_dir=args.features_dir,
        )

        if len(args.model) > 1:
            for model in args.model[1:]:
                more_feature = get_preloaded_features(
                    args.subj, stimulus_list, model, features_dir=args.features_dir
                )
                feature_mat = np.hstack((feature_mat, more_feature))

                model_name_to_save += "_" + model

    if args.subset_data is not None:
        subset_cat = args.subset_data
        print("Subsetting training data with criteria: " + subset_cat)
        if (
            "no" in subset_cat
        ):  # selecting the trials that didnt contain certain categories
            subset_cat = subset_cat.split("_")[-1]
            subset_trial_id = load_subset_trials(stimulus_list, subset_cat, negcat=True)
        else:
            subset_trial_id = load_subset_trials(stimulus_list, subset_cat)
        br_data = br_data[subset_trial_id, :]
        feature_mat = feature_mat[subset_trial_id, :]
        model_name_to_save += "_" + args.subset_data + "_subset"

    print("=======================")
    print("Running ridge encoding model on :")
    print(model_name_to_save)

    print("Feature size is: " + str(feature_mat.shape))
    print("=======================")

    if not args.get_features_only:
        run(
            feature_mat,
            br_data,
            model_name=model_name_to_save,
            test=args.test,
            fix_testing=args.fix_testing,
            cv=args.cv,
            output_dir=args.output_dir,
        )
