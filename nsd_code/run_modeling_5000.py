"""
This scripts takes an features space and runs encoding models (ridge regression) to
predict NSD brain data.
"""
import argparse
import numpy as np
import pickle
from encodingmodel.encoding_model import fit_encoding_model, permutation_test
from featureprep.feature_prep import get_features


def run(
        fm,
        br,
        model_name,
        test,
        notest,
        whole_brain,
        fix_testing,
        cv,
        # model_list,
):
    print("Features are {}. Using whole brain data: {}".format(model_name, whole_brain))

    if not test:
        print("Fitting Encoding Models")
        fit_encoding_model(
            fm,
            br,
            model_name=model_name,
            ROI=not whole_brain,
            subj=args.subj,
            fix_testing=fix_testing,
            cv=cv,
            saving=True,
            # model_list=model_list
        )
    if not notest:
        print("Running Permutation Test")
        permutation_test(
            fm,
            br,
            model_name=model_name,
            ROI=not whole_brain,
            subj=args.subj,
            # split_by_runs=split_by_runs,
            permute_y=args.permute_y,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="please specify features to model from"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        help="input the name of the features."
             "Options are: convnet, scenenet, response,surface_normal_latent, surface_normal_subsample, RT, etc",
    )
    # parser.add_argument(
    #     "--layer", type=str, default="", help="input name of the convolutional layer or task"
    # )
    parser.add_argument(
        "--notest", action="store_true", help="run encoding model with permutation test"
    )
    parser.add_argument(
        "--test", action="store_true", help="running permutation testing only"
    )
    parser.add_argument(
        "--roi", default=False, action="store_true", help="use roi data for modeling"
    )
    parser.add_argument(
        "--subj", type=int, default=1, help="specify which subject to build model on"
    )

    parser.add_argument(
        "--zscored", action="store_true", help="load zscored data"
    )

    parser.add_argument(
        "--fix_testing",
        action="store_true",
        help="used fixed sampling for training and testing",
    )
    parser.add_argument(
        "--cv", action="store_true", default=False, help="run cross-validation"
    )
    parser.add_argument(
        "--permute_y",
        action="store_true",
        default=False,
        help="permute test label but not training label",
    )
    args = parser.parse_args()

    mask_tag = ""
    if args.roi:
        mask_tag += "_roi_only"

    if args.zscored:
        mask_tag += "_zscore"

    brain_path = "output/averaged_cortical_responses_zscored_by_run_subj%02d%s.npy" % (args.subj, mask_tag)

    # Load brain data
    br_data = np.load(brain_path)

    # take the responses to the first 4916 trials
    br_data = br_data[:4916, :]

    print("Brain response size is: " + str(br_data.shape))


    # Load feature spaces
    print("Running ridge on " + args.model)

    stimulus_with_repeat = np.load("output/coco_ID_of_repeats_subj%02d.npy" % args.subj)
    try:
        assert stimulus_with_repeat.shape[1] ==3
    except AssertionError:
        print(stimulus_with_repeat.shape)

    stimulus_list = stimulus_with_repeat[:,0] #All subjects should have same orders

    feature_mat = get_features(
        args.subj,
        stimulus_list,
        args.model,
    )

    if len(feature_mat.shape) > 2:
        feature_mat = np.squeeze(feature_mat)

    # take the first 4916 stimuli
    feature_mat = feature_mat[:4916,:]

    print("Feature size is: " + str(feature_mat.shape))

    model_name_to_save = (
            args.model
            + mask_tag
    )

    run(
        feature_mat,
        br_data,
        model_name=model_name_to_save,
        test=args.test,
        notest=args.notest,
        whole_brain=not args.roi,
        fix_testing=args.fix_testing,
        cv=args.cv,
    )
