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
    whole_brain,
    fix_testing,
    cv,
    output_dir,
):
    print("Features are {}. Using whole brain data: {}".format(model_name, whole_brain))
    if test:
        print("Running Permutation Test")
        permutation_test(
            fm,
            br,
            model_name=model_name,
            ROI=not whole_brain,
            subj=args.subj,
            permute_y=args.permute_y,
            output_dir=output_dir,
        )

    else:
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
            output_dir=output_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="please specify features to model from"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        nargs="+",
        help="input the names of the features."
        "Options are: convnet, scenenet, response,surface_normal_latent, surface_normal_subsample, RT, etc",
    )
    # parser.add_argument(
    #     "--layer", type=str, default="", help="input name of the convolutional layer or task"
    # )
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
    parser.add_argument(
        "--get_features_only",
        action="store_true",
        default=False,
        help="only generate features but not running the encoding models",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output"
    )
    args = parser.parse_args()

    brain_path = (
        "%s/cortical_voxels/averaged_cortical_responses_zscored_by_run_subj%02d.npy"
        % (args.output_dir, args.subj)
    )

    # Load brain data
    br_data = np.load(brain_path)
    print("Brain response size is: " + str(br_data.shape))

    # Load feature spaces
    print("Running ridge on :")
    print(args.model)

    stimulus_list = np.load("%s/coco_ID_of_repeats_subj%02d.npy" % (args.output_dir, args.subj))

    feature_mat = get_features(args.subj, stimulus_list, args.model[0],)
    model_name_to_save = args.model[0]

    if len(args.model) > 1:
        for model in args.model[1:]:
            more_feature = get_features(args.subj, stimulus_list, model,)
            feature_mat = np.hstack((feature_mat, more_feature))

            model_name_to_save += "_" + model
    
    print("Feature size is: " + str(feature_mat.shape))

    if not args.get_features_only:
        run(
            feature_mat,
            br_data,
            model_name=model_name_to_save,
            test=args.test,
            whole_brain=not args.roi,
            fix_testing=args.fix_testing,
            cv=args.cv,
            output_dir = args.output_dir,
        )
