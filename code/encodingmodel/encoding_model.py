import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import (
    KFold,
    PredefinedSplit,
    train_test_split,
    GroupShuffleSplit,
    ShuffleSplit,
)
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from util.util import pearson_corr, empirical_p
from encodingmodel.ridge import RidgeCVEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def ridge_cv(
    X,
    y,
    run_group=None,
    pca=False,
    tol=8,
    nfold=7,
    cv=False,
    fix_testing=False,
    permute_y=False,
):
    # fix_tsesting can be True (42), False, and a seed
    if fix_testing is True:
        fix_testing_state = 42
    elif fix_testing is False:
        fix_testing_state = None
    else:
        fix_testing_state = fix_testing

    scoring = lambda y, yhat: -torch.nn.functional.mse_loss(yhat, y)

    alphas = torch.from_numpy(
        np.logspace(-tol, 1 / 2 * np.log10(X.shape[1]) + tol, 100)
    )

    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=fix_testing_state
    )

    X_train = torch.from_numpy(X_train).to(dtype=torch.float64).to(device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float64).to(device)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float64).to(device)

    # model selection

    if cv:
        kfold = KFold(n_splits=nfold)
    else:
        tr_index, _ = next(
            ShuffleSplit(test_size=0.15).split(
                X_train, y_train
            )  # split training and testing
        )
        # set predefined train and validation split
        test_fold = np.zeros(X_train.shape[0])
        test_fold[tr_index] = -1
        kfold = PredefinedSplit(test_fold)
        assert kfold.get_n_splits() == 1

    clf = RidgeCVEstimator(alphas, kfold, scoring, scale_X=False)

    print("Fitting ridge models...")

    clf.fit(X_train, y_train)

    weights, bias = clf.get_model_weights_and_bias()

    print("Making predictions using ridge models...")
    yhat = clf.predict(X_test).cpu().numpy()
    try:
        rsqs = [r2_score(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]
    except ValueError:  # debugging for NaNs in subj 5
        print("Ytest: NaNs? Finite?")
        print(np.any(np.isnan(y_test)))
        print(np.all(np.isfinite(y_test)))
        print("Yhat: NaNs? Finite?")
        print(np.any(np.isnan(yhat)))
        print(np.all(np.isfinite(yhat)))

    corrs = [pearsonr(y_test[:, i], yhat[:, i]) for i in range(y_test.shape[1])]

    if not permute_y:
        return (
            corrs,
            rsqs,
            clf.mean_cv_scores.cpu().numpy(),
            clf.best_l_scores.cpu().numpy(),
            clf.best_l_idxs.cpu().numpy(),
            [yhat, y_test],
            weights.cpu().numpy(),
            bias.cpu().numpy(),
        )

    else:  # permutation testings
        print("running permutation test (permutating test labels 5000 times).")
        repeat = 5000
        corrs_dist = list()
        label_idx = np.arange(y_test.shape[0])
        for _ in tqdm(range(repeat)):
            np.random.shuffle(label_idx)
            y_test_perm = y_test[label_idx, :]
            perm_corrs = pearson_corr(y_test_perm, yhat, rowvar=False)
            corrs_dist.append(perm_corrs)
        corr_only = [r[0] for r in corrs]
        p = empirical_p(corr_only, np.array(corrs_dist))
        assert len(p) == y_test.shape[1]
        return corrs_dist, p, None


def fit_encoding_model(
    X,
    y,
    model_name=None,
    subj=1,
    fix_testing=False,
    cv=False,
    saving=True,
    permute_y=False,
    output_dir=None,
):

    model_name += "_whole_brain"

    if cv:
        print("Running cross validation")

    if output_dir is None:
        outpath = "output/encoding_results/subj%d/" % subj
    else:
        outpath = "%s/encoding_results/subj%d/" % (output_dir, subj)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    (
        corrs_array,
        rsqs_array,
        cv_array,
        l_score_array,
        best_l_array,
        predictions_array,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    assert (
        y.shape[0] == X.shape[0]
    )  # test that shape of features spaces and the brain are the same

    corrs_array, *cv_outputs = ridge_cv(
        X,
        y,
        cv=False,
        fix_testing=fix_testing,
        permute_y=permute_y,
    )

    if permute_y:  # if running permutation just return subsets of the output
        # save correaltions
        np.save(
            "output/permutation_results/subj%s/permutation_test_on_test_data_corr_%s_whole_brain.npy"
            % (subj, model_name),
            np.array(corrs_array),
        )
        # save p-values
        pickle.dump(
            cv_outputs[0],
            open(
                "output/permutation_results/subj%s/permutation_test_on_test_data_pvalue_%s.p"
                % (subj, model_name),
                "wb",
            ),
        )
        # return np.array(corrs_array), cv_outputs[0]

    if saving:
        pickle.dump(corrs_array, open(outpath + "corr_%s.p" % model_name, "wb"))

        if len(cv_outputs) > 0:
            pickle.dump(cv_outputs[0], open(outpath + "rsq_%s.p" % model_name, "wb"))
            pickle.dump(
                cv_outputs[1],
                open(outpath + "cv_score_%s.p" % model_name, "wb"),
            )
            pickle.dump(
                cv_outputs[2],
                open(outpath + "l_score_%s.p" % model_name, "wb"),
            )
            pickle.dump(
                cv_outputs[3],
                open(outpath + "best_l_%s.p" % model_name, "wb"),
            )

            if fix_testing:
                pickle.dump(
                    cv_outputs[4],
                    open(outpath + "pred_%s.p" % model_name, "wb"),
                )

            np.save("%sweights_%s.npy" % (outpath, model_name), cv_outputs[5])
            np.save("%sbias_%s.npy" % (outpath, model_name), cv_outputs[6])

    return np.array(corrs_array), None


def permutation_test(
    X,
    y,
    model_name,
    repeat=5000,
    subj=1,
    pca=False,
    permute_y=True,  # rather than permute training
    output_dir=None,
):
    """
    Running permutation test (permute the label 5000 times).
    """
    model_name += "_whole_brain"
    if outdir is None:
        outdir = "output/permutation_results/subj%d/" % subj
    else:
        outdir = "%s/subj%d/" % (output_dir, subj)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print("Running permutation test of {} for {} times".format(model_name, repeat))
    corr_dists, rsq_dists = list(), list()
    if permute_y:  # permute inside ridge cv
        print("Permutation testing by permuting test data.")
        _ = fit_encoding_model(
            X,
            y,
            model_name=model_name,
            subj=subj,
            cv=False,
            saving=False,
            permute_y=True,
            fix_testing=False,
            output_dir=output_dir,
        )
    else:
        label_idx = np.arange(X.shape[0])
        for _ in tqdm(range(repeat)):
            np.random.shuffle(label_idx)
            X_perm = X[label_idx, :]
            corrs_array, *cv_outputs = fit_encoding_model(
                X_perm,
                y,
                model_name=model_name,
                subj=subj,
                cv=False,
                saving=False,
                fix_testing=False,
            )
        corr_dists.append(corrs_array)
        # rsq_dists.append(rsqs_array)

        pickle.dump(
            corr_dists,
            open(
                "output/permutation_results/subj%s/permutation_test_on_training_data_corr_%s.p"
                % (subj, model_name),
                "wb",
            ),
        )
