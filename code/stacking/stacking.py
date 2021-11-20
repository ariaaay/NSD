import numpy as np
import matplotlib.pyplot as plt
import cortex
import os

import scipy.io as sio
import mne
from scipy.stats import zscore
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

# %load_ext autoreload
# %autoreload 2
from ridge_tools import cross_val_ridge, R2, corr, R2r
from utils import delay_mat, smooth_run_not_masked, load_and_process


from cvxopt import matrix, solvers

solvers.options["show_progress"] = False
# import gurobipy as gp
# from gurobipy import GRB


score_f = R2


def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))
    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items :] = n_folds - 1
    return ind


from ridge_tools import cross_val_ridge, ridge


def quad_stack(PP, n_features):
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    return np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(
        n_features,
    )


def feat_ridge_CV(
    train_feature, train_data, test_feature, method="cross_val_ridge", n_folds=10
):

    if np.all(train_feature == 0):  # if zero predictor
        weights = np.zeros((train_feature.shape[1], train_data.shape[1]))
        preds_train = np.zeros_like(train_data)
    else:
        ind_nested = CV_ind(train_data.shape[0], n_folds=n_folds)
        preds_train = np.zeros_like(train_data)
        # weights = ridge(train_features[FEATURE],train_data, 1)

        for i_nested in range(n_folds):
            train_data_nested = np.nan_to_num(
                zscore(train_data[ind_nested != i_nested])
            )
            train_features_nested = np.nan_to_num(
                zscore(train_feature[ind_nested != i_nested])
            )
            test_features_nested = np.nan_to_num(
                zscore(train_feature[ind_nested == i_nested])
            )

            weights = ridge(train_features_nested, train_data_nested, 1)
            # preds_train[FEATURE][ind_nested==i_nested] = test_features_nested.dot(weights)

            if method == "simple_ridge":
                weights = ridge(train_feature, train_data, 100)
            elif method == "cross_val_ridge":
                if train_feature.shape[1] > train_feature.shape[0]:
                    weights, __ = cross_val_ridge(
                        train_features_nested,
                        train_data_nested,
                        n_splits=10,
                        lambdas=np.array([10 ** i for i in range(-6, 10)]),
                        do_plot=False,
                        method="plain",
                    )
                else:
                    weights, __ = cross_val_ridge(
                        train_features_nested,
                        train_data_nested,
                        n_splits=10,
                        lambdas=np.array([10 ** i for i in range(-6, 10)]),
                        do_plot=False,
                        method="plain",
                    )
            #                 elif method == 'simple_ridge_1':
            #                     weights = ridge(train_features[FEATURE],train_data, 1)
            # preds_train =  np.dot(train_feature, weights)
            preds_train[ind_nested == i_nested] = test_features_nested.dot(weights)

    err = train_data - preds_train

    # predict the test data also before overwriting the weights:
    preds_test = np.dot(test_feature, weights)
    # preds_test[FEATURE,test_ind] = zscore(preds_test[FEATURE][test_ind])
    # single feature space predictions, computed over a fold
    # r2s_folds[ind_num,FEATURE,:] = score_f(preds_test[FEATURE,test_ind],test_data)
    r2s_train_fold = score_f(preds_train, train_data)
    var_train_fold = np.var(preds_train, axis=0)

    return preds_train, err, preds_test, r2s_train_fold, var_train_fold


def stacking_CV_fmri(
    data,
    features,
    method="cross_val_ridge",
    n_folds=4,
    with_subs=True,
    with_subc=False,
    with_concat=True,
):

    # INPUTS: data (ntime*nvoxels), features (list of ntime*ndim), method = what to use to train,
    #         n_folds = number of cross-val folds

    n_time = data.shape[0]
    n_voxels = data.shape[1]
    n_features = len(features)

    ind = CV_ind(n_time, n_folds=n_folds)

    if with_concat == True:
        n_feat = n_features - 1
    else:
        n_feat = n_features

    # easier to store r2s in an array and access them programatically than to maintain a different
    # variable for each
    r2s = np.zeros((n_features, n_voxels))
    var = np.zeros((n_features, n_voxels))
    r2s_sub = np.zeros((n_feat, n_voxels))
    vars_sub = np.zeros((n_feat, n_voxels))
    r2c_sub = np.zeros((n_feat, n_voxels))
    varc_sub = np.zeros((n_feat, n_voxels))
    # r2s_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_train_folds = np.zeros((n_folds, n_features, n_voxels))
    var_train_folds = np.zeros((n_folds, n_features, n_voxels))
    r2s_weighted = np.zeros((n_features, n_voxels))
    var_weighted = np.zeros((n_features, n_voxels))
    # r2s_weighted_fold = np.zeros((n_folds, n_features, n_voxels))
    # stacked_r2s_fold = np.zeros((n_folds, n_voxels))
    stacked_train_r2s_fold = np.zeros((n_folds, n_voxels))
    stacked_pred = np.zeros((n_time, n_voxels))
    # concat_pred = np.zeros((n_time, n_voxels))
    stacked_pred_sub = np.zeros((n_feat, n_time, n_voxels))
    concat_pred_sub = np.zeros((n_feat, n_time, n_voxels))
    preds_test = np.zeros((n_features, n_time, n_voxels))
    weighted_pred = np.zeros((n_feat, n_time, n_voxels))

    S_average = np.zeros((n_voxels, n_feat))

    # DO BY FOLD
    for ind_num in range(n_folds):
        train_ind = ind != ind_num
        test_ind = ind == ind_num

        # split data
        train_data = data[train_ind]
        train_features = [F[train_ind] for F in features]

        test_data = data[test_ind]
        test_features = [F[test_ind] for F in features]

        # normalize data  <= WE SHOULD ZSCORE BY TRAIN/TEST
        train_data = np.nan_to_num(zscore(train_data))
        test_data = np.nan_to_num(zscore(test_data))

        train_features = [np.nan_to_num(zscore(F)) for F in train_features]
        test_features = [np.nan_to_num(zscore(F)) for F in test_features]

        err = dict()
        preds_train = dict()

        for FEATURE in range(n_features):
            (
                preds_train[FEATURE],
                error,
                preds_test[FEATURE, test_ind],
                r2s_train_folds[ind_num, FEATURE, :],
                var_train_folds[ind_num, FEATURE, :],
            ) = feat_ridge_CV(
                train_features[FEATURE],
                train_data,
                test_features[FEATURE],
                method=method,
            )

            if (with_concat == True) and (FEATURE == n_feat):
                pass
            else:
                err[FEATURE] = error

        if with_subc:
            for ifeature in range(n_feat - 1):
                tmp_features_train = train_features.copy()
                del tmp_features_train[ifeature]
                tmp_features_test = test_features.copy()
                del tmp_features_test[ifeature]
                train_feature_c = np.hstack(tmp_features_train)
                test_feature_c = np.hstack(tmp_features_test)

                _, _, concat_pred_sub[ifeature, test_ind, :], _, _ = feat_ridge_CV(
                    train_feature_c, train_data, test_feature_c, method=method
                )

        P = np.zeros((n_voxels, n_feat, n_feat))
        for i in range(n_feat):
            for j in range(n_feat):
                P[:, i, j] = np.mean(err[i] * err[j], 0)

        S = np.zeros((n_voxels, n_feat))

        stacked_pred_train = np.zeros_like(train_data)

        for i in range(0, n_voxels):
            PP = matrix(P[i])
            S[i, :] = quad_stack(PP, n_feat)
            # combine the predictions from the individual feature spaces for voxel i
            z = np.array(
                [preds_test[feature_j, test_ind, i] for feature_j in range(n_feat)]
            )
            # if i==0:
            # print(z.shape) # to make sure
            # multiply the predictions by S[i,:]
            stacked_pred[test_ind, i] = np.dot(S[i, :], z)

            # perform subtraction analysis:
            if with_subs:
                for ifeature in range(n_feat):
                    ind_feature = np.ones(n_feat)
                    ind_feature[ifeature] = 0
                    PP2 = matrix(P[i][ind_feature == 1][:, ind_feature == 1])
                    S_feature = quad_stack(PP2, n_feat - 1)
                    stacked_pred_sub[ifeature, test_ind, i] = np.dot(
                        S_feature, z[ind_feature == 1, :]
                    )

            # combine the training predictions from the individual feature spaces for voxel i
            z = np.array([preds_train[feature_j][:, i] for feature_j in range(n_feat)])
            stacked_pred_train[:, i] = np.dot(S[i, :], z)
        #             if score_f(stacked_pred_train[:,i],train_data[:,i])>0.1:
        #                 a = blablabla

        S_average += S

        # stacked prediction, computed over a fold
        # stacked_r2s_fold[ind_num,:] = score_f(stacked_pred[test_ind],test_data)
        stacked_train_r2s_fold[ind_num, :] = score_f(stacked_pred_train, train_data)

        for FEATURE in range(n_feat):
            # weight the predictions according to S:
            # weighted single feature space predictions, computed over a fold
            weighted_pred[FEATURE, test_ind] = (
                preds_test[FEATURE, test_ind] * S[:, FEATURE]
            )
    #             r2s_weighted_fold[ind_num,FEATURE,:] = score_f(weighted_pred[FEATURE,test_ind],test_data)

    # compute overall
    for FEATURE in range(n_features):
        r2s[FEATURE, :] = score_f(preds_test[FEATURE], data)
        var[FEATURE, :] = np.var(preds_test[FEATURE], axis=0)

    for FEATURE in range(n_feat):
        r2s_weighted[FEATURE, :] = score_f(weighted_pred[FEATURE], data)
        var_weighted[FEATURE, :] = np.var(weighted_pred[FEATURE], axis=0)

    stacked_r2s = score_f(stacked_pred, data)
    stacked_var = np.var(stacked_pred, axis=0)

    for FEATURE in range(n_feat):
        r2s_sub[FEATURE, :] = stacked_r2s - score_f(stacked_pred_sub[FEATURE], data)
        vars_sub[FEATURE, :] = np.var(stacked_pred_sub[FEATURE], axis=0)
        r2c_sub[FEATURE, :] = r2s[-1, :] - score_f(concat_pred_sub[FEATURE], data)
        varc_sub[FEATURE, :] = np.var(concat_pred_sub[FEATURE], axis=0)

    r2s_train = r2s_train_folds.mean(0)
    var_train = var_train_folds.mean(0)
    stacked_train = stacked_train_r2s_fold.mean(0)
    S_average = S_average / n_folds

    Result = dict()
    Result["r2s"] = r2s
    Result["var"] = var
    Result["stacked_r2s"] = stacked_r2s
    Result["stacked_var"] = stacked_var
    Result["r2s_weighted"] = r2s_weighted
    Result["var_weighted"] = var_weighted
    Result["r2s_sub"] = r2s_sub
    Result["vars_sub"] = vars_sub
    Result["r2c_sub"] = r2c_sub
    Result["varc_sub"] = varc_sub
    Result["r2s_train"] = r2s_train
    Result["var_train"] = var_train
    Result["stacked_train"] = stacked_train
    Result["S_average"] = S_average

    return Result
