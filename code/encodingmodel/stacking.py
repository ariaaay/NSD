from cvxopt import matrix, solvers
from sklearn.metrics import r2_score

from util.util import *

# Adapted from Ruogu Lin


def stack(err_list, yhat, y_test):
    """
    :param err_list: an 2D array of prediction error from each feature spaces mxn (m = # of features, n = # of voxels)
    :param yhat: a 3D array of prediction by different feature spaces (m x t x n) (m, n as above, t = # of trials)
    :param y_test: Test data (t x n)
    :return:
    """
    n_voxels = y_test.shape[1]
    n_features = err_list.shape[0]
    n_trials = y_test.shape[0]

    assert err_list.shape == (n_features, n_voxels)
    assert yhat.shape == (n_features, n_trials, n_voxels)

    P = np.zeros((n_voxels, n_features, n_features))
    # import pdb;pdb.set_trace()

    for i in range(n_features):
        for j in range(n_features):
            P[:, i, j] = (
                err_list[i] * err_list[j]
            )  # err is a list of errors for from each individual models
    # import pdb;pdb.set_trace()

    # PROGRAMATICALLY SET THIS FROM THE NUMBER OF FEATURES
    q = matrix(np.zeros((n_features)))
    G = matrix(-np.eye(n_features, n_features))
    h = matrix(np.zeros(n_features))
    A = matrix(np.ones((1, n_features)))
    b = matrix(np.ones(1))

    S = np.zeros((n_voxels, n_features))

    stacked_yhat = np.zeros_like(y_test)

    for i in range(0, n_voxels):
        PP = matrix(P[i])
        S[i, :] = np.array(solvers.qp(PP, q, G, h, A, b)["x"]).reshape(n_features,)
        # combine the predictions from the individual feature spaces for voxel i
        z = np.array([yhat[feature_j, :, i] for feature_j in range(n_features)])
        # if i == 0:
        # print(z.shape)  # to make sure
        # multiply the predictions by S[i,:]
        stacked_yhat[:, i] = np.dot(S[i, :], z)
    stacked_r2s = r2_score(stacked_yhat, y_test, multioutput="raw_values")
    return S, stacked_r2s
