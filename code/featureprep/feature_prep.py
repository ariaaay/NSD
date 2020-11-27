"This scripts load feature spaces and prepares it for encoding model"
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd

# from util.util import *
# from featureprep.conv_autoencoder import Autoencoder, preprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_features(subj, stim_list, model):
    """
    :param subj: subject ID
    :param stim_list: a list of COCO IDs for the stimuli images
    :param model: models to extract features from
    :return featmat: a matrix of features that matches with the order of brain data
    """
    subj = int(subj)
    print("Getting features for %s, for subject %d" % (model, subj))
    # events also has a stim list, it is same as the "stim_lists.txt"; but repetition is not indicated in the
    # file name.

    try:
        if subj == 0:
            featmat = np.load("features/%s.npy" % model)
        else:
            featmat = np.load("features/subj%d/%s.npy" % (subj, model))

    except FileNotFoundError:
        if "taskrepr" in model:
            # latent space in taskonomy, model should be in the format of "taskrepr_X", e.g. taskrep_curvature
            task = "_".join(model.split("_")[1:])
            repr_dir = "/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli/{}".format(
                task
            )

            featmat = []
            print("stimulus length is: " + str(len(stim_list)))
            for img_id in tqdm(stim_list):
                try:
                    fpath = "%s/%d.npy" % (repr_dir, img_id)
                    repr = np.load(fpath).flatten()
                except FileNotFoundError:
                    fpath = "%s/COCO_train2014_%012d.npy" % (repr_dir, img_id)
                    repr = np.load(fpath).flatten()
                featmat.append(repr)
            featmat = np.array(featmat)

        if (
            "convnet" in model
        ):  # model should be "convnet_vgg16" to load "feat_vgg16.npy"
            model_id = model.split("_")[1:]
            feat_name = "_".join(model_id)
            print("Loading convnet model %s..." % feat_name)
            # this extracted feature is order based on nsd ID (order in the stimulus info file)
            try:
                all_feat = np.load("features/feat_%s.npy" % feat_name)
            except FileNotFoundError:
                all_feat = np.load(
                    "/lab_data/tarrlab/common/datasets/features/NSD/feat_%s.npy"
                    % feat_name
                )  # on clsuter

            stim = pd.read_pickle(
                "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
            )

            featmat = []
            for img_id in tqdm(stim_list):
                try:
                    # extract the nsd ID corresponding to the coco ID in the stimulus list
                    stim_ind = stim["nsdId"][stim["cocoId"] == img_id]
                    # extract the repective features for that nsd ID
                    featmat.append(all_feat[stim_ind, :])
                except IndexError:
                    print("COCO Id Not Found: " + str(img_id))
            featmat = np.array(featmat)

        if subj == 0:  # meaning it is features for all subjects
            np.save("features/%s.npy" % (model), featmat)
        else:
            np.save("features/subj%d/%s.npy" % (subj, model), featmat)

        print("feature shape is " + str(featmat.shape[0]))

    if len(feature_mat.shape) > 2:
        feature_mat = np.squeeze(feature_mat)
    return featmat
