"This scripts load feature spaces and prepares it for encoding model"
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable

# from util.util import *
# from featureprep.conv_autoencoder import Autoencoder, preprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def get_features(subj, stim_list, model, layer=None):
    print("Getting features for {}{}, for subject {}".format(model, layer, subj))
    # events also has a stim list, it is same as the "stim_lists.txt"; but repetition is not indicated in the file name.

    if "taskrepr" in model:
        # latent space in taskonomy, model should be in the format of "taskrep_X", e.g. taskrep_curvature
        task = "_".join(model.split("_")[1:])
        repr_dir = "/lab_data/tarrlab/yuanw3/taskonomy_features/genStimuli/{}".format(task)

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


        print("feature shape is " + str(featmat.shape[0]))

    return featmat
