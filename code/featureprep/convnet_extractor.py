import os
import pickle
import json
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms, utils, models
from scipy.ndimage import gaussian_filter
from collections import namedtuple
from sklearn.decomposition import PCA

from util.model_config import conv_layers, fc_layers
from util.util import pool_size



class Vgg19(nn.Module):
    def __init__(self, layer, extract_conv=True):
        super(Vgg19, self).__init__()
        conv_layers = {"conv1": 6, "conv2": 13, "conv3": 26, "conv4": 39, "conv5": 52}
        fc_layers = {"fc6": 1, "fc7": 4}
        self.extract_conv = extract_conv
        if self.extract_conv:
            self.layer_ind = conv_layers[layer]
        else:
            self.layer_ind = fc_layers[layer]

        # load models from PyTorch
        vgg19_bn = models.vgg19_bn(pretrained=True)
        vgg19_bn.to(device)
        for param in vgg19_bn.parameters():
            param.requires_grad = False
        vgg19_bn.eval()

        features = list(vgg19_bn.features)
        self.features = nn.ModuleList(features).eval()
        self.adaptivepool = vgg19_bn.avgpool
        if (
            not self.extract_conv
        ):  # if need fc layer then add those linear layers into the forward pass
            self.classifiers = nn.Sequential(
                *list(vgg19_bn.classifier.children())
            ).eval()

    def forward(self, x, subsample):
        results = []
        for ii, layer in enumerate(self.features):
            x = layer(x)
            if self.extract_conv and self.layer_ind == ii:
                if subsample == "avgpool":
                    k = pool_size(x.data, 20000, adaptive=True)
                    results = (
                        nn.functional.adaptive_avg_pool2d(x.data, (k, k))
                        .cpu()
                        .flatten()
                        .numpy()
                    )
                elif subsample == "pca":
                    if (
                        self.layer_ind == conv_layers["conv1"]
                    ):  # need to reduce dimension of the first layer by half for PCA
                        results = (
                            nn.functional.avg_pool2d(x.data, (2, 2))
                            .cpu()
                            .flatten()
                            .numpy()
                            .astype(np.float16)
                        )
                    else:
                        results = x.cpu().flatten().numpy().astype(np.float16)
                else:
                    results = x.cpu().flatten().numpy()
                break

        if not self.extract_conv:
            x = self.adaptivepool(x)
            x = x.view(-1)
            for ii, layer in enumerate(self.classifiers):
                x = layer(x)
                if self.layer_ind == ii:
                    results = x.view(-1).data.cpu().numpy()
                    break
        return results