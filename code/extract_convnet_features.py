import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms, utils, models

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from util.util import pytorch_pca


import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

preprocess = transforms.Compose(
    [
        # transforms.Resize(375),
        transforms.ToTensor()
    ]
)

# Load Images
stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images"

stim = pd.read_pickle(
    "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
)
all_coco_ids = stim.cocoId
all_images_paths = list()
all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
print("Number of Images: {}".format(len(all_images_paths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("layer", type=str, help="input name of the convolutional layer")
    parser.add_argument(
        "--model",
        type=str,
        default="vgg19",
        help="input name of the model to extract layer from",
    )
    parser.add_argument(
        "--subsample",
        type=str,
        default="avgpool",
        help="Please specify the method to subsample convolutional layers. Options are PCA and "
        "avgpool.",
    )
    parser.add_argument(
        "--subsampling_size",
        type=int,
        default=20000,
        help="Specify the target size for subsampling",
    )
    parser.add_argument("--cpu", action="store_true", help="cpu only for subsmapling.")

    args = parser.parse_args()
    print("Feature are subsampling with " + args.subsample)
    subsample_tag = "_" + args.subsample

    if 'conv' in args.layer:
        extract_conv = True
    else:
        extract_conv = False

    if args.model == "vgg19":
        from featureprep.convnet_extractor import Vgg19

        print("Extracting features from Vgg19_bn")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        model = Vgg19(args.layer, extract_conv).eval()

    elif args.model == "alexnet":
        from featureprep.convnet_extractor import AlexNet

        print("Extracting features from Alexnet...")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        model = AlexNet(args.layer, extract_conv).eval()

    # if there's only cpu, extracting feature is too slow so load pre-computed features
    if args.cpu and args.subsample == "pca":
        try:
            all_features = np.load(
                "/lab_data/tarrlab/common/datasets/features/NSD/feat_%s_%s%s.npy"
                % (args.model, args.layer, subsample_tag)
            )
        except FileNotFoundError:
            pass
    else:
        all_features = []
        for p in tqdm(all_images_paths):
            img = Image.open(p)
            input = Variable(preprocess(img).unsqueeze_(0)).to(device)
            out = model.forward(input, args.subsample, args.subsampling_size)

            all_features.append(out)
        all_features = np.array(all_features)
        # print(all_features.dtype)

    # PCA
    if args.subsample == "pca":
        print("Running PCA...")
        if args.cpu:
            pca = PCA()
            all_features = pca.fit_transform(all_features.astype(np.float16))
        else:
            all_features_full = torch.from_numpy(all_features).to(device)
            all_features = pytorch_pca(all_features_full).cpu()

    # Saving

    print(all_features.shape)
    np.save(
        "/lab_data/tarrlab/common/datasets/features/NSD/feat_%s_%s%s.npy"
        % (args.model, args.layer, subsample_tag),
        all_features,
    )
    # pickle.dump(all_images_paths, open('../outputs/convnet_features/convnet_image_orders_{}.p'.format(args.layer), 'wb'))

    # save the imtermediate product of PCAs for future use
    if args.subsample == "pca":
        # Save to file in the current working directory
        from joblib import dump

        joblib_filename = "pca_model_{}_{}.pkl".format(args.subsample, args.layer)
        dump(pca, joblib_filename)
