import argparse

import pandas as pd
import numpy as np
from torch._C import Value
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
# from sklearn.decomposition import PCA

import torch
from torchvision import transforms
import torch.nn as nn

import blip_models

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def extract_last_layer_feature(model):
    preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    all_images_paths = ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
    print("Number of Images: {}".format(len(all_images_paths)))
    # if model == "clip":
    #     from  blip_models import CLIP_VITB16
    #     model = CLIP_VITB16()   
        
    # elif model == "simclr":
    #     from  blip_models import SIMCLR_VITB16
    #     model = SIMCLR_VITB16()

    ckpt_path = "models/%s_base_25ep.pt" % model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(blip_models, old_args.model)(rand_embed=False,
        ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    all_features = []

    for p in tqdm(all_images_paths):
        image = preprocess(Image.open(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            # print(image_features.shape)

        all_features.append(image_features.cpu().data.numpy())
    all_features = np.array(all_features)
    return all_features



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subj", default=1, type=int)
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/features",
    )
    parser.add_argument(
        "--project_output_dir",
        type=str,
        default="/user_data/yuanw3/project_outputs/NSD/output",
    )
    parser.add_argument(
        "--model",
        type=str,
    )

    args = parser.parse_args()
    stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"

    print(args)
    if args.subj == 0:
        for s in range(8):
            print("Extracting subj%01d" % (s + 1))
            feature_output_dir = "%s/subj%01d" % (args.feature_dir, (s + 1))
            all_coco_ids = np.load(
                "%s/coco_ID_of_repeats_subj%02d.npy"
                % (args.project_output_dir, (s + 1))
            )
            try:
                np.load("%s/YFCC_%s.npy" % (feature_output_dir, args.model))
            except FileNotFoundError:
                feat = extract_last_layer_feature(args.model)
                np.save("%s/YFCC_%s.npy" % (feature_output_dir, args.model), feat)

    else:
        all_coco_ids = np.load(
            "%s/coco_ID_of_repeats_subj%02d.npy" % (args.project_output_dir, args.subj)
        )
        feature_output_dir = "%s/subj%01d" % (args.feature_dir, args.subj)
        feat = extract_last_layer_feature(args.model)
        np.save("%s/YFCC_%s.npy" % (feature_output_dir, args.model), feat)
