import copy
import argparse

import pandas as pd
import numpy as np
from torch._C import Value
from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA

import torch
import torchvision
import torch.nn as nn

import torchextractor as tx

import clip
from util.util import pytorch_pca
from util.data_util import load_top1_objects_in_COCO, load_objects_in_COCO

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_captions(cid):
    annIds = train_caps.getAnnIds(imgIds=[cid])
    anns = train_caps.loadAnns(annIds)
    if anns == []:
        annIds = val_caps.getAnnIds(imgIds=[cid])
        anns = val_caps.loadAnns(annIds)

    if anns == []:
        print("no captions extracted for image: " + str(cid))

    captions = [d["caption"] for d in anns]
    return captions


def load_object_caption_overlap(cid):
    caption = load_captions(cid)
    objs = load_objects_in_COCO(cid)
    for k in expand_dict.keys():
        if k in objs:
            objs += expand_dict[k]
    all_caps = ""
    for c in caption:
        all_caps += c
    obj_intersect = [o for o in objs if o in all_caps]

    print(objs)
    print(all_caps)
    print(obj_intersect)
    return obj_intersect


def extract_object_base_text_feature():
    model, _ = clip.load("ViT-B/32", device=device)
    all_features = []
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            objects = load_objects_in_COCO(cid)
            expression = "a photo of " + " ".join(objects)
            text = clip.tokenize(expression).to(device)
            cap_emb = model.encode_text(text).cpu().data.numpy()
            all_features.append(cap_emb)

    all_features = np.array(all_features).squeeze()
    print("Feature shape is: " + str(all_features.shape))
    np.save(
        "%s/clip_object.npy" % feature_output_dir,
        all_features,
    )


def extract_top1_obejct_base_text_feature():
    model, _ = clip.load("ViT-B/32", device=device)
    all_features = []
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            obj = load_top1_objects_in_COCO(cid)
            text = clip.tokenize("a photo of a " + obj).to(device)
            cap_emb = model.encode_text(text).cpu().data.numpy()
            all_features.append(cap_emb)

    all_features = np.array(all_features).squeeze()
    print("Feature shape is: " + str(all_features.shape))
    np.save(
        "%s/clip_top1_object.npy" % feature_output_dir,
        all_features,
    )


def extract_obj_cap_intersect_text_feature():
    model, _ = clip.load("ViT-B/32", device=device)
    all_features = []
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            overlaps = load_object_caption_overlap(cid)
            text = clip.tokenize(overlaps).to(device)
            cap_emb = model.encode_text(text).cpu().data.numpy()
            all_features.append(cap_emb)

    all_features = np.array(all_features)
    print("Feature shape is: " + str(all_features.shape))
    np.save(
        "%s/clip_object_caption_overlap.npy" % feature_output_dir,
        all_features,
    )
    return


# def extract_visual_resnet_prePCA_feature():
#     LOI_ResNet_vision = [
#         "visual.bn1",
#         "visual.avgpool",
#         "visual.layer1.2.bn3",
#         "visual.layer2.3.bn3",
#         "visual.layer3.5.bn3",
#         "visual.layer4.2.bn3",
#         "visual.attnpool",
#     ]
#     model, preprocess = clip.load("RN50", device=device)
#     model_visual = tx.Extractor(model, LOI_ResNet_vision)
#     compressed_features = [copy.copy(e) for _ in range(8) for e in [[]]]
#     subsampling_size = 5000

#     print("Extracting ResNet features")
#     for cid in tqdm(all_coco_ids):
#         with torch.no_grad():
#             image_path = "%s/%s.jpg" % (stimuli_dir, cid)
#             image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#             captions = load_captions(cid)
#             text = clip.tokenize(captions).to(device)

#             _, features = model_visual(image, text)

#             for i, f in enumerate(features.values()):
#                 # print(f.size())
#                 if len(f.size()) > 3:
#                     c = f.data.shape[1]  # number of channels
#                     k = int(np.floor(np.sqrt(subsampling_size / c)))
#                     tmp = nn.functional.adaptive_avg_pool2d(f.data, (k, k))
#                     # print(tmp.size())
#                     compressed_features[i].append(tmp.squeeze().cpu().numpy().flatten())
#                 else:
#                     compressed_features[i].append(
#                         f.squeeze().data.cpu().numpy().flatten()
#                     )

#     for l, f in enumerate(compressed_features):
#         np.save("%s/visual_layer_resnet_prePCA_%01d.npy" % (feature_output_dir, l), f)
#         # compressed_features_array.append(np.array(f))


def extract_visual_resnet_prePCA_feature():
    LOI_ResNet_vision = [
        "visual.bn1",
        "visual.avgpool",
        "visual.layer1.2.bn3",
        "visual.layer2.3.bn3",
        "visual.layer3.5.bn3",
        "visual.layer4.2.bn3",
        "visual.attnpool",
    ]
    model, preprocess = clip.load("RN50", device=device)
    model_visual = tx.Extractor(model, LOI_ResNet_vision)
    compressed_features = [copy.copy(e) for _ in range(8) for e in [[]]]
    subsampling_size = 5000

    print("Extracting ResNet features")
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            captions = load_captions(cid)
            text = clip.tokenize(captions).to(device)

            _, features = model_visual(image, text)

            for i, f in enumerate(features.values()):
                # print(f.size())
                if len(f.size()) > 3:
                    c = f.data.shape[1]  # number of channels
                    k = int(np.floor(np.sqrt(subsampling_size / c)))
                    tmp = nn.functional.adaptive_avg_pool2d(f.data, (k, k))
                    # print(tmp.size())
                    compressed_features[i].append(tmp.squeeze().cpu().numpy().flatten())
                else:
                    compressed_features[i].append(
                        f.squeeze().data.cpu().numpy().flatten()
                    )

    for l, f in enumerate(compressed_features):
        np.save("%s/visual_layer_resnet_prePCA_%01d.npy" % (feature_output_dir, l), f)


def extract_visual_resnet_feature():
    for l in range(7):
        try:
            f = np.load(
                "%s/visual_layer_resnet_prePCA_%01d.npy" % (feature_output_dir, l)
            )
        except FileNotFoundError:
            extract_visual_resnet_prePCA_feature()
            f = np.load(
                "%s/visual_layer_resnet_prePCA_%01d.npy" % (feature_output_dir, l)
            )

        print("Running PCA")
        print("feature shape: ")
        print(f.shape)
        pca = PCA(n_components=min(f.shape[0], 64), svd_solver="auto")

        fp = pca.fit_transform(f)
        print("Feature %01d has shape of:" % l)
        print(fp.shape)

        np.save("%s/visual_layer_resnet_%01d.npy" % (feature_output_dir, l), fp)


def extract_visual_transformer_feature():
    model, preprocess = clip.load("ViT-B/32", device=device)
    LOI_transformer_vision = [
        "visual.transformer.resblocks.%01d.ln_2" % i for i in range(12)
    ]
    model_visual = tx.Extractor(model, LOI_transformer_vision)
    compressed_features = [copy.copy(e) for _ in range(12) for e in [[]]]

    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            captions = load_captions(cid)
            text = clip.tokenize(captions).to(device)

            _, features = model_visual(image, text)

            for i, f in enumerate(features.values()):
                compressed_features[i].append(f.squeeze().cpu().data.numpy().flatten())

    compressed_features = np.array(compressed_features)

    for l, f in enumerate(compressed_features):
        pca = PCA(n_components=min(f.shape[0], 64), whiten=True, svd_solver="full")
        try:
            fp = pca.fit_transform(f)
        except ValueError:
            print(fp.shape)

        print("Feature %01d has shape of:" % l)
        print(fp.shape)

        np.save("%s/visual_layer_%01d.npy" % (feature_output_dir, l), fp)


def extract_text_layer_feature():
    model, preprocess = clip.load("ViT-B/32", device=device)
    LOI_text = ["transformer.resblocks.%01d.ln_2" % i for i in range(12)]
    text_features = [copy.copy(e) for _ in range(12) for e in [[]]]
    model_text = tx.Extractor(model, LOI_text)
    for cid in tqdm(all_coco_ids):
        with torch.no_grad():
            image_path = "%s/%s.jpg" % (stimuli_dir, cid)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            captions = load_captions(cid)

            layer_features = [
                copy.copy(e) for _ in range(12) for e in [[]]
            ]  # layer_features is 12 x 5 x m
            for caption in captions:
                text = clip.tokenize(caption).to(device)
                _, features = model_text(image, text)
                # features is a feature dictionary for all layers, each image, each caption
                for i, layer in enumerate(LOI_text):
                    layer_features[i].append(
                        features[layer].cpu().data.numpy().squeeze().flatten()
                    )

                # print(np.array(layer_features).shape)
            avg_features = np.mean(np.array(layer_features), axis=1)  # 12 x m

        for i in range(len(LOI_text)):
            text_features[i].append(avg_features[i])

    text_features = np.array(text_features)
    # print(text_features.shape) # 12 x 10000 x m

    for l, f in enumerate(text_features):
        pca = PCA(n_components=min(f.shape[0], 64), whiten=True, svd_solver="full")
        try:
            fp = pca.fit_transform(f)
        except ValueError:
            print(fp.shape)

        print("Feature %01d has shape of:" % l)
        print(fp.shape)

        np.save("%s/text_layer_%01d.npy" % (feature_output_dir, l), fp)


def extract_last_layer_feature(model_name="ViT-B/32", modality="vision"):
    all_images_paths = list()
    all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
    print("Number of Images: {}".format(len(all_images_paths)))
    model, preprocess = clip.load(model_name, device=device)

    if modality == "vision":
        all_features = []

        for p in tqdm(all_images_paths):
            image = preprocess(Image.open(p)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)

            all_features.append(image_features.cpu().data.numpy())
        all_features = np.array(all_features)
        # print(all_features.shape)
        # np.save(
        #     "/lab_data/tarrlab/common/datasets/features/NSD/clip_.npy" % model_name,
        #     all_features,
        # )
        return all_features
    elif modality == "text": # this is subject specific
        # extract text feature of image titles
        all_text_features = []
        for cid in tqdm(all_coco_ids):
            with torch.no_grad():
                captions = load_captions(cid)
                # print(captions)
                embs = []
                for caption in captions:
                    text = clip.tokenize(caption).to(device)
                    embs.append(model.encode_text(text).cpu().data.numpy())

                mean_emb = np.mean(np.array(embs), axis=0).squeeze()

                all_text_features.append(mean_emb)

        all_text_features = np.array(all_text_features)
        print(all_text_features.shape)
        return all_text_features


parser = argparse.ArgumentParser()
# parser.add_argument("--subj", default=1, type=int)
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
args = parser.parse_args()
stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"
# feature_output_dir = "%s/subj%01d" % (args.feature_dir, args.subj)

# all_coco_ids = np.load(
#     "%s/coco_ID_of_repeats_subj%02d.npy" % (args.project_output_dir, args.subj)
# )

from pycocotools.coco import COCO

trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"
train_caps = COCO(trainFile)
val_caps = COCO(valFile)

expand_dict = {}
expand_dict["person"] = ["man", "men", "women", "woman", "people", "guys", "people"]

# extract_object_base_text_feature()
# extract_top1_obejct_base_text_feature()
# extract_visual_resnet_feature()
# extract_visual_transformer_feature()

for s in range(7):
    print("Extracting subj%01d" % (s+2))
    feature_output_dir = "%s/subj%01d" % (args.feature_dir, (s+2))
    all_coco_ids = np.load(
        "%s/coco_ID_of_repeats_subj%02d.npy" % (args.project_output_dir, (s+2))
    )
    try:
        np.load("%s/clip_text.npy" % feature_output_dir)
    except FileNotFoundError:
        text_feat = extract_last_layer_feature(modality="text")
        np.save("%s/clip_text.npy" % feature_output_dir, text_feat)
    
    try:
        np.load("%s/clip_visual_resnet.npy" % feature_output_dir)
    except FileNotFoundError:
        visual_res_feat = extract_last_layer_feature(model_name="RN50", modality="vision")
        np.save("%s/clip_visual_resnet.npy" % feature_output_dir, visual_res_feat)
