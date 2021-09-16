import copy

import pandas as pd
import numpy as np
from torch._C import Value
from tqdm import tqdm
from PIL import Image

from sklearn.decomposition import PCA

import torch
import torchvision

from pycocotools.coco import COCO
import torchextractor as tx

import clip
from util.util import pytorch_pca

device = "cuda" if torch.cuda.is_available() else "cpu"

trainFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_train2017.json"
valFile = "/lab_data/tarrlab/common/datasets/coco_annotations/captions_val2017.json"
train_caps=COCO(trainFile)
val_caps=COCO(valFile)

def load_captions(cid):
    annIds = train_caps.getAnnIds(imgIds=[cid])
    anns = train_caps.loadAnns(annIds)
    if anns == []:
        annIds = val_caps.getAnnIds(imgIds=[cid])
        anns = val_caps.loadAnns(annIds)

    if anns == []:
        print("no captions extracted for image: " + str(cid))
    
    captions = [d['caption'] for d in anns]
    return captions


# Load Images
subj = 1

stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images/images"
feature_output_dir = "/user_data/yuanw3/project_outputs/NSD/features/subj%01d" % subj
project_output_dir = "/user_data/yuanw3/project_outputs/NSD/output"

# stim = pd.read_pickle(
#     "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
# )
# all_coco_ids = stim.cocoId

all_coco_ids = np.load("%s/coco_ID_of_repeats_subj%02d.npy" % (project_output_dir, subj))

all_features = []
model, preprocess = clip.load("ViT-B/32", device=device)

# # extract image features
# all_images_paths = list()
# all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
# print("Number of Images: {}".format(len(all_images_paths)))
# for p in tqdm(all_images_paths):
#     image = preprocess(Image.open(p)).unsqueeze(0).to(device)
#         # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image)
        
#         # logits_per_image, logits_per_text = model(image, text)
#         # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#     all_features.append(image_features.cpu().data.numpy())
# all_features = np.array(all_features)
# print(all_features.shape)
# np.save("/lab_data/tarrlab/common/datasets/features/NSD/clip.npy", all_features)

# extract text feature of image titles
# all_text_features = []
# for cid in tqdm(all_coco_ids):
#     with torch.no_grad():
#         captions = load_captions(cid)
#         # print(captions)
#         text = clip.tokenize(captions).to(device)
#         cap_emb = model.encode_text(text).cpu().data.numpy()
#         all_text_features.append(cap_emb)

# all_text_features = np.array(all_text_features)
# print(all_text_features.shape)
# np.save("/lab_data/tarrlab/common/datasets/features/NSD/clip_text.npy", all_text_features)


# intermediate features
LOI_vision = ["visual.transformer.resblocks.%01d.ln_2" % i for i in range(12)]
LOI_text  = ["transformer.resblocks.%01d.ln_2" % i for i in range(12)]

# For visual features
# model = tx.Extractor(model, LOI_vision)
# compressed_features = [copy.copy(e) for _ in range(12) for e in [[]]]

# for cid in tqdm(all_coco_ids):
#     with torch.no_grad():
#         image_path = "%s/%s.jpg" % (stimuli_dir, cid)
#         image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

#         captions = load_captions(cid)
#         text = clip.tokenize(captions).to(device)

#         _, features = model(image, text)

#         for i, f in enumerate(features.values()):
#             compressed_features[i].append(f.squeeze().cpu().data.numpy().flatten())

# for text features
text_features = [copy.copy(e) for _ in range(12) for e in [[]]]
model = tx.Extractor(model, LOI_text)
for cid in tqdm(all_coco_ids[:3]):
    with torch.no_grad():
        image_path = "%s/%s.jpg" % (stimuli_dir, cid)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        captions = load_captions(cid)

        layer_features = [copy.copy(e) for _ in range(12) for e in [[]]] # layer_features is 12 x 5 x m
        for caption in captions:
            text = clip.tokenize(caption).to(device)
            _, features = model(image, text)
            # features is a feature dictionary for all layers, each image, each caption
            for i, layer in enumerate(LOI_text):
                layer_features[i].append(features[layer].cpu().data.numpy().squeeze().flatten())

            print(np.array(layer_features).shape)
        avg_features = np.mean(np.array(layer_features), axis=1) # 12 x m

    for i in range(len(LOI_text)):  
        text_features[i].append(avg_features[i])

text_features = np.array(text_features)
print(text_features.shape) # 12 x 10000 x m

for l, f in enumerate(text_features):
    pca = PCA(n_components=min(f.shape[0], 512), whiten=True, svd_solver="full")
    try:
        fp = pca.fit_transform(f)
    except ValueError:
        print(fp.shape)

    print("Feature %01d has shape of:" % l)
    print(fp.shape)

    np.save("%s/text_layer_%01d.npy" % (feature_output_dir, l), fp)

        
