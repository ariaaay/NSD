import clip
import torch
import json

import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image

from pycocotools.coco import COCO

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
stimuli_dir = "/lab_data/tarrlab/common/datasets/NSD_images"

stim = pd.read_pickle(
    "/lab_data/tarrlab/common/datasets/NSD/nsddata/experiments/nsd/nsd_stim_info_merged.pkl"
)
all_coco_ids = stim.cocoId
all_images_paths = list()
all_images_paths += ["%s/%s.jpg" % (stimuli_dir, id) for id in all_coco_ids]
print("Number of Images: {}".format(len(all_images_paths)))

all_features = []
model, preprocess = clip.load("ViT-B/32", device=device)

# extract image features
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
all_text_features = []
for cid in tqdm(all_coco_ids):
    with torch.no_grad():
        captions = load_captions(cid)
        # print(captions)
        text = clip.tokenize(captions).to(device)
        cap_emb = model.encode_text(text).cpu().data.numpy()
        all_text_features.append(cap_emb)

all_text_features = np.array(all_text_features)
print(all_text_features.shape)
np.save("/lab_data/tarrlab/common/datasets/features/NSD/clip_text.npy", all_text_features)