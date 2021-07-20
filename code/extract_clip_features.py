import clip
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for p in tqdm(all_images_paths):
    image = preprocess(Image.open(p)).unsqueeze(0).to(device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print(image_features.shape)

    all_features.append(image_features.data.numpy())
all_features = np.array(all_features)
print(all_features.shape)
np.save("/lab_data/tarrlab/common/datasets/features/NSD/clip.npy", all_features)