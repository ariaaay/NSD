import base64

import json
import csv
import sys
import numpy as np

# import pandas as pd
from tqdm import tqdm

from util.tsv_file import TSVFile

data_path = "/user_data/yuanw3/project_outputs/NSD/features/general/model_0060000/"
id2ind = json.load(open("%s/imageid2idx.json" % data_path))
all_coco_ids = np.load(
    "/user_data/yuanw3/project_outputs/NSD/output/coco_ID_of_repeats_subj01.npy"
)
oscar_ids = list(id2ind.keys())
feat_file = "%s/features.tsv" % data_path
feat_tsv = TSVFile(feat_file)
num_rows = feat_tsv.num_rows()

for i in tqdm(range(num_rows)):
    tmp = feat_tsv.seek(i)

    if i == 0:
        features = np.zeros((len(all_coco_ids), 2054))

    img_id = int(tmp[0])
    if img_id in all_coco_ids:
        nsd_idx = list(all_coco_ids).index(img_id)
        num_boxes = int(tmp[1])
        feat = np.frombuffer(base64.b64decode(tmp[-1]), dtype=np.float32).reshape(
            (num_boxes, -1)
        )
        features[nsd_idx, :] = np.mean(feat, axis=0)
        # print(feat.shape)
        # features[nsd_idx, :] = np.frombuffer(base64.b64decode(tmp[2]), np.float32)

    # if i > 30:
    #     break

print(np.array(features).shape)

np.save(
    "/user_data/yuanw3/project_outputs/NSD/features/oscar/subj01.npy",
    np.array(features),
)
